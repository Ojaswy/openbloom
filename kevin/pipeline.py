"""
Kevin async pipeline.

Orchestration order for a single filing:
  1. Resolve ticker -> CIK  (edgar/feed.py)
  2. Fetch recent 8-K filing list  (edgar/feed.py)
  3. For each filing:
     a. Resolve exhibit URLs  (edgar/feed.py)
     b. Fetch exhibit HTML    (edgar/fetcher.py)
     c. Classify items        (edgar/classifier.py)
     d. Parse EPS             (parse/eps.py)
     e. Parse metrics         (parse/metrics.py)
     f. LLM fallback if low confidence  (parse/llm_fallback.py)
     g. Analyse tone          (analyze/sentiment.py)
     h. Generate signal       (analyze/signals.py)
  4. Return list[KevinBrief] sorted by filed_at desc

All network I/O is async; concurrency is bounded by cfg.MAX_CONCURRENT_FILINGS.
"""
from __future__ import annotations

import asyncio
import logging
import re
from datetime import datetime, timezone

import httpx

from kevin.analyze.sentiment import analyse_tone
from kevin.analyze.signals import generate_signal
from kevin.config import cfg
from kevin.edgar.classifier import classify_items, is_earnings_filing
from kevin.edgar.feed import (
    resolve_exhibit_urls,
    search_by_cik,
    search_8k_filings,
)
from kevin.edgar.fetcher import build_client, fetch_filing_exhibit, throttle
from kevin.market import fetch_consensus
from kevin.models import FilingIndex, FilingMetrics, KevinBrief
from kevin.parse.eps import extract_eps
from kevin.parse.llm_fallback import get_llm_client, llm_extract_eps
from kevin.parse.metrics import (
    extract_gross_margin,
    extract_guidance,
    extract_operating_margin,
    extract_revenue,
)

log = logging.getLogger(__name__)

# Module-level compiled patterns for _strip_html — avoids recompiling on every call
_TAG_RE     = re.compile(r"<[^>]+>")
_SPACE_RE   = re.compile(r"[ \t]+")
_NEWLINE_RE = re.compile(r"\n{3,}")


# CIK resolution caches (per process)

_cik_cache: dict[str, str] = {}
_ticker_map: dict[str, str] | None = None   # ticker.upper() -> cik_str
_ticker_map_lock = asyncio.Lock()


async def _load_ticker_map(client: httpx.AsyncClient) -> dict[str, str]:
    """
    Fetch and cache SEC company_tickers.json (one round-trip per process).
    Lock prevents duplicate fetches when multiple tickers resolve concurrently.
    """
    global _ticker_map
    if _ticker_map is not None:
        return _ticker_map
    async with _ticker_map_lock:
        if _ticker_map is not None:   # re-check inside lock
            return _ticker_map
        try:
            await throttle()
            resp = await client.get(
                "https://www.sec.gov/files/company_tickers.json",
                headers={"User-Agent": cfg.EDGAR_UA},
                timeout=20,
            )
            resp.raise_for_status()
            _ticker_map = {
                v["ticker"].upper(): str(v["cik_str"])
                for v in resp.json().values()
                if "ticker" in v and "cik_str" in v
            }
        except Exception:
            _ticker_map = {}
    return _ticker_map


async def _resolve_cik(client: httpx.AsyncClient, ticker: str) -> str | None:
    """Resolve ticker -> CIK via the SEC company_tickers.json (cached per process)."""
    upper = ticker.upper()
    if upper in _cik_cache:
        return _cik_cache[upper]
    ticker_map = await _load_ticker_map(client)
    cik = ticker_map.get(upper)
    if cik:
        _cik_cache[upper] = cik
    return cik


# Single filing processor

async def _process_filing(
    client:   httpx.AsyncClient,
    index:    FilingIndex,
    ticker:   str,
    llm:      object | None,
) -> KevinBrief | None:
    """Full pipeline for one FilingIndex -> KevinBrief. Returns None on hard failure."""
    errors: list[str] = []

    # 1. Resolve exhibit URLs if not already populated
    if not index.exhibit_urls:
        try:
            index.exhibit_urls = await resolve_exhibit_urls(client, index.cik, index.accession)
        except Exception as e:
            errors.append(f"exhibit_url_resolution: {e}")

    exhibit_url = index.exhibit_urls[0] if index.exhibit_urls else None

    # 2. Fetch exhibit HTML
    html: str = ""
    if exhibit_url:
        fetched = await fetch_filing_exhibit(client, exhibit_url)
        if fetched:
            html = fetched
        else:
            errors.append("no_exhibit_html")
    else:
        errors.append("no_exhibit_html")

    # 3. Classify items
    plain_text   = _strip_html(html)
    index.items  = classify_items(plain_text, index.items or [])

    # 4. Parse EPS
    llm_used = False
    eps_basic, eps_diluted, eps_adjusted, top_score = extract_eps(html)

    # LLM fallback when regex confidence is low
    if top_score < cfg.LLM_CONFIDENCE_THRESHOLD and llm is not None:
        log.info("[%s] Low EPS confidence (%.0f) — invoking LLM fallback", ticker, top_score)
        llm_result = await llm_extract_eps(html[:8000], llm)
        if llm_result is not None:
            eps_basic = llm_result
            llm_used  = True

    # 5. Parse other metrics
    revenue_mm, rev_yoy = extract_revenue(plain_text)
    gross_margin        = extract_gross_margin(plain_text)
    op_margin           = extract_operating_margin(plain_text)
    guidance            = extract_guidance(plain_text)

    # Best available GAAP EPS (prefer basic over diluted)
    eps_primary: float | None = None
    if eps_basic and eps_basic.value is not None:
        eps_primary = eps_basic.value
    elif eps_diluted and eps_diluted.value is not None:
        eps_primary = eps_diluted.value

    metrics = FilingMetrics(
        eps_basic        = eps_basic,
        eps_diluted      = eps_diluted,
        eps_adjusted     = eps_adjusted,
        eps_primary      = eps_primary,
        revenue_mm       = revenue_mm,
        revenue_yoy_pct  = rev_yoy,
        gross_margin     = gross_margin,
        operating_margin = op_margin,
        guidance         = guidance,
        eps_estimate     = None,   # caller can inject post-hoc
        rev_estimate_mm  = None,
    )

    # 6. Analyse tone
    tone = analyse_tone(plain_text)

    # 7. Generate signal
    signal = generate_signal(
        index,
        metrics,
        tone,
        exhibit_url = exhibit_url,
        llm_used    = llm_used,
        ticker      = ticker.upper(),
    )

    return KevinBrief(
        index          = index,
        metrics        = metrics,
        tone           = tone,
        signal         = signal,
        raw_text_chars = len(plain_text),
        parsed_at      = datetime.now(timezone.utc),
        errors         = errors,
    )


def _strip_html(html: str) -> str:
    """Fast HTML -> plain-text using module-level compiled patterns."""
    text = _TAG_RE.sub(" ", html)
    text = _SPACE_RE.sub(" ", text)
    text = _NEWLINE_RE.sub("\n\n", text)
    return text.strip()


# Public API

async def analyse_ticker(
    ticker:        str,
    *,
    since_days:    int  = cfg.DEFAULT_LOOKBACK_DAYS,
    earnings_only: bool = True,
    max_filings:   int  = 10,
) -> list[KevinBrief]:
    """
    Fetch and analyse all recent 8-K filings for a ticker.

    Returns a list of KevinBrief, newest first.
    Set earnings_only=False to include non-earnings 8-Ks (M&A, departures, etc.).
    """
    llm = get_llm_client()

    async with build_client() as client:
        cik = await _resolve_cik(client, ticker)
        if not cik:
            log.warning("Could not resolve CIK for ticker: %s", ticker)
            filings = await search_8k_filings(
                client, ticker=ticker, since_days=since_days, max_results=max_filings
            )
        else:
            filings = await search_by_cik(
                client, cik, since_days=since_days, max_results=max_filings
            )
            for f in filings:
                f.ticker = ticker.upper()

        if earnings_only:
            filings = [
                f for f in filings
                if is_earnings_filing(f.items) or not f.items
            ]

        if not filings:
            log.info("No filings found for %s in last %d days", ticker, since_days)
            return []

        sem = asyncio.Semaphore(cfg.MAX_CONCURRENT_FILINGS)

        async def _bounded(index: FilingIndex) -> KevinBrief | None:
            async with sem:
                try:
                    return await _process_filing(client, index, ticker, llm)
                except Exception as e:
                    log.error("Failed processing %s: %s", index.accession, e)
                    return None

        results = await asyncio.gather(*(_bounded(f) for f in filings))

    briefs = [r for r in results if r is not None]
    briefs.sort(key=lambda b: b.index.filed_at, reverse=True)

    # Inject analyst consensus (best-effort; never fails the pipeline)
    try:
        consensus = await asyncio.to_thread(fetch_consensus, ticker)
        eps_est = consensus.get("eps_estimate")
        rev_est = consensus.get("rev_estimate_mm")
        for brief in briefs:
            if eps_est is not None:
                brief.metrics.eps_estimate = eps_est
            if rev_est is not None:
                brief.metrics.rev_estimate_mm = rev_est
    except Exception:
        pass

    return briefs


async def analyse_filing(
    accession_or_url: str,
    ticker:  str = "UNKNOWN",
) -> KevinBrief | None:
    """
    Analyse a single filing by accession number or direct exhibit URL.

    Useful for ad-hoc analysis of a specific 8-K.
    """
    llm = get_llm_client()

    async with build_client() as client:
        if accession_or_url.startswith("http"):
            exhibit_url = accession_or_url
            index = FilingIndex(
                accession    = "direct",
                cik          = "0",
                ticker       = ticker,
                company      = ticker,
                filed_at     = datetime.now(timezone.utc),
                period       = None,
                exhibit_urls = [exhibit_url],
            )
        else:
            acc    = accession_or_url.replace("-", "")
            cik    = accession_or_url.split("-")[0].lstrip("0") or "0"
            dashed = accession_or_url if "-" in accession_or_url else \
                     f"{acc[:10]}-{acc[10:12]}-{acc[12:]}"
            index = FilingIndex(
                accession    = dashed,
                cik          = cik,
                ticker       = ticker,
                company      = ticker,
                filed_at     = datetime.now(timezone.utc),
                period       = None,
                exhibit_urls = [],
            )

        return await _process_filing(client, index, ticker, llm)
