"""
EDGAR full-text search feed + CIK lookup.

Uses:
  - EDGAR EFTS (https://efts.sec.gov)  → free, no key required
  - EDGAR company search API            → CIK resolution
  - EDGAR filing index                  → exhibit URL resolution

Rate limiting is handled by the caller (fetcher.py).
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx

from kevin.config import cfg
from kevin.edgar.fetcher import throttle
from kevin.models import FilingIndex, FilingItemType


# ── Helpers ───────────────────────────────────────────────────────────────────

_HEADERS = {
    "User-Agent":      cfg.EDGAR_UA,
    "Accept-Encoding": "gzip",
}

_ITEM_PATTERN = re.compile(r"item\s+(\d+\.\d+[A-Za-z]?)", re.I)
# Bare format used by EDGAR submissions JSON metadata: "2.02,9.01"
_ITEM_BARE_PATTERN = re.compile(r"\b(\d+\.\d+[A-Za-z]?)\b")

_KNOWN_ITEMS = {i.value: i for i in FilingItemType}


def _parse_items(text: str) -> list[FilingItemType]:
    """
    Extract 8-K item numbers from filing text or EDGAR submissions metadata.

    Handles two formats:
      - "Item 2.02 ..." (filing text / index HTML)
      - "2.02,9.01"     (EDGAR submissions JSON metadata — no prefix)
    Returns [] for empty input so callers can distinguish "unknown" from "OTHER".
    """
    if not text.strip():
        return []
    found: list[FilingItemType] = []
    # Primary: requires "item" prefix (filing text)
    for m in _ITEM_PATTERN.finditer(text):
        key = m.group(1)
        item = _KNOWN_ITEMS.get(key, FilingItemType.OTHER)
        if item not in found:
            found.append(item)
    if not found:
        # Fallback: bare numbers from EDGAR submissions metadata
        for m in _ITEM_BARE_PATTERN.finditer(text):
            key = m.group(1)
            if key in _KNOWN_ITEMS:          # only add recognised items; skip 9.01 etc.
                item = _KNOWN_ITEMS[key]
                if item not in found:
                    found.append(item)
    return found or [FilingItemType.OTHER]


def _accession_to_parts(acc: str) -> tuple[str, str]:
    """'0001234567-24-000001' → (cik='1234567', nodash='0001234567240000001')"""
    clean = acc.replace("-", "")
    cik   = acc.split("-")[0].lstrip("0") or "0"
    return cik, clean


# ── Public API ────────────────────────────────────────────────────────────────

async def search_8k_filings(
    client: httpx.AsyncClient,
    *,
    ticker: str | None      = None,
    cik: str | None         = None,
    since_days: int         = 30,
    item: str               = "2.02",
    max_results: int        = 40,
) -> list[FilingIndex]:
    """
    Search EDGAR EFTS for recent 8-K filings containing a specific item.

    Pass either `ticker` or `cik`; if neither, returns market-wide filings.
    """
    since_date = (datetime.now(timezone.utc) - timedelta(days=since_days)).date().isoformat()

    params: dict[str, Any] = {
        "q":              f'"Item {item}"',
        "forms":          "8-K",
        "dateRange":      "custom",
        "startdt":        since_date,
        "_source":        "*",
        "hits.hits.total.value": "true",
    }
    if cik:
        params["entity"] = cik
    elif ticker:
        params["q"] = f'"Item {item}" "{ticker}"'

    url = "https://efts.sec.gov/LATEST/search-index"
    await throttle()
    resp = await client.get(url, params=params, headers=_HEADERS, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    hits  = data.get("hits", {}).get("hits", [])[:max_results]
    filings: list[FilingIndex] = []

    for hit in hits:
        src = hit.get("_source", {})
        acc = src.get("accession_no") or hit.get("_id", "")
        if not acc:
            continue

        raw_cik, _ = _accession_to_parts(acc)

        filed_str = src.get("file_date", "")
        try:
            filed_at = datetime.fromisoformat(filed_str).replace(tzinfo=timezone.utc)
        except Exception:
            filed_at = datetime.now(timezone.utc)

        filings.append(FilingIndex(
            accession  = acc,
            cik        = raw_cik,
            ticker     = ticker,   # caller-supplied; file_num is a list, not a ticker
            company    = src.get("entity_name", "Unknown"),
            filed_at   = filed_at,
            period     = src.get("period_of_report"),
            items      = [],   # populated later via classifier
            exhibit_urls = [],
        ))

    return filings


async def resolve_exhibit_urls(
    client: httpx.AsyncClient,
    cik: str,
    accession: str,
) -> list[str]:
    """
    Fetch the EDGAR filing index page and extract all EX-99.1 / EX-99
    HTML exhibit URLs (earnings releases are almost always EX-99.1).
    """
    _, nodash = _accession_to_parts(accession)
    index_url = (
        f"https://www.sec.gov/Archives/edgar/data/{cik}/{nodash}/"
        f"{accession}-index.htm"
    )
    try:
        await throttle()
        resp = await client.get(index_url, headers=_HEADERS, timeout=10)
        if resp.status_code != 200:
            return []
        html = resp.text

        # Find EX-99.x exhibit links (earnings release is almost always EX-99.1)
        exhibit_re = re.compile(
            r'<td[^>]*>EX-99(?:\.\d+)?</td>.*?<a[^>]+href="([^"]+\.htm[l]?)"',
            re.I | re.S
        )
        matches = exhibit_re.findall(html)

        # Fallback: any .htm link in the document
        if not matches:
            matches = re.findall(r'href="([^"]+\.htm[l]?)"', html, re.I)

        urls: list[str] = []
        for path in matches:
            if path.startswith("http"):
                urls.append(path)
            elif path.startswith("/"):
                urls.append(f"https://www.sec.gov{path}")
            else:
                urls.append(
                    f"https://www.sec.gov/Archives/edgar/data/{cik}/{nodash}/{path}"
                )
        return urls[:3]   # first 3 is plenty; 1 is almost always the earnings release
    except Exception:
        return []


async def search_by_cik(
    client: httpx.AsyncClient,
    cik: str,
    *,
    since_days: int = 30,
    max_results: int = 10,
) -> list[FilingIndex]:
    """Fetch recent 8-K filings for a specific CIK via the submissions JSON."""
    url = f"https://data.sec.gov/submissions/CIK{cik.zfill(10)}.json"
    try:
        await throttle()
        resp = await client.get(url, headers=_HEADERS, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return []

    company = data.get("name", "Unknown")
    recent  = data.get("filings", {}).get("recent", {})

    forms   = recent.get("form",           [])
    accns   = recent.get("accessionNumber",[])
    dates   = recent.get("filingDate",     [])
    periods = recent.get("reportDate",     [])
    # items_col available in newer submissions
    items_col = recent.get("items", [""] * len(forms))

    since_dt = datetime.now(timezone.utc) - timedelta(days=since_days)
    results: list[FilingIndex] = []

    for form, acc, date_str, period, items_raw in zip(forms, accns, dates, periods, items_col):
        if form not in ("8-K", "8-K/A"):
            continue
        try:
            filed_at = datetime.fromisoformat(date_str).replace(tzinfo=timezone.utc)
        except Exception:
            continue
        if filed_at < since_dt:
            break   # submissions are newest-first; safe to stop

        results.append(FilingIndex(
            accession  = acc,
            cik        = cik,
            ticker     = None,
            company    = company,
            filed_at   = filed_at,
            period     = period or None,
            items      = _parse_items(str(items_raw)),
            exhibit_urls = [],
            form       = form,
        ))
        if len(results) >= max_results:
            break

    return results
