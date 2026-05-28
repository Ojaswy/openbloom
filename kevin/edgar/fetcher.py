"""
Async HTTP fetcher with:
  - EDGAR-compliant rate limiting (≤10 req/s, 0.1s between requests)
  - Disk cache (avoids re-fetching filed documents — they never change)
  - Transparent gzip / encoding handling
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from pathlib import Path

import httpx

from kevin.config import cfg

log = logging.getLogger(__name__)

_HEADERS = {
    "User-Agent":      cfg.EDGAR_UA,
    "Accept-Encoding": "gzip, deflate",
    "Accept":          "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

# Minimum seconds between consecutive requests (EDGAR ToS: 10 req/s)
_MIN_INTERVAL = 1.0 / max(cfg.EDGAR_RATE, 1)

_last_request_time: float = 0.0
_rate_lock = asyncio.Lock()


async def _throttle() -> None:
    """Enforce per-process rate limit for all EDGAR requests."""
    global _last_request_time
    async with _rate_lock:
        now   = time.monotonic()
        gap   = _MIN_INTERVAL - (now - _last_request_time)
        if gap > 0:
            await asyncio.sleep(gap)
        _last_request_time = time.monotonic()


async def throttle() -> None:
    """Public rate-limit gate — call before any direct client.get() to an EDGAR host."""
    await _throttle()


def _cache_path(url: str) -> Path:
    key = hashlib.sha256(url.encode()).hexdigest()
    cfg.EDGAR_CACHE.mkdir(parents=True, exist_ok=True)
    return cfg.EDGAR_CACHE / f"{key}.html"


def build_client() -> httpx.AsyncClient:
    """Create a shared httpx client with sensible timeouts."""
    return httpx.AsyncClient(
        timeout=httpx.Timeout(30.0, connect=10.0),
        follow_redirects=True,
        http2=False,
    )


async def fetch_url(
    client: httpx.AsyncClient,
    url: str,
    *,
    use_cache: bool = True,
) -> str | None:
    """
    Fetch a URL and return its text content.
    Caches responses to disk so filed documents are fetched once only.
    Returns None on any non-recoverable error.
    """
    cache_file = _cache_path(url)

    if use_cache and cache_file.exists():
        log.debug("Cache hit: %s", url)
        return cache_file.read_text(encoding="utf-8", errors="replace")

    await _throttle()
    try:
        resp = await client.get(url, headers=_HEADERS)
        if resp.status_code == 429:
            log.warning("EDGAR 429 — backing off 5s then retrying")
            await asyncio.sleep(5)
            await _throttle()
            resp = await client.get(url, headers=_HEADERS)

        if resp.status_code != 200:
            log.warning("HTTP %s for %s", resp.status_code, url)
            return None

        text = resp.text
        if use_cache:
            cache_file.write_text(text, encoding="utf-8")
        return text

    except (httpx.TimeoutException, httpx.NetworkError) as exc:
        log.error("Network error fetching %s: %s", url, exc)
        return None


async def fetch_filing_exhibit(
    client: httpx.AsyncClient,
    exhibit_url: str,
) -> str | None:
    """Convenience wrapper — always caches (filings are immutable)."""
    return await fetch_url(client, exhibit_url, use_cache=True)


async def fetch_json(
    client: httpx.AsyncClient,
    url: str,
    *,
    use_cache: bool = False,
) -> dict | None:
    """Fetch a JSON endpoint and return parsed dict."""
    text = await fetch_url(client, url, use_cache=use_cache)
    if text is None:
        return None
    try:
        return json.loads(text)
    except Exception as exc:
        log.error("JSON parse error for %s: %s", url, exc)
        return None
