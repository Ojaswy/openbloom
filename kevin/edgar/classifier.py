"""
8-K item type classifier.

Detects which SEC 8-K items are present in a filing from:
  1. The EDGAR submissions JSON `items` field (most reliable)
  2. The filing index HTML
  3. The exhibit text itself (fallback)

Also determines MarketSession from the filing timestamp.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone

from kevin.models import FilingItemType, MarketSession


# ── Item detection ────────────────────────────────────────────────────────────

# Map of text patterns → item type (order matters — more specific first)
_ITEM_REGEXES: list[tuple[re.Pattern, FilingItemType]] = [
    (re.compile(r"item\s+2\.02",               re.I), FilingItemType.EARNINGS),
    (re.compile(r"item\s+4\.02",               re.I), FilingItemType.RESTATEMENT),
    (re.compile(r"item\s+1\.01",               re.I), FilingItemType.MA_AGREEMENT),
    (re.compile(r"item\s+5\.02",               re.I), FilingItemType.DEPARTURE),
    (re.compile(r"item\s+8\.01",               re.I), FilingItemType.MATERIAL_EVENT),
]

# Additional heuristics when item numbers not explicit
_EARNINGS_PHRASES = re.compile(
    r"(?:earnings|results of operations|quarterly results|quarterly earnings"
    r"|per\s+(?:common\s+)?share|eps\b|revenue|net income|net loss)",
    re.I
)
_RESTATEMENT_PHRASES = re.compile(
    r"(?:restat|non-reliance|material weakness|internal control)",
    re.I
)
_MA_PHRASES = re.compile(
    r"(?:merger\s+agreement|acquisition|definitive\s+agreement|transaction\s+agreement)",
    re.I
)
_DEPARTURE_PHRASES = re.compile(
    r"(?:chief executive|chief financial|ceo|cfo|president|director)\s+"
    r"(?:resigned|retirement|depart|appointed|named)",
    re.I
)


def classify_items(text: str, items_from_meta: list[FilingItemType] | None = None) -> list[FilingItemType]:
    """
    Classify which 8-K items are present.
    Prefers metadata items (from EDGAR submissions JSON) but supplements
    with text-based detection when metadata is empty or generic.
    """
    found: set[FilingItemType] = set(items_from_meta or [])
    found.discard(FilingItemType.OTHER)

    for pat, item in _ITEM_REGEXES:
        if pat.search(text):
            found.add(item)

    # Heuristic fallback when no explicit items found
    if not found:
        if _EARNINGS_PHRASES.search(text):
            found.add(FilingItemType.EARNINGS)
        if _RESTATEMENT_PHRASES.search(text):
            found.add(FilingItemType.RESTATEMENT)
        if _MA_PHRASES.search(text):
            found.add(FilingItemType.MA_AGREEMENT)
        if _DEPARTURE_PHRASES.search(text):
            found.add(FilingItemType.DEPARTURE)

    return list(found) if found else [FilingItemType.OTHER]


def is_earnings_filing(items: list[FilingItemType]) -> bool:
    return FilingItemType.EARNINGS in items


def is_high_risk_filing(items: list[FilingItemType], text: str) -> bool:
    """
    Returns True if this filing contains signals that warrant immediate
    human review regardless of bull/bear score.
    """
    risky = {FilingItemType.RESTATEMENT, FilingItemType.MA_AGREEMENT}
    if risky.intersection(items):
        return True
    # Check for going concern language in text
    if re.search(r"going\s+concern|substantial\s+doubt", text, re.I):
        return True
    return False


# ── Market session ────────────────────────────────────────────────────────────

def get_market_session(filed_at: datetime) -> MarketSession:
    """
    Classify filing timestamp into trading session.
    Uses zoneinfo (stdlib 3.9+) for DST-correct ET conversion.
    Falls back to static -4 offset if zone data unavailable.
    """
    try:
        from zoneinfo import ZoneInfo
        et = filed_at.astimezone(ZoneInfo("America/New_York"))
        et_hour, et_minute = et.hour, et.minute
    except Exception:
        # Fallback: EDT approximation (off by 1hr in Nov-Mar)
        et_hour   = (filed_at.hour - 4) % 24
        et_minute = filed_at.minute

    weekday = filed_at.weekday()   # 0=Mon ... 6=Sun
    if weekday >= 5:
        return MarketSession.WEEKEND

    if et_hour < 9 or (et_hour == 9 and et_minute < 30):
        return MarketSession.PRE_MARKET
    if et_hour >= 16:
        return MarketSession.POST_MARKET
    return MarketSession.INTRADAY
