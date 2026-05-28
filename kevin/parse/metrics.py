"""
Revenue, gross margin, operating margin, and guidance extractor.

Design: regex-first with unit normalisation (billions→millions).
Each metric carries a confidence score (0-100).
"""
from __future__ import annotations

import re

from kevin.models import GuidanceRange


# ── Unit normalisation ────────────────────────────────────────────────────────

def _to_millions(value: float, unit_str: str) -> float:
    u = unit_str.lower().strip()
    if u.startswith("b"):    # billion / B
        return value * 1_000
    if u.startswith("t"):    # trillion (rare but exists)
        return value * 1_000_000
    return value             # assume millions


# ── Revenue ───────────────────────────────────────────────────────────────────

_REV_PATTERNS: list[re.Pattern] = [
    # "$12.3 billion in net revenue"  /  "revenue of $12.3B"
    re.compile(
        r"(?:net\s+)?(?:revenue|net\s+sales|total\s+revenue|total\s+sales)"
        r"[^$\d]{0,60}\$\s*([\d,]+(?:\.\d+)?)\s*(billion|million|B|M|bn|mm)\b",
        re.I
    ),
    # "$12.3 billion revenue"
    re.compile(
        r"\$\s*([\d,]+(?:\.\d+)?)\s*(billion|million|B|M|bn|mm)\s+"
        r"(?:in\s+)?(?:net\s+)?(?:revenue|net\s+sales|total\s+revenue)",
        re.I
    ),
    # Revenue was $12,345.6 (implied millions from document header)
    re.compile(
        r"(?:net\s+)?(?:revenue|net\s+sales|total\s+revenue)"
        r"[^$\d]{0,30}\$\s*([\d,]+(?:\.\d+)?)\s*(?:\n|$|\s{2,})",
        re.I
    ),
]

_YOY_PATTERN = re.compile(
    r"(?:revenue|sales)\s+(?:grew|increased|declined|decreased|fell|rose)"
    r"[^%]{0,40}([\d.]+)\s*%\s*(?:year.over.year|yoy|y\/y)",
    re.I
)


def extract_revenue(text: str) -> tuple[float | None, float | None]:
    """Return (revenue_mm, yoy_pct).  yoy_pct is a fraction (0.12 = +12%)."""
    revenue: float | None = None
    for pat in _REV_PATTERNS:
        m = pat.search(text)
        if m:
            try:
                raw  = float(m.group(1).replace(",", ""))
                unit = m.group(2) if pat.groups >= 2 else "M"
                rev  = _to_millions(raw, unit)
                # Sanity: SEC 8-K filers must have > $10M revenue to be plausible
                if rev < 10:
                    continue
                revenue = rev
                break
            except (IndexError, ValueError):
                continue

    yoy: float | None = None
    m2 = _YOY_PATTERN.search(text)
    if m2:
        try:
            pct_raw = float(m2.group(1))
            # Detect direction word
            ctx = m2.group(0).lower()
            sign = -1 if any(w in ctx for w in ("declin", "decreas", "fell")) else 1
            yoy = sign * pct_raw / 100
        except ValueError:
            pass

    return revenue, yoy


# ── Gross margin ──────────────────────────────────────────────────────────────

_GM_PATTERNS: list[re.Pattern] = [
    re.compile(r"gross\s+(?:profit\s+)?margin[^%\d]{0,40}([\d.]+)\s*%", re.I),
    re.compile(r"gross\s+margin\s+(?:was|of|improved to|declined to)[^%\d]{0,20}([\d.]+)\s*%", re.I),
]


def extract_gross_margin(text: str) -> float | None:
    for pat in _GM_PATTERNS:
        m = pat.search(text)
        if m:
            try:
                return float(m.group(1)) / 100
            except ValueError:
                continue
    return None


# ── Operating margin ──────────────────────────────────────────────────────────

_OM_PATTERNS: list[re.Pattern] = [
    re.compile(r"operating\s+(?:income\s+)?margin[^%\d]{0,40}([\d.]+)\s*%", re.I),
    re.compile(r"operating\s+margin\s+(?:was|of)[^%\d]{0,20}([\d.]+)\s*%", re.I),
]


def extract_operating_margin(text: str) -> float | None:
    for pat in _OM_PATTERNS:
        m = pat.search(text)
        if m:
            try:
                return float(m.group(1)) / 100
            except ValueError:
                continue
    return None


# ── Guidance ──────────────────────────────────────────────────────────────────

_GUIDANCE_PATTERNS: list[re.Pattern] = [
    # "$X.XX to $Y.YY per share"
    re.compile(
        r"\$\s*(\d+(?:\.\d+)?)\s+(?:to|–|-)\s+\$\s*(\d+(?:\.\d+)?)\s+"
        r"(?:per\s+(?:diluted\s+)?(?:common\s+)?share|eps)",
        re.I
    ),
    # "guidance of $X.XX to $Y.YY"
    re.compile(
        r"(?:guidance|outlook|expect(?:s|ed|ing)?)"
        r"[^$\d]{0,40}\$\s*(\d+(?:\.\d+)?)\s+(?:to|–|-)\s+\$\s*(\d+(?:\.\d+)?)",
        re.I
    ),
    # "earnings per share of approximately $X.XX"
    re.compile(
        r"(?:guidance|outlook|expect(?:s|ed|ing)?)"
        r"[^$\d]{0,60}\$\s*(\d+(?:\.\d+)?)\s+(?:per\s+(?:diluted\s+)?share|eps)",
        re.I
    ),
]

# Period detection for guidance
_PERIOD_PATTERNS = re.compile(
    r"\b(Q[1-4]\s+(?:FY)?\d{2,4}|(?:first|second|third|fourth)\s+quarter\s+(?:fiscal\s+)?\d{4}"
    r"|full[\s-]?year\s+\d{4}|FY\s*\d{2,4})\b",
    re.I
)

# Guidance direction — raised vs lowered — from surrounding prose
_GUIDANCE_RAISED_RE = re.compile(
    r"(?:raise[sd]?|increas(?:e[sd]?|ing)|lift(?:ed|ing)?|upward)"
    r".{0,80}"
    r"(?:guidance|outlook|forecast|expectation)",
    re.I
)
_GUIDANCE_LOWERED_RE = re.compile(
    r"(?:lower(?:ed|s|ing)?|cut|reduc(?:e[sd]?|ing)|decreas(?:e[sd]?|ing)"
    r"|withdr(?:ew|awn?)|downward)"
    r".{0,80}"
    r"(?:guidance|outlook|forecast|expectation)",
    re.I
)


def extract_guidance(text: str) -> GuidanceRange | None:
    """Extract forward guidance midpoint from press release language."""
    for pat in _GUIDANCE_PATTERNS:
        m = pat.search(text)
        if not m:
            continue
        try:
            lo = float(m.group(1))
            if pat.groups >= 2 and m.group(2):
                hi  = float(m.group(2))
                mid = (lo + hi) / 2
            else:
                lo, hi, mid = lo, lo, lo

            # Sanity check — guidance EPS should be in plausible range
            if not (0 <= mid <= 500):
                continue

            # Widen context window for period and direction detection
            ctx_start = max(0, m.start() - 400)
            ctx       = text[ctx_start: m.end() + 400]

            # Period
            period: str | None = None
            pm = _PERIOD_PATTERNS.search(ctx)
            if pm:
                period = pm.group(1)

            # Guidance direction — raised / lowered — from surrounding prose
            is_raised: bool | None = None
            if _GUIDANCE_RAISED_RE.search(ctx):
                is_raised = True
            elif _GUIDANCE_LOWERED_RE.search(ctx):
                is_raised = False

            return GuidanceRange(
                lo=lo, hi=hi, midpoint=mid,
                metric="eps", period=period,
                is_raised=is_raised,
            )
        except (ValueError, IndexError):
            continue
    return None
