"""
EPS extractor — enhanced port of the Trexquant parser (trex/parser.py).

Key improvements over the original:
  - Confidence score on every candidate (0-100)
  - Separate GAAP-basic, GAAP-diluted, non-GAAP paths with priority ranking
  - Jaccard similarity against canonical label corpus
  - Magnitude plausibility check (EPS virtually never > 500 for normal stocks)
  - Footnote marker suppression (avoids selecting "1", "2" etc.)
  - Returns EPSResult, not raw float — includes provenance

The extraction pipeline:
  1. Table scan (highest fidelity — structured data)
  2. Text-fallback regex (for text-only press releases)
  3. If best_score < threshold → caller should invoke LLM fallback
"""
from __future__ import annotations

import math
import re
import string
from dataclasses import dataclass, field

from bs4 import BeautifulSoup

from kevin.models import EPSResult, EPSType


# ── Constants (borrowed + extended from trex/parser.py) ───────────────────────

NUM_RE = re.compile(r"\(?\$?-?\d{1,3}(?:,\d{3})*(?:\.\d+)?\)?|\(?\.\d+\)?")

ADJUSTED_KEYWORDS = [
    "non-gaap", "non gaap", "adjusted", "core", "pro forma",
    "as adjusted", "excluding", "normalized",
]
NON_EPS_KEYWORDS = [
    "dividend", "dividends", "distribution", "book value",
    "common dividends", "dividend per share", "accumulated other comprehensive",
    "comprehensive loss", "income taxes payable", "operating income",
    "foreign currency", "currency exchange", "gain on sale",
]
EPS_TRIGGERS = [
    "per share", "eps", "earnings per share", "loss per share",
    "net income per", "net earnings per", "net (loss) income",
    "net loss per", "income per share",
]

CANONICAL_LABELS = [
    "Net income per common share",
    "Net income per share attributable",
    "Basic income (loss) per share",
    "Net loss per common share",
    "Loss Per Common Share: Basic",
    "Basic net income per share",
    "Adjusted EPS",
    "Income (loss) per share—basic",
    "NET EARNINGS PER COMMON SHARE - BASIC",
    "Basic Earnings per Share",
    "Net (loss) income per share attributable to the Company",
    "Earnings per common share",
    "Economic EPS",
    "Net loss per share",
    "Net income per common share",
    "Basic and diluted earnings per share",
    "Earnings per share",
    "Basic and diluted net (loss) income per share",
    "Basic and diluted loss per share",
    "Net income (loss) per share - basic",
    "Earnings per share - basic",
    "Earnings per share - diluted",
    "(Loss) earnings per share",
    "Basic earnings per share",
    "Basic earnings (loss) per share",
    "Net (loss) income per diluted common share",
    "Earnings Per Diluted Share",
    "Earnings per ordinary share",
]

# ── Measurement-header exclusions (these rows are table headers, not EPS rows) ──
# Ported from trex/parser.py — critical for avoiding date-number false positives.
_MEASUREMENT_PATTERNS: list[re.Pattern] = [
    re.compile(r"\bin\s+millions\b",            re.I),
    re.compile(r"\bin\s+thousands\b",           re.I),
    re.compile(r"\bexcept\s+per\s+share\b",     re.I),
    re.compile(r"\bexcept\s+per\s+share\s+amounts\b", re.I),
    re.compile(r"\bexcept\s+per\s+share\s+data\b", re.I),
    re.compile(r"\bin\s+millions\s+except\s+per\s+share\b", re.I),
    re.compile(r"\(\s*in\s+millions",           re.I),
    re.compile(r"\$\s*in\s+thousands",          re.I),
    re.compile(r"\(in\s+millions",              re.I),
    re.compile(r"\(in\s+thousands",             re.I),
    re.compile(r"\bin\s+millions,?\s+except",   re.I),
    re.compile(r"dollars\s+in\s+(?:thousands|millions)", re.I),
]

# Minimum score a candidate must reach before we trust it.
# Below this, the filing goes to LLM fallback.
_MIN_TRUST_SCORE = 20.0

# Pre-normalised canonical set
_STOP = frozenset(
    "the and of to a for per share shares common company inc co llc corp".split()
)


def _normalise(s: str) -> str:
    if not s:
        return ""
    s = s.replace("–", "-").replace("—", "-").replace("\xa0", " ").strip()
    s = s.translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))
    return " ".join(w.lower() for w in s.split() if w.strip() and w.lower() not in _STOP)


def _jaccard(a: str, b: str) -> float:
    sa, sb = set(a.split()), set(b.split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


_CANON_NORM = [_normalise(c) for c in CANONICAL_LABELS]


# ── Number cleaner ────────────────────────────────────────────────────────────

def clean_num(tok: str) -> float | None:
    if not tok:
        return None
    s = tok.strip().replace("\xa0", " ").strip()
    neg = False
    if s.startswith("(") and s.endswith(")"):
        neg, s = True, s[1:-1].strip()
    s = re.sub(r"[$,]", "", s)
    if s.startswith(("−", "-")):
        neg, s = True, s.lstrip("−-").strip()
    if s.startswith("."):
        s = "0" + s
    try:
        v = float(s)
        return -abs(v) if neg else v
    except ValueError:
        m = re.search(r"-?\d[\d,]*\.\d+|-?\d[\d,]*",
                      tok.replace("(", "-").replace(")", ""))
        if m:
            try:
                return float(m.group(0).replace(",", ""))
            except ValueError:
                return None
    return None


# ── Candidate dataclass ───────────────────────────────────────────────────────

@dataclass
class Candidate:
    value:       float
    eps_type:    EPSType
    gaap_like:   bool
    label:       str
    snippet:     str
    score:       float
    col_index:   int = 0
    source:      str = "regex_table"
    best_j:      float = 0.0
    token_raw:   str = ""
    origin_is_labelcell: bool = False


# ── Scoring ───────────────────────────────────────────────────────────────────

def _score(
    label: str,
    full_row: str,
    col_index: int,
    value: float,
    eps_type: EPSType,
    gaap_like: bool,
    token_raw: str,
    origin_is_labelcell: bool,
) -> tuple[float, float]:
    """Return (score, best_jaccard) for a candidate."""
    lab  = label.lower()
    full = full_row.lower()
    labn = _normalise(label)

    # Jaccard against canonical label corpus
    best_j = max((_jaccard(labn, cn) for cn in _CANON_NORM), default=0.0)
    score  = 40 * best_j

    # Keyword bonuses
    if "net earnings per common share" in full or "net earnings per common share" in lab:
        score += 30
    if "net (loss) income per" in full or "net loss per" in full:
        score += 25
    if "earnings per share" in full or "per share" in full or "eps" in full:
        score += 10
    if "basic" in lab:
        score += 8
    if "dilut" in lab:
        score += 4
    if gaap_like:
        score += 6

    # Heavy penalties
    if any(k in lab for k in NON_EPS_KEYWORDS):
        score -= 80
    if any(k in lab for k in ADJUSTED_KEYWORDS):
        score -= 40

    # Net income WITHOUT per-share → big penalty (it's not EPS)
    if ("net income" in lab or "net loss" in lab) and \
       "per share" not in lab and "eps" not in lab and "earnings per" not in lab:
        score -= 60

    # Prefer leftmost column (latest quarter usually leftmost)
    score += max(0, 5 - col_index)

    # Magnitude plausibility — EPS is almost always < 100 for normal stocks
    if abs(value) <= 50:
        score += 5
    elif abs(value) <= 100:
        score -= 10   # borderline — penalise but don't reject
    elif abs(value) <= 500:
        score -= 25
    else:
        score -= 50

    # Decimal bonus (EPS almost always has decimals like 0.89, 1.23)
    if "." in token_raw:
        score += 6
    else:
        # Integer with no decimal is likely a date fragment or footnote
        score -= 15

    # Footnote suppression (small integer in label cell)
    if origin_is_labelcell:
        stripped = re.sub(r"[^\d\-]", "", token_raw.strip())
        if re.fullmatch(r"\d{1,2}", stripped):
            score -= 100

    return score, best_j


# ── Table scanner ─────────────────────────────────────────────────────────────

def _scan_tables(soup: BeautifulSoup) -> list[Candidate]:
    candidates: list[Candidate] = []

    for table in soup.find_all("table"):
        for tr in table.find_all("tr"):
            cells = tr.find_all(["td", "th"])
            if len(cells) < 2:
                continue

            label    = cells[0].get_text(" ", strip=True)
            full_row = tr.get_text(" ", strip=True)
            low_full = full_row.lower()

            # Skip measurement-header rows (e.g. "Dollars in thousands, except per share data")
            # These contain "per share" but are NOT EPS rows — they are table headers.
            # This is the single most important gate: without it, date numbers (31, 30)
            # in the header row get picked up as EPS values.
            if any(pat.search(full_row) for pat in _MEASUREMENT_PATTERNS):
                continue

            # Gate: row must contain EPS-relevant language
            label_ok = any(t in label.lower() for t in EPS_TRIGGERS)
            row_ok   = "per share" in low_full and \
                       any(t in low_full for t in ["earnings", "income", "loss", "eps"])
            if not label_ok and not row_ok:
                continue
            if any(k in low_full for k in NON_EPS_KEYWORDS):
                continue

            # Harvest numbers from non-label cells
            tokens: list[tuple[float, int, str, bool]] = []
            for cidx, cell in enumerate(cells[1:], start=1):
                txt = cell.get_text(" ", strip=True).replace("\xa0", "")
                for tok in NUM_RE.findall(txt):
                    v = clean_num(tok)
                    if v is not None:
                        tokens.append((v, cidx - 1, tok, False))

            # Fallback: numbers from label cell
            if not tokens:
                for tok in NUM_RE.findall(cells[0].get_text(" ", strip=True)):
                    v = clean_num(tok)
                    if v is not None:
                        tokens.append((v, 0, tok, True))

            eps_type = (
                EPSType.BASIC   if "basic"   in label.lower() else
                EPSType.DILUTED if "dilut"   in label.lower() else
                EPSType.ADJUSTED if any(k in label.lower() for k in ADJUSTED_KEYWORDS) else
                EPSType.UNKNOWN
            )
            gaap_like = not any(k in (label + full_row).lower() for k in ADJUSTED_KEYWORDS)

            for ti, (val, cidx, raw, from_label) in enumerate(tokens):
                if val is None or math.isnan(val) or abs(val) > 5000:
                    continue
                s, j = _score(label, full_row, ti, val, eps_type, gaap_like, raw, from_label)
                candidates.append(Candidate(
                    value=val, eps_type=eps_type, gaap_like=gaap_like,
                    label=label, snippet=full_row[:500],
                    score=s, col_index=ti, source="regex_table",
                    best_j=j, token_raw=raw, origin_is_labelcell=from_label,
                ))

    return candidates


# ── Text fallback ─────────────────────────────────────────────────────────────

_TEXT_PATTERNS: list[tuple[re.Pattern, EPSType]] = [
    (re.compile(
        r"basic\s+(?:earnings|net\s+income|net\s+(?:loss|earnings))"
        r"(?:\s+\(loss\))?\s+per\s+(?:common\s+)?share"
        r"[\s:,\-—]{0,15}([\(]?\$?-?\d[\d,]*\.?\d*\)?)",
        re.I
    ), EPSType.BASIC),
    (re.compile(
        r"diluted\s+(?:earnings|net\s+income|net\s+(?:loss|earnings))"
        r"(?:\s+\(loss\))?\s+per\s+(?:common\s+)?share"
        r"[\s:,\-—]{0,15}([\(]?\$?-?\d[\d,]*\.?\d*\)?)",
        re.I
    ), EPSType.DILUTED),
    (re.compile(
        r"net\s+(?:income|loss)(?:\s+per\s+share)?"
        r"[\s:,\-—]{0,15}([\(]?\$?-?\d[\d,]*\.?\d*\)?)",
        re.I
    ), EPSType.UNKNOWN),
    (re.compile(
        r"earnings\s+per\s+(?:common\s+)?share"
        r"[\s:,\-—]{0,15}([\(]?\$?-?\d[\d,]*\.?\d*\)?)",
        re.I
    ), EPSType.UNKNOWN),
    (re.compile(
        r"loss\s+per\s+(?:common\s+)?share"
        r"[\s:,\-—]{0,15}([\(]?\$?-?\d[\d,]*\.?\d*\)?)",
        re.I
    ), EPSType.UNKNOWN),
]


def _scan_text(text: str) -> list[Candidate]:
    candidates: list[Candidate] = []
    for pat, eps_type in _TEXT_PATTERNS:
        for m in pat.finditer(text):
            num_tok = m.group(1)
            val     = clean_num(num_tok)
            if val is None or abs(val) > 50:
                continue
            label    = m.group(0)[:m.start(1) - m.start()].strip()
            full_str = m.group(0)
            if any(k in full_str.lower() for k in NON_EPS_KEYWORDS):
                continue
            gaap_like = not any(k in full_str.lower() for k in ADJUSTED_KEYWORDS)
            s, j = _score(label, full_str, 0, val, eps_type, gaap_like, num_tok, False)
            candidates.append(Candidate(
                value=val, eps_type=eps_type, gaap_like=gaap_like,
                label=label, snippet=full_str[:500],
                score=s, col_index=0, source="regex_text",
                best_j=j, token_raw=num_tok,
            ))
    return candidates


# ── Selector ──────────────────────────────────────────────────────────────────

def _select_best(candidates: list[Candidate]) -> Candidate | None:
    valid = [c for c in candidates
             if not any(k in (c.label + c.snippet).lower() for k in NON_EPS_KEYWORDS)
             and c.score >= _MIN_TRUST_SCORE]   # hard minimum — garbage below this
    if not valid:
        return None
    return max(valid, key=lambda c: (
        c.score,
        0 if c.gaap_like else -1,
        -c.col_index,
        0 if c.eps_type == EPSType.BASIC else 1,
    ))

# ── Public entry point ────────────────────────────────────────────────────────

def extract_eps(html: str) -> tuple["EPSResult | None", "EPSResult | None", "EPSResult | None", float]:
    """
    Extract basic, diluted and adjusted EPS from an HTML filing.

    Returns (basic, diluted, adjusted, best_score).
    best_score < cfg.LLM_CONFIDENCE_THRESHOLD -> caller should run LLM fallback.
    """
    soup = BeautifulSoup(html, "lxml")
    candidates = _scan_tables(soup)

    if not candidates:
        plain = soup.get_text(" ", strip=True)
        candidates = _scan_text(plain)

    if not candidates:
        return None, None, None, 0.0

    # Partition by EPS type
    basic_cands    = [c for c in candidates if c.eps_type == EPSType.BASIC]
    diluted_cands  = [c for c in candidates if c.eps_type == EPSType.DILUTED]
    adjusted_cands = [c for c in candidates if c.eps_type == EPSType.ADJUSTED]

    def _to_result(cand: "Candidate | None") -> "EPSResult | None":
        if cand is None:
            return None
        conf = min(100.0, max(0.0, cand.score))
        return EPSResult(
            value      = cand.value,
            eps_type   = cand.eps_type,
            confidence = conf,
            source     = cand.source,
            label      = cand.label,
            snippet    = cand.snippet,
        )

    # Fall back to UNKNOWN-typed candidates (text fallback) only — never
    # fall back to a DILUTED candidate for the basic slot, as that mislabels it.
    unknown_cands = [c for c in candidates if c.eps_type == EPSType.UNKNOWN]
    best_basic    = _select_best(basic_cands) or _select_best(unknown_cands)
    best_diluted  = _select_best(diluted_cands)
    best_adjusted = _select_best(adjusted_cands)

    top_score = max((c.score for c in candidates), default=0.0)

    return (
        _to_result(best_basic),
        _to_result(best_diluted),
        _to_result(best_adjusted),
        top_score,
    )
