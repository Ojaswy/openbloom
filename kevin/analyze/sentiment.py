"""
Lexical sentiment & tone analysis of 8-K earnings press releases.

No ML model required - pure word-count heuristics calibrated to the
financial filing register.  Fast enough to run on every document.

Output: ToneAnalysis with:
  - bull/bear/hedge word counts
  - certainty score (0-100)
  - dominant sentiment enum
  - risk flags (restatement, investigation, going-concern, etc.)
  - key bullish/bearish phrases extracted from the text
  - qa_tension_score: estimated from Q&A section language
"""
from __future__ import annotations

import re

from kevin.models import Sentiment, ToneAnalysis


# Lexicons

BULL_WORDS: frozenset[str] = frozenset({
    "record", "strong", "exceeded", "outperformed", "beat", "robust",
    "accelerat", "growth", "momentum", "inflection", "ramp", "expand",
    "margin expansion", "raised", "increase", "raised guidance", "ahead of",
    "above", "exceeded expectations", "exceptional", "outstanding", "solid",
    "breakthrough", "milestone", "demand", "backlog", "win", "awarded",
    "strategic", "traction", "adoption", "broad-based", "broad based",
    "diversified", "resilient", "confidence", "optimistic", "excited",
    "strong pipeline", "all-time high", "all time high", "record revenue",
    "record earnings", "record bookings", "revenue growth", "earnings growth",
})

BEAR_WORDS: frozenset[str] = frozenset({
    "decline", "miss", "below", "disappoint", "shortfall", "headwind",
    "pressure", "weak", "softness", "slowdown", "decelerat", "contraction",
    "loss", "impairment", "write-down", "write down", "restructur",
    "layoff", "reduction in force", "workforce reduction", "inventory",
    "destocking", "macro", "uncertain", "challenging", "difficult",
    "lower than expected", "worse than", "fell short", "reduced guidance",
    "lowered guidance", "cut guidance", "withdrew guidance", "suspended",
    "delayed", "cancelled", "terminated", "litigation", "lawsuit",
    "investigation", "subpoena", "violation", "fine", "penalty",
    "default", "covenant", "liquidity", "solvency", "dilution",
})

HEDGE_WORDS: frozenset[str] = frozenset({
    "may", "might", "could", "should", "would", "subject to", "contingent",
    "if", "assuming", "depends", "uncertain", "potential", "possible",
    "anticipate", "expect", "believe", "estimate", "approximately",
    "around", "roughly", "guidance range", "range of", "between",
    "forward-looking", "forward looking", "risk", "no assurance",
    "no guarantee", "there can be no", "we cannot predict", "we cannot assure",
})

# Risk flag patterns - each is a (label, regex) pair

_RISK_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("restatement",       re.compile(r"restat(?:e|ed|ement|ing)", re.I)),
    ("material_weakness", re.compile(r"material\s+weakness", re.I)),
    ("going_concern",     re.compile(r"going.concern|substantial\s+doubt", re.I)),
    ("sec_investigation", re.compile(r"SEC\s+(?:investigation|inquiry|subpoena|formal\s+order|enforcement)", re.I)),
    ("doj_investigation", re.compile(r"DOJ\s+(?:investigation|subpoena)|department\s+of\s+justice\s+(?:investigation|subpoena)", re.I)),
    # Require active filing language, not just boilerplate risk-factor mention
    ("class_action",      re.compile(
        r"(?:filed|commenced|initiated|pending|alleged)\s+.{0,60}class.action"
        r"|class.action\s+.{0,60}(?:filed|commenced|pending|alleged|against\s+(?:us|the\s+company))"
        r"|securities\s+(?:fraud\s+)?(?:lawsuit|litigation)\s+(?:filed|pending|alleged)",
        re.I
    )),
    ("guidance_withdrawn", re.compile(r"withdraw(?:n|ing)?\s+(?:its\s+)?(?:full.year\s+)?guidance", re.I)),
    ("covenant_breach",   re.compile(r"covenant\s+(?:breach|waiver|violation|default)", re.I)),
    ("impairment",        re.compile(r"goodwill\s+impairment|impairment\s+charge", re.I)),
    ("ceo_departure",     re.compile(r"CEO.{0,30}(?:resigned|resign|depart|retirement|stepping\s+down)", re.I)),
    ("workforce_cut",     re.compile(r"workforce\s+reduction|reduction\s+in\s+force|layoff|\brif\b", re.I)),
    ("delisting_risk",    re.compile(r"delist(?:ing)?|nasdaq\s+notice|nyse\s+notice", re.I)),
]


# Q&A tension heuristics

_QA_MARKERS = re.compile(
    r"(?:^|\n)\s*(?:Q:|Analyst:|Question:)", re.I | re.M
)
_TENSION_WORDS = re.compile(
    r"(?:concerned|worried|declining|slowdown|pressure|miss|why did|why has"
    r"|what happened|challenge|disappoint|can you explain|unpack|pushback"
    r"|credibility|guidance cut|revised down|why the change|clarify)",
    re.I
)


def _qa_tension(text: str) -> float | None:
    """Estimate analyst tension in the Q&A transcript section (0-100)."""
    m = _QA_MARKERS.search(text)
    if not m:
        return None
    qa_text = text[m.start():]

    q_count      = len(_QA_MARKERS.findall(qa_text))
    tension_hits = len(_TENSION_WORDS.findall(qa_text))
    if q_count == 0:
        return None

    # Rate per question, capped at 100
    rate = min(100.0, (tension_hits / q_count) * 25)
    return round(rate, 1)


# Key phrase extraction

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _key_phrases(text: str, n: int = 5) -> list[str]:
    """
    Extract the most signal-rich sentences (highest combined bull+bear density).
    Returns up to n sentences, each <= 200 chars.
    """
    sentences = _SENT_SPLIT.split(text)
    scored: list[tuple[float, str]] = []

    for sent in sentences:
        sent = sent.strip()
        if len(sent) < 20 or len(sent) > 500:
            continue
        low = sent.lower()
        bull_hits = sum(1 for w in BULL_WORDS if w in low)
        bear_hits = sum(1 for w in BEAR_WORDS if w in low)
        if bull_hits + bear_hits == 0:
            continue
        scored.append((bull_hits + bear_hits * 1.5, sent[:200]))

    scored.sort(key=lambda x: -x[0])
    return [s for _, s in scored[:n]]


# Main public function

def analyse_tone(text: str) -> ToneAnalysis:
    """
    Full tone analysis of a filing text.
    text should be plain text (HTML already stripped).
    """
    low = text.lower()

    # Word counts — substring match catches stems (e.g. "restructur" -> "restructuring")
    bull_count  = sum(1 for w in BULL_WORDS  if w in low)
    bear_count  = sum(1 for w in BEAR_WORDS  if w in low)
    hedge_count = sum(1 for w in HEDGE_WORDS if w in low)

    # Risk flags
    risk_flags: list[str] = [
        label for label, pat in _RISK_PATTERNS if pat.search(text)
    ]

    # Dominant sentiment
    if bull_count > bear_count * 1.4:
        dominant = Sentiment.BULLISH
    elif bear_count > bull_count * 1.4:
        dominant = Sentiment.BEARISH
    elif abs(bull_count - bear_count) <= max(2, (bull_count + bear_count) * 0.15):
        dominant = Sentiment.MIXED
    else:
        dominant = Sentiment.NEUTRAL

    # Certainty score: inversely related to hedge density
    total_words   = max(sum(1 for _ in re.finditer(r"\b\w+\b", low)), 1)
    hedge_density = hedge_count / total_words * 1000   # per-thousand words
    certainty     = max(0.0, min(100.0, 100 - hedge_density * 8))

    return ToneAnalysis(
        bull_word_count    = bull_count,
        bear_word_count    = bear_count,
        hedge_word_count   = hedge_count,
        certainty_score    = round(certainty, 1),
        dominant_sentiment = dominant,
        risk_flags         = risk_flags,
        qa_tension_score   = _qa_tension(text),
        key_phrases        = _key_phrases(text),
    )
