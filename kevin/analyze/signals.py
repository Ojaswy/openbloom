"""
Signal generation — converts parsed metrics + tone into a tradable Signal.

Bull/bear scores are computed from a weighted combination of:
  - EPS beat/miss vs estimate
  - Revenue beat/miss vs estimate
  - Guidance delta (raised / lowered / maintained)
  - Tone sentiment (bull/bear word ratio)
  - Risk flag count (each flag depresses confidence)
  - Market session (pre/post market amplifies impact)

The output Signal is the atom that a quant strategy consumes.
All scores are 0-100 integers; nulls are explicit.
"""
from __future__ import annotations

import math
from datetime import datetime

from kevin.analyze.sentiment import ToneAnalysis
from kevin.edgar.classifier import get_market_session
from kevin.models import (
    FilingIndex,
    FilingMetrics,
    MarketSession,
    Sentiment,
    Signal,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_pct(actual: float | None, estimate: float | None) -> float | None:
    """(actual - estimate) / |estimate|, or None if either is missing."""
    if actual is None or estimate is None or estimate == 0:
        return None
    return (actual - estimate) / abs(estimate)


def _clamp(v: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, v))


# ── Scoring weights ───────────────────────────────────────────────────────────
# Each weight is the maximum number of bull/bear points that component can
# contribute.  They sum to 100 on each side.

W_EPS_BEAT      = 30   # EPS beat/miss
W_REV_BEAT      = 20   # Revenue beat/miss
W_GUIDANCE      = 25   # Guidance raised/maintained/lowered
W_TONE          = 15   # Sentiment word ratio
W_CERTAINTY     = 10   # Management certainty score


def _eps_component(metrics: FilingMetrics) -> tuple[float, float]:
    """Returns (bull_pts, bear_pts) for the EPS beat/miss."""
    pct = _safe_pct(metrics.eps_primary, metrics.eps_estimate)
    if pct is None:
        return 0.0, 0.0

    # Sigmoid-like mapping: ±5% → ~50% of weight, ±20%+ → ~100%
    magnitude = _clamp(abs(pct) / 0.20 * W_EPS_BEAT, 0, W_EPS_BEAT)
    if pct > 0:
        return magnitude, 0.0
    else:
        return 0.0, magnitude


def _rev_component(metrics: FilingMetrics) -> tuple[float, float]:
    """Returns (bull_pts, bear_pts) for revenue beat/miss."""
    pct = _safe_pct(metrics.revenue_mm, metrics.rev_estimate_mm)
    if pct is None:
        # Partial signal from YoY growth if no estimate
        if metrics.revenue_yoy_pct is not None:
            g = metrics.revenue_yoy_pct
            score = _clamp(abs(g) / 0.30 * W_REV_BEAT, 0, W_REV_BEAT)
            return (score, 0.0) if g > 0 else (0.0, score)
        return 0.0, 0.0

    magnitude = _clamp(abs(pct) / 0.10 * W_REV_BEAT, 0, W_REV_BEAT)
    return (magnitude, 0.0) if pct > 0 else (0.0, magnitude)


def _guidance_component(metrics: FilingMetrics) -> tuple[float, float]:
    """Returns (bull_pts, bear_pts) for guidance signal."""
    g = metrics.guidance
    if g is None:
        return 0.0, 0.0

    if g.is_raised is True:
        return float(W_GUIDANCE), 0.0
    if g.is_raised is False:
        return 0.0, float(W_GUIDANCE)

    # Guidance midpoint vs current EPS as proxy for direction
    if g.midpoint is not None and metrics.eps_primary is not None:
        delta_pct = (g.midpoint - metrics.eps_primary) / abs(metrics.eps_primary) \
                    if metrics.eps_primary != 0 else 0
        pts = _clamp(abs(delta_pct) / 0.15 * W_GUIDANCE, 0, W_GUIDANCE)
        return (pts, 0.0) if delta_pct > 0 else (0.0, pts)

    # Maintained guidance → small bull signal (certainty)
    return float(W_GUIDANCE * 0.4), 0.0


def _tone_component(tone: ToneAnalysis) -> tuple[float, float]:
    """Returns (bull_pts, bear_pts) from sentiment word ratio."""
    total = tone.bull_word_count + tone.bear_word_count
    if total == 0:
        return 0.0, 0.0

    bull_ratio = tone.bull_word_count / total
    bear_ratio = tone.bear_word_count / total

    bull_pts = _clamp(bull_ratio * W_TONE * 2, 0, W_TONE)
    bear_pts = _clamp(bear_ratio * W_TONE * 2, 0, W_TONE)
    return bull_pts, bear_pts


def _certainty_component(tone: ToneAnalysis) -> tuple[float, float]:
    """High certainty → small bull boost; low certainty → small bear drag."""
    score = tone.certainty_score / 100.0
    if score > 0.6:
        return (score - 0.6) / 0.4 * W_CERTAINTY, 0.0
    elif score < 0.4:
        return 0.0, (0.4 - score) / 0.4 * W_CERTAINTY
    return 0.0, 0.0


def _risk_penalty(tone: ToneAnalysis) -> float:
    """Each risk flag subtracts from the final confidence score."""
    return min(40.0, len(tone.risk_flags) * 10.0)


# ── Verdict prose ─────────────────────────────────────────────────────────────

def _verdict(
    bull: float,
    bear: float,
    metrics: FilingMetrics,
    tone: ToneAnalysis,
    index: FilingIndex,
) -> str:
    direction = "bullish" if bull > bear + 10 else "bearish" if bear > bull + 10 else "mixed"

    eps_str = ""
    if metrics.eps_primary is not None:
        eps_str = f"EPS of ${metrics.eps_primary:.2f}"
        if metrics.eps_estimate is not None:
            diff   = metrics.eps_primary - metrics.eps_estimate
            symbol = "beat" if diff > 0 else "missed"
            eps_str += f" ({symbol} estimate by ${abs(diff):.2f})"

    rev_str = ""
    if metrics.revenue_mm is not None:
        rev_m = metrics.revenue_mm
        if rev_m >= 1_000:
            rev_str = f"revenue of ${rev_m/1000:.2f}B"
        else:
            rev_str = f"revenue of ${rev_m:.0f}M"

    guide_str = ""
    if metrics.guidance and metrics.guidance.midpoint is not None:
        guide_str = f"Guidance midpoint ${metrics.guidance.midpoint:.2f}/share"
        if metrics.guidance.is_raised:
            guide_str += " (raised)"
        elif metrics.guidance.is_raised is False:
            guide_str += " (lowered)"

    risk_str = ""
    if tone.risk_flags:
        risk_str = f" Risk flags: {', '.join(tone.risk_flags)}."

    parts = [p for p in [eps_str, rev_str, guide_str] if p]
    body  = ". ".join(parts)

    return f"{index.company} — {direction.upper()}. {body}.{risk_str}".strip()


# ── Main entry point ──────────────────────────────────────────────────────────

def generate_signal(
    index:   FilingIndex,
    metrics: FilingMetrics,
    tone:    ToneAnalysis,
    *,
    exhibit_url: str | None = None,
    llm_used:    bool       = False,
    ticker:      str | None = None,
) -> Signal:
    """
    Synthesise a tradable Signal from index metadata, parsed metrics,
    and tone analysis.
    """
    # ── Component scoring ──────────────────────────────────────────────────
    bull, bear = 0.0, 0.0

    for fn in (_eps_component, _rev_component, _guidance_component):
        b, r = fn(metrics)
        bull += b
        bear += r

    b, r = _tone_component(tone)
    bull += b
    bear += r

    b, r = _certainty_component(tone)
    bull += b
    bear += r

    bull = _clamp(bull)
    bear = _clamp(bear)

    # ── Confidence ────────────────────────────────────────────────────────
    # Base: average of best EPS extraction confidence + tone certainty
    eps_conf = 0.0
    for er in [metrics.eps_basic, metrics.eps_diluted]:
        if er is not None:
            eps_conf = max(eps_conf, er.confidence)

    base_conf  = (eps_conf + tone.certainty_score) / 2
    risk_pen   = _risk_penalty(tone)
    confidence = _clamp(base_conf - risk_pen)

    # Q&A tension penalty: elevated analyst pushback further reduces confidence
    # (tension_score > 50 → up to 15-point drag, proportional above the threshold)
    if tone.qa_tension_score is not None and tone.qa_tension_score > 50:
        tension_pen = _clamp((tone.qa_tension_score - 50) / 50 * 15, 0.0, 15.0)
        confidence  = _clamp(confidence - tension_pen)

    # ── Surprise metrics ──────────────────────────────────────────────────
    eps_surprise     = None
    eps_surprise_pct = None
    if metrics.eps_primary is not None and metrics.eps_estimate is not None:
        eps_surprise     = round(metrics.eps_primary - metrics.eps_estimate, 4)
        eps_surprise_pct = _safe_pct(metrics.eps_primary, metrics.eps_estimate)
        if eps_surprise_pct is not None:
            eps_surprise_pct = round(eps_surprise_pct, 4)

    rev_surprise_pct = _safe_pct(metrics.revenue_mm, metrics.rev_estimate_mm)
    if rev_surprise_pct is not None:
        rev_surprise_pct = round(rev_surprise_pct, 4)

    # ── Market session ────────────────────────────────────────────────────
    session = get_market_session(index.filed_at)

    # ── Items as strings ──────────────────────────────────────────────────
    item_strs = [item.value for item in index.items]

    return Signal(
        ticker           = ticker or index.ticker or "UNKNOWN",
        company          = index.company,
        accession        = index.accession,
        period           = index.period,
        filed_at         = index.filed_at,
        market_session   = session,
        eps              = metrics.eps_primary,
        eps_surprise     = eps_surprise,
        eps_surprise_pct = eps_surprise_pct,
        revenue_mm       = metrics.revenue_mm,
        rev_surprise_pct = rev_surprise_pct,
        bull_score       = round(bull, 1),
        bear_score       = round(bear, 1),
        confidence       = round(confidence, 1),
        verdict          = _verdict(bull, bear, metrics, tone, index),
        risk_flags       = tone.risk_flags,
        items            = item_strs,
        exhibit_url      = exhibit_url,
        llm_used         = llm_used,
    )
