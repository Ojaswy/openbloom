"""Pydantic v2 data models — every field quant-friendly (explicit nulls, no ambiguity)."""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ── Enums ─────────────────────────────────────────────────────────────────────

class EPSType(str, Enum):
    BASIC         = "basic"
    DILUTED       = "diluted"
    BASIC_DILUTED = "basic_diluted"   # reported as one combined figure
    ADJUSTED      = "adjusted"        # non-GAAP
    UNKNOWN       = "unknown"


class FilingItemType(str, Enum):
    """Subset of SEC 8-K item types Kevin cares about."""
    EARNINGS        = "2.02"   # Results of Operations / Earnings
    MATERIAL_EVENT  = "8.01"   # Other events (M&A rumours, FDA, etc.)
    MA_AGREEMENT    = "1.01"   # Entry into material agreement
    DEPARTURE       = "5.02"   # C-suite departure
    RESTATEMENT     = "4.02"   # Non-reliance on financial statements
    AMENDMENT       = "8.01A"
    OTHER           = "other"


class Sentiment(str, Enum):
    BULLISH  = "bullish"
    BEARISH  = "bearish"
    NEUTRAL  = "neutral"
    MIXED    = "mixed"


class MarketSession(str, Enum):
    PRE_MARKET  = "pre_market"    # Filed before 09:30 ET
    POST_MARKET = "post_market"   # Filed after 16:00 ET
    INTRADAY    = "intraday"
    WEEKEND     = "weekend"
    UNKNOWN     = "unknown"


# ── Sub-models ────────────────────────────────────────────────────────────────

class EPSResult(BaseModel):
    """Single extracted EPS figure with extraction provenance."""
    value:       float | None
    eps_type:    EPSType
    confidence:  float            = Field(ge=0, le=100, description="0-100 extraction confidence")
    source:      str              = Field(description="'regex_table' | 'regex_text' | 'llm'")
    label:       str | None       = Field(None, description="Row label from filing table")
    snippet:     str | None       = Field(None, description="Raw text around the match (≤500 chars)")


class GuidanceRange(BaseModel):
    lo:          float | None
    hi:          float | None
    midpoint:    float | None
    metric:      str              = "eps"   # eps | revenue | margin
    period:      str | None       = None    # e.g. "Q2 FY2025"
    is_raised:   bool | None      = None    # True if raised vs prior guidance


class FilingMetrics(BaseModel):
    """All structured metrics extracted from a single filing document."""
    # EPS
    eps_basic:        EPSResult | None = None
    eps_diluted:      EPSResult | None = None
    eps_adjusted:     EPSResult | None = None
    eps_primary:      float | None     = None   # Best available EPS value (GAAP basic preferred)

    # Revenue
    revenue_mm:       float | None     = None   # USD millions
    revenue_yoy_pct:  float | None     = None   # e.g. 0.12 = +12%

    # Margin
    gross_margin:     float | None     = None   # 0-1
    operating_margin: float | None     = None   # 0-1

    # Guidance
    guidance:         GuidanceRange | None = None

    # Consensus (populated if available from external source)
    eps_estimate:     float | None     = None
    rev_estimate_mm:  float | None     = None


class ToneAnalysis(BaseModel):
    """Lexical tone/sentiment analysis of the filing text."""
    bull_word_count:    int
    bear_word_count:    int
    hedge_word_count:   int           # "may", "could", "subject to", etc.
    certainty_score:    float         = Field(ge=0, le=100)
    dominant_sentiment: Sentiment
    risk_flags:         list[str]     = Field(
        default_factory=list,
        description="Detected material risk phrases (restatement, investigation, going concern, etc.)"
    )
    qa_tension_score:   float | None  = Field(
        None, ge=0, le=100,
        description="Estimated analyst-management tension in Q&A section"
    )
    key_phrases:        list[str]     = Field(default_factory=list)


class Signal(BaseModel):
    """
    Tradable signal output — what a quant strategy actually consumes.
    Everything here is timestamp-anchored and null-explicit.
    """
    ticker:           str
    company:          str
    accession:        str           # EDGAR accession number (canonical ID)
    period:           str | None    # Reporting period e.g. "Q1 FY2025"
    filed_at:         datetime      # UTC filing timestamp
    market_session:   MarketSession

    # Core metrics
    eps:              float | None
    eps_surprise:     float | None  # actual - estimate (null if no estimate)
    eps_surprise_pct: float | None  # (actual - estimate) / abs(estimate)
    revenue_mm:       float | None
    rev_surprise_pct: float | None

    # Scores  (0-100)
    bull_score:       float
    bear_score:       float
    confidence:       float         # Overall extraction + analysis confidence

    # Qualitative
    verdict:          str | None    = None
    risk_flags:       list[str]     = Field(default_factory=list)
    items:            list[str]     = Field(default_factory=list, description="8-K items detected")

    # Provenance
    exhibit_url:      str | None    = None
    llm_used:         bool          = False


class FilingIndex(BaseModel):
    """Raw metadata from EDGAR before any parsing."""
    accession:    str
    cik:          str
    ticker:       str | None
    company:      str
    filed_at:     datetime
    period:       str | None
    items:        list[FilingItemType] = Field(default_factory=list)
    exhibit_urls: list[str]            = Field(default_factory=list)
    form:         str = "8-K"


class KevinBrief(BaseModel):
    """Full analysis output — the complete Kevin record for one filing."""
    index:    FilingIndex
    metrics:  FilingMetrics
    tone:     ToneAnalysis
    signal:   Signal
    raw_text_chars: int = 0          # Size of the exhibit text processed
    parsed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    errors:    list[str] = Field(default_factory=list)   # Non-fatal warnings

    def to_signal_row(self) -> dict[str, Any]:
        """Flat dict suitable for CSV export of the signal layer."""
        s = self.signal
        return {
            "ticker":           s.ticker,
            "company":          s.company,
            "accession":        s.accession,
            "period":           s.period,
            "filed_at":         s.filed_at.isoformat(),
            "market_session":   s.market_session.value,
            "eps":              s.eps,
            "eps_surprise":     s.eps_surprise,
            "eps_surprise_pct": s.eps_surprise_pct,
            "revenue_mm":       s.revenue_mm,
            "rev_surprise_pct": s.rev_surprise_pct,
            "bull_score":       s.bull_score,
            "bear_score":       s.bear_score,
            "confidence":       s.confidence,
            "risk_flags":       "|".join(s.risk_flags),
            "items":            "|".join(s.items),
            "llm_used":         s.llm_used,
            "exhibit_url":      s.exhibit_url,
        }
