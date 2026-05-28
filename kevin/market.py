"""
Market data — analyst consensus estimates via yfinance.

Supplements EDGAR data with EPS and revenue consensus for beat/miss computation.
All functions are synchronous; call via asyncio.to_thread() from async code.
"""
from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)


def fetch_consensus(ticker: str) -> dict[str, Any]:
    """
    Fetch analyst consensus estimates for *ticker*.

    Returns {"eps_estimate": float|None, "rev_estimate_mm": float|None}.
    All failures are caught; both fields default to None.

    Synchronous — use asyncio.to_thread(fetch_consensus, ticker) from async code.
    """
    result: dict[str, Any] = {"eps_estimate": None, "rev_estimate_mm": None}
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)

        # ── EPS estimate ──────────────────────────────────────────────────────
        # Primary: earnings_history carries quarterly epsEstimate (most accurate)
        try:
            hist = t.earnings_history
            if hist is not None and not hist.empty:
                col = next(
                    (c for c in ("epsEstimate", "eps_estimate") if c in hist.columns),
                    None,
                )
                if col:
                    valid = hist[hist[col].notna()]
                    if not valid.empty:
                        result["eps_estimate"] = float(valid[col].iloc[-1])
        except Exception:
            pass

        # Fallback: forwardEps / 4  (annual analyst estimate → quarterly proxy)
        if result["eps_estimate"] is None:
            try:
                fwd = t.info.get("forwardEps")
                if fwd:
                    result["eps_estimate"] = round(float(fwd) / 4, 4)
            except Exception:
                pass

        # ── Revenue estimate ──────────────────────────────────────────────────
        try:
            rev_df = t.revenue_estimate
            if rev_df is not None and not rev_df.empty and "avg" in rev_df.columns:
                # "0q" = current quarter; fall back to the first available row
                row = rev_df.loc["0q"] if "0q" in rev_df.index else rev_df.iloc[0]
                avg = float(row["avg"] if hasattr(row, "__getitem__") else row)
                result["rev_estimate_mm"] = round(avg / 1e6, 2)
        except Exception:
            pass

    except Exception as exc:
        log.debug("Consensus fetch failed for %s: %s", ticker, exc)

    return result
