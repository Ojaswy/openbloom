#!/usr/bin/env python3
"""
Kevin quick-start demo.

Runs a full pipeline on NVIDIA's most recent 8-K filing:
  1. Fetches the filing from EDGAR (no API key required)
  2. Parses EPS, revenue, guidance, gross margin using regex
  3. Analyses management tone
  4. Generates a tradable Signal

Run from the kevin/ directory:
    pip install -r requirements.txt
    python examples/demo.py
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

# Ensure the kevin package is importable from this location
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kevin.pipeline import analyse_ticker


async def main() -> None:
    print("* KEVIN - Demo  (EDGAR live fetch, no API key required)\n")

    ticker     = "NVDA"
    since_days = 365   # look back a full year to ensure at least one filing is returned

    print(f"Fetching recent 8-K earnings filings for {ticker}...")
    briefs = await analyse_ticker(
        ticker,
        since_days    = since_days,
        earnings_only = True,
        max_filings   = 3,
    )

    if not briefs:
        print(f"No earnings filings found for {ticker} in the last {since_days} days.")
        print("Try increasing --since or check EDGAR directly.")
        return

    count = len(briefs)
    print(f"Found {count} filing{'s' if count != 1 else ''}.\n")

    for brief in briefs:
        s = brief.signal
        m = brief.metrics
        t = brief.tone

        print("-" * 60)
        print(f"  Ticker     : {s.ticker}")
        print(f"  Company    : {s.company}")
        print(f"  Filed      : {s.filed_at.strftime('%Y-%m-%d')}  ({s.market_session.value})")
        print(f"  Period     : {s.period or '—'}")
        print()
        print(f"  EPS        : ${s.eps:.2f}" if s.eps else "  EPS        : —")
        if s.eps_surprise_pct is not None:
            sign = "+" if s.eps_surprise_pct >= 0 else ""
            word = "BEAT" if s.eps_surprise_pct >= 0 else "MISS"
            print(f"  EPS Surp.  : {sign}{s.eps_surprise_pct*100:.2f}%  [{word}]")
        if m.revenue_mm:
            rev_b = m.revenue_mm / 1000
            print(f"  Revenue    : ${rev_b:.2f}B")
        if m.guidance and m.guidance.midpoint:
            print(f"  Guidance   : ${m.guidance.midpoint:.2f}/share  "
                  f"[{m.guidance.lo:.2f}–{m.guidance.hi:.2f}]")
        if m.gross_margin:
            print(f"  Gross Marg : {m.gross_margin*100:.1f}%")
        print()
        print(f"  Bull Score : {s.bull_score:.0f}/100")
        print(f"  Bear Score : {s.bear_score:.0f}/100")
        print(f"  Confidence : {s.confidence:.0f}/100")
        if s.risk_flags:
            print(f"  Risk Flags : {', '.join(s.risk_flags)}")
        print()
        print(f"  Verdict    : {s.verdict}")
        print()
        if t.key_phrases:
            print("  Key phrases:")
            for phrase in t.key_phrases[:3]:
                print(f"    > {phrase}")
        if brief.errors:
            print(f"  [warnings] : {', '.join(brief.errors)}")
        print()

    # Export full signal list as JSON
    signals = [b.to_signal_row() for b in briefs]
    out_path = Path(__file__).parent / "demo_signals.json"
    out_path.write_text(json.dumps(signals, indent=2, default=str))
    print(f"Signal JSON written to: {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
