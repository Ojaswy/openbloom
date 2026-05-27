# ◎ Kevin

> *"Actually, I'm kind of a big deal on Wall Street."*

**Kevin** is a production-grade SEC 8-K earnings intelligence engine built for quantitative hedge funds. It scrapes, parses, and analyses earnings filings from EDGAR in seconds — outputting structured signals ready to plug into any alpha strategy.

---

## Why 8-Ks?

The SEC 8-K (specifically **Item 2.02 – Results of Operations**) is the primary source for:

| Signal | Alpha Source |
|--------|-------------|
| EPS beat/miss vs estimate | Post-earnings drift (PEAD) |
| Revenue surprise | Revenue momentum strategies |
| Guidance revision | Analyst estimate revision chains |
| Management tone shift | NLP sentiment alpha |
| Filing timestamp (pre/post market) | Event-timing strategies |
| Risk flags (restatement, investigation) | Short-side triggers |

---

## Architecture

```
kevin/
├── kevin/
│   ├── config.py            # Env-var driven config (LLM provider, EDGAR UA, cache)
│   ├── models.py            # Pydantic v2 — EPSResult, Signal, KevinBrief, etc.
│   ├── edgar/
│   │   ├── feed.py          # EDGAR EFTS full-text search + CIK lookup
│   │   ├── fetcher.py       # Async HTTP fetcher, rate-limited (10 req/s), disk cache
│   │   └── classifier.py    # 8-K item type detection + market session
│   ├── parse/
│   │   ├── eps.py           # EPS extractor: table scan → text fallback
│   │   ├── metrics.py       # Revenue, margin, guidance extractors
│   │   └── llm_fallback.py  # LLM extraction when regex confidence < threshold
│   ├── analyze/
│   │   ├── sentiment.py     # Lexical tone analysis, risk flags, key phrases
│   │   └── signals.py       # Bull/bear scoring, beat/miss, Signal generation
│   └── pipeline.py          # Async orchestrator (ticker → list[KevinBrief])
└── cli.py                   # Click CLI: scan, filing, batch, watch
```

### Parsing pipeline

```
EDGAR EFTS search
       │
       ▼
CIK lookup + exhibit URL resolution
       │
       ▼
Async HTML fetch (cached to disk)
       │
       ▼
8-K item classifier  →  [earnings, M&A, restatement, ...]
       │
       ▼
EPS extractor (regex: table scan → text fallback)
       │  confidence < threshold?
       ▼
LLM fallback (Anthropic / Ollama / claude -p)
       │
       ▼
Metrics extractor (revenue, margin, guidance)
       │
       ▼
Tone analyser (bull/bear lexicon, risk flags, Q&A tension)
       │
       ▼
Signal generator  →  KevinBrief
```

---

## Quickstart

```bash
cd fin-engine/kevin
pip install -r requirements.txt

# Scan NVDA's last 30 days of 8-K earnings filings
python cli.py scan NVDA

# Multiple tickers, CSV output (quant-friendly)
python cli.py scan NVDA MSFT AAPL --out csv --save signals.csv

# Single filing by accession number
python cli.py filing 0001045810-25-000021 --ticker NVDA

# Enable LLM fallback (Anthropic, requires ANTHROPIC_API_KEY)
python cli.py scan TSLA --llm anthropic

# Enable LLM fallback (local Ollama, no key required)
python cli.py scan TSLA --llm ollama

# Bulk process from ticker list
echo -e "NVDA\nMSFT\nAAPL\nTSLA\nMETA" > tickers.txt
python cli.py batch tickers.txt --save morning_signals.csv

# Watch for new filings every 5 minutes
python cli.py watch NVDA MSFT --interval 300
```

### Demo (no setup required)

```bash
python examples/demo.py
```

---

## Output

### Table (default)

```
◎ KEVIN — Earnings Signal Summary
 Ticker  Filed       Session      EPS     Surprise   Rev $M   Bull  Bear  Conf  Flags
 NVDA    2025-02-26  post_market  $0.89   +4.7%      $39.3B   88    14    82    —
 MSFT    2025-01-29  post_market  $3.23   +3.9%      $69.6B   72    24    78    —
 TSLA    2025-01-29  post_market  $0.73   -5.2%      $25.7B   42    68    61    —
```

### Signal JSON (`--out signals`)

```json
{
  "ticker": "NVDA",
  "company": "NVIDIA Corporation",
  "accession": "0001045810-25-000021",
  "period": "Q4 FY2025",
  "filed_at": "2025-02-26T21:00:00+00:00",
  "market_session": "post_market",
  "eps": 0.89,
  "eps_surprise": 0.04,
  "eps_surprise_pct": 0.0471,
  "revenue_mm": 39331.0,
  "rev_surprise_pct": 0.034,
  "bull_score": 88.0,
  "bear_score": 14.0,
  "confidence": 82.0,
  "risk_flags": [],
  "items": ["2.02"],
  "llm_used": false,
  "exhibit_url": "https://www.sec.gov/Archives/edgar/data/..."
}
```

### CSV (`--out csv`)

One row per filing — flat signal table ideal for pandas or any quant pipeline.

---

## LLM Fallback

The EPS regex achieves ~94% accuracy on structured press releases. The remaining ~6% are edge cases: inline text-only releases, XBRL-only filings, non-standard tables.

When regex confidence falls below threshold (default: score < 40), Kevin optionally invokes an LLM:

| Provider | Config | Notes |
|----------|--------|-------|
| `anthropic` | `KEVIN_LLM=anthropic` + `ANTHROPIC_API_KEY` | claude-haiku-4-5, fast + cheap |
| `ollama` | `KEVIN_LLM=ollama` | Local llama3.1 via Ollama REST |
| `claude` | `KEVIN_LLM=claude` | Subprocess call to `claude -p` CLI |
| `none` | `KEVIN_LLM=none` (default) | Regex-only, no external calls |

The LLM sees only a ~2,500 character snippet of the filing — not the full document.

---

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `KEVIN_LLM` | `none` | LLM provider |
| `KEVIN_LLM_THRESHOLD` | `40` | Confidence score below which LLM is invoked |
| `KEVIN_ANTHROPIC_MODEL` | `claude-haiku-4-5-20251001` | Anthropic model |
| `KEVIN_OLLAMA_MODEL` | `llama3.1` | Ollama model |
| `KEVIN_OLLAMA_HOST` | `http://localhost:11434` | Ollama host |
| `KEVIN_EDGAR_UA` | `kevin-fin-engine ...` | EDGAR User-Agent (required by SEC ToS) |
| `KEVIN_EDGAR_RATE` | `8` | Max EDGAR requests per second |
| `KEVIN_CACHE_DIR` | `~/.kevin/cache` | Disk cache location |
| `KEVIN_LOOKBACK_DAYS` | `30` | Default lookback window |
| `KEVIN_CONCURRENCY` | `5` | Max concurrent filing downloads |

---

## Signals for quant strategies

Kevin outputs a `Signal` object per filing. Useful fields for common strategies:

**Post-earnings drift (PEAD)**
```python
signals = [b.to_signal_row() for b in briefs]
df = pd.DataFrame(signals)
df["direction"] = df["eps_surprise_pct"].apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0)
```

**Pre-market vs post-market alpha**
```python
pre = df[df["market_session"] == "pre_market"]
post = df[df["market_session"] == "post_market"]
```

**Risk flag short screen**
```python
risky = df[df["risk_flags"].str.contains("restatement|going_concern|sec_investigation")]
```

**Sentiment momentum**
```python
df["sentiment_score"] = df["bull_score"] - df["bear_score"]
```

---

## Requirements

- Python 3.11+
- No API key required for EDGAR scraping
- Optional: `ANTHROPIC_API_KEY` for LLM fallback
- Optional: [Ollama](https://ollama.ai) running locally for local LLM fallback

---

*Kevin is named after Kevin Malone from The Office — who, despite appearances, always knew exactly where the money was.*
