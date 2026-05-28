#!/usr/bin/env python3
"""
Kevin Terminal — Bloomberg-style financial intelligence TUI v2

Usage:
    python tui.py                            # 20-stock default watchlist
    python tui.py NVDA TSLA AMD              # custom watchlist
    python tui.py --llm anthropic            # enable Anthropic Claude
    python tui.py --llm ollama               # use local Ollama
    python tui.py --llm ollama --model llama3.2

Keys:
    q / Ctrl+C   Quit          r   Hard refresh
    /            Chat input    ↑↓  Navigate watchlist
    Escape       Back to list
"""
from __future__ import annotations

import asyncio
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

# ── Dependency gate ───────────────────────────────────────────────────────────
def _require(*pkgs: str) -> None:
    missing = [p for p in pkgs if not __import__("importlib").util.find_spec(p.split(">=")[0])]
    if missing:
        print(f"pip install {' '.join(missing)}")
        sys.exit(1)

_require("textual", "yfinance")

from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import DataTable, Footer, Input, RichLog, Static
from rich.text import Text
import yfinance as yf

from kevin.config import LLMProvider, cfg

# ── Default 20-stock watchlist ────────────────────────────────────────────────
DEFAULT_TICKERS = [
    # Mega-cap tech
    "NVDA", "MSFT", "AAPL", "GOOGL", "AMZN", "META", "TSLA",
    # Semis
    "AMD", "AVGO", "TSM",
    # Finance
    "JPM", "GS", "V",
    # Healthcare / Pharma
    "LLY", "UNH",
    # Energy
    "XOM",
    # Consumer / Other
    "COST", "WMT",
    # ETFs (market pulse)
    "SPY", "QQQ",
]
CHART_PERIOD   = "30d"
PRICE_REFRESH  = 60   # seconds
MAX_FILINGS    = 5

# ── yfinance data fetchers ────────────────────────────────────────────────────
def _safe(v: Any) -> float | None:
    if v is None:
        return None
    try:
        f = float(v)
        return None if math.isnan(f) else f
    except (TypeError, ValueError):
        return None

def _fetch_quote(ticker: str) -> dict[str, Any]:
    """Synchronous — runs in thread pool."""
    try:
        t  = yf.Ticker(ticker)
        fi = t.fast_info
        price = _safe(getattr(fi, "last_price",    None))
        prev  = _safe(getattr(fi, "previous_close", None))
        dh    = _safe(getattr(fi, "day_high",        None))
        dl    = _safe(getattr(fi, "day_low",         None))
        vol   = _safe(getattr(fi, "last_volume",     None))
        avg3m = _safe(getattr(fi, "three_month_average_volume", None))
        w52h  = _safe(getattr(fi, "fifty_two_week_high", None))
        w52l  = _safe(getattr(fi, "fifty_two_week_low",  None))
        mktcap= _safe(getattr(fi, "market_cap",     None))

        # Fallback for stale/NaN price
        if price is None:
            h = t.history(period="2d")
            if not h.empty:
                price = float(h["Close"].iloc[-1])
                prev  = float(h["Close"].iloc[-2]) if len(h) > 1 else price

        if price is None:
            return {"ticker": ticker, "ok": False}

        prev     = prev or price
        chg      = price - prev
        pct      = chg / prev * 100 if prev else 0.0
        vol_ratio= (vol / avg3m) if (vol and avg3m and avg3m > 0) else None
        day_rng  = ((dh - dl) / price * 100) if (dh and dl and price) else None

        return dict(
            ticker=ticker, ok=True,
            price=price, change=chg, pct=pct,
            day_high=dh, day_low=dl, day_range_pct=day_rng,
            vol=vol, avg_vol=avg3m, vol_ratio=vol_ratio,
            wk52_high=w52h, wk52_low=w52l, mktcap=mktcap,
        )
    except Exception:
        return {"ticker": ticker, "ok": False}

def _fetch_ohlcv(ticker: str) -> list[dict]:
    """Synchronous — runs in thread pool."""
    try:
        hist = yf.Ticker(ticker).history(period=CHART_PERIOD, interval="1d", auto_adjust=True)
        return [
            dict(
                date=str(ts.date()),
                open=float(r["Open"]), high=float(r["High"]),
                low=float(r["Low"]),   close=float(r["Close"]),
                volume=float(r["Volume"]),
            )
            for ts, r in hist.iterrows()
        ]
    except Exception:
        return []

# ── Technical indicators ──────────────────────────────────────────────────────
def _calc_technicals(candles: list[dict], quote: dict) -> dict[str, Any]:
    if not candles:
        return {}

    closes = [c["close"]  for c in candles]
    highs  = [c["high"]   for c in candles]
    lows   = [c["low"]    for c in candles]
    vols   = [c["volume"] for c in candles]
    result: dict[str, Any] = {}

    # ── RSI (14) ──────────────────────────────────────
    if len(closes) >= 15:
        deltas = [closes[i] - closes[i-1] for i in range(1, len(closes))]
        gains  = [max(d, 0)  for d in deltas[-14:]]
        losses = [-min(d, 0) for d in deltas[-14:]]
        ag, al = sum(gains)/14, sum(losses)/14
        result["rsi"] = 100.0 if al == 0 else round(100 - 100/(1 + ag/al), 1)

    # ── ATR (14) ──────────────────────────────────────
    if len(candles) >= 15:
        trs = [
            max(highs[i]-lows[i], abs(highs[i]-closes[i-1]), abs(lows[i]-closes[i-1]))
            for i in range(1, len(candles))
        ]
        result["atr"]     = round(sum(trs[-14:])/14, 2)
        result["atr_pct"] = round(result["atr"] / closes[-1] * 100, 2)

    # ── Momentum ──────────────────────────────────────
    if len(closes) >= 6:
        result["mom5"]  = round((closes[-1]-closes[-6])/closes[-6]*100, 2)
    if len(closes) >= 11:
        result["mom10"] = round((closes[-1]-closes[-11])/closes[-11]*100, 2)

    # ── VWAP (30-day) ─────────────────────────────────
    if vols:
        typicals = [(highs[i]+lows[i]+closes[i])/3 for i in range(len(candles))]
        total_v  = sum(vols)
        if total_v > 0:
            vwap = sum(t*v for t, v in zip(typicals, vols)) / total_v
            result["vwap"]     = round(vwap, 2)
            result["vwap_dev"] = round((closes[-1]-vwap)/vwap*100, 2)

    # ── 52-week position ──────────────────────────────
    w52h = quote.get("wk52_high") or max(highs)
    w52l = quote.get("wk52_low")  or min(lows)
    if w52h > w52l:
        result["wk52_pos"]  = round((closes[-1]-w52l)/(w52h-w52l)*100, 1)
        result["wk52_high"] = w52h
        result["wk52_low"]  = w52l

    # ── Price velocity (|daily move| / ATR — > 1 = strong) ──
    if "atr" in result and result["atr"] > 0:
        result["velocity"] = round(abs(quote.get("change", 0)) / result["atr"], 2)

    # ── Gap at open ───────────────────────────────────
    if len(candles) >= 2:
        result["gap_pct"] = round(
            (candles[-1]["open"] - candles[-2]["close"]) / candles[-2]["close"] * 100, 2
        )

    return result

# ── ASCII candlestick + volume renderer ───────────────────────────────────────
def _render_chart(candles: list[dict], width: int, height: int) -> Text:
    AXIS = 9

    if not candles or width < AXIS + 4 or height < 6:
        out = Text()
        out.append("\n" * max(0, height // 2 - 1))
        out.append("  no data  ".center(width), style="color(240)")
        return out

    # Volume bars take bottom 3 rows
    VOL_H  = 3
    candle_h = max(3, height - VOL_H - 1)   # -1 for separator row

    highs  = [c["high"]   for c in candles]
    lows   = [c["low"]    for c in candles]
    vols   = [c["volume"] for c in candles]

    p_max = max(highs) * 1.001
    p_min = min(lows)  * 0.999
    p_rng = p_max - p_min or 1.0

    def to_row(p: float) -> int:
        return max(0, min(candle_h-1, candle_h-1 - int((p-p_min)/p_rng*(candle_h-1))))

    chart_w = width - AXIS
    n       = min(len(candles), chart_w // 2)
    candles = candles[-n:]
    vols    = vols[-n:]

    BLANK = (" ", "")
    grid  = [[BLANK]*width for _ in range(height)]

    # Price axis
    for row in range(candle_h):
        if row % 4 == 0 or row == candle_h - 1:
            pct   = (candle_h-1-row) / max(candle_h-1, 1)
            price = p_min + pct*p_rng
            lbl   = f"{price:>7.2f} "
            for col, ch in enumerate(lbl[:AXIS]):
                grid[row][col] = (ch, "color(240)")
        grid[row][AXIS-1] = ("│", "color(238)")

    # Candles
    for i, c in enumerate(candles):
        x = AXIS + i*2
        if x >= width:
            break
        h_r, l_r = to_row(c["high"]), to_row(c["low"])
        o_r, cl_r = to_row(c["open"]), to_row(c["close"])
        top, bot = min(o_r, cl_r), max(o_r, cl_r)
        bull = c["close"] >= c["open"]
        col  = "bright_green" if bull else "bright_red"
        for row in range(h_r, l_r+1):
            if 0 <= row < candle_h:
                ch = ("█" if bull else "▓") if top <= row <= bot else "│"
                grid[row][x] = (ch, col)

    # Separator row
    sep_row = candle_h
    grid[sep_row][AXIS-1] = ("┤", "color(238)")
    for col in range(AXIS, width):
        grid[sep_row][col] = ("─", "color(238)")

    # Volume bars (rows candle_h+1 … height-1)
    max_vol = max(vols) if vols else 1
    vol_rows = height - candle_h - 1
    for i, (c, vol) in enumerate(zip(candles, vols)):
        x = AXIS + i*2
        if x >= width:
            break
        bar_h = max(1, int(vol / max_vol * vol_rows)) if vol else 0
        bull  = c["close"] >= c["open"]
        col   = "bright_green" if bull else "bright_red"
        for row in range(height-bar_h, height):
            grid[row][x] = ("▄", col)

    # Assemble
    out = Text()
    for r, row in enumerate(grid):
        for ch, sty in row:
            out.append(ch, style=sty or None)
        if r < height - 1:
            out.append("\n")
    return out

# ── Unified LLM caller ────────────────────────────────────────────────────────
async def _call_llm(
    system: str,
    messages: list[dict],
    *,
    provider: LLMProvider,
    anthropic_key: str | None = None,
    anthropic_model: str      = cfg.ANTHROPIC_MODEL,
    ollama_host: str          = cfg.OLLAMA_HOST,
    ollama_model: str         = cfg.OLLAMA_MODEL,
) -> str:
    if provider == LLMProvider.ANTHROPIC:
        if not anthropic_key:
            return "[No API key — set ANTHROPIC_API_KEY]"
        import anthropic as _ant
        client = _ant.AsyncAnthropic(api_key=anthropic_key)
        msg = await client.messages.create(
            model=anthropic_model, max_tokens=400,
            system=system, messages=messages,
        )
        return msg.content[0].text

    if provider == LLMProvider.OLLAMA:
        import httpx
        all_msgs = [{"role": "system", "content": system}] + messages
        try:
            async with httpx.AsyncClient(base_url=ollama_host, timeout=120) as http:
                resp = await http.post("/api/chat", json={
                    "model": ollama_model, "messages": all_msgs,
                    "stream": False, "options": {"temperature": 0.3},
                })
                resp.raise_for_status()
                return resp.json()["message"]["content"]
        except Exception as e:
            return f"[Ollama error: {e}]"

    return ""

# ── Mini bar renderer ─────────────────────────────────────────────────────────
def _bar(ratio: float, width: int = 10, style: str = "bright_yellow") -> Text:
    ratio  = max(0.0, min(1.0, ratio))
    filled = int(ratio * width)
    t = Text()
    t.append("█" * filled,          style=style)
    t.append("░" * (width - filled), style="color(237)")
    return t

def _val(v: float | None, fmt: str = ".2f", pre: str = "", suf: str = "") -> str:
    if v is None:
        return "—"
    return f"{pre}{v:{fmt}}{suf}"

# ── Widgets ───────────────────────────────────────────────────────────────────

class PriceBar(Widget):
    """Top live ticker tape."""
    quotes: reactive[dict] = reactive({})

    DEFAULT_CSS = "PriceBar { height: 1; dock: top; background: #160b00; padding: 0 1; }"

    def render(self) -> Text:
        t = Text(overflow="fold")
        t.append("  KEVIN  ", style="bold black on #e68000")
        t.append("  ")
        for sym, q in self.quotes.items():
            if not q.get("ok"):
                t.append(f" {sym} ···  ", style="color(240)")
                continue
            p   = q["price"]
            pct = q["pct"]
            col = "bright_green" if pct >= 0 else "bright_red"
            t.append(f" {sym} ", style="bold white")
            t.append(f"${p:.2f} ", style=col)
            t.append(f"{'▲' if pct>=0 else '▼'}{abs(pct):.2f}%", style=col)
            t.append("  ", style="color(236)")
        t.append(f"  {datetime.now().strftime('%H:%M:%S')} ", style="color(240)")
        return t


class WatchlistPane(Widget):
    DEFAULT_CSS = """
    WatchlistPane { height: 1fr; }
    WatchlistPane DataTable { height: 1fr; background: #080808; }
    """

    def __init__(self, tickers: list[str], **kw: Any) -> None:
        super().__init__(**kw)
        self._tickers = tickers

    def compose(self) -> ComposeResult:
        yield Static(" ◈ WATCHLIST", classes="panel-title")
        yield DataTable(cursor_type="row", id="wl-tbl")

    def on_mount(self) -> None:
        t = self.query_one("#wl-tbl", DataTable)
        t.add_column("SYM",   key="sym",   width=6)
        t.add_column("PRICE", key="price", width=9)
        t.add_column("CHG%",  key="chg",   width=7)
        t.add_column("V×",    key="vratio", width=4)
        for s in self._tickers:
            t.add_row(s, "···", "···", "···", key=s)

    def refresh_quotes(self, quotes: dict[str, Any]) -> None:
        t = self.query_one("#wl-tbl", DataTable)
        for sym, q in quotes.items():
            if not q.get("ok"):
                for col in ("price", "chg", "vratio"):
                    t.update_cell(sym, col, Text("---", style="color(240)"))
                continue
            pct = q["pct"]
            col = "bright_green" if pct >= 0 else "bright_red"
            vr  = q.get("vol_ratio")
            vc  = ("bright_red" if (vr or 0) > 2 else
                   "bright_yellow" if (vr or 0) > 1 else "color(240)")
            t.update_cell(sym, "price",  Text(f"{q['price']:>8.2f}", style=col))
            t.update_cell(sym, "chg",    Text(f"{'+'if pct>=0 else ''}{pct:.2f}%", style=col))
            t.update_cell(sym, "vratio", Text(_val(vr, ".1f", suf="×"), style=vc))

    @on(DataTable.RowSelected, "#wl-tbl")
    def _sel(self, ev: DataTable.RowSelected) -> None:
        key = ev.row_key.value if ev.row_key else None
        if key:
            self.app.select_ticker(str(key))  # type: ignore[attr-defined]


class ChartPane(Widget):
    DEFAULT_CSS = """
    ChartPane { height: 1fr; }
    #canvas { height: 1fr; padding: 0 1; }
    """

    def __init__(self, **kw: Any) -> None:
        super().__init__(**kw)
        self._ticker  = "—"
        self._candles: list[dict] = []

    def compose(self) -> ComposeResult:
        yield Static(" ◈ CHART", classes="panel-title", id="chart-title")
        yield Static("", id="canvas")

    def load(self, ticker: str, candles: list[dict]) -> None:
        self._ticker, self._candles = ticker, candles
        self._draw()

    def _draw(self) -> None:
        self.query_one("#chart-title", Static).update(
            f" ◈  {self._ticker}  ·  {CHART_PERIOD.upper()}  CANDLES + VOLUME"
        )
        canvas = self.query_one("#canvas", Static)
        w = max(20, self.size.width - 2)
        h = max(8,  self.size.height - 3)
        canvas.update(_render_chart(self._candles, w, h))

    def on_resize(self) -> None:
        self._draw()


class MetricsPane(Widget):
    DEFAULT_CSS = """
    MetricsPane { height: 7; border-top: solid #2a1400; padding: 0 1; }
    """

    def compose(self) -> ComposeResult:
        yield Static(" ◈ KEY METRICS", classes="panel-title")
        yield Static("", id="mb")

    def load(self, brief: Any | None, quote: dict | None) -> None:
        from rich.table import Table as RT
        m, s, q = (brief.metrics if brief else None), (brief.signal if brief else None), (quote or {})
        g = RT.grid(padding=(0, 3))
        for _ in range(4):
            g.add_column(width=16)

        def kv(k: str, v: str, vc: str = "bright_white") -> tuple:
            return Text(k, style="color(240)"), Text(v, style=vc)

        mktcap = q.get("mktcap")
        mc_str = (f"${mktcap/1e12:.2f}T" if mktcap and mktcap >= 1e12
                  else (f"${mktcap/1e9:.1f}B" if mktcap else "—"))

        eps_str = _val(m.eps_primary if m else None, ".2f", "$")
        rev_str = _val(m.revenue_mm/1000 if (m and m.revenue_mm) else None, ".2f", "$", "B")
        gm_str  = _val(m.gross_margin*100 if (m and m.gross_margin) else None, ".1f", suf="%")
        om_str  = _val(m.operating_margin*100 if (m and m.operating_margin) else None, ".1f", suf="%")
        bull    = s.bull_score if s else None
        bear    = s.bear_score if s else None
        conf    = s.confidence if s else None
        bc      = "bright_green" if (bull or 0) > (bear or 0) else "bright_red"

        g.add_row(*kv("EPS (GAAP)", eps_str), *kv("Revenue",    rev_str))
        g.add_row(*kv("Gross Mar",  gm_str),  *kv("Op Margin",  om_str))
        g.add_row(*kv("Bull/Bear",  f"{_val(bull,'.0f')} / {_val(bear,'.0f')}", bc),
                  *kv("Confidence", _val(conf, ".0f", suf="/100"), "bright_cyan"))
        g.add_row(*kv("Mkt Cap",    mc_str),
                  *kv("Day Range",  _val(q.get("day_range_pct"), ".2f", suf="%")))

        if m and m.guidance and m.guidance.midpoint is not None:
            gg = m.guidance
            raised = ("▲ RAISED" if gg.is_raised else "▼ LOWERED" if gg.is_raised is False else "GUIDANCE")
            gc     = "bright_green" if gg.is_raised else ("bright_red" if gg.is_raised is False else "bright_yellow")
            lo_s   = f"${gg.lo:.2f}" if gg.lo is not None else "?"
            hi_s   = f"${gg.hi:.2f}" if gg.hi is not None else "?"
            guide  = f"{lo_s}–{hi_s}  mid ${gg.midpoint:.2f}"
            if gg.period:
                guide += f"  {gg.period}"
            g.add_row(Text(raised, style=f"bold {gc}"), Text(guide, style="white"), Text(""), Text(""))

        self.query_one("#mb", Static).update(g)


class FilingsPane(Widget):
    DEFAULT_CSS = """
    FilingsPane { height: 1fr; border-top: solid #2a1400; }
    FilingsPane RichLog { height: 1fr; padding: 0 1; background: #080808; }
    """

    def compose(self) -> ComposeResult:
        yield Static(" ◈ EDGAR FILINGS", classes="panel-title")
        yield RichLog(id="fl", markup=False, highlight=False, wrap=False)

    def load(self, briefs: list[Any]) -> None:
        log = self.query_one("#fl", RichLog)
        log.clear()
        if not briefs:
            log.write(Text("  —  no recent filings", style="color(240)"))
            return
        for b in briefs[:MAX_FILINGS]:
            s, m = b.signal, b.metrics
            vcol = "green" if s.bull_score > s.bear_score else "red"
            row  = Text()
            row.append(f"  {b.index.filed_at.strftime('%Y-%m-%d')} ", style="color(240)")
            row.append("8-K ", style="white")
            row.append(f" {'BEAT' if vcol=='green' else 'MISS'} ",
                       style=f"bold black on {vcol}")
            log.write(row)
            detail = Text()
            detail.append(f"   EPS {_val(m.eps_primary,'.2f','$'):>8}", style="white")
            detail.append(f"  Rev {_val(m.revenue_mm/1000 if m.revenue_mm else None,'.1f','$','B'):>8}", style="white")
            if m.gross_margin:
                detail.append(f"  GM {m.gross_margin*100:.1f}%", style="color(240)")
            log.write(detail)
            log.write(Text(""))


class SignalsPane(Widget):
    """HFT-relevant technical indicators panel."""
    DEFAULT_CSS = """
    SignalsPane {
        height: 14;
        border-top: solid #2a1400;
        padding: 0 1;
    }
    """

    def compose(self) -> ComposeResult:
        yield Static(" ◈ TECHNICALS", classes="panel-title")
        yield Static("", id="sig-body")

    def load(self, technicals: dict, quote: dict) -> None:
        from rich.table import Table as RT
        g = RT.grid(padding=(0, 1))
        g.add_column(width=9)   # label
        g.add_column(width=11)  # bar / value
        g.add_column(width=10)  # value / flag
        g.add_column(width=9)   # label2
        g.add_column(width=11)  # val2

        def row(la: str, bar_or_val: Any, flag: str, flag_col: str,
                lb: str = "", vb: str = "", vb_col: str = "white") -> None:
            g.add_row(
                Text(la, style="color(240)"),
                bar_or_val,
                Text(flag, style=flag_col),
                Text(lb, style="color(240)"),
                Text(vb, style=vb_col),
            )

        # RSI
        rsi = technicals.get("rsi")
        if rsi is not None:
            rsi_col = ("bright_red" if rsi > 70 else "bright_green" if rsi < 30 else "bright_yellow")
            rsi_lbl = ("OVERBOUGHT" if rsi > 70 else "OVERSOLD" if rsi < 30 else "NEUTRAL")
            row("RSI(14)", _bar(rsi/100, 10, rsi_col), f"{rsi:.1f}  {rsi_lbl}", rsi_col,
                "Velocity", _val(technicals.get("velocity"), ".2f", suf="×"),
                "bright_red" if (technicals.get("velocity") or 0) > 1.5 else "white")

        # ATR
        atr = technicals.get("atr")
        atr_pct = technicals.get("atr_pct")
        row("ATR(14)", Text(_val(atr, ".2f", "$"), style="white"),
            _val(atr_pct, ".2f", suf="% pr"), "color(240)",
            "Gap", _val(technicals.get("gap_pct"), "+.2f", suf="%"),
            "bright_green" if (technicals.get("gap_pct") or 0) > 0 else "bright_red")

        # VWAP
        vwap    = technicals.get("vwap")
        vwap_dev= technicals.get("vwap_dev")
        vd_col  = "bright_green" if (vwap_dev or 0) >= 0 else "bright_red"
        row("VWAP", Text(_val(vwap, ".2f", "$"), style="white"),
            _val(vwap_dev, "+.2f", suf="%"), vd_col,
            "Day Sprd", _val(quote.get("day_range_pct"), ".2f", suf="%"), "color(240)")

        # Momentum
        m5  = technicals.get("mom5")
        m10 = technicals.get("mom10")
        m5c  = "bright_green" if (m5 or 0)  >= 0 else "bright_red"
        m10c = "bright_green" if (m10 or 0) >= 0 else "bright_red"
        row("Mom 5D",  Text(_val(m5, "+.2f", suf="%"),  style=m5c),  "", "",
            "Mom 10D", _val(m10, "+.2f", suf="%"), m10c)

        # 52-week bar
        pos = technicals.get("wk52_pos")
        w52h = technicals.get("wk52_high") or quote.get("wk52_high")
        w52l = technicals.get("wk52_low")  or quote.get("wk52_low")
        if pos is not None:
            pos_col = ("bright_red" if pos > 85 else "bright_green" if pos < 15 else "bright_yellow")
            pos_bar = _bar(pos/100, 10, pos_col)
            row("52W Pos", pos_bar, f"{pos:.0f}%", pos_col,
                "Range",
                f"${_val(w52l,'.0f')}–${_val(w52h,'.0f')}", "color(240)")

        # Volume
        vr = quote.get("vol_ratio")
        if vr is not None:
            vr_col = ("bright_red" if vr > 2.5 else "bright_yellow" if vr > 1.2 else "color(240)")
            vr_lbl = ("SURGE" if vr > 2.5 else "HOT" if vr > 1.5 else "ACTIVE" if vr > 1.0 else "QUIET")
            row("Vol/Avg", _bar(min(vr/3, 1.0), 10, vr_col), f"{vr:.2f}× {vr_lbl}", vr_col,
                "Mkt Cap", (
                    f"${quote.get('mktcap',0)/1e12:.2f}T"
                    if (quote.get("mktcap") or 0) >= 1e12
                    else f"${(quote.get('mktcap') or 0)/1e9:.0f}B"
                ), "color(240)")

        self.query_one("#sig-body", Static).update(g)

    def loading(self, ticker: str) -> None:
        self.query_one("#sig-body", Static).update(
            Text(f"  Computing technicals for {ticker}…", style="color(240)")
        )


class AnalysisPane(Widget):
    DEFAULT_CSS = """
    AnalysisPane { height: 1fr; }
    AnalysisPane RichLog { height: 1fr; padding: 0 1; background: #080808; }
    """

    def compose(self) -> ComposeResult:
        yield Static(" ◈ ANALYSIS", classes="panel-title")
        yield RichLog(id="al", markup=False, highlight=False, wrap=True)

    def _log(self) -> RichLog:
        return self.query_one("#al", RichLog)

    def clear(self) -> None:
        self._log().clear()

    def write_structured(self, brief: Any) -> None:
        log = self._log()
        log.clear()
        m, s = brief.metrics, brief.signal
        bull    = (s.bull_score or 0) > (s.bear_score or 0)
        vcol    = "green" if bull else "red"

        # ── Header ────────────────────────────────────────
        hdr = Text()
        hdr.append(f"  {brief.index.company[:24]:24}", style="bold white")
        hdr.append(f" {'▲ BULLISH' if bull else '▼ BEARISH'} ", style=f"bold black on {vcol}")
        log.write(hdr)
        log.write(Text(
            f"  Filed {brief.index.filed_at.strftime('%b %d %Y')}  "
            f"·  {brief.index.period or ''}  ·  conf {s.confidence:.0f}/100",
            style="color(240)"
        ))
        log.write(Text("  " + "─" * 36, style="color(238)"))

        # ── Earnings ──────────────────────────────────────
        if m.eps_primary or m.revenue_mm:
            log.write(Text("  EARNINGS", style="bold #e68000"))
            if m.eps_primary:
                log.write(Text(f"   EPS        {_val(m.eps_primary, '.2f', '$'):>10}", style="white"))
            if m.revenue_mm:
                log.write(Text(f"   Revenue    {_val(m.revenue_mm/1000, '.2f', '$', 'B'):>10}", style="white"))
            if m.gross_margin:
                log.write(Text(f"   Gross Mar  {_val(m.gross_margin*100, '.1f', suf='%'):>10}", style="white"))
            if m.operating_margin:
                log.write(Text(f"   Op Margin  {_val(m.operating_margin*100, '.1f', suf='%'):>10}", style="white"))

        # ── Guidance ──────────────────────────────────────
        if m.guidance and m.guidance.midpoint is not None:
            g = m.guidance
            raised = ("▲ RAISED" if g.is_raised else
                      "▼ LOWERED" if g.is_raised is False else "GUIDANCE")
            gc     = ("bright_green" if g.is_raised else
                      "bright_red"   if g.is_raised is False else "bright_yellow")
            log.write(Text(""))
            log.write(Text(f"  {raised}", style=f"bold {gc}"))
            lo_s = f"${g.lo:.2f}" if g.lo is not None else "?"
            hi_s = f"${g.hi:.2f}" if g.hi is not None else "?"
            log.write(Text(f"   {lo_s} – {hi_s}  (mid ${g.midpoint:.2f})", style="white"))
            if g.period:
                log.write(Text(f"   {g.period}", style="color(240)"))

        # ── Signal summary ────────────────────────────────
        log.write(Text(""))
        log.write(Text(
            f"  Bull {_val(s.bull_score,'.0f')}  ·  Bear {_val(s.bear_score,'.0f')}  ·  Conf {_val(s.confidence,'.0f')}",
            style="bright_green" if bull else "bright_red"
        ))
        if s.risk_flags:
            log.write(Text(f"  ⚠  {', '.join(s.risk_flags[:3])}", style="bright_red"))
        log.write(Text(""))

    def end_llm(self, full_text: str, brief: Any, provider_name: str) -> None:
        """Finalise: re-write structured section then append LLM text."""
        self.write_structured(brief)
        log = self._log()
        log.write(Text("  " + "─" * 36, style="color(238)"))
        log.write(Text(f"  LLM ({provider_name})", style="color(240)"))
        for line in full_text.split("\n"):
            log.write(Text(f"  {line}", style="white"))

    def write_line(self, text: Text) -> None:
        """Append one line — public alternative to direct _log() access."""
        self._log().write(text)

    def show_error(self, msg: str) -> None:
        """Append a red error line."""
        self._log().write(Text(f"  {msg}", style="bright_red"))

    def loading(self, ticker: str) -> None:
        log = self._log()
        log.clear()
        log.write(Text(f"  Loading {ticker}…", style="color(240)"))


class ChatPane(Widget):
    DEFAULT_CSS = """
    ChatPane { height: 16; border-top: solid #2a1400; }
    ChatPane RichLog { height: 1fr; padding: 0 1; background: #080808; }
    ChatPane Input {
        height: 3; dock: bottom; background: #0d0d0d;
        border: solid #2a1400;
    }
    ChatPane Input:focus { border: solid #e68000; }
    """

    def compose(self) -> ComposeResult:
        yield Static(" ◈ CHAT  [dim](/ to focus)[/dim]", classes="panel-title", markup=True)
        yield RichLog(id="cl", markup=False, highlight=False, wrap=True)
        yield Input(id="ci", placeholder="Ask about markets, charts, filings…")

    def on_mount(self) -> None:
        # Persisted message lines for replay when clearing the thinking indicator
        self._lines: list[Text] = []

    def _log(self) -> RichLog:
        return self.query_one("#cl", RichLog)

    def _replay(self) -> None:
        """Clear and re-render all saved lines (used to remove thinking indicator)."""
        log = self._log()
        log.clear()
        for line in self._lines:
            log.write(line)

    def add_msg(self, role: str, text: str) -> None:
        if role == "user":
            line = Text(f"  ▶  {text}", style="bright_yellow")
            self._lines.append(line)
            self._log().write(line)
            self._lines.append(Text(""))
            self._log().write(Text(""))
        else:
            # Replace the "thinking" placeholder with the real reply
            self._replay()
            for raw_line in text.split("\n"):
                line = Text(f"  {raw_line}", style="white")
                self._lines.append(line)
                self._log().write(line)
            self._lines.append(Text(""))
            self._log().write(Text(""))

    def thinking(self) -> None:
        """Show a temporary thinking indicator (not saved to _lines)."""
        self._log().write(Text("  Kevin  ···", style="color(240)"))

    @on(Input.Submitted, "#ci")
    async def _submit(self, ev: Input.Submitted) -> None:
        msg = ev.value.strip()
        if not msg:
            return
        ev.input.clear()
        ev.input.disabled = True
        await self.app.handle_chat(msg)   # type: ignore[attr-defined]
        ev.input.disabled = False
        ev.input.focus()


# ── Application ───────────────────────────────────────────────────────────────
class KevinTerminal(App[None]):
    CSS = """
    Screen { background: #0a0a0a; color: #d8d8d8; }

    .panel-title {
        background: #1a0900;
        color: #e68000;
        height: 1;
        padding: 0 1;
        text-style: bold;
    }

    #main { height: 1fr; }

    #left  { width: 29; border-right: solid #281200; }
    #center{ width: 1fr; border-right: solid #281200; }
    #right { width: 46; }

    DataTable          { background: #080808; }
    DataTable > .datatable--header  { background: #1a0900; color: #e68000; text-style: bold; }
    DataTable > .datatable--cursor  { background: #2a1400; }
    DataTable > .datatable--fixed   { background: #1a0900; }

    RichLog { background: #080808; scrollbar-color: #e68000; }

    Footer               { background: #1a0900; color: #e68000; }
    Footer .footer--key  { color: #e68000; text-style: bold; }
    Footer .footer--description { color: #a05000; }
    """

    BINDINGS = [
        Binding("q",      "quit",       "Quit"),
        Binding("r",      "refresh",    "Refresh"),
        Binding("/",      "chat",       "Chat"),
        Binding("escape", "focus_list", "List", show=False),
    ]

    def __init__(
        self,
        tickers: list[str],
        llm_override: LLMProvider | None = None,
        ollama_model: str | None = None,
        ollama_host:  str | None = None,
    ) -> None:
        super().__init__()
        self.tickers      = tickers
        self._selected    = tickers[0]
        self._quotes:  dict[str, Any]  = {}
        self._briefs:  dict[str, list] = {}
        self._candles: dict[str, list] = {}
        self._techs:   dict[str, dict] = {}
        self._history: list[dict]      = []

        # LLM config (CLI can override env)
        self._llm_prov    = llm_override or cfg.LLM_PROVIDER
        self._ant_key     = cfg.ANTHROPIC_API_KEY
        self._ant_model   = cfg.ANTHROPIC_MODEL
        self._oll_host    = ollama_host  or cfg.OLLAMA_HOST
        self._oll_model   = ollama_model or cfg.OLLAMA_MODEL

    @property
    def _llm_name(self) -> str:
        return {
            LLMProvider.ANTHROPIC: "Claude",
            LLMProvider.OLLAMA:    f"Ollama/{self._oll_model}",
            LLMProvider.CLAUDE:    "Claude CLI",
        }.get(self._llm_prov, "none")

    @property
    def _llm_ready(self) -> bool:
        if self._llm_prov == LLMProvider.ANTHROPIC:
            return bool(self._ant_key)
        return self._llm_prov in (LLMProvider.OLLAMA, LLMProvider.CLAUDE)

    # ── Layout ────────────────────────────────────────────────────────────────
    def compose(self) -> ComposeResult:
        yield PriceBar(id="pb")
        with Horizontal(id="main"):
            with Vertical(id="left"):
                yield WatchlistPane(self.tickers, id="wl")
                yield FilingsPane(id="fp")
            with Vertical(id="center"):
                yield ChartPane(id="chart")
                yield MetricsPane(id="metrics")
            with Vertical(id="right"):
                yield SignalsPane(id="sig")
                yield AnalysisPane(id="ap")
                yield ChatPane(id="cp")
        yield Footer()

    def on_mount(self) -> None:
        self._fetch_prices()
        self._fetch_edgar(self._selected)
        self.set_interval(1,             self._tick)
        self.set_interval(PRICE_REFRESH, self._fetch_prices)

    def _tick(self) -> None:
        self.query_one("#pb", PriceBar).refresh()

    # ── Workers ───────────────────────────────────────────────────────────────
    @work(thread=True, exclusive=True, group="prices")
    def _fetch_prices(self) -> None:
        import concurrent.futures
        results: dict[str, Any] = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(self.tickers), 10)) as ex:
            futs = {ex.submit(_fetch_quote, t): t for t in self.tickers}
            for f in concurrent.futures.as_completed(futs):
                t = futs[f]
                try:    results[t] = f.result()
                except Exception: results[t] = {"ticker": t, "ok": False}
        self.call_from_thread(self._apply_quotes, results)
        # Also refresh chart for whatever is currently selected
        selected = self._selected
        ohlcv = _fetch_ohlcv(selected)
        self.call_from_thread(self._apply_chart, selected, ohlcv)

    @work(thread=True, exclusive=True, group="chart")
    def _fetch_chart(self, ticker: str) -> None:
        """Refresh OHLCV + technicals for a single ticker (used on selection change)."""
        ohlcv = _fetch_ohlcv(ticker)
        self.call_from_thread(self._apply_chart, ticker, ohlcv)

    @work(exclusive=True, group="edgar")
    async def _fetch_edgar(self, ticker: str) -> None:
        from kevin.pipeline import analyse_ticker
        self.query_one("#ap", AnalysisPane).loading(ticker)
        self.query_one("#sig", SignalsPane).loading(ticker)
        try:
            briefs = await analyse_ticker(
                ticker, since_days=90, earnings_only=True, max_filings=MAX_FILINGS
            )
            self._briefs[ticker] = briefs
            self.query_one("#fp",      FilingsPane).load(briefs)
            self.query_one("#metrics", MetricsPane).load(
                briefs[0] if briefs else None,
                self._quotes.get(ticker),
            )
            await self._run_analysis(ticker, briefs)
        except Exception as e:
            ap = self.query_one("#ap", AnalysisPane)
            ap.clear()
            ap.show_error(f"Error: {e}")

    # ── UI updaters ───────────────────────────────────────────────────────────
    def _apply_quotes(self, quotes: dict[str, Any]) -> None:
        self._quotes = quotes
        self.query_one("#pb", PriceBar).quotes = quotes
        self.query_one("#wl", WatchlistPane).refresh_quotes(quotes)
        if self._selected in self._briefs:
            self.query_one("#metrics", MetricsPane).load(
                (self._briefs[self._selected] or [None])[0],
                quotes.get(self._selected),
            )

    def _apply_chart(self, ticker: str, candles: list[dict]) -> None:
        self._candles[ticker] = candles
        techs = _calc_technicals(candles, self._quotes.get(ticker, {}))
        self._techs[ticker] = techs
        self.query_one("#chart", ChartPane).load(ticker, candles)
        self.query_one("#sig",   SignalsPane).load(techs, self._quotes.get(ticker, {}))

    # ── Ticker selection ──────────────────────────────────────────────────────
    def select_ticker(self, ticker: str) -> None:
        self._selected = ticker
        self.query_one("#ap",      AnalysisPane).loading(ticker)
        self.query_one("#fp",      FilingsPane).load([])
        self.query_one("#metrics", MetricsPane).load(None, None)
        # Refresh chart only — quotes are refreshed on the price timer
        self._fetch_chart(ticker)
        if ticker not in self._briefs:
            self._fetch_edgar(ticker)
        else:
            self.query_one("#fp",      FilingsPane).load(self._briefs[ticker])
            self.query_one("#metrics", MetricsPane).load(
                (self._briefs[ticker] or [None])[0],
                self._quotes.get(ticker),
            )
            if ticker in self._techs:
                self.query_one("#sig", SignalsPane).load(
                    self._techs[ticker], self._quotes.get(ticker, {})
                )
            self.run_worker(self._run_analysis(ticker, self._briefs[ticker]), exclusive=True)

    # ── LLM analysis ─────────────────────────────────────────────────────────
    async def _run_analysis(self, ticker: str, briefs: list[Any]) -> None:
        ap = self.query_one("#ap", AnalysisPane)
        if not briefs:
            ap.clear()
            ap.write_line(Text(f"  No recent earnings for {ticker}.", style="color(240)"))
            return

        brief = briefs[0]
        ap.write_structured(brief)

        if not self._llm_ready:
            ap.write_line(Text(""))
            ap.write_line(Text(
                "  ── Set KEVIN_LLM=anthropic/ollama for AI commentary ──",
                style="color(238)"
            ))
            return

        ctx    = self._build_context(ticker)
        system = (
            "You are Kevin, a sharp sell-side analyst. Given SEC 8-K data, write a "
            "150-word filing brief: EPS beat/miss vs consensus, revenue trajectory, "
            "margin direction, guidance surprise, and one key risk. End with a "
            "one-sentence trade verdict."
        )
        msgs = [{"role": "user", "content": f"Analyse {ticker}:\n{ctx}"}]

        try:
            full = await _call_llm(
                system, msgs,
                provider        = self._llm_prov,
                anthropic_key   = self._ant_key,
                anthropic_model = self._ant_model,
                ollama_host     = self._oll_host,
                ollama_model    = self._oll_model,
            )
            ap.end_llm(full, brief, self._llm_name)
        except Exception as e:
            ap.show_error(f"LLM error: {e}")

    # ── Chat ──────────────────────────────────────────────────────────────────
    async def handle_chat(self, msg: str) -> None:
        cp = self.query_one("#cp", ChatPane)
        cp.add_msg("user", msg)
        self._history.append({"role": "user", "content": msg})

        if not self._llm_ready:
            cp.add_msg("assistant", f"Configure LLM — run: python tui.py --llm anthropic  (or ollama)")
            return

        cp.thinking()
        system = (
            f"You are Kevin, a concise financial analyst. "
            f"Answer in under 180 words, use specific numbers from context.\n\n"
            f"Context:\n{self._build_context(self._selected)}"
        )
        try:
            reply = await _call_llm(
                system,
                self._history[-14:],
                provider        = self._llm_prov,
                anthropic_key   = self._ant_key,
                anthropic_model = self._ant_model,
                ollama_host     = self._oll_host,
                ollama_model    = self._oll_model,
            )
        except Exception as e:
            reply = f"Error: {e}"

        self._history.append({"role": "assistant", "content": reply})
        cp.add_msg("assistant", reply)

    # ── Context builder ───────────────────────────────────────────────────────
    def _build_context(self, ticker: str) -> str:
        parts: list[str] = []
        q = self._quotes.get(ticker, {})
        if q.get("ok"):
            parts.append(f"{ticker}: ${q['price']:.2f}  ({q['pct']:+.2f}% today)")
            if q.get("vol_ratio"):
                parts.append(f"Volume ratio: {q['vol_ratio']:.2f}×")

        tc = self._techs.get(ticker, {})
        if tc.get("rsi"):
            parts.append(f"RSI(14): {tc['rsi']:.1f}")
        if tc.get("atr"):
            parts.append(f"ATR(14): ${tc['atr']:.2f}  ({tc.get('atr_pct',0):.2f}% of price)")
        if tc.get("mom5"):
            parts.append(f"5-day momentum: {tc['mom5']:+.2f}%")
        if tc.get("vwap_dev"):
            parts.append(f"VWAP deviation: {tc['vwap_dev']:+.2f}%")

        for b in self._briefs.get(ticker, [])[:1]:
            m, s = b.metrics, b.signal
            parts.append(f"Latest 8-K: {b.index.filed_at.strftime('%Y-%m-%d')}  {b.index.company}")
            if m.eps_primary:    parts.append(f"EPS: ${m.eps_primary:.2f}")
            if m.revenue_mm:     parts.append(f"Revenue: ${m.revenue_mm/1000:.2f}B")
            if m.gross_margin:   parts.append(f"Gross margin: {m.gross_margin*100:.1f}%")
            if m.guidance:
                g = m.guidance
                parts.append(f"Guidance: ${g.lo:.2f}–${g.hi:.2f}")
            if s.risk_flags:     parts.append(f"Risks: {', '.join(s.risk_flags)}")
            if s.verdict:        parts.append(f"Signal: {s.verdict}")
        return "\n".join(parts) or f"No data loaded for {ticker}."

    # ── Actions ───────────────────────────────────────────────────────────────
    def action_refresh(self) -> None:
        self._briefs.pop(self._selected, None)
        self._candles.pop(self._selected, None)
        self._techs.pop(self._selected, None)
        self._fetch_prices()
        self._fetch_edgar(self._selected)

    def action_chat(self) -> None:
        self.query_one("#ci", Input).focus()

    def action_focus_list(self) -> None:
        self.query_one("#wl-tbl", DataTable).focus()


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(
        description="Kevin Terminal — Bloomberg-style TUI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python tui.py\n"
            "  python tui.py NVDA TSLA AMD\n"
            "  python tui.py --llm anthropic\n"
            "  python tui.py --llm ollama --model llama3.2\n"
            "  python tui.py --llm ollama --host http://192.168.1.10:11434\n"
        ),
    )
    p.add_argument("tickers",   nargs="*",          default=DEFAULT_TICKERS, metavar="TICKER")
    p.add_argument("--llm",     choices=["anthropic", "ollama", "none"], default=None,
                   help="LLM provider (overrides KEVIN_LLM env)")
    p.add_argument("--model",   default=None, metavar="NAME",
                   help="Ollama model name (default: llama3.1)")
    p.add_argument("--host",    default=None, metavar="URL",
                   help="Ollama host (default: http://localhost:11434)")
    args = p.parse_args()

    tickers = [t.upper() for t in args.tickers] or DEFAULT_TICKERS

    # Map CLI --llm to LLMProvider
    prov_map = {"anthropic": LLMProvider.ANTHROPIC, "ollama": LLMProvider.OLLAMA, "none": LLMProvider.NONE}
    llm_override = prov_map.get(args.llm) if args.llm else None

    KevinTerminal(
        tickers,
        llm_override = llm_override,
        ollama_model = args.model,
        ollama_host  = args.host,
    ).run()
