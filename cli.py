#!/usr/bin/env python3
"""
Kevin CLI - SEC 8-K earnings intelligence engine.

Usage:
  kevin scan NVDA                          # last 30 days of NVDA 8-Ks
  kevin scan NVDA MSFT AAPL               # multiple tickers
  kevin scan NVDA --since 7 --out json    # 1-week, JSON output
  kevin scan NVDA --out csv               # CSV signal file
  kevin scan NVDA --llm anthropic         # enable LLM fallback
  kevin filing 0001045810-25-000021       # analyse one specific filing
  kevin filing https://www.sec.gov/...    # by direct exhibit URL
  kevin batch tickers.txt                 # bulk process from file
  kevin watch NVDA --interval 300         # poll every 5 min for new filings
"""
from __future__ import annotations

import asyncio
import csv
import json
import os
import sys
import time
from io import StringIO
from pathlib import Path

import click
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table

from kevin.config import LLMProvider, cfg
from kevin.models import KevinBrief

console = Console(highlight=False)


# Output formatters

def _json_out(briefs: list[KevinBrief]) -> str:
    return json.dumps(
        [b.model_dump(mode="json") for b in briefs],
        indent=2, default=str
    )


def _signals_json_out(briefs: list[KevinBrief]) -> str:
    return json.dumps(
        [b.signal.model_dump(mode="json") for b in briefs],
        indent=2, default=str
    )


def _csv_out(briefs: list[KevinBrief]) -> str:
    if not briefs:
        return ""
    rows = [b.to_signal_row() for b in briefs]
    buf  = StringIO()
    w    = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue()


def _rich_table(briefs: list[KevinBrief]) -> Table:
    t = Table(
        title="Kevin - Earnings Signal Summary",
        box=box.SIMPLE_HEAVY,
        show_header=True,
        header_style="bold cyan",
        border_style="dim",
        expand=True,
    )
    t.add_column("Ticker",   style="bold yellow", no_wrap=True, width=8)
    t.add_column("Filed",    style="dim",         no_wrap=True, width=12)
    t.add_column("Session",  style="dim",         no_wrap=True, width=12)
    t.add_column("EPS",      justify="right",     no_wrap=True, width=8)
    t.add_column("Surprise", justify="right",     no_wrap=True, width=10)
    t.add_column("Rev $M",   justify="right",     no_wrap=True, width=10)
    t.add_column("Bull",     justify="right",     no_wrap=True, width=6)
    t.add_column("Bear",     justify="right",     no_wrap=True, width=6)
    t.add_column("Conf",     justify="right",     no_wrap=True, width=6)
    t.add_column("Flags",    style="dim red",     no_wrap=False)

    for b in briefs:
        s    = b.signal
        eps  = f"${s.eps:.2f}" if s.eps is not None else "-"
        surp = ""
        if s.eps_surprise_pct is not None:
            pct   = s.eps_surprise_pct * 100
            sign  = "+" if pct >= 0 else ""
            color = "green" if pct >= 0 else "red"
            surp  = f"[{color}]{sign}{pct:.1f}%[/{color}]"
        rev = (
            f"${s.revenue_mm/1000:.1f}B" if s.revenue_mm and s.revenue_mm >= 1000
            else (f"${s.revenue_mm:.0f}M" if s.revenue_mm else "-")
        )
        bull_c = "green" if s.bull_score > 60 else "yellow" if s.bull_score > 40 else "red"
        bear_c = "red"   if s.bear_score > 60 else "yellow" if s.bear_score > 40 else "green"
        flags  = ", ".join(s.risk_flags[:3]) if s.risk_flags else ""

        t.add_row(
            s.ticker,
            b.index.filed_at.strftime("%Y-%m-%d"),
            s.market_session.value.replace("_", " "),
            eps,
            surp or "-",
            rev,
            f"[{bull_c}]{s.bull_score:.0f}[/{bull_c}]",
            f"[{bear_c}]{s.bear_score:.0f}[/{bear_c}]",
            f"{s.confidence:.0f}",
            flags or "-",
        )
    return t


def _rich_detail(brief: KevinBrief) -> None:
    """Pretty-print full detail for a single brief."""
    s  = brief.signal
    m  = brief.metrics
    to = brief.tone

    bull_c = "green" if s.bull_score > 60 else "yellow" if s.bull_score > 40 else "red"
    bear_c = "red"   if s.bear_score > 60 else "yellow" if s.bear_score > 40 else "green"

    eps_line = f"${s.eps:.2f}" if s.eps is not None else "-"
    surp_line = ""
    if s.eps_surprise_pct is not None:
        color = "green" if s.eps_surprise_pct >= 0 else "red"
        sign  = "+" if s.eps_surprise_pct >= 0 else ""
        surp_line = (
            f"  surprise [{color}]{sign}{s.eps_surprise_pct*100:.1f}%[/{color}]"
        )

    rev_line = "-"
    if s.revenue_mm is not None:
        rev_line = (
            f"${s.revenue_mm/1000:.2f}B" if s.revenue_mm >= 1000
            else f"${s.revenue_mm:.0f}M"
        )

    header = (
        f"[bold yellow]{s.ticker}[/bold yellow]  [dim]{brief.index.company}[/dim]\n"
        f"[dim]{brief.index.period or ''}  filed {s.filed_at.strftime('%Y-%m-%d %H:%M')} UTC  "
        f"{s.market_session.value.replace('_', ' ')}[/dim]\n\n"
        f"EPS: [bold]{eps_line}[/bold]{surp_line}\n"
        f"Revenue: [bold]{rev_line}[/bold]\n"
        f"Bull: [{bull_c}]{s.bull_score:.0f}[/{bull_c}]  "
        f"Bear: [{bear_c}]{s.bear_score:.0f}[/{bear_c}]  Conf: {s.confidence:.0f}"
    )
    console.print(Panel(header, title="Kevin", border_style="yellow"))

    if s.verdict:
        console.print(Panel(s.verdict, title="Verdict", border_style="cyan"))

    mt = Table(box=box.MINIMAL, show_header=False, expand=False)
    mt.add_column("", style="dim", width=24)
    mt.add_column("", style="bold")
    if m.eps_basic:
        mt.add_row("EPS Basic (GAAP)",
                   f"${m.eps_basic.value:.2f}  conf={m.eps_basic.confidence:.0f}  src={m.eps_basic.source}")
    if m.eps_diluted:
        mt.add_row("EPS Diluted (GAAP)", f"${m.eps_diluted.value:.2f}")
    if m.eps_adjusted:
        mt.add_row("EPS Adjusted",       f"${m.eps_adjusted.value:.2f}")
    if m.guidance:
        g = m.guidance
        mt.add_row("Guidance midpoint",
                   f"${g.midpoint:.2f}  [{g.lo:.2f}-{g.hi:.2f}]  {g.period or ''}")
    if m.gross_margin:
        mt.add_row("Gross margin",     f"{m.gross_margin*100:.1f}%")
    if m.operating_margin:
        mt.add_row("Operating margin", f"{m.operating_margin*100:.1f}%")
    console.print(mt)

    if s.risk_flags:
        console.print(f"[bold red]Risk flags:[/bold red] {', '.join(s.risk_flags)}")
    if to.key_phrases:
        console.print("\n[bold dim]Key phrases:[/bold dim]")
        for phrase in to.key_phrases[:3]:
            console.print(f"  [dim]>[/dim] {phrase}")
    if s.exhibit_url:
        console.print(f"\n[dim]Source: {s.exhibit_url}[/dim]")
    if brief.errors:
        console.print(f"[dim yellow]Warnings: {', '.join(brief.errors)}[/dim yellow]")


# LLM option helper

def _apply_llm_option(llm: str) -> None:
    """Override the config LLM provider from CLI flag."""
    try:
        os.environ["KEVIN_LLM"] = llm.lower()
        cfg.LLM_PROVIDER = LLMProvider(llm.lower())
    except ValueError:
        console.print(f"[red]Unknown LLM provider: {llm}. Choices: anthropic, ollama, claude, none[/red]")
        sys.exit(1)


def _emit(text: str, path: str | None) -> None:
    if path:
        Path(path).write_text(text, encoding="utf-8")
        console.print(f"[green]Saved to {path}[/green]")
    else:
        console.print(text)


# CLI group

@click.group()
@click.version_option("0.1.0", prog_name="kevin")
def cli() -> None:
    """Kevin - SEC 8-K earnings intelligence engine."""
    _DEFAULT_UA = "kevin-fin-engine research@fin-engine.local"
    if cfg.EDGAR_UA == _DEFAULT_UA:
        console.print(
            "[yellow]⚠  KEVIN_EDGAR_UA is using a placeholder value. "
            "SEC ToS requires a real User-Agent. "
            "Set: KEVIN_EDGAR_UA='YourOrg/1.0 you@example.com'[/yellow]"
        )


# scan

@cli.command()
@click.argument("tickers", nargs=-1, required=True)
@click.option("--since",  default=30,      show_default=True, help="Lookback window in days")
@click.option("--out",    default="table", show_default=True,
              type=click.Choice(["table", "json", "signals", "csv"]), help="Output format")
@click.option("--llm",    default="none",  show_default=True,
              help="LLM provider for low-confidence fallback (anthropic|ollama|claude|none)")
@click.option("--all",    "all_filings", is_flag=True, default=False,
              help="Include non-earnings 8-Ks (M&A, departures, etc.)")
@click.option("--save",   default=None, help="Save output to this file path")
@click.option("--detail", is_flag=True, default=False,
              help="Show full detail for each brief (table mode only)")
def scan(tickers, since, out, llm, all_filings, save, detail) -> None:
    """Scan recent 8-K filings for one or more TICKERS."""
    _apply_llm_option(llm)

    from kevin.pipeline import analyse_ticker

    async def _run_all() -> list[KevinBrief]:
        tasks = [
            analyse_ticker(t, since_days=since, earnings_only=not all_filings)
            for t in tickers
        ]
        results = await asyncio.gather(*tasks)
        return [b for group in results for b in group]

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task(f"Scanning {', '.join(t.upper() for t in tickers)}...", total=None)
        all_briefs = asyncio.run(_run_all())
        progress.update(task, description=f"Done — {len(all_briefs)} filing(s)")

    if not all_briefs:
        console.print("[yellow]No filings found.[/yellow]")
        return

    all_briefs.sort(key=lambda b: b.index.filed_at, reverse=True)

    if out == "table":
        console.print(_rich_table(all_briefs))
        if detail:
            for b in all_briefs:
                _rich_detail(b)
    elif out == "json":
        _emit(_json_out(all_briefs), save)
    elif out == "signals":
        _emit(_signals_json_out(all_briefs), save)
    elif out == "csv":
        _emit(_csv_out(all_briefs), save)


# filing

@cli.command()
@click.argument("accession_or_url")
@click.option("--ticker", default="UNKNOWN", help="Ticker for labelling")
@click.option("--out",    default="table",   show_default=True,
              type=click.Choice(["table", "json", "signals", "csv"]))
@click.option("--llm",    default="none",    show_default=True,
              help="LLM provider (anthropic|ollama|claude|none)")
@click.option("--save",   default=None,      help="Save output to this file path")
def filing(accession_or_url, ticker, out, llm, save) -> None:
    """Analyse a single 8-K by ACCESSION number or direct exhibit URL."""
    _apply_llm_option(llm)

    from kevin.pipeline import analyse_filing

    with console.status(f"Fetching and analysing {accession_or_url}..."):
        brief = asyncio.run(analyse_filing(accession_or_url, ticker=ticker))

    if brief is None:
        console.print("[red]Analysis failed - could not retrieve filing.[/red]")
        sys.exit(1)

    if out == "table":
        _rich_detail(brief)
    elif out == "json":
        _emit(_json_out([brief]), save)
    elif out == "signals":
        _emit(_signals_json_out([brief]), save)
    elif out == "csv":
        _emit(_csv_out([brief]), save)


# batch

@cli.command()
@click.argument("ticker_file", type=click.Path(exists=True))
@click.option("--since",  default=30,              show_default=True)
@click.option("--out",    default="csv",            show_default=True,
              type=click.Choice(["json", "signals", "csv"]))
@click.option("--llm",    default="none",           show_default=True)
@click.option("--save",   default="kevin_signals.csv", show_default=True)
def batch(ticker_file, since, out, llm, save) -> None:
    """
    Bulk process a list of tickers from a plain-text file (one ticker per line).

    Example:
        echo -e "NVDA\\nMSFT\\nAAPL" > tickers.txt
        kevin batch tickers.txt --save signals.csv
    """
    _apply_llm_option(llm)
    tickers = [
        line.strip().upper()
        for line in Path(ticker_file).read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    if not tickers:
        console.print("[red]No tickers found in file.[/red]")
        sys.exit(1)

    console.print(f"[cyan]Processing {len(tickers)} tickers...[/cyan]")

    from kevin.pipeline import analyse_ticker

    # Run all tickers concurrently inside a single event loop — no overhead of
    # creating/destroying an event loop per ticker.
    async def _run_all() -> list[KevinBrief]:
        tasks   = [analyse_ticker(t, since_days=since) for t in tickers]
        results = await asyncio.gather(*tasks)
        return [b for group in results for b in group]

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(f"Scanning {len(tickers)} tickers...", total=None)
        all_briefs = asyncio.run(_run_all())
        progress.update(task, description=f"Done — {len(all_briefs)} brief(s)")

    all_briefs.sort(key=lambda b: b.index.filed_at, reverse=True)

    if out == "csv":
        output = _csv_out(all_briefs)
    elif out == "signals":
        output = _signals_json_out(all_briefs)
    else:
        output = _json_out(all_briefs)

    _emit(output, save)
    console.print(f"[green]{len(all_briefs)} briefs written to {save}[/green]")


# watch

@cli.command()
@click.argument("tickers", nargs=-1, required=True)
@click.option("--interval", default=300, show_default=True, help="Poll interval in seconds")
@click.option("--since",    default=7,   show_default=True,
              help="Lookback window in days (sets how far back the first poll reaches)")
@click.option("--llm",      default="none", show_default=True)
@click.option("--out",      default="table", type=click.Choice(["table", "json", "csv"]))
def watch(tickers, interval, since, llm, out) -> None:
    """
    Poll EDGAR for new 8-K filings for TICKERS every INTERVAL seconds.

    On first run, shows a summary of all filings found in the lookback window
    so you can confirm it is working.  After that, only NEW filings trigger output.

    Press Ctrl-C to stop.
    """
    _apply_llm_option(llm)
    from kevin.pipeline import analyse_ticker

    ticker_str = ", ".join(t.upper() for t in tickers)

    async def _watch_loop() -> None:
        seen: set[str] = set()
        first_poll = True

        console.print(
            f"[cyan]Kevin watching {ticker_str}"
            f" - polling every {interval}s, lookback {since}d. Ctrl-C to stop.[/cyan]"
        )

        while True:
            tasks  = [analyse_ticker(t, since_days=since) for t in tickers]
            groups = await asyncio.gather(*tasks)
            briefs = [b for group in groups for b in group]
            new    = [b for b in briefs if b.index.accession not in seen]
            for b in new:
                seen.add(b.index.accession)

            if first_poll:
                if new:
                    console.print(
                        f"[dim]Loaded {len(new)} existing filing(s) from the last {since}d "
                        f"— watching for new ones.[/dim]"
                    )
                    if out == "table":
                        console.print(_rich_table(new))
                else:
                    console.print(
                        f"[dim]No filings found in the last {since}d for {ticker_str}. "
                        f"Watching for new ones — try --since to extend the window.[/dim]"
                    )
                first_poll = False
            else:
                for b in new:
                    console.print(
                        f"\n[bold yellow]NEW FILING[/bold yellow]  "
                        f"[yellow]{b.signal.ticker}[/yellow]  {b.index.accession}"
                    )
                    if out == "table":
                        _rich_detail(b)
                    elif out == "json":
                        console.print_json(_json_out([b]))
                    elif out == "csv":
                        console.print(_csv_out([b]))

            console.print(
                f"[dim]{time.strftime('%H:%M:%S')} - tracking {len(seen)} filing(s),"
                f" next poll in {interval}s...[/dim]"
            )
            await asyncio.sleep(interval)

    try:
        asyncio.run(_watch_loop())
    except KeyboardInterrupt:
        console.print("\n[yellow]Watch stopped.[/yellow]")


if __name__ == "__main__":
    cli()
