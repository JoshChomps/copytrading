"""CLI entrypoint: python -m copytrade <command>."""

from __future__ import annotations

import asyncio
import json
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import typer

from copytrade.config import load_settings
from copytrade.store import db as store

app = typer.Typer(help="Disclosure-based copy-trading research pipeline.", no_args_is_help=True)


def _open():
    settings = load_settings()
    conn = store.connect(settings.db_path)
    store.run_migrations(conn)
    return conn, settings


def _fmt(value, spec: str, dash: str = ", ") -> str:
    """Format a possibly-missing number for a fixed-width report cell.

    Reports are full of statistics that are legitimately undefined (too few
    clusters, no control, an empty subset). Printing a bare 'nan' reads as a
    bug; printing a dash reads as 'not applicable'.
    """
    return spec.format(value) if pd.notna(value) else dash


@app.command()
def migrate():
    """Apply pending SQL migrations."""
    conn, _ = _open()
    ran = store.run_migrations(conn)
    typer.echo(f"applied: {ran or 'nothing (up to date)'}")


@app.command()
def demo(
    seed: int = 42, start: str = "2019-01-01", end: str = "2024-12-31",
    force: bool = typer.Option(False, help="allow seeding into a DB that already holds real data"),
):
    """Seed a deterministic synthetic dataset (offline end-to-end demo).

    Refuses to write into a DB containing REAL ingested trades, synthetic and
    real data must never mix (use COPYTRADE_DB_PATH=data/demo.db)."""
    from copytrade.demo.synthetic import generate

    conn, settings = _open()
    real = conn.execute("SELECT COUNT(*) FROM trades WHERE source != 'demo'").fetchone()[0]
    if real and not force:
        typer.echo(f"REFUSING: {settings.db_path} already holds {real} real trades; "
                   "demo data would contaminate them. Use COPYTRADE_DB_PATH=data/demo.db "
                   "(what `make demo` does) or --force if you truly mean it.")
        raise typer.Exit(1)
    summary = generate(conn, seed=seed, start=start, end=end)
    typer.echo(json.dumps(summary, indent=2))


@app.command("backfill-form4")
def backfill_form4(days: int = 30):
    """Ingest recent SEC Form 4 filings via the EDGAR daily indexes."""
    from copytrade.ingest.edgar import EdgarClient, backfill_days
    from copytrade.ingest.form4 import parse_form4, store_form4

    conn, settings = _open()

    async def run():
        client = EdgarClient(settings.edgar_user_agent, settings.edgar_rps, settings.raw_dir)
        inserted = filings = 0
        try:
            for day in backfill_days(date.today(), days):
                refs = await client.daily_form4_accessions(day)
                typer.echo(f"{day}: {len(refs)} Form 4 filings")
                for ref in refs:
                    try:
                        xml = await client.fetch_filing_xml(ref)
                        if not xml:
                            continue
                        accepted = await client.fetch_acceptance_datetime(ref) \
                            or f"{day} 17:00:00"
                        n = store_form4(conn, parse_form4(xml), ref.accession, accepted,
                                        raw_ref=str(Path(settings.raw_dir) / f"{ref.accession}.xml"))
                        inserted += n
                        filings += 1
                    except Exception as e:  # keep going; one bad filing shouldn't kill a backfill
                        typer.echo(f"  ! {ref.accession}: {e}")
        finally:
            await client.aclose()
        typer.echo(f"done: {filings} filings parsed, {inserted} trades inserted")

    asyncio.run(run())


@app.command("backfill-congress")
def backfill_congress(
    years: int = 2,
    json_file: str = typer.Option("", help="load a local JSON dump instead of calling Finnhub"),
):
    """Ingest congressional trades (Finnhub aggregator, or a local JSON dump)."""
    from copytrade.ingest.congress import (
        FinnhubCongress,
        JsonFileCongress,
        load_universe,
        store_congress_trades,
    )

    conn, settings = _open()

    async def run():
        source = None
        if json_file:
            source = JsonFileCongress(json_file)
            trades, skipped = await source.fetch()
        else:
            symbols = load_universe(settings.congress_universe_file)
            if not symbols:
                typer.echo(f"universe file missing/empty: {settings.congress_universe_file} "
                           "(one ticker per line, '#' comments ok)")
                raise typer.Exit(1)
            source = FinnhubCongress(settings.finnhub_api_key)
            frm = (date.today() - timedelta(days=365 * years)).isoformat()
            trades, skipped = await source.fetch(symbols, frm, date.today().isoformat())
        inserted, deduped = store_congress_trades(conn, trades)
        typer.echo(
            f"done: {inserted} inserted, {deduped} duplicates, "
            f"{skipped} skipped (missing filing/disclosure date, never backdated)"
        )
        errors = getattr(source, "errors", {})
        if errors:
            typer.secho(f"WARNING: {len(errors)} symbols failed "
                        f"(e.g. {dict(list(errors.items())[:3])})", fg=typer.colors.YELLOW)
            if getattr(source, "all_failed_uniformly", False):
                typer.secho(
                    "every symbol failed identically, your Finnhub tier likely does "
                    "not include /stock/congressional-trading (it moved to premium for "
                    "some tiers). Alternatives: backfill-quiver (paid) or the "
                    "authoritative house_clerk/senate_efd scrapers (free).",
                    fg=typer.colors.YELLOW,
                )

    asyncio.run(run())


@app.command("ingest-senate")
def ingest_senate(
    repo_dir: str = typer.Option("data/senate-stock-watcher", help="where to clone/find the data repo"),
    no_network: bool = typer.Option(False, help="skip clone/pull, use existing checkout"),
):
    """Ingest REAL Senate PTRs (with real disclosure dates) from the
    senate-stock-watcher GitHub repo. Works even where sec.gov/finnhub are
    blocked: only needs git access to github.com. Backtest-grade data
    (~2013-2021); see docs/CANADA.md for live-signal options."""
    from copytrade.ingest.senate_watcher import clone_or_pull, ingest_repo

    conn, settings = _open()
    if not no_network:
        clone_or_pull(repo_dir)
    res = ingest_repo(conn, repo_dir)
    typer.echo(json.dumps(res, indent=2))
    if res.get("staleness_warning"):
        typer.secho(f"WARNING: {res['staleness_warning']}", fg=typer.colors.YELLOW)
    # party/state enrichment (unitedstates/congress-legislators, also on
    # raw.githubusercontent so it works wherever the senate clone worked)
    try:
        from copytrade.ingest.legislators import enrich_congress_filers, load_index

        stats = enrich_congress_filers(conn, load_index(settings.raw_dir))
        typer.echo(f"party enrichment: {stats['matched']} matched, "
                   f"{stats['unmatched']} unmatched"
                   + (f" ({', '.join(stats['unmatched_names'][:5])}...)"
                      if stats["unmatched"] else ""))
    except Exception as e:  # noqa: BLE001, enrichment is additive, never fatal
        typer.secho(f"party enrichment skipped: {e}", fg=typer.colors.YELLOW)


@app.command("ingest-committees")
def ingest_committees(
    as_of: str = typer.Option("", help="snapshot date; defaults to today"),
):
    """Ingest congressional COMMITTEE assignments + jurisdictions.

    This is the last untested channel: the academic claim (Ziobrowski et al.)
    is that members trade better INSIDE their committee's jurisdiction, which
    is where an informational advantage would plausibly sit. Every other angle
    here treats filers as interchangeable and all came back null.

    SNAPSHOT, NOT HISTORY. Upstream publishes current membership only, so rows
    are stamped `as_of`. Applying today's roster to a 2016 trade would be
    look-ahead; `jurisdiction` enforces the date restriction rather than
    trusting the caller to remember it."""
    from copytrade.ingest.committees import refresh

    conn, _ = _open()
    res = refresh(conn, as_of or None)
    typer.echo(json.dumps(res, indent=2))
    if res["filers_no_bioguide"]:
        typer.secho(
            f"{res['filers_no_bioguide']} congress filers have no bioguide id and "
            f"cannot be placed on a committee, run `enrich-legislators` first.",
            fg=typer.colors.YELLOW)
    if res["filers_no_seat"]:
        typer.secho(
            f"{res['filers_no_seat']} filers have a bioguide but no current seat "
            f"(left Congress, or serve on no committee). Any jurisdiction test "
            f"silently excludes them.", fg=typer.colors.YELLOW)


@app.command("ingest-committees-history")
def ingest_committees_history(
    repo_dir: str = typer.Option("data/congress-legislators",
                                 help="where to clone/find the upstream repo"),
    min_gap_days: int = typer.Option(60, help="approx. spacing between snapshots"),
    since: str = typer.Option("2014-01-01", help="ignore revisions before this"),
):
    """POINT-IN-TIME committee membership, walked from the upstream git history.

    This is what makes the jurisdiction test usable. With only the current
    roster, scoring a 2016 trade means applying 2026 committee assignments
    look-ahead: so the test has to be restricted to the current Congress and
    loses ~96% of the sample. The upstream repo has 200+ historical revisions
    of the membership file; walking them gives the roster actually in force on
    each filing date.

    Uses a blobless clone (`--filter=blob:none`), so the history is cheap to
    fetch and blobs are pulled only for the revisions actually read."""
    from copytrade.ingest.committees import refresh_history

    conn, _ = _open()
    res = refresh_history(conn, repo_dir, min_gap_days, since)
    typer.echo(json.dumps(res, indent=2))
    if res["snapshots_in_db"] > 1:
        typer.secho(
            f"\n{res['snapshots_in_db']} snapshots spanning {res['earliest']} → "
            f"{res['latest']}. `jurisdiction` will now score each trade against "
            f"the roster in force on its filing date, so the current-Congress "
            f"restriction no longer applies.", fg=typer.colors.GREEN)


@app.command("enrich-legislators")
def enrich_legislators(refresh: bool = typer.Option(False, help="re-download the YAML")):
    """(Re)match congress filers to unitedstates/congress-legislators for
    party/state metadata (powers the party-cohort analysis)."""
    from copytrade.ingest.legislators import enrich_congress_filers, load_index

    conn, settings = _open()
    stats = enrich_congress_filers(conn, load_index(settings.raw_dir, refresh=refresh))
    typer.echo(json.dumps(stats, indent=2))


@app.command("fx-refresh")
def fx_refresh(start: str = "2012-01-01"):
    """Cache daily USD/CAD (Bank of Canada Valet, falling back to the Fed
    H.10 mirror). Real observed rates only."""
    from copytrade.ingest.fx import refresh_usdcad

    conn, _ = _open()
    typer.echo(json.dumps(refresh_usdcad(conn, start), indent=2))


@app.command()
def allocate(
    tfsa_room: float = typer.Option(109_000.0, help="your CRA-verified TFSA room, CAD"),
    taxable_capital: float = typer.Option(100_000.0, help="capital for the margin sleeve, CAD"),
):
    """TFSA vs margin sleeve plan from the current ranking (Canadian resident:
    short-turnover copying stays OUT of the TFSA, CRA business-income risk)."""
    from copytrade.accounts.allocation import allocate_sleeves
    from copytrade.backtest.costs import get_cost_model
    from copytrade.rank.rank import build_ranking

    conn, settings = _open()
    trades = store.trades_frame(conn, instruments=["stock"])
    prices_m = store.load_price_matrix(conn)
    if settings.benchmark not in prices_m.columns:
        typer.echo(f"no {settings.benchmark} prices cached, run `python -m copytrade prices` "
                   "first (needs yfinance network access; see docs/CANADA.md obstacle map)")
        raise typer.Exit(1)
    ranking, _ = build_ranking(
        trades, prices_m, prices_m[settings.benchmark].dropna(),
        get_cost_model(settings.cost_model), horizon=settings.horizon,
        min_trades_rank=settings.min_trades_rank,
        same_day_close=settings.same_day_close, cutoff=settings.same_day_cutoff,
        amount_basis=settings.amount_basis, trade_notional=settings.trade_notional,
    )
    plan = allocate_sleeves(ranking, trades, tfsa_room, taxable_capital)
    if not plan.assignments.empty:
        typer.echo(plan.assignments.to_string(index=False))
    typer.echo(f"\nTFSA deployed: ${plan.tfsa_deployed:,.0f}   "
               f"margin deployed: ${plan.margin_deployed:,.0f}")
    for n in plan.notes:
        typer.echo(f"  * {n}")


@app.command("orders-from-signals")
def orders_from_signals(
    since: str = typer.Option("", help="only trades filed on/after this timestamp"),
    notional: float = typer.Option(0.0, help="per-order notional; 0 = settings.trade_notional"),
):
    """Create local paper orders from BUY filings by top-ranked filers."""
    from copytrade.broker.paper import record_signal_orders
    from copytrade.poll import top_filer_stats

    conn, settings = _open()
    watch = set(top_filer_stats(settings))
    if not watch:
        typer.echo("no ranking.csv yet: run `report` first so there is a watchlist")
        raise typer.Exit(1)
    n = record_signal_orders(
        conn, notional=notional or settings.trade_notional,
        since=since or None, watchlist=watch,
    )
    typer.echo(f"created {n} paper orders (watchlist: {len(watch)} filers)")


@app.command("orders-fill")
def orders_fill():
    """Fill open local paper orders at the first close AFTER order creation."""
    from copytrade.broker.paper import fill_open_orders

    conn, _ = _open()
    typer.echo(json.dumps(fill_open_orders(conn), indent=2))


@app.command("orders-push-alpaca")
def orders_push_alpaca():
    """Mirror open paper orders into an Alpaca PAPER account (keys from env)."""
    import os

    from copytrade.broker.paper import AlpacaPaper, push_orders_to_alpaca

    conn, _ = _open()
    client = AlpacaPaper(
        os.environ.get("ALPACA_API_KEY_ID", ""),
        os.environ.get("ALPACA_API_SECRET_KEY", ""),
        os.environ.get("ALPACA_PAPER_BASE_URL", "https://paper-api.alpaca.markets"),
    )
    try:
        typer.echo(json.dumps(push_orders_to_alpaca(conn, client), indent=2))
    finally:
        client.close()


@app.command("orders-list")
def orders_list(limit: int = 50):
    """Show the most recent paper orders."""
    conn, _ = _open()
    rows = conn.execute(
        "SELECT order_id, created_at, broker, account, ticker, side, notional, "
        "status, fill_price, fill_date, note FROM orders "
        "ORDER BY order_id DESC LIMIT ?", (limit,),
    ).fetchall()
    for r in rows:
        typer.echo(" | ".join("" if v is None else str(v) for v in r))
    if not rows:
        typer.echo("(no orders)")


@app.command("backfill-form345")
def backfill_form345(
    quarter: str = typer.Argument(..., help="e.g. 2024q1"),
    zip_file: str = typer.Option("", help="use a local ZIP instead of downloading"),
    limit: int = typer.Option(0, help="stop after N transaction rows (validation runs)"),
):
    """Bulk insider backfill from the SEC DERA Insider Transactions Data Sets
    (one quarterly ZIP = every Form 3/4/5 that quarter; day-precision anchor)."""
    from copytrade.ingest.form345_bulk import download_quarter, ingest_zip

    conn, settings = _open()
    if zip_file:
        path = Path(zip_file)
    else:
        year, q = quarter.lower().split("q")
        path = asyncio.run(download_quarter(
            int(year), int(q), Path(settings.raw_dir) / "dera", settings.edgar_user_agent
        ))
    typer.echo(json.dumps(ingest_zip(conn, path, limit=limit), indent=2))


@app.command("backfill-13f")
def backfill_13f(
    cik: str = typer.Argument(..., help="fund manager CIK, e.g. 0001067983 (Berkshire)"),
    quarters: int = typer.Option(5, help="most recent quarters to load (deltas need 2+)"),
):
    """Ingest a manager's 13F-HR holdings and derive quarter-over-quarter
    buy/sell deltas, anchored at each filing's ACCEPTANCE time (never the
    quarter end). CUSIPs map to tickers via OpenFIGI (free, keyless tier)."""
    from copytrade.ingest.thirteenf import backfill_13f_cik

    conn, settings = _open()
    res = asyncio.run(backfill_13f_cik(
        conn, cik, quarters, settings.edgar_user_agent, settings.edgar_rps,
        settings.raw_dir, settings.openfigi_api_key,
    ))
    typer.echo(json.dumps(res, indent=2))


@app.command("backfill-quiver")
def backfill_quiver(limit: int = typer.Option(0, help="ingest only the first N records (validation)")):
    """One-shot pull of QuiverQuant's bulk live congress feed (both chambers,
    no symbol blind spot). Needs COPYTRADE_QUIVER_API_KEY. Run with
    --limit 20 first and eyeball the rows."""
    from copytrade.ingest.quiver import fetch_live, parse_quiver_payload
    from copytrade.ingest.congress import store_congress_trades

    conn, settings = _open()
    if not settings.quiver_api_key:
        typer.echo("no COPYTRADE_QUIVER_API_KEY configured")
        raise typer.Exit(1)
    payload = asyncio.run(fetch_live(settings.quiver_api_key))
    if limit:
        payload = payload[:limit]
    trades, skipped = parse_quiver_payload(payload)
    ins, dedup = store_congress_trades(conn, trades)
    typer.echo(json.dumps({"records": len(payload), "inserted": ins,
                           "deduped": dedup, "skipped_unusable": skipped}, indent=2))


# The 'powerful people' whose 13F options are worth watching. CIKs are stable
# SEC identifiers; verified against EDGAR full-text search.
FAMOUS_FUNDS = {
    "Scion (Michael Burry)": "0001649339",
    "Berkshire Hathaway (Buffett)": "0001067983",
    "Pershing Square (Ackman)": "0001336528",
    "Bridgewater (Dalio)": "0001350694",
    "Appaloosa (Tepper)": "0001656456",
    "Third Point (Loeb)": "0001040273",
    "Greenlight (Einhorn)": "0001079114",
    "Duquesne (Druckenmiller)": "0001536411",
}


@app.command("backfill-funds")
def backfill_funds(
    quarters: int = typer.Option(6, help="quarters of 13F history per fund"),
    only: str = typer.Option("", help="substring filter on fund name; empty = all"),
):
    """Backfill famous hedge-fund 13F holdings (Burry/Buffett/Ackman/...),
    deriving QoQ buy/sell deltas AND put/call option positions for the
    derivatives section. Needs data.sec.gov (works off-sandbox / in Actions)."""
    from copytrade.ingest.thirteenf import backfill_13f_cik

    conn, settings = _open()
    funds = {k: v for k, v in FAMOUS_FUNDS.items() if only.lower() in k.lower()}

    async def run():
        out = {}
        for name, cik in funds.items():
            try:
                res = await backfill_13f_cik(
                    conn, cik, quarters, settings.edgar_user_agent, settings.edgar_rps,
                    settings.raw_dir, settings.openfigi_api_key)
                out[name] = res
            except Exception as e:  # noqa: BLE001, one fund must not sink the rest
                out[name] = {"error": str(e)[:200]}
            typer.echo(f"  {name}: {out[name]}")
        return out

    asyncio.run(run())


@app.command("news-refresh")
def news_refresh(
    tickers: str = typer.Option("", help="comma-separated; default = tickers of open orders + recent watchlist buys"),
    days: int = typer.Option(14, help="Finnhub lookback window"),
):
    """Refresh company news (Finnhub with your key, else keyless GDELT) for
    alert/dashboard context. News is context, never a backtest factor."""
    from copytrade.ingest.news import refresh_news

    conn, settings = _open()
    if tickers:
        syms = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    else:
        syms = [r[0] for r in conn.execute(
            "SELECT DISTINCT ticker FROM orders WHERE status='new' "
            "UNION SELECT DISTINCT ticker FROM trades "
            "WHERE side='buy' AND filing_datetime >= DATETIME('now','-14 days') "
            "AND ticker IS NOT NULL LIMIT 50"
        )]
    if not syms:
        typer.echo("no tickers to refresh (pass --tickers or create orders first)")
        raise typer.Exit(0)
    frm = (date.today() - timedelta(days=days)).isoformat()
    res = asyncio.run(refresh_news(
        conn, syms, settings.finnhub_api_key, frm, date.today().isoformat()
    ))
    typer.echo(json.dumps(res, indent=2))


@app.command()
def prices(start: str = "2015-01-01", end: str = ""):
    """Cache daily adjusted closes for every traded ticker + SPY/TLT
    (yfinance -> Tiingo (if key) -> stooq fallback chain, provenance reported)."""
    from copytrade.ingest.market import refresh_prices

    conn, settings = _open()
    # RSP (equal-weight S&P) and IWM (small-cap) are benchmarks for the factor
    # / breadth tests; TLT for the 60/40 sleeve; QQQ (growth/high-return),
    # VTI (total market), SCHD (dividend/defensive) for the popular-ETF scorecard
    res = refresh_prices(conn, start, end or date.today().isoformat(),
                         extra_tickers=(settings.benchmark, "TLT", "RSP", "IWM",
                                        "QQQ", "VTI", "SCHD"),
                         tiingo_api_key=settings.tiingo_api_key)
    typer.echo(json.dumps(res, indent=2))


@app.command()
def rank(top: int = 20):
    """Print the filer ranking (lag-adjusted alpha at the configured horizon)."""
    from copytrade.backtest.costs import get_cost_model
    from copytrade.rank.rank import build_ranking

    conn, settings = _open()
    trades = store.trades_frame(conn, instruments=["stock"])
    prices_m = store.load_price_matrix(conn)
    if settings.benchmark not in prices_m.columns:
        typer.echo(f"no {settings.benchmark} prices cached, run `python -m copytrade prices` "
                   "first (needs yfinance network access; see docs/CANADA.md obstacle map)")
        raise typer.Exit(1)
    ranking, diag = build_ranking(
        trades, prices_m, prices_m[settings.benchmark].dropna(),
        get_cost_model(settings.cost_model), horizon=settings.horizon,
        min_trades_rank=settings.min_trades_rank,
        same_day_close=settings.same_day_close, cutoff=settings.same_day_cutoff,
        amount_basis=settings.amount_basis, trade_notional=settings.trade_notional,
    )
    typer.echo(f"diagnostics: {diag}")
    if not ranking.empty:
        typer.echo(ranking.head(top).to_string(index=False))


@app.command()
def profits(
    top: int = 20,
    min_buys: int = typer.Option(1, help="include filers with at least this many priced buys"),
):
    """Per-filer track record (realized returns) with NO significance gate
    low-frequency high-conviction pickers surface here even when they can't
    clear the walk-forward's n>=30 'copyable' bar."""
    from copytrade.backtest.costs import get_cost_model
    from copytrade.rank.profit import filer_track_records

    conn, settings = _open()
    trades = store.trades_frame(conn, instruments=["stock"])
    prices_m = store.load_price_matrix(conn)
    if settings.benchmark not in prices_m.columns:
        typer.echo(f"no {settings.benchmark} prices cached, run `python -m copytrade prices` first")
        raise typer.Exit(1)
    tr = filer_track_records(
        trades, prices_m, prices_m[settings.benchmark].dropna(),
        get_cost_model(settings.cost_model),
        same_day_close=settings.same_day_close, cutoff=settings.same_day_cutoff,
        min_buys=min_buys,
    )
    if tr.empty:
        typer.echo("no priced track records (need prices + at least one matchable buy)")
        raise typer.Exit(0)
    show = tr.head(top)[["filer", "kind", "party", "n_buys_priced", "n_closed",
                         "win_rate", "mean_return", "total_pnl_per_10k",
                         "annualized_per_pick", "avg_holding_days", "confidence"]]
    typer.echo(show.to_string(index=False))


@app.command()
def lag():
    """Disclosure-lag decomposition: how stale each source's signals are by
    the time you can legally act, and which filers disclose fastest (the only
    real lever on congressional reporting lag). See docs/SCRAPING.md for the
    pipeline-lag minimization playbook."""
    from copytrade.rank.lag import fastest_filers, lag_by_source

    conn, _ = _open()
    trades = store.trades_frame(conn, instruments=None)
    by_src = lag_by_source(trades)
    typer.echo("=== Reporting lag (transaction -> filing), by source ===")
    typer.echo(by_src.to_string(index=False) if not by_src.empty else "(no dated trades)")
    fast = fastest_filers(trades)
    if not fast.empty:
        typer.echo("\n=== Fastest-disclosing filers (freshest signals) ===")
        typer.echo(fast.to_string(index=False))
    typer.echo("\nReporting lag is the filer's own delay (congress: up to 45d by law) "
               "and cannot be shrunk by software, prefer fast disclosers. Pipeline "
               "lag (our poll cadence + EDGAR real-time feed) is minimized in config; "
               "see docs/SCRAPING.md.")


@app.command("cftc-refresh")
def cftc_refresh(
    markets: str = typer.Option("S&P 500,NASDAQ,GOLD,CRUDE OIL",
                                help="comma-separated market name filters; empty = all"),
):
    """Cache the latest CFTC Commitments of Traders (futures positioning of
    large speculators). Context/framing only, never a priced backtest input."""
    from copytrade.ingest.cftc import refresh_cot

    conn, _ = _open()
    mkts = [m.strip() for m in markets.split(",") if m.strip()] if markets else None
    typer.echo(json.dumps(refresh_cot(conn, markets=mkts), indent=2))


@app.command()
def derivatives(horizon: int = 0):
    """Derivatives & futures section: framing of stock vs option vs futures
    bets by powerful people. Options show the UNDERLYING's directional move
    from filing (not option P&L, leverage is unmeasured); futures show CFTC
    large-speculator net positioning."""
    from copytrade.backtest.costs import get_cost_model
    from copytrade.rank.derivatives import (
        cot_snapshot,
        instrument_framing,
        option_direction_study,
        stock_vs_option_by_filer,
    )
    from copytrade.rank.profit import filer_track_records

    conn, settings = _open()
    h = horizon or settings.horizon
    all_trades = store.trades_frame(conn, instruments=None)   # every instrument
    typer.echo("=== Instrument framing (all disclosed activity) ===")
    fr = instrument_framing(all_trades)
    typer.echo(fr.to_string(index=False) if not fr.empty else "(no trades)")

    prices_m = store.load_price_matrix(conn)
    if settings.benchmark in prices_m.columns:
        bench = prices_m[settings.benchmark].dropna()
        opt = option_direction_study(all_trades, prices_m, bench, horizon=h,
                                     same_day_close=settings.same_day_close,
                                     cutoff=settings.same_day_cutoff)
        typer.echo(f"\n=== Options: underlying directional edge at {h}d (filing-anchored) ===")
        if opt.empty:
            typer.echo("(no option trades priceable, options are stored but leverage is unmeasured)")
        else:
            typer.echo(f"{len(opt)} option signals, mean signed underlying excess "
                       f"{opt['underlying_excess'].mean():+.2%}, "
                       f"hit rate {(opt['underlying_excess'] > 0).mean():.0%}")
            stock_track = filer_track_records(
                store.trades_frame(conn, instruments=["stock"]), prices_m, bench,
                get_cost_model(settings.cost_model),
                same_day_close=settings.same_day_close, cutoff=settings.same_day_cutoff)
            framed = stock_vs_option_by_filer(stock_track, opt)
            if not framed.empty:
                typer.echo("\nStock edge vs option edge, per filer (is the edge picking or leverage?):")
                typer.echo(framed.head(15).to_string(index=False))
    else:
        typer.echo("\n(no benchmark prices, run `prices` for the option directional study)")

    typer.echo("\n=== Futures: CFTC large-speculator net positioning ===")
    snap = cot_snapshot(conn)
    typer.echo(snap.to_string(index=False) if not snap.empty else
               "(no COT data: run `python -m copytrade cftc-refresh`)")


@app.command()
def mirror(
    filer: str = typer.Argument(..., help="filer name (substring match), e.g. 'Pelosi'"),
    exit_on_sell: bool = typer.Option(True, help="also copy the filer's sells as exits"),
):
    """Mirror ONE named filer's buys vs SPY, net of costs, point-in-time
    (filing-anchored). The honest 'copy one famous member' test, concentrated,
    no significance gate, no pretence the filer was selectable in advance.
    Options are excluded (unpriceable), so a member whose real edge is options
    will look weaker here than the headlines, that gap is the point."""
    from copytrade.backtest import benchmarks as bm
    from copytrade.backtest import metrics
    from copytrade.backtest.costs import get_cost_model
    from copytrade.backtest.engine import walk_forward

    conn, settings = _open()
    trades = store.trades_frame(conn, instruments=["stock"])
    prices_m = store.load_price_matrix(conn)
    if settings.benchmark not in prices_m.columns:
        typer.echo(f"no {settings.benchmark} prices cached, run `prices` first")
        raise typer.Exit(1)
    bench = prices_m[settings.benchmark].dropna()
    names = sorted({n for n in trades["filer_name"].dropna().unique()
                    if filer.lower() in n.lower()})
    if not names:
        typer.echo(f"no filer matching {filer!r}. Try `profits` to see available names.")
        raise typer.Exit(1)
    if len(names) > 1:
        typer.echo(f"'{filer}' matches {len(names)}: {names}, mirroring all of them together")
    cost = get_cost_model(settings.cost_model)
    common = dict(horizon=settings.horizon, max_weight=0.20, min_trades=1,
                  trade_notional=settings.trade_notional,
                  same_day_close=settings.same_day_close, cutoff=settings.same_day_cutoff)
    wf = walk_forward(trades, prices_m, bench, cost, only_filers=set(names), **common)
    if wf is None or wf.equity.empty:
        typer.echo(f"{names}: no priceable stock buys to mirror (options are excluded)")
        raise typer.Exit(0)
    curves = {f"Mirror {filer}": wf.equity}
    if exit_on_sell:
        wf_x = walk_forward(trades, prices_m, bench, cost, only_filers=set(names),
                            exit_on_filer_sell=True, **common)
        if wf_x is not None:
            curves[f"Mirror {filer} + copy-exit"] = wf_x.equity
    spy = bm.buy_and_hold(bench.loc[wf.equity.index[0]:wf.equity.index[-1]], cost)
    curves[f"{settings.benchmark} buy & hold"] = spy
    typer.echo(f"\nMirroring {names}  ({wf.equity.index[0].date()} → {wf.equity.index[-1].date()}, "
               f"net of {cost.name})\n")
    typer.echo(f"{'strategy':<34} {'total':>9} {'CAGR':>7} {'Sharpe':>7} {'Sortino':>8} {'maxDD':>7}")
    for name, eq in curves.items():
        s = metrics.perf_stats(eq.dropna())
        if not s:
            continue
        sortino = f"{s['sortino']:.2f}" if pd.notna(s.get("sortino")) else "n/a"
        typer.echo(f"{name:<34} {s['total_return']:>+8.1%} {s['cagr']:>+6.1%} "
                   f"{s['sharpe']:>7.2f} {sortino:>8} {s['max_drawdown']:>6.1%}")
    n_opt = conn.execute(
        "SELECT COUNT(*) FROM trades t JOIN filers f USING(filer_id) "
        "WHERE t.instrument != 'stock' AND (" +
        " OR ".join(["f.name LIKE ?"] * len(names)) + ")",
        [f"%{n}%" for n in names],
    ).fetchone()[0]
    if n_opt:
        typer.secho(f"\nnote: {n_opt} non-stock (option/other) trades by this filer are "
                    "EXCLUDED: their real leverage is not captured here.", fg=typer.colors.YELLOW)


@app.command()
def recent(
    since: str = typer.Option("2022-01-01", help="ISO date; analyze only trades FILED on/after this"),
    mirrors: str = typer.Option("Pelosi,Greene,Crenshaw", help="comma-separated filers to mirror in-window"),
):
    """Recent-era viability: is copying powerful people worth it NOW, not on
    the 2012-2026 average? Restricts to trades FILED (actionable) on/after
    `since`, then compares copy-all and top-N selection against SPY over the
    SAME window, all point-in-time and net of costs, plus who's actually been
    hot lately and named single-filer mirrors. Same discipline as the full
    backtest; just a recent slice."""
    from copytrade.backtest import benchmarks as bm
    from copytrade.backtest import metrics
    from copytrade.backtest.costs import get_cost_model
    from copytrade.backtest.engine import walk_forward
    from copytrade.rank.profit import filer_track_records

    conn, settings = _open()
    cost = get_cost_model(settings.cost_model)
    trades = store.trades_frame(conn, instruments=["stock"], since=since)
    prices_m = store.load_price_matrix(conn)
    if trades.empty or settings.benchmark not in prices_m.columns:
        typer.echo("no in-window stock trades or no benchmark prices, run `prices` first")
        raise typer.Exit(1)
    bench = prices_m[settings.benchmark].dropna()
    n_buys = int((trades["side"] == "buy").sum())
    typer.echo(f"=== Recent-era copy-trading viability (filed ≥ {since}) ===")
    typer.echo(f"{len(trades)} stock trades, {n_buys} buys, "
               f"{trades['filer_name'].nunique()} filers, net of {cost.name}\n")

    def show(curves: dict):
        typer.echo(f"{'strategy':<34} {'total':>9} {'CAGR':>7} {'Sharpe':>7} {'maxDD':>7}")
        for name, eq in curves.items():
            s = metrics.perf_stats(eq.dropna())
            if s:
                typer.echo(f"{name:<34} {s['total_return']:>+8.1%} {s['cagr']:>+6.1%} "
                           f"{s['sharpe']:>7.2f} {s['max_drawdown']:>6.1%}")

    common = dict(horizon=settings.horizon, max_weight=settings.max_weight,
                  trade_notional=settings.trade_notional,
                  same_day_close=settings.same_day_close, cutoff=settings.same_day_cutoff)
    curves: dict = {}
    wf_all = walk_forward(trades, prices_m, bench, cost, copy_all=True, min_trades=1, **common)
    if wf_all is not None and not wf_all.equity.empty:
        curves["Copy ALL filers (congress index)"] = wf_all.equity
        wf_top = walk_forward(trades, prices_m, bench, cost, top_n=settings.top_n,
                              min_trades=settings.min_trades_walkforward, rank_years=1, **common)
        if wf_top is not None and not wf_top.equity.empty:
            curves[f"Copy top-{settings.top_n} (selected)"] = wf_top.equity
        span = curves["Copy ALL filers (congress index)"].index
        curves[f"{settings.benchmark} buy & hold"] = bm.buy_and_hold(bench.loc[span[0]:span[-1]], cost)
    show(curves)

    typer.echo("\n=== Who's been hot in-window (point-in-time track record) ===")
    tr = filer_track_records(trades, prices_m, bench, cost,
                             same_day_close=settings.same_day_close, cutoff=settings.same_day_cutoff)
    if tr is not None and not tr.empty:
        cols = ["filer", "kind", "party", "n_buys_priced", "win_rate", "mean_return"]
        cols = [c for c in cols if c in tr.columns]
        top = tr.sort_values("mean_return", ascending=False).head(12)
        typer.echo(top[cols].to_string(index=False))

    names = [m.strip() for m in mirrors.split(",") if m.strip()]
    typer.echo("\n=== Named single-filer mirrors, in-window vs SPY ===")
    for who in names:
        who_names = sorted({n for n in trades["filer_name"].dropna().unique() if who.lower() in n.lower()})
        if not who_names:
            typer.echo(f"  {who}: no in-window stock buys")
            continue
        wf = walk_forward(trades, prices_m, bench, cost, only_filers=set(who_names),
                          min_trades=1, **common)
        if wf is None or wf.equity.empty:
            typer.echo(f"  {who}: no priceable in-window stock buys")
            continue
        span = wf.equity.index
        m = metrics.perf_stats(wf.equity.dropna())
        spy = metrics.perf_stats(bm.buy_and_hold(bench.loc[span[0]:span[-1]], cost).dropna())
        typer.echo(f"  {who:<10} {m['total_return']:>+7.1%} (SPY {spy['total_return']:>+7.1%})  "
                   f"[{span[0].date()}→{span[-1].date()}]")


@app.command()
def roster(
    min_buys: int = typer.Option(15, help="min priced buys (full history) for a filer to be analyzed"),
    min_window_buys: int = typer.Option(5, help="min buys in a window to score that window"),
    top: int = typer.Option(25, help="rows to show in each per-window ranking"),
):
    """Individual-by-individual, across windows. For EVERY filer with a real
    sample, mirror them (point-in-time, cost-net) over full history / 2020+ /
    2022+ / 2024+ and measure ALPHA vs SPY over their own active span. Then
    rank by CONSISTENCY: who beats the market in the most windows, not who
    got one lucky year. This is the honest answer to 'which people are
    actually worth following regardless of regime'."""
    import numpy as np

    from copytrade.backtest import benchmarks as bm
    from copytrade.backtest import metrics
    from copytrade.backtest.costs import get_cost_model
    from copytrade.backtest.engine import walk_forward

    conn, settings = _open()
    cost = get_cost_model(settings.cost_model)
    trades = store.trades_frame(conn, instruments=["stock"])
    prices_m = store.load_price_matrix(conn)
    if trades.empty or settings.benchmark not in prices_m.columns:
        typer.echo("no trades or no benchmark prices, run `prices` first")
        raise typer.Exit(1)
    bench = prices_m[settings.benchmark].dropna()
    common = dict(horizon=settings.horizon, max_weight=settings.max_weight,
                  trade_notional=settings.trade_notional,
                  same_day_close=settings.same_day_close, cutoff=settings.same_day_cutoff)

    buys_all = trades[trades["side"] == "buy"]
    counts = buys_all["filer_name"].value_counts()
    names = [n for n in counts.index if pd.notna(n) and counts[n] >= min_buys]
    windows = [("full", None), ("2020+", "2020-01-01"),
               ("2022+", "2022-01-01"), ("2024+", "2024-01-01")]
    typer.echo(f"=== Per-individual mirror alpha vs SPY, {len(names)} filers "
               f"(≥{min_buys} buys), net of {cost.name} ===\n")

    # pre-slice the frame once per window, then mirror each filer within it
    windowed = {}
    for wname, since in windows:
        windowed[wname] = trades if since is None else trades[
            trades["filing_datetime"] >= pd.Timestamp(since)]

    def mirror(sub, name):
        n = int(((sub["filer_name"] == name) & (sub["side"] == "buy")).sum())
        if n < min_window_buys:
            return None
        wf = walk_forward(sub, prices_m, bench, cost, only_filers={name}, min_trades=1, **common)
        if wf is None or wf.equity.empty or len(wf.equity) < 2:
            return None
        span = wf.equity.index
        m = metrics.perf_stats(wf.equity.dropna())
        spy = metrics.perf_stats(bm.buy_and_hold(bench.loc[span[0]:span[-1]], cost).dropna())
        if not m or not spy:
            return None
        return {"n": n, "total": m["total_return"], "spy": spy["total_return"],
                "alpha": m["total_return"] - spy["total_return"],
                "sharpe": m["sharpe"], "maxdd": m["max_drawdown"]}

    rows: dict[str, dict] = {}
    for name in names:
        rec = {"filer": name, "kind": buys_all[buys_all["filer_name"] == name]["filer_kind"].iloc[0]}
        for wname, _ in windows:
            rec[wname] = mirror(windowed[wname], name)
        rows[name] = rec

    def _cell(cell):
        """Render one roster window cell; distinct from the module-level
        _fmt, which formats a single number."""
        return f"{cell['alpha']:>+7.1%}(n{cell['n']})" if cell else ", "

    # per-window rankings by alpha
    for wname, _ in windows:
        scored = [(r["filer"], r["kind"], r[wname]) for r in rows.values() if r[wname]]
        scored.sort(key=lambda x: x[2]["alpha"], reverse=True)
        typer.echo(f"\n----- Best individuals, window {wname} (by alpha vs SPY over their span) -----")
        typer.echo(f"{'filer':<26}{'kind':<9}{'total':>8}{'SPY':>8}{'alpha':>8}{'Shrp':>6}{'maxDD':>7}{'n':>5}")
        for nm, kind, c in scored[:top]:
            typer.echo(f"{nm[:25]:<26}{kind:<9}{c['total']:>+7.1%} {c['spy']:>+7.1%} "
                       f"{c['alpha']:>+7.1%}{c['sharpe']:>6.2f}{c['maxdd']:>7.1%}{c['n']:>5}")

    # consistency leaderboard: beat SPY in how many scored windows?
    board = []
    for r in rows.values():
        cells = [r[w] for w, _ in windows if r[w]]
        if not cells:
            continue
        beat = sum(1 for c in cells if c["alpha"] > 0)
        board.append((r["filer"], r["kind"], beat, len(cells),
                      float(np.mean([c["alpha"] for c in cells])), r))
    board.sort(key=lambda x: (x[2], x[4]), reverse=True)
    typer.echo("\n\n===== CONSISTENCY LEADERBOARD (beat SPY in the most windows) =====")
    typer.echo(f"{'filer':<26}{'kind':<9}{'beat/scored':>12}{'avgAlpha':>10}   per-window alpha (full/2020+/2022+/2024+)")
    for nm, kind, beat, scored_n, avg_a, r in board[:40]:
        cells = "  ".join(_cell(r[w]) for w, _ in windows)
        typer.echo(f"{nm[:25]:<26}{kind:<9}{beat:>6}/{scored_n:<5}{avg_a:>+9.1%}   {cells}")


@app.command()
def insiders(
    horizon: int = 0,
    benchmark: str = typer.Option("", help="excess-return benchmark ticker; blank = config default (SPY). "
                                           "Use IWM for a size-matched (small-cap) comparison."),
    routine_years: int = typer.Option(3, help="Cohen-Malloy: same-month trading in >= this many years => routine"),
    cluster_window: int = typer.Option(7, help="days: cluster = >= min-insiders buying one ticker within this window"),
    min_insiders: int = typer.Option(3, help="distinct insiders to count as a cluster buy"),
    max_tickers: int = typer.Option(1500, help="cap price fetch to the most-bought insider tickers"),
):
    """Insider Form 4 signal test, the fast (T+2) channel. Do CLUSTER buys
    (several officers, same name, short window) and OPPORTUNISTIC (non-routine)
    insiders beat SPY at the horizon, point-in-time and net of costs? Compares
    each subset to the all-insider-buys baseline with a clustered t-stat."""
    from datetime import date as _date

    from copytrade.backtest.costs import get_cost_model
    from copytrade.backtest.engine import run_event_study
    from copytrade.ingest.market import fetch_daily_closes
    from copytrade.rank.insiders import (
        cluster_buy_ids,
        cluster_ticker_summary,
        routine_filer_ids,
        subset_stats,
        top_quartile_mask,
    )

    conn, settings = _open()
    h = horizon or settings.horizon
    hcol = f"h{h}"
    ins = store.trades_frame(conn, kinds=["insider"], instruments=["stock"])
    if ins.empty:
        typer.echo("no insider data: run `backfill-form345` first (DERA bulk Form 3/4/5)")
        raise typer.Exit(1)
    buys = ins[ins["side"] == "buy"].copy()
    typer.echo(f"=== Insider Form 4 signal test: {len(ins)} insider trades, "
               f"{len(buys)} buys, {buys['filer_name'].nunique()} insiders, horizon {h}d ===")

    bench_sym = benchmark or settings.benchmark
    # bound the price universe to the most-bought insider tickers (+benchmark)
    top_tk = buys["ticker"].value_counts().head(max_tickers).index.tolist()
    prices_m = store.load_price_matrix(conn)
    have = set(prices_m.columns)
    need = [t for t in [*top_tk, bench_sym] if t not in have]
    if need:
        df, prov = fetch_daily_closes(need, "2019-06-01", _date.today().isoformat(),
                                      settings.tiingo_api_key)
        if len(df):
            store.upsert_prices(conn, df)
            prices_m = store.load_price_matrix(conn)
    if bench_sym not in prices_m.columns:
        typer.echo(f"no benchmark prices for {bench_sym}")
        raise typer.Exit(1)
    bench = prices_m[bench_sym].dropna()
    typer.echo(f"(excess returns measured vs {bench_sym})")

    priced = buys[buys["ticker"].isin(prices_m.columns)].copy()
    es = run_event_study(priced, prices_m, bench, get_cost_model(settings.cost_model),
                         [h], anchor="filing", same_day_close=settings.same_day_close,
                         cutoff=settings.same_day_cutoff, amount_basis="mid",
                         trade_notional=settings.trade_notional)
    pt = es.per_trade
    if pt.empty:
        typer.echo("no priceable insider buys at this horizon")
        raise typer.Exit(0)

    cluster_ids = cluster_buy_ids(priced, cluster_window, min_insiders)
    routine_ids = routine_filer_ids(ins, routine_years)     # history over ALL insider trades
    pt["is_cluster"] = pt["trade_id"].isin(cluster_ids)
    pt["is_routine"] = pt["filer_id"].isin(routine_ids)
    big, big_info = top_quartile_mask(pt["amount"])
    if big_info["degenerate"]:
        typer.secho(
            f"NOTE: the 75th percentile sits on a tied amount "
            f"({big_info['threshold']:,.0f}); 'large' is "
            f"{big_info['fraction']:.1%} of trades, not 25%.",
            fg=typer.colors.YELLOW)

    subsets = {
        "ALL insider buys": pt,
        "  cluster buys": pt[pt["is_cluster"]],
        "  non-cluster": pt[~pt["is_cluster"]],
        "  opportunistic (non-routine)": pt[~pt["is_routine"]],
        "  routine": pt[pt["is_routine"]],
        "  large ($ top quartile)": pt[big],
        "  cluster & opportunistic": pt[pt["is_cluster"] & ~pt["is_routine"]],
    }
    typer.echo(f"\n{'subset':<30}{'n':>7}{'meanExc':>9}{'medExc':>8}{'hit%':>7}{'t-stat':>8}")
    for name, sub in subsets.items():
        s = subset_stats(sub, hcol)
        if not s["n"]:
            typer.echo(f"{name:<30}{0:>7}")
            continue
        typer.echo(f"{name:<30}{s['n']:>7}{s['mean_excess']:>+8.2%}{s['median']:>+8.2%}"
                   f"{s['hit_rate']*100:>6.0f}{s['tstat']:>8.2f}")
    typer.echo("\n  (meanExc = mean forward excess return vs SPY at horizon, net of costs; "
               "t-stat clustered by execution date, |t|>2 is the bar)")

    cs = cluster_ticker_summary(pt, cluster_ids, hcol)
    if not cs.empty:
        typer.echo("\n=== Most-clustered names (distinct insiders buying together) ===")
        show = cs.assign(mean_excess=(cs["mean_excess"]*100).round(1),
                         hit_rate=(cs["hit_rate"]*100).round(0))
        typer.echo(show.rename(columns={"mean_excess": "meanExc%", "hit_rate": "hit%"}).to_string(index=False))


@app.command("backfill-13d")
def backfill_13d(years: str = typer.Option("2021,2022,2023,2024,2025", help="comma-separated years")):
    """Ingest SEC Schedule 13D activist stakes (>5%) from EDGAR quarterly
    indexes: the fast, event-driven channel. Filing-anchored buys of the
    target company."""
    from copytrade.ingest.thirteend import refresh_13d

    conn, settings = _open()
    yrs = [int(y) for y in years.split(",") if y.strip()]
    res = asyncio.run(refresh_13d(conn, settings, yrs))
    typer.echo(json.dumps(res, indent=2))


@app.command()
def activism(
    horizons: str = typer.Option("5,21,63", help="forward horizons (days) to test"),
    max_tickers: int = typer.Option(1500, help="cap price fetch to the most-targeted names"),
):
    """13D activist event study: forward EXCESS return of the TARGET after a
    Schedule 13D is filed, at several horizons (pops are fast, so short
    horizons matter). Point-in-time (filing-anchored), net of costs, clustered
    t-stat vs SPY."""
    from datetime import date as _date

    from copytrade.backtest.costs import get_cost_model
    from copytrade.backtest.engine import run_event_study
    from copytrade.ingest.market import fetch_daily_closes
    from copytrade.rank.insiders import subset_stats

    conn, settings = _open()
    df = store.trades_frame(conn, kinds=["fund"], sides=["buy"], instruments=["stock"])
    buys = df[df["source"] == "edgar_13d"].copy() if "source" in df.columns else df.iloc[0:0]
    if buys.empty:
        typer.echo("no 13D data: run `backfill-13d` first")
        raise typer.Exit(1)
    hs = [int(h) for h in horizons.split(",") if h.strip()]
    typer.echo(f"=== 13D activist event study: {len(buys)} target filings, "
               f"{buys['ticker'].nunique()} tickers, excess vs {settings.benchmark}, net of {settings.cost_model} ===")

    top_tk = buys["ticker"].value_counts().head(max_tickers).index.tolist()
    prices_m = store.load_price_matrix(conn)
    need = [t for t in [*top_tk, settings.benchmark] if t not in set(prices_m.columns)]
    if need:
        pdf, _ = fetch_daily_closes(need, "2019-06-01", _date.today().isoformat(), settings.tiingo_api_key)
        if len(pdf):
            store.upsert_prices(conn, pdf)
            prices_m = store.load_price_matrix(conn)
    if settings.benchmark not in prices_m.columns:
        typer.echo("no benchmark prices")
        raise typer.Exit(1)
    bench = prices_m[settings.benchmark].dropna()
    priced = buys[buys["ticker"].isin(prices_m.columns)].copy()
    es = run_event_study(priced, prices_m, bench, get_cost_model(settings.cost_model),
                         hs, anchor="filing", same_day_close=settings.same_day_close,
                         cutoff=settings.same_day_cutoff, trade_notional=settings.trade_notional)
    pt = es.per_trade
    if pt.empty:
        typer.echo("no priceable 13D targets")
        raise typer.Exit(0)
    typer.echo(f"\n{'horizon':<9}{'n':>7}{'meanExc':>9}{'medExc':>8}{'hit%':>7}{'t-stat':>8}")
    for h in hs:
        s = subset_stats(pt, f"h{h}")
        if s["n"]:
            typer.echo(f"{str(h)+'d':<9}{s['n']:>7}{s['mean_excess']:>+8.2%}{s['median']:>+8.2%}"
                       f"{s['hit_rate']*100:>6.0f}{s['tstat']:>8.2f}")
    typer.echo("\n  (excess vs SPY at each horizon, net of costs; clustered t-stat, |t|>2 = real; "
               "13D pops are fast, so the short horizons are the fair test)")


@app.command()
def factors():
    """Factor attribution: is the congressional basket's mild edge real alpha
    or just a SIZE/breadth tilt? Regresses the copy-ALL basket's monthly
    returns on Fama-French factors (Mkt-RF, SMB, HML, optionally Mom) and on the
    equal-weight RSP and SPY benchmarks. A positive significant SMB beta with
    an insignificant alpha = 'it's the size factor, not skill'."""
    from copytrade.backtest import benchmarks as bm
    from copytrade.backtest.costs import get_cost_model
    from copytrade.backtest.engine import walk_forward
    from copytrade.ingest.famafrench import fetch_ff_factors
    from copytrade.rank.factors import factor_regression, single_benchmark

    conn, settings = _open()
    cost = get_cost_model(settings.cost_model)
    trades = store.trades_frame(conn, instruments=["stock"])
    prices_m = store.load_price_matrix(conn)
    if trades.empty or settings.benchmark not in prices_m.columns:
        typer.echo("no trades or benchmark prices, run `prices` first")
        raise typer.Exit(1)
    bench = prices_m[settings.benchmark].dropna()
    wf = walk_forward(trades, prices_m, bench, cost, copy_all=True, min_trades=1,
                      horizon=settings.horizon, max_weight=settings.max_weight,
                      trade_notional=settings.trade_notional,
                      same_day_close=settings.same_day_close, cutoff=settings.same_day_cutoff)
    if wf is None or wf.equity.empty:
        typer.echo("copy-all basket produced no equity curve")
        raise typer.Exit(1)
    basket_m = wf.equity.resample("ME").last().pct_change().dropna()
    typer.echo(f"=== Factor attribution of the copy-ALL basket "
               f"({wf.equity.index[0].date()} → {wf.equity.index[-1].date()}, "
               f"{len(basket_m)} months) ===")

    try:
        ff = fetch_ff_factors(getattr(settings, "edgar_user_agent", "copytrade-research"))
    except Exception as e:  # noqa: BLE001
        ff = None
        typer.secho(f"(Fama-French download failed: {str(e)[:120]})", fg=typer.colors.YELLOW)
    if ff is not None and not ff.empty:
        factors = ("Mkt-RF", "SMB", "HML", "Mom") if "Mom" in ff.columns else ("Mkt-RF", "SMB", "HML")
        reg = factor_regression(basket_m, ff, factors=factors)
        if reg:
            typer.echo(f"\nFama-French regression (basket excess ~ factors), "
                       f"n={reg['n_months']} months, R²={reg['r2']:.2f}")
            typer.echo(f"{'term':<10}{'coef':>10}{'t':>8}{'sig':>6}")
            for term in ["alpha", *factors]:
                if term in reg:
                    d = reg[term]
                    sig = "***" if d["p"] < 0.01 else "**" if d["p"] < 0.05 else "*" if d["p"] < 0.1 else ""
                    coef = d["coef"] * (100 if term != "alpha" else 100)
                    typer.echo(f"{term:<10}{coef:>+9.2f}%{d['t']:>8.2f}{sig:>6}")
            typer.echo(f"annualized alpha: {reg['alpha_annualized']:+.1%}  "
                       f"(alpha t={reg['alpha']['t']:+.2f})")
            smb = reg.get("SMB", {})
            if smb:
                verdict = ("SIZE factor explains it (SMB+ significant, alpha not)"
                           if smb["coef"] > 0 and smb["p"] < 0.05
                           and reg["alpha"]["p"] >= 0.05
                           else "SMB not the whole story")
                typer.echo(f"read: {verdict}")

    typer.echo("\nSingle-benchmark regressions (basket monthly ~ benchmark):")
    typer.echo(f"{'benchmark':<18}{'beta':>7}{'ann.alpha':>11}{'alpha t':>9}{'R²':>7}")
    for label in [settings.benchmark, "RSP", "IWM"]:
        if label in prices_m.columns:
            b_m = prices_m[label].dropna().resample("ME").last().pct_change().dropna()
            sb = single_benchmark(basket_m, b_m)
            if sb:
                typer.echo(f"{label:<18}{sb['beta']:>7.2f}{sb['alpha_annualized']:>+10.1%}"
                           f"{sb['alpha_t']:>9.2f}{sb['r2']:>7.2f}")
    typer.echo("\n  (high R² + ~zero, insignificant alpha => that benchmark explains the basket; "
               "RSP = equal-weight S&P, the breadth proxy)")


@app.command()
def netedge(
    horizons: str = typer.Option("21,63,126", help="consensus-CAR horizons (days)"),
    cluster_window: int = typer.Option(45, help="days: consensus = >= min-filers same ticker in window"),
    min_filers: int = typer.Option(3, help="distinct filers for a consensus event"),
):
    """The FAIR test of a tradeable edge (not 'beat SPY'): (1) a standalone
    absolute/risk scorecard, (2) the BETA-NEUTRAL long/short alpha, long the
    basket, short beta x SPY, so market direction is hedged and only the signal
    remains: with its Sharpe and t-stat, and (3) consensus CAR at 1-6 month
    horizons. All point-in-time, net of Canadian costs."""
    import numpy as np

    from copytrade.backtest import benchmarks as bm
    from copytrade.backtest import metrics
    from copytrade.backtest.costs import get_cost_model
    from copytrade.backtest.engine import run_event_study, walk_forward
    from copytrade.rank.factors import long_short_returns
    from copytrade.rank.insiders import cluster_buy_ids, subset_stats

    conn, settings = _open()
    cost = get_cost_model(settings.cost_model)
    trades = store.trades_frame(conn, instruments=["stock"])
    prices_m = store.load_price_matrix(conn)
    if trades.empty or settings.benchmark not in prices_m.columns:
        typer.echo("no trades or benchmark prices, run `prices` first")
        raise typer.Exit(1)
    bench = prices_m[settings.benchmark].dropna()
    common = dict(horizon=settings.horizon, max_weight=settings.max_weight,
                  trade_notional=settings.trade_notional,
                  same_day_close=settings.same_day_close, cutoff=settings.same_day_cutoff)

    wf_all = walk_forward(trades, prices_m, bench, cost, copy_all=True, min_trades=1, **common)
    wf_top = walk_forward(trades, prices_m, bench, cost, top_n=settings.top_n,
                          min_trades=settings.min_trades_walkforward, **common)
    if wf_all is None or wf_all.equity.empty:
        typer.echo("copy-all produced no equity curve")
        raise typer.Exit(1)
    span = wf_all.equity.index

    # ---- 1. absolute scorecard (no beat-SPY gate) ----
    curves = {"Copy ALL (congress index)": wf_all.equity}
    if wf_top is not None and not wf_top.equity.empty:
        curves[f"Copy top-{settings.top_n}"] = wf_top.equity
    curves[f"{settings.benchmark} buy & hold"] = bm.buy_and_hold(bench.loc[span[0]:span[-1]], cost)
    if "TLT" in prices_m.columns and prices_m["TLT"].notna().sum() > 50:
        curves["60/40 SPY/TLT"] = bm.sixty_forty(bench.loc[span[0]:span[-1]],
                                                 prices_m["TLT"], cost)
    typer.echo("=== 1. Absolute scorecard (standalone risk/return, no beat-SPY gate) ===")
    typer.echo(f"{'strategy':<28}{'total':>8}{'CAGR':>7}{'Shrp':>6}{'Sort':>6}{'Clmr':>6}{'maxDD':>7}")
    for name, eq in curves.items():
        s = metrics.perf_stats(eq.dropna())
        if s:
            typer.echo(f"{name:<28}{s['total_return']:>+7.0%}{s['cagr']:>+6.1%}{s['sharpe']:>6.2f}"
                       f"{(s['sortino'] if pd.notna(s['sortino']) else 0):>6.2f}"
                       f"{(s['calmar'] if pd.notna(s['calmar']) else 0):>6.2f}{s['max_drawdown']:>7.1%}")

    # ---- 2. beta-neutral long/short alpha ----
    basket_m = wf_all.equity.resample("ME").last().pct_change().dropna()
    spy_m = bench.loc[span[0]:span[-1]].resample("ME").last().pct_change().dropna()
    typer.echo("\n=== 2. Beta-neutral long/short (long basket, short beta x SPY): pure alpha ===")
    typer.echo(f"{'hedge':<20}{'beta':>6}{'ann.alpha':>11}{'Sharpe':>8}{'t-stat':>8}")
    for label, beta in (("OLS beta-neutral", None), ("dollar-neutral (β=1)", 1.0)):
        ls, b = long_short_returns(basket_m, spy_m, beta=beta)
        if ls.empty:
            continue
        n = len(ls)
        ann = float((1 + ls.mean()) ** 12 - 1)
        sharpe = float(ls.mean() / ls.std() * np.sqrt(12)) if ls.std() > 0 else float("nan")
        tstat = float(ls.mean() / (ls.std() / np.sqrt(n))) if ls.std() > 0 else float("nan")
        typer.echo(f"{label:<20}{b:>6.2f}{ann:>+10.1%}{sharpe:>8.2f}{tstat:>8.2f}")
    typer.echo("  (t-stat on the monthly long/short mean; |t|>2 = real alpha independent of market beta)")

    # ---- 3. consensus CAR at short/medium horizons ----
    hs = [int(h) for h in horizons.split(",") if h.strip()]
    buys = trades[(trades["side"] == "buy") & (trades["filer_kind"] == "congress")]
    cluster_ids = cluster_buy_ids(buys, cluster_window, min_filers)
    consensus = buys[buys["trade_id"].isin(cluster_ids)]
    typer.echo(f"\n=== 3. Consensus CAR (>= {min_filers} members, same name within {cluster_window}d) ===")
    if consensus.empty:
        typer.echo("(no consensus events)")
    else:
        es = run_event_study(consensus, prices_m, bench, cost, hs, anchor="filing",
                             same_day_close=settings.same_day_close, cutoff=settings.same_day_cutoff,
                             trade_notional=settings.trade_notional)
        pt = es.per_trade
        typer.echo(f"{len(consensus)} consensus buys, {consensus['ticker'].nunique()} tickers")
        typer.echo(f"{'horizon':<9}{'n':>7}{'CAR':>9}{'medCAR':>8}{'hit%':>7}{'t-stat':>8}")
        for h in hs:
            s = subset_stats(pt, f"h{h}")
            if s["n"]:
                typer.echo(f"{str(h)+'d':<9}{s['n']:>7}{s['mean_excess']:>+8.2%}{s['median']:>+8.2%}"
                           f"{s['hit_rate']*100:>6.0f}{s['tstat']:>8.2f}")
    typer.echo("  (CAR = cumulative abnormal return vs SPY, net of costs; short horizons are the fair test)")


@app.command()
def greeks(
    ma_window: int = typer.Option(200, help="regime filter: hold basket only while SPY > this-day SMA"),
    risk_off: str = typer.Option("cash", help="where the regime filter parks when risk-off: 'cash' or 'tlt'"),
    switch_cost_bps: float = typer.Option(10.0, help="round-trip cost per regime flip, basis points"),
    regime_etfs: str = typer.Option("QQQ,SPY", help="also show regime-filtered versions of these ETFs "
                                                    "(the filter is a risk rule, not a copy rule)"),
    cad: bool = typer.Option(False, "--cad", help="report in CAD, adds the currency translation "
                                                  "effect a Canadian actually realises"),
):
    """The FULL metrics battery (not just alpha/beta) for the copy-ALL basket, a
    REGIME-FILTERED version of it, and the popular ETFs a real investor would
    actually choose (QQQ growth, VTI total-market, IWM small-cap, SCHD dividend,
    SPY). For each: return, vol, downside dev, Sharpe/Sortino/Calmar, max-DD,
    skew/kurtosis, daily VaR/CVaR-95, Omega, tail ratio, %-up days, and the
    market-relative 'greeks' vs SPY, beta (the equity delta), Jensen alpha,
    R2, Treynor, tracking error, info ratio, up/down capture. Net of Canadian
    costs, point-in-time."""
    from copytrade.backtest import benchmarks as bm
    from copytrade.backtest import metrics
    from copytrade.backtest.costs import get_cost_model
    from copytrade.backtest.engine import walk_forward
    from copytrade.backtest.regime import regime_equity

    conn, settings = _open()
    cost = get_cost_model(settings.cost_model)
    trades = store.trades_frame(conn, instruments=["stock"])
    prices_m = store.load_price_matrix(conn)
    if trades.empty or settings.benchmark not in prices_m.columns:
        typer.echo("no trades or benchmark prices, run `prices` first")
        raise typer.Exit(1)
    bench = prices_m[settings.benchmark].dropna()
    common = dict(horizon=settings.horizon, max_weight=settings.max_weight,
                  trade_notional=settings.trade_notional,
                  same_day_close=settings.same_day_close, cutoff=settings.same_day_cutoff)
    wf_all = walk_forward(trades, prices_m, bench, cost, copy_all=True, min_trades=1, **common)
    if wf_all is None or wf_all.equity.empty:
        typer.echo("copy-all produced no equity curve")
        raise typer.Exit(1)
    basket = wf_all.equity.dropna()
    span0, span1 = basket.index[0], basket.index[-1]

    # SPY daily returns over the span = the benchmark for every relative metric
    bench_rets = bench.loc[span0:span1].pct_change().dropna()

    # regime-filtered basket: SPY (with pre-span history for the MA) is the gauge
    risk_off_ret = None
    if risk_off.lower() == "tlt" and "TLT" in prices_m.columns:
        risk_off_ret = prices_m["TLT"].dropna().pct_change().dropna()
    regime_eq, reg_info = regime_equity(basket, bench, ma_window=ma_window,
                                        risk_off_ret=risk_off_ret,
                                        switch_cost=switch_cost_bps / 1e4)

    curves: dict[str, pd.Series] = {"Copy ALL": basket}
    off_label = "TLT" if risk_off_ret is not None else "cash"
    curves[f"Regime>{ma_window}d→{off_label}"] = regime_eq
    etfs = bm.etf_buy_hold(prices_m, cost, start=span0, end=span1)
    for t, eq in etfs.items():
        curves[t] = eq

    # The regime filter is a RISK rule, not a copy-signal rule, it applies to
    # any holding. Since the ETFs beat the copy basket outright, the honest
    # comparison has to include regime-filtered ETFs too, otherwise the basket
    # is being credited with a drawdown improvement anything could have had.
    reg_etf_info: dict[str, dict] = {}
    for t in [s.strip().upper() for s in regime_etfs.split(",") if s.strip()]:
        if t not in etfs:
            continue
        eq, info = regime_equity(etfs[t], bench, ma_window=ma_window,
                                 risk_off_ret=risk_off_ret,
                                 switch_cost=switch_cost_bps / 1e4)
        if not eq.empty:
            curves[f"{t}|regime"] = eq
            reg_etf_info[t] = info

    # CAD reporting: holding a USD asset is also a long-USD/short-CAD position.
    # Translate every curve (and the benchmark used for relative metrics) so the
    # comparison stays internally consistent in the currency the investor spends.
    fx_note = ""
    if cad:
        from copytrade.backtest.currency import load_usdcad, to_cad

        usdcad = load_usdcad(conn)
        if usdcad.empty:
            typer.secho("no USD/CAD history, run `fx-refresh` first; "
                        "reporting in USD.", fg=typer.colors.YELLOW)
        else:
            converted, infos = {}, {}
            for name, eq in curves.items():
                c, info = to_cad(eq, usdcad)
                if not c.empty:
                    converted[name] = c
                    infos[name] = info
            if converted:
                curves = converted
                bench_cad, binfo = to_cad(bench.loc[span0:span1], usdcad)
                bench_rets = bench_cad.pct_change().dropna()
                ref = next(iter(infos.values()))
                # The table now covers a SHORTER window than the USD run, so the
                # header span must be restated, otherwise the truncation gets
                # read as a currency effect. Measured: SPY CAGR 13.79% (USD from
                # 2015-01-05) vs 15.07% (USD from 2017-01-03) vs 15.64% (CAD from
                # 2017-01-03): i.e. most of the apparent "CAD uplift" was the
                # shorter window, not FX.
                span0, span1 = ref["cad_span"]
                bench_usd_same = bench.loc[span0:span1]
                usd_same_cagr = None
                if len(bench_usd_same) > 2:
                    yrs = (span1 - span0).days / 365.25
                    if yrs > 0:
                        usd_same_cagr = (float(bench_usd_same.iloc[-1]
                                               / bench_usd_same.iloc[0])
                                         ** (1 / yrs) - 1)
                fx_note = (
                    f"\nCAD reporting ON. Covered window {span0.date()} → "
                    f"{span1.date()} ({ref['dropped_days']} days dropped: "
                    f"{ref['dropped_days'] - ref['dropped_after_last_fx']} before the "
                    f"first BoC observation, {ref['dropped_after_last_fx']} after the "
                    f"last ({pd.Timestamp(ref['fx_last_observation']).date()}), rates "
                    f"are never back-filled or carried indefinitely forward)."
                    f"\nUSD/CAD moved {binfo['fx_return']:+.1%} over that window."
                )
                if usd_same_cagr is not None:
                    fx_note += (
                        f"\nLIKE-FOR-LIKE: {settings.benchmark} CAGR over the SAME "
                        f"window is {usd_same_cagr:+.2%} in USD. Compare against the "
                        f"CAD row above to isolate the currency effect, comparing "
                        f"to a full-span USD run would credit FX with the "
                        f"truncated window."
                    )

    stats = {name: metrics.full_metrics(eq, bench_rets=bench_rets) for name, eq in curves.items()}
    stats = {k: v for k, v in stats.items() if v}
    names = list(stats)

    rows = [
        ("total_return", "Total return", "{:+.0%}"),
        ("cagr", "CAGR", "{:+.1%}"),
        ("vol", "Volatility (ann)", "{:.1%}"),
        ("downside_dev", "Downside dev", "{:.1%}"),
        ("max_drawdown", "Max drawdown", "{:.1%}"),
        ("sharpe", "Sharpe", "{:.2f}"),
        ("sortino", "Sortino", "{:.2f}"),
        ("calmar", "Calmar", "{:.2f}"),
        ("omega", "Omega", "{:.2f}"),
        ("skew", "Skew (daily)", "{:+.2f}"),
        ("kurtosis", "Excess kurtosis", "{:.1f}"),
        ("var95", "VaR 95% (1d)", "{:.2%}"),
        ("cvar95", "CVaR 95% (1d)", "{:.2%}"),
        ("tail_ratio", "Tail ratio", "{:.2f}"),
        ("pct_positive_days", "% up days", "{:.0%}"),
        ("best_day", "Best day", "{:+.1%}"),
        ("worst_day", "Worst day", "{:+.1%}"),
        ("beta", "Beta (≈delta) vs SPY", "{:.2f}"),
        ("alpha_annualized", "Jensen alpha (ann)", "{:+.1%}"),
        ("corr", "Corr vs SPY", "{:.2f}"),
        ("r2", "R² vs SPY", "{:.2f}"),
        ("treynor", "Treynor", "{:.2f}"),
        ("tracking_error", "Tracking error", "{:.1%}"),
        ("info_ratio", "Information ratio", "{:.2f}"),
        ("up_capture", "Up capture", "{:.0%}"),
        ("down_capture", "Down capture", "{:.0%}"),
    ]
    w = 12
    ccy = "CAD" if (cad and fx_note) else "USD"
    typer.echo(f"=== Full metrics battery ({span0.date()} → {span1.date()}, "
               f"net of {settings.cost_model}, {ccy}) ===")
    header = f"{'metric':<22}" + "".join(f"{n[:w]:>{w}}" for n in names)
    typer.echo(header)
    typer.echo("-" * len(header))
    for key, label, fmt in rows:
        cells = ""
        for n in names:
            v = stats[n].get(key)
            cells += f"{(fmt.format(v) if v is not None and pd.notna(v) else ', '):>{w}}"
        typer.echo(f"{label:<22}{cells}")
    typer.echo(f"\nRegime filter: {reg_info.get('switches','?')} switches, "
               f"{reg_info.get('time_in_market',0):.0%} of days in-market "
               f"(SPY {ma_window}d SMA rule, {switch_cost_bps:.0f}bps/flip, parks in {off_label}).")
    if reg_etf_info:
        detail = ", ".join(f"{t} {i.get('switches','?')} switches"
                           for t, i in reg_etf_info.items())
        typer.echo(f"Regime-filtered ETFs use the SAME rule ({detail}), compare "
                   f"'X|regime' against 'Copy ALL' + regime to see whether the copy "
                   f"signal adds anything beyond the risk rule itself.")
    if fx_note:
        typer.echo(fx_note)
    typer.echo("Note: 'greeks' here are the equity-portfolio analogues, beta is the market delta; "
               "skew/kurtosis capture tail convexity. True option greeks (gamma/theta/vega) apply "
               "only to the options positions, covered in `derivatives`.")


@app.command()
def conviction(
    horizon: int = 0,
    cluster_window: int = typer.Option(30, help="days: cluster = >= min-members buying one ticker within this window"),
    min_members: int = typer.Option(3, help="distinct members to count as a cluster buy"),
    abnormal_mult: float = typer.Option(2.0, help="buy >= this x the member's own median = 'abnormal' conviction"),
):
    """Congress CONVICTION signals: does filtering to high-conviction buys beat
    the copy-everyone baseline? Compares ALL congress buys vs cluster (>= N
    members piling into one ticker in a window), large ($ top quartile),
    abnormal (a buy far above that member's own median), and cluster+large
    each point-in-time, cost-net, with a clustered t-stat vs SPY."""
    from copytrade.backtest.costs import get_cost_model
    from copytrade.backtest.engine import run_event_study
    from copytrade.rank.insiders import (
        abnormal_size_ids_pit,
        cluster_buy_ids,
        cluster_ticker_summary,
        subset_stats,
        top_quartile_mask,
    )

    conn, settings = _open()
    h = horizon or settings.horizon
    hcol = f"h{h}"
    buys = store.trades_frame(conn, kinds=["congress"], sides=["buy"], instruments=["stock"])
    prices_m = store.load_price_matrix(conn)
    if buys.empty or settings.benchmark not in prices_m.columns:
        typer.echo("no congress buys or no benchmark prices, run `prices` first")
        raise typer.Exit(1)
    bench = prices_m[settings.benchmark].dropna()
    es = run_event_study(buys, prices_m, bench, get_cost_model(settings.cost_model),
                         [h], anchor="filing", same_day_close=settings.same_day_close,
                         cutoff=settings.same_day_cutoff, amount_basis=settings.amount_basis,
                         trade_notional=settings.trade_notional)
    pt = es.per_trade
    if pt.empty:
        typer.echo("no priceable congress buys")
        raise typer.Exit(0)
    typer.echo(f"=== Congress conviction signals: {len(buys)} buys, {len(pt)} priceable, "
               f"horizon {h}d, excess vs {settings.benchmark}, net of {settings.cost_model} ===")

    cluster_ids = cluster_buy_ids(buys, cluster_window, min_members)
    # point-in-time: the whole-sample median is look-ahead in a return test
    abnormal_ids = abnormal_size_ids_pit(buys, abnormal_mult)
    pt["is_cluster"] = pt["trade_id"].isin(cluster_ids)
    pt["is_abnormal"] = pt["trade_id"].isin(abnormal_ids)
    big, big_info = top_quartile_mask(pt["amount"])
    if big_info["degenerate"]:
        typer.secho(
            f"NOTE: amounts are disclosed RANGES; the 75th percentile sits on a "
            f"tied bucket ({big_info['threshold']:,.0f}), so 'large' is "
            f"{big_info['fraction']:.1%} of trades, not 25%.",
            fg=typer.colors.YELLOW)

    subsets = {
        "ALL congress buys": pt,
        f"  cluster (>={min_members} members/{cluster_window}d)": pt[pt["is_cluster"]],
        "  large ($ top quartile)": pt[big],
        f"  abnormal (>={abnormal_mult}x own median)": pt[pt["is_abnormal"]],
        "  cluster & large": pt[pt["is_cluster"] & big],
    }
    typer.echo(f"\n{'subset':<34}{'n':>7}{'meanExc':>9}{'medExc':>8}{'hit%':>7}{'t-stat':>8}")
    for name, sub in subsets.items():
        s = subset_stats(sub, hcol)
        if not s["n"]:
            typer.echo(f"{name:<34}{0:>7}")
            continue
        typer.echo(f"{name:<34}{s['n']:>7}{s['mean_excess']:>+8.2%}{s['median']:>+8.2%}"
                   f"{s['hit_rate']*100:>6.0f}{s['tstat']:>8.2f}")
    typer.echo("\n  (meanExc = forward excess vs benchmark at horizon, net of costs; "
               "clustered t-stat: |t|>2 is the bar for a real signal)")

    cs = cluster_ticker_summary(pt, cluster_ids, hcol)
    if not cs.empty:
        typer.echo("\n=== Most-clustered congress tickers (members piling in together) ===")
        show = cs.assign(mean_excess=(cs["mean_excess"]*100).round(1),
                         hit_rate=(cs["hit_rate"]*100).round(0))
        typer.echo(show.rename(columns={"mean_excess": "meanExc%", "hit_rate": "hit%"}).to_string(index=False))


@app.command()
def indicators(
    horizon: int = 0,
    control_per_date: int = typer.Option(3, help="control stocks sampled per execution date"),
    seed: int = typer.Option(0, help="control-sample seed"),
    top: int = typer.Option(3, help="show quintile detail for the N strongest features"),
):
    """WHAT characterised the trades that worked, disclosures as a LENS.

    Every other analysis asks "does mirroring these people pay?" (no: gross
    excess return is ~0 at every horizon, and costs make it negative). This
    asks a different question: that null is about the MEAN, and a zero mean is
    perfectly consistent with large, predictable dispersion around it.

    So: conditional on a disclosure happening, do observable characteristics
    momentum, volatility, drawdown, market regime, trade size, disclosure lag,
    cluster intensity: predict which ones worked?

    THE CONTROL DECIDES WHETHER THIS MEANS ANYTHING. If momentum predicts
    returns among congress-bought stocks, that is not a fact about Congress;
    momentum predicts returns among all stocks. Every feature is therefore
    tested twice: on the disclosed trades, and on random tickers drawn on the
    SAME dates from the same priced universe. Only the DIFFERENCE can be
    attributed to the disclosure. A feature that works equally well in both is
    a generic factor you could trade directly, without waiting 45 days."""
    from copytrade.backtest.costs import get_cost_model
    from copytrade.backtest.engine import run_event_study
    from copytrade.rank.features import (
        CONTEXT_FEATURES,
        PRICE_FEATURES,
        TRADE_FEATURES,
        build_control,
        build_disclosed,
        compare_features,
        quintile_spread,
    )

    conn, settings = _open()
    h = horizon or settings.horizon
    cost = get_cost_model(settings.cost_model)
    buys = store.trades_frame(conn, kinds=["congress"], sides=["buy"], instruments=["stock"])
    prices_m = store.load_price_matrix(conn)
    if buys.empty or settings.benchmark not in prices_m.columns:
        typer.echo("no congress buys or no benchmark prices, run `prices` first")
        raise typer.Exit(1)
    bench = prices_m[settings.benchmark].dropna()
    es = run_event_study(buys, prices_m, bench, cost, [h], anchor="filing",
                         same_day_close=settings.same_day_close,
                         cutoff=settings.same_day_cutoff,
                         amount_basis=settings.amount_basis,
                         trade_notional=settings.trade_notional)
    if es.per_trade.empty:
        typer.echo("no priceable congress buys")
        raise typer.Exit(0)

    # chamber from filers.meta, not from the ingest source name
    chamber_by_filer: dict = {}
    for row in conn.execute("SELECT filer_id, meta FROM filers WHERE kind='congress'"):
        try:
            meta = json.loads(row["meta"]) if row["meta"] else {}
        except (TypeError, ValueError):
            meta = {}
        if meta.get("chamber"):
            chamber_by_filer[row["filer_id"]] = meta["chamber"]

    typer.echo(f"building point-in-time features for {len(es.per_trade)} disclosed buys...")
    disc = build_disclosed(es.per_trade, prices_m, bench, h, trades=buys,
                           chamber_by_filer=chamber_by_filer)
    typer.echo(f"building matched control ({control_per_date}/date, same dates, "
               f"same universe)...")
    # pass the COST MODEL, so the control pays the same price-dependent round
    # trip the disclosed arm pays rather than a flat sample mean
    ctrl = build_control(disc, prices_m, bench, h, cost,
                         per_date=control_per_date, seed=seed,
                         trade_notional=settings.trade_notional)

    feats = PRICE_FEATURES + CONTEXT_FEATURES + TRADE_FEATURES
    cmp = compare_features(disc, ctrl, feats, horizon=h)
    typer.echo(f"\n=== Indicators: {len(disc)} disclosed vs {len(ctrl)} control "
               f"observations, {h}d net excess vs {settings.benchmark} ===")
    typer.echo("IC = monthly Spearman rank correlation between the feature and the")
    typer.echo("forward return. ic_diff = disclosed - control; FDR is applied to it.\n")
    def _row(r):
        return (f"{r['feature']:<16}{r['n']:>7}{r['months']:>5}"
                f"{_fmt(r['ic_disclosed'], '{:+.3f}'):>9}"
                f"{_fmt(r['t_disclosed'], '{:.2f}'):>8}"
                f"{_fmt(r['ic_control'], '{:+.3f}'):>9}"
                f"{_fmt(r['ic_diff'], '{:+.3f}'):>9}"
                f"{_fmt(r['t_diff'], '{:.2f}'):>8}"
                f"{_fmt(r['q_fdr'], '{:.3f}'):>9}")

    header = (f"{'feature':<16}{'n':>7}{'mo':>5}{'IC_disc':>9}{'t_disc':>8}"
              f"{'IC_ctrl':>9}{'IC_diff':>9}{'t_diff':>8}{'q(FDR)':>9}  flag")

    ctrlable = cmp[cmp["controllable"]]
    only = cmp[~cmp["controllable"]]

    typer.echo("A. CONTROLLABLE features, a random stock has these too, so the")
    typer.echo("   control isolates what the disclosure adds.")
    typer.echo(header)
    for _, r in ctrlable.iterrows():
        flag = "** DISCLOSURE-SPECIFIC" if r["disclosure_specific"] else ""
        if not flag and pd.notna(r["t_disclosed"]) and abs(r["t_disclosed"]) > 2:
            flag = "predictive, but NOT vs control (generic factor)"
        typer.echo(f"{_row(r)}  {flag}")

    if not only.empty:
        typer.echo("\nB. DISCLOSURE-ONLY features, no control is DEFINABLE (a random")
        typer.echo("   stock has no filing lag, trade size or filer). These can never")
        typer.echo("   be 'generic factors'; they are also never proof of an edge,")
        typer.echo("   because there is nothing to compare them against.")
        typer.echo(header)
        for _, r in only.iterrows():
            flag = ("predictive within disclosures (uncontrolled)"
                    if pd.notna(r["t_disclosed"]) and abs(r["t_disclosed"]) > 2 else "")
            typer.echo(f"{_row(r)}  {flag}")

    winners = cmp[cmp["disclosure_specific"]]
    typer.echo("")
    if winners.empty:
        typer.secho(
            f"No CONTROLLABLE feature is disclosure-specific after FDR "
            f"({len(ctrlable)} tested). For those, whatever predicts returns "
            f"here predicts them just as well in random stocks on the same "
            f"dates: a generic factor, not information carried by the filing."
            + (f" The {len(only)} disclosure-only feature(s) in section B were "
               f"NOT tested against a control and this verdict says nothing "
               f"about them." if not only.empty else ""),
            fg=typer.colors.YELLOW)
    else:
        typer.secho(f"{len(winners)} feature(s) predict returns among disclosed "
                    f"trades SPECIFICALLY:", fg=typer.colors.GREEN)
        typer.echo(winners[["feature", "ic_disclosed", "ic_control",
                            "ic_diff", "t_diff", "q_fdr"]].to_string(index=False))

    ranked = cmp.dropna(subset=["ic_disclosed"]).head(top)
    for _, r in ranked.iterrows():
        g = quintile_spread(disc, r["feature"])
        if g.empty:
            continue
        typer.echo(f"\n--- {r['feature']}: mean {h}d net excess by quintile "
                   f"(disclosed trades) ---")
        show = g.assign(mean=(g["mean"] * 100).round(2),
                        median=(g["median"] * 100).round(2),
                        hit=(g["hit"] * 100).round(0))
        typer.echo(show.rename(columns={"mean": "mean%", "median": "med%",
                                        "hit": "hit%"}).to_string())
        t_txt = _fmt(g.attrs["t"], "{:.2f}", dash="undefined")
        typer.echo(f"  top-minus-bottom spread: {g.attrs['spread']:+.2%} "
                   f"(paired-month t={t_txt}, {g.attrs['paired_months']} months "
                   f"with both buckets): check MONOTONICITY; a spread driven by "
                   f"one extreme bucket is usually an outlier")
        if g.attrs["paired_months"] == 0:
            typer.echo("    (market-wide feature: whole months land in one bucket, "
                       "so a within-month spread test is not defined)")


@app.command()
def leverage(
    ma_window: int = typer.Option(200, help="regime filter MA window for the filtered variants"),
    financing: float = typer.Option(None, help="annual margin rate; defaults to leverage.DEFAULT_FINANCING"),
    maintenance: float = typer.Option(None, help="maintenance margin; defaults to leverage.DEFAULT_MAINTENANCE (IBKR-CA/CIRO 30%)"),
    sensitivity: bool = typer.Option(True, help="also scan maintenance 25/30/35/50%, the answer moves a lot"),
    paths: int = typer.Option(2000, help="bootstrap paths"),
    tolerance: float = typer.Option(0.05, help="acceptable probability of a margin call"),
):
    """HOW HARD CAN I PUSH THIS? Leverage + margin-call survivability.

    The copy basket has no significant alpha, so leverage cannot manufacture
    edge: 2x a zero-alpha book is 2x the beta and 2x the drawdown, minus
    financing. What leverage CAN do is turn a low-drawdown asset into a
    higher-return one at the same risk of ruin, which is precisely why the
    regime filter's drawdown reduction matters more than its CAGR cost.

    Reports the realised path AND a circular block bootstrap, because "2x
    would have survived 2015-2026" is one draw from the distribution and is
    contaminated by knowing this decade ended well. The number that should
    drive sizing is the bootstrapped probability of a margin call."""
    from copytrade.backtest import benchmarks as bm
    from copytrade.backtest.costs import get_cost_model
    from copytrade.backtest.engine import walk_forward
    from copytrade.backtest.leverage import (
        DEFAULT_FINANCING,
        DEFAULT_MAINTENANCE,
        MAINTENANCE_GRID,
        leverage_scan,
        max_safe_leverage,
    )
    from copytrade.backtest.regime import regime_equity

    # resolve from the module constants rather than re-hardcoding in the CLI
    financing = DEFAULT_FINANCING if financing is None else financing
    maintenance = DEFAULT_MAINTENANCE if maintenance is None else maintenance

    conn, settings = _open()
    cost = get_cost_model(settings.cost_model)
    trades = store.trades_frame(conn, instruments=["stock"])
    prices_m = store.load_price_matrix(conn)
    if trades.empty or settings.benchmark not in prices_m.columns:
        typer.echo("no trades or benchmark prices, run `prices` first")
        raise typer.Exit(1)
    bench = prices_m[settings.benchmark].dropna()
    wf = walk_forward(trades, prices_m, bench, cost, copy_all=True, min_trades=1,
                      horizon=settings.horizon, max_weight=settings.max_weight,
                      trade_notional=settings.trade_notional,
                      same_day_close=settings.same_day_close,
                      cutoff=settings.same_day_cutoff)
    if wf is None or wf.equity.empty:
        typer.echo("copy-all produced no equity curve")
        raise typer.Exit(1)
    basket = wf.equity.dropna()
    span0, span1 = basket.index[0], basket.index[-1]

    candidates: dict[str, pd.Series] = {"Copy ALL": basket}
    reg, _ = regime_equity(basket, bench, ma_window=ma_window)
    if not reg.empty:
        candidates[f"Copy ALL + regime{ma_window}"] = reg
    for t, eq in bm.etf_buy_hold(prices_m, cost, start=span0, end=span1).items():
        if t in ("SPY", "QQQ"):
            candidates[t] = eq
            r, _ = regime_equity(eq, bench, ma_window=ma_window)
            if not r.empty:
                candidates[f"{t} + regime{ma_window}"] = r

    typer.echo(f"=== Leverage & margin-call survivability ({span0.date()} → "
               f"{span1.date()}, financing {financing:.1%}, maintenance "
               f"{maintenance:.0%}, {paths} bootstrap paths) ===")
    for name, eq in candidates.items():
        scan = leverage_scan(eq.pct_change().dropna(),
                             financing_rate=financing, maintenance=maintenance,
                             n_paths=paths)
        if scan.empty:
            continue
        safe = max_safe_leverage(scan, tolerance)
        typer.echo(f"\n--- {name} ---")
        typer.echo(f"{'L':>5}{'realised ret':>14}{'realised DD':>13}"
                   f"{'boot med ret':>14}{'boot p05':>10}{'med DD':>9}"
                   f"{'P(call)':>9}{'LIQ':>5}")
        for _, r in scan.iterrows():
            # LIQ marks a realised path that was FORCE-CLOSED. Without it a
            # wiped-out account is indistinguishable from an ordinary loss
            # and because equity freezes at liquidation, its realised drawdown
            # reads SHALLOWER than a surviving path's, exactly where the user
            # is being told to size off that number.
            liq = "YES" if bool(r["realised_liquidated"]) else ""
            typer.echo(
                f"{r['leverage']:>5.2f}{r['realised_total_return']:>+13.0%}"
                f"{r['realised_max_dd']:>+13.1%}{r['boot_median_return']:>+13.0%}"
                f"{r['boot_p05_return']:>+10.0%}{r['boot_median_max_dd']:>+9.1%}"
                f"{r['margin_call_prob']:>9.1%}{liq:>5}"
            )
        if scan["realised_liquidated"].any():
            typer.secho("  LIQ = the realised path was force-closed; it stopped "
                        "participating from that day, so its 'realised DD' "
                        "understates the actual ruin.", fg=typer.colors.RED)
        typer.secho(f"  max leverage with P(margin call) <= {tolerance:.0%}: {safe:.2f}x "
                    f"(at {maintenance:.0%} maintenance)",
                    fg=typer.colors.GREEN if safe > 1 else typer.colors.YELLOW)

        if sensitivity:
            # One maintenance number is not an answer: the safe rung moves with
            # it, and brokers set their own. Show the band.
            cells = []
            for m in MAINTENANCE_GRID:
                sc = leverage_scan(eq.pct_change().dropna(),
                                   financing_rate=financing, maintenance=m,
                                   n_paths=max(400, paths // 4))
                if sc.empty:
                    continue
                cells.append(f"{m:.0%}→{max_safe_leverage(sc, tolerance):.2f}x")
            if cells:
                typer.echo(f"  maintenance sensitivity: {'  '.join(cells)}")

    typer.echo("\nNOT modelled (all make real leverage WORSE, never better): intraday "
               "gaps through the maintenance level, brokers raising maintenance in "
               "stress, borrow availability, and the tax cost of a forced sale.")


@app.command()
def jurisdiction(
    horizon: int = 0,
    since: str = typer.Option("", help="only trades filed on/after this date; "
                                       "defaults to the current Congress start"),
):
    """Do members trade better INSIDE their committee's jurisdiction?

    The last untested hypothesis, and the one the academic literature actually
    makes (Ziobrowski et al.): an Armed Services member plausibly knows
    something about defence procurement that they do not about supermarkets.
    Every other angle here pools all filers and all trades, which would wash
    out an effect concentrated in a minority of them.

    TWO HONESTY CONSTRAINTS, BOTH ENFORCED RATHER THAN DOCUMENTED. Committee
    membership upstream is a CURRENT snapshot, so the test is restricted to
    trades filed while that roster applied, scoring a 2016 trade with today's
    committees is the same look-ahead the 'abnormal' subset had. And the
    committee->sector map is hand-authored judgement, deliberately conservative,
    with Appropriations and Budget excluded because their remit spans the whole
    economy and would mark nearly everything in-jurisdiction."""
    from copytrade.backtest.costs import get_cost_model
    from copytrade.backtest.engine import run_event_study
    from copytrade.rank.jurisdiction import (
        CURRENT_CONGRESS_START,
        build_member_jurisdictions,
        build_pit_jurisdictions,
        classify,
        classify_pit,
        compare,
        eligibility_pit,
        eligibility_report,
        load_sector_map,
    )

    conn, settings = _open()
    h = horizon or settings.horizon
    hcol = f"h{h}"
    since = since or CURRENT_CONGRESS_START
    buys = store.trades_frame(conn, kinds=["congress"], sides=["buy"],
                              instruments=["stock"])
    prices_m = store.load_price_matrix(conn)
    if buys.empty or settings.benchmark not in prices_m.columns:
        typer.echo("no congress buys or no benchmark prices, run `prices` first")
        raise typer.Exit(1)

    # Prefer the point-in-time path whenever the history has been walked: it
    # scores each trade against the roster in force on its OWN filing date,
    # which removes the look-ahead that forced the current-Congress cut-off and
    # recovers the ~96% of the sample that restriction discarded.
    pit_lookup, n_snapshots = build_pit_jurisdictions(conn)
    use_pit = n_snapshots > 1
    juris = build_member_jurisdictions(conn)
    sectors = load_sector_map(conn)
    if not juris and not use_pit:
        typer.secho("no committee jurisdictions, run `ingest-committees` "
                    "(and `enrich-legislators` first)", fg=typer.colors.RED)
        raise typer.Exit(1)
    if not sectors:
        typer.secho("sector_map is empty, run `sectors-refresh` first",
                    fg=typer.colors.RED)
        raise typer.Exit(1)

    bench = prices_m[settings.benchmark].dropna()
    es = run_event_study(buys, prices_m, bench, get_cost_model(settings.cost_model),
                         [h], anchor="filing",
                         same_day_close=settings.same_day_close,
                         cutoff=settings.same_day_cutoff,
                         amount_basis=settings.amount_basis,
                         trade_notional=settings.trade_notional)
    pt = es.per_trade
    if pt.empty:
        typer.echo("no priceable congress buys")
        raise typer.Exit(0)

    typer.echo(f"=== Committee jurisdiction vs traded sector "
               f"(horizon {h}d, excess vs {settings.benchmark}, net) ===")
    if use_pit:
        elig = eligibility_pit(pt, pit_lookup, sectors)
        typer.secho(f"\nPOINT-IN-TIME: {n_snapshots} membership snapshots, each "
                    f"trade is scored against the roster in force on its own "
                    f"filing date, so no date restriction is needed.",
                    fg=typer.colors.GREEN)
        typer.echo(f"{len(sectors)} tickers have a sector.")
        typer.echo(f"{'priced buys':<34}{elig['total_priced_buys']:>8}")
        typer.echo(f"{'  with a known sector':<34}{elig['with_sector']:>8}")
        typer.echo(f"{'  with a known jurisdiction':<34}{elig['with_jurisdiction']:>8}")
        typer.echo(f"{'  ELIGIBLE':<34}{elig['eligible']:>8}")
        lab = classify_pit(pt, pit_lookup, sectors)
    else:
        elig = eligibility_report(pt, juris, sectors, since)
        typer.secho("\nSNAPSHOT-ONLY: current membership only, so the test is "
                    "restricted to trades filed on/after the current Congress. "
                    "Run `ingest-committees-history` to recover the full sample.",
                    fg=typer.colors.YELLOW)
        typer.echo(f"\n{len(juris)} filers have a mapped jurisdiction; "
                   f"{len(sectors)} tickers have a sector.")
        typer.echo(f"{'priced buys':<34}{elig['total_priced_buys']:>8}")
        typer.echo(f"{'  filed on/after ' + since:<34}{elig['after_snapshot_date']:>8}")
        typer.echo(f"{'  ...with a known sector':<34}{elig['of_those_with_sector']:>8}")
        typer.echo(f"{'  ...with a known jurisdiction':<34}{elig['of_those_with_jurisdiction']:>8}")
        typer.echo(f"{'  ELIGIBLE':<34}{elig['eligible']:>8}")
        typer.echo(f"\n  ({elig['attrition_note']})")
        lab = classify(pt, juris, sectors, since)
    if lab.empty:
        typer.secho("\nNo eligible trades, the snapshot window contains no "
                    "priced buys by a mapped filer in a mapped sector.",
                    fg=typer.colors.YELLOW)
        raise typer.Exit(0)

    res = compare(lab, hcol, h)
    typer.echo(f"\n{'group':<24}{'n':>7}{'mean':>10}{'median':>10}{'hit%':>7}{'t':>8}{'blk':>5}")
    for label, key in (("in jurisdiction", "in_jurisdiction"),
                       ("out of jurisdiction", "out_of_jurisdiction")):
        g = res[key]
        typer.echo(f"{label:<24}{g['n']:>7}"
                   f"{_fmt(g['mean'], '{:+.2%}'):>10}{_fmt(g['median'], '{:+.2%}'):>10}"
                   f"{_fmt((g['hit'] or 0)*100 if g['hit'] is not None else None, '{:.0f}'):>7}"
                   f"{_fmt(g.get('t'), '{:.2f}'):>8}{g.get('blocks', 0):>5}")

    typer.echo(f"\nPaired monthly difference (in − out): "
               f"{_fmt(res['diff_mean'], '{:+.2%}')} "
               f"(t={_fmt(res['diff_t'], '{:.2f}')}, "
               f"p={_fmt(res['diff_p'], '{:.3f}')}, "
               f"{res['paired_months']} months with both groups)")

    sig = (res["diff_t"] is not None and abs(res["diff_t"]) > 2
           and (res["diff_mean"] or 0) > 0)
    n_in = res["in_jurisdiction"]["n"]
    blocks = min(res["in_jurisdiction"].get("blocks") or 0,
                 res["out_of_jurisdiction"].get("blocks") or 0)
    # A null from a thin sample is NOT evidence of absence, and this design
    # discards ~96% of the history by construction. Say which one this is.
    underpowered = n_in < 200 or blocks < 8 or res["paired_months"] < 24
    typer.echo("")
    if sig:
        typer.secho("JURISDICTION EFFECT DETECTED, buys inside the member's own "
                    "committee remit outperform their other buys. This is the "
                    "first positive result in the project; treat it as a lead to "
                    "replicate, not a conclusion.", fg=typer.colors.GREEN)
    elif underpowered:
        typer.secho(
            f"UNDERPOWERED: NOT a null. Only {n_in} in-jurisdiction trades over "
            f"{res['paired_months']} paired months ({blocks} independent blocks) "
            f"survive the snapshot restriction, so this test could only detect a "
            f"very large effect. The hypothesis is UNTESTED at usable power, not "
            f"refuted. Historical committee membership would be needed to test it "
            f"properly.", fg=typer.colors.YELLOW)
    else:
        typer.secho("No significant jurisdiction effect. Members' trades inside "
                    "their own committee's remit do not beat their trades "
                    "outside it.", fg=typer.colors.YELLOW)
    if use_pit:
        typer.echo(
            f"\nCaveats that bound this result: the committee→sector map is "
            f"hand-authored judgement (deliberately conservative; Appropriations "
            f"and Budget excluded as economy-wide), and only {len(sectors)} "
            f"tickers have a sector, so trades in unmapped names are excluded. "
            f"Membership is resolved point-in-time from {n_snapshots} snapshots, "
            f"so this result is NOT limited by the roster date.")
    else:
        typer.echo("\nCaveats that bound this result: committee membership is a "
                   "current snapshot (hence the date restriction), the "
                   "committee→sector map is hand-authored judgement, and the "
                   "eligible sample is a small fraction of the history.")


@app.command()
def timescale(
    kind: str = typer.Option("congress", help="congress | insider, which disclosure channel"),
    min_n: int = typer.Option(30, help="cells with fewer priced trades are flagged underpowered"),
    show_all: bool = typer.Option(False, help="print every grid cell, not just the band summaries"),
):
    """WHERE ON THE TIME AXIS is there a signal? Sweeps {subset x holding
    horizon} from 1 day to 2 years, point-in-time and cost-net, and corrects
    for multiple testing (Benjamini-Hochberg FDR + Bonferroni).

    The earlier 'no edge' verdict tested ONE horizon (63d). This tests the
    whole axis, split into the two sleeves the strategy was always meant to
    have: SHORT (1-21d, event-driven post-filing drift) and LONG (42-504d,
    durable tilt). Sweeping without correction is data dredging, so the
    correction is not optional here, a cell is only a finding if it survives
    FDR."""
    from copytrade.backtest.costs import get_cost_model
    from copytrade.backtest.engine import run_event_study
    from copytrade.rank.insiders import (
        abnormal_size_ids_pit,
        cluster_buy_ids,
        top_quartile_mask,
    )
    from copytrade.rank.timescale import (
        ALL_HORIZONS,
        band_summary,
        decay_profile,
        horizon_sweep,
    )

    conn, settings = _open()
    kinds = ["congress"] if kind == "congress" else ["insider"]
    buys = store.trades_frame(conn, kinds=kinds, sides=["buy"], instruments=["stock"])
    prices_m = store.load_price_matrix(conn)
    if buys.empty or settings.benchmark not in prices_m.columns:
        typer.echo("no buys or no benchmark prices, run `prices` first")
        raise typer.Exit(1)
    bench = prices_m[settings.benchmark].dropna()
    es = run_event_study(buys, prices_m, bench, get_cost_model(settings.cost_model),
                         ALL_HORIZONS, anchor="filing",
                         same_day_close=settings.same_day_close,
                         cutoff=settings.same_day_cutoff,
                         amount_basis=settings.amount_basis,
                         trade_notional=settings.trade_notional)
    pt = es.per_trade
    if pt.empty:
        typer.echo("no priceable buys")
        raise typer.Exit(0)

    cluster_ids = cluster_buy_ids(buys, 30, 3)
    # point-in-time variant: the whole-sample median would label a 2016 trade
    # "abnormal" using 2026 data, which breaks the point-in-time invariant in
    # the SUBSET DEFINITION even though execution stays clean
    abnormal_ids = abnormal_size_ids_pit(buys, 2.0)
    big, big_info = top_quartile_mask(pt["amount"])
    if big_info["degenerate"]:
        typer.secho(
            f"NOTE: congress amounts are disclosed RANGES, so the 75th "
            f"percentile sits on a tied bucket ({big_info['threshold']:,.0f}); "
            f"the 'large' subset is {big_info['fraction']:.1%} of trades, not "
            f"25%. Read that row accordingly.", fg=typer.colors.YELLOW)
    subsets = {
        "ALL buys": pd.Series(True, index=pt.index),
        "cluster (>=3/30d)": pt["trade_id"].isin(cluster_ids),
        "large ($ top quartile)": big,
        "abnormal (>=2x own median)": pt["trade_id"].isin(abnormal_ids),
    }

    grid = horizon_sweep(pt, subsets, ALL_HORIZONS, min_n=min_n)
    n_tests = int(grid["n_tests"].iloc[0]) if not grid.empty else 0
    typer.echo(f"=== Timescale sweep: {len(buys)} {kind} buys, {len(pt)} priceable, "
               f"excess vs {settings.benchmark}, net of {settings.cost_model} ===")
    typer.echo(f"{n_tests} testable cells -> multiple-testing correction applied "
               f"across the WHOLE grid (BH-FDR primary, Bonferroni backstop).")
    typer.echo("t_mo  = clustered by calendar month (comparable with earlier work)\n"
               "t_adj = clustered on NON-OVERLAPPING blocks as long as the holding\n"
               "        period, because two 504-day positions opened a month apart\n"
               "        share ~96% of their window and are not independent draws.\n"
               "blk   = how many independent blocks that leaves. FDR keys off t_adj;\n"
               "        fewer than 5 blocks is flagged rather than trusted.\n")

    for band in ("short", "long"):
        g = grid[grid["band"] == band]
        if g.empty:
            continue
        label = ("SHORT band (1-21d), event-driven / post-filing drift"
                 if band == "short" else
                 "LONG band (42-504d): durable tilt / position sleeve")
        typer.echo(f"--- {label} ---")
        typer.echo(f"{'subset':<28}{'h':>5}{'n':>7}{'mean':>9}{'hit%':>6}"
                   f"{'t_mo':>7}{'t_adj':>7}{'blk':>5}{'q(FDR)':>9}  flag")
        for _, r in g.iterrows():
            if not show_all and r["n"] == 0:
                continue
            flag = ""
            if r["survives_fdr"]:
                flag = "** SURVIVES FDR"
            elif r["raw_sig_only"]:
                flag = "raw only (expected from noise at this grid size)"
            elif r["underpowered"] and r["n"]:
                flag = ("too few independent blocks"
                        if r["blocks"] < 5 else "underpowered")
            typer.echo(f"{r['subset']:<28}{r['horizon']:>5}{r['n']:>7}"
                       f"{_fmt(r['mean'], '{:+.2%}'):>9}{_fmt(r['hit']*100, '{:.0f}'):>6}"
                       f"{_fmt(r['t_month'], '{:.2f}'):>7}{_fmt(r['t'], '{:.2f}'):>7}"
                       f"{r['blocks']:>5}{_fmt(r['q_fdr'], '{:.3f}'):>9}  {flag}")
        typer.echo("")

    typer.echo("--- Band verdict ---")
    bs = band_summary(grid)
    typer.echo(bs.to_string(index=False))

    survivors = grid[grid["survives_fdr"]]
    typer.echo("")
    if survivors.empty:
        typer.secho(
            "NO cell survives FDR correction. Read this precisely: it means the "
            "signal is not located at some other horizon either, the earlier "
            "63d null was not a horizon artifact.", fg=typer.colors.YELLOW)
    else:
        typer.secho(f"{len(survivors)} cell(s) survive FDR correction:",
                    fg=typer.colors.GREEN)
        typer.echo(survivors[["subset", "horizon", "band", "n", "mean", "t", "q_fdr"]]
                   .to_string(index=False))

    typer.echo("\n--- Decay shape, ALL buys (is there a drift curve at all?) ---")
    dp = decay_profile(grid, "ALL buys")
    show = dp.assign(mean=(dp["mean"] * 100).round(2), hit=(dp["hit"] * 100).round(0))
    typer.echo(show.rename(columns={"mean": "mean%", "hit": "hit%"}).to_string(index=False))
    typer.echo("  (a real short-lived signal RISES then flattens; a random one wanders)")


@app.command()
def exits(
    horizon: int = 0,
    kind: str = typer.Option("congress", help="congress | insider"),
):
    """HOW TO EXIT: does any path-dependent exit rule beat 'hold N days'?

    Every earlier result used a single fixed-horizon exit, which is a modelling
    convenience rather than a strategy. This tests trailing stops, take-profits,
    stop-losses and combinations against that baseline, on the same trades and
    the same point-in-time execution.

    The column that matters is `vs_base`, the PAIRED difference against the
    baseline. A rule can look profitable just by inheriting the baseline's
    return; only a positive paired difference means the rule added something.
    Rule parameters are a dredging surface, so BH-FDR is applied across the
    whole grid of paired comparisons."""
    from copytrade.backtest.costs import get_cost_model
    from copytrade.backtest.engine import run_event_study
    from copytrade.rank.exits import rule_grid

    conn, settings = _open()
    h = horizon or settings.horizon
    kinds = ["congress"] if kind == "congress" else ["insider"]
    buys = store.trades_frame(conn, kinds=kinds, sides=["buy"], instruments=["stock"])
    prices_m = store.load_price_matrix(conn)
    if buys.empty or settings.benchmark not in prices_m.columns:
        typer.echo("no buys or no benchmark prices, run `prices` first")
        raise typer.Exit(1)
    bench = prices_m[settings.benchmark].dropna()
    es = run_event_study(buys, prices_m, bench, get_cost_model(settings.cost_model),
                         [h], anchor="filing", same_day_close=settings.same_day_close,
                         cutoff=settings.same_day_cutoff,
                         amount_basis=settings.amount_basis,
                         trade_notional=settings.trade_notional)
    if es.per_trade.empty:
        typer.echo("no priceable buys")
        raise typer.Exit(0)

    grid = rule_grid(es.paths, es.per_trade, baseline_h=h)
    typer.echo(f"=== Exit rules: {len(es.per_trade)} {kind} buys, baseline hold {h}d, "
               f"excess vs {settings.benchmark}, net of {settings.cost_model} ===")
    typer.echo(f"{'rule':<26}{'n':>6}{'mean':>9}{'hit%':>6}{'hold':>7}"
               f"{'t_own':>8}{'vs_base':>10}{'t_vs':>7}{'q(FDR)':>9}")
    for _, r in grid.iterrows():
        typer.echo(
            f"{r['rule']:<26}{r['n']:>6}{_fmt(r['mean'], '{:+.2%}'):>9}"
            f"{_fmt(r['hit']*100, '{:.0f}'):>6}{_fmt(r['mean_hold'], '{:.0f}'):>7}"
            f"{_fmt(r['t_own'], '{:.2f}'):>8}{_fmt(r['vs_base'], '{:+.2%}'):>10}"
            f"{_fmt(r['t_vs_base'], '{:.2f}'):>7}{_fmt(r['q_fdr'], '{:.3f}'):>9}"
        )
    winners = grid[grid["beats_baseline"]]
    typer.echo("")
    if winners.empty:
        typer.secho("No exit rule beats the fixed-horizon baseline after FDR "
                    "correction. Exit engineering is not where the edge is.",
                    fg=typer.colors.YELLOW)
    else:
        typer.secho(f"{len(winners)} rule(s) beat the baseline after correction:",
                    fg=typer.colors.GREEN)
        typer.echo(winners[["rule", "mean", "vs_base", "t_vs_base", "q_fdr"]]
                   .to_string(index=False))
    typer.echo("\n  (exits are simulated on excess-vs-benchmark paths, so 'trail' means "
               "giving back that much RELATIVE to the benchmark, not a broker stop order)")


@app.command("who-to-copy")
def who_to_copy(
    horizon: int = 0,
    freq: str = typer.Option("Y", help="ranking window: Y=calendar year, Q=quarter"),
    min_trades: int = typer.Option(5, help="trades a filer needs in BOTH windows to be ranked"),
    leaderboard: bool = typer.Option(False, help="also print the (dangerous) full-sample ranking"),
):
    """Is there a persistent skill difference between filers, OUT OF SAMPLE?

    Copying the in-sample top-N returned +101% vs copy-all's +212%, which shows
    that ranking on past return and keeping the winners fails. This asks the
    question underneath it: does a filer's edge in one period predict the next?
    Rank filers on a TRAIN window only, then measure them on the following
    TEST window. Reports Spearman rank correlation and the top-minus-bottom
    tercile spread, with t-stats across window pairs.

    A null here is strongly actionable: it means there is no stable 'who', so
    copy-everyone is the correct design and effort belongs on WHEN/HOW LONG
    (`timescale`) and risk control (`greeks`)."""
    from copytrade.backtest.costs import get_cost_model
    from copytrade.backtest.engine import run_event_study
    from copytrade.rank.persistence import filer_leaderboard, persistence_report

    conn, settings = _open()
    h = horizon or settings.horizon
    hcol = f"h{h}"
    buys = store.trades_frame(conn, kinds=["congress"], sides=["buy"], instruments=["stock"])
    prices_m = store.load_price_matrix(conn)
    if buys.empty or settings.benchmark not in prices_m.columns:
        typer.echo("no congress buys or no benchmark prices, run `prices` first")
        raise typer.Exit(1)
    bench = prices_m[settings.benchmark].dropna()
    es = run_event_study(buys, prices_m, bench, get_cost_model(settings.cost_model),
                         [h], anchor="filing", same_day_close=settings.same_day_close,
                         cutoff=settings.same_day_cutoff,
                         amount_basis=settings.amount_basis,
                         trade_notional=settings.trade_notional)
    pt = es.per_trade
    if pt.empty:
        typer.echo("no priceable congress buys")
        raise typer.Exit(0)

    rep = persistence_report(pt, hcol, freq=freq, min_trades=min_trades)
    typer.echo(f"=== Who to copy: OUT-OF-SAMPLE filer persistence "
               f"(horizon {h}d, excess vs {settings.benchmark}, net) ===")
    if rep["pairs"] == 0:
        typer.echo("insufficient data: no adjacent windows with enough ranked filers")
        raise typer.Exit(0)

    typer.echo(f"\n{rep['pairs']} adjacent {freq}-window pairs, filers needing "
               f">={min_trades} trades in both\n")
    d = rep["detail"]
    show = d.assign(top_tercile=(d["top_tercile"] * 100).round(2),
                    bottom_tercile=(d["bottom_tercile"] * 100).round(2),
                    spread=(d["spread"] * 100).round(2),
                    spearman=d["spearman"].round(3))
    typer.echo(show[["train", "test", "filers", "spearman", "top_tercile",
                     "bottom_tercile", "spread"]].to_string(index=False))

    typer.echo("\n--- Pooled verdict ---")
    typer.echo(f"mean Spearman rho      : {rep['mean_spearman']:+.3f}  "
               f"(t={rep['t_spearman']:.2f}, p={rep['p_spearman']:.3f})")
    typer.echo(f"mean tercile spread    : {rep['mean_spread']:+.2%}  "
               f"(t={rep['t_spread']:.2f}, p={rep['p_spread']:.3f})")
    typer.echo(f"pairs with +ve spread  : {rep['pairs_positive']}/{rep['pairs']}  "
               f"(sign test p={rep['sign_p']:.3f})")

    persists = (abs(rep["t_spread"]) > 2) if rep["t_spread"] == rep["t_spread"] else False
    typer.echo("")
    if persists and rep["mean_spread"] > 0:
        typer.secho("PERSISTENCE DETECTED, past filer edge predicts future edge. "
                    "Selection is worth building on.", fg=typer.colors.GREEN)
    else:
        typer.secho("NO significant persistence, a filer's past edge does not predict "
                    "their next-period edge. There is no stable 'who' to copy; "
                    "copy-everyone is the correct design.", fg=typer.colors.YELLOW)

    if leaderboard:
        lb = filer_leaderboard(pt, hcol)
        typer.echo("\n--- Full-sample ranking (NOT a buy list, see the verdict above) ---")
        typer.echo(lb.head(25).to_string())


@app.command()
def timeline(recent_since: int = typer.Option(2022, help="years >= this get the recent-attention block")):
    """Full-timeline, year-by-year: the copy-ALL congressional basket vs SPY vs
    top-N selection, every calendar year from first data to now, so you can
    see exactly which years following the crowd helped and which it hurt.
    Recent years get extra attention (quarterly split + cumulative alpha)."""
    from copytrade.backtest import benchmarks as bm
    from copytrade.backtest import metrics
    from copytrade.backtest.costs import get_cost_model
    from copytrade.backtest.engine import walk_forward

    conn, settings = _open()
    cost = get_cost_model(settings.cost_model)
    trades = store.trades_frame(conn, instruments=["stock"])
    prices_m = store.load_price_matrix(conn)
    if trades.empty or settings.benchmark not in prices_m.columns:
        typer.echo("no trades or no benchmark prices, run `prices` first")
        raise typer.Exit(1)
    bench = prices_m[settings.benchmark].dropna()
    common = dict(horizon=settings.horizon, max_weight=settings.max_weight,
                  trade_notional=settings.trade_notional,
                  same_day_close=settings.same_day_close, cutoff=settings.same_day_cutoff)

    wf_all = walk_forward(trades, prices_m, bench, cost, copy_all=True, min_trades=1, **common)
    if wf_all is None or wf_all.equity.empty:
        typer.echo("copy-all produced no equity curve")
        raise typer.Exit(1)
    span = wf_all.equity.index
    wf_top = walk_forward(trades, prices_m, bench, cost, top_n=settings.top_n,
                          min_trades=settings.min_trades_walkforward, **common)
    spy_eq = bm.buy_and_hold(bench.loc[span[0]:span[-1]], cost)

    a_all = metrics.annual_returns(wf_all.equity)
    a_spy = metrics.annual_returns(spy_eq)
    a_top = metrics.annual_returns(wf_top.equity) if wf_top is not None and not wf_top.equity.empty else None
    # equal-weight RSP: the breadth proxy, if the basket ≈ RSP, its "edge" is
    # just equal-weighting, buyable off the shelf
    a_rsp = (metrics.annual_returns(bm.buy_and_hold(prices_m["RSP"].dropna().loc[span[0]:span[-1]], cost))
             if "RSP" in prices_m.columns else None)

    # per-year disclosed buys + active filers (activity context)
    buys = trades[trades["side"] == "buy"].dropna(subset=["filing_datetime"]).copy()
    buys["y"] = buys["filing_datetime"].dt.year
    by_year = buys.groupby("y").agg(n_buys=("trade_id", "count"),
                                    filers=("filer_name", "nunique"))

    typer.echo(f"=== Full timeline: copy-ALL basket vs SPY, by calendar year "
               f"({span[0].date()} → {span[-1].date()}, net of {cost.name}) ===\n")
    typer.echo(f"{'year':<6}{'copyAll':>9}{'SPY':>9}{'RSPew':>9}{'alpha':>9}{'topN':>9}{'buys':>8}{'filers':>8}")
    cum = 1.0
    cum_spy = 1.0
    cum_rsp = 1.0
    for y in sorted(a_all.index):
        ca = a_all.get(y, float('nan'))
        sp = a_spy.get(y, float('nan'))
        rs = a_rsp.get(y, float('nan')) if a_rsp is not None else float('nan')
        tp = a_top.get(y) if a_top is not None else None
        alpha = ca - sp if pd.notna(ca) and pd.notna(sp) else float('nan')
        if pd.notna(ca):
            cum *= (1 + ca)
        if pd.notna(sp):
            cum_spy *= (1 + sp)
        if pd.notna(rs):
            cum_rsp *= (1 + rs)
        nb = int(by_year["n_buys"].get(y, 0))
        nf = int(by_year["filers"].get(y, 0))
        mark = "  <<" if y >= recent_since else ""
        tp_s = f"{tp:>+8.1%}" if tp is not None and pd.notna(tp) else ", "
        rs_s = f"{rs:>+8.1%}" if pd.notna(rs) else ", "
        typer.echo(f"{y:<6}{ca:>+8.1%} {sp:>+8.1%} {rs_s} {alpha:>+8.1%} {tp_s}{nb:>8}{nf:>8}{mark}")
    typer.echo(f"\ncumulative: copyAll {cum-1:+.1%}  vs  SPY {cum_spy-1:+.1%}"
               f"{'  vs  RSP(ew) ' + format(cum_rsp-1, '+.1%') if a_rsp is not None else ''}  "
               f"(cumulative alpha vs SPY {cum-cum_spy:+.1%})")

    # recent-years attention: how many recent years did copy-all win, and by how much
    recent = [y for y in a_all.index if y >= recent_since]
    if recent:
        wins = sum(1 for y in recent if pd.notna(a_all.get(y)) and pd.notna(a_spy.get(y))
                   and a_all[y] > a_spy[y])
        avg_alpha = float(pd.Series([a_all[y] - a_spy[y] for y in recent
                                     if pd.notna(a_all.get(y)) and pd.notna(a_spy.get(y))]).mean())
        typer.echo(f"\nrecent years (≥{recent_since}): copy-all beat SPY in {wins}/{len(recent)} years, "
                   f"avg annual alpha {avg_alpha:+.1%}")


@app.command("sectors-refresh")
def sectors_refresh(limit: int = typer.Option(800, help="max NEW tickers to map this run"),
                    min_trades: int = typer.Option(1, help="only map tickers with >= this many trades")):
    """Populate the ticker->sector map from yfinance (cached; resumable). Maps
    the most-traded unmapped tickers first, up to `limit` per run."""
    from copytrade.ingest.sectors import refresh_sectors

    conn, _ = _open()
    typer.echo(json.dumps(refresh_sectors(conn, limit=limit, min_trades=min_trades), indent=2))


@app.command()
def holdings(
    filer: str = typer.Option("", help="one filer, e.g. 'Cleo Fields'"),
    top: int = typer.Option(25, help="rows to print"),
):
    """What a filer is DISCLOSED to be in, direction and attention, not a portfolio.

    A PTR discloses transactions, not holdings: positions opened before the
    tracking window are invisible, and every size is a disclosed RANGE, so a
    netted position is a band that widens with each transaction rather than a
    number. Consumer trackers render this as a "portfolio"; this prints the
    bounds instead.
    """
    from copytrade.rank.holdings import filer_positions, filer_summary, most_held

    conn, _ = _open()
    trades = store.trades_frame(conn, instruments=None)
    if trades.empty:
        typer.echo("no trades: run an ingest first")
        raise typer.Exit(1)
    pos = filer_positions(trades, filer or None)
    if pos.empty:
        typer.echo(f"no disclosed positions for {filer!r}" if filer else "no positions")
        raise typer.Exit(1)

    money = lambda v: f"${v/1000:,.0f}k" if abs(v) >= 1000 else f"${v:,.0f}"  # noqa: E731
    if filer:
        typer.echo(f"=== {filer}: disclosed net positions ===")
        typer.echo(f"{'ticker':<8}{'buys':>5}{'sells':>6}{'net (low)':>12}"
                   f"{'net (high)':>12}  {'last':<12}status")
        for r in pos.head(top).itertuples():
            typer.echo(f"{r.ticker:<8}{r.n_buys:>5}{r.n_sells:>6}"
                       f"{money(r.net_low):>12}{money(r.net_high):>12}  "
                       f"{r.last_disclosed:<12}{r.status}")
    else:
        typer.echo("=== most active filers by disclosed gross flow ===")
        typer.echo(f"{'filer':<28}{'tickers':>8}{'net (low)':>12}{'net (high)':>12}  last")
        for r in filer_summary(pos, top=top).itertuples():
            typer.echo(f"{r.filer[:27]:<28}{r.tickers:>8}"
                       f"{money(r.net_low):>12}{money(r.net_high):>12}  {r.last_disclosed}")
        typer.echo("\n=== most crowded names (filers with a net-buy disclosure) ===")
        typer.echo(f"{'ticker':<8}{'filers':>7}{'buys':>6}{'net (high)':>13}  last")
        for r in most_held(pos, top=15).itertuples():
            typer.echo(f"{r.ticker:<8}{r.filers:>7}{r.buys:>6}"
                       f"{money(r.net_high):>13}  {r.last_disclosed}")

    typer.secho(
        "\nA PTR discloses TRANSACTIONS, not holdings. Positions opened before "
        "this window are invisible, sizes are ranges, and a net-zero band is "
        "NOT a closed position.", fg=typer.colors.YELLOW)


@app.command("prices-qa")
def prices_qa(
    dry_run: bool = typer.Option(False, "--dry-run", help="report without recording"),
    show: int = typer.Option(40, help="how many offenders to print"),
):
    """Screen every price series for values that are positive but not credible.

    `upsert_prices` already rejects adj_close <= 0. That guard was written after
    four OTC ADRs produced a -587,200% return, and it removed the negative ticks
    while leaving the rest of the same broken series behind, DAIUF still ran
    5.1e-07 to 45.31 and surfaced as the best-performing name on the dashboard.
    This finds the residue: sub-penny artifacts, unadjusted reverse splits,
    series oscillating between two scales, and multi-year gaps read as one
    session. Nothing is deleted; offenders are quarantined and excluded at
    `load_price_matrix`, so their trades join the counted 'unpriceable' bucket.
    Real events survive on purpose (DJT +357%, GME +135%).
    """
    from copytrade.store.price_qa import refresh_quarantine, scan_all

    conn, _ = _open()
    findings = scan_all(conn)
    out = refresh_quarantine(conn, dry_run=dry_run)
    typer.echo(json.dumps({k: v for k, v in out.items() if k != "tickers"}, indent=2))
    if findings:
        typer.echo(f"\n{'ticker':<10}{'reason':<32}{'min':>12}{'max':>14}"
                   f"{'range':>12}{'>100% days':>12}")
        for f in sorted(findings, key=lambda x: -x["detail"]["range_ratio"])[:show]:
            d = f["detail"]
            typer.echo(f"{f['ticker']:<10}{f['reason']:<32}{d['min']:>12.4g}"
                       f"{d['max']:>14.4g}{d['range_ratio']:>12.4g}"
                       f"{d['sessions_over_100pct']:>12}")
    if dry_run:
        typer.secho("\n(dry run: nothing recorded)", fg=typer.colors.YELLOW)
    else:
        typer.secho("\nRebuild the dashboard so the panels drop these: "
                    "python -m copytrade dashboard-build", fg=typer.colors.GREEN)


@app.command()
def areas(
    horizon: int = 0,
    focus: str = typer.Option("Lisa Mcclain,Mark Dr Green,Thomas Suozzi,Nancy Pelosi,Marjorie Taylor Greene",
                              help="comma-separated filers to profile individually"),
):
    """What areas do these people pursue, and where are their picks actually
    good? Sector attribution (net forward return per area), the best/worst/most-
    crowded individual names, and a per-filer footprint. Point-in-time, cost-net."""
    from copytrade.backtest.costs import get_cost_model
    from copytrade.backtest.engine import run_event_study
    from copytrade.rank.areas import filer_focus, sector_attribution, ticker_leaders

    conn, settings = _open()
    h = horizon or settings.horizon
    hcol = f"h{h}"
    trades = store.trades_frame(conn, sides=["buy"], instruments=["stock"])
    prices_m = store.load_price_matrix(conn)
    if trades.empty or settings.benchmark not in prices_m.columns:
        typer.echo("no buys or no benchmark prices, run `prices` first")
        raise typer.Exit(1)
    bench = prices_m[settings.benchmark].dropna()
    es = run_event_study(trades, prices_m, bench, get_cost_model(settings.cost_model),
                         [h], anchor="filing", same_day_close=settings.same_day_close,
                         cutoff=settings.same_day_cutoff, amount_basis=settings.amount_basis,
                         trade_notional=settings.trade_notional)
    pt = es.per_trade
    sector_map = pd.read_sql_query("SELECT ticker, sector, industry FROM sector_map", conn)
    mapped = sector_map["ticker"].nunique() if not sector_map.empty else 0
    typer.echo(f"=== Area attribution: {len(pt)} priced buys, horizon {h}d, "
               f"net of {settings.cost_model}; {mapped} tickers sector-mapped ===")
    if mapped == 0:
        typer.secho("(no sector map yet, run `sectors-refresh`; ticker-level analysis still valid)",
                    fg=typer.colors.YELLOW)

    sec = sector_attribution(pt, sector_map, hcol, min_buys=25)
    typer.echo("\n--- Sectors: where the disclosed buys actually worked (sorted by net return) ---")
    if not sec.empty:
        show = sec.assign(win_rate=(sec["win_rate"]*100).round(0),
                          mean_ret=(sec["mean_ret"]*100).round(1),
                          median_ret=(sec["median_ret"]*100).round(1),
                          notional=(sec["notional"]/1e6).round(1))
        typer.echo(show.rename(columns={"notional": "notional_$M", "win_rate": "win%",
                                        "mean_ret": "meanRet%", "median_ret": "medRet%"}).to_string(index=False))

    lead = ticker_leaders(pt, hcol, min_buys=10, top=12)
    for label, key in [("BEST names (mean net return, n≥10)", "best"),
                       ("WORST names (n≥10)", "worst"),
                       ("MOST-CROWDED names (what they pile into)", "most_traded")]:
        d = lead[key]
        typer.echo(f"\n--- {label} ---")
        if not d.empty:
            dd = d.assign(win_rate=(d["win_rate"]*100).round(0), mean_ret=(d["mean_ret"]*100).round(1))
            typer.echo(dd.rename(columns={"win_rate": "win%", "mean_ret": "meanRet%"}).to_string(index=False))

    typer.echo("\n\n=== Per-filer footprint (what each one pursues, and whether it works) ===")
    names_avail = set(pt["filer_name"].dropna().unique())
    for want in [f.strip() for f in focus.split(",") if f.strip()]:
        matches = sorted({n for n in names_avail if want.lower() in n.lower()})
        for name in matches[:1]:  # first match per requested filer
            ff = filer_focus(pt, sector_map, hcol, name, top=8)
            if ff["tickers"].empty:
                continue
            typer.echo(f"\n----- {name} -----")
            secs = ff["sectors"].head(5).assign(mean_ret=(ff["sectors"].head(5)["mean_ret"]*100).round(1))
            typer.echo("  areas pursued (by # buys): " +
                       ", ".join(f"{r.sector} (n={r.n}, {r.mean_ret:+.1f}%)" for r in secs.itertuples()))
            tk = ff["tickers"].assign(win_rate=(ff["tickers"]["win_rate"]*100).round(0),
                                      mean_ret=(ff["tickers"]["mean_ret"]*100).round(1))
            typer.echo(tk.rename(columns={"win_rate": "win%", "mean_ret": "meanRet%"}).to_string(index=False))


@app.command()
def report():
    """Write ranking.csv, decay plots, walk-forward equity vs benchmarks, README."""
    from copytrade.report.report import generate_report

    conn, settings = _open()
    res = generate_report(conn, settings)
    typer.echo(f"wrote {len(res['files'])} files to {settings.reports_dir}/")
    for name, s in res["stats"].items():
        if s:
            typer.echo(f"  {name}: total {s['total_return']:+.1%}, "
                       f"CAGR {s['cagr']:+.1%}, sharpe {s['sharpe']:.2f}, "
                       f"maxDD {s['max_drawdown']:.1%}")


@app.command()
def poll():
    """Continuous scraping loop (source-registry scheduler): all enabled
    disclosure feeds at their own cadences, webhook alerts on actionable
    trades by top-ranked filers. Informational only, no orders."""
    from copytrade.scrape import run_scheduler

    conn, settings = _open()
    asyncio.run(run_scheduler(conn, settings))


@app.command("scrape-once")
def scrape_once_cmd():
    """Run every enabled source exactly once and print a per-source summary
    (cron-friendly: the recommended shape on a small VPS). See docs/SCRAPING.md."""
    from copytrade.scrape.scheduler import scrape_once

    conn, settings = _open()
    results = asyncio.run(scrape_once(conn, settings))
    if not results:
        typer.echo("no sources enabled (check finnhub key / scrape_sources)")
        return
    for r in results:
        status = f"ERROR {r.error}" if r.error else f"+{r.inserted} new, {r.deduped} dup, {r.skipped} skip"
        typer.echo(f"  [{r.source}] {status}" + (f", {r.note}" if r.note else ""))


@app.command("scrape-status")
def scrape_status():
    """Show each source's last-run heartbeat (liveness/health)."""
    conn, _ = _open()
    rows = conn.execute(
        "SELECT key, value FROM poll_state WHERE key LIKE 'scrape:%:heartbeat' ORDER BY key"
    ).fetchall()
    if not rows:
        typer.echo("no scrape heartbeats yet, run `scrape-once` or `poll`")
        return
    for key, value in rows:
        src = key.split(":")[1]
        typer.echo(f"  {src}: {value}")


@app.command("scrape-house")
def scrape_house(
    year: int = typer.Option(0, help="calendar year; 0 = current"),
    reparse: bool = typer.Option(False, help="clear state + delete existing House rows, "
                                             "then rebuild (use after a parser change, e.g. options)"),
):
    """One-shot US House PTR scrape (authoritative Clerk feed). Needs
    disclosures-clerk.house.gov + pdfplumber for PDF text; see docs/SCRAPING.md."""
    from datetime import datetime

    from copytrade.ingest.house_clerk import scrape_recent_ptrs

    # Without a PDF text layer every PTR is counted as "needs OCR" and silently
    # skipped, which looks like a successful run that ingested nothing. House
    # PTRs are the only post-2021 congressional source, so fail loudly instead.
    try:
        import pdfplumber  # noqa: F401
    except ImportError:
        typer.secho(
            "pdfplumber is not installed, every PTR would be skipped as "
            "'needs OCR' and this run would ingest nothing.\n"
            "    pip install pdfplumber==0.11.4",
            fg=typer.colors.RED,
        )
        raise typer.Exit(1)

    conn, settings = _open()
    yrs = [year] if year else [datetime.now().year]
    res = asyncio.run(scrape_recent_ptrs(conn, settings, years=yrs, reparse=reparse))
    typer.echo(json.dumps(res.__dict__, indent=2, default=str))


@app.command("dashboard-build")
def dashboard_build(
    only: str = typer.Option("", help="comma-separated subset, e.g. 'health,scorecard'"),
    leverage_paths: int = typer.Option(1200, help="bootstrap paths for the leverage panel"),
):
    """Precompute every dashboard panel to reports/dashboard/*.json.

    The dashboard serves these from disk. They are not computed per request
    because the timescale sweep, the leverage bootstrap and the feature/control
    build each take minutes, a request handler would time the browser out
    long before finishing. Re-run after new data lands; the UI shows the build
    timestamp so a stale panel is visible as stale rather than passed off as
    live.

    Exits non-zero if any panel did not rebuild, so a partial run cannot be
    mistaken for a complete one."""
    from datetime import datetime, timezone

    from copytrade.web import analyses as an

    names = [s.strip() for s in only.split(",") if s.strip()] or list(an.BUILDERS)
    unknown = [n for n in names if n not in an.BUILDERS]
    if unknown:
        typer.secho(f"unknown: {unknown}; options: {sorted(an.BUILDERS)}",
                    fg=typer.colors.RED)
        raise typer.Exit(1)

    started = datetime.now(timezone.utc)
    failed: list[str] = []
    for name in names:
        typer.echo(f"building {name}...")
        try:
            kwargs = {"paths": leverage_paths} if name == "leverage" else {}
            payload = an.BUILDERS[name](**kwargs)
            p = an.save(name, payload)
            note = payload.get("error") or payload.get("verdict", "")
            typer.secho(f"  -> {p}  {note[:110]}",
                        fg=typer.colors.RED if payload.get("error") else typer.colors.GREEN)
        except Exception as e:  # noqa: BLE001, one bad panel must not block the rest
            typer.secho(f"  FAILED: {type(e).__name__}: {e}", fg=typer.colors.RED)
            failed.append(name)

    # Verify what actually landed on disk. Building panel-by-panel means a run
    # can die partway and leave a MIXTURE of vintages, which is worse than a
    # stale dashboard because each panel looks fine on its own, the numbers
    # simply no longer describe the same data. That is not hypothetical: a run
    # that died on the last builder left `scorecard_cad` eleven days behind the
    # rest, and the exit code alone did not say so.
    stale = [n for n in names if not an.was_built_since(n, started)]
    if stale or failed:
        typer.secho(
            f"\n{len(stale)} of {len(names)} panels did NOT rebuild in this run: "
            f"{sorted(set(stale) | set(failed))}\n"
            "The dashboard would serve these alongside fresh ones, so its panels "
            "would describe different data. Re-run before trusting any figure.",
            fg=typer.colors.RED,
        )
        raise typer.Exit(1)

    typer.secho(f"\nall {len(names)} panels rebuilt", fg=typer.colors.GREEN)
    typer.echo("Serve it with:  python -m copytrade serve")


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8000):
    """Serve the web dashboard + JSON API."""
    import uvicorn

    uvicorn.run("copytrade.web.app:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    app()
