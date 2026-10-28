"""Live asyncio pollers (plan §5): Form 4 Atom feed every few minutes,
congress aggregator hourly. On each newly inserted trade by a top-ranked
filer, post a webhook alert. Graceful resume: dedup is DB-level
(UNIQUE(source, source_uid)), last-seen accession kept in poll_state.
"""

from __future__ import annotations

import asyncio
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from copytrade.alerts.webhook import format_alert, send_alert
from copytrade.store import db as store


def top_filer_stats(settings) -> dict[str, dict]:
    """Filer -> trailing stats from the last generated ranking (reports/ranking.csv)."""
    path = Path(settings.reports_dir) / "ranking.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    alpha_col = next((c for c in df.columns if c.startswith("lag_alpha")), None)
    top = df[df.get("copyable", False) == True]  # noqa: E712
    if top.empty:
        top = df.head(settings.top_n)
    return {
        r["filer"]: {
            "lag_alpha": r.get(alpha_col, 0.0), "tstat": r.get("tstat", 0.0),
            "n_trades": r.get("n_trades", 0), "hit_rate": r.get("hit_rate", 0.0),
        }
        for _, r in top.iterrows()
    }


async def maybe_alert(conn, settings, filer_name: str, trade: dict, watchlist: dict):
    if filer_name not in watchlist:
        return
    text = format_alert(trade, watchlist[filer_name])
    text = enrich_alert(conn, settings, trade, text)
    await send_alert(settings.webhook_url, text, settings.webhook_format)


def enrich_alert(conn, settings, trade: dict, text: str) -> str:
    """Prepend a cluster-buy flag and append recent headlines (context only)."""
    from copytrade.ingest.news import latest_news
    from copytrade.rank.signals import live_cluster_size

    ticker, side = trade.get("ticker"), trade.get("side")
    if ticker and side == "buy":
        n = live_cluster_size(conn, ticker, str(trade.get("filing_datetime")))
        if n >= 2:
            text = f"🔥 **CLUSTER BUY**, {n} distinct filers bought {ticker} in 10 days\n" + text
    if ticker:
        heads = latest_news(conn, ticker, limit=3)
        if heads:
            text += "\nrecent news:" + "".join(
                f"\n  • {h['published_at'][:10]} [{h['headline'][:90]}](<{h['url']}>)"
                for h in heads
            )
    return text


async def poll_form4(conn, settings, watchlist: dict):
    from copytrade.ingest.edgar import EdgarClient
    from copytrade.ingest.form4 import parse_form4, store_form4

    client = EdgarClient(settings.edgar_user_agent, settings.edgar_rps, settings.raw_dir)
    try:
        while True:
            try:
                refs = await client.recent_form4_accessions()
                last_seen = store.get_state(conn, "form4_last_accession")
                new_last = refs[0].accession if refs else last_seen
                for ref in refs:
                    if ref.accession == last_seen:
                        break
                    xml = await client.fetch_filing_xml(ref)
                    if not xml:
                        continue
                    accepted = await client.fetch_acceptance_datetime(ref) \
                        or f"{date.today()} 17:00:00"
                    filing = parse_form4(xml)
                    n = store_form4(conn, filing, ref.accession, accepted)
                    if n and filing.owner_name:
                        for tx in filing.transactions:
                            if tx.side:
                                await maybe_alert(conn, settings, filing.owner_name, {
                                    "filer_name": filing.owner_name, "kind": "insider",
                                    "side": tx.side, "ticker": filing.issuer_symbol,
                                    "filing_datetime": accepted, "transaction_date": tx.date,
                                    "amount_low": (tx.shares or 0) * (tx.price or 0),
                                    "amount_high": (tx.shares or 0) * (tx.price or 0),
                                }, watchlist)
                if new_last:
                    store.set_state(conn, "form4_last_accession", new_last)
            except Exception as e:
                print(f"[form4 poll] {e}")
            await asyncio.sleep(settings.poll_form4_secs)
    finally:
        await client.aclose()


async def poll_congress(conn, settings, watchlist: dict):
    from copytrade.ingest.congress import FinnhubCongress, store_congress_trades

    if not settings.finnhub_api_key:
        print("[congress poll] no Finnhub key configured, skipping")
        return
    from copytrade.ingest.congress import load_universe

    symbols = load_universe(settings.congress_universe_file)
    if not symbols:
        print("[congress poll] empty universe, skipping")
        return
    source = FinnhubCongress(settings.finnhub_api_key)
    while True:
        try:
            frm = (date.today() - timedelta(days=14)).isoformat()
            trades, _ = await source.fetch(symbols, frm, date.today().isoformat())
            for t in trades:
                filer = store.resolve_alias(conn, t.name)
                inserted, _ = store_congress_trades(conn, [t])
                if inserted:
                    await maybe_alert(conn, settings, filer, {
                        "filer_name": filer, "kind": "congress", "side": t.side,
                        "ticker": t.ticker, "filing_datetime": t.filing_date,
                        "transaction_date": t.transaction_date,
                        "amount_low": t.amount_low, "amount_high": t.amount_high,
                    }, watchlist)
        except Exception as e:
            print(f"[congress poll] {e}")
        await asyncio.sleep(settings.poll_congress_secs)


async def run_pollers(conn, settings):
    """Back-compat entrypoint: delegates to the source-registry scheduler
    (copytrade/scrape). The standalone poll_form4/poll_congress coroutines
    above are retained for reference and direct use."""
    from copytrade.scrape import run_scheduler

    await run_scheduler(conn, settings)
