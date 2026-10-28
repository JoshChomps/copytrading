"""Unified async scheduler: the 'constantly scrape' loop.

One task per source, each firing at the source's cadence. Cross-cutting
concerns live here, once, instead of being copy-pasted into every source:

  * per-source exponential backoff on failure (a flaky feed slows itself,
    never the others; a persistently dead source is disabled after
    `max_consecutive_failures` with a logged reason);
  * alert dispatch through the shared watchlist + webhook;
  * a heartbeat row in poll_state per source (last_run, last_ok, counts) so
    a dashboard/health check can see liveness;
  * graceful cancellation.

The single-shot `scrape_once` helper runs every source one time (for cron
deployments that prefer an external timer over a long-lived process, the
recommended shape on a small Vancouver VPS; see docs/SCRAPING.md).
"""

from __future__ import annotations

import asyncio
import json
import time

from copytrade.alerts.webhook import send_alert
from copytrade.scrape.base import ScrapeResult, Source
from copytrade.store import db as store

MAX_CONSECUTIVE_FAILURES = 8
BACKOFF_CAP_SECS = 3600


async def _dispatch_alerts(conn, settings, result: ScrapeResult, watchlist: dict) -> None:
    if not settings.webhook_url or not watchlist:
        return
    from copytrade.poll import enrich_alert
    from copytrade.alerts.webhook import format_alert

    for trade in result.alertable:
        name = trade.get("filer_name")
        if name in watchlist:
            text = enrich_alert(conn, settings, trade, format_alert(trade, watchlist[name]))
            await send_alert(settings.webhook_url, text, settings.webhook_format)


def _record_heartbeat(conn, source: str, result: ScrapeResult, ok: bool) -> None:
    store.set_state(conn, f"scrape:{source}:heartbeat", json.dumps({
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"), "ok": ok,
        "inserted": result.inserted, "deduped": result.deduped,
        "skipped": result.skipped, "error": result.error, "note": result.note,
        # a degraded run is still an `ok` run: it ingested what it could reach
        "degraded": result.degraded,
    }))


async def _run_source_safely(conn, settings, source: Source, watchlist: dict) -> ScrapeResult:
    try:
        result = await source.run_once(conn, settings)
    except Exception as e:  # noqa: BLE001, one bad source must never kill the loop
        result = ScrapeResult(source=source.name, error=f"{type(e).__name__}: {e}")
    _record_heartbeat(conn, source.name, result, ok=result.error is None)
    if result.error is None:
        await _dispatch_alerts(conn, settings, result, watchlist)
    return result


async def _source_loop(conn, settings, source: Source, watchlist: dict) -> None:
    failures = 0
    while True:
        result = await _run_source_safely(conn, settings, source, watchlist)
        if result.error:
            failures += 1
            if failures >= MAX_CONSECUTIVE_FAILURES:
                print(f"[{source.name}] disabled after {failures} failures: {result.error}")
                return
            delay = min(source.cadence_secs * (2 ** failures), BACKOFF_CAP_SECS)
            print(f"[{source.name}] error ({failures}): {result.error}; retry in {delay}s")
        else:
            failures = 0
            delay = source.cadence_secs
            if result.degraded:
                # not a failure: the run ingested what it could reach, so it
                # keeps its cadence and its alerts, but the counts are a floor
                print(f"[{source.name}] {result.degraded}")
            if result.inserted or result.note:
                print(f"[{source.name}] +{result.inserted} new, "
                      f"{result.deduped} dup{f'; {result.note}' if result.note else ''}")
        await asyncio.sleep(delay)


async def run_scheduler(conn, settings, sources: list[Source] | None = None) -> None:
    from copytrade.scrape.registry import build_sources, build_watchlist

    sources = sources or build_sources(settings)
    watchlist = build_watchlist(settings)
    auth = [s.name for s in sources if s.authoritative]
    agg = [s.name for s in sources if not s.authoritative]
    print(f"scraper: {len(sources)} sources "
          f"(authoritative backbone: {auth or 'none'}; aggregators: {agg or 'none'}); "
          f"watching {len(watchlist)} filers; "
          f"alerts {'ON' if settings.webhook_url else 'OFF'}")
    if not sources:
        print("no sources enabled: set finnhub key / universe, or check scrape_sources")
        return
    await asyncio.gather(*(_source_loop(conn, settings, s, watchlist) for s in sources))


async def scrape_once(conn, settings, sources: list[Source] | None = None) -> list[ScrapeResult]:
    """Run every enabled source exactly once (cron-friendly)."""
    from copytrade.scrape.registry import build_sources, build_watchlist

    sources = sources or build_sources(settings)
    watchlist = build_watchlist(settings)
    results = []
    for s in sources:
        results.append(await _run_source_safely(conn, settings, s, watchlist))
    return results
