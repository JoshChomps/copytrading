"""QuiverQuant congressional trading, the paid upgrade for LIVE congress.

Why it earns a key: unlike Finnhub's symbol-keyed endpoint, Quiver's
/beta/live/congresstrading returns the BULK of recent trades across BOTH
chambers in one call, no watchlist blind spot, with report (disclosure)
dates. It is still a third-party aggregator, so it stays classified
non-authoritative in the scrape registry (the Clerk/eFD scrapers remain the
source of record); in practice it is the easiest complete live feed.

Parser follows Quiver's documented response schema (Representative,
BioGuideID, ReportDate, TransactionDate, Ticker, Transaction, Range/Amount,
House, Party). quiverquant.com is unreachable from this dev sandbox, so
validate one live pull (`backfill-quiver --limit 20`) before trusting a bulk
run: records missing a ReportDate are counted and skipped, never backdated.
"""

from __future__ import annotations

from datetime import datetime

import httpx

from copytrade.ingest.congress import CongressTrade, store_congress_trades
from copytrade.ingest.senate_watcher import parse_amount
from copytrade.scrape.base import BaseSource, ScrapeResult

LIVE_URL = "https://api.quiverquant.com/beta/live/congresstrading"


def _iso(s: str | None) -> str | None:
    for fmt in ("%Y-%m-%d", "%m/%d/%Y"):
        try:
            return datetime.strptime((s or "")[:10], fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


def parse_quiver_payload(payload: list[dict]) -> tuple[list[CongressTrade], int]:
    trades: list[CongressTrade] = []
    skipped = 0
    for r in payload or []:
        name = (r.get("Representative") or r.get("Senator") or r.get("Name") or "").strip()
        ticker = (r.get("Ticker") or "").strip().upper()
        filing = _iso(r.get("ReportDate") or r.get("Filed"))
        tx_date = _iso(r.get("TransactionDate") or r.get("Traded"))
        if not name or not ticker or not filing or not tx_date:
            skipped += 1
            continue
        raw_amount = r.get("Range") or r.get("Amount")
        lo, hi = parse_amount(raw_amount) if isinstance(raw_amount, str) else (
            (float(raw_amount), None) if raw_amount else (None, None)
        )
        t = (r.get("Transaction") or "").lower()
        side = "buy" if "purchase" in t else "sell" if "sale" in t else None
        chamber = "Senate" if "senate" in (r.get("House") or "").lower() else "House"
        trades.append(CongressTrade(
            name=name.title(), chamber=chamber, ticker=ticker, side=side,
            transaction_date=tx_date, filing_date=filing,
            amount_low=lo, amount_high=hi, instrument="stock",
            source="quiver_congress", raw=r,
        ))
    return trades, skipped


async def fetch_live(api_key: str, timeout: float = 60.0) -> list[dict]:
    async with httpx.AsyncClient(
        timeout=timeout, headers={"Authorization": f"Bearer {api_key}"}
    ) as client:
        resp = await client.get(LIVE_URL)
        resp.raise_for_status()
        return resp.json()


class QuiverCongressSource(BaseSource):
    name = "quiver_congress"
    kind = "congress"
    cadence_secs = 3600
    authoritative = False   # bulk (no symbol blind spot), but still third-party

    async def run_once(self, conn, settings) -> ScrapeResult:
        res = ScrapeResult(source=self.name)
        if not settings.quiver_api_key:
            return ScrapeResult(source=self.name, note="no Quiver key, skipped")
        trades, skipped = parse_quiver_payload(await fetch_live(settings.quiver_api_key))
        res.skipped = skipped
        for t in trades:
            ins, dedup = store_congress_trades(conn, [t])
            res.inserted += ins
            res.deduped += dedup
            if ins and t.side == "buy":
                res.alertable.append({
                    "filer_name": t.name, "kind": "congress", "side": t.side,
                    "ticker": t.ticker, "filing_datetime": t.filing_date,
                    "transaction_date": t.transaction_date,
                    "amount_low": t.amount_low, "amount_high": t.amount_high,
                })
        return res
