"""Congressional trading ingestion (STOCK Act PTRs) via aggregators.

The aggregator sits behind a small interface so it is swappable (plan §2.2):
  * FinnhubCongress: /stock/congressional-trading, symbol-keyed, free tier.
  * JsonFileCongress: load a local JSON dump (any aggregator's export),
    used for fixtures/tests and for users with Quiver/Unusual Whales exports.

Rules: filing (disclosure) date is REQUIRED, records without it are counted
and skipped, never backdated. Amounts are RANGES; both bounds are stored.
Options are kept but flagged instrument='option' (MVP backtests stock only).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Protocol

import httpx


@dataclass
class CongressTrade:
    name: str
    chamber: str | None
    ticker: str
    side: str | None            # buy | sell | None (exchange/other)
    transaction_date: str       # YYYY-MM-DD
    filing_date: str            # YYYY-MM-DD, point-in-time anchor
    amount_low: float | None
    amount_high: float | None
    instrument: str = "stock"
    source: str = "aggregator"
    raw: dict | None = None
    uid: str | None = None      # override: dedup key EXCLUDING filing_date
                                # (amended PTRs re-report the same trade with a
                                # later receipt date; the earliest wins)

    @property
    def source_uid(self) -> str:
        if self.uid:
            return self.uid
        key = json.dumps(
            [self.name, self.ticker, self.side, self.transaction_date,
             self.amount_low, self.amount_high, self.filing_date],
            sort_keys=True,
        )
        return hashlib.sha1(key.encode()).hexdigest()[:20]


def load_universe(path: str | Path) -> list[str]:
    """Ticker-per-line file -> symbols. Skips blanks and '#' comments
    a comment line sent to a symbol-keyed API as a ticker 403s the run
    (found the hard way in CI)."""
    p = Path(path)
    if not p.exists():
        return []
    return [
        s.strip().upper()
        for s in p.read_text(encoding="utf-8").splitlines()
        if s.strip() and not s.lstrip().startswith("#")
    ]


class CongressSource(Protocol):
    async def fetch(self, symbols: Iterable[str], from_date: str, to_date: str) -> tuple[list[CongressTrade], int]:
        """Returns (trades, n_skipped_missing_filing_date)."""
        ...


def _side_from_type(tx_type: str) -> str | None:
    t = (tx_type or "").lower()
    if "purchase" in t or t == "buy":
        return "buy"
    if "sale" in t or t == "sell":
        return "sell"
    return None


def _instrument(asset_name: str | None) -> str:
    """Classify a congress asset description into the shared instrument
    taxonomy. Distinguishing call vs put matters: the derivatives study signs
    the underlying's move by direction (long call = bullish, long put =
    bearish), so collapsing both to a bare 'option' would default puts to a
    wrong bullish sign. Mirrors house_clerk.house_instrument so Finnhub- and
    House-sourced options classify identically."""
    a = (asset_name or "").lower()
    if not ("option" in a or "call" in a or "put" in a):
        return "stock"
    if "call" in a:
        return "call_option"
    if "put" in a:
        return "put_option"
    return "option"


def parse_finnhub_payload(payload: dict) -> tuple[list[CongressTrade], int]:
    """Finnhub /stock/congressional-trading response -> normalized trades."""
    trades: list[CongressTrade] = []
    skipped = 0
    for rec in payload.get("data", []):
        filing = rec.get("filingDate") or rec.get("disclosureDate")
        tx_date = rec.get("transactionDate")
        symbol = (rec.get("symbol") or payload.get("symbol") or "").upper()
        if not filing or not tx_date or not symbol:
            skipped += 1
            continue
        trades.append(
            CongressTrade(
                name=(rec.get("name") or "").strip(),
                chamber=rec.get("position"),
                ticker=symbol,
                side=_side_from_type(rec.get("transactionType", "")),
                transaction_date=tx_date[:10],
                filing_date=filing[:10],
                amount_low=rec.get("amountFrom"),
                amount_high=rec.get("amountTo"),
                instrument=_instrument(rec.get("assetName")),
                source="finnhub_congress",
                raw=rec,
            )
        )
    return trades, skipped


class FinnhubCongress:
    """Symbol-keyed fetcher; free tier is ~60 req/min so we space requests.

    Per-symbol errors are collected in `self.errors` (symbol -> status/reason)
    instead of aborting the whole backfill, one bad symbol or a
    tier-restricted endpoint must not sink a pipeline run. If EVERY symbol
    fails identically, the caller can conclude the endpoint itself is
    unavailable for this API tier."""

    BASE = "https://finnhub.io/api/v1/stock/congressional-trading"

    def __init__(self, api_key: str, rps: float = 0.9, transport: httpx.AsyncBaseTransport | None = None):
        if not api_key:
            raise ValueError("FinnhubCongress requires an API key (COPYTRADE_FINNHUB_API_KEY)")
        self.api_key = api_key
        self._interval = 1.0 / rps
        self._transport = transport
        self.errors: dict[str, str] = {}

    async def fetch(self, symbols, from_date, to_date):
        trades: list[CongressTrade] = []
        skipped = 0
        self.errors = {}
        self._n_symbols = len(list(symbols))
        async with httpx.AsyncClient(timeout=30, transport=self._transport) as client:
            for sym in symbols:
                params = {"symbol": sym, "from": from_date, "to": to_date, "token": self.api_key}
                for backoff in (2, 4, 8, 16, None):
                    try:
                        resp = await client.get(self.BASE, params=params)
                    except httpx.TransportError as e:
                        self.errors[sym] = f"transport: {e}"
                        break
                    if resp.status_code == 429 and backoff:
                        await asyncio.sleep(backoff)
                        continue
                    if resp.status_code != 200:
                        self.errors[sym] = f"http {resp.status_code}"
                        break
                    t, s = parse_finnhub_payload(resp.json())
                    trades.extend(t)
                    skipped += s
                    break
                await asyncio.sleep(self._interval)
        return trades, skipped

    @property
    def all_failed_uniformly(self) -> bool:
        """True only when EVERY queried symbol failed with the same error
        the signature of a tier-restricted endpoint, not a bad symbol."""
        return (
            bool(self.errors)
            and len(self.errors) == getattr(self, "_n_symbols", -1)
            and len(set(self.errors.values())) == 1
        )


class JsonFileCongress:
    """Local JSON dump: either a Finnhub-shaped payload or a bare list of records."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    async def fetch(self, symbols=None, from_date=None, to_date=None):
        data = json.loads(self.path.read_text(encoding="utf-8"))
        payload = data if isinstance(data, dict) else {"data": data}
        return parse_finnhub_payload(payload)


def store_congress_trades(conn, trades: list[CongressTrade]) -> tuple[int, int]:
    """Returns (inserted, deduped)."""
    from copytrade.store import db as store

    inserted = deduped = 0
    for t in trades:
        if not t.name:
            deduped += 1
            continue
        filer_id = store.upsert_filer(
            conn, t.name, "congress", {"chamber": t.chamber} if t.chamber else None
        )
        new = store.insert_trade(
            conn,
            filer_id=filer_id,
            source=t.source,
            source_uid=t.source_uid,
            ticker=t.ticker,
            side=t.side,
            instrument=t.instrument,
            transaction_date=t.transaction_date,
            # PTRs carry a disclosure DATE; anchor conservatively to end of
            # business day so execution is never earlier than availability.
            filing_datetime=f"{t.filing_date} 17:00:00",
            amount_low=t.amount_low,
            amount_high=t.amount_high,
        )
        inserted += int(new)
        deduped += int(not new)
    conn.commit()
    return inserted, deduped
