"""Paper trading (plan Phase 8), two layers, zero live-order capability.

1. Local paper ledger (`orders` table): signals from top-ranked filers become
   order rows; `fill_open_orders` fills each at the first cached close
   STRICTLY AFTER the order's creation timestamp, the exact execution
   assumption the backtest makes, so paper results and backtest results are
   comparable apples-to-apples.

2. AlpacaPaper: thin REST client for https://paper-api.alpaca.markets so the
   same signals can be mirrored into a real paper account (live quotes, real
   fill simulation). Paper only: the client REFUSES any non-paper base URL
   unless COPYTRADE_ALLOW_LIVE=1 is set explicitly. Alpaca paper accounts are
   free and need no US residency; for eventual live execution as a Canadian
   the recommended platform is IBKR Canada (see docs/CANADA.md), its API
   integration is deliberately NOT implemented until paper results justify it.

Network note: paper-api.alpaca.markets is unreachable from restricted
sandboxes (403 via proxy); run `orders-push-alpaca` from your own machine.
"""

from __future__ import annotations

import os
import sqlite3

import httpx

ALPACA_PAPER_BASE = "https://paper-api.alpaca.markets"


# ------------------------------------------------------------ local ledger --

def record_signal_orders(
    conn: sqlite3.Connection,
    account_for_filer: dict[str, str] | None = None,
    notional: float = 10_000.0,
    since: str | None = None,
    watchlist: set[str] | None = None,
) -> int:
    """Turn recent BUY filings by watched filers into paper orders.

    account_for_filer: filer name -> 'tfsa' | 'margin' (from allocate_sleeves);
    unknown filers default to 'paper'. Dedup via UNIQUE(trade_id, account).
    Returns number of orders created.
    """
    q = (
        "SELECT t.trade_id, t.ticker, f.name FROM trades t "
        "JOIN filers f USING(filer_id) "
        "WHERE t.side='buy' AND t.instrument='stock' AND t.ticker IS NOT NULL"
    )
    args: list = []
    if since:
        q += " AND t.filing_datetime >= ?"
        args.append(since)
    created = 0
    for trade_id, ticker, filer in conn.execute(q, args).fetchall():
        if watchlist is not None and filer not in watchlist:
            continue
        account = (account_for_filer or {}).get(filer, "paper")
        cur = conn.execute(
            "INSERT OR IGNORE INTO orders(trade_id, ticker, side, notional, account, note) "
            "VALUES (?,?,?,?,?,?)",
            (trade_id, ticker, "buy", notional, account, f"copy {filer}"),
        )
        created += cur.rowcount
    conn.commit()
    return created


def fill_open_orders(conn: sqlite3.Connection) -> dict:
    """Fill 'new' local-paper orders at the first cached close AFTER
    created_at (never same-day: THE rule). Orders whose ticker has no later
    close stay open; that is honest, not a bug."""
    filled = still_open = 0
    rows = conn.execute(
        "SELECT order_id, ticker, notional, created_at FROM orders "
        "WHERE status='new' AND broker='paper'"
    ).fetchall()
    for order_id, ticker, notional, created_at in rows:
        px = conn.execute(
            "SELECT date, adj_close FROM prices WHERE ticker=? AND date > DATE(?) "
            "ORDER BY date LIMIT 1",
            (ticker, created_at),
        ).fetchone()
        if px is None:
            still_open += 1
            continue
        fill_date, fill_price = px
        conn.execute(
            "UPDATE orders SET status='filled', fill_price=?, fill_date=?, "
            "qty=? WHERE order_id=?",
            (fill_price, fill_date, (notional or 0) / fill_price, order_id),
        )
        filled += 1
    conn.commit()
    return {"filled": filled, "still_open": still_open}


# ---------------------------------------------------------------- alpaca ----

class AlpacaPaper:
    """Minimal Alpaca paper-trading client (market notional orders only)."""

    def __init__(self, key_id: str, secret: str, base_url: str = ALPACA_PAPER_BASE):
        if "paper" not in base_url and os.environ.get("COPYTRADE_ALLOW_LIVE") != "1":
            raise ValueError(
                f"refusing non-paper base URL {base_url!r}, set COPYTRADE_ALLOW_LIVE=1 "
                "only when you truly mean it"
            )
        if not key_id or not secret:
            raise ValueError("Alpaca keys missing (ALPACA_API_KEY_ID / ALPACA_API_SECRET_KEY)")
        self._client = httpx.Client(
            base_url=base_url,
            headers={"APCA-API-KEY-ID": key_id, "APCA-API-SECRET-KEY": secret},
            timeout=30,
        )

    def account(self) -> dict:
        r = self._client.get("/v2/account")
        r.raise_for_status()
        return r.json()

    def submit_notional(self, symbol: str, notional: float, side: str) -> dict:
        r = self._client.post("/v2/orders", json={
            "symbol": symbol, "notional": round(notional, 2), "side": side,
            "type": "market", "time_in_force": "day",
        })
        r.raise_for_status()
        return r.json()

    def close(self) -> None:
        self._client.close()


def push_orders_to_alpaca(conn: sqlite3.Connection, client: AlpacaPaper) -> dict:
    """Mirror open local-paper orders into the Alpaca paper account. Each
    pushed order is recorded (broker -> alpaca_paper, ext_ref = Alpaca id) so
    re-runs never double-submit."""
    pushed = failed = 0
    rows = conn.execute(
        "SELECT order_id, ticker, side, notional FROM orders "
        "WHERE status='new' AND broker='paper' AND ext_ref IS NULL"
    ).fetchall()
    for order_id, ticker, side, notional in rows:
        try:
            res = client.submit_notional(ticker, notional or 0.0, side)
        except httpx.HTTPError as e:
            conn.execute("UPDATE orders SET note=COALESCE(note,'') || ? WHERE order_id=?",
                         (f" [alpaca error: {e}]", order_id))
            failed += 1
            continue
        conn.execute(
            "UPDATE orders SET broker='alpaca_paper', ext_ref=? WHERE order_id=?",
            (res.get("id"), order_id),
        )
        pushed += 1
    conn.commit()
    return {"pushed": pushed, "failed": failed}
