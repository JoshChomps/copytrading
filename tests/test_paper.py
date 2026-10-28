"""Paper-order ledger: fills must follow THE rule (first close strictly AFTER
order creation: never same-day), mirroring the backtest exactly."""

import pandas as pd
import pytest

from copytrade.broker.paper import AlpacaPaper, fill_open_orders, record_signal_orders
from copytrade.store import db as store


def _seed(conn):
    filer = store.upsert_filer(conn, "Sen. Test", "congress", None)
    store.insert_trade(
        conn, filer_id=filer, source="senate_watcher", source_uid="t1",
        ticker="ZZTOP", side="buy", instrument="stock",
        transaction_date="2024-01-05", filing_datetime="2024-01-10 17:00:00",
        amount_low=1001.0, amount_high=15000.0,
    )
    store.upsert_prices(conn, pd.DataFrame({
        "ticker": ["ZZTOP"] * 3,
        "date": ["2024-01-10", "2024-01-11", "2024-01-12"],
        "adj_close": [100.0, 110.0, 120.0],
    }))
    return filer


def test_orders_created_filtered_and_deduped(conn):
    _seed(conn)
    n = record_signal_orders(conn, notional=5_000, watchlist={"Sen. Test"})
    assert n == 1
    assert record_signal_orders(conn, notional=5_000, watchlist={"Sen. Test"}) == 0  # dedup
    assert record_signal_orders(conn, notional=5_000, watchlist={"Somebody Else"}) == 0


def test_fill_is_strictly_after_creation(conn):
    _seed(conn)
    record_signal_orders(conn, notional=5_000, watchlist={"Sen. Test"})
    # force a deterministic creation timestamp on the filing date
    conn.execute("UPDATE orders SET created_at='2024-01-10 17:00:00'")
    conn.commit()
    res = fill_open_orders(conn)
    assert res == {"filled": 1, "still_open": 0}
    row = conn.execute("SELECT fill_date, fill_price, qty FROM orders").fetchone()
    assert row[0] == "2024-01-11"          # NOT 2024-01-10, no same-day fill
    assert row[1] == 110.0
    assert row[2] == pytest.approx(5_000 / 110.0)


def test_unpriced_order_stays_open(conn):
    _seed(conn)
    record_signal_orders(conn, notional=5_000, watchlist={"Sen. Test"})
    conn.execute("UPDATE orders SET created_at='2024-01-12 17:00:00'")  # after last close
    conn.commit()
    assert fill_open_orders(conn) == {"filled": 0, "still_open": 1}


def test_alpaca_refuses_live_url(monkeypatch):
    monkeypatch.delenv("COPYTRADE_ALLOW_LIVE", raising=False)
    with pytest.raises(ValueError, match="paper"):
        AlpacaPaper("k", "s", base_url="https://api.alpaca.markets")


def test_alpaca_requires_keys():
    with pytest.raises(ValueError, match="keys"):
        AlpacaPaper("", "")
