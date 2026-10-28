"""A transaction date after its own filing is impossible and must not survive.

The House Clerk PDFs are OCR of scanned forms and produced 3031-04-30 (filed
2021-05-03), 2220-04-07, 2202-09-19 and ~80 quieter day/month typos. Those
dates only feed the disclosure-lag and zero-lag decay statistics, the backtest
executes off `filing_datetime`: so the fix nulls the date and KEEPS the trade.
Deleting the row would silently shrink the sample the headline results rest on.
"""

from __future__ import annotations

import pandas as pd
import pytest

from copytrade.store import db as store


@pytest.fixture()
def conn(tmp_path):
    c = store.connect(tmp_path / "t.db")
    store.run_migrations(c)
    return c


def _insert(conn, txn, filing="2021-05-03 17:00:00", uid="u1"):
    fid = store.upsert_filer(conn, "Test Member", "congress")
    store.insert_trade(
        conn, filer_id=fid, source="house_clerk", source_uid=uid, ticker="AAPL",
        side="buy", instrument="stock", transaction_date=txn,
        filing_datetime=filing, amount_low=1000.0, amount_high=15000.0,
    )
    return conn.execute(
        "SELECT transaction_date FROM trades WHERE source_uid=?", (uid,)
    ).fetchone()[0]


@pytest.mark.parametrize("bad", ["3031-04-30", "2220-04-07", "2202-09-19",
                                 "2035-09-01", "2030-06-01", "2021-05-04"])
def test_transaction_date_after_filing_is_nulled(conn, bad):
    assert _insert(conn, bad) is None


@pytest.mark.parametrize("good", ["2021-05-03", "2021-04-01", "2019-02-27",
                                  "2013-03-15"])
def test_plausible_transaction_dates_survive(conn, good):
    assert _insert(conn, good, uid=good) == good


def test_the_trade_itself_is_kept_not_deleted(conn):
    """Nulling the date must not cost the sample a trade, `filing_datetime` is
    the execution anchor and it is untouched."""
    _insert(conn, "3031-04-30")
    row = conn.execute(
        "SELECT COUNT(*), MAX(filing_datetime) FROM trades").fetchone()
    assert row[0] == 1
    assert row[1] == "2021-05-03 17:00:00"


def test_long_positive_lag_is_not_an_error(conn):
    """603 live trades are disclosed more than two years late. Lateness is the
    phenomenon this project measures, not a defect to clean away."""
    assert _insert(conn, "2013-03-15", filing="2019-05-13 17:00:00") == "2013-03-15"


def test_sanitize_is_pure_and_reports_its_reason():
    ok, reason = store.sanitize_transaction_date("2021-01-01", "2021-05-03 17:00:00")
    assert ok == "2021-01-01" and reason is None
    bad, reason = store.sanitize_transaction_date("3031-04-30", "2021-05-03 17:00:00")
    assert bad is None and reason == "after_filing"


def test_missing_inputs_pass_through():
    assert store.sanitize_transaction_date(None, "2021-05-03")[0] is None
    assert store.sanitize_transaction_date("2021-01-01", None)[0] == "2021-01-01"


def test_unparseable_date_is_preserved_for_diagnosis():
    """Left as-is on purpose: trades_frame coerces it to NaT anyway, and keeping
    the raw string is what makes the upstream parser bug findable from the DB."""
    v, reason = store.sanitize_transaction_date("not-a-date", "2021-05-03")
    assert v == "not-a-date" and reason is None


def test_same_day_transaction_and_filing_is_valid(conn):
    """Filing the same day you trade is fast, not impossible."""
    assert _insert(conn, "2021-05-03", filing="2021-05-03 17:00:00") == "2021-05-03"


def test_migration_cleans_preexisting_rows(tmp_path):
    """Migration 008 must fix databases built before the ingest guard existed."""
    c = store.connect(tmp_path / "legacy.db")
    store.run_migrations(c)
    fid = store.upsert_filer(c, "Legacy", "congress")
    # bypass insert_trade to simulate a row written by the old code path
    c.execute(
        "INSERT INTO trades(filer_id, source, source_uid, ticker, side, instrument,"
        " transaction_date, filing_datetime) VALUES (?,?,?,?,?,?,?,?)",
        (fid, "house_clerk", "legacy-1", "AAPL", "buy", "stock",
         "3031-04-30", "2021-05-03 17:00:00"))
    c.commit()
    c.execute("DELETE FROM schema_migrations WHERE id='008_trade_date_sanity.sql'")
    c.commit()
    store.run_migrations(c)
    assert c.execute("SELECT transaction_date FROM trades "
                     "WHERE source_uid='legacy-1'").fetchone()[0] is None


def test_lag_statistics_exclude_the_nulled_rows(conn):
    """The whole point of nulling rather than clamping: a bogus date must not
    contribute a bogus lag."""
    from copytrade.rank.lag import reporting_lag_days

    _insert(conn, "3031-04-30", uid="bad")
    _insert(conn, "2021-04-03", uid="good")
    trades = store.trades_frame(conn, instruments=["stock"])
    lag = reporting_lag_days(trades).dropna()
    assert len(lag) == 1
    assert lag.iloc[0] == 30
