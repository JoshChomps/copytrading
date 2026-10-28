import pandas as pd

from copytrade.rank.signals import (
    cluster_buy_events,
    cluster_sizes,
    live_cluster_size,
    routine_filers,
)
from copytrade.store import db as store


def _trades(rows):
    df = pd.DataFrame(rows, columns=["filer_id", "ticker", "side", "filing_datetime"])
    df["filing_datetime"] = pd.to_datetime(df["filing_datetime"])
    return df


def test_cluster_counts_distinct_filers_trailing_only():
    t = _trades([
        (1, "AAPL", "buy", "2024-01-02"),
        (2, "AAPL", "buy", "2024-01-08"),   # within 10d of filer 1 -> cluster 2
        (1, "AAPL", "buy", "2024-01-09"),   # same filer again -> still 2 distinct
        (3, "AAPL", "buy", "2024-03-01"),   # window expired -> back to 1
        (1, "MSFT", "buy", "2024-01-02"),   # different ticker -> independent
    ])
    sizes = cluster_sizes(t)
    assert sizes.tolist() == [1, 2, 2, 1, 1]
    # the FIRST filer's trade must never see the later filing (trailing only)
    assert sizes.iloc[0] == 1


def test_cluster_sells_do_not_pollute_buy_clusters():
    t = _trades([
        (1, "AAPL", "sell", "2024-01-02"),
        (2, "AAPL", "buy", "2024-01-03"),
    ])
    assert cluster_sizes(t).loc[1] == 1


def test_cluster_buy_events_threshold():
    t = _trades([
        (1, "AAPL", "buy", "2024-01-02"),
        (2, "AAPL", "buy", "2024-01-05"),
    ])
    t["trade_id"] = [10, 11]
    ev = cluster_buy_events(t, min_filers=2)
    assert len(ev) == 1 and ev.iloc[0]["trade_id"] == 11 and ev.iloc[0]["cluster_size"] == 2


def test_routine_filer_same_month_three_consecutive_years():
    rows = [(1, "AAPL", "sell", f"{y}-06-15") for y in (2019, 2020, 2021)]  # routine
    rows += [(2, "AAPL", "buy", d) for d in ("2019-06-15", "2021-06-15", "2023-06-15")]  # gaps
    rows += [(3, "AAPL", "buy", d) for d in ("2019-01-05", "2020-07-11", "2021-11-30")]
    r = routine_filers(_trades(rows))
    assert bool(r[1]) is True
    assert bool(r[2]) is False   # non-consecutive years
    assert bool(r[3]) is False   # different months


def test_live_cluster_size_sql(conn):
    f1 = store.upsert_filer(conn, "A", "congress", None)
    f2 = store.upsert_filer(conn, "B", "congress", None)
    for i, (f, dt) in enumerate([(f1, "2024-01-02 17:00:00"), (f2, "2024-01-08 17:00:00")]):
        store.insert_trade(conn, filer_id=f, source="s", source_uid=f"u{i}",
                           ticker="AAPL", side="buy", instrument="stock",
                           transaction_date="2024-01-01", filing_datetime=dt)
    conn.commit()
    assert live_cluster_size(conn, "AAPL", "2024-01-08 17:00:00") == 2
    assert live_cluster_size(conn, "AAPL", "2024-01-02 17:00:00") == 1  # trailing only
