import numpy as np
import pandas as pd

from copytrade.backtest.costs import get_cost_model
from copytrade.rank.profit import confidence_label, filer_track_records


def _world():
    cal = pd.bdate_range("2020-01-01", "2020-12-31")
    # XX doubles over the year, YY halves, linearly
    px = pd.DataFrame({
        "XX": np.linspace(100, 200, len(cal)),
        "YY": np.linspace(100, 50, len(cal)),
        "SPY": np.linspace(100, 110, len(cal)),
    }, index=cal)
    bench = px["SPY"]
    return px, bench, cal


def _t(rows):
    df = pd.DataFrame(rows, columns=[
        "trade_id", "filer_id", "filer_name", "filer_kind", "side",
        "ticker", "transaction_date", "filing_datetime"])
    df["filing_datetime"] = pd.to_datetime(df["filing_datetime"])
    df["transaction_date"] = pd.to_datetime(df["transaction_date"])
    return df


def test_confidence_labels():
    assert confidence_label(1) == "anecdotal"
    assert confidence_label(3) == "low"
    assert confidence_label(10) == "medium"
    assert confidence_label(30) == "high"


def test_winner_and_loser_track_records():
    px, bench, cal = _world()
    trades = _t([
        # filer 1: one buy of the doubler, held to year end (open)
        (1, 1, "Good Picker", "congress", "buy", "XX", "2020-01-01", "2020-01-02 17:00"),
        # filer 2: one buy of the halver
        (2, 2, "Bad Picker", "congress", "buy", "YY", "2020-01-01", "2020-01-02 17:00"),
    ])
    tr = filer_track_records(trades, px, bench, get_cost_model("zero"), horizon_cap=252)
    by = tr.set_index("filer")
    assert by.loc["Good Picker", "mean_return"] > 0.7      # ~ +95% gross
    assert by.loc["Bad Picker", "mean_return"] < -0.3
    assert by.loc["Good Picker", "win_rate"] == 1.0
    assert by.loc["Bad Picker", "win_rate"] == 0.0
    # sorted by total P&L: good picker first
    assert tr.iloc[0]["filer"] == "Good Picker"


def test_low_frequency_filer_is_included_and_labeled():
    """The whole point: a 2-trade filer must NOT be dropped."""
    px, bench, cal = _world()
    trades = _t([
        (1, 5, "Rare Genius", "congress", "buy", "XX", "2020-01-01", "2020-01-02 17:00"),
        (2, 5, "Rare Genius", "congress", "buy", "XX", "2020-06-01", "2020-06-02 17:00"),
    ])
    tr = filer_track_records(trades, px, bench, get_cost_model("zero"))
    assert len(tr) == 1
    assert tr.iloc[0]["n_buys_priced"] == 2
    assert tr.iloc[0]["confidence"] == "anecdotal"


def test_fifo_matches_buy_to_later_sell():
    px, bench, cal = _world()
    trades = _t([
        (1, 7, "Trader", "congress", "buy", "XX", "2020-01-01", "2020-01-02 17:00"),
        (2, 7, "Trader", "congress", "sell", "XX", "2020-06-30", "2020-07-01 17:00"),
    ])
    tr = filer_track_records(trades, px, bench, get_cost_model("zero"))
    r = tr.iloc[0]
    assert r["n_closed"] == 1
    assert 0 < r["avg_holding_days"] < 200      # ~ half a year, closed early
    # return is the mid-year appreciation, not full-year
    assert 0.3 < r["mean_return"] < 0.7


def test_overlapping_positions_aggregate_additively_not_compounded():
    px, bench, cal = _world()
    # 5 identical buys of the doubler -> additive P&L ~ 5 * single, NOT (1+r)^5
    rows = [(i, 9, "Multi", "congress", "buy", "XX", "2020-01-01", "2020-01-02 17:00")
            for i in range(5)]
    tr = filer_track_records(_t(rows), px, bench, get_cost_model("zero"))
    r = tr.iloc[0]
    per_pick = r["mean_return"]
    assert r["total_pnl_per_10k"] ==  __import__("pytest").approx(per_pick * 5 * 10_000, rel=1e-6)


def test_empty_and_no_prices():
    px, bench, cal = _world()
    assert filer_track_records(pd.DataFrame(), px, bench, get_cost_model("zero")).empty
    trades = _t([(1, 1, "X", "congress", "buy", "NOPRICE", "2020-01-01", "2020-01-02 17:00")])
    assert filer_track_records(trades, px, bench, get_cost_model("zero")).empty
