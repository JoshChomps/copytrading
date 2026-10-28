"""Execution-mapping and no-look-ahead proofs (plan P4 acceptance criteria)."""

from datetime import time as dtime

import numpy as np
import pandas as pd
import pytest

from copytrade.backtest.costs import CostModel, get_cost_model
from copytrade.backtest.engine import execution_position, run_event_study

CAL = pd.bdate_range("2024-01-01", "2024-06-28")
ZERO = get_cost_model("zero")


# ------------------------------------------------------ execution mapping --

def test_filing_after_close_executes_next_day():
    # Wed 2024-02-07 16:35 -> Thu close
    pos = execution_position(pd.Timestamp("2024-02-07 16:35"), CAL)
    assert CAL[pos] == pd.Timestamp("2024-02-08")


def test_intraday_filing_still_next_day_when_conservative():
    pos = execution_position(pd.Timestamp("2024-02-07 10:00"), CAL, same_day_close=False)
    assert CAL[pos] == pd.Timestamp("2024-02-08")


def test_same_day_toggle_respects_cutoff():
    early = execution_position(pd.Timestamp("2024-02-07 10:00"), CAL,
                               same_day_close=True, cutoff=dtime(15, 30))
    late = execution_position(pd.Timestamp("2024-02-07 15:45"), CAL,
                              same_day_close=True, cutoff=dtime(15, 30))
    assert CAL[early] == pd.Timestamp("2024-02-07")
    assert CAL[late] == pd.Timestamp("2024-02-08")


def test_weekend_filing_executes_monday():
    pos = execution_position(pd.Timestamp("2024-02-10 12:00"), CAL)  # Saturday
    assert CAL[pos] == pd.Timestamp("2024-02-12")


def test_filing_beyond_calendar_returns_none():
    assert execution_position(pd.Timestamp("2024-06-28 17:00"), CAL) is None


# ------------------------------------------------------------ no lookahead --

def _pop_fixture():
    """Stock pops +20% BETWEEN transaction date and filing date. A correct
    engine cannot capture the pop; one that peeks at transaction_date does."""
    spy = pd.Series(100.0, index=CAL)
    px = pd.Series(100.0, index=CAL)
    px[CAL >= "2024-02-06"] = 120.0  # pop the day after the insider bought
    prices = pd.DataFrame({"POP": px})
    trades = pd.DataFrame([{
        "trade_id": 1, "filer_id": 1, "filer_name": "A", "filer_kind": "insider",
        "side": "buy", "ticker": "POP",
        "transaction_date": pd.Timestamp("2024-02-05"),
        "filing_datetime": pd.Timestamp("2024-02-08 09:00"),
        "amount_low": 10_000.0, "amount_high": 10_000.0,
        "shares": 100.0, "price": 100.0,
    }])
    return trades, prices, spy


def test_filing_anchor_cannot_capture_pre_filing_pop():
    trades, prices, spy = _pop_fixture()
    es = run_event_study(trades, prices, spy, ZERO, [5], anchor="filing")
    assert len(es.per_trade) == 1
    assert es.per_trade["h5"].iloc[0] == pytest.approx(0.0, abs=1e-12)
    # entry must be strictly after the filing timestamp
    assert es.per_trade["entry_date"].iloc[0] > pd.Timestamp("2024-02-08")


def test_transaction_anchor_shows_what_the_lag_destroyed():
    trades, prices, spy = _pop_fixture()
    es = run_event_study(trades, prices, spy, ZERO, [5], anchor="transaction")
    assert es.per_trade["h5"].iloc[0] == pytest.approx(0.20, abs=1e-12)
    # the difference between the two anchors IS the lag-destroyed alpha
    es_f = run_event_study(trades, prices, spy, ZERO, [5], anchor="filing")
    gap = es.per_trade["h5"].iloc[0] - es_f.per_trade["h5"].iloc[0]
    assert gap == pytest.approx(0.20, abs=1e-12)


def test_costs_are_subtracted_at_horizons():
    trades, prices, spy = _pop_fixture()
    cost = CostModel(name="x", spread_slippage_bps=50, sec_fee_on_sells=False)  # 1% roundtrip
    es = run_event_study(trades, prices, spy, cost, [5], anchor="filing")
    assert es.per_trade["h5"].iloc[0] == pytest.approx(-0.01, abs=1e-12)


def test_unpriceable_counted_and_excluded():
    trades, prices, spy = _pop_fixture()
    ghost = trades.copy()
    ghost["ticker"] = "GONE"
    ghost["trade_id"] = 2
    both = pd.concat([trades, ghost], ignore_index=True)
    es = run_event_study(both, prices, spy, ZERO, [5], anchor="filing")
    assert es.unpriceable == 1 and len(es.per_trade) == 1


def test_excess_is_vs_benchmark():
    trades, prices, spy = _pop_fixture()
    rising_spy = pd.Series(np.linspace(100, 110, len(CAL)), index=CAL)
    es = run_event_study(trades, prices, rising_spy, ZERO, [5], anchor="transaction")
    assert es.per_trade["h5"].iloc[0] < 0.20  # SPY's rise is netted out


def test_amount_for_basis_zero_bound_and_open_ended():
    """Audit fix: a legit 0.0 bound must not be dropped by `or`-coalescing,
    and open-ended buckets ('Over $50M': high=None) must resolve."""
    import pandas as pd

    from copytrade.backtest.engine import amount_for_basis

    class Row:
        amount_low = 0.0
        amount_high = 100.0

    assert amount_for_basis(Row, "low", fallback=999.0) == 0.0
    assert amount_for_basis(Row, "mid", fallback=999.0) == 50.0

    class OpenEnded:
        amount_low = 50_000_000.0
        amount_high = None

    assert amount_for_basis(OpenEnded, "high", 0.0) == 50_000_000.0
    assert amount_for_basis(OpenEnded, "mid", 0.0) == 50_000_000.0

    class Empty:
        amount_low = float("nan")
        amount_high = None

    assert amount_for_basis(Empty, "mid", fallback=123.0) == 123.0


def _wf_frames():
    """Tiny two-filer world: filer 1 buys then sells 5 bd later."""
    import numpy as np
    import pandas as pd

    cal = pd.bdate_range("2019-01-01", "2021-12-31")
    bench = pd.Series(np.linspace(100, 120, len(cal)), index=cal)
    px = pd.DataFrame({"XX": np.linspace(50, 80, len(cal))}, index=cal)
    rows = []
    tid = 0
    for year in (2019, 2020, 2021):
        for m in range(1, 11):
            tid += 1
            buy_dt = pd.Timestamp(f"{year}-{m:02d}-05 17:00:00")
            rows.append(dict(trade_id=tid, filer_id=1, filer_name="F1",
                             filer_kind="congress", side="buy", ticker="XX",
                             transaction_date=buy_dt.normalize() - pd.Timedelta(days=3),
                             filing_datetime=buy_dt,
                             amount_low=1000.0, amount_high=2000.0))
            tid += 1
            rows.append(dict(trade_id=tid, filer_id=1, filer_name="F1",
                             filer_kind="congress", side="sell", ticker="XX",
                             transaction_date=buy_dt.normalize() + pd.Timedelta(days=5),
                             filing_datetime=buy_dt + pd.Timedelta(days=7),
                             amount_low=1000.0, amount_high=2000.0))
    return pd.DataFrame(rows), px, bench


def test_exit_on_filer_sell_shortens_holds():
    from copytrade.backtest.costs import get_cost_model
    from copytrade.backtest.engine import walk_forward

    trades, px, bench = _wf_frames()
    cost = get_cost_model("zero")
    base = walk_forward(trades, px, bench, cost, horizon=63, min_trades=5)
    early = walk_forward(trades, px, bench, cost, horizon=63, min_trades=5,
                         exit_on_filer_sell=True)
    assert base is not None and early is not None
    # the filer sells ~7 calendar days after each buy filing: copied holds
    # must be far shorter than the 63-bd backstop
    hold_base = (base.taxlots["exit_date"] - base.taxlots["entry_date"]).map(lambda d: d.days)
    hold_early = (early.taxlots["exit_date"] - early.taxlots["entry_date"]).map(lambda d: d.days)
    assert hold_early.median() < hold_base.median()
    assert hold_early.median() <= 10


def test_copy_all_needs_no_training_years():
    from copytrade.backtest.costs import get_cost_model
    from copytrade.backtest.engine import walk_forward

    trades, px, bench = _wf_frames()
    trades = trades[trades["filing_datetime"].dt.year == 2019]  # single year
    cost = get_cost_model("zero")
    assert walk_forward(trades, px, bench, cost, horizon=21, min_trades=5) is None
    wf = walk_forward(trades, px, bench, cost, horizon=21, min_trades=5, copy_all=True)
    assert wf is not None and len(wf.equity) > 0
    assert wf.picks and all(v == ["(all filers)"] for v in wf.picks.values())


def test_only_filers_mirrors_single_named_filer():
    from copytrade.backtest.costs import get_cost_model
    from copytrade.backtest.engine import walk_forward

    trades, px, bench = _wf_frames()   # one filer, "F1"
    trades = trades.copy()
    # add a second filer whose buys should be IGNORED by the mirror
    other = trades[trades["side"] == "buy"].head(3).copy()
    other["filer_id"] = 2
    other["filer_name"] = "F2"
    other["ticker"] = "XX"
    trades = pd.concat([trades, other], ignore_index=True)
    cost = get_cost_model("zero")
    wf = walk_forward(trades, px, bench, cost, horizon=21, min_trades=1,
                      only_filers={"F1"})
    assert wf is not None
    # only F1's names appear in picks; F2 excluded
    assert all(v == ["(all filers)"] for v in wf.picks.values())
    assert not wf.taxlots.empty
    assert (wf.taxlots["filer"] == "F1").all()

    assert walk_forward(trades, px, bench, cost, horizon=21, min_trades=1,
                        only_filers={"Nobody"}) is None
