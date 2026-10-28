import numpy as np
import pandas as pd
import pytest

from copytrade.backtest.benchmarks import buy_and_hold, dca, sixty_forty
from copytrade.backtest.costs import CostModel, get_cost_model

ZERO = get_cost_model("zero")
CAL = pd.bdate_range("2020-01-01", "2021-12-31")


def test_buy_and_hold_is_price_ratio():
    px = pd.Series(np.linspace(100, 150, len(CAL)), index=CAL)
    eq = buy_and_hold(px, ZERO)
    assert eq.iloc[0] == pytest.approx(1.0)
    assert eq.iloc[-1] == pytest.approx(1.5)


def test_buy_and_hold_entry_cost():
    px = pd.Series(100.0, index=CAL)
    cost = CostModel(name="x", spread_slippage_bps=100, sec_fee_on_sells=False)  # 1%/side
    eq = buy_and_hold(px, cost)
    assert eq.iloc[-1] == pytest.approx(0.99)


def test_dca_flat_market_ends_fully_invested():
    px = pd.Series(100.0, index=CAL)
    eq = dca(px, ZERO, months=12)
    assert eq.iloc[-1] == pytest.approx(1.0)
    # early on most capital is idle cash
    assert eq.iloc[5] == pytest.approx(1.0)


def test_dca_lags_lump_sum_in_rising_market():
    px = pd.Series(np.linspace(100, 200, len(CAL)), index=CAL)
    lump = buy_and_hold(px, ZERO)
    drip = dca(px, ZERO, months=12)
    assert drip.iloc[-1] < lump.iloc[-1]
    assert drip.iloc[-1] > 1.0


def test_sixty_forty_between_the_legs():
    spy = pd.Series(np.linspace(100, 160, len(CAL)), index=CAL)   # +60%
    tlt = pd.Series(np.linspace(100, 110, len(CAL)), index=CAL)   # +10%
    eq = sixty_forty(spy, tlt, ZERO)
    assert 1.10 < eq.iloc[-1] < 1.60
    # roughly 0.6*60% + 0.4*10% = 40% before rebalance drift
    assert eq.iloc[-1] == pytest.approx(1.40, abs=0.06)


def test_new_perf_stats_fields():
    import numpy as np
    import pandas as pd

    from copytrade.backtest.metrics import bootstrap_outperf_pvalue, perf_stats

    rng = np.random.default_rng(7)
    idx = pd.bdate_range("2020-01-01", periods=500)
    # deterministic zig-zag with positive drift: +2%, -1%, +2%, ...
    rets = np.tile([0.02, -0.01], 250)
    up = pd.Series(np.cumprod(1 + rets), index=idx)
    s = perf_stats(up)
    assert {"sortino", "calmar"} <= set(s)
    assert s["sortino"] > s["sharpe"] > 0  # downside vol < total vol

    flat = pd.Series(rng.normal(0.0, 0.01, 500), index=idx)
    strong = flat + 0.003          # same noise, constant positive excess
    p_good = bootstrap_outperf_pvalue(strong, flat, n_boot=500)
    p_null = bootstrap_outperf_pvalue(flat, flat, n_boot=500)
    assert p_good == 0.0           # every resampled mean is exactly +0.003
    assert p_null == 1.0           # excess identically 0 -> never > 0


def test_annual_returns_chains_across_year_boundaries():
    import numpy as np
    import pandas as pd

    from copytrade.backtest.metrics import annual_returns

    # daily calendar spanning 3 years; construct known year-end levels
    idx = pd.bdate_range("2021-06-01", "2023-12-29")
    # linear ramp 100 -> 120 is fine; instead pin exact year-ends by building
    eq = pd.Series(100.0, index=idx)
    eq.loc[:"2021-12-31"] = 110.0   # 2021 ends at 110 (partial: 100->110 = +10%)
    eq.loc["2022-01-01":"2022-12-31"] = 132.0  # 2022 ends 132 (110->132 = +20%)
    eq.loc["2023-01-01":] = 118.8   # 2023 ends 118.8 (132->118.8 = -10%)
    eq.iloc[0] = 100.0
    ann = annual_returns(eq)
    assert abs(ann.loc[2021] - 0.10) < 1e-6
    assert abs(ann.loc[2022] - 0.20) < 1e-6
    assert abs(ann.loc[2023] + 0.10) < 1e-6
    assert list(ann.index) == [2021, 2022, 2023]   # no duplicate years
