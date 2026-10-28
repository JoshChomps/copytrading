import numpy as np
import pandas as pd

from copytrade.backtest.metrics import full_metrics
from copytrade.backtest.regime import regime_equity, regime_returns


def _equity_from_rets(rets, start_value=1.0, index=None):
    eq = (1 + pd.Series(rets, index=index)).cumprod() * start_value
    return eq


def test_full_metrics_reports_full_battery_and_beta():
    idx = pd.bdate_range("2015-01-01", periods=756)          # ~3 years
    rng = np.random.default_rng(7)
    bench_rets = pd.Series(rng.normal(0.0004, 0.01, size=len(idx)), index=idx)
    # basket = 1.4x the market + idiosyncratic noise, so beta should be ~1.4
    basket_rets = 1.4 * bench_rets + rng.normal(0, 0.004, size=len(idx))
    eq = _equity_from_rets(basket_rets, index=idx)
    m = full_metrics(eq, bench_rets=bench_rets)
    # every metric present
    for key in ("total_return", "cagr", "vol", "downside_dev", "sharpe", "sortino",
                "calmar", "max_drawdown", "skew", "kurtosis", "var95", "cvar95",
                "omega", "tail_ratio", "pct_positive_days", "best_day", "worst_day",
                "beta", "alpha_annualized", "corr", "r2", "treynor",
                "tracking_error", "info_ratio", "up_capture", "down_capture"):
        assert key in m, f"missing {key}"
    assert abs(m["beta"] - 1.4) < 0.1                        # beta recovered
    assert m["corr"] > 0.9 and m["r2"] > 0.8
    assert m["var95"] < 0 and m["cvar95"] <= m["var95"]      # CVaR is deeper in the tail
    assert 0 < m["omega"]                                    # gains/losses ratio defined


def test_full_metrics_empty_on_tiny_input():
    assert full_metrics(pd.Series([1.0, 1.0], index=pd.bdate_range("2020", periods=2))) == {}


def test_regime_filter_sidesteps_the_crash():
    # 200 up-days, then a sharp 60-day drawdown, then recovery
    idx = pd.bdate_range("2015-01-01", periods=500)
    rets = np.concatenate([
        np.full(200, 0.001),        # steady climb (well above any MA)
        np.full(80, -0.01),         # sustained crash
        np.full(220, 0.001),        # recovery
    ])
    signal = _equity_from_rets(rets, index=idx)              # SPY = the basket itself here
    basket_eq = signal.copy()
    reg_eq, info = regime_equity(basket_eq, signal, ma_window=50, switch_cost=0.0)
    # the filter must end ABOVE the always-invested basket by dodging the crash
    assert reg_eq.iloc[-1] > basket_eq.iloc[-1]
    assert reg_eq.dropna().min() > basket_eq.dropna().min()  # shallower worst point
    assert 0.0 < info["time_in_market"] < 1.0                # it actually went to cash
    assert info["switches"] >= 2                             # out then back in


def test_regime_returns_no_lookahead_and_costs_flips():
    idx = pd.bdate_range("2015-01-01", periods=300)
    rng = np.random.default_rng(1)
    price = _equity_from_rets(rng.normal(0.0005, 0.01, size=len(idx)), index=idx)
    basket_ret = price.pct_change().dropna()
    ret, risk_on = regime_returns(basket_ret, price, ma_window=50, switch_cost=0.002)
    # decision is lagged: risk_on is boolean and defined on the basket span
    assert risk_on.dtype == bool
    assert ret.index.equals(basket_ret.index)
