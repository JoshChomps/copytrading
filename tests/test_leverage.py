import numpy as np
import pandas as pd

from copytrade.backtest.leverage import (
    leverage_scan,
    levered_path,
    levered_paths_vec,
    max_dd,
    max_safe_leverage,
)


def test_unlevered_path_reproduces_the_asset():
    r = np.array([0.01, -0.02, 0.03, -0.01])
    eq, liq = levered_path(r, leverage=1.0, financing_rate=0.0)
    assert liq is None, "an unlevered account can never be margin-called"
    assert np.isclose(eq[-1], np.prod(1 + r))


def test_daily_rebalance_scales_each_day_return():
    r = np.array([0.01, -0.02, 0.03])
    eq, _ = levered_path(r, leverage=2.0, financing_rate=0.0, maintenance=0.0,
                         rebalance="daily")
    assert np.isclose(eq[-1], np.prod(1 + 2 * r))


def test_fixed_loan_is_assets_minus_debt_not_a_scaled_return():
    # The default mode holds a fixed loan, so terminal equity is
    # L*prod(1+r) - (L-1), NOT prod(1 + L*r).
    r = np.array([0.01, -0.02, 0.03])
    eq, _ = levered_path(r, leverage=2.0, financing_rate=0.0, maintenance=0.0)
    assert np.isclose(eq[-1], 2.0 * np.prod(1 + r) - 1.0)
    assert not np.isclose(eq[-1], np.prod(1 + 2 * r))


def test_financing_is_charged_only_on_the_borrowed_portion():
    r = np.zeros(252)
    # fixed loan: the borrowed dollar compounds up at exp(+0.05)
    eq2, _ = levered_path(r, leverage=2.0, financing_rate=0.05, maintenance=0.0)
    assert np.isclose(eq2[-1], 2.0 - np.exp(0.05), rtol=1e-4)
    # 3x borrows 2x -> twice the debt
    eq3, _ = levered_path(r, leverage=3.0, financing_rate=0.05, maintenance=0.0)
    assert np.isclose(eq3[-1], 3.0 - 2.0 * np.exp(0.05), rtol=1e-4)
    # unlevered borrows nothing and pays nothing, in either mode
    for mode in ("none", "daily"):
        eq1, _ = levered_path(r, leverage=1.0, financing_rate=0.05, rebalance=mode)
        assert np.isclose(eq1[-1], 1.0)


def test_slow_grind_down_calls_a_fixed_loan_but_not_a_daily_reset():
    # THE reason the default is 'none'. A steady -1%/day bleed destroys a real
    # margin account, while daily rebalancing resets the ratio every morning
    # and never triggers. Modelling this wrong understates risk of ruin.
    r = np.full(60, -0.01)
    _, liq_fixed = levered_path(r, leverage=2.0, financing_rate=0.05,
                                maintenance=0.25, rebalance="none")
    _, liq_daily = levered_path(r, leverage=2.0, financing_rate=0.05,
                                maintenance=0.25, rebalance="daily")
    assert liq_fixed is not None, "a fixed loan must be called by a sustained decline"
    assert liq_daily is None, "daily reset never sees the cumulative damage"


def test_margin_call_fires_on_a_big_drop_and_is_absorbing():
    # At 4x, a -20% day wipes out ~80% of equity -> ratio collapses
    r = np.array([-0.20, 0.50, 0.50])
    eq, liq = levered_path(r, leverage=4.0, financing_rate=0.0, maintenance=0.25)
    assert liq == 0, "liquidation must fire on the first breaching day"
    # the later +50% days must NOT rescue the account
    assert eq[-1] == eq[1]


def test_higher_leverage_calls_sooner():
    r = np.array([-0.12, -0.05, 0.02])
    _, liq2 = levered_path(r, leverage=2.0, financing_rate=0.0, maintenance=0.25)
    _, liq4 = levered_path(r, leverage=4.0, financing_rate=0.0, maintenance=0.25)
    assert liq4 is not None
    assert liq2 is None or liq2 >= liq4


def test_vectorised_matches_the_scalar_implementation():
    rng = np.random.default_rng(0)
    paths = rng.normal(0.0004, 0.012, size=(40, 500))
    for L in (1.0, 1.5, 2.0, 3.0):
        term, dds, breached = levered_paths_vec(paths, L, 0.05, 0.25)
        for j in range(0, 40, 7):
            eq, liq = levered_path(paths[j], L, 0.05, 0.25)
            assert np.isclose(term[j], eq[-1], rtol=1e-9), f"L={L} path={j}"
            assert np.isclose(dds[j], max_dd(eq), rtol=1e-9)
            assert bool(breached[j]) == (liq is not None)


def _series(n=1500, mu=0.0004, sd=0.011, seed=3):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=n)
    return pd.Series(rng.normal(mu, sd, n), index=idx)


def test_scan_margin_call_probability_increases_with_leverage():
    scan = leverage_scan(_series(), leverages=(1.0, 2.0, 3.0, 4.0),
                         n_paths=300, seed=1)
    p = scan["margin_call_prob"].to_numpy()
    assert p[0] == 0.0, "unlevered can never be called"
    assert np.all(np.diff(p) >= -1e-12), "risk of ruin must be monotone in leverage"


def test_scan_drawdown_deepens_with_leverage():
    scan = leverage_scan(_series(), leverages=(1.0, 1.5, 2.0), n_paths=300, seed=2)
    dd = scan["boot_median_max_dd"].to_numpy()
    assert np.all(np.diff(dd) <= 1e-12), "deeper drawdown at higher leverage"


def test_max_safe_leverage_respects_the_tolerance():
    scan = leverage_scan(_series(sd=0.02), leverages=(1.0, 2.0, 3.0, 4.0),
                         n_paths=300, seed=4)
    safe = max_safe_leverage(scan, tolerance=0.05)
    called = scan.loc[scan["leverage"] == safe, "margin_call_prob"].iloc[0]
    assert called <= 0.05
    higher = scan[scan["leverage"] > safe]
    assert (higher["margin_call_prob"] > 0.05).all()


def test_a_calmer_asset_supports_more_leverage():
    # The core claim behind the regime filter: lower drawdown -> more leverage
    # survivable at the same risk of ruin.
    calm = leverage_scan(_series(sd=0.007, seed=5), leverages=(1.0, 2.0, 3.0),
                         n_paths=300, seed=5)
    wild = leverage_scan(_series(sd=0.020, seed=5), leverages=(1.0, 2.0, 3.0),
                         n_paths=300, seed=5)
    assert max_safe_leverage(calm) >= max_safe_leverage(wild)


def test_short_series_returns_empty_rather_than_guessing():
    assert leverage_scan(_series(n=50)).empty
