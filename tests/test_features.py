import numpy as np
import pandas as pd

from copytrade.rank.features import (
    _monthly_ic,
    build_control,
    compare_features,
    forward_excess,
    information_coefficient,
    price_features,
    quintile_spread,
)


def _bench_arrays(n=400, seed=0):
    rng = np.random.default_rng(seed)
    bv = 100 * np.cumprod(1 + rng.normal(0.0003, 0.01, n))
    s = pd.Series(bv)
    return bv, s.rolling(200, min_periods=200).mean().values, \
        (s.pct_change().rolling(63, min_periods=20).std().values * np.sqrt(252))


def test_features_never_see_the_execution_bar_or_later():
    """THE look-ahead guard. Features are built at pos-1, so mutating every
    price from the execution bar onward must not change a single feature."""
    bv, bma, bvol = _bench_arrays()
    rng = np.random.default_rng(1)
    vals = 50 * np.cumprod(1 + rng.normal(0.0004, 0.012, 400))
    pos = 300

    before = price_features(vals, pos, bv, bma, bvol)
    poisoned = vals.copy()
    poisoned[pos:] *= 100.0          # catastrophic future move
    after = price_features(poisoned, pos, bv, bma, bvol)

    for k in before:
        a, b = before[k], after[k]
        assert (np.isnan(a) and np.isnan(b)) or np.isclose(a, b), (
            f"feature {k} changed when only FUTURE prices moved -> look-ahead"
        )


def test_features_do_change_when_the_past_changes():
    # sanity in the other direction, so the guard above isn't vacuous
    bv, bma, bvol = _bench_arrays()
    rng = np.random.default_rng(2)
    vals = 50 * np.cumprod(1 + rng.normal(0.0004, 0.012, 400))
    pos = 300
    before = price_features(vals, pos, bv, bma, bvol)
    changed = vals.copy()
    changed[:pos - 1] *= 0.5
    after = price_features(changed, pos, bv, bma, bvol)
    assert not np.isclose(before["mom_21"], after["mom_21"])


def test_forward_excess_is_return_minus_benchmark_minus_cost():
    vals = np.array([10.0, 11.0, 12.0])
    bench = np.array([100.0, 100.0, 105.0])
    got = forward_excess(vals, bench, 0, 2, cost=0.01)
    assert np.isclose(got, (12 / 10 - 1) - (105 / 100 - 1) - 0.01)


def test_forward_excess_is_nan_past_the_end():
    vals = np.array([10.0, 11.0])
    bench = np.array([100.0, 101.0])
    assert np.isnan(forward_excess(vals, bench, 0, 10, 0.0))


def _panel(n_months=48, per_month=40, effect=0.0, seed=0):
    """Synthetic disclosed-style panel with a controllable feature->return link."""
    rng = np.random.default_rng(seed)
    rows = []
    for m in range(n_months):
        d = pd.Timestamp("2018-01-15") + pd.DateOffset(months=m)
        x = rng.normal(0, 1, per_month)
        y = effect * x + rng.normal(0, 1, per_month)
        for xi, yi in zip(x, y):
            rows.append({"feat": xi, "fwd": yi, "entry_date": d})
    return pd.DataFrame(rows)


def test_ic_is_null_on_noise():
    df = _panel(effect=0.0, seed=5)
    r = information_coefficient(df, "feat")
    assert abs(r["ic"]) < 0.06
    assert abs(r["t"]) < 2


def test_ic_detects_a_planted_relationship():
    df = _panel(effect=0.8, seed=6)
    r = information_coefficient(df, "feat")
    assert r["ic"] > 0.4
    assert r["t"] > 5


def test_ic_handles_binary_features():
    """Regression: a >=3-distinct-values guard silently excluded every binary
    feature (market regime, repeat-buy, chamber), reporting them as 0 months."""
    rng = np.random.default_rng(7)
    rows = []
    for m in range(36):
        d = pd.Timestamp("2019-01-15") + pd.DateOffset(months=m)
        flag = rng.integers(0, 2, 30)
        fwd = 0.5 * flag + rng.normal(0, 1, 30)
        for f_, y in zip(flag, fwd):
            rows.append({"flag": float(f_), "fwd": y, "entry_date": d})
    df = pd.DataFrame(rows)
    r = information_coefficient(df, "flag")
    assert r["months"] > 20, "binary features must not be silently dropped"
    assert r["ic"] > 0.1 and r["t"] > 2


def test_ic_returns_nan_for_a_constant_feature():
    df = _panel(seed=8).assign(feat=1.0)
    r = information_coefficient(df, "feat")
    assert np.isnan(r["ic"]) and r["months"] == 0


def test_control_is_drawn_on_the_same_dates():
    cal = pd.bdate_range("2016-01-01", periods=600)
    bench = pd.Series(100 * np.cumprod(1 + np.random.default_rng(3).normal(
        0.0003, 0.01, len(cal))), index=cal)
    rng = np.random.default_rng(4)
    prices = pd.DataFrame(
        {f"T{i}": 20 * np.cumprod(1 + rng.normal(0.0003, 0.015, len(cal)))
         for i in range(12)}, index=cal)
    disclosed = pd.DataFrame({"entry_date": cal[250:260]})
    ctrl = build_control(disclosed, prices, bench, horizon=21, cost=0.003,
                         per_date=3, seed=0)
    assert not ctrl.empty
    assert set(ctrl["entry_date"]) <= set(disclosed["entry_date"])
    assert ctrl.groupby("entry_date").size().max() <= 3


def test_compare_applies_fdr_and_flags_nothing_on_matched_noise():
    """When disclosed and control share the same data-generating process, no
    feature may be labelled disclosure-specific."""
    a = _panel(effect=0.6, seed=11).rename(columns={"feat": "mom_21"})
    b = _panel(effect=0.6, seed=12).rename(columns={"feat": "mom_21"})
    out = compare_features(a, b, ["mom_21"])
    assert not out.empty
    # strongly predictive in BOTH -> not disclosure-specific
    assert out["ic_disclosed"].iloc[0] > 0.3
    assert out["ic_control"].iloc[0] > 0.3
    assert not out["disclosure_specific"].iloc[0]


def test_compare_flags_a_feature_that_only_works_in_the_disclosed_sample():
    a = _panel(effect=1.5, seed=13).rename(columns={"feat": "mom_21"})
    b = _panel(effect=0.0, seed=14).rename(columns={"feat": "mom_21"})
    out = compare_features(a, b, ["mom_21"])
    assert out["disclosure_specific"].iloc[0], (
        "a feature predictive ONLY among disclosed trades must be flagged"
    )


def test_quintile_spread_is_monotone_for_a_strong_feature():
    df = _panel(effect=1.5, seed=15).rename(columns={"feat": "mom_21"})
    g = quintile_spread(df, "mom_21")
    assert len(g) == 5
    assert g["mean"].is_monotonic_increasing
    assert g.attrs["spread"] > 0


def test_monthly_ic_returns_one_row_per_usable_month():
    df = _panel(n_months=24, seed=16)
    m = _monthly_ic(df, "feat")
    assert len(m) == 24


# --------------------------------------------------- review fixes (2026-07-25)

def _overlapping_panel(n_months=120, per_month=30, seed=0):
    """Monthly ICs whose forward windows overlap, as they do at h=63."""
    rng = np.random.default_rng(seed)
    rows = []
    for m in range(n_months):
        d = pd.Timestamp("2016-01-15") + pd.DateOffset(months=m)
        x = rng.normal(0, 1, per_month)
        rows += [{"feat": xi, "fwd": yi, "entry_date": d}
                 for xi, yi in zip(x, rng.normal(0, 1, per_month))]
    return pd.DataFrame(rows)


def test_ic_t_uses_non_overlapping_blocks_at_long_horizons():
    """At h=63 adjacent monthly ICs share ~2/3 of their outcome window. Testing
    them as independent gave P(|t|>2) = 27.5% on a pure null against a nominal
    5%. Blocking must collapse the effective sample."""
    df = _overlapping_panel()
    short = information_coefficient(df, "feat", horizon=21)
    long_ = information_coefficient(df, "feat", horizon=252)
    assert short["blocks"] > long_["blocks"], "longer horizon => fewer blocks"
    assert long_["blocks"] <= 12, "a 252d horizon over 10y leaves ~10 blocks"


def test_control_only_predictor_is_not_flagged_disclosure_specific():
    """Sign guard. p_diff is TWO-SIDED, so without it a feature that predicts
    returns only in the CONTROL, the strongest evidence it is a generic
    factor: was printed in green as a discovery."""
    disclosed = _panel(effect=0.0, seed=21).rename(columns={"feat": "mom_21"})
    control = _panel(effect=1.5, seed=22).rename(columns={"feat": "mom_21"})
    out = compare_features(disclosed, control, ["mom_21"], horizon=21)
    assert out["ic_diff"].iloc[0] < 0, "control predicts, disclosed does not"
    assert not out["disclosure_specific"].iloc[0]


def test_trade_only_features_are_marked_uncontrollable():
    """A control cannot exist for a filing lag, a random stock has none. Those
    rows must be flagged, not silently reported as 'tested and failed'."""
    disclosed = _panel(effect=1.0, seed=23).rename(columns={"feat": "lag_days"})
    control = disclosed.copy()
    control["lag_days"] = np.nan
    out = compare_features(disclosed, control, ["lag_days"], horizon=21)
    assert not out["controllable"].iloc[0]
    assert not out["disclosure_specific"].iloc[0]


def test_fdr_correction_is_exercised_across_a_real_grid():
    """Single-feature grids make m=1 and q==p, so no correction is tested."""
    rng = np.random.default_rng(31)
    feats = [f"f{i}" for i in range(12)]
    disclosed = _panel(effect=0.0, seed=32)
    control = _panel(effect=0.0, seed=33)
    for f in feats:
        disclosed[f] = rng.normal(0, 1, len(disclosed))
        control[f] = rng.normal(0, 1, len(control))
    out = compare_features(disclosed, control, feats, horizon=21)
    tested = out.dropna(subset=["p_diff", "q_fdr"])
    assert len(tested) >= 10
    assert (tested["q_fdr"] >= tested["p_diff"] - 1e-12).all()
    assert tested["q_fdr"].gt(tested["p_diff"]).any(), "m>1 must actually correct"
    assert out["disclosure_specific"].sum() == 0, "pure noise yields no winner"


def test_quintile_spread_bails_when_buckets_collapse():
    """qcut drops duplicate EDGES, not duplicate values: a ~97%-zero feature
    has nunique()>=5 while all five edges are 0, collapsing to one bucket and
    reporting spread 0.0 with a t built from float residuals."""
    rng = np.random.default_rng(41)
    n = 400
    feat = np.zeros(n)
    feat[:12] = rng.integers(1, 9, 12)          # thin tail, nunique >= 5
    df = pd.DataFrame({
        "n_filers_30d": feat,
        "fwd": rng.normal(0, 0.05, n),
        "entry_date": pd.Timestamp("2020-01-15") + pd.to_timedelta(
            rng.integers(0, 900, n), unit="D"),
    })
    assert df["n_filers_30d"].nunique() >= 5
    assert quintile_spread(df, "n_filers_30d").empty


def test_quintile_t_measures_the_spread_not_the_return_level():
    """Both buckets drawn from an identical distribution => true spread is 0.
    Pooling top with negated bottom returns leaked the sample MEAN into the
    statistic (reproduced at t=-3.61 on a zero spread)."""
    rng = np.random.default_rng(43)
    rows = []
    for m in range(60):
        d = pd.Timestamp("2018-01-15") + pd.DateOffset(months=m)
        # feature is market-wide: a whole month sits in one bucket
        level = rng.normal(-0.03, 0.01)
        f = float(m)
        rows += [{"feat": f, "fwd": level + rng.normal(0, 0.02),
                  "entry_date": d} for _ in range(30)]
    df = pd.DataFrame(rows)
    g = quintile_spread(df, "feat")
    assert not g.empty
    # For a MARKET-WIDE feature no month contains both the top and bottom
    # bucket, so a within-month paired spread is genuinely undefined and must
    # report NaN. The old pooled statistic instead leaked the return level and
    # reported t = -3.61 on a true spread of exactly zero.
    assert np.isnan(g.attrs["t"])
    assert g.attrs["paired_months"] == 0


def test_quintile_t_is_defined_for_a_cross_sectional_feature():
    """When both buckets appear within the same month the paired t exists
    so the NaN above is a property of market-wide features, not a dead code
    path."""
    rng = np.random.default_rng(45)
    rows = []
    for m in range(48):
        d = pd.Timestamp("2018-01-15") + pd.DateOffset(months=m)
        x = rng.normal(0, 1, 40)
        rows += [{"feat": xi, "fwd": 0.02 * xi + rng.normal(0, 0.01),
                  "entry_date": d} for xi in x]
    g = quintile_spread(pd.DataFrame(rows), "feat")
    assert not g.empty
    assert g.attrs["paired_months"] >= 40
    assert g.attrs["t"] > 5, "a real cross-sectional spread must be detected"
