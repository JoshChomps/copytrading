import numpy as np
import pandas as pd

from copytrade.rank.timescale import (
    ALL_HORIZONS,
    band_of,
    band_summary,
    benjamini_hochberg,
    clustered_test,
    horizon_sweep,
)


def test_bh_is_monotone_and_never_below_raw_p():
    # BH q-values must be non-decreasing in p and always >= the raw p-value
    p = np.array([0.001, 0.01, 0.03, 0.2, 0.5, 0.9])
    q = benjamini_hochberg(p)
    assert np.all(np.diff(q) >= -1e-12), "q must be monotone in p"
    assert np.all(q >= p - 1e-12), "correction can only make p less significant"
    assert np.all(q <= 1.0)


def test_bh_kills_a_lone_marginal_hit():
    # THE core guard: one p=0.03 among 20 tests is what a null grid produces.
    # It must NOT survive FDR, this is the bug the old single-horizon
    # analysis could not have caught.
    p = np.concatenate([[0.03], np.linspace(0.2, 0.99, 19)])
    q = benjamini_hochberg(p)
    assert q[0] > 0.05, "a marginal raw hit in a 20-test grid must not survive FDR"


def test_bh_keeps_a_genuinely_strong_hit():
    p = np.concatenate([[1e-6], np.linspace(0.2, 0.99, 19)])
    q = benjamini_hochberg(p)
    assert q[0] < 0.05, "a strong effect must still survive correction"


def test_bh_ignores_nan_cells_in_the_test_count():
    # NaN cells (too few clusters to test) must not inflate m and over-correct
    p_clean = np.array([0.01, 0.4, 0.6])
    p_padded = np.array([0.01, 0.4, 0.6, np.nan, np.nan])
    q_clean = benjamini_hochberg(p_clean)
    q_padded = benjamini_hochberg(p_padded)
    assert np.allclose(q_clean, q_padded[:3], equal_nan=True)
    assert np.isnan(q_padded[3:]).all()


def test_band_split_covers_every_horizon():
    assert {band_of(h) for h in ALL_HORIZONS} == {"short", "long"}
    assert band_of(1) == "short" and band_of(21) == "short"
    assert band_of(42) == "long" and band_of(504) == "long"


def test_clustered_test_needs_three_clusters():
    # two monthly clusters is not enough to test a mean
    vals = pd.Series([0.01, 0.02])
    dates = pd.Series(pd.to_datetime(["2024-01-05", "2024-02-05"]))
    t, p, n = clustered_test(vals, dates)
    assert np.isnan(t) and np.isnan(p) and n == 2


def test_clustered_test_clusters_by_month_not_by_trade():
    # 30 trades in ONE month must not look like 30 independent observations.
    # With a single cluster the test is undefined, which is the point:
    # same-month overlapping bets cannot manufacture significance.
    vals = pd.Series([0.05] * 30)
    dates = pd.Series(pd.to_datetime(["2024-01-10"] * 30))
    t, p, n = clustered_test(vals, dates)
    assert n == 1 and np.isnan(t)


def _per_trade(n_per_month=4, months=24, effect=0.0, seed=0):
    """Synthetic per-trade frame shaped like run_event_study's output."""
    rng = np.random.default_rng(seed)
    dates, rows = [], []
    for m in range(months):
        d = pd.Timestamp("2020-01-15") + pd.DateOffset(months=m)
        for _ in range(n_per_month):
            dates.append(d)
    df = pd.DataFrame({"entry_date": dates})
    df["amount"] = 10000.0
    for h in ALL_HORIZONS:
        df[f"h{h}"] = rng.normal(effect, 0.08, len(df))
    return df


def test_sweep_on_pure_noise_finds_no_survivor():
    # A null dataset swept across the full grid must yield zero FDR survivors.
    pt = _per_trade(effect=0.0, seed=42)
    subsets = {"ALL buys": pd.Series(True, index=pt.index)}
    grid = horizon_sweep(pt, subsets, ALL_HORIZONS, min_n=10)
    assert not grid.empty
    assert grid["survives_fdr"].sum() == 0, "noise must not produce a finding"
    assert (grid["n_tests"] == grid["p"].notna().sum()).all()


def test_sweep_flags_underpowered_cells():
    pt = _per_trade(n_per_month=1, months=6, seed=1)
    subsets = {"ALL buys": pd.Series(True, index=pt.index)}
    grid = horizon_sweep(pt, subsets, ALL_HORIZONS, min_n=30)
    assert grid["underpowered"].all(), "6 trades/cell must be flagged underpowered"
    # underpowered cells can never be reported as findings
    assert grid["survives_fdr"].sum() == 0


def test_sweep_detects_a_real_planted_effect():
    # Sanity in the other direction: a large, consistent effect must be found,
    # otherwise the test is just always saying "no".
    pt = _per_trade(n_per_month=8, months=36, effect=0.05, seed=7)
    subsets = {"ALL buys": pd.Series(True, index=pt.index)}
    grid = horizon_sweep(pt, subsets, ALL_HORIZONS, min_n=30)
    assert grid["survives_fdr"].sum() > 0, "a real 5% effect must survive FDR"


def test_band_summary_reports_both_bands():
    pt = _per_trade(n_per_month=6, months=30, seed=3)
    subsets = {"ALL buys": pd.Series(True, index=pt.index)}
    grid = horizon_sweep(pt, subsets, ALL_HORIZONS, min_n=30)
    bs = band_summary(grid)
    assert set(bs["band"]) == {"short", "long"}
    assert (bs["survives_fdr"] <= bs["raw_p<0.05"]).all()


def test_block_test_collapses_clusters_at_long_horizons():
    from copytrade.rank.timescale import block_test, clustered_test

    # 60 months of data, one observation per month
    dates = pd.Series(pd.date_range("2015-01-15", periods=60, freq="MS"))
    vals = pd.Series(np.linspace(0.01, 0.02, 60))
    _, _, n_month = clustered_test(vals, dates)
    _, _, n_block = block_test(vals, dates, horizon_days=504)  # ~24-month blocks
    assert n_month == 60
    assert n_block <= 3, "a 2-year horizon over 5 years leaves ~2-3 independent blocks"


def test_block_test_matches_monthly_clustering_at_short_horizons():
    from copytrade.rank.timescale import block_test, clustered_test

    dates = pd.Series(pd.date_range("2015-01-15", periods=48, freq="MS"))
    vals = pd.Series(np.linspace(-0.01, 0.03, 48))
    t_m, _, n_m = clustered_test(vals, dates)
    t_b, _, n_b = block_test(vals, dates, horizon_days=21)  # 1-month blocks
    assert n_b == n_m and np.isclose(t_b, t_m)


def test_overlap_adjustment_flags_long_horizon_cells_as_underpowered():
    # Long-horizon cells backed by too few independent blocks must never be
    # reported as findings, however large their monthly-clustered t-stat.
    pt = _per_trade(n_per_month=20, months=30, effect=0.05, seed=5)
    subsets = {"ALL buys": pd.Series(True, index=pt.index)}
    grid = horizon_sweep(pt, subsets, [504], min_n=10)
    row = grid.iloc[0]
    assert row["blocks"] < 5
    assert row["underpowered"]
    assert not row["survives_fdr"], "too few independent blocks cannot be a finding"
