"""Timescale sweep: WHERE on the horizon axis does a signal live, if anywhere?

Motivation. Earlier work fixed the holding horizon at 63 trading days and found
no significant edge. That is one cell of a two-dimensional grid: a signal could
be real but short-lived (post-filing drift that decays within days) or slow
(a tilt that only pays over a year). Testing one horizon and declaring a null
tests the horizon, not the hypothesis.

So this module sweeps the whole grid: {trade subset} x {holding horizon}, each
cell point-in-time and cost-net, reusing `run_event_study` so execution stays
on the single `execution_position` choke point.

THE CATCH, AND WHY THIS MODULE IS MOSTLY ABOUT STATISTICS. Sweeping a grid is
data dredging unless it is corrected. A 6-subset x 10-horizon grid is 60
tests; at alpha=0.05 you expect ~3 "significant" cells from pure noise, and
the largest |t| in a null grid is routinely >2. Reporting the best cell
without correction is exactly how backtests lie.

Two corrections are therefore applied to EVERY sweep:

  * Benjamini-Hochberg FDR (q-value): controls the expected fraction of false
    discoveries among the cells you call significant. This is the primary bar.
  * Bonferroni: controls the probability of ANY false positive. Very strict;
    reported as the conservative backstop.

A cell is only reported as a finding if it survives FDR. Raw |t|>2 that does
not survive is labelled as such, because in a grid this size that is the
expected behaviour of noise.

The horizon grid is deliberately split into two bands, matching the two
sleeves this strategy was always meant to have:

  SHORT (1-21 trading days), event-driven. Is there post-filing drift? This
      is the tactical sleeve: fast, high-turnover, and the band where a
      disclosure-reaction effect would have to appear if it exists at all.
  LONG (42-504 trading days), position/beta. Does copying express a durable
      tilt worth holding? This is the buy-and-hold sleeve.

The bands are analysed separately because they are different strategies with
different costs, different tax treatment (see docs/CANADA.md), and different
failure modes: pooling them hides both.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

# Trading-day horizons. Short band probes the drift window disclosure-reaction
# effects would occupy; long band probes durable tilt. 504 ~= two years.
# Tuples, not lists: these are module-level constants used as default
# arguments, and a mutable default is one in-place .sort()/.append() away
# from silently reconfiguring the grid for the whole process.
SHORT_HORIZONS = (1, 2, 3, 5, 10, 21)
LONG_HORIZONS = (42, 63, 126, 252, 504)
ALL_HORIZONS = SHORT_HORIZONS + LONG_HORIZONS


def band_of(h: int) -> str:
    """Which sleeve a horizon belongs to.

    Raises on an unknown horizon rather than defaulting to "long": a
    caller-supplied h=30 was silently filed under a band the report
    labels "LONG band (42-504d)".
    """
    if h in SHORT_HORIZONS:
        return "short"
    if h in LONG_HORIZONS:
        return "long"
    raise ValueError(
        f"horizon {h} is in neither band; add it to SHORT_HORIZONS or "
        f"LONG_HORIZONS so it is reported under an accurate label")


def benjamini_hochberg(pvals: np.ndarray) -> np.ndarray:
    """BH step-up FDR q-values, NaN-safe and order-preserving.

    q_i = min over k>=rank(i) of (p_(k) * m / k), clipped to <=1. NaNs (cells
    with too few clusters to test) are excluded from m so they neither dilute
    nor inflate the correction.
    """
    p = np.asarray(pvals, dtype=float)
    q = np.full(p.shape, np.nan)
    ok = ~np.isnan(p)
    m = int(ok.sum())
    if m == 0:
        return q
    idx = np.flatnonzero(ok)
    order = idx[np.argsort(p[idx], kind="mergesort")]
    ranked = p[order] * m / np.arange(1, m + 1)
    # step-up: enforce monotonicity from the largest p downward
    q_sorted = np.minimum.accumulate(ranked[::-1])[::-1]
    q[order] = np.clip(q_sorted, 0.0, 1.0)
    return q


def clustered_test(values: pd.Series, dates: pd.Series) -> tuple[float, float, int]:
    """(t-stat, two-sided p, n_clusters) for the mean, clustered by entry month.

    Mirrors `metrics.clustered_tstat` but also returns the p-value, which the
    multiple-testing correction needs. Clustering by month is what stops
    overlapping positions from inflating significance: trades entered in the
    same month share market conditions and are not independent draws.
    """
    df = pd.DataFrame(
        {"v": values.values, "m": pd.to_datetime(dates).dt.to_period("M")}
    ).dropna()
    if df.empty:
        return np.nan, np.nan, 0
    cm = df.groupby("m")["v"].mean()
    n = int(len(cm))
    if n < 3:  # need >=3 monthly clusters for a meaningful t
        return np.nan, np.nan, n
    t, p = stats.ttest_1samp(cm, 0.0)
    return float(t), float(p), n


def block_test(
    values: pd.Series, dates: pd.Series, horizon_days: int
) -> tuple[float, float, int]:
    """Overlap-aware version of `clustered_test`.

    WHY THIS EXISTS. Clustering by calendar month assumes months are
    independent draws. That is fine at a 5-day horizon; it is badly wrong at
    504 days, where a position opened in January and one opened in February
    share ~96% of their holding period. Monthly clustering would treat those
    as two independent observations and roughly double-count the evidence, so
    a long-horizon t-stat computed that way is inflated, sometimes enough to
    manufacture a "finding" out of one lucky two-year stretch.

    The fix is to cluster on NON-OVERLAPPING blocks as long as the holding
    period itself (horizon_days / 21 ~= months), so each block mean really is
    an independent draw. The honest cost is brutal: at a 504-day horizon a
    ten-year sample contains only ~5 independent blocks, and the test says so
    by returning a tiny cluster count rather than a confident t.

    That is the point. A long-horizon result resting on 5 independent
    observations should look weak, because it is.
    """
    block_months = max(1, int(round(horizon_days / 21)))
    df = pd.DataFrame(
        {"v": values.values, "d": pd.to_datetime(dates.values)}
    ).dropna()
    if df.empty:
        return np.nan, np.nan, 0
    months = df["d"].dt.to_period("M")
    origin = months.min()
    # integer month offset from the first observation, floored into blocks
    offset = (months - origin).apply(lambda x: x.n)
    df["block"] = offset // block_months
    bm = df.groupby("block")["v"].mean()
    n = int(len(bm))
    if n < 3:
        return np.nan, np.nan, n
    t, p = stats.ttest_1samp(bm, 0.0)
    return float(t), float(p), n


def horizon_sweep(
    per_trade: pd.DataFrame,
    subsets: dict[str, pd.Series],
    horizons: tuple[int, ...] = ALL_HORIZONS,
    min_n: int = 30,
) -> pd.DataFrame:
    """Full {subset x horizon} grid of point-in-time, cost-net excess returns.

    `subsets` maps a label to a boolean mask over `per_trade` rows, so every
    cell is computed from the same event study, one execution path, many
    views of it.

    Cells with fewer than `min_n` priced trades are computed but flagged
    `underpowered`, because a null from 12 trades is not evidence of absence.
    Returns one row per cell with the raw test plus BH-FDR and Bonferroni
    corrections applied ACROSS THE WHOLE GRID.
    """
    rows: list[dict] = []
    for label, mask in subsets.items():
        sub = per_trade[mask] if mask is not None else per_trade
        for h in horizons:
            col = f"h{h}"
            if col not in sub.columns:
                continue
            s = sub[col].dropna()
            n = int(len(s))
            if n == 0:
                rows.append({
                    "subset": label, "horizon": h, "band": band_of(h), "n": 0,
                    "mean": np.nan, "median": np.nan, "hit": np.nan,
                    "t_month": np.nan, "p_month": np.nan, "clusters": 0,
                    "t": np.nan, "p": np.nan, "blocks": 0,
                    "underpowered": True,
                })
                continue
            dates = sub.loc[s.index, "entry_date"]
            # monthly clustering: comparable with earlier work, but too
            # generous once the holding period exceeds a month
            t_m, p_m, nc = clustered_test(s, dates)
            # non-overlapping blocks as long as the horizon: the honest test
            t_b, p_b, nb = block_test(s, dates, h)
            rows.append({
                "subset": label, "horizon": h, "band": band_of(h), "n": n,
                "mean": float(s.mean()), "median": float(s.median()),
                "hit": float((s > 0).mean()),
                "t_month": t_m, "p_month": p_m, "clusters": nc,
                # `t`/`p` are the overlap-adjusted figures, everything
                # downstream (FDR, survivorship) keys off these deliberately
                "t": t_b, "p": p_b, "blocks": nb,
                "underpowered": n < min_n or nb < 5,
            })
    grid = pd.DataFrame(rows)
    if grid.empty:
        return grid
    grid["q_fdr"] = benjamini_hochberg(grid["p"].values)
    m = int(grid["p"].notna().sum())
    grid["p_bonf"] = np.clip(grid["p"] * max(m, 1), 0, 1)
    grid["n_tests"] = m
    # a finding must clear FDR *and* have enough data to mean anything
    grid["survives_fdr"] = (grid["q_fdr"] < 0.05) & (~grid["underpowered"])
    grid["raw_sig_only"] = (grid["p"] < 0.05) & (~grid["survives_fdr"])
    return grid


def band_summary(grid: pd.DataFrame) -> pd.DataFrame:
    """Per-band verdict: how many cells were tested, how many cleared raw
    significance, and how many survived correction. The gap between the last
    two columns is the data-dredging tax, if 'raw' is large and 'fdr' is 0,
    the raw hits were noise."""
    if grid.empty:
        return grid
    out = []
    for band, g in grid.groupby("band"):
        testable = g["p"].notna()
        out.append({
            "band": band,
            "cells": int(len(g)),
            "testable": int(testable.sum()),
            "raw_p<0.05": int((g["p"] < 0.05).sum()),
            "survives_fdr": int(g["survives_fdr"].sum()),
            "best_t": float(g["t"].abs().max()) if testable.any() else np.nan,
            "best_mean": float(g.loc[g["t"].abs().idxmax(), "mean"])
            if testable.any() and g["t"].notna().any() else np.nan,
        })
    return pd.DataFrame(out).sort_values("band", ascending=False)


def decay_profile(grid: pd.DataFrame, subset: str) -> pd.DataFrame:
    """Mean excess return by horizon for one subset, the information-decay
    shape. A genuine short-lived signal rises then flattens/decays; a random
    one wanders. Reading the SHAPE guards against over-reading any single
    cell's t-stat."""
    g = grid[grid["subset"] == subset].sort_values("horizon")
    return g[["horizon", "band", "n", "mean", "median", "hit", "t", "q_fdr"]]
