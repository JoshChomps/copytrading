"""Who to copy: tested the only way that can't fool you: OUT OF SAMPLE.

The earlier finding was that copying the top-N "best" filers badly underperformed copying everyone (+101% vs +212% on the corrected
sample). That is the signature of selection on noise: rank
filers by realised performance, and with ~100 filers the top of the list is
dominated by whoever got lucky, not whoever is skilled. Their luck does not
repeat, so the "best" basket underperforms the naive one.

But "in-sample top-N fails" does not answer the actual question. It only shows
that RANKING BY PAST RETURN AND KEEPING THE WINNERS is not a strategy. The
question underneath is:

    Is there ANY persistent skill differential between filers at all?

That is a testable property of the data, independent of any particular
selection rule: does a filer's edge in one period predict their edge in the
next? This module tests exactly that, and nothing else.

METHOD. Chop history into consecutive non-overlapping windows. For each
adjacent (train, test) pair, rank filers by mean excess return in train, then
observe their mean excess in test. Two readouts:

  1. Rank correlation (Spearman) between train rank and test performance,
     computed per window pair and averaged. Zero => no persistence => no
     selection rule based on past performance can work, and copy-everyone is
     the correct design.
  2. Tercile spread: mean test-window excess of the top third minus the bottom
     third, by train rank. This is the tradeable version of the same question,
     and its t-stat (across window pairs) is the significance bar.

Both are strictly out-of-sample by construction: the ranking uses only data
from before the test window, so there is no look-ahead. Filers must clear
`min_trades` in BOTH windows to enter a pair, otherwise the ranking is noise
built on two or three trades.

WHAT A NULL HERE MEANS. If persistence is zero, that is a genuinely useful,
strongly-actionable result: it says stop hunting for whom to copy, because the
data contains no stable who. It redirects effort to WHEN and HOW LONG (see
`timescale.py`) and to risk management (`regime.py`), which is where the
remaining leverage actually is.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def _window_key(dates: pd.Series, freq: str) -> pd.Series:
    """Map entry dates to a window label. 'Y' = calendar year, '2Q'/'Q' etc.
    accepted for finer slicing when the sample supports it."""
    d = pd.to_datetime(dates)
    if freq.upper() in ("Y", "A"):
        return d.dt.year.astype(str)
    return d.dt.to_period(freq).astype(str)


def filer_window_table(
    per_trade: pd.DataFrame, hcol: str, freq: str = "Y", min_trades: int = 5
) -> pd.DataFrame:
    """Mean excess return per (filer, window), restricted to filers with at
    least `min_trades` priced trades in that window."""
    df = per_trade[["filer_name", "entry_date", hcol]].dropna(subset=[hcol]).copy()
    if df.empty:
        return pd.DataFrame(columns=["filer_name", "window", "mean_excess", "n"])
    df["window"] = _window_key(df["entry_date"], freq)
    g = df.groupby(["filer_name", "window"])[hcol].agg(["mean", "count"])
    g = g[g["count"] >= min_trades].reset_index()
    return g.rename(columns={"mean": "mean_excess", "count": "n"})


def persistence_pairs(table: pd.DataFrame) -> list[tuple[str, str, pd.DataFrame]]:
    """Adjacent (train_window, test_window, merged_filers) triples.

    Only filers present in BOTH windows are kept, the question is whether a
    given filer's edge carries forward, which is undefined for someone who
    stopped trading.
    """
    windows = sorted(table["window"].unique())
    # `windows` only contains periods where SOME filer cleared min_trades, so
    # list-adjacency is not calendar-adjacency: with --freq Q a sparse quarter
    # drops out entirely and ("2018Q2", "2018Q4") would be reported as a
    # train -> NEXT-period test across a quarter-long gap. The question this
    # module asks is specifically about the *next* period, so a gap changes the
    # estimand and must be skipped rather than silently relabelled.
    def _adjacent(a: str, b: str) -> bool:
        try:
            return (pd.Period(b) - pd.Period(a)).n == 1
        except Exception:  # noqa: BLE001, plain-year labels aren't Periods
            try:
                return int(b) - int(a) == 1
            except ValueError:
                return True          # unknown labelling: don't silently drop
    out = []
    for a, b in zip(windows, windows[1:]):
        if not _adjacent(a, b):
            continue
        tr = table[table["window"] == a][["filer_name", "mean_excess", "n"]]
        te = table[table["window"] == b][["filer_name", "mean_excess", "n"]]
        m = tr.merge(te, on="filer_name", suffixes=("_train", "_test"))
        if len(m) >= 5:  # need enough filers to rank meaningfully
            out.append((a, b, m))
    return out


def persistence_report(
    per_trade: pd.DataFrame, hcol: str, freq: str = "Y", min_trades: int = 5
) -> dict:
    """Full out-of-sample persistence test.

    Returns per-pair detail plus the pooled verdict: mean Spearman rho, the
    tercile spread and its t-stat across window pairs, and the count of pairs
    where the spread was positive (a sign test that doesn't assume normality).
    """
    table = filer_window_table(per_trade, hcol, freq, min_trades)
    pairs = persistence_pairs(table)
    rows: list[dict] = []
    for a, b, m in pairs:
        rho, p = stats.spearmanr(m["mean_excess_train"], m["mean_excess_test"])
        # tercile spread: rank on TRAIN only, measure on TEST only
        k = max(1, len(m) // 3)
        ranked = m.sort_values("mean_excess_train", ascending=False)
        top = ranked.head(k)["mean_excess_test"].mean()
        bot = ranked.tail(k)["mean_excess_test"].mean()
        rows.append({
            "train": a, "test": b, "filers": int(len(m)),
            "spearman": float(rho) if rho == rho else np.nan,
            "p_spearman": float(p) if p == p else np.nan,
            "top_tercile": float(top), "bottom_tercile": float(bot),
            "spread": float(top - bot),
        })
    detail = pd.DataFrame(rows)
    if detail.empty:
        return {"detail": detail, "pairs": 0, "verdict": "insufficient data"}

    spreads = detail["spread"].dropna()
    rhos = detail["spearman"].dropna()
    t_spread, p_spread = (stats.ttest_1samp(spreads, 0.0)
                          if len(spreads) >= 3 else (np.nan, np.nan))
    t_rho, p_rho = (stats.ttest_1samp(rhos, 0.0)
                    if len(rhos) >= 3 else (np.nan, np.nan))
    return {
        "detail": detail,
        "pairs": int(len(detail)),
        "mean_spearman": float(rhos.mean()) if len(rhos) else np.nan,
        "t_spearman": float(t_rho) if t_rho == t_rho else np.nan,
        "p_spearman": float(p_rho) if p_rho == p_rho else np.nan,
        "mean_spread": float(spreads.mean()) if len(spreads) else np.nan,
        "t_spread": float(t_spread) if t_spread == t_spread else np.nan,
        "p_spread": float(p_spread) if p_spread == p_spread else np.nan,
        "pairs_positive": int((spreads > 0).sum()),
        "sign_p": float(stats.binomtest(int((spreads > 0).sum()), len(spreads), 0.5).pvalue)
        if len(spreads) else np.nan,
    }


def filer_leaderboard(
    per_trade: pd.DataFrame, hcol: str, min_trades: int = 20
) -> pd.DataFrame:
    """Full-sample per-filer stats.

    DELIBERATELY NOT A BUY LIST. It is displayed only so the persistence result
    can be interpreted: if persistence is zero, the ordering here is a
    ranking of luck, and acting on it is the documented way to turn the
    copy-everyone basket (+212%) into the top-N basket (+101%).
    Consumers should read `persistence_report` first.
    """
    df = per_trade[["filer_name", "entry_date", hcol]].dropna(subset=[hcol])
    g = df.groupby("filer_name")[hcol].agg(["count", "mean", "median"])
    g = g[g["count"] >= min_trades]
    if g.empty:
        return g
    from copytrade.backtest.metrics import clustered_tstat

    ts = {}
    for name, sub in df[df["filer_name"].isin(g.index)].groupby("filer_name"):
        t, _ = clustered_tstat(sub[hcol], sub["entry_date"])
        ts[name] = t
    g["t"] = pd.Series(ts)
    return g.sort_values("mean", ascending=False)
