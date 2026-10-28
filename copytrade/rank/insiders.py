"""Corporate-insider (Form 4) signal tests, the fast disclosure channel.

Insider *buys* (transaction code P) are the informative minority of Form 4
activity, and two subsets have documented predictive power:

  * CLUSTER buys: several DISTINCT insiders of the same issuer buying within a
    short window. Consensus/conviction beats a lone buyer (Lakonishok & Lee).
  * OPPORTUNISTIC insiders: Cohen, Malloy & Pomorski (2012) "Decoding Inside
    Information": an insider who trades the *same calendar month* year after
    year is a ROUTINE trader (liquidity/comp-driven, uninformative); everyone
    else is OPPORTUNISTIC, and only opportunistic trades predict returns.

These are pure functions over the trades frame; the CLI scores each subset's
forward excess return vs SPY, point-in-time (filing-anchored), net of costs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def cluster_buy_ids(buys: pd.DataFrame, window_days: int = 7,
                    min_insiders: int = 3) -> set:
    """Trade_ids of buys that are part of a cluster: >= min_insiders DISTINCT
    insiders buying the SAME ticker within a trailing window_days of this
    buy's filing. O(n^2) within each ticker group (groups are small)."""
    out: set = set()
    if buys.empty:
        return out
    b = buys.dropna(subset=["filing_datetime", "ticker", "filer_id", "trade_id"])
    win = np.timedelta64(window_days, "D")
    for _, g in b.groupby("ticker"):
        g = g.sort_values("filing_datetime")
        times = g["filing_datetime"].values
        fids = g["filer_id"].values
        tids = g["trade_id"].values
        for i in range(len(g)):
            mask = (times <= times[i]) & (times > times[i] - win)
            if len(set(fids[mask])) >= min_insiders:
                out.add(int(tids[i]))
    return out


def routine_filer_ids(trades: pd.DataFrame, min_years: int = 3) -> set:
    """Cohen-Malloy ROUTINE filers: those with any calendar month traded in
    >= min_years distinct years. Their trades are uninformative; all other
    (OPPORTUNISTIC) filers are where the signal lives."""
    if trades.empty:
        return set()
    t = trades.dropna(subset=["filer_id", "transaction_date"]).copy()
    t["m"] = t["transaction_date"].dt.month
    t["y"] = t["transaction_date"].dt.year
    routine: set = set()
    for fid, g in t.groupby("filer_id"):
        if (g.groupby("m")["y"].nunique() >= min_years).any():
            routine.add(int(fid))
    return routine


def abnormal_size_ids_pit(buys: pd.DataFrame, mult: float = 2.0,
                          min_history: int = 3) -> set:
    """POINT-IN-TIME 'abnormally large for this filer'.

    Same idea as `abnormal_size_ids`, but each buy is compared to the EXPANDING
    median of that filer's STRICTLY PRIOR buys, so the label uses only
    information that existed when the trade was disclosed.

    Use this anywhere the subset feeds a return test. `abnormal_size_ids` uses
    the filer's whole-sample median and is look-ahead: measured on the live
    congress sample the two rules disagree on 364 trades (18.5% of the subset),
    every one of them selected using data from the future.
    """
    out: set = set()
    if buys.empty or "amount_mid" not in buys.columns:
        return out
    need = ["filer_id", "trade_id", "amount_mid", "filing_datetime"]
    if not set(need) <= set(buys.columns):
        return out
    b = buys.dropna(subset=need).sort_values("filing_datetime")
    for _, g in b.groupby("filer_id", sort=False):
        amt = g["amount_mid"].to_numpy(dtype=float)
        tids = g["trade_id"].to_numpy()
        for i in range(min_history, len(g)):
            prior = amt[:i]                      # strictly before this buy
            med = float(np.median(prior))
            if med > 0 and amt[i] >= mult * med:
                out.add(int(tids[i]))
    return out


def top_quartile_mask(amounts: pd.Series) -> tuple[pd.Series, dict]:
    """Boolean mask for the largest-quartile trades, safe against TIED amounts.

    Congress dollar figures are disclosed as RANGES and stored as bucket
    midpoints, so the distribution is a few spikes rather than a continuum: on
    the live sample 74% of buys share the single value 8000.5 (the
    $1,001-$15,000 bucket). `amount >= quantile(0.75)` therefore sweeps the
    whole mode in and selected 98.5% of trades, a "large" subset that was
    simply ALL trades relabelled, which also inflated the multiple-testing
    family with duplicate cells.

    Strict `>` is the correct comparison against a tied quantile. The returned
    info dict reports the fraction actually selected so callers can surface a
    warning when the buckets make a clean quartile impossible.
    """
    a = pd.to_numeric(amounts, errors="coerce")
    q = a.quantile(0.75)
    mask = a > q
    frac = float(mask.mean()) if len(a) else 0.0
    return mask.fillna(False), {
        "threshold": float(q) if pd.notna(q) else float("nan"),
        "fraction": frac,
        "degenerate": bool(frac < 0.10 or frac > 0.40),
    }


def abnormal_size_ids(buys: pd.DataFrame, mult: float = 2.0,
                      min_history: int = 3) -> set:
    """Trade_ids of buys that are ABNORMALLY LARGE for that filer: amount_mid
    >= mult x the filer's own median buy size (filers with >= min_history
    priced buys). A conviction proxy, an outsized bet relative to habit
    generic across congress members and insiders.

    NOT POINT-IN-TIME: the median spans the filer's whole history, so a 2016
    trade is judged against 2026 trades. Fine for descriptive attribution;
    use `abnormal_size_ids_pit` for anything that scores forward returns.
    """
    out: set = set()
    if buys.empty or "amount_mid" not in buys.columns:
        return out
    b = buys.dropna(subset=["filer_id", "trade_id", "amount_mid"])
    for _, g in b.groupby("filer_id"):
        if len(g) < min_history:
            continue
        med = g["amount_mid"].median()
        if not (med and med > 0):
            continue
        out.update(int(t) for t in g[g["amount_mid"] >= mult * med]["trade_id"])
    return out


def subset_stats(per_trade: pd.DataFrame, hcol: str) -> dict:
    """n, mean/median excess, hit rate, clustered t-stat for a per-trade subset.
    The t-stat clusters by execution date so overlapping same-day bets don't
    inflate significance."""
    from copytrade.backtest.metrics import clustered_tstat

    s = per_trade[hcol].dropna()
    if len(s) == 0:
        return {"n": 0, "mean_excess": np.nan, "median": np.nan,
                "hit_rate": np.nan, "tstat": np.nan}
    tstat, _ = clustered_tstat(s, per_trade.loc[s.index, "entry_date"])
    return {"n": int(len(s)), "mean_excess": float(s.mean()),
            "median": float(s.median()), "hit_rate": float((s > 0).mean()),
            "tstat": float(tstat)}


def cluster_ticker_summary(per_trade: pd.DataFrame, cluster_ids: set,
                           hcol: str, top: int = 15) -> pd.DataFrame:
    """Most-clustered tickers (by distinct insiders) with their forward excess."""
    cl = per_trade[per_trade["trade_id"].isin(cluster_ids)]
    if cl.empty:
        return pd.DataFrame()
    g = cl.groupby("ticker").agg(
        cluster_buys=("trade_id", "count"),
        insiders=("filer_id", "nunique"),
        mean_excess=(hcol, "mean"),
        hit_rate=(hcol, lambda s: float((s > 0).mean())),
    ).reset_index().sort_values("insiders", ascending=False)
    return g.head(top)
