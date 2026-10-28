"""What characterised the trades that worked, disclosures as a lens, not a feed.

A DIFFERENT QUESTION FROM EVERYTHING ELSE IN THIS REPO. Every other analysis
asks "does mirroring these people pay?" and the answer is a well-established
no: gross excess return is indistinguishable from zero at every horizon, and
costs make it reliably negative.

But that is a statement about the MEAN. A zero average is perfectly consistent
with large and predictable dispersion around it, winners and losers that
cancel out. So the question here is:

    conditional on a disclosure happening, do observable characteristics of
    the trade and its context predict which ones worked?

If yes, the disclosures stop being a copy feed and become a filtered universe:
a place to look, with a separate rule deciding what to act on.

THE CONTROL IS THE WHOLE POINT. Suppose momentum predicts returns among
congress-bought stocks. That is not a finding about Congress, momentum
predicts returns among *all* stocks. Without a control you rediscover the
factor zoo and mislabel it as political insight.

So every feature is tested twice: once on the disclosed trades, and once on a
MATCHED CONTROL: random tickers drawn on the same dates from the same priced
universe, with identical features and identical forward-return construction.
The quantity of interest is the DIFFERENCE in predictive power:

    IC(disclosed) - IC(control)

Only that difference can be attributed to the disclosure. A feature that works
equally well in both samples is a generic factor you could trade directly,
without waiting 45 days for a filing.

POINT-IN-TIME. Every feature is built from data strictly BEFORE the execution
close (through `entry_pos - 1`). The forward return starts at `entry_pos`,
which is where `execution_position` put it. No feature may touch the outcome
window.

MULTIPLE TESTING. This is the widest search in the project, so the same
BH-FDR correction is applied across the whole feature grid, on the DIFFERENCE
statistic rather than the raw one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from copytrade.rank.timescale import benjamini_hochberg


def _block_of(months: pd.PeriodIndex, horizon: int) -> np.ndarray:
    """Group month periods into NON-OVERLAPPING blocks as long as the holding
    period, so each block mean is an independent draw.

    Monthly ICs computed on `horizon`-day forward returns are NOT independent:
    at h=63 adjacent months share ~2/3 of their outcome window. Testing them
    with a plain one-sample t inflated the reported |t| by ~1.9x and produced
    P(|t|>2) = 27.5% on a pure-null feature against a nominal 5%. Blocking is
    the same correction `timescale.block_test` applies to the sweep.
    """
    block_months = max(1, int(round(horizon / 21)))
    origin = months.min()
    offset = np.array([(m - origin).n for m in months])
    return offset // block_months


def _t_from_blocked(values: pd.Series, months: pd.PeriodIndex,
                    horizon: int) -> tuple[float, float, int]:
    """One-sample t on block means. Returns (t, p, n_blocks)."""
    if len(values) == 0:
        return np.nan, np.nan, 0
    blocks = _block_of(months, horizon)
    bm = pd.Series(values.to_numpy(), index=blocks).groupby(level=0).mean()
    n = int(len(bm))
    if n < 3:
        return np.nan, np.nan, n
    t, p = stats.ttest_1samp(bm, 0.0)
    return float(t), float(p), n

# Features are grouped so the report can say which KIND of information paid.
PRICE_FEATURES = [
    "mom_21", "mom_63", "mom_252", "vol_63", "dd_from_high", "rel_str_63",
]
CONTEXT_FEATURES = ["mkt_above_ma", "mkt_vol_63"]
TRADE_FEATURES = ["amount", "lag_days", "n_filers_30d", "is_repeat", "is_senate"]


def _safe_ret(vals: np.ndarray, i: int, back: int) -> float:
    """Return over the `back` closes ending at i, or NaN if unavailable."""
    j = i - back
    if j < 0 or i < 0:
        return np.nan
    a, b = vals[j], vals[i]
    if not np.isfinite(a) or not np.isfinite(b) or a <= 0:
        return np.nan
    return b / a - 1.0


def price_features(vals: np.ndarray, pos: int, bench_vals: np.ndarray,
                   bench_ma: np.ndarray, bench_vol: np.ndarray) -> dict:
    """Point-in-time price/context features as of the close BEFORE execution.

    `pos` is the execution index; everything is computed at `pos - 1` so no
    feature can see the bar it is used to trade.
    """
    i = pos - 1
    out = {f: np.nan for f in PRICE_FEATURES + CONTEXT_FEATURES}
    if i < 1:
        return out
    out["mom_21"] = _safe_ret(vals, i, 21)
    out["mom_63"] = _safe_ret(vals, i, 63)
    out["mom_252"] = _safe_ret(vals, i, 252)

    lo = max(0, i - 63)
    seg = vals[lo:i + 1]
    seg = seg[np.isfinite(seg) & (seg > 0)]
    if len(seg) > 5:
        r = np.diff(seg) / seg[:-1]
        out["vol_63"] = float(np.std(r) * np.sqrt(252))
    lo2 = max(0, i - 252)
    seg2 = vals[lo2:i + 1]
    seg2 = seg2[np.isfinite(seg2) & (seg2 > 0)]
    if len(seg2) > 5 and np.isfinite(vals[i]) and vals[i] > 0:
        out["dd_from_high"] = float(vals[i] / np.max(seg2) - 1.0)

    b63 = _safe_ret(bench_vals, i, 63)
    if np.isfinite(out["mom_63"]) and np.isfinite(b63):
        out["rel_str_63"] = out["mom_63"] - b63
    if i < len(bench_ma) and np.isfinite(bench_ma[i]) and np.isfinite(bench_vals[i]):
        out["mkt_above_ma"] = float(bench_vals[i] > bench_ma[i])
    if i < len(bench_vol):
        out["mkt_vol_63"] = float(bench_vol[i])
    return out


def forward_excess(vals: np.ndarray, bench_vals: np.ndarray, pos: int,
                   horizon: int, cost: float) -> float:
    """Net excess return over `horizon` closes from the execution close.

    Identical construction for disclosed and control observations, otherwise
    the comparison would be between two different measurements.
    """
    end = pos + horizon
    if end >= len(vals) or pos < 0:
        return np.nan
    p0, p1 = vals[pos], vals[end]
    b0, b1 = bench_vals[pos], bench_vals[end]
    if not all(np.isfinite(x) for x in (p0, p1, b0, b1)) or p0 <= 0 or b0 <= 0:
        return np.nan
    return (p1 / p0 - 1.0) - (b1 / b0 - 1.0) - cost


def build_disclosed(per_trade: pd.DataFrame, prices: pd.DataFrame,
                    bench: pd.Series, horizon: int,
                    trades: pd.DataFrame | None = None,
                    chamber_by_filer: dict | None = None) -> pd.DataFrame:
    """Feature matrix + forward return for each priced disclosed buy.

    The forward return is READ FROM `per_trade[f"h{horizon}"]`, which
    `run_event_study` already computed at the same `entry_pos` with the same
    cost model. Recomputing it here would fork the definition of "net excess
    return" in two, so the engine's non-positive-price guard would have to be
    maintained twice (project rule: new analyses must consume run_event_study).
    """
    cal = bench.index
    px = prices.reindex(cal).ffill()
    col = {c: i for i, c in enumerate(px.columns)}
    pv = px.values
    bv = bench.values
    bma = pd.Series(bv, index=cal).rolling(200, min_periods=200).mean().values
    bvol = (pd.Series(bv, index=cal).pct_change()
            .rolling(63, min_periods=20).std().values * np.sqrt(252))
    hcol = f"h{horizon}"

    # Chamber comes from filers.meta, NOT from the ingest source name. Matching
    # only source=='senate_watcher' mislabels senate_efd and quiver_congress as
    # House: and since senate_watcher stops at 2021-03-09 while house_clerk
    # runs to 2026, that flag was really a proxy for DATE, not chamber.
    chamber = chamber_by_filer or {}
    src: dict = {}
    if trades is not None and "source" in trades.columns:
        src = dict(zip(trades["trade_id"], trades["source"]))

    # trailing cluster intensity + repeat-buy flag, both strictly backward
    hist_ticker: dict[str, list] = {}
    hist_pair: set = set()

    rows = []
    pt = per_trade.sort_values("entry_date")
    for t in pt.itertuples():
        ci = col.get(t.ticker)
        if ci is None:
            continue
        pos = int(t.entry_pos)
        f = price_features(pv[:, ci], pos, bv, bma, bvol)
        f["amount"] = float(t.amount)
        f["lag_days"] = float(t.lag_days) if pd.notna(t.lag_days) else np.nan
        prev = hist_ticker.get(t.ticker, [])
        cutoff = t.entry_date - pd.Timedelta(days=30)
        f["n_filers_30d"] = float(len({fid for d, fid in prev if d >= cutoff}))
        f["is_repeat"] = float((t.filer_id, t.ticker) in hist_pair)
        ch = chamber.get(t.filer_id)
        if ch is None:                      # fall back to the ingest source
            s_ = src.get(t.trade_id)
            ch = ("Senate" if s_ in ("senate_watcher", "senate_efd")
                  else "House" if s_ == "house_clerk" else None)
        f["is_senate"] = np.nan if ch is None else float(str(ch).lower() == "senate")
        f["fwd"] = float(getattr(t, hcol)) if pd.notna(getattr(t, hcol, np.nan)) else np.nan
        f["entry_date"] = t.entry_date
        f["ticker"] = t.ticker
        rows.append(f)
        hist_ticker.setdefault(t.ticker, []).append((t.entry_date, t.filer_id))
        hist_pair.add((t.filer_id, t.ticker))
    return pd.DataFrame(rows)


def build_control(disclosed: pd.DataFrame, prices: pd.DataFrame,
                  bench: pd.Series, horizon: int, cost,
                  per_date: int = 3, seed: int = 0,
                  trade_notional: float = 10_000.0) -> pd.DataFrame:
    """Matched control: random priced tickers on the SAME execution dates.

    Matching on date is what makes the comparison fair, it holds the market
    environment fixed, so any difference in predictive power is about which
    stocks were chosen, not when.

    `cost` may be a CostModel (preferred) or a flat fraction. A CostModel
    charges the same PRICE-DEPENDENT round trip the disclosed arm pays: with
    ibkr_ca that ranges ~26bps on a $200 stock to ~226bps on a $0.50 one, and
    share price correlates with mom_252, vol_63 and dd_from_high, precisely
    the features ic_diff is meant to isolate. Charging the disclosed arm a
    varying cost and the control a constant one put that asymmetry straight
    into the headline statistic.
    """
    cal = bench.index
    px = prices.reindex(cal)
    raw = px.values                      # NOT forward-filled, see below
    pv = px.ffill().values
    bv = bench.values
    cols = list(px.columns)
    col_idx = np.arange(len(cols))
    bma = pd.Series(bv, index=cal).rolling(200, min_periods=200).mean().values
    bvol = (pd.Series(bv, index=cal).pct_change()
            .rolling(63, min_periods=20).std().values * np.sqrt(252))
    rng = np.random.default_rng(seed)
    pos_of = {d: i for i, d in enumerate(cal)}

    # Eligibility demands a REAL print, exactly as engine.py requires a real
    # entry price. Testing the forward-filled matrix instead let delisted
    # tickers stay drawable forever at a frozen price: 71 of 4,766 control rows
    # were dead names with vol_63 == mom_21 == dd_from_high == 0.0 exactly and
    # mean forward excess -5.21% vs +0.008% for live controls. Because the ICs
    # are Spearman those are not diffuse noise, they are exact ties at the
    # extreme rank of two features with systematically negative outcomes.
    valid = np.isfinite(raw) & (raw > 0)

    flat_cost = cost if isinstance(cost, (int, float)) else None

    rows = []
    for d in disclosed["entry_date"].unique():
        pos = pos_of.get(pd.Timestamp(d))
        if pos is None or pos < 1:
            continue
        elig = col_idx[valid[pos]]
        if len(elig) == 0:
            continue
        for ci in rng.choice(elig, size=min(per_date, len(elig)), replace=False):
            f = price_features(pv[:, ci], pos, bv, bma, bvol)
            # trade-level features are undefined for a non-disclosed stock
            for k in TRADE_FEATURES:
                f[k] = np.nan
            p0 = raw[pos, ci]
            c = (flat_cost if flat_cost is not None
                 else cost.roundtrip_frac(trade_notional, trade_notional / p0))
            f["fwd"] = forward_excess(pv[:, ci], bv, pos, horizon, c)
            f["entry_date"] = pd.Timestamp(d)
            f["ticker"] = cols[ci]
            rows.append(f)
    return pd.DataFrame(rows)


def information_coefficient(df: pd.DataFrame, feature: str,
                            horizon: int = 63) -> dict:
    """Spearman IC between a feature and forward return, plus a clustered t.

    Rank correlation rather than a linear fit: it is robust to the fat tails
    and outliers that dominate single-stock returns, and it answers the
    practical question (does ranking on this feature sort outcomes?) directly.

    The t-stat comes from the monthly IC series aggregated into
    NON-OVERLAPPING blocks as long as `horizon`. Two corrections stack here:
    monthly grouping stops same-month trades from being counted as independent,
    and blocking stops adjacent MONTHS from being counted as independent when
    their forward-return windows overlap (at h=63 they share ~2/3 of the
    window). Without the second step, a pure-null feature reported |t|>2 on
    27.5% of runs against a nominal 5%.
    """
    sub = df[[feature, "fwd", "entry_date"]].dropna()
    empty = {"feature": feature, "n": len(sub), "ic": np.nan,
             "t": np.nan, "p": np.nan, "months": 0, "blocks": 0}
    if len(sub) < 50:
        return empty
    # >=2 levels, not >=3: binary features (market regime, repeat-buy,
    # chamber) are legitimate predictors and a Spearman IC on two levels is
    # just a rank-sum test. Requiring 3 silently dropped them entirely.
    if sub[feature].nunique() < 2:
        return empty
    m = _monthly_ic(df, feature)
    if len(m) < 3:
        return {**empty, "months": int(len(m))}
    t, p, nb = _t_from_blocked(m["ic"], m.index, horizon)
    return {"feature": feature, "n": int(len(sub)), "ic": float(m["ic"].mean()),
            "t": t, "p": p, "months": int(len(m)), "blocks": nb}


def compare_features(disclosed: pd.DataFrame, control: pd.DataFrame,
                     features: list[str], horizon: int = 63) -> pd.DataFrame:
    """Per-feature IC in both samples, and the difference that matters.

    `ic_diff` is disclosed minus control, tested on the per-month PAIRED
    difference of ICs (so the market environment cancels), aggregated into
    non-overlapping blocks as long as `horizon` (so overlapping forward
    windows do not masquerade as independent months). FDR runs across the
    feature grid on that difference.

    `controllable` marks features for which a control is even DEFINABLE.
    Trade-level features (amount, disclosure lag, cluster intensity) do not
    exist for a random stock, so they can never be compared and must never be
    described as generic factors, the previous code left their q_fdr as NaN,
    and `NaN < 0.05` being False made "could not be tested" indistinguishable
    from "tested and failed".
    """
    rows = []
    for f in features:
        d = information_coefficient(disclosed, f, horizon)
        has_ctrl = f in control.columns and control[f].notna().any()
        c = (information_coefficient(control, f, horizon) if has_ctrl else
             {"ic": np.nan, "t": np.nan, "p": np.nan, "n": 0, "months": 0})
        t_diff, p_diff, nb = np.nan, np.nan, 0
        if has_ctrl:
            dm = _monthly_ic(disclosed, f)
            cm = _monthly_ic(control, f)
            both = dm.join(cm, how="inner", lsuffix="_d", rsuffix="_c").dropna()
            if len(both) >= 3:
                t_diff, p_diff, nb = _t_from_blocked(
                    both.iloc[:, 0] - both.iloc[:, 1], both.index, horizon)
        ic_d, ic_c = d["ic"], c["ic"]
        rows.append({
            "feature": f, "n": d["n"], "months": d["months"],
            "controllable": bool(has_ctrl),
            "ic_disclosed": ic_d, "t_disclosed": d["t"],
            "ic_control": ic_c, "t_control": c["t"],
            "ic_diff": (ic_d - ic_c) if np.isfinite(ic_d) and np.isfinite(ic_c) else np.nan,
            "t_diff": t_diff, "p_diff": p_diff, "diff_blocks": nb,
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["q_fdr"] = benjamini_hochberg(out["p_diff"].values)
    # Sign guard. p_diff is TWO-SIDED, so without it a feature that predicts
    # returns only in the CONTROL, the strongest possible evidence that it is
    # a generic factor: was reported in green as a disclosure-specific
    # discovery (reproduced at ic_diff=-0.867, q=2.8e-37). Both sibling modules
    # guard direction; this one did not.
    out["disclosure_specific"] = (
        (out["q_fdr"] < 0.05) & (out["ic_diff"] > 0) & out["controllable"]
    )
    return out.sort_values("ic_disclosed", key=lambda s: s.abs(), ascending=False)


def _monthly_ic(df: pd.DataFrame, feature: str) -> pd.DataFrame:
    """Per-month Spearman IC series for one feature (index = month)."""
    if feature not in df.columns:
        return pd.DataFrame()
    sub = df[[feature, "fwd", "entry_date"]].dropna()
    if sub.empty:
        return pd.DataFrame()
    month = pd.to_datetime(sub["entry_date"]).dt.to_period("M")
    recs = {}
    for m, g in sub.groupby(month):
        if len(g) >= 10 and g[feature].nunique() >= 2:
            rho, _ = stats.spearmanr(g[feature], g["fwd"])
            if np.isfinite(rho):
                recs[m] = rho
    return pd.DataFrame({"ic": pd.Series(recs)})


def quintile_spread(df: pd.DataFrame, feature: str, q: int = 5) -> pd.DataFrame:
    """Mean forward return by feature quintile, the tradeable form of an IC.

    An IC says the ranking works; this says how much it is worth and whether
    the relationship is monotone. A non-monotone 'spread' driven by one
    extreme bucket is usually an outlier artifact, not a signal.
    """
    sub = df[[feature, "fwd", "entry_date"]].dropna()
    if len(sub) < 100 or sub[feature].nunique() < q:
        return pd.DataFrame()
    try:
        sub = sub.assign(bucket=pd.qcut(sub[feature], q, labels=False, duplicates="drop"))
    except ValueError:
        return pd.DataFrame()
    # The nunique() guard above is NOT sufficient: qcut drops duplicate quantile
    # EDGES, not duplicate values. A feature that is ~97% zeros with a thin tail
    # (n_filers_30d) has nunique() >= q while all q edges are 0, collapsing to a
    # single bucket: which then reported spread = 0.0 and a t built from float
    # residuals ~1e-18 under a header saying "by quintile".
    if sub["bucket"].nunique() < q:
        return pd.DataFrame()

    g = sub.groupby("bucket")["fwd"].agg(["count", "mean", "median"])
    g["hit"] = sub.groupby("bucket")["fwd"].apply(lambda s: (s > 0).mean())
    top, bot = g.index.max(), g.index.min()
    spread = g.loc[top, "mean"] - g.loc[bot, "mean"]

    # PAIRED per-month difference of bucket means. Pooling top returns with
    # NEGATED bottom returns only estimates (mean_top - mean_bot) when the two
    # buckets are equally represented WITHIN each month; qcut equalises counts
    # sample-wide, not per month. For market-wide features whole months land in
    # one bucket, so the old statistic measured the return LEVEL: reproduced at
    # t = -3.61 on two buckets drawn from an identical distribution (true
    # spread exactly 0).
    month = pd.to_datetime(sub["entry_date"]).dt.to_period("M")
    per_month = (sub.assign(m=month)
                 .groupby(["m", "bucket"])["fwd"].mean().unstack("bucket"))
    t = np.nan
    if top in per_month.columns and bot in per_month.columns:
        d = (per_month[top] - per_month[bot]).dropna()
        if len(d) >= 3:
            t = float(stats.ttest_1samp(d, 0.0)[0])
    g.attrs["spread"] = float(spread)
    g.attrs["t"] = t
    g.attrs["paired_months"] = int(len(d)) if t == t else 0
    return g
