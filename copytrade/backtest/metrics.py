"""Performance and significance metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

TRADING_DAYS = 252


def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    return float((equity / peak - 1).min())


def perf_stats(equity: pd.Series) -> dict:
    """CAGR/vol/Sharpe/Sortino/Calmar/max-DD from an equity curve (daily closes)."""
    equity = equity.dropna()
    if len(equity) < 2:
        return {}
    rets = equity.pct_change().dropna()
    years = len(rets) / TRADING_DAYS
    total = float(equity.iloc[-1] / equity.iloc[0] - 1)
    cagr = float((1 + total) ** (1 / years) - 1) if years > 0 else np.nan
    vol = float(rets.std() * np.sqrt(TRADING_DAYS))
    sharpe = float(rets.mean() / rets.std() * np.sqrt(TRADING_DAYS)) if rets.std() > 0 else np.nan
    downside = rets[rets < 0].std()
    sortino = float(rets.mean() / downside * np.sqrt(TRADING_DAYS)) if downside and downside > 0 else np.nan
    mdd = max_drawdown(equity)
    return {
        "total_return": total,
        "cagr": cagr,
        "vol": vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": float(cagr / abs(mdd)) if mdd < 0 and pd.notna(cagr) else np.nan,
        "max_drawdown": mdd,
        "days": len(rets),
    }


def full_metrics(equity: pd.Series, bench_rets: pd.Series | None = None,
                 rf_annual: float = 0.0) -> dict:
    """The complete risk/return battery for an equity curve (daily closes).

    Beyond CAGR/Sharpe this returns the full set a discretionary allocator
    actually reads: volatility and downside deviation, Sortino/Calmar,
    max-drawdown, the return distribution's skew and (excess) kurtosis, daily
    historical VaR/CVaR at 95%, the Omega ratio, tail ratio, % positive days,
    and best/worst day. If `bench_rets` (the benchmark's daily returns) is
    given, it adds the market-relative "greeks": beta (the equity analogue of
    delta: sensitivity to the market), annualized Jensen alpha, correlation
    and R2, Treynor ratio, tracking error, information ratio, and up/down
    capture. rf_annual is the risk-free rate used for Sharpe/Sortino/Treynor
    excess (0 by default; caller may pass a T-bill proxy)."""
    equity = equity.dropna()
    if len(equity) < 3:
        return {}
    rets = equity.pct_change().dropna()
    if rets.std() == 0 or len(rets) < 2:
        return {}
    rf_daily = (1 + rf_annual) ** (1 / TRADING_DAYS) - 1
    excess = rets - rf_daily
    years = len(rets) / TRADING_DAYS
    total = float(equity.iloc[-1] / equity.iloc[0] - 1)
    cagr = float((1 + total) ** (1 / years) - 1) if years > 0 else np.nan
    vol = float(rets.std() * np.sqrt(TRADING_DAYS))
    downside = rets[rets < 0].std()
    mdd = max_drawdown(equity)
    var95 = float(rets.quantile(0.05))
    tail_losses = rets[rets <= var95]
    gains = float(rets[rets > 0].sum())
    losses = float(-rets[rets < 0].sum())
    q95, q05 = float(rets.quantile(0.95)), var95
    out = {
        "total_return": total,
        "cagr": cagr,
        "vol": vol,
        "downside_dev": float(downside * np.sqrt(TRADING_DAYS)) if downside and downside > 0 else np.nan,
        "sharpe": float(excess.mean() / rets.std() * np.sqrt(TRADING_DAYS)),
        "sortino": float(excess.mean() / downside * np.sqrt(TRADING_DAYS)) if downside and downside > 0 else np.nan,
        "calmar": float(cagr / abs(mdd)) if mdd < 0 and pd.notna(cagr) else np.nan,
        "max_drawdown": mdd,
        "skew": float(rets.skew()),
        "kurtosis": float(rets.kurt()),                       # excess kurtosis (fat tails > 0)
        "var95": var95,                                       # daily 5% historical VaR
        "cvar95": float(tail_losses.mean()) if len(tail_losses) else np.nan,
        "omega": float(gains / losses) if losses > 0 else np.nan,
        "tail_ratio": float(q95 / abs(q05)) if q05 != 0 else np.nan,
        "pct_positive_days": float((rets > 0).mean()),
        "best_day": float(rets.max()),
        "worst_day": float(rets.min()),
        "days": len(rets),
    }
    if bench_rets is not None:
        b = bench_rets.reindex(rets.index).dropna()
        r = rets.reindex(b.index).dropna()
        b = b.reindex(r.index)
        if len(b) > 20 and float(b.var(ddof=1)) > 0:
            beta = float(r.cov(b) / b.var(ddof=1))
            alpha_daily = float(r.mean() - beta * b.mean())
            active = (r - b)
            up, down = b > 0, b < 0
            out.update({
                "beta": beta,                                 # ~ delta to the market
                "alpha_annualized": float((1 + alpha_daily) ** TRADING_DAYS - 1),
                "corr": float(r.corr(b)),
                "r2": float(r.corr(b) ** 2),
                "treynor": float((excess.reindex(r.index).mean() * TRADING_DAYS) / beta) if beta != 0 else np.nan,
                "tracking_error": float(active.std() * np.sqrt(TRADING_DAYS)),
                "info_ratio": float(active.mean() / active.std() * np.sqrt(TRADING_DAYS)) if active.std() > 0 else np.nan,
                "up_capture": float(r[up].mean() / b[up].mean()) if up.any() and b[up].mean() != 0 else np.nan,
                "down_capture": float(r[down].mean() / b[down].mean()) if down.any() and b[down].mean() != 0 else np.nan,
            })
    return out


def annual_returns(equity: pd.Series) -> pd.Series:
    """Calendar-year total returns from a daily equity curve, chained across
    year boundaries (so no Dec->Jan gap is dropped). The first year is a
    partial-year return from the series start; index is the integer year."""
    equity = equity.dropna()
    if len(equity) < 2:
        return pd.Series(dtype=float)
    year_end = equity.resample("YE").last()
    base = pd.Series([equity.iloc[0]], index=[equity.index[0]])
    chain = pd.concat([base, year_end])
    ann = chain.pct_change().dropna()
    ann.index = ann.index.year
    return ann[~ann.index.duplicated(keep="last")]


def bootstrap_outperf_pvalue(
    strat_rets: pd.Series, bench_rets: pd.Series, n_boot: int = 2000, seed: int = 0
) -> float:
    """One-sided bootstrap p-value that the strategy's mean daily EXCESS
    return over the benchmark is <= 0 (small p => outperformance unlikely to
    be luck). IID resampling of paired daily excess returns, a coarse guard
    against celebrating noise, not a full deflated-Sharpe treatment (the
    walk-forward already handles selection bias; this handles sampling
    noise)."""
    ex = (strat_rets - bench_rets.reindex(strat_rets.index)).dropna().values
    if len(ex) < 20:
        return np.nan
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(ex), size=(n_boot, len(ex)))
    means = ex[idx].mean(axis=1)
    return float((means <= 0).mean())


def clustered_tstat(values: pd.Series, dates: pd.Series) -> tuple[float, int]:
    """t-stat of the mean, clustered by calendar month of entry (plan §4.2):
    trades filed in the same month share market conditions, so we test across
    monthly cluster means rather than pretending trades are independent."""
    df = pd.DataFrame({"v": values.values, "m": pd.to_datetime(dates).dt.to_period("M")}).dropna()
    if df.empty:
        return np.nan, 0
    cluster_means = df.groupby("m")["v"].mean()
    if len(cluster_means) < 2:
        return np.nan, len(cluster_means)
    t, _ = stats.ttest_1samp(cluster_means, 0.0)
    return float(t), int(len(cluster_means))


def hit_rate(values: pd.Series) -> float:
    v = values.dropna()
    return float((v > 0).mean()) if len(v) else np.nan


def turnover_class(trades: pd.DataFrame) -> str:
    """Median holding period from FIFO-matched buy->sell pairs per ticker,
    classified short (<63 td) / medium (63-252) / long (>252), the user's
    long/short-term filter. 'long' when a filer never sells."""
    holds: list[int] = []
    for _, g in trades.dropna(subset=["transaction_date"]).groupby("ticker"):
        buys = sorted(g[g["side"] == "buy"]["transaction_date"])
        sells = sorted(g[g["side"] == "sell"]["transaction_date"])
        bi = 0
        for s in sells:
            while bi < len(buys) and buys[bi] <= s:
                holds.append(int(np.busday_count(buys[bi].date(), s.date())))
                bi += 1
                break
    if not holds:
        return "long"
    med = float(np.median(holds))
    if med < 63:
        return "short"
    if med <= 252:
        return "medium"
    return "long"
