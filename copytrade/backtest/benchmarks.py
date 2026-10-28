"""Low-effort long-term strategies the copy-trading edge must beat.

If copying disclosures cannot outperform these NET OF COSTS, the honest
conclusion is "buy the index". All curves start at 1.0 on the same date and
pay the same cost model:

  * SPY buy & hold: one purchase, hold forever.
  * SPY DCA (12m): capital drip-fed monthly over the first year, then held
                      (uninvested cash earns 0; the classic lump-sum-averse plan).
  * 60/40 SPY/TLT: quarterly-rebalanced stock/bond mix, costs on turnover.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from copytrade.backtest.costs import CostModel


def buy_and_hold(price: pd.Series, cost: CostModel, portfolio_value: float = 250_000.0) -> pd.Series:
    price = price.dropna()
    p0 = float(price.iloc[0])
    entry = cost.side_frac(portfolio_value, portfolio_value / p0, "buy")
    return (price / p0) * (1 - entry)


def dca(price: pd.Series, cost: CostModel, months: int = 12,
        portfolio_value: float = 250_000.0) -> pd.Series:
    price = price.dropna()
    contrib_dates = price.groupby(price.index.to_period("M")).apply(lambda s: s.index[0])
    contrib_dates = list(contrib_dates.iloc[:months])
    per = 1.0 / len(contrib_dates)
    units = 0.0
    cash = 1.0
    eq = pd.Series(index=price.index, dtype=float)
    ci = 0
    for d, p in price.items():
        if ci < len(contrib_dates) and d >= contrib_dates[ci]:
            notional = per * portfolio_value
            fee = cost.cost_per_side(notional, notional / p, "buy") / portfolio_value
            units += (per - fee) / p
            cash -= per
            ci += 1
        eq[d] = cash + units * p
    return eq


def sixty_forty(spy: pd.Series, tlt: pd.Series, cost: CostModel,
                rebalance_months: int = 3, portfolio_value: float = 250_000.0) -> pd.Series:
    df = pd.concat([spy.rename("spy"), tlt.rename("tlt")], axis=1).dropna()
    if df.empty:
        return pd.Series(dtype=float)
    w = {"spy": 0.6, "tlt": 0.4}
    units = {}
    p0 = df.iloc[0]
    for k in w:
        notional = w[k] * portfolio_value
        fee = cost.cost_per_side(notional, notional / p0[k], "buy") / portfolio_value
        units[k] = (w[k] - fee) / p0[k]
    eq = pd.Series(index=df.index, dtype=float)
    last_rebalance = df.index[0].to_period("M")
    for d, row in df.iterrows():
        value = units["spy"] * row["spy"] + units["tlt"] * row["tlt"]
        months_since = (d.to_period("M") - last_rebalance).n
        if months_since >= rebalance_months:
            total_fee = 0.0
            for k in w:
                traded = abs(w[k] * value - units[k] * row[k])
                if traded > 1e-12:
                    side = "buy" if w[k] * value > units[k] * row[k] else "sell"
                    total_fee += cost.cost_per_side(
                        traded * portfolio_value, traded * portfolio_value / row[k], side
                    ) / portfolio_value
            value -= total_fee
            for k in w:
                units[k] = w[k] * value / row[k]
            last_rebalance = d.to_period("M")
        eq[d] = units["spy"] * row["spy"] + units["tlt"] * row["tlt"]
    return eq


POPULAR_ETFS = {
    "SPY": "S&P 500 (large-cap core)",
    "QQQ": "Nasdaq-100 (growth / high-return)",
    "VTI": "Total US market",
    "IWM": "Russell 2000 (small-cap)",
    "SCHD": "Dividend (defensive / 'safety')",
}


def etf_buy_hold(prices: pd.DataFrame, cost: CostModel, tickers=tuple(POPULAR_ETFS),
                 start=None, end=None, portfolio_value: float = 250_000.0) -> dict[str, pd.Series]:
    """Buy&hold equity curves (cost-net) for whichever popular ETFs are present
    in `prices`, over [start, end]. Used by the full-metrics scorecard so the
    basket is compared against the ETFs a real investor would actually pick, not
    only SPY."""
    window = prices.loc[start:end] if (start is not None or end is not None) else prices
    out = {}
    for t in tickers:
        if t in window.columns and window[t].notna().sum() > 50:
            out[t] = buy_and_hold(window[t].dropna(), cost, portfolio_value)
    return out


def benchmark_suite(prices: pd.DataFrame, cost: CostModel,
                    start=None, end=None, spy: str = "SPY", tlt: str = "TLT",
                    portfolio_value: float = 250_000.0) -> pd.DataFrame:
    """Equity curves for every benchmark available in `prices`, over [start, end]."""
    window = prices.loc[start:end] if (start is not None or end is not None) else prices
    out = {}
    if spy in window.columns:
        s = window[spy].dropna()
        out["SPY buy & hold"] = buy_and_hold(s, cost, portfolio_value)
        out["SPY DCA (12m)"] = dca(s, cost, portfolio_value=portfolio_value)
        if tlt in window.columns and window[tlt].notna().sum() > 50:
            out["60/40 SPY/TLT"] = sixty_forty(s, window[tlt], cost, portfolio_value=portfolio_value)
    return pd.DataFrame(out)
