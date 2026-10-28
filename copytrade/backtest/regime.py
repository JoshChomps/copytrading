"""Trend / regime filter for the copy-trading basket.

The one experiment aimed squarely at a higher-risk-appetite, profit-first
investor: the copy-ALL basket's return is essentially market beta (the
factor/beta-neutral tests showed no independent alpha), so the only honest
way to improve *profit per unit of drawdown* without giving up that beta is
to sidestep the worst of the bear markets. A classic, hard-to-overfit rule
does exactly that: hold the basket only while a broad-market gauge (SPY) is
above its long moving average; step to cash (or bonds) when it is below.

Point-in-time discipline: the decision on day t uses the signal's close and
moving average through day t-1 only (`.shift(1)`), so there is no look-ahead.
Every regime flip pays a round-trip switching cost.

This is not a promise of higher return, a trend filter usually trades a
little CAGR for a much shallower drawdown, which is the trade a leveraged
investor most needs (a smaller drawdown is what makes leverage survivable).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def regime_returns(basket_ret: pd.Series, signal_price: pd.Series,
                   ma_window: int = 200, risk_off_ret: pd.Series | None = None,
                   switch_cost: float = 0.001) -> tuple[pd.Series, pd.Series]:
    """Daily returns of the regime-filtered basket.

    risk-on  (signal_price > its `ma_window` SMA, evaluated with yesterday's
             data): earn the basket's return.
    risk-off (at/below the MA): earn `risk_off_ret` if given (e.g. TLT daily
             returns), else 0 (cash).
    Each flip subtracts `switch_cost` (a round-trip fraction, ~10 bps default)
    that day. `signal_price` should carry history BEFORE the basket span so the
    first day already has a full moving average.

    Returns (regime_daily_returns, risk_on_flag) aligned to basket_ret.index.
    """
    ma = signal_price.rolling(ma_window, min_periods=ma_window).mean()
    # decision uses data through yesterday, applied to today's return
    risk_on = (signal_price > ma).shift(1)
    risk_on = risk_on.reindex(basket_ret.index).fillna(False).astype(bool)
    if risk_off_ret is None:
        off = pd.Series(0.0, index=basket_ret.index)
    else:
        off = risk_off_ret.reindex(basket_ret.index).fillna(0.0)
    ret = basket_ret.where(risk_on, off).astype(float)
    flips = risk_on.astype(int).diff().abs().fillna(0.0)
    ret = ret - flips * switch_cost
    return ret.rename("regime"), risk_on.rename("risk_on")


def regime_equity(basket_equity: pd.Series, signal_price: pd.Series,
                  ma_window: int = 200, risk_off_ret: pd.Series | None = None,
                  switch_cost: float = 0.001) -> tuple[pd.Series, dict]:
    """Regime-filtered equity curve (starts at the basket's first value) plus a
    small stats dict: number of switches and fraction of days in-market."""
    basket_equity = basket_equity.dropna()
    if len(basket_equity) < 3:
        return pd.Series(dtype=float), {}
    basket_ret = basket_equity.pct_change().dropna()
    ret, risk_on = regime_returns(basket_ret, signal_price, ma_window,
                                  risk_off_ret, switch_cost)
    eq = (1 + ret).cumprod() * float(basket_equity.iloc[0])
    eq = pd.concat([pd.Series([float(basket_equity.iloc[0])],
                              index=[basket_equity.index[0]]), eq])
    ro = risk_on.reindex(ret.index).fillna(False)
    info = {
        "switches": int(ro.astype(int).diff().abs().fillna(0).sum()),
        "time_in_market": float(ro.mean()),
        "ma_window": ma_window,
    }
    return eq.rename("regime"), info
