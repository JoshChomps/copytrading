"""Leverage and margin-call survivability, "how hard can I push this?"

This is the direct answer to a profit-first, higher-risk-appetite question,
and it is the one place where the regime filter earns its keep. The copy
basket has no significant alpha, so leverage cannot manufacture edge: 2x a
zero-alpha book is 2x the beta and 2x the drawdown, minus financing. What
leverage CAN do is convert a low-drawdown asset into a higher-return one at
the same risk of ruin, which is why the drawdown reduction from the regime
filter matters more than its CAGR cost.

WHY A SINGLE HISTORICAL PATH IS NOT AN ANSWER. The obvious approach is to
multiply the realised daily returns by 2 and read off the max drawdown. That
answers "would 2x have survived 2015-2026", which is one draw from the
distribution of outcomes and is contaminated by the fact that we already know
this particular decade ended well. The question that actually determines
position size is:

    across plausible orderings of this return process, how often does a
    levered account get liquidated?

So the module reports both: the realised path (what did happen) and a
CIRCULAR BLOCK BOOTSTRAP over the daily returns (what could plausibly have
happened). Blocks, not IID draws, because volatility clusters, crashes are
consecutive bad days, and IID resampling destroys exactly the structure that
causes margin calls, which would understate the risk.

THE MECHANICS MODELLED
  * Financing. Only the borrowed portion accrues interest, daily, at
    `financing_rate` (IBKR-CA margin is a benchmark rate plus a spread; the
    default here is a placeholder the caller should set deliberately).
  * Maintenance margin. A forced liquidation triggers when account equity
    falls below `maintenance` of position value. Defaults to 30% (IBKR-CA /
    CIRO), not the US Reg-T 25% floor. Brokers raise maintenance in stress,
    which this does NOT model, so results here remain optimistic.
  * Liquidation is terminal and absorbing. Once hit, the account is marked
    liquidated at that day's equity. Real brokers partially liquidate and let
    you continue, which can end better OR much worse; treating it as an
    absorbing failure keeps the metric interpretable as "probability of a
    catastrophic outcome".

WHAT THIS DOES NOT MODEL: intraday gaps (a crash through the maintenance level
between closes liquidates worse than the close implies), borrow availability,
margin-rate changes, or the tax consequences of a forced sale in a taxable
sleeve. All of these make real leverage worse than modelled, never better.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# IBKR-CA margin runs a spread over the policy rate. 5.5% is defensible for a
# small USD balance at tier-1 pricing in mid-2026, but it is applied FLAT over
# 2015-2026, a window in which the real rate ran ~1.6% (2015-16, 2020-21) to
# ~6.8% (2023-24); the period average is nearer 3.0-3.5%. Measured on SPY at
# 2x over the full span, bootstrap median terminal return is +650% at 2.0%
# financing vs +584% at 5.5%, so this default is conservative by roughly 40pp
# of levered wealth. Set it deliberately from the caller.
DEFAULT_FINANCING = 0.055

# Maintenance margin. NOT the US Reg-T 25% floor: the broker modelled here is
# IBKR CANADA (CIRO), where standard margin on a listed equity >= $2.00 runs
# 30% (reduced-margin list) to 50%, and house minimums often exceed 25%.
# The digit matters enormously: measured on real SPY 2015-01..2026-07 with
# 4,000 bootstrap paths, P(margin call) at 2x is 11.6% at 25%, 17.0% at 30%,
# 25.3% at 35% and 96.1% at 50%. Mechanically 2x liquidates on a 33.3% asset
# decline at 25% but on 28.6% at 30%, and SPY's realised max drawdown over
# this sample is -33.7%. Always report a sensitivity band, never one number.
DEFAULT_MAINTENANCE = 0.30
MAINTENANCE_GRID = (0.25, 0.30, 0.35, 0.50)


def levered_path(
    daily_ret: np.ndarray,
    leverage: float,
    financing_rate: float = DEFAULT_FINANCING,
    maintenance: float = DEFAULT_MAINTENANCE,
    trading_days: int = 252,
    rebalance: str = "none",
) -> tuple[np.ndarray, int | None]:
    """Equity curve of a levered account, and the liquidation index.

    TWO MODES, AND THE DEFAULT MATTERS ENORMOUSLY.

    rebalance='none' (DEFAULT: the realistic retail case). Borrow once at the
        start and hold. The loan accrues interest; the margin ratio E/A
        degrades as the assets fall, so a prolonged grind-down liquidates you.
        This is how real margin accounts actually blow up.

    rebalance='daily' (the textbook idealisation, e.g. a daily-reset LETF).
        Leverage is restored to L every day, which RESETS the margin ratio to
        1/L each morning. Consequently only a single catastrophic day can
        trigger a call and a slow 50% decline never does, which badly
        understates real risk of ruin. Offered for comparison with LETF
        products, not as a model of a retail margin account.

    In both modes equity earns the levered asset return less financing on the
    borrowed portion. Liquidation is absorbing: equity freezes at that day's
    value. Returns (equity_curve, liquidation_index_or_None).
    """
    f_daily = financing_rate / trading_days
    n = len(daily_ret)
    eq = np.empty(n + 1)
    eq[0] = 1.0
    liq: int | None = None

    if rebalance == "daily":
        for i, r in enumerate(daily_ret):
            ratio = (1.0 + leverage * r) / (leverage * (1.0 + r)) if leverage else 1.0
            eq[i + 1] = eq[i] * (1.0 + leverage * r - (leverage - 1.0) * f_daily)
            if eq[i + 1] <= 0:
                eq[i + 1] = 0.0
                liq = i
                eq[i + 2:] = 0.0
                break
            if leverage > 1.0 and ratio < maintenance:
                liq = i
                eq[i + 2:] = eq[i + 1]
                break
        return eq, liq

    # fixed-loan: assets compound with the market, the debt compounds with
    # interest, and equity is the difference, so the ratio decays in a
    # drawdown instead of resetting.
    assets = leverage
    debt = leverage - 1.0
    for i, r in enumerate(daily_ret):
        assets *= (1.0 + r)
        debt *= (1.0 + f_daily)
        e = assets - debt
        eq[i + 1] = max(e, 0.0)
        if e <= 0:
            liq = i
            eq[i + 2:] = 0.0
            break
        if leverage > 1.0 and (e / assets) < maintenance:
            liq = i
            eq[i + 2:] = eq[i + 1]
            break
    return eq, liq


def max_dd(equity: np.ndarray) -> float:
    """Max drawdown of an equity path (negative fraction)."""
    peak = np.maximum.accumulate(equity)
    with np.errstate(divide="ignore", invalid="ignore"):
        dd = equity / peak - 1.0
    return float(np.nanmin(dd))


def levered_paths_vec(
    rets: np.ndarray,
    leverage: float,
    financing_rate: float = DEFAULT_FINANCING,
    maintenance: float = DEFAULT_MAINTENANCE,
    trading_days: int = 252,
    rebalance: str = "none",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorised `levered_path` over many paths at once.

    `rets` is (n_paths, n_days). Same accounting as the scalar version, this
    exists purely because the bootstrap needs millions of day-steps and a
    Python loop over them takes minutes.

    Returns (terminal_equity, max_drawdown, breached) per path, with
    liquidation applied as an absorbing state: equity is frozen at the
    liquidation day's value and `breached` flags the paths that were called.
    """
    f_daily = financing_rate / trading_days
    n_paths, n_days = rets.shape

    if rebalance == "daily":
        step = leverage * rets - (leverage - 1.0) * f_daily
        eq = np.cumprod(1.0 + step, axis=1)
        eq = np.hstack([np.ones((n_paths, 1)), eq])
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = (1.0 + leverage * rets) / (leverage * (1.0 + rets))
        breached = (eq[:, 1:] <= 0)
        if leverage > 1.0:
            breached = breached | (ratio < maintenance)
    else:
        # fixed loan: assets compound with the market, debt with interest
        assets = leverage * np.cumprod(1.0 + rets, axis=1)
        debt = (leverage - 1.0) * np.power(1.0 + f_daily, np.arange(1, n_days + 1))
        e = assets - debt[None, :]
        eq = np.hstack([np.ones((n_paths, 1)), np.maximum(e, 0.0)])
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(assets > 0, e / assets, -np.inf)
        breached = (e <= 0)
        if leverage > 1.0:
            breached = breached | (ratio < maintenance)

    n_paths, n_days = rets.shape
    any_breach = breached.any(axis=1)
    first = np.where(any_breach, breached.argmax(axis=1), n_days - 1)

    # freeze equity from the liquidation day onward
    day_idx = np.arange(n_days + 1)[None, :]
    frozen_at = (first + 1)[:, None]
    eq_frozen = np.where(day_idx <= frozen_at, eq, np.take_along_axis(
        eq, np.clip(frozen_at, 0, n_days), axis=1))
    eq_frozen = np.where(eq_frozen < 0, 0.0, eq_frozen)

    peak = np.maximum.accumulate(eq_frozen, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        dd = eq_frozen / peak - 1.0
    return eq_frozen[:, -1], np.nanmin(dd, axis=1), any_breach


def _block_bootstrap(
    rets: np.ndarray, n_paths: int, block: int, rng: np.random.Generator
) -> np.ndarray:
    """CIRCULAR block bootstrap: resample fixed-length contiguous blocks with
    wraparound, so volatility clustering (and therefore crash structure)
    survives the resampling.

    Deliberately named circular, not "stationary": the stationary bootstrap
    (Politis-Romano 1994) draws GEOMETRICALLY distributed block lengths,
    which this does not. Both are valid here; measured block sensitivity on
    SPY is mild (P(call) at 2x: 9.6% at 5d, 11.6% at 21d, 6.2% at 252d)."""
    n = len(rets)
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=(n_paths, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]) % n
    return rets[idx.reshape(n_paths, -1)[:, :n]]


def leverage_scan(
    daily_ret: pd.Series,
    leverages: tuple[float, ...] = (1.0, 1.25, 1.5, 2.0, 3.0),
    financing_rate: float = DEFAULT_FINANCING,
    maintenance: float = DEFAULT_MAINTENANCE,
    n_paths: int = 2000,
    block: int = 21,
    seed: int = 0,
    rebalance: str = "none",
) -> pd.DataFrame:
    """Realised and bootstrapped outcomes for a range of leverage levels.

    Realised columns describe the one path history actually delivered.
    Bootstrap columns describe the distribution: median and 5th-percentile
    terminal wealth, median max drawdown, and, the number that should drive
    sizing: the fraction of paths that hit a margin call.
    """
    r = daily_ret.dropna().to_numpy(dtype=float)
    if len(r) < 100:
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    boot = _block_bootstrap(r, n_paths, block, rng)

    rows = []
    for L in leverages:
        eq, liq = levered_path(r, L, financing_rate, maintenance,
                               rebalance=rebalance)
        term, dds, breached = levered_paths_vec(
            boot, L, financing_rate, maintenance, rebalance=rebalance)
        calls = int(breached.sum())
        rows.append({
            "leverage": L,
            "realised_total_return": float(eq[-1] - 1.0),
            "realised_max_dd": max_dd(eq),
            "realised_liquidated": liq is not None,
            "boot_median_return": float(np.median(term) - 1.0),
            "boot_p05_return": float(np.percentile(term, 5) - 1.0),
            "boot_median_max_dd": float(np.median(dds)),
            "boot_worst_max_dd": float(np.min(dds)),
            "margin_call_prob": calls / n_paths,
        })
    return pd.DataFrame(rows)


def max_safe_leverage(scan: pd.DataFrame, tolerance: float = 0.05) -> float:
    """Highest leverage in the scan whose bootstrapped margin-call probability
    stays at or below `tolerance`. Returns 1.0 when even unlevered exceeds it
    (which would indicate a badly mis-specified maintenance level)."""
    ok = scan[scan["margin_call_prob"] <= tolerance]
    return float(ok["leverage"].max()) if not ok.empty else 1.0
