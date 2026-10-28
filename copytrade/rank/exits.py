"""Exit rules: how to hold and when to get out.

Every result so far used ONE exit: hold a fixed number of trading days, then
sell. That is a modelling convenience, not a strategy. Two positions with the
same 63-day return can have completely different paths, and a rule that reacts
to the path (trail a stop, take a profit, cut a loser) can change the realised
distribution even when the fixed-horizon mean is unchanged. So "no edge at 63d
fixed-horizon" does not by itself mean "no edge exploitable by any exit".

This module tests path-dependent exits against the fixed-horizon baseline.

WHAT IT OPERATES ON. `run_event_study` already returns `paths`: a
(n_trades, H+1) array of GROSS cumulative EXCESS return versus the benchmark,
one row per trade, starting at 0 on the execution close. Exits are simulated on
those rows, so execution still comes from the single `execution_position`
choke point: this adds no second execution path.

Because the paths are excess-vs-benchmark, every rule here is a RELATIVE rule:
a "trailing stop" means giving back that much relative to the benchmark, not
in absolute price. That is the correct frame for a strategy whose stated goal
is beating an ETF, but it is a different object from a broker-side stop order,
and it is deliberately not called one.

Round-trip costs are subtracted per trade from `per_trade['roundtrip_cost']`,
so every number below is net.

THE DREDGING WARNING APPLIES HERE TOO, HARDER. Exit rules have free
parameters, and sweeping them across a grid will always surface a "best" cell.
The same Benjamini-Hochberg correction used in `timescale.py` is applied
across the whole rule grid, and a rule is only reported as a finding if it
beats the fixed-horizon baseline by an amount that survives correction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from copytrade.rank.timescale import benjamini_hochberg, block_test


def _first_true(mask: np.ndarray) -> int | None:
    """Index of the first True, or None. Row-wise helper for trigger search."""
    idx = np.flatnonzero(mask)
    return int(idx[0]) if idx.size else None


def apply_exit(
    path: np.ndarray,
    trail: float | None = None,
    take_profit: float | None = None,
    stop_loss: float | None = None,
    max_h: int | None = None,
    require_full: bool = True,
) -> tuple[float, int]:
    """Simulate one trade's exit. Returns (gross excess return, holding days).

    Evaluated day by day in order, so whichever trigger fires first wins, no
    look-ahead at the full path. `max_h` caps the hold (a time stop).

    `require_full` (default True) makes a trade whose `max_h` window has NOT
    yet elapsed return NaN, exactly as `run_event_study` writes NaN into
    `h{max_h}`. Without it the position was closed at the last available bar
    and reported as if it were a full-horizon hold: on the live sample the 55
    trades entered in the final three months were held 3-56 days and averaged
    +4.91% against -0.35% for the rest, so the "fixed 63d baseline" printed
    n=12750/-0.33% where every other command reports n=12695/-0.3509%.

    Trailing NaNs otherwise mark days beyond available price history (a
    delisted name); such a position is closed at the last valid observation.
    """
    valid = np.flatnonzero(~np.isnan(path))
    if valid.size == 0:
        return np.nan, 0
    last = int(valid[-1])
    if max_h is not None:
        if require_full and last < int(max_h):
            # horizon has not elapsed for this trade, match the engine's NaN
            return np.nan, 0
        last = min(last, int(max_h))
    seg = path[: last + 1]

    trigger = last
    running_max = np.fmax.accumulate(np.nan_to_num(seg, nan=-np.inf))
    cands: list[int] = []
    if trail is not None:
        t = _first_true(seg <= running_max - trail)
        if t is not None:
            cands.append(t)
    if take_profit is not None:
        t = _first_true(seg >= take_profit)
        if t is not None:
            cands.append(t)
    if stop_loss is not None:
        t = _first_true(seg <= -stop_loss)
        if t is not None:
            cands.append(t)
    # day 0 is the entry close; a rule cannot fire before the position exists
    cands = [c for c in cands if c > 0]
    if cands:
        trigger = min(min(cands), last)
    val = seg[trigger]
    if np.isnan(val):
        val = seg[valid[valid <= trigger][-1]] if (valid <= trigger).any() else np.nan
    return float(val), int(trigger)


def run_rule(
    paths: np.ndarray,
    per_trade: pd.DataFrame,
    trail: float | None = None,
    take_profit: float | None = None,
    stop_loss: float | None = None,
    max_h: int | None = None,
    require_full: bool = True,
) -> pd.DataFrame:
    """Apply one exit rule to every trade. Returns a per-trade frame with the
    NET excess return and the realised holding period.

    `require_full` propagates to `apply_exit`; it keeps the trade universe
    identical to the engine's `h{max_h}` column so the baseline is the same
    object every other command reports.
    """
    out = np.full(len(paths), np.nan)
    held = np.zeros(len(paths), dtype=int)
    for i in range(len(paths)):
        out[i], held[i] = apply_exit(paths[i], trail, take_profit, stop_loss,
                                     max_h, require_full=require_full)
    cost = per_trade["roundtrip_cost"].to_numpy(dtype=float)
    return pd.DataFrame({
        "net_excess": out - cost,
        "held_days": held,
        "entry_date": per_trade["entry_date"].to_numpy(),
    })


def rule_grid(
    paths: np.ndarray,
    per_trade: pd.DataFrame,
    baseline_h: int,
    trails: tuple[float, ...] = (0.05, 0.10, 0.15),
    take_profits: tuple[float, ...] = (0.10, 0.20),
    stop_losses: tuple[float, ...] = (0.10, 0.20),
) -> pd.DataFrame:
    """Compare exit rules against the fixed-horizon baseline.

    The baseline is 'hold `baseline_h` days', which is what every earlier
    result used. Each alternative is tested two ways:

      * its own mean net excess return and clustered t-stat (is it profitable
        at all?), and
      * the PAIRED difference against the baseline on the same trades, with
        its own clustered t-stat (does the rule ADD anything?). The paired
        test is the one that matters, a rule can look profitable purely by
        inheriting whatever the baseline already had.

    BH-FDR is applied across the paired tests, since that is the family of
    comparisons being searched.
    """
    base = run_rule(paths, per_trade, max_h=baseline_h)
    # Keyed on the PARAMETER TUPLE, not the formatted label. Percent-formatting
    # collides (0.10 and 0.104 both render "10%"), which silently dropped one
    # rule from the grid AND shrank the BH family below the number of rules
    # actually searched: under-correcting the survivors.
    rules: list[tuple[str, dict]] = [
        (f"fixed {baseline_h}d (baseline)", {"max_h": baseline_h})
    ]
    seen: set[tuple] = set()

    def _add(label: str, kw: dict) -> None:
        key = (kw.get("trail"), kw.get("take_profit"), kw.get("stop_loss"))
        if key in seen:
            return
        seen.add(key)
        rules.append((label, kw))

    for tr in trails:
        _add(f"trail {tr*100:.4g}%", {"trail": tr, "max_h": baseline_h})
    for tp in take_profits:
        _add(f"take-profit {tp*100:.4g}%", {"take_profit": tp, "max_h": baseline_h})
    for sl in stop_losses:
        _add(f"stop-loss {sl*100:.4g}%", {"stop_loss": sl, "max_h": baseline_h})
    for tr in trails:
        for tp in take_profits:
            _add(f"trail {tr*100:.4g}% + TP {tp*100:.4g}%",
                 {"trail": tr, "take_profit": tp, "max_h": baseline_h})

    rows: list[dict] = []
    for label, kw in rules:
        r = run_rule(paths, per_trade, **kw)
        s = r["net_excess"].dropna()
        if s.empty:
            continue
        # Overlap-aware: these are `baseline_h`-day holds, so positions opened
        # in consecutive months share most of their window and monthly
        # clustering would treat them as independent draws, measured to
        # inflate |t| by ~1.9x at h=63. Cluster on non-overlapping blocks as
        # long as the holding period instead (same correction as timescale).
        t_own, p_own, _ = block_test(s, r.loc[s.index, "entry_date"], baseline_h)
        is_base = label.endswith("(baseline)")
        if is_base:
            t_d, p_d = np.nan, np.nan
        else:
            diff = (r["net_excess"] - base["net_excess"]).dropna()
            t_d, p_d, _ = block_test(diff, r.loc[diff.index, "entry_date"],
                                     baseline_h)
        rows.append({
            "rule": label,
            "n": int(len(s)),
            "mean": float(s.mean()),
            "median": float(s.median()),
            "hit": float((s > 0).mean()),
            "mean_hold": float(r.loc[s.index, "held_days"].mean()),
            "t_own": t_own,
            "vs_base": np.nan if is_base else float(
                (r["net_excess"] - base["net_excess"]).dropna().mean()),
            "t_vs_base": t_d,
            "p_vs_base": p_d,
        })
    grid = pd.DataFrame(rows)
    if grid.empty:
        return grid
    grid["q_fdr"] = benjamini_hochberg(grid["p_vs_base"].values)
    grid["beats_baseline"] = (grid["q_fdr"] < 0.05) & (grid["vs_base"] > 0)
    return grid.sort_values("mean", ascending=False)
