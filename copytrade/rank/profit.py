"""Per-filer track record: profit analysis at ANY trade count.

The ranking layer (rank.py) gates 'copyable' on statistical significance
(t>2, n>=30) so the walk-forward isn't fooled by lucky streaks. But a filer
who buys three names a year and triples them is exactly who you want to find,
and that filer will never clear n>=30. This module answers the separate
question "how have this filer's picks actually done?" with NO minimum trade
count: it just labels low-sample records honestly.

Two complementary profit measures, because congress amounts are ranges:

  * position return: each BUY's price appreciation from its execution fill
    to the matched SELL fill (FIFO, same filer+ticker) or, if still open, to
    the last available close. Amount-independent, so it works for congress
    (ranges) and insiders (exact) alike. This is the honest "did the pick go
    up" measure.
  * realized $ P&L: only for filings with exact shares*price (Form 4);
    congress round trips report an approximate band from the amount ranges.

Everything is point-in-time: entries and exits both use execution_position
(first close after the FILING), never the transaction date. A win is a
positive net-of-costs position return.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from copytrade.backtest.costs import CostModel
from copytrade.backtest.engine import execution_position, parse_cutoff

TRADING_DAYS = 252


def confidence_label(n: int) -> str:
    """Small-sample honesty, not a gate."""
    if n >= 30:
        return "high"
    if n >= 10:
        return "medium"
    if n >= 3:
        return "low"
    return "anecdotal"


@dataclass
class Position:
    ticker: str
    entry_date: pd.Timestamp
    entry_price: float
    exit_date: pd.Timestamp | None
    exit_price: float | None
    ret_gross: float
    ret_net: float
    holding_days: int
    closed: bool


@dataclass
class TrackRecord:
    filer_id: int
    filer: str
    kind: str
    party: str | None
    n_buys_priced: int
    n_closed: int
    win_rate: float
    mean_return: float          # net, per position
    median_return: float
    total_pnl_per_10k: float    # $ from putting 10k into EVERY pick (additive
                                # positions overlap in time, so never compounded)
    annualized_per_pick: float  # mean position return annualized by avg hold
    avg_holding_days: float
    best: float
    worst: float
    confidence: str
    positions: list[Position] = field(default_factory=list)


def _match_positions(
    filer_trades: pd.DataFrame,
    ffill: pd.DataFrame,
    cal: pd.DatetimeIndex,
    cost: CostModel,
    same_day_close: bool,
    cutoff,
    horizon_cap: int,
) -> list[Position]:
    """FIFO-match a filer's buys to their later sells per ticker, pricing both
    legs at the first close after each filing. Unmatched buys are marked open
    at the last available close (or horizon_cap days out, whichever is first).
    `ffill` is the calendar-aligned forward-filled price matrix, built ONCE by
    the caller (never per filer)."""
    positions: list[Position] = []
    for ticker, g in filer_trades.groupby("ticker"):
        if ticker not in ffill.columns:
            continue
        col = ffill[ticker]
        buys, sells = [], []
        for r in g.itertuples():
            pos = execution_position(r.filing_datetime, cal, same_day_close, cutoff)
            if pos is None:
                continue
            (buys if r.side == "buy" else sells if r.side == "sell" else []).append(pos)
        buys.sort()
        sells.sort()
        si = 0
        for bpos in buys:
            entry_px = col.iloc[bpos]
            if not np.isfinite(entry_px) or entry_px <= 0:
                continue
            while si < len(sells) and sells[si] <= bpos:
                si += 1
            if si < len(sells):
                xpos = sells[si]
                si += 1
                closed = True
            else:
                xpos = min(bpos + horizon_cap, len(cal) - 1)
                closed = False
            exit_px = col.iloc[xpos]
            if not np.isfinite(exit_px) or exit_px <= 0:
                continue
            gross = exit_px / entry_px - 1.0
            net = gross - cost.roundtrip_frac(10_000.0, 10_000.0 / entry_px)
            positions.append(Position(
                ticker=ticker,
                entry_date=cal[bpos], entry_price=float(entry_px),
                exit_date=cal[xpos], exit_price=float(exit_px),
                ret_gross=float(gross), ret_net=float(net),
                holding_days=int(np.busday_count(cal[bpos].date(), cal[xpos].date())),
                closed=closed,
            ))
    return positions


def filer_track_records(
    trades: pd.DataFrame,
    prices: pd.DataFrame,
    bench: pd.Series,
    cost: CostModel,
    same_day_close: bool = False,
    cutoff: str = "15:30",
    horizon_cap: int = 252,
    min_buys: int = 1,
) -> pd.DataFrame:
    """One row per filer with a priced track record. NO significance gate
    `confidence` flags the sample size instead. Sorted by total compounded
    net return so the best pickers surface regardless of frequency."""
    if trades.empty or bench is None or len(bench) == 0:
        return pd.DataFrame()
    cal = bench.index
    cut = parse_cutoff(cutoff)
    ffill = prices.reindex(cal).ffill()   # built ONCE, shared across all filers
    records = []
    for (fid, name, kind), g in trades.groupby(["filer_id", "filer_name", "filer_kind"]):
        party = g["filer_party"].iloc[0] if "filer_party" in g.columns else None
        positions = _match_positions(g, ffill, cal, cost, same_day_close, cut, horizon_cap)
        if len(positions) < min_buys:
            continue
        rets = np.array([p.ret_net for p in positions])
        closed = [p for p in positions if p.closed]
        holds = np.array([p.holding_days for p in positions if p.holding_days > 0])
        avg_hold = float(holds.mean()) if len(holds) else np.nan
        mean_ret = float(rets.mean())
        # positions overlap in time -> aggregate ADDITIVELY, never compounded:
        # dollars from staking 10k on every pick
        total_pnl_10k = float(rets.sum() * 10_000.0)
        # per-pick annualized edge: what one average bet earns, scaled to a year
        ann = (
            float((1.0 + mean_ret) ** (TRADING_DAYS / avg_hold) - 1.0)
            if avg_hold and avg_hold > 0 and mean_ret > -1 else np.nan
        )
        records.append({
            "filer_id": int(fid), "filer": name, "kind": kind, "party": party,
            "n_buys_priced": len(positions), "n_closed": len(closed),
            "win_rate": float((rets > 0).mean()),
            "mean_return": mean_ret, "median_return": float(np.median(rets)),
            "total_pnl_per_10k": total_pnl_10k, "annualized_per_pick": ann,
            "avg_holding_days": avg_hold,
            "best": float(rets.max()), "worst": float(rets.min()),
            "confidence": confidence_label(len(positions)),
        })
    df = pd.DataFrame(records)
    if df.empty:
        return df
    return df.sort_values("total_pnl_per_10k", ascending=False).reset_index(drop=True)
