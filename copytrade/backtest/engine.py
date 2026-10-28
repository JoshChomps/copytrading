"""Point-in-time backtest engine.

THE rule (plan §0.2): a trade is only actionable at its filing timestamp.
Execution is the first market close AFTER filing_datetime (optionally the
same day's close when filed before the cutoff, a sensitivity toggle).
`execution_position` is the single choke point for signal->execution mapping;
any code path that peeks at transaction_date for execution is a bug. The
transaction-date anchor exists ONLY to draw the hypothetical zero-lag decay
curve for comparison.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import time as dtime

import numpy as np
import pandas as pd

from copytrade.backtest.costs import CostModel


class LookaheadError(AssertionError):
    pass


def parse_cutoff(s: str) -> dtime:
    hh, mm = s.split(":")
    return dtime(int(hh), int(mm))


def execution_position(
    signal_dt: pd.Timestamp,
    cal: pd.DatetimeIndex,
    same_day_close: bool = False,
    cutoff: dtime = dtime(15, 30),
) -> int | None:
    """Index into `cal` of the close where a signal at `signal_dt` executes.

    Conservative default: next trading day's close. If `same_day_close` and
    the signal lands on a trading day before `cutoff` (ET), that day's close.
    Returns None if execution would fall beyond the calendar (too recent).
    """
    day = signal_dt.normalize()
    pos = int(cal.searchsorted(day))
    if pos < len(cal) and cal[pos] == day:
        if same_day_close and signal_dt.time() < cutoff:
            return pos
        pos += 1
    if pos >= len(cal):
        return None
    # look-ahead guard: executing close must not precede the signal
    if cal[pos] < day:
        raise LookaheadError(f"execution {cal[pos]} precedes signal {signal_dt}")
    return pos


def amount_for_basis(row, basis: str, fallback: float) -> float:
    """Congress amounts are RANGES, never treat the midpoint as exact.
    basis selects low/mid/high for sensitivity runs (plan §7).
    Explicit None checks: `or` would silently drop a legitimate 0.0 bound,
    and open-ended buckets ('Over $50,000,000') have only one bound."""
    low = row.amount_low if pd.notna(row.amount_low) else None
    high = row.amount_high if pd.notna(row.amount_high) else None
    if low is None and high is None:
        return fallback
    lo = low if low is not None else high
    hi = high if high is not None else low
    if basis == "low":
        return lo
    if basis == "high":
        return hi
    return (lo + hi) / 2


@dataclass
class EventStudyResult:
    per_trade: pd.DataFrame          # one row per priced trade, net returns per horizon
    paths: np.ndarray                # (n_trades, H+1) GROSS cumulative excess return paths
    max_h: int
    unpriceable: int
    total: int

    def decay_curve(self, mask: pd.Series | None = None) -> np.ndarray:
        m = self.paths if mask is None else self.paths[np.asarray(mask, dtype=bool)]
        if len(m) == 0:
            return np.full(self.max_h + 1, np.nan)
        with np.errstate(invalid="ignore"):
            return np.nanmean(m, axis=0)


def run_event_study(
    trades: pd.DataFrame,
    prices: pd.DataFrame,
    bench: pd.Series,
    cost_model: CostModel,
    horizons: list[int],
    anchor: str = "filing",
    same_day_close: bool = False,
    cutoff: str = "15:30",
    amount_basis: str = "mid",
    trade_notional: float = 10_000.0,
) -> EventStudyResult:
    """Event study of post-signal excess returns vs the benchmark.

    anchor='filing'      -> executes after filing_datetime (the real strategy)
    anchor='transaction' -> hypothetical zero-lag copy at transaction-day close
                            (ONLY for measuring alpha destroyed by the lag)

    per_trade horizon columns are NET of the cost model's round trip; `paths`
    are gross information-decay curves (costs are a constant shift, applied in
    tables and portfolio sims, not in the decay geometry).
    """
    if anchor not in ("filing", "transaction"):
        raise ValueError(anchor)
    cal = bench.index
    max_h = max(horizons)
    px = prices.reindex(cal)
    px_ffill = px.ffill()
    col_pos = {c: i for i, c in enumerate(px.columns)}
    raw_vals = px.values
    ffill_vals = px_ffill.values
    bench_vals = bench.values
    cutoff_t = parse_cutoff(cutoff)

    rows: list[dict] = []
    paths: list[np.ndarray] = []
    unpriceable = 0

    for t in trades.itertuples():
        if anchor == "filing":
            signal_dt = t.filing_datetime
            pos = execution_position(signal_dt, cal, same_day_close, cutoff_t)
        else:
            # zero-lag hypothetical: same-day close on the transaction date
            signal_dt = t.transaction_date
            if pd.isna(signal_dt):
                unpriceable += 1
                continue
            pos = execution_position(signal_dt, cal, same_day_close=True, cutoff=dtime(23, 59))
        if pos is None:
            continue
        ci = col_pos.get(t.ticker)
        if ci is None:
            unpriceable += 1
            continue
        p0 = raw_vals[pos, ci]
        # A non-positive entry price is a provider error, not a price. Dividing
        # by it flips the sign of the return ratio and produces impossible
        # values that survive every downstream average. Guarded here as well as
        # at the storage layer because one corrupt series is enough to move a
        # whole-sample mean by hundreds of percent.
        if np.isnan(p0) or p0 <= 0:
            unpriceable += 1
            continue

        end = min(pos + max_h, len(cal) - 1)
        seg = ffill_vals[pos : end + 1, ci]
        # A non-positive price part-way along the path is a provider error, but
        # it must be treated as END OF DATA (like a delisting) rather than as a
        # reason to reject the trade. Rejecting outright would make a 1-day
        # horizon's SAMPLE depend on prices up to max(horizons) days in the
        # FUTURE: with the full sweep grid a single corrupt tick ~500 days out
        # silently removed the trade from the short band too, conditioning
        # those results on the name surviving another two years.
        bad = np.flatnonzero(~(seg > 0))
        if bad.size:
            if bad[0] == 0:            # cannot happen (p0 > 0), guarded anyway
                unpriceable += 1
                continue
            seg = seg[: bad[0]]
            end = pos + len(seg) - 1
        upath = seg / p0 - 1.0
        bpath = bench_vals[pos : end + 1] / bench_vals[pos] - 1.0
        ex = np.full(max_h + 1, np.nan)
        ex[: end - pos + 1] = upath - bpath

        amount = amount_for_basis(t, amount_basis, trade_notional)
        rt = cost_model.roundtrip_frac(amount, amount / p0)

        row = {
            "trade_id": t.trade_id,
            "filer_id": t.filer_id,
            "filer_name": t.filer_name,
            "kind": t.filer_kind,
            "side": t.side,
            "ticker": t.ticker,
            "entry_pos": pos,
            "entry_date": cal[pos],
            "amount": amount,
            # The execution price is retained so a caller can REPRICE this
            # trade under a different cost model without re-running the whole
            # event study: shares = amount / entry_price is the only other
            # input `roundtrip_frac` needs, and the gross paths are identical
            # across models by construction.
            "entry_price": float(p0),
            "lag_days": (
                np.busday_count(t.transaction_date.date(), cal[pos].date())
                if pd.notna(t.transaction_date) else np.nan
            ),
            "roundtrip_cost": rt,
        }
        for h in horizons:
            row[f"h{h}"] = ex[h] - rt if h <= end - pos and not np.isnan(ex[h]) else np.nan
        rows.append(row)
        paths.append(ex)

    per_trade = pd.DataFrame(rows)
    return EventStudyResult(
        per_trade=per_trade,
        paths=np.vstack(paths) if paths else np.empty((0, max_h + 1)),
        max_h=max_h,
        unpriceable=unpriceable,
        total=len(trades),
    )


# ------------------------------------------------------------ portfolio ----

@dataclass
class Position:
    ticker: str
    col: int
    units: float
    entry_pos: int
    exit_pos: int
    entry_value: float
    filer_name: str
    entry_price: float


@dataclass
class WalkForwardResult:
    equity: pd.Series
    daily_returns: pd.Series
    picks: dict[int, list[str]]           # test year -> filer names
    taxlots: pd.DataFrame                 # closed-lot ledger (CRA capital-gains export)
    skipped_full: int = 0
    stale_exits: int = 0                  # exits priced off a forward-filled (possibly delisted) close
    stats: dict = field(default_factory=dict)


def walk_forward(
    trades: pd.DataFrame,
    prices: pd.DataFrame,
    bench: pd.Series,
    cost_model: CostModel,
    horizon: int = 63,
    top_n: int = 10,
    min_trades: int = 5,
    max_weight: float = 0.05,
    rank_years: int = 2,
    same_day_close: bool = False,
    cutoff: str = "15:30",
    amount_basis: str = "mid",
    trade_notional: float = 10_000.0,
    portfolio_value: float = 250_000.0,
    exit_on_filer_sell: bool = False,
    copy_all: bool = False,
    only_filers: set[str] | None = None,
) -> WalkForwardResult | None:
    """Walk-forward 'copy the top-N filers' simulation.

    Equity is simulated normalized to 1.0; `portfolio_value` sets the dollar
    size used to compute fixed/min/per-share commissions realistically.

    Rank filers on trailing net lag-alpha over years [t-rank_years, t-1], then
    copy their BUY filings during year t, selection uses only past data, so
    no look-ahead in filer choice either (plan §4.2).

    exit_on_filer_sell: also copy the EXIT, close a position at the first
    close after the same filer files a SELL in that ticker (point-in-time:
    the sell's filing anchor, never its transaction date). The fixed horizon
    remains the backstop.
    copy_all: no selection at all, copy every filer's buys from the first
    year onward. This is the 'congress index' baseline (what NANC/KRUZ-style
    products do); if top-N selection cannot beat it, the ranking layer adds
    nothing.
    only_filers: mirror EXACTLY these filers (by name), copy every buy they
    file, no ranking, no significance gate. This is the honest 'copy one
    famous member' test (e.g. a Pelosi mirror): concentrated, no look-ahead
    in timing (still filing-anchored), but no pretence that the filer was
    selectable in advance. Implies copy_all-style year coverage.
    """
    buys = trades[trades["side"] == "buy"]
    if only_filers:
        buys = buys[buys["filer_name"].isin(only_filers)]
        copy_all = True   # no ranking window needed when the filer is named
    if buys.empty:
        return None
    es = run_event_study(
        buys, prices, bench, cost_model, [horizon], anchor="filing",
        same_day_close=same_day_close, cutoff=cutoff,
        amount_basis=amount_basis, trade_notional=trade_notional,
    )
    pt = es.per_trade
    if pt.empty:
        return None
    pt = pt.assign(entry_year=pt["entry_date"].dt.year)
    year_set = set(pt["entry_year"].unique())
    if copy_all:
        test_years = sorted(year_set)
    else:
        test_years = [
            y for y in sorted(year_set)
            if any(v in year_set for v in range(y - rank_years, y))
        ]
    if not test_years:
        return None

    cal = bench.index
    px = prices.reindex(cal)
    raw_vals = px.values                  # NaN where no real print (delisted/halted)
    px = px.ffill()
    col_pos = {c: i for i, c in enumerate(px.columns)}
    vals = px.values

    picks: dict[int, list[str]] = {}
    med = pt["amount"].median()
    # float(nan) is truthy: guard explicitly or an all-NaN amount column
    # poisons every position weight downstream
    median_amount = float(med) if pd.notna(med) and med > 0 else trade_notional

    # copy-the-exit: per (filer, ticker) sorted execution positions of the
    # filer's OWN sell filings: the earliest one after entry trims exit_pos
    sell_pos: dict[tuple[int, str], list[int]] = {}
    if exit_on_filer_sell:
        cutoff_t = parse_cutoff(cutoff)
        sells = trades[trades["side"] == "sell"].dropna(subset=["filing_datetime"])
        for s in sells.itertuples():
            pos = execution_position(s.filing_datetime, cal, same_day_close, cutoff_t)
            if pos is not None:
                sell_pos.setdefault((s.filer_id, s.ticker), []).append(pos)
        for v in sell_pos.values():
            v.sort()

    def planned_exit(filer_id: int, ticker: str, entry_pos: int) -> int:
        exit_pos = entry_pos + horizon
        for p in sell_pos.get((filer_id, ticker), ()):
            if p > entry_pos:
                return min(exit_pos, p)
        return exit_pos

    planned: list[tuple[int, int, float, str, str, float]] = []
    for y in test_years:
        if copy_all:
            picks[y] = ["(all filers)"]
            test = pt[pt["entry_year"] == y]
        else:
            # only training trades whose horizon RESOLVED before the test year
            # starts may inform the ranking, otherwise selection peeks ahead
            year_start_pos = int(cal.searchsorted(pd.Timestamp(year=y, month=1, day=1)))
            train = pt[
                pt["entry_year"].isin(range(y - rank_years, y))
                & (pt["entry_pos"] + horizon <= year_start_pos)
            ]
            scores = (
                train.groupby(["filer_id", "filer_name"])[f"h{horizon}"]
                .agg(["mean", "count"])
                .reset_index()
            )
            scores = scores[scores["count"] >= min_trades].sort_values("mean", ascending=False)
            top = scores.head(top_n)
            picks[y] = top["filer_name"].tolist()
            top_ids = set(top["filer_id"])
            test = pt[(pt["entry_year"] == y) & (pt["filer_id"].isin(top_ids))]
        for t in test.itertuples():
            w = max_weight * float(np.clip(t.amount / median_amount, 0.25, 1.0))
            planned.append((int(t.entry_pos),
                            planned_exit(t.filer_id, t.ticker, int(t.entry_pos)),
                            w, t.ticker, t.filer_name, t.amount))

    if not planned:
        return None
    planned.sort(key=lambda x: x[0])

    start_pos = planned[0][0]
    end_pos = min(max(p[1] for p in planned), len(cal) - 1)

    equity = 1.0
    cash = 1.0
    open_pos: list[Position] = []
    lots: list[dict] = []
    eq_path = []
    skipped_full = 0
    stale_exits = 0
    plan_i = 0

    for i in range(start_pos, end_pos + 1):
        # 1) exits at today's close
        still: list[Position] = []
        for p in open_pos:
            if p.exit_pos <= i or i == end_pos:
                exit_px = vals[i, p.col]
                if np.isnan(raw_vals[i, p.col]):
                    # no real print today: the fill uses a forward-filled price,
                    # which overstates liquidity for delisted names, counted
                    # and disclosed, not hidden
                    stale_exits += 1
                value = p.units * exit_px
                fee = cost_model.cost_per_side(
                    value * portfolio_value, value * portfolio_value / max(exit_px, 1e-9), "sell"
                ) / portfolio_value
                cash += value - fee
                lots.append({
                    "ticker": p.ticker, "filer": p.filer_name,
                    "entry_date": cal[p.entry_pos].date(), "exit_date": cal[i].date(),
                    "entry_value": round(p.entry_value, 6), "exit_value": round(value - fee, 6),
                    "pnl": round(value - fee - p.entry_value, 6),
                })
            else:
                still.append(p)
        open_pos = still

        # 2) entries at today's close
        while plan_i < len(planned) and planned[plan_i][0] == i:
            entry_pos_, exit_pos_, w, ticker, filer, amount = planned[plan_i]
            plan_i += 1
            ci = col_pos.get(ticker)
            if ci is None or np.isnan(vals[i, ci]):
                continue
            invested = sum(p.units * vals[i, p.col] for p in open_pos)
            avail = cash
            target = w * (invested + cash)
            if target > avail:
                if avail < 0.25 * target:
                    skipped_full += 1
                    continue
                target = avail
            entry_px = vals[i, ci]
            fee = cost_model.cost_per_side(
                target * portfolio_value, target * portfolio_value / entry_px, "buy"
            ) / portfolio_value
            if target - fee <= 0:
                skipped_full += 1
                continue
            cash -= target
            open_pos.append(Position(
                ticker=ticker, col=ci, units=(target - fee) / entry_px,
                entry_pos=i, exit_pos=min(exit_pos_, end_pos),
                entry_value=target, filer_name=filer, entry_price=entry_px,
            ))

        equity = cash + sum(p.units * vals[i, p.col] for p in open_pos)
        eq_path.append(equity)

    idx = cal[start_pos : end_pos + 1]
    eq = pd.Series(eq_path, index=idx, name="copytrade")
    rets = eq.pct_change().fillna(0.0)
    return WalkForwardResult(
        equity=eq, daily_returns=rets, picks=picks,
        taxlots=pd.DataFrame(lots), skipped_full=skipped_full,
        stale_exits=stale_exits,
    )
