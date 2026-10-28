"""Derivatives & futures: a SEPARATE section for leveraged bets.

Stocks are copyable and priceable, so they drive the main backtest. Options
and futures are neither cleanly copyable (range-based disclosures, no strike)
nor priceable from what filers report, but the DIRECTION of a powerful
person's leveraged bet is still information. This module frames the three
instrument classes side by side, honestly:

  * options (congress PTRs, 13F put/call holdings, insider derivative table):
    we cannot price the option, but we CAN measure the UNDERLYING's forward
    excess return from the filing date, signed by the bet's direction
    (long call / short put = bullish → +underlying; long put = bearish →
    −underlying). That is a directional hit-rate, explicitly NOT an option
    P&L: the leverage that makes these bets lucrative is unmeasured.
  * futures: no named individual discloses one, so the proxy is CFTC COT
    large-speculator net positioning (ingest/cftc.py), shown as a snapshot,
    not a copy signal.

The whole point is the FRAMING: how does a filer's directional edge on
options compare to their stock edge? A member whose options win but whose
stocks don't is telling you their edge is timing/leverage, not stock picking.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from copytrade.backtest.costs import CostModel
from copytrade.backtest.engine import execution_position, parse_cutoff

OPTION_INSTRUMENTS = ("option", "call_option", "put_option", "stock_option")


def classify_option(instrument: str, asset_desc: str | None) -> tuple[str, int]:
    """Return (kind, direction_sign) where kind ∈ {call, put, option} and
    sign is +1 for a bullish bet on the underlying, -1 for bearish, from the
    text of the disclosure. Buying a call or writing a put is bullish; buying
    a put is bearish. When ambiguous, assume the long/bullish reading (+1)."""
    text = f"{instrument} {asset_desc or ''}".lower()
    if "put" in text:
        # 'sold put' / 'short put' is bullish; a bare/'bought' put is bearish
        bullish = any(w in text for w in ("sold", "short", "write", "written"))
        return "put", (1 if bullish else -1)
    if "call" in text:
        bearish = any(w in text for w in ("sold", "short", "write", "written"))
        return "call", (-1 if bearish else 1)
    return "option", 1


def option_direction_study(
    trades: pd.DataFrame,
    prices: pd.DataFrame,
    bench: pd.Series,
    horizon: int = 63,
    same_day_close: bool = False,
    cutoff: str = "15:30",
) -> pd.DataFrame:
    """Per option trade: signed forward excess return of the UNDERLYING over
    `horizon`, anchored at filing. Returns a per-trade frame (empty if no
    option trades / no prices)."""
    opts = trades[trades["instrument"].isin(OPTION_INSTRUMENTS)].copy()
    if opts.empty or bench is None or len(bench) == 0:
        return pd.DataFrame()
    cal = bench.index
    cut = parse_cutoff(cutoff)
    ffill = prices.reindex(cal).ffill()
    bench_v = bench.values
    rows = []
    for t in opts.itertuples():
        if t.ticker not in ffill.columns or pd.isna(t.filing_datetime):
            continue
        pos = execution_position(t.filing_datetime, cal, same_day_close, cut)
        if pos is None:
            continue
        end = min(pos + horizon, len(cal) - 1)
        col = ffill[t.ticker].values
        p0 = col[pos]
        if not np.isfinite(p0) or p0 <= 0:
            continue
        under = col[end] / p0 - 1.0
        bmk = bench_v[end] / bench_v[pos] - 1.0
        # direction comes from the persisted instrument (call_option/put_option),
        # the only long/short signal the disclosures reliably encode, the raw
        # asset text is not carried on the trades row, so no description is
        # passed here (written/covered-option nuance is genuinely unavailable)
        kind, sign = classify_option(t.instrument, None)
        excess = (under - bmk) * sign
        rows.append({
            "filer_id": t.filer_id, "filer": t.filer_name, "kind": t.filer_kind,
            "ticker": t.ticker, "opt_kind": kind, "direction": sign,
            "underlying_excess": excess, "filing": t.filing_datetime,
        })
    return pd.DataFrame(rows)


def instrument_framing(trades: pd.DataFrame) -> pd.DataFrame:
    """Counts by instrument class across all filers, the top-level framing
    of how much of the disclosed activity is stock vs option vs other."""
    if trades.empty:
        return pd.DataFrame()
    def bucket(instr: str) -> str:
        if instr == "stock":
            return "stock"
        if instr in OPTION_INSTRUMENTS:
            return "option"
        if instr == "ca_material_change":
            return "canada_material_change"
        return instr or "unknown"
    g = trades.assign(cls=trades["instrument"].map(bucket))
    out = g.groupby("cls").agg(
        trades=("trade_id", "count"),
        filers=("filer_id", "nunique"),
        tickers=("ticker", "nunique"),
    ).reset_index().sort_values("trades", ascending=False)
    return out


def stock_vs_option_by_filer(
    stock_track: pd.DataFrame, opt_study: pd.DataFrame, min_each: int = 3
) -> pd.DataFrame:
    """Join a filer's stock track record (mean return) against their option
    directional edge: the money question: is their edge stock-picking or
    leverage/timing?"""
    if opt_study.empty:
        return pd.DataFrame()
    opt = (opt_study.groupby(["filer_id", "filer"])
           .agg(opt_n=("underlying_excess", "size"),
                opt_dir_excess=("underlying_excess", "mean"),
                opt_hit=("underlying_excess", lambda s: float((s > 0).mean())))
           .reset_index())
    if stock_track is None or stock_track.empty:
        opt["stock_mean_return"] = np.nan
        return opt[opt["opt_n"] >= min_each].sort_values("opt_dir_excess", ascending=False)
    st = stock_track[["filer", "mean_return"]].rename(columns={"mean_return": "stock_mean_return"})
    merged = opt.merge(st, on="filer", how="left")
    return merged[merged["opt_n"] >= min_each].sort_values("opt_dir_excess", ascending=False)


def cot_snapshot(conn, markets: tuple[str, ...] = ("S&P 500", "NASDAQ", "GOLD", "CRUDE OIL")) -> pd.DataFrame:
    """Latest CFTC large-speculator net positioning for headline markets."""
    rows = []
    for m in markets:
        r = conn.execute(
            "SELECT report_date, market, noncomm_long, noncomm_short, open_interest "
            "FROM cot_positioning WHERE UPPER(market) LIKE ? ORDER BY report_date DESC LIMIT 1",
            (f"%{m.upper()}%",),
        ).fetchone()
        if r:
            long_, short_ = r["noncomm_long"] or 0, r["noncomm_short"] or 0
            rows.append({
                "market": r["market"], "as_of": r["report_date"],
                "spec_long": long_, "spec_short": short_,
                "spec_net": long_ - short_,
                "spec_net_pct_oi": (long_ - short_) / r["open_interest"]
                if r["open_interest"] else np.nan,
            })
    return pd.DataFrame(rows)
