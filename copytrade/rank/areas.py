"""Area / holdings attribution: not 'who to copy' but 'what do these people
pursue, and where are their picks actually good?'

Inputs are the per-trade frame from the event study (one row per priced BUY,
with a net forward-return column h{horizon}) joined to a ticker->sector map.
Everything here is point-in-time already, the returns come from filing-anchored
execution: so a sector's mean return is 'what following buys in this area,
after disclosure, net of costs, would have returned over the horizon'.
"""

from __future__ import annotations

import pandas as pd


def _attribute(df: pd.DataFrame, by: str, hcol: str) -> pd.DataFrame:
    g = df.groupby(by).agg(
        n_buys=("trade_id", "count"),
        n_filers=("filer_id", "nunique"),
        n_tickers=("ticker", "nunique"),
        notional=("amount", "sum"),
        win_rate=(hcol, lambda s: float((s > 0).mean())),
        mean_ret=(hcol, "mean"),
        median_ret=(hcol, "median"),
    ).reset_index()
    return g


def sector_attribution(per_trade: pd.DataFrame, sector_map: pd.DataFrame,
                       hcol: str, min_buys: int = 20) -> pd.DataFrame:
    """Per sector: how much activity (n_buys, notional, breadth) and how it
    performed (win rate, mean/median net forward return). Sorted by mean
    return so the 'areas they're actually successful in' surface at the top."""
    if per_trade.empty:
        return pd.DataFrame()
    df = per_trade.dropna(subset=[hcol]).copy()
    if sector_map is not None and not sector_map.empty:
        df = df.merge(sector_map[["ticker", "sector"]], on="ticker", how="left")
    else:
        df["sector"] = None
    df["sector"] = df["sector"].fillna("Unknown")
    g = _attribute(df, "sector", hcol)
    return g[g["n_buys"] >= min_buys].sort_values("mean_ret", ascending=False).reset_index(drop=True)


def ticker_leaders(per_trade: pd.DataFrame, hcol: str, min_buys: int = 8,
                   top: int = 15) -> dict[str, pd.DataFrame]:
    """Best / worst / most-crowded individual names. 'best' and 'worst' need a
    minimum sample; 'most_traded' is raw popularity (what they pile into)."""
    if per_trade.empty:
        return {"best": pd.DataFrame(), "worst": pd.DataFrame(), "most_traded": pd.DataFrame()}
    df = per_trade.dropna(subset=[hcol])
    g = df.groupby("ticker").agg(
        n_buys=("trade_id", "count"),
        n_filers=("filer_id", "nunique"),
        win_rate=(hcol, lambda s: float((s > 0).mean())),
        mean_ret=(hcol, "mean"),
    ).reset_index()
    sample = g[g["n_buys"] >= min_buys]
    return {
        "best": sample.sort_values("mean_ret", ascending=False).head(top).reset_index(drop=True),
        "worst": sample.sort_values("mean_ret").head(top).reset_index(drop=True),
        "most_traded": g.sort_values("n_buys", ascending=False).head(top).reset_index(drop=True),
    }


def filer_focus(per_trade: pd.DataFrame, sector_map: pd.DataFrame, hcol: str,
                filer_name: str, top: int = 8) -> dict[str, pd.DataFrame]:
    """One filer's footprint: the tickers and sectors they pursue most, each
    with the net forward return those picks delivered, 'what are they into,
    and does it work for them?'"""
    df = per_trade[per_trade["filer_name"] == filer_name].dropna(subset=[hcol]).copy()
    if df.empty:
        return {"tickers": pd.DataFrame(), "sectors": pd.DataFrame()}
    if sector_map is not None and not sector_map.empty:
        df = df.merge(sector_map[["ticker", "sector"]], on="ticker", how="left")
    else:
        df["sector"] = None
    df["sector"] = df["sector"].fillna("Unknown")
    tk = df.groupby("ticker").agg(
        n=("trade_id", "count"), win_rate=(hcol, lambda s: float((s > 0).mean())),
        mean_ret=(hcol, "mean"),
    ).reset_index().sort_values("n", ascending=False).head(top).reset_index(drop=True)
    sec = df.groupby("sector").agg(
        n=("trade_id", "count"), mean_ret=(hcol, "mean"),
    ).reset_index().sort_values("n", ascending=False).reset_index(drop=True)
    return {"tickers": tk, "sectors": sec}
