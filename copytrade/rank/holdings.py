"""Disclosed net positions per filer, "what is this member actually in?"

WHY THIS EXISTS. Every consumer-facing tracker in this space (InsiderWave,
Quiver, Unusual Whales) leads with a PORTFOLIO view, and the question a reader
actually arrives with is phrased that way: "Cleo Fields holds TSMC." This
project had no answer to it. It stores transactions and analyses transactions;
asking it what someone HOLDS produced nothing.

THE HONEST VERSION OF THAT FEATURE, AND WHY IT IS NOT A PORTFOLIO.

A Periodic Transaction Report discloses TRANSACTIONS, not holdings. Three
consequences follow, and any product that renders a PTR stream as a "portfolio"
without stating them is overclaiming:

  1. **Positions opened before the window are invisible.** A member who bought
     Apple in 2019 and never traded it again has no PTR row, so this module
     cannot see it. Annual Financial Disclosure filings do list holdings, this
     project does not ingest them.
  2. **Sizes are RANGES.** "$100,001 - $250,000" is the disclosure; the true
     figure is unknown. Netting a buy band against a sell band therefore gives
     a BAND, never a number, and the bands widen with every transaction. A
     single scalar "position size" would be fabricated precision.
  3. **A net-zero band is not a closed position.** Selling "$1,001 - $15,000"
     against a "$1,001 - $15,000" buy is consistent with both a full exit and a
     90% remaining stake.

So this returns a *disclosed net* with explicit low/high bounds and a
qualitative status, and refuses to state a share count or a market value.

WHAT IT IS GOOD FOR. Direction and attention: which names a filer has been
accumulating versus distributing, how recently, and how concentrated their
disclosed activity is. That is a real and useful reading of the data, and it is
strictly weaker than "portfolio", which is the point.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: Below this the netted band is small enough relative to its own width that
#: calling it "open" or "closed" would be reading noise.
_AMBIGUOUS = 0.25


def _bounds(g: pd.DataFrame) -> tuple[float, float]:
    """(low, high) disclosed notional for a set of transactions.

    `amount_high` is NULL for open-ended top buckets ("Over $50,000,000" and
    the $X+ tiers), so the high bound falls back to the low bound rather than
    to zero: treating an unbounded bucket as worth nothing would understate
    exactly the largest trades.
    """
    lo = float(g["amount_low"].fillna(0).sum())
    hi = float(g["amount_high"].fillna(g["amount_low"]).fillna(0).sum())
    return lo, max(hi, lo)


def filer_positions(trades: pd.DataFrame, filer: str | None = None) -> pd.DataFrame:
    """Per (filer, ticker) disclosed net position.

    Columns: n_buys/n_sells, first/last disclosure, the buy and sell notional
    bands, the netted band, and a `status` that names what the data can support.
    """
    if trades.empty:
        return pd.DataFrame()
    df = trades.dropna(subset=["ticker"]).copy()
    if filer:
        df = df[df["filer_name"].str.lower() == filer.lower()]
    if df.empty:
        return pd.DataFrame()
    df = df[df["side"].isin(["buy", "sell"])]

    rows = []
    for (name, ticker), g in df.groupby(["filer_name", "ticker"], sort=False):
        buys, sells = g[g["side"] == "buy"], g[g["side"] == "sell"]
        blo, bhi = _bounds(buys)
        slo, shi = _bounds(sells)
        # widest honest interval: most-bought-least-sold .. least-bought-most-sold
        net_hi, net_lo = bhi - slo, blo - shi
        span = max(bhi, shi, 1.0)
        # `net_long` is the machine-readable verdict; `status` is prose for the
        # UI. They are separate on purpose, `most_held` used to filter on the
        # English label, so rewording a string silently emptied the panel.
        net_long = False
        if not len(sells):
            status, net_long = "accumulating (no disclosed sale)", True
        elif not len(buys):
            status = "reducing (no disclosed purchase in window)"
        elif net_lo > _AMBIGUOUS * span:
            status, net_long = "net buyer", True
        elif net_hi < -_AMBIGUOUS * span:
            status = "net seller"
        else:
            status = "ambiguous (bands overlap)"
        last = g["filing_datetime"].max()
        rows.append({
            "filer": name, "ticker": ticker,
            "n_buys": int(len(buys)), "n_sells": int(len(sells)),
            "buy_low": blo, "buy_high": bhi,
            "sell_low": slo, "sell_high": shi,
            "net_low": net_lo, "net_high": net_hi,
            "first_disclosed": str(pd.Timestamp(g["filing_datetime"].min()).date()),
            "last_disclosed": str(pd.Timestamp(last).date()),
            "status": status, "net_long": net_long,
            "instrument": (g["instrument"].mode().iloc[0]
                           if "instrument" in g and not g["instrument"].isna().all()
                           else "stock"),
        })
    out = pd.DataFrame(rows)
    return out.sort_values(["filer", "net_high"], ascending=[True, False]).reset_index(drop=True)


def filer_summary(positions: pd.DataFrame, top: int = 25) -> pd.DataFrame:
    """Per filer: breadth, disclosed gross flow, and how recent the activity is."""
    if positions.empty:
        return pd.DataFrame()
    g = positions.groupby("filer").agg(
        tickers=("ticker", "nunique"),
        disclosures=("n_buys", "sum"),
        sells=("n_sells", "sum"),
        buy_notional_low=("buy_low", "sum"),
        buy_notional_high=("buy_high", "sum"),
        sell_notional_low=("sell_low", "sum"),
        sell_notional_high=("sell_high", "sum"),
        last_disclosed=("last_disclosed", "max"),
    ).reset_index()
    g["net_low"] = g["buy_notional_low"] - g["sell_notional_high"]
    g["net_high"] = g["buy_notional_high"] - g["sell_notional_low"]
    g["gross_low"] = g["buy_notional_low"] + g["sell_notional_low"]
    return g.sort_values("gross_low", ascending=False).head(top).reset_index(drop=True)


def most_held(positions: pd.DataFrame, min_filers: int = 3, top: int = 25) -> pd.DataFrame:
    """Tickers the most filers are net-long by disclosure, the crowded names.

    This is the closest honest analogue to a consumer tracker's "most held"
    leaderboard. It counts FILERS WITH A NET-BUY DISCLOSURE, not holders: a
    member who bought before the window and never traded again is invisible
    here, and one who bought once in 2015 still counts.
    """
    if positions.empty:
        return pd.DataFrame()
    net_long = positions[positions["net_long"]]
    if net_long.empty:
        return pd.DataFrame()
    g = net_long.groupby("ticker").agg(
        filers=("filer", "nunique"),
        buys=("n_buys", "sum"),
        net_low=("net_low", "sum"),
        net_high=("net_high", "sum"),
        last_disclosed=("last_disclosed", "max"),
    ).reset_index()
    g = g[g["filers"] >= min_filers]
    return g.sort_values(["filers", "net_high"], ascending=False).head(top).reset_index(drop=True)
