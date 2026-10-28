"""Disclosure-lag decomposition, and how low it can actually go.

The delay between a filer's trade and the moment YOU can act on it has three
distinct parts. Only the third is a software problem:

  1. REPORTING lag  (transaction_date -> filing_date): the filer's own delay.
     Congress: up to 45 calendar days by law (STOCK Act), and members choose
     where in that window to file. Insiders (Form 4): 2 business days by law
     , structurally tiny. THIS DOMINATES for congress and no pipeline can
     shrink it; the only lever is preferring fast-disclosing filers.
  2. PUBLICATION lag (filing -> public availability): the office posting the
     document. Minutes for EDGAR; hours-to-days for House Clerk / Senate eFD.
     Addressable only by source choice, not by us.
  3. PIPELINE lag    (publication -> we act): our poll cadence + execution
     rule. Fully ours: real-time EDGAR feed (~minutes), tight poll intervals,
     and next-close execution.

This module measures part 1 from the data we already store (both dates are
in `trades`), because it is the biggest and the only one that varies by
filer: so "minimize lag" operationally means **weight toward filers who
disclose fast**, which the decay_lag_speed chart already shows retains more
alpha. Parts 2-3 are configuration, documented in docs/SCRAPING.md.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def reporting_lag_days(trades: pd.DataFrame) -> pd.Series:
    """Whole calendar days from transaction_date to filing DATE (the filer's
    own delay). Compares dates, not timestamps, the 17:00 end-of-day filing
    anchor must not inflate the lag by a fractional day. NaN where either
    date is missing."""
    tx = pd.to_datetime(trades["transaction_date"], errors="coerce").dt.normalize()
    fil = pd.to_datetime(trades["filing_datetime"], errors="coerce").dt.normalize()
    return (fil - tx).dt.days


def lag_by_source(trades: pd.DataFrame) -> pd.DataFrame:
    """Reporting-lag distribution per source: median, p90, and the share of
    filings inside common windows. The '<=2d' column is where insiders live;
    congress clusters near the 45-day cap."""
    if trades.empty:
        return pd.DataFrame()
    df = trades.assign(lag=reporting_lag_days(trades)).dropna(subset=["lag"])
    df = df[df["lag"] >= 0]                       # drop clock-skew/bad rows
    rows = []
    for source, g in df.groupby("source"):
        lag = g["lag"]
        rows.append({
            "source": source, "n": len(lag),
            "median_days": round(float(lag.median()), 1),
            "p90_days": round(float(lag.quantile(0.90)), 1),
            "pct_le_2d": round(float((lag <= 2).mean()), 3),
            "pct_le_14d": round(float((lag <= 14).mean()), 3),
            "pct_le_45d": round(float((lag <= 45).mean()), 3),
        })
    return pd.DataFrame(rows).sort_values("median_days")


def fastest_filers(trades: pd.DataFrame, min_trades: int = 10, top: int = 15) -> pd.DataFrame:
    """Filers ranked by how fast they disclose, the ones whose signals are
    freshest by the time you can act. Preferring these is the only real lever
    on congressional reporting lag."""
    if trades.empty:
        return pd.DataFrame()
    df = trades.assign(lag=reporting_lag_days(trades)).dropna(subset=["lag"])
    df = df[df["lag"] >= 0]
    g = (df.groupby(["filer_id", "filer_name"])
         .agg(n=("lag", "size"), median_lag=("lag", "median"))
         .reset_index())
    g = g[g["n"] >= min_trades].sort_values("median_lag")
    g["median_lag"] = g["median_lag"].round(1)
    return g.head(top)
