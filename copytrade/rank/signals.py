"""Signal refinements from the insider-trading literature, computed purely
from filings already in the DB (no external data, no look-ahead):

* Cluster buys: multiple DISTINCT filers buying the same ticker within a
  trailing window. Consensus buying is the best-documented amplifier of the
  insider-purchase signal; a cluster count at filing time uses only past
  filings, so it is point-in-time safe.
* Routine filers: adapted from Cohen, Frazzini & Malloy (2012), "Decoding
  Inside Information" (J. Finance): insiders who trade in the SAME calendar
  month year after year are calendar-driven (options vesting, diversification
  programs) and their trades carry little information; the "opportunistic"
  remainder carries the alpha. A filer is flagged routine here if some
  calendar month contains their filings in >= min_years consecutive years.

Both are surfaced as columns/flags, they refine ranking and alerts, they do
not silently change portfolio construction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def cluster_sizes(
    trades: pd.DataFrame,
    window_days: int = 10,
    sides: tuple[str, ...] = ("buy",),
) -> pd.Series:
    """Per-trade count of DISTINCT filers (self included) that filed the same
    side on the same ticker within the trailing `window_days` calendar days
    ending at each trade's filing time. Index = trades.index.

    Trailing-only by construction: a filing can never see later filings, so
    the flag is safe to use in backtests and live alerts alike.
    """
    out = pd.Series(1, index=trades.index, dtype=int)
    sub = trades[trades["side"].isin(sides)]
    win = pd.Timedelta(days=window_days).to_timedelta64()
    for _, g in sub.groupby("ticker", sort=False):
        g = g.sort_values("filing_datetime")
        times = g["filing_datetime"].to_numpy()
        filers = g["filer_id"].to_numpy()
        sizes = np.empty(len(g), dtype=int)
        # O(n) sliding window: distinct-filer count maintained incrementally
        # (the per-row set rebuild was O(n*window))
        counts: dict = {}
        distinct = start = 0
        for i in range(len(g)):
            c = counts.get(filers[i], 0)
            counts[filers[i]] = c + 1
            distinct += c == 0
            while times[i] - times[start] > win:
                f = filers[start]
                counts[f] -= 1
                distinct -= counts[f] == 0
                start += 1
            sizes[i] = distinct
        out.loc[g.index] = sizes
    return out


def cluster_buy_events(
    trades: pd.DataFrame, window_days: int = 10, min_filers: int = 2
) -> pd.DataFrame:
    """The buy trades whose trailing cluster reached `min_filers`, with the
    cluster size attached: feed for reports and alert enrichment."""
    buys = trades[trades["side"] == "buy"]
    if buys.empty:
        return buys.assign(cluster_size=pd.Series(dtype=int))
    sizes = cluster_sizes(buys, window_days)
    flagged = buys.assign(cluster_size=sizes)
    return flagged[flagged["cluster_size"] >= min_filers]


def routine_filers(trades: pd.DataFrame, min_years: int = 3) -> pd.Series:
    """filer_id -> True when some calendar month holds their filings in
    >= min_years CONSECUTIVE years (calendar-driven trading pattern)."""
    result: dict[int, bool] = {}
    sub = trades.dropna(subset=["filing_datetime"])
    for filer_id, g in sub.groupby("filer_id"):
        dt = g["filing_datetime"]
        routine = False
        for _, years in dt.groupby(dt.dt.month):
            ys = sorted(years.dt.year.unique())
            run = 1
            for a, b in zip(ys, ys[1:]):
                run = run + 1 if b == a + 1 else 1
                if run >= min_years:
                    routine = True
                    break
            if routine:
                break
        result[filer_id] = routine
    return pd.Series(result, dtype=bool)


def live_cluster_size(
    conn, ticker: str, filing_datetime: str, window_days: int = 10, side: str = "buy"
) -> int:
    """DB-side trailing cluster count for ONE new filing (poller/alert path)."""
    row = conn.execute(
        "SELECT COUNT(DISTINCT filer_id) FROM trades "
        "WHERE ticker=? AND side=? AND filing_datetime <= ? "
        "AND filing_datetime >= DATETIME(?, ?)",
        (ticker, side, filing_datetime, filing_datetime, f"-{int(window_days)} days"),
    ).fetchone()
    return int(row[0] or 0)
