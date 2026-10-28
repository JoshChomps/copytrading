"""Screens for price series that are positive but not credible.

WHY THIS EXISTS. `upsert_prices` rejects `adj_close <= 0`, which was written
after four foreign OTC ADRs produced a -587,200% return and dragged the
whole-sample mean to -310%. That fix removed the negative ticks and left the
rest of the same broken series in place, so the failure came back wearing a
different sign: DAIUF still ran from 5.1e-07 to 45.31 and showed up as the
single best-performing name on the dashboard at +251% mean over 8 buys.

WHAT A BROKEN SERIES ACTUALLY LOOKS LIKE. Four distinct shapes appear in this
data, and only the first is caught by a naive "reject big daily moves" rule:

  * sub-penny adjusted closes: AOZOF bottoms out at 3.7e-24, which is not a
    price by any reading;
  * a series that OSCILLATES between two scales, as if two instruments were
    merged under one symbol, ITC alternates 45 <-> 7,800 on 312 sessions,
    BM alternates 0.07 <-> 7.2 on 116;
  * an unadjusted reverse split, leaving a pre-split segment orders of
    magnitude away from the post-split one, WKHS tops out at 124,000
    against a 2.34 low;
  * a multi-year GAP read as one session, CHK goes 3.05 (2021-03-01) to
    81.46 (2024-10-04), a +2,571% "day".

THE RULE MUST NOT EAT REAL EVENTS. Genuine one-session moves above +100% do
happen: DJT +357% on the DWAC merger, GME +135% in the squeeze. Both survive
every screen below, which is the constraint that shaped them, a flat cap on
daily moves would have discarded real market history to remove provider noise.
So the screens all combine a move test with a RANGE test, or count repeated
offences, because one violent day is a market and five is a data feed.

Nothing is deleted. Quarantine is recorded and applied at `load_price_matrix`,
the single place every analysis reads prices, and the affected trades fall into
the existing "unpriceable" bucket that reports already count and disclose.
"""

from __future__ import annotations

import json
import sqlite3

import numpy as np
import pandas as pd

#: An adjusted close below this is a provider artifact, not a price. The real
#: floor for a US-listed name a member of Congress disclosed buying is orders of
#: magnitude above it; the offenders here bottom out at 1e-5 and 3.7e-24.
MIN_PLAUSIBLE_PRICE = 1e-3

#: One session above +100% is a market event (GME, DJT). Five is a data feed.
MAX_DOUBLING_SESSIONS = 5

#: No equity's ADJUSTED price legitimately spans four orders of magnitude over
#: a decade: adjustment is what removes splits. GME's real range is 124x.
MAX_RANGE_RATIO = 10_000

#: A softer range bar, but only when paired with an impossible single session.
SUSPECT_RANGE_RATIO = 1_000
SUSPECT_DAY_MOVE = 3.0          # +300%


def scan_ticker(prices: pd.Series) -> tuple[str, dict] | None:
    """Screen one ticker's adjusted-close series.

    Returns (reason, detail) when it fails, else None.
    """
    p = pd.Series(prices).dropna().to_numpy(dtype=float)
    if len(p) < 20:
        return None
    lo, hi = float(np.min(p)), float(np.max(p))
    if lo <= 0:
        return "non_positive", {"min": lo}
    ratio = hi / lo
    r = p[1:] / p[:-1] - 1.0
    doublings = int((r > 1.0).sum())
    best = float(np.max(r)) if len(r) else 0.0
    detail = {"min": lo, "max": hi, "range_ratio": ratio,
              "sessions_over_100pct": doublings, "max_session_move": best,
              "n": int(len(p))}

    if lo < MIN_PLAUSIBLE_PRICE:
        return "sub_penny", detail
    if doublings >= MAX_DOUBLING_SESSIONS:
        return "repeated_impossible_sessions", detail
    if ratio > MAX_RANGE_RATIO:
        return "implausible_range", detail
    if ratio > SUSPECT_RANGE_RATIO and best > SUSPECT_DAY_MOVE:
        return "scale_break", detail
    return None


def screen_all(conn, review_limit: int = 25) -> tuple[list[dict], list[dict]]:
    """One pass over `prices` producing BOTH the failures and the review tier.

    The price table is ~7.6M rows; loading it twice (once to find offenders,
    once to find near-misses) and re-running `scan_ticker` over every group a
    second time was the single most expensive thing the integrity panel did.
    Callers that need both lists should use this rather than `scan_all` +
    `review_candidates`.
    """
    df = pd.read_sql_query(
        "SELECT ticker, date, adj_close FROM prices ORDER BY ticker, date", conn)
    if df.empty:
        return [], []
    excluded = quarantined_tickers(conn)
    failures, review = [], []
    for ticker, g in df.groupby("ticker", sort=True):
        series = g["adj_close"]
        hit = scan_ticker(series)
        if hit:
            reason, detail = hit
            failures.append({"ticker": ticker, "reason": reason, "detail": detail})
            continue
        if ticker in excluded:
            continue
        p = series.dropna().to_numpy(dtype=float)
        if len(p) < 20 or p.min() <= 0:
            continue
        ratio = float(p.max() / p.min())
        if ratio <= REVIEW_RANGE_RATIO:
            continue
        r = p[1:] / p[:-1] - 1.0
        review.append({"ticker": ticker, "range_ratio": ratio,
                       "min": float(p.min()), "max": float(p.max()),
                       "max_session_move": float(np.max(r)) if len(r) else 0.0,
                       "n": int(len(p))})
    review.sort(key=lambda d: -d["range_ratio"])
    return failures, review[:review_limit]


def scan_all(conn) -> list[dict]:
    """Screen every ticker in `prices`. Returns one row per failure."""
    return screen_all(conn)[0]


def refresh_quarantine(conn, dry_run: bool = False) -> dict:
    """Re-run the screens and record the failures.

    Tickers manually `released` stay released, the screens are heuristics over
    provider data and a human overrule must survive the next scan, or the
    override is worthless.
    """
    findings = scan_all(conn)
    released = {r[0] for r in conn.execute(
        "SELECT ticker FROM price_quarantine WHERE released=1")}
    if not dry_run:
        for f in findings:
            if f["ticker"] in released:
                continue
            conn.execute(
                "INSERT INTO price_quarantine(ticker, reason, detail) VALUES (?,?,?) "
                "ON CONFLICT(ticker) DO UPDATE SET reason=excluded.reason, "
                "detail=excluded.detail, detected_at=CURRENT_TIMESTAMP",
                (f["ticker"], f["reason"], json.dumps(f["detail"])))
        conn.commit()
    by_reason: dict[str, int] = {}
    for f in findings:
        by_reason[f["reason"]] = by_reason.get(f["reason"], 0) + 1
    affected = 0
    if findings:
        names = [f["ticker"] for f in findings]
        marks = ",".join("?" * len(names))
        affected = conn.execute(
            f"SELECT COUNT(*) FROM trades WHERE ticker IN ({marks}) "
            "AND instrument='stock'", names).fetchone()[0]
    return {
        "screened": conn.execute("SELECT COUNT(DISTINCT ticker) FROM prices").fetchone()[0],
        "quarantined": len(findings),
        "by_reason": by_reason,
        "trades_affected": affected,
        "released_kept": len(released),
        "tickers": [f["ticker"] for f in findings],
        "dry_run": dry_run,
    }


#: A series can span three orders of magnitude legitimately, a correctly
#: adjusted reverse split does exactly that. FCEL runs 6,220 -> 18.50 and CELH
#: runs 0.12 -> 43.81, both real. Series above this bar that still PASS every
#: screen are surfaced for a human to glance at rather than excluded, because
#: the alternative is tuning a heuristic until it happens to agree with me.
REVIEW_RANGE_RATIO = 500


def review_candidates(conn, limit: int = 25) -> list[dict]:
    """Series that pass every screen but sit far enough out to be worth an eye.

    Not a defect list. It exists because the screens are heuristics over
    provider data, and the honest way to run a heuristic is to show what it
    nearly caught: a reader who recognises a name here can overrule it in
    either direction, which is the check a threshold cannot perform for itself.
    """
    return screen_all(conn, review_limit=limit)[1]


def quarantined_tickers(conn) -> set[str]:
    """Tickers to exclude from the price matrix.

    Only a MISSING TABLE is tolerated, databases created before migration 010
    legitimately lack it. Every other failure is re-raised: this guard is the
    single thing keeping 27 known-corrupt series out of every analysis, and a
    blanket `except` would let a locked or damaged database silently readmit
    them with no signal anywhere, including in the integrity panel whose job is
    to catch exactly that.
    """
    try:
        return {r[0] for r in conn.execute(
            "SELECT ticker FROM price_quarantine WHERE released=0")}
    except sqlite3.OperationalError as e:
        if "no such table" in str(e).lower():
            return set()
        raise
