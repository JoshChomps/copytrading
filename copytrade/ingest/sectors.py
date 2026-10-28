"""Ticker -> sector/industry mapping via yfinance, cached in the DB.

This is reference data, not a signal: it lets the analysis ask "which AREAS
do these filers pursue, and where do their picks actually work?" yfinance is
keyless but per-ticker and CI-throttled, so we (a) map the most-traded tickers
first, they carry the bulk of the volume, and (b) cache in sector_map so the
mapping fills in progressively across runs and is never re-fetched. A ticker
yfinance can't classify is simply left unmapped (-> 'Unknown' downstream),
never guessed.

FAILURES ARE CACHED TOO, and that is the point of `sector_attempts`. An
earlier revision stored only successes, so a ticker yfinance could not
classify was re-requested on every single run. The observed behaviour was a
first pass mapping 521 of 800 requested, and then every later pass reporting
`mapped: 0`: not because the map was complete (2,568 tickers were still
unmapped) but because re-asking for the previous run's 279 failures tripped
yfinance's rate limiter, after which nothing succeeded. A status of "0 new"
that actually means "the provider is refusing" is the kind of quiet
degradation this project treats as a bug.

So: every attempt is recorded, recent failures are skipped, and a run that
looks throttled says THROTTLED and stops rather than burning the rest of the
budget against a closed door.
"""

from __future__ import annotations

import time

#: A ticker that failed lookup is not retried for this long. Long enough that
#: repeated runs make progress on genuinely new names instead of re-grinding
#: known failures; short enough that a throttled batch is retried the next day.
RETRY_AFTER_DAYS = 7

#: Give up on a ticker after this many failed attempts. Delisted and foreign
#: OTC lines are permanently unclassifiable and there is no point paying a
#: request for them forever.
MAX_ATTEMPTS = 3

#: If this many consecutive lookups fail from the very start of a run, treat
#: the provider as throttled and stop. Mirrors the Tiingo 429 latch: a wall of
#: failures is a provider state, not 60 individually-bad tickers.
THROTTLE_STREAK = 25


class Throttled(RuntimeError):
    """Raised internally when the provider appears to be rate-limiting."""


def fetch_sectors(tickers: list[str], sleep: float = 0.0
                  ) -> tuple[dict[str, tuple[str, str]], dict[str, str], bool]:
    """Look up sectors for `tickers`.

    Returns (mapped, failures, throttled) where `mapped` is
    {ticker: (sector, industry)}, `failures` is {ticker: reason}, and
    `throttled` says the run stopped early because the provider stopped
    answering. Failures are reported, never faked.
    """
    import yfinance as yf

    out: dict[str, tuple[str, str]] = {}
    failures: dict[str, str] = {}
    streak = 0
    for t in tickers:
        try:
            info = yf.Ticker(t).info or {}
            sector = info.get("sector")
            if sector:
                out[t] = (sector, info.get("industry") or "")
                streak = 0
            else:
                failures[t] = "no_sector"
                streak += 1
        except Exception:  # noqa: BLE001, one bad ticker must not stop the sweep
            failures[t] = "error"
            streak += 1
        # A long unbroken failure streak is the signature of rate limiting.
        # This deliberately does NOT require an empty `out`: a run that maps a
        # handful of tickers and THEN hits the limiter is the common case, and
        # gating on "nothing succeeded yet" meant such a run ground through the
        # remaining hundreds, recorded them all as failed attempts, and so put
        # perfectly classifiable tickers into the 7-day cooling-off window.
        if streak >= THROTTLE_STREAK:
            # Only the trailing streak is attributable to the limiter; failures
            # from before it are genuine per-ticker misses and keep their reason.
            for bad in list(failures)[-streak:]:
                failures[bad] = "throttled"
            return out, failures, True
        if sleep:
            time.sleep(sleep)
    return out, failures, False


def refresh_sectors(conn, limit: int = 800, min_trades: int = 1,
                    retry_after_days: int = RETRY_AFTER_DAYS) -> dict:
    """Map the most-traded still-unmapped stock tickers (up to `limit` this
    run). Idempotent + resumable: already-mapped tickers are skipped, and so
    are tickers whose lookup failed recently or has failed `MAX_ATTEMPTS`
    times, so repeated runs make progress on new names instead of re-grinding
    known failures."""
    counts = conn.execute(
        "SELECT ticker, COUNT(*) c FROM trades "
        "WHERE instrument='stock' AND ticker IS NOT NULL "
        "GROUP BY ticker HAVING c >= ?",
        (min_trades,),
    ).fetchall()
    have = {r[0] for r in conn.execute("SELECT ticker FROM sector_map")}
    cooling = {
        r[0] for r in conn.execute(
            "SELECT ticker FROM sector_attempts "
            "WHERE attempts >= ? OR julianday('now') - julianday(last_attempt) < ?",
            (MAX_ATTEMPTS, retry_after_days),
        )
    }
    skip = have | cooling
    ranked = [r[0] for r in sorted(counts, key=lambda r: -r[1])]
    want = [t for t in ranked if t not in skip][:limit]

    got, failures, throttled = fetch_sectors(want)
    for t, (sector, industry) in got.items():
        conn.execute(
            "INSERT OR REPLACE INTO sector_map(ticker, sector, industry) VALUES (?,?,?)",
            (t, sector, industry),
        )
        conn.execute("DELETE FROM sector_attempts WHERE ticker=?", (t,))
    for t, reason in failures.items():
        # `reason` already marks the throttled tail; stamping the whole batch
        # as throttled would hide which failures were genuine per-ticker misses.
        conn.execute(
            "INSERT INTO sector_attempts(ticker, attempts, last_attempt, last_status) "
            "VALUES (?, 1, CURRENT_TIMESTAMP, ?) "
            "ON CONFLICT(ticker) DO UPDATE SET "
            "  attempts = sector_attempts.attempts + 1, "
            "  last_attempt = CURRENT_TIMESTAMP, "
            "  last_status = excluded.last_status",
            (t, reason),
        )
    conn.commit()

    total_unmapped = len([t for t in ranked if t not in have]) - len(got)
    return {
        "requested": len(want),
        "mapped": len(got),
        "failed": len(failures),
        "throttled": throttled,
        "total_mapped": len(have) + len(got),
        "still_unmapped": max(total_unmapped, 0),
        "cooling_off": len(cooling),
        "note": (
            "provider stopped answering: this is a rate limit, NOT a signal "
            "that the remaining tickers are unclassifiable; retry later"
            if throttled else None),
    }
