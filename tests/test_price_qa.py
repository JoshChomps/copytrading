"""Price series that are positive but not credible must not reach an analysis.

The `adj_close <= 0` guard in `upsert_prices` was written after four OTC ADRs
produced a -587,200% return. It removed the negative ticks and left the rest of
the same broken series in place, so the failure came back with a positive sign:
DAIUF still ran 5.1e-07 to 45.31 and surfaced as the best-performing name on the
dashboard at +251% mean over 8 buys.

The hard constraint on any screen written here: it must not eat real market
history. DJT rose 357% in one session on the DWAC merger and GME 135% in the
squeeze. Both are real, both must survive.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from copytrade.store import db as store
from copytrade.store import price_qa


def _series(values):
    return pd.Series(values, dtype=float)


def _ramp(n, start, end):
    return list(np.geomspace(start, end, n))


# ------------------------------------------------------------ the offenders --

def test_sub_penny_series_is_caught():
    """AOZOF bottoms out at 3.7e-24, which is not a price by any reading."""
    s = _series(_ramp(400, 3.65e-24, 14.02))
    hit = price_qa.scan_ticker(s)
    assert hit is not None


def test_oscillating_between_two_scales_is_caught():
    """ITC alternates 45 <-> 7,800 across 312 sessions, as if two instruments
    were merged under one symbol."""
    s = _series([45.0 if i % 2 else 7800.0 for i in range(400)])
    reason, _ = price_qa.scan_ticker(s)
    assert reason == "repeated_impossible_sessions"


def test_unadjusted_reverse_split_is_caught():
    """A pre-split segment orders of magnitude from the post-split one."""
    s = _series([2.34] * 200 + [124_000.0] * 200)
    assert price_qa.scan_ticker(s) is not None


def test_multi_year_gap_read_as_one_session_is_caught():
    """CHK goes 3.05 (2021-03-01) to 81.46 (2024-10-04), a +2,571% 'day'."""
    s = _series(_ramp(200, 5811.0, 3.05) + _ramp(200, 81.46, 20.0))
    assert price_qa.scan_ticker(s) is not None


# --------------------------------------------------- what must NOT be caught --

def test_a_real_squeeze_survives():
    """GME: +135% in one session, 124x range. Real market history."""
    s = _series(_ramp(1500, 0.7, 37.0) + [86.88] + _ramp(1600, 40.0, 25.0))
    assert price_qa.scan_ticker(s) is None


def test_a_real_merger_pop_survives():
    """DJT: +357% on the DWAC merger, 13.8x range."""
    s = _series(_ramp(600, 9.96, 9.96) + [45.5, 94.2] + _ramp(600, 90.0, 20.0))
    assert price_qa.scan_ticker(s) is None


def test_a_correctly_adjusted_reverse_split_survives():
    """FCEL legitimately runs 6,220 -> 18.50 after compounding reverse splits:
    a smooth 336x decay with no impossible session. Range alone must not
    condemn a series."""
    s = _series(_ramp(3000, 6220.0, 18.5))
    assert price_qa.scan_ticker(s) is None


def test_a_genuine_hundredfold_grower_survives():
    """CELH ran 0.12 -> 43.81 over the sample. Real."""
    s = _series(_ramp(3000, 0.1167, 43.81))
    assert price_qa.scan_ticker(s) is None


def test_an_ordinary_benchmark_survives():
    rng = np.random.default_rng(0)
    s = _series(100 * np.cumprod(1 + rng.normal(0.0004, 0.011, 3000)))
    assert price_qa.scan_ticker(s) is None


def test_a_short_series_is_not_judged():
    """Too few observations to screen; silence beats a guess."""
    assert price_qa.scan_ticker(_series([1.0, 500.0, 0.001])) is None


# --------------------------------------------------------------- the wiring --

@pytest.fixture()
def conn(tmp_path):
    from pathlib import Path

    c = store.connect(tmp_path / "t.db")
    store.run_migrations(c, Path(__file__).resolve().parents[1] / "migrations")
    dates = pd.bdate_range("2020-01-01", periods=400)
    good = pd.DataFrame({"ticker": "SPY", "date": dates,
                         "adj_close": np.linspace(300, 400, 400)})
    bad = pd.DataFrame({"ticker": "DAIUF", "date": dates,
                        "adj_close": np.geomspace(5.13e-07, 45.31, 400)})
    store.upsert_prices(c, pd.concat([good, bad]))
    return c


def test_quarantine_excludes_the_ticker_from_the_price_matrix(conn):
    assert "DAIUF" in store.load_price_matrix(conn).columns
    price_qa.refresh_quarantine(conn)
    cols = store.load_price_matrix(conn).columns
    assert "DAIUF" not in cols, "a quarantined series must not reach an analysis"
    assert "SPY" in cols, "the screen must not take good data with it"


def test_rows_are_not_deleted(conn):
    """`prices` is re-populated from providers, so a delete would be undone by
    the next refresh: and would destroy the evidence the provider is wrong."""
    price_qa.refresh_quarantine(conn)
    assert conn.execute("SELECT COUNT(*) FROM prices WHERE ticker='DAIUF'"
                        ).fetchone()[0] == 400
    assert "DAIUF" in store.load_price_matrix(conn, include_quarantined=True).columns


def test_dry_run_records_nothing(conn):
    out = price_qa.refresh_quarantine(conn, dry_run=True)
    assert out["quarantined"] == 1
    assert conn.execute("SELECT COUNT(*) FROM price_quarantine").fetchone()[0] == 0
    assert "DAIUF" in store.load_price_matrix(conn).columns


def test_a_manual_release_survives_the_next_scan(conn):
    """The screens are heuristics over provider data. A human overrule that the
    next scan silently reverted would be worthless."""
    price_qa.refresh_quarantine(conn)
    conn.execute("UPDATE price_quarantine SET released=1 WHERE ticker='DAIUF'")
    conn.commit()
    price_qa.refresh_quarantine(conn)
    assert conn.execute(
        "SELECT released FROM price_quarantine WHERE ticker='DAIUF'"
    ).fetchone()[0] == 1
    assert "DAIUF" in store.load_price_matrix(conn).columns


def test_refresh_reports_the_affected_trade_count(conn):
    fid = store.upsert_filer(conn, "M", "congress")
    store.insert_trade(conn, filer_id=fid, source="s", source_uid="u1",
                       ticker="DAIUF", side="buy", instrument="stock",
                       transaction_date="2020-02-03",
                       filing_datetime="2020-02-20 17:00:00")
    conn.commit()
    out = price_qa.refresh_quarantine(conn)
    assert out["trades_affected"] == 1
    assert out["by_reason"].get("sub_penny") == 1


def test_quarantined_tickers_is_empty_without_the_table(tmp_path):
    """Older databases predate migration 010; the lookup must degrade quietly
    rather than break every price load."""
    c = store.connect(tmp_path / "old.db")
    c.execute("CREATE TABLE prices (ticker TEXT, date TEXT, adj_close REAL)")
    assert price_qa.quarantined_tickers(c) == set()


def test_review_tier_lists_near_misses_without_excluding_them(conn):
    price_qa.refresh_quarantine(conn)
    dates = pd.bdate_range("2020-01-01", periods=400)
    # FCEL's real span in this dataset: 16,980 -> 3.73, a 4,552x range produced
    # by compounding reverse splits that WERE adjusted correctly.
    store.upsert_prices(conn, pd.DataFrame({
        "ticker": "FCEL", "date": dates,
        "adj_close": np.geomspace(16_980.0, 3.73, 400)}))
    review = price_qa.review_candidates(conn)
    names = [r["ticker"] for r in review]
    assert "FCEL" in names, "a 4,552x range should be surfaced for a human"
    assert "FCEL" in store.load_price_matrix(conn).columns, "but not excluded"
    assert "DAIUF" not in names, "already quarantined, not a near miss"
