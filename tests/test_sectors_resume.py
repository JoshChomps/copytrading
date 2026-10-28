"""Sector lookups must remember their FAILURES, not just their successes.

The bug this pins: `sector_map` stored only tickers yfinance could classify, so
a ticker it failed on stayed "unmapped" and was re-requested every run. Observed
behaviour was pass 1 mapping 521 of 800 and every later pass reporting
`mapped: 0`: not because the map was complete (2,568 tickers were still
unmapped) but because re-asking for the previous run's failures tripped the rate
limiter. A status of "0 new" that actually means "the provider is refusing" is
the kind of quiet degradation that makes a coverage gap look like a finished job.
"""

from __future__ import annotations

import pytest

from copytrade.ingest import sectors as S
from copytrade.store import db as store


@pytest.fixture()
def conn(tmp_path):
    c = store.connect(tmp_path / "t.db")
    store.run_migrations(c)
    fid = store.upsert_filer(c, "M", "congress")
    for i, t in enumerate(["AAA", "BBB", "CCC", "DDD"]):
        store.insert_trade(c, filer_id=fid, source="s", source_uid=f"u{i}",
                           ticker=t, side="buy", instrument="stock",
                           transaction_date="2020-01-02",
                           filing_datetime="2020-01-10 17:00:00")
    return c


def _fake(mapped: dict, failures: dict, throttled: bool = False):
    return lambda tickers, sleep=0.0: (
        {t: v for t, v in mapped.items() if t in tickers},
        {t: v for t, v in failures.items() if t in tickers},
        throttled,
    )


def test_failures_are_recorded_and_not_immediately_retried(conn, monkeypatch):
    monkeypatch.setattr(S, "fetch_sectors",
                        _fake({"AAA": ("Technology", "Software")},
                              {"BBB": "no_sector", "CCC": "error", "DDD": "no_sector"}))
    first = S.refresh_sectors(conn, limit=10)
    assert first["mapped"] == 1 and first["failed"] == 3

    # second run must NOT re-request the three failures
    seen = {}

    def spy(tickers, sleep=0.0):
        seen["asked"] = list(tickers)
        return {}, {}, False

    monkeypatch.setattr(S, "fetch_sectors", spy)
    second = S.refresh_sectors(conn, limit=10)
    assert seen["asked"] == []
    assert second["cooling_off"] == 3


def test_a_cooled_off_ticker_is_retried_once_the_window_passes(conn, monkeypatch):
    monkeypatch.setattr(S, "fetch_sectors", _fake({}, {"AAA": "error"}))
    S.refresh_sectors(conn, limit=10)
    # retry_after_days=0 simulates the window having elapsed
    monkeypatch.setattr(S, "fetch_sectors",
                        _fake({"AAA": ("Energy", "Oil")}, {}))
    out = S.refresh_sectors(conn, limit=10, retry_after_days=0)
    assert out["mapped"] == 1
    assert conn.execute("SELECT sector FROM sector_map WHERE ticker='AAA'"
                        ).fetchone()[0] == "Energy"


def test_success_clears_the_failure_record(conn, monkeypatch):
    monkeypatch.setattr(S, "fetch_sectors", _fake({}, {"AAA": "error"}))
    S.refresh_sectors(conn, limit=10)
    monkeypatch.setattr(S, "fetch_sectors", _fake({"AAA": ("Energy", "Oil")}, {}))
    S.refresh_sectors(conn, limit=10, retry_after_days=0)
    assert conn.execute("SELECT COUNT(*) FROM sector_attempts WHERE ticker='AAA'"
                        ).fetchone()[0] == 0


def test_a_ticker_is_abandoned_after_max_attempts(conn, monkeypatch):
    monkeypatch.setattr(S, "fetch_sectors", _fake({}, {"AAA": "error"}))
    for _ in range(S.MAX_ATTEMPTS):
        S.refresh_sectors(conn, limit=10, retry_after_days=0)
    seen = {}

    def spy(tickers, sleep=0.0):
        seen["asked"] = list(tickers)
        return {}, {}, False

    monkeypatch.setattr(S, "fetch_sectors", spy)
    S.refresh_sectors(conn, limit=10, retry_after_days=0)
    assert "AAA" not in seen["asked"]


def test_throttling_is_reported_as_throttling_not_as_completion(conn, monkeypatch):
    """The distinction that matters: 'the provider refused' must never be
    presented the same way as 'there is nothing left to map'.

    `fetch_sectors` labels the throttled tail itself, so the stub returns the
    reason it really would, `refresh_sectors` no longer overwrites the whole
    batch, which used to hide which failures were genuine per-ticker misses.
    """
    monkeypatch.setattr(S, "fetch_sectors",
                        _fake({}, {"AAA": "throttled", "BBB": "throttled"},
                              throttled=True))
    out = S.refresh_sectors(conn, limit=10)
    assert out["throttled"] is True
    assert out["mapped"] == 0
    assert out["still_unmapped"] > 0        # NOT "complete"
    assert "rate limit" in out["note"]
    assert conn.execute(
        "SELECT COUNT(*) FROM sector_attempts WHERE last_status='throttled'"
    ).fetchone()[0] == 2


def test_still_unmapped_is_reported_so_zero_new_is_never_ambiguous(conn, monkeypatch):
    monkeypatch.setattr(S, "fetch_sectors",
                        _fake({"AAA": ("Technology", "Software")}, {}))
    out = S.refresh_sectors(conn, limit=1)
    assert out["mapped"] == 1
    assert out["still_unmapped"] == 3


def test_throttle_streak_stops_a_hopeless_run_early(monkeypatch):
    """Detection lives in fetch_sectors: a long unbroken failure streak with
    nothing mapped is a provider state, not N individually-bad tickers."""
    class FakeTicker:
        def __init__(self, t): pass
        @property
        def info(self): raise RuntimeError("429 Too Many Requests")

    monkeypatch.setitem(__import__("sys").modules, "yfinance",
                        type("m", (), {"Ticker": FakeTicker}))
    mapped, failures, throttled = S.fetch_sectors([f"T{i}" for i in range(200)])
    assert throttled is True
    assert not mapped
    # stopped early rather than burning all 200 requests against a closed door
    assert len(failures) == S.THROTTLE_STREAK


def test_throttle_fires_even_after_some_tickers_mapped(monkeypatch):
    """The guard used to require that NOTHING had succeeded yet.

    A run that maps a handful and THEN hits the limiter is the common case, and
    gating on an empty result meant such a run ground through the remaining
    hundreds, recorded them all as failed attempts, and so pushed perfectly
    classifiable tickers into the 7-day cooling-off window.
    """
    calls = {"n": 0}

    class FakeTicker:
        def __init__(self, t):
            self.t = t

        @property
        def info(self):
            calls["n"] += 1
            if self.t in ("T0", "T1", "T2"):
                return {"sector": "Technology", "industry": "Software"}
            raise RuntimeError("429 Too Many Requests")

    monkeypatch.setitem(__import__("sys").modules, "yfinance",
                        type("m", (), {"Ticker": FakeTicker}))
    mapped, failures, throttled = S.fetch_sectors([f"T{i}" for i in range(300)])
    assert throttled is True, "three early successes must not disable detection"
    assert len(mapped) == 3
    # stopped shortly after the streak, not after all 300
    assert calls["n"] == 3 + S.THROTTLE_STREAK


def test_only_the_throttled_tail_is_labelled_throttled(monkeypatch):
    """Failures from before the limiter kicked in are genuine per-ticker misses
    and must keep their own reason, or the cache cannot tell them apart."""
    class FakeTicker:
        def __init__(self, t):
            self.t = t

        @property
        def info(self):
            if self.t == "EARLY":
                return {}          # genuine 'no_sector'
            if self.t == "OK":
                return {"sector": "Energy", "industry": "Oil"}
            raise RuntimeError("429")

    monkeypatch.setitem(__import__("sys").modules, "yfinance",
                        type("m", (), {"Ticker": FakeTicker}))
    _, failures, throttled = S.fetch_sectors(
        ["EARLY", "OK"] + [f"X{i}" for i in range(S.THROTTLE_STREAK)])
    assert throttled is True
    assert failures["EARLY"] == "no_sector", "pre-throttle miss keeps its reason"
    assert failures["X0"] == "throttled"
