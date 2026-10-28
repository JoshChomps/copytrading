import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from copytrade.web import analyses as an

ROOT = Path(__file__).resolve().parents[1]


def test_clean_strips_nan_and_inf():
    """NaN/Inf are not valid JSON. json.dumps emits bare NaN/Infinity tokens
    that JSON.parse rejects, and a lenient parser turns them into 0, which is
    how a missing statistic becomes a fabricated one on screen."""
    out = an._clean({
        "a": float("nan"), "b": float("inf"), "c": -float("inf"),
        "d": np.float64("nan"), "e": 1.5, "f": np.int64(3),
        "g": [float("nan"), 2.0], "h": np.True_,
    })
    assert out["a"] is None and out["b"] is None and out["c"] is None
    assert out["d"] is None
    assert out["e"] == 1.5 and out["f"] == 3
    assert out["g"] == [None, 2.0]
    assert out["h"] is True
    # the whole payload must round-trip through strict JSON
    json.loads(json.dumps(out))


def test_clean_serialises_timestamps():
    out = an._clean({"t": pd.Timestamp("2026-07-25")})
    assert isinstance(out["t"], str) and out["t"].startswith("2026-07-25")
    json.loads(json.dumps(out))


def test_save_and_load_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(an, "cache_dir", lambda: tmp_path)
    p = an.save("demo", {"verdict": "ok", "rows": [{"x": float("nan")}]})
    assert p.exists()
    got = an.load("demo")
    assert got["verdict"] == "ok"
    assert got["rows"][0]["x"] is None
    assert "built_at" in got, "the UI needs a timestamp to show staleness"


def test_load_missing_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr(an, "cache_dir", lambda: tmp_path)
    assert an.load("never_built") is None


def test_load_corrupt_returns_none_rather_than_raising(tmp_path, monkeypatch):
    """A half-written cache file must degrade to 'not built', not 500 the API."""
    monkeypatch.setattr(an, "cache_dir", lambda: tmp_path)
    (tmp_path / "broken.json").write_text("{not json", encoding="utf-8")
    assert an.load("broken") is None


def test_every_builder_is_reachable_by_name():
    for name, fn in an.BUILDERS.items():
        assert callable(fn), name


def test_the_ui_fetches_exactly_the_panels_that_can_be_built():
    """The front end hard-codes the panel list it requests. If that list and
    BUILDERS drift apart, the failure is silent in both directions: a panel the
    UI never asks for is built and never seen, and a panel it asks for but
    cannot be built renders as a permanently empty card.

    This is not hypothetical, `scorecard_cad` was reachable from the UI and
    the API but was NOT in BUILDERS, so it could not be rebuilt by name and was
    skipped whenever the build died before the special case that produced it.
    """
    import re

    html = (ROOT / "copytrade" / "web" / "static" / "index.html").read_text(encoding="utf-8")
    m = re.search(r"const NAMES = \[(.*?)\];", html, re.S)
    assert m, "could not find the UI's NAMES list"
    ui = set(re.findall(r'"([a-z_]+)"', m.group(1)))

    assert ui == set(an.BUILDERS), (
        f"UI-only: {sorted(ui - set(an.BUILDERS))}  "
        f"builder-only: {sorted(set(an.BUILDERS) - ui)}"
    )


def test_scorecard_cad_is_a_first_class_panel():
    """It is the same builder with a different argument, which is exactly why
    it kept being special-cased, and special cases are what go stale."""
    assert "scorecard_cad" in an.BUILDERS
    assert an.BUILDERS["scorecard_cad"].func is an.build_scorecard
    assert an.BUILDERS["scorecard_cad"].keywords == {"cad": True}


# ---- partial builds must not pass for complete ones ----------------------
#
# Panels are written one at a time, so a run that dies partway leaves a MIXTURE
# of vintages on disk. Each panel still looks fine alone; what breaks is that
# they no longer describe the same data, and the doc's cross-panel statements
# stop holding. A run that died on the last builder once left scorecard_cad
# eleven days behind the rest, and nothing in the exit code said so.

def test_was_built_since_accepts_a_panel_written_by_this_run(tmp_path, monkeypatch):
    from datetime import datetime, timezone

    monkeypatch.setattr(an, "cache_dir", lambda: tmp_path)
    started = datetime.now(timezone.utc)
    an.save("health", {"ok": True})
    assert an.was_built_since("health", started)


def test_was_built_since_rejects_a_panel_left_over_from_an_earlier_run(tmp_path, monkeypatch):
    from datetime import datetime, timedelta, timezone

    monkeypatch.setattr(an, "cache_dir", lambda: tmp_path)
    an.save("health", {"ok": True})          # written "before" the run starts
    started = datetime.now(timezone.utc) + timedelta(minutes=5)
    assert not an.was_built_since("health", started)


def test_was_built_since_rejects_a_panel_that_was_never_written(tmp_path, monkeypatch):
    from datetime import datetime, timezone

    monkeypatch.setattr(an, "cache_dir", lambda: tmp_path)
    assert not an.was_built_since("never_built", datetime.now(timezone.utc))


def test_was_built_since_reads_built_at_not_the_file_mtime(tmp_path, monkeypatch):
    """A panel that is copied, synced or restored gets a fresh mtime over stale
    contents. The stamp inside the payload is the one that means anything."""
    from datetime import datetime, timezone

    monkeypatch.setattr(an, "cache_dir", lambda: tmp_path)
    p = tmp_path / "health.json"
    p.write_text(json.dumps({"built_at": "2020-01-01T00:00:00+00:00"}), encoding="utf-8")
    assert p.stat().st_mtime > 0          # freshly touched on disk...
    started = datetime(2026, 1, 1, tzinfo=timezone.utc)
    assert not an.was_built_since("health", started)   # ...but old inside


def test_dashboard_build_exits_nonzero_when_a_panel_does_not_rebuild(tmp_path, monkeypatch):
    """The exit code is the whole point of the check: the run that left
    scorecard_cad eleven days stale still reported success to the shell, so
    nothing downstream had any reason to look."""
    from typer.testing import CliRunner

    from copytrade import cli

    monkeypatch.setattr(an, "cache_dir", lambda: tmp_path)

    def boom(**kw):
        raise RuntimeError("builder exploded")

    monkeypatch.setattr(an, "BUILDERS", {"health": lambda **kw: {"ok": True},
                                         "timescale": boom})

    result = CliRunner().invoke(cli.app, ["dashboard-build", "--only", "health,timescale"])
    assert result.exit_code == 1, result.output
    assert "timescale" in result.output
    assert "did NOT rebuild" in result.output


def test_dashboard_build_exits_zero_when_everything_rebuilds(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from copytrade import cli

    monkeypatch.setattr(an, "cache_dir", lambda: tmp_path)
    monkeypatch.setattr(an, "BUILDERS", {"health": lambda **kw: {"ok": True}})

    result = CliRunner().invoke(cli.app, ["dashboard-build", "--only", "health"])
    assert result.exit_code == 0, result.output
    assert "all 1 panels rebuilt" in result.output


def test_builders_degrade_on_an_empty_database(tmp_path, monkeypatch):
    """With no trades, every panel must return an {'error': ...} payload rather
    than raising: one empty panel must not take the dashboard down."""
    from copytrade.config import Settings
    from copytrade.store import db as store

    db = tmp_path / "empty.db"
    conn = store.connect(db)
    from pathlib import Path
    store.run_migrations(conn, Path(__file__).resolve().parents[1] / "migrations")
    conn.close()

    monkeypatch.setattr(an, "load_settings",
                        lambda *a, **k: Settings(db_path=str(db),
                                                 reports_dir=str(tmp_path)))
    an.reset_shared()
    # `health` and `quality` describe the DATABASE rather than a backtest, so
    # both are meaningful (and must not error) when it is empty.
    for name, fn in an.BUILDERS.items():
        if name in ("health", "quality"):
            continue
        payload = fn()
        assert isinstance(payload, dict), name
        assert "error" in payload, f"{name} should report an error, got {payload.keys()}"


def test_quality_builder_passes_its_checks_on_a_clean_empty_database(tmp_path, monkeypatch):
    """An empty DB has nothing wrong with it. The integrity panel must say so
    rather than reporting emptiness as corruption, a false alarm here would
    train the reader to ignore the one panel that must never be ignored."""
    from pathlib import Path

    from copytrade.config import Settings
    from copytrade.store import db as store

    db = tmp_path / "empty3.db"
    conn = store.connect(db)
    store.run_migrations(conn, Path(__file__).resolve().parents[1] / "migrations")
    conn.close()
    monkeypatch.setattr(an, "load_settings",
                        lambda *a, **k: Settings(db_path=str(db),
                                                 reports_dir=str(tmp_path)))
    an.reset_shared()
    q = an.build_quality()
    failing = [c["check"] for c in q["checks"] if not c["ok"]]
    # benchmarks are legitimately missing from an empty DB; nothing else may fail
    assert failing == ["benchmarks priced"], failing
    json.loads(json.dumps(an._clean(q)))


def test_quality_catches_a_planted_corruption(tmp_path, monkeypatch):
    """The checks must actually fire. A panel of checks that can only pass is
    decoration."""
    from pathlib import Path

    from copytrade.config import Settings
    from copytrade.store import db as store

    db = tmp_path / "dirty.db"
    conn = store.connect(db)
    store.run_migrations(conn, Path(__file__).resolve().parents[1] / "migrations")
    fid = store.upsert_filer(conn, "X", "congress")
    # write straight past insert_trade's guard, as the old code path did
    conn.execute(
        "INSERT INTO trades(filer_id, source, source_uid, ticker, side, instrument,"
        " transaction_date, filing_datetime) VALUES (?,?,?,?,?,?,?,?)",
        (fid, "house_clerk", "x1", "AAPL", "buy", "stock",
         "3031-04-30", "2021-05-03 17:00:00"))
    conn.commit()
    conn.close()
    monkeypatch.setattr(an, "load_settings",
                        lambda *a, **k: Settings(db_path=str(db),
                                                 reports_dir=str(tmp_path)))
    an.reset_shared()
    q = an.build_quality()
    bad = {c["check"]: c for c in q["checks"] if not c["ok"]}
    assert "transaction date after its own filing" in bad
    assert bad["transaction date after its own filing"]["value"] == 1
    assert q["n_failed"] >= 1
    assert "FAILED" in q["verdict"]


def test_shared_memo_is_keyed_on_the_database(tmp_path, monkeypatch):
    """A cached price matrix answering for the wrong database is the quietest
    possible way to fabricate results, so the memo key carries the db path."""
    from pathlib import Path

    from copytrade.config import Settings
    from copytrade.store import db as store

    paths = []
    for name in ("a.db", "b.db"):
        p = tmp_path / name
        c = store.connect(p)
        store.run_migrations(c, Path(__file__).resolve().parents[1] / "migrations")
        c.close()
        paths.append(p)

    an.reset_shared()
    keys = []
    for p in paths:
        monkeypatch.setattr(an, "load_settings",
                            lambda *a, _p=p, **k: Settings(db_path=str(_p),
                                                           reports_dir=str(tmp_path)))
        an._ctx()
        keys.append(an._db_key())
    assert keys[0] != keys[1]
    ctx_keys = [k for k in an._SHARED if isinstance(k, tuple) and k[0] == "ctx"]
    assert len(ctx_keys) == 2, "each database must get its own cached context"
    an.reset_shared()


def test_every_shared_memo_key_carries_the_database(tmp_path, monkeypatch):
    """_ctx and _trades were keyed on the db path but _copy_all and
    _event_study were not: and those two hold the walk-forward basket and the
    per-trade event study, the objects almost every headline number is derived
    from. A second database in the same process would have been served the
    first one's results with no error."""
    from pathlib import Path

    from copytrade.config import Settings
    from copytrade.store import db as store

    db = tmp_path / "keyed.db"
    conn = store.connect(db)
    store.run_migrations(conn, Path(__file__).resolve().parents[1] / "migrations")
    conn.close()
    monkeypatch.setattr(an, "load_settings",
                        lambda *a, **k: Settings(db_path=str(db),
                                                 reports_dir=str(tmp_path)))
    an.reset_shared()
    an._copy_all()
    an._event_study([63])
    for key in an._SHARED:
        assert isinstance(key, tuple), f"{key!r} is not scoped to a database"
        assert str(db) in key, f"memo key {key[0]!r} omits the database path"
    an.reset_shared()


def test_health_builder_works_on_an_empty_database(tmp_path, monkeypatch):
    from pathlib import Path

    from copytrade.config import Settings
    from copytrade.store import db as store

    db = tmp_path / "empty2.db"
    conn = store.connect(db)
    store.run_migrations(conn, Path(__file__).resolve().parents[1] / "migrations")
    conn.close()
    monkeypatch.setattr(an, "load_settings",
                        lambda *a, **k: Settings(db_path=str(db),
                                                 reports_dir=str(tmp_path)))
    h = an.build_health()
    assert h["trades"] == 0
    assert h["verdict"] == "degraded", "no benchmarks priced => degraded"
    assert any("benchmark" in n for n in h["notes"])
    json.loads(json.dumps(an._clean(h)))


def test_health_price_coverage_counts_the_traded_universe_only(tmp_path, monkeypatch):
    """`priced` + `missing` must equal `traded`, they are a partition of one
    universe, so any other total is an impossible aggregate (§2.8).

    The bug this pins: `priced` counted every ticker in the `prices` table,
    which also holds benchmarks and option-only names that are not in the
    instrument='stock' denominator. On the live sample that published
    2,990 priced + 631 missing against a universe of 3,335, overstated
    coverage as 89.7% against a true 81.1%, and, because the inflated figure
    cleared the `cov < 0.85` gate, let the first panel on the dashboard report
    verdict "ok" with no notes at a coverage level it exists to flag.
    """
    from pathlib import Path

    import pandas as pd

    from copytrade.config import Settings
    from copytrade.store import db as store

    db = tmp_path / "coverage.db"
    conn = store.connect(db)
    store.run_migrations(conn, Path(__file__).resolve().parents[1] / "migrations")
    fid = store.upsert_filer(conn, "X", "congress")
    # traded universe = 3 stock tickers; OPT is traded but not as stock, so it
    # is outside the denominator and must not count toward coverage either
    for i, (tkr, instrument) in enumerate(
        [("AAA", "stock"), ("BBB", "stock"), ("CCC", "stock"), ("OPT", "call_option")]
    ):
        store.insert_trade(
            conn, filer_id=fid, source="house_clerk", source_uid=f"u{i}",
            ticker=tkr, side="buy", instrument=instrument,
            transaction_date="2021-05-01", filing_datetime="2021-05-03 17:00:00")
    # priced: 2 of the 3 stock names, plus a benchmark and the option-only name
    # that exist in `prices` but are NOT part of the traded stock universe
    store.upsert_prices(conn, pd.DataFrame([
        {"ticker": t, "date": "2021-05-03", "adj_close": 10.0}
        for t in ("AAA", "BBB", "SPY", "OPT")
    ]))
    conn.commit()
    conn.close()
    monkeypatch.setattr(an, "load_settings",
                        lambda *a, **k: Settings(db_path=str(db),
                                                 reports_dir=str(tmp_path)))
    an.reset_shared()
    h = an.build_health()

    assert h["traded_tickers"] == 3
    assert h["priced_tickers"] == 2, "SPY and OPT are not in the traded stock universe"
    assert h["missing_prices"] == 1
    assert h["priced_tickers"] + h["missing_prices"] == h["traded_tickers"]
    assert h["coverage"] == pytest.approx(2 / 3)
    assert h["coverage"] <= 1.0
    # and the gate must actually fire at that coverage
    assert any("traded tickers are priced" in n for n in h["notes"])


def test_api_rejects_unknown_analysis_name():
    from fastapi.testclient import TestClient

    from copytrade.web.app import app

    c = TestClient(app)
    r = c.get("/api/analysis/not_a_real_panel")
    assert r.status_code == 404
    assert "unknown analysis" in r.text


def test_api_reports_unbuilt_panels_in_the_index():
    from fastapi.testclient import TestClient

    from copytrade.web.app import app

    c = TestClient(app)
    r = c.get("/api/analysis")
    assert r.status_code == 200
    body = r.json()
    for name in an.BUILDERS:
        assert name in body
        assert "built" in body[name] and "built_at" in body[name]
