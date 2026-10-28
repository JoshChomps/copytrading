import json
from pathlib import Path

import pandas as pd
import pytest

from copytrade.ingest.committees import (
    link_filers,
    member_committees,
    parse_committees,
    parse_membership,
    store,
)
from copytrade.rank.jurisdiction import (
    BROAD_EXCLUDED,
    COMMITTEE_SECTORS,
    SECTORS,
    classify,
    compare,
    eligibility_report,
    jurisdiction_sectors,
)

MIGRATIONS = Path(__file__).resolve().parents[1] / "migrations"


def _db(tmp_path):
    from copytrade.store import db as store_

    conn = store_.connect(tmp_path / "j.db")
    store_.run_migrations(conn, MIGRATIONS)
    return conn


# ------------------------------------------------------------------ ingest --

def test_parse_committees_prefixes_subcommittee_ids():
    """Upstream subcommittee ids are relative ('15' under HSAG) but the
    membership file names them HSAG15, without the prefix the join silently
    finds nothing."""
    raw = [{
        "thomas_id": "HSAG", "type": "house", "name": "Agriculture",
        "jurisdiction": "farms", "subcommittees": [
            {"thomas_id": "15", "name": "Livestock"},
        ],
    }]
    out = parse_committees(raw)
    ids = {c["thomas_id"]: c for c in out}
    assert set(ids) == {"HSAG", "HSAG15"}
    assert ids["HSAG15"]["parent_id"] == "HSAG"
    assert ids["HSAG"]["parent_id"] is None


def test_parse_membership_drops_entries_without_a_bioguide():
    """Name-matching a committee seat would risk attributing it to the wrong
    person, so an entry with no id is dropped rather than guessed."""
    out = parse_membership({"HSAG": [
        {"name": "A Member", "bioguide": "B001"},
        {"name": "No Id Here"},
    ]})
    assert [m["bioguide"] for m in out] == ["B001"]


def test_member_committees_rolls_subcommittees_up_to_the_parent(tmp_path):
    """Jurisdiction is defined at committee level, so a subcommittee seat must
    confer the parent's remit."""
    conn = _db(tmp_path)
    store(conn,
          parse_committees([{
              "thomas_id": "HSAG", "type": "house", "name": "Ag",
              "subcommittees": [{"thomas_id": "15", "name": "Livestock"}]}]),
          parse_membership({"HSAG15": [{"name": "X", "bioguide": "B001"}]}),
          as_of="2026-01-01")
    assert member_committees(conn, "B001") == {"HSAG"}


def test_link_filers_separates_no_bioguide_from_no_seat(tmp_path):
    """Two different failures with different fixes: one needs
    `enrich-legislators`, the other means the member simply left Congress."""
    conn = _db(tmp_path)
    store(conn, parse_committees([{"thomas_id": "HSAG", "type": "house", "name": "Ag"}]),
          parse_membership({"HSAG": [{"name": "Seated", "bioguide": "B001"}]}),
          as_of="2026-01-01")
    conn.executemany(
        "INSERT INTO filers(name, kind, meta) VALUES (?,?,?)",
        [("Seated", "congress", json.dumps({"bioguide": "B001"})),
         ("Left Congress", "congress", json.dumps({"bioguide": "B999"})),
         ("Unmatched", "congress", None)],
    )
    conn.commit()
    res = link_filers(conn)
    assert res["filers_with_seat"] == 1
    assert res["filers_no_seat"] == 1
    assert res["filers_no_bioguide"] == 1


# ------------------------------------------------------------ jurisdiction --

def test_every_mapped_sector_is_a_real_yfinance_sector():
    """A typo in the map would silently never match, producing a permanent and
    invisible null."""
    for cid, secs in COMMITTEE_SECTORS.items():
        unknown = secs - SECTORS
        assert not unknown, f"{cid} maps to unknown sector(s): {unknown}"


def test_broad_committees_are_excluded_from_the_map():
    """Appropriations and Budget touch the whole economy; including them would
    mark nearly every trade in-jurisdiction and dissolve the comparison."""
    for cid in BROAD_EXCLUDED:
        assert cid not in COMMITTEE_SECTORS
    assert jurisdiction_sectors({"HSAP", "SSBU"}) == set()


def test_jurisdiction_sectors_unions_committees_and_ignores_broad_ones():
    got = jurisdiction_sectors({"HSAS", "SSHR", "HSAP"})
    assert "Industrials" in got and "Healthcare" in got


def _per_trade(rows):
    df = pd.DataFrame(rows)
    df["entry_date"] = pd.to_datetime(df["entry_date"])
    return df


def test_classify_excludes_trades_before_the_snapshot_window():
    """THE look-ahead guard: scoring a 2016 trade with a 2026 committee roster
    is the same error `abnormal_size_ids_pit` was written to remove."""
    pt = _per_trade([
        {"filer_id": 1, "ticker": "LMT", "entry_date": "2016-06-01", "h63": 0.05},
        {"filer_id": 1, "ticker": "LMT", "entry_date": "2025-06-01", "h63": 0.05},
    ])
    lab = classify(pt, {1: {"Industrials"}}, {"LMT": "Industrials"},
                   since="2025-01-03")
    assert len(lab) == 1
    assert lab["entry_date"].iloc[0].year == 2025
    assert bool(lab["in_jurisdiction"].iloc[0]) is True


def test_classify_marks_out_of_jurisdiction_trades():
    pt = _per_trade([
        {"filer_id": 1, "ticker": "KO", "entry_date": "2025-06-01", "h63": 0.01},
    ])
    lab = classify(pt, {1: {"Industrials"}}, {"KO": "Consumer Defensive"},
                   since="2025-01-03")
    assert bool(lab["in_jurisdiction"].iloc[0]) is False


def test_classify_drops_filers_without_a_mapped_jurisdiction():
    """A member with no seat must be EXCLUDED, not counted as out-of-
    jurisdiction: that would pad the control group with people the hypothesis
    says nothing about."""
    pt = _per_trade([
        {"filer_id": 1, "ticker": "KO", "entry_date": "2025-06-01", "h63": 0.01},
        {"filer_id": 2, "ticker": "KO", "entry_date": "2025-06-01", "h63": 0.01},
    ])
    lab = classify(pt, {1: {"Industrials"}}, {"KO": "Consumer Defensive"},
                   since="2025-01-03")
    assert list(lab["filer_id"]) == [1]


def test_classify_drops_tickers_without_a_sector():
    pt = _per_trade([
        {"filer_id": 1, "ticker": "ZZZZ", "entry_date": "2025-06-01", "h63": 0.01},
    ])
    assert classify(pt, {1: {"Industrials"}}, {"KO": "Consumer Defensive"},
                    since="2025-01-03").empty


def test_eligibility_report_discloses_the_attrition():
    pt = _per_trade([
        {"filer_id": 1, "ticker": "LMT", "entry_date": "2016-06-01", "h63": 0.05},
        {"filer_id": 1, "ticker": "LMT", "entry_date": "2025-06-01", "h63": 0.05},
        {"filer_id": 9, "ticker": "LMT", "entry_date": "2025-07-01", "h63": 0.05},
    ])
    rep = eligibility_report(pt, {1: {"Industrials"}}, {"LMT": "Industrials"},
                             since="2025-01-03")
    assert rep["total_priced_buys"] == 3
    assert rep["after_snapshot_date"] == 2
    assert rep["eligible"] == 1, "filer 9 has no jurisdiction"


def test_compare_returns_an_error_on_an_empty_frame():
    assert "error" in compare(pd.DataFrame(), "h63", 63)


def test_compare_detects_a_planted_jurisdiction_effect():
    rows = []
    for m in range(36):
        d = pd.Timestamp("2023-01-15") + pd.DateOffset(months=m)
        for _ in range(12):
            rows.append({"entry_date": d, "in_jurisdiction": True, "h63": 0.05})
            rows.append({"entry_date": d, "in_jurisdiction": False, "h63": 0.00})
    res = compare(pd.DataFrame(rows), "h63", 63)
    assert res["diff_mean"] > 0.04
    assert res["diff_t"] > 5
    assert res["paired_months"] == 36


def test_compare_is_null_when_both_groups_match():
    import numpy as np

    rng = np.random.default_rng(0)
    rows = []
    for m in range(36):
        d = pd.Timestamp("2023-01-15") + pd.DateOffset(months=m)
        for _ in range(12):
            rows.append({"entry_date": d, "in_jurisdiction": True,
                         "h63": float(rng.normal(0, 0.05))})
            rows.append({"entry_date": d, "in_jurisdiction": False,
                         "h63": float(rng.normal(0, 0.05))})
    res = compare(pd.DataFrame(rows), "h63", 63)
    assert abs(res["diff_t"]) < 2


# ------------------------------------- overlap-aware pairing on the headline --
#
# The group-level t's always clustered on non-overlapping blocks, but the
# paired DIFFERENCE, the number the verdict actually quotes, paired by
# calendar month and ran a one-sample t over 129 monthly differences. At a
# 63-day horizon consecutive months share ~2/3 of their holding window, so
# those are not 129 independent draws. Same correction as `timescale.block_test`.

def _paired_frame(n_months=36, effect=0.0, seed=0):
    import numpy as np

    rng = np.random.default_rng(seed)
    rows = []
    for m in range(n_months):
        d = pd.Timestamp("2020-01-15") + pd.DateOffset(months=m)
        for _ in range(12):
            rows.append({"entry_date": d, "in_jurisdiction": True,
                         "h63": effect + float(rng.normal(0, 0.05))})
            rows.append({"entry_date": d, "in_jurisdiction": False,
                         "h63": float(rng.normal(0, 0.05))})
    return pd.DataFrame(rows)


def test_headline_pairing_uses_blocks_as_long_as_the_horizon():
    res = compare(_paired_frame(), "h63", 63)
    # 63 trading days ~ 3 months, so 36 months collapse to 12 blocks
    assert res["paired_block_months"] == 3
    assert res["paired_blocks"] == 12
    assert res["paired_months"] == 36        # legacy figure still reported


def test_block_pairing_deflates_t_when_months_are_correlated():
    """This is the defect, reproduced.

    Overlapping holding periods make consecutive months share most of their
    window, so their differences move together. Monthly pairing treats them as
    independent draws and inflates the t; block pairing does not. The shared
    per-quarter shock below is exactly what a 63-day hold induces.
    """
    import numpy as np

    rng = np.random.default_rng(7)
    rows = []
    for m in range(36):
        d = pd.Timestamp("2020-01-15") + pd.DateOffset(months=m)
        shared = 0.02 if (m // 3) % 2 == 0 else -0.01   # persists across a block
        for _ in range(12):
            rows.append({"entry_date": d, "in_jurisdiction": True,
                         "h63": shared + float(rng.normal(0, 0.002))})
            rows.append({"entry_date": d, "in_jurisdiction": False, "h63": 0.0})
    res = compare(pd.DataFrame(rows), "h63", 63)
    assert abs(res["diff_t"]) < abs(res["diff_t_month"])


def test_a_one_day_horizon_still_pairs_by_month():
    """At h=1 there is no overlap to correct, so blocks collapse to months and
    the two figures agree, the correction must not fire where it is not needed."""
    res = compare(_paired_frame(), "h63", 1)
    assert res["paired_block_months"] == 1
    assert res["paired_blocks"] == res["paired_months"] == 36
    assert res["diff_t"] == pytest.approx(res["diff_t_month"])


def test_the_correction_is_free_when_observations_really_are_independent():
    """Blocking averages k months together but divides the sample by k, and
    with genuinely iid months those cancel. The correction should cost nothing
    where there is no overlap to correct, otherwise it would just be a penalty."""
    res = compare(_paired_frame(effect=0.01, seed=3), "h63", 63)
    assert abs(res["diff_t"]) == pytest.approx(abs(res["diff_t_month"]), rel=0.35)


def test_block_pairing_still_finds_a_large_real_effect():
    """Being conservative must not mean being blind."""
    res = compare(_paired_frame(effect=0.05, seed=1), "h63", 63)
    assert res["diff_mean"] > 0.04
    assert res["diff_t"] > 5


def test_unpaired_periods_are_dropped_not_scored_as_zero():
    """A period containing only one group carries no information about the
    contrast; counting it as a zero difference would dilute a real effect."""
    rows = []
    for m in range(24):
        d = pd.Timestamp("2020-01-15") + pd.DateOffset(months=m)
        rows.append({"entry_date": d, "in_jurisdiction": True, "h63": 0.05})
        if m % 6 != 0:      # some months have no out-of-jurisdiction trade
            rows.append({"entry_date": d, "in_jurisdiction": False, "h63": 0.0})
    res = compare(pd.DataFrame(rows), "h63", 63)
    assert res["diff_mean"] == pytest.approx(0.05)


# ------------------------------------------------- point-in-time membership --

def test_thin_revisions_keeps_a_spaced_grid_and_the_newest():
    from copytrade.ingest.committees import thin_revisions

    revs = [("a", "2015-01-01"), ("b", "2015-01-20"), ("c", "2015-04-01"),
            ("d", "2015-04-10"), ("e", "2026-07-01")]
    kept = thin_revisions(revs, min_gap_days=60)
    assert [k[0] for k in kept] == ["a", "c", "e"]
    assert kept[-1][0] == revs[-1][0], "the newest revision must always survive"


def test_snapshot_for_never_borrows_a_future_roster():
    """THE look-ahead guard for the historical path. A trade predating every
    snapshot must resolve to {} so the caller excludes it, scoring it with the
    earliest available roster would import assignments that did not exist."""
    from copytrade.ingest.committees import snapshot_for

    snaps = [("2015-01-01", {"B1": {"HSAG"}}), ("2020-01-01", {"B1": {"HSAS"}})]
    assert snapshot_for(snaps, "2014-06-01") == {}
    assert snapshot_for(snaps, "2016-06-01") == {"B1": {"HSAG"}}
    assert snapshot_for(snaps, "2025-01-01") == {"B1": {"HSAS"}}


def test_load_snapshots_rolls_subcommittees_up_and_groups_by_date(tmp_path):
    from copytrade.ingest.committees import load_snapshots

    conn = _db(tmp_path)
    store(conn,
          parse_committees([{"thomas_id": "HSAG", "type": "house", "name": "Ag",
                             "subcommittees": [{"thomas_id": "15", "name": "L"}]}]),
          [], as_of="2020-01-01")
    conn.executemany(
        "INSERT INTO committee_members(thomas_id,bioguide,as_of) VALUES (?,?,?)",
        [("HSAG15", "B1", "2015-01-01"), ("HSAG", "B1", "2020-01-01")])
    conn.commit()
    snaps = load_snapshots(conn)
    assert [s[0] for s in snaps] == ["2015-01-01", "2020-01-01"]
    assert snaps[0][1]["B1"] == {"HSAG"}, "subcommittee rolls up to parent"


def test_pit_lookup_tracks_a_member_moving_between_committees(tmp_path):
    """The whole point of the historical walk: the same member's jurisdiction
    must differ before and after a committee change."""
    from copytrade.rank.jurisdiction import build_pit_jurisdictions

    conn = _db(tmp_path)
    store(conn, parse_committees([
        {"thomas_id": "HSAS", "type": "house", "name": "Armed Services"},
        {"thomas_id": "SSHR", "type": "senate", "name": "HELP"}]),
        [], as_of="2020-01-01")
    conn.executemany(
        "INSERT INTO committee_members(thomas_id,bioguide,as_of) VALUES (?,?,?)",
        [("HSAS", "B1", "2015-01-01"), ("SSHR", "B1", "2021-01-01")])
    conn.execute("INSERT INTO filers(filer_id, name, kind, meta) VALUES (?,?,?,?)",
                 (1, "Mover", "congress", json.dumps({"bioguide": "B1"})))
    conn.commit()

    lookup, n = build_pit_jurisdictions(conn)
    assert n == 2
    assert lookup(1, "2014-01-01") is None, "before any snapshot => excluded"
    assert lookup(1, "2016-06-01") == {"Industrials"}
    assert lookup(1, "2022-06-01") == {"Healthcare"}


def test_classify_pit_uses_each_trades_own_date(tmp_path):
    from copytrade.rank.jurisdiction import classify_pit

    def lookup(filer_id, on_date):
        return {"Industrials"} if on_date < "2020-01-01" else {"Healthcare"}

    pt = _per_trade([
        {"filer_id": 1, "ticker": "LMT", "entry_date": "2016-06-01", "h63": 0.05},
        {"filer_id": 1, "ticker": "LMT", "entry_date": "2022-06-01", "h63": 0.05},
    ])
    lab = classify_pit(pt, lookup, {"LMT": "Industrials"})
    assert len(lab) == 2
    # same member, same ticker, opposite classification because the roster moved
    assert list(lab["in_jurisdiction"]) == [True, False]


def test_classify_pit_excludes_trades_with_no_roster():
    from copytrade.rank.jurisdiction import classify_pit

    pt = _per_trade([
        {"filer_id": 1, "ticker": "LMT", "entry_date": "2010-01-01", "h63": 0.05},
    ])
    assert classify_pit(pt, lambda f, d: None, {"LMT": "Industrials"}).empty
