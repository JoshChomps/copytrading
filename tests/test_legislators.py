"""Legislator matching against a REAL fixture (entries copied verbatim from
unitedstates/congress-legislators for senators present in the senate data)."""

import json

import yaml

from copytrade.ingest.legislators import LegislatorIndex, enrich_congress_filers
from copytrade.store import db as store
from tests.conftest import FIXTURES


def _index():
    recs = yaml.safe_load((FIXTURES / "legislators_real_sample.yaml").read_text())
    return LegislatorIndex(recs)


def test_matches_real_senate_watcher_name_variants():
    idx = _index()
    # exact messy strings observed in the real senate dataset
    assert _party(idx, "Pat Roberts") == "Republican"            # nickname match
    assert _party(idx, "A. Mitchell Mcconnell, Jr.") == "Republican"  # initial + middle
    assert _party(idx, "Ron L Wyden") == "Democrat"
    assert _party(idx, "David A Perdue , Jr") == "Republican"    # stray space + suffix
    assert _party(idx, "Angus S King, Jr.") == "Independent"


def _party(idx, name):
    hit = idx.match(name)
    return hit["party"] if hit else None


def test_no_match_is_none_never_guessed():
    idx = _index()
    assert idx.match("Zebulon Nobody") is None
    assert idx.match("Roberts") is None          # bare last name: ambiguous by design
    assert idx.match("Quincy Roberts") is None   # last matches, forename doesn't


def test_era_filter_excludes_pre_stock_act_senators():
    # a real historical record: an 1815-1821 senator must never be a candidate
    ancient = [{
        "name": {"first": "Jonathan", "last": "Roberts"},
        "terms": [{"type": "sen", "start": "1815-12-04", "end": "1821-03-03",
                   "state": "PA", "party": "Republican"}],
    }]
    idx = LegislatorIndex(ancient)
    assert idx.match("Jonathan Roberts") is None


def test_enrich_filers_updates_meta(conn):
    f = store.upsert_filer(conn, "Pat Roberts", "congress", {"chamber": "Senate"})
    store.upsert_filer(conn, "Unknown Person", "congress", None)
    stats = enrich_congress_filers(conn, _index())
    assert stats["matched"] == 1 and stats["unmatched"] == 1
    meta = json.loads(conn.execute(
        "SELECT meta FROM filers WHERE filer_id=?", (f,)).fetchone()[0])
    assert meta["party"] == "Republican" and meta["state"] == "KS"
    assert meta["chamber"] == "Senate"  # existing meta preserved
