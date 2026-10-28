"""Congressional committee assignments, the last untested channel.

WHY THIS EXISTS. Every angle this project has tested treats filers as
interchangeable, and all of them came back null: no persistence between filers,
no horizon carries a signal, no conviction subset separates, no exit rule
helps. The one hypothesis that has never been tested here is the one the
academic literature actually makes (Ziobrowski et al. 2004/2011): that members
trade better INSIDE THEIR COMMITTEE'S JURISDICTION, because that is where the
informational advantage plausibly sits. A defence-committee member buying a
defence contractor is a different object from the same member buying a
supermarket chain.

Testing that needs three things the DB did not have: who sits on what, what
each committee covers, and what sector each traded ticker belongs to.

THE HONEST LIMITATION, STATED UP FRONT. The upstream source publishes
`committee-membership-current.yaml`, CURRENT membership only. There is no
historical membership file. So this ingest captures a SNAPSHOT, stamped with
`as_of`, and any return test built on it must restrict itself to trades filed
while that snapshot was in force. Applying today's roster to a 2016 trade would
be precisely the look-ahead that `abnormal_size_ids_pit` was written to remove.
`copytrade jurisdiction` enforces the restriction rather than trusting the
caller to remember it, and reports the resulting sample size so an underpowered
answer is visible as underpowered.

Members are linked by `bioguide`, which `legislators.py` already writes into
`filers.meta`: so this joins cleanly onto the filers the backtest uses.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import httpx
import yaml

BASE = "https://unitedstates.github.io/congress-legislators"
COMMITTEES_URL = f"{BASE}/committees-current.yaml"
MEMBERSHIP_URL = f"{BASE}/committee-membership-current.yaml"
# path inside the upstream git repo, for the historical walk
MEMBERSHIP_FILE = "committee-membership-current.yaml"


def fetch_yaml(url: str, timeout: int = 120) -> object:
    r = httpx.get(url, timeout=timeout, follow_redirects=True)
    r.raise_for_status()
    return yaml.safe_load(r.text)


def parse_committees(raw: list[dict]) -> list[dict]:
    """Flatten committees + their subcommittees into one row per body.

    Subcommittee ids upstream are relative (e.g. '15' under HSAG), so they are
    prefixed with the parent to make a globally unique key that matches how the
    membership file names them (HSAG15).
    """
    out: list[dict] = []
    for c in raw or []:
        tid = c.get("thomas_id")
        if not tid:
            continue
        out.append({
            "thomas_id": tid, "type": c.get("type"), "name": c.get("name"),
            "jurisdiction": c.get("jurisdiction"), "parent_id": None,
        })
        for sc in c.get("subcommittees") or []:
            sid = sc.get("thomas_id")
            if not sid:
                continue
            out.append({
                "thomas_id": f"{tid}{sid}", "type": c.get("type"),
                "name": sc.get("name"), "jurisdiction": sc.get("jurisdiction"),
                "parent_id": tid,
            })
    return out


def parse_membership(raw: dict) -> list[dict]:
    """One row per (committee, member). Entries without a bioguide are dropped
    rather than name-matched: a wrong join here would silently attribute a
    committee seat to the wrong person."""
    out: list[dict] = []
    for tid, members in (raw or {}).items():
        for m in members or []:
            bio = m.get("bioguide")
            if not bio:
                continue
            out.append({
                "thomas_id": tid, "bioguide": bio, "name": m.get("name"),
                "party": m.get("party"), "rank": m.get("rank"),
                "title": m.get("title"),
            })
    return out


def store(conn, committees: list[dict], members: list[dict],
          as_of: str | None = None) -> dict:
    as_of = as_of or date.today().isoformat()
    conn.executemany(
        "INSERT INTO committees(thomas_id,type,name,jurisdiction,parent_id,as_of) "
        "VALUES (:thomas_id,:type,:name,:jurisdiction,:parent_id,:as_of) "
        "ON CONFLICT(thomas_id) DO UPDATE SET type=excluded.type, "
        "name=excluded.name, jurisdiction=excluded.jurisdiction, "
        "parent_id=excluded.parent_id, as_of=excluded.as_of",
        [{**c, "as_of": as_of} for c in committees],
    )
    conn.executemany(
        "INSERT INTO committee_members(thomas_id,bioguide,name,party,rank,title,as_of) "
        "VALUES (:thomas_id,:bioguide,:name,:party,:rank,:title,:as_of) "
        "ON CONFLICT(thomas_id,bioguide,as_of) DO UPDATE SET "
        "party=excluded.party, rank=excluded.rank, title=excluded.title",
        [{**m, "as_of": as_of} for m in members],
    )
    conn.commit()
    return {"committees": len(committees), "memberships": len(members),
            "as_of": as_of}


def link_filers(conn) -> dict:
    """How many congress filers in the DB can be matched to a committee seat.

    Reported rather than assumed: a filer with no bioguide (unmatched by
    `enrich-legislators`) or one who has since left Congress cannot be placed
    on a committee, and any jurisdiction test silently excludes them.
    """
    seats = {r[0] for r in conn.execute("SELECT DISTINCT bioguide FROM committee_members")}
    matched, unmatched, no_bio = [], [], []
    for row in conn.execute("SELECT filer_id, name, meta FROM filers WHERE kind='congress'"):
        try:
            meta = json.loads(row["meta"]) if row["meta"] else {}
        except (TypeError, ValueError):
            meta = {}
        bio = meta.get("bioguide")
        if not bio:
            no_bio.append(row["name"])
        elif bio in seats:
            matched.append(row["name"])
        else:
            unmatched.append(row["name"])
    return {
        "filers_with_seat": len(matched),
        "filers_no_seat": len(unmatched),
        "filers_no_bioguide": len(no_bio),
        "examples_no_seat": sorted(unmatched)[:8],
        "examples_no_bioguide": sorted(no_bio)[:8],
    }


def refresh(conn, as_of: str | None = None) -> dict:
    committees = parse_committees(fetch_yaml(COMMITTEES_URL))
    members = parse_membership(fetch_yaml(MEMBERSHIP_URL))
    res = store(conn, committees, members, as_of)
    res.update(link_filers(conn))
    res["with_jurisdiction"] = sum(1 for c in committees if c.get("jurisdiction"))
    return res


REPO_URL = "https://github.com/unitedstates/congress-legislators"


def clone_or_pull(repo_dir: str | Path) -> Path:
    """Blobless clone of the upstream repo (or fetch if present).

    `--filter=blob:none` fetches history without file contents, so the clone is
    fast; blobs are pulled on demand by `git show`. That matters because the
    point of this is to walk 200+ historical revisions of one file.
    """
    import subprocess

    d = Path(repo_dir)
    if (d / ".git").exists():
        subprocess.run(["git", "-C", str(d), "fetch", "--quiet", "origin"],
                       check=False, timeout=600)
    else:
        d.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--filter=blob:none", "--quiet",
                        REPO_URL, str(d)], check=True, timeout=1800)
    return d


def membership_revisions(repo_dir: str | Path) -> list[tuple[str, str]]:
    """[(commit_sha, iso_date)] for every revision of the membership file,
    oldest first."""
    import subprocess

    out = subprocess.run(
        ["git", "-C", str(repo_dir), "log", "--reverse",
         "--format=%H %ad", "--date=short", "--", MEMBERSHIP_FILE],
        capture_output=True, text=True, check=True, timeout=600).stdout
    revs = []
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 2:
            revs.append((parts[0], parts[1]))
    return revs


def thin_revisions(revs: list[tuple[str, str]], min_gap_days: int = 60
                   ) -> list[tuple[str, str]]:
    """Keep roughly one snapshot per `min_gap_days`, plus the newest.

    Committee membership changes slowly, most commits are single-seat
    corrections: so storing all 200+ revisions multiplies the table for no
    resolution gain. A ~2-month grid is finer than the quarterly cadence at
    which assignments actually move.
    """
    kept: list[tuple[str, str]] = []
    last: date | None = None
    for sha, d in revs:
        cur = date.fromisoformat(d)
        if last is None or (cur - last).days >= min_gap_days:
            kept.append((sha, d))
            last = cur
    if revs and (not kept or kept[-1][0] != revs[-1][0]):
        kept.append(revs[-1])
    return kept


def refresh_history(conn, repo_dir: str | Path = "data/congress-legislators",
                    min_gap_days: int = 60, since: str = "2014-01-01") -> dict:
    """Ingest POINT-IN-TIME committee membership from the upstream git history.

    This is what makes the jurisdiction test usable. With only the current
    roster, scoring a 2016 trade means applying 2026 committee assignments
    look-ahead: so the test has to be restricted to the current Congress and
    loses ~96% of the sample. Walking the history gives the roster that was
    actually in force on each filing date.
    """
    import subprocess

    d = clone_or_pull(repo_dir)
    revs = [r for r in membership_revisions(d) if r[1] >= since]
    picked = thin_revisions(revs, min_gap_days)

    # committee metadata (jurisdictions) changes rarely; current is enough
    committees = parse_committees(fetch_yaml(COMMITTEES_URL))
    store(conn, committees, [], as_of=date.today().isoformat())

    total = 0
    failed = 0
    for sha, day in picked:
        try:
            raw = subprocess.run(
                ["git", "-C", str(d), "show", f"{sha}:{MEMBERSHIP_FILE}"],
                capture_output=True, text=True, check=True, timeout=300,
                encoding="utf-8", errors="replace").stdout
            members = parse_membership(yaml.safe_load(raw))
        except Exception:  # noqa: BLE001, one bad revision must not stop the walk
            failed += 1
            continue
        if not members:
            continue
        conn.executemany(
            "INSERT INTO committee_members"
            "(thomas_id,bioguide,name,party,rank,title,as_of) "
            "VALUES (:thomas_id,:bioguide,:name,:party,:rank,:title,:as_of) "
            "ON CONFLICT(thomas_id,bioguide,as_of) DO NOTHING",
            [{**m, "as_of": day} for m in members],
        )
        total += len(members)
    conn.commit()

    snaps = [r[0] for r in conn.execute(
        "SELECT DISTINCT as_of FROM committee_members ORDER BY as_of")]
    res = {
        "revisions_available": len(revs),
        "snapshots_ingested": len(picked),
        "snapshots_in_db": len(snaps),
        "memberships_written": total,
        "failed_revisions": failed,
        "earliest": snaps[0] if snaps else None,
        "latest": snaps[-1] if snaps else None,
    }
    res.update(link_filers(conn))
    return res


def load_snapshots(conn) -> list[tuple[str, dict[str, set[str]]]]:
    """All membership snapshots as [(as_of, {bioguide: {committee_id}})],
    oldest first, with subcommittees rolled up to their parent.

    Loaded once and bisected per trade, a per-trade SQL lookup across 12k
    trades and 20+ snapshots would dominate the analysis runtime.
    """
    parents = {r[0]: (r[1] or r[0]) for r in
               conn.execute("SELECT thomas_id, parent_id FROM committees")}
    out: dict[str, dict[str, set[str]]] = {}
    for tid, bio, as_of in conn.execute(
            "SELECT thomas_id, bioguide, as_of FROM committee_members"):
        out.setdefault(as_of, {}).setdefault(bio, set()).add(parents.get(tid, tid))
    return sorted(out.items())


def snapshot_for(snapshots: list[tuple[str, dict]], on_date: str) -> dict:
    """The membership roster in force on `on_date`, the latest snapshot at or
    before it. Returns {} when the date precedes every snapshot, so callers
    exclude the trade rather than scoring it with a roster from the future."""
    import bisect

    keys = [s[0] for s in snapshots]
    i = bisect.bisect_right(keys, on_date) - 1
    return snapshots[i][1] if i >= 0 else {}


def member_committees(conn, bioguide: str, as_of: str | None = None) -> set[str]:
    """Top-level committee ids for a member (subcommittees rolled up to parent,
    since jurisdiction is set at committee level)."""
    q = ("SELECT cm.thomas_id, c.parent_id FROM committee_members cm "
         "LEFT JOIN committees c USING(thomas_id) WHERE cm.bioguide=?")
    args: list = [bioguide]
    if as_of:
        q += " AND cm.as_of=?"
        args.append(as_of)
    return {(r[1] or r[0]) for r in conn.execute(q, args)}
