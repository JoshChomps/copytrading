"""Legislator metadata enrichment, party/state for congress filers.

Source: github.com/unitedstates/congress-legislators (the canonical open
dataset behind most congress-tracking tools), YAML in-tree so it is
reachable via raw.githubusercontent even from git-only sandboxes.

Why party matters (research context, see docs/CANADA.md §research): the
NANC/KRUZ ETF pair and the follow-up literature show the two party cohorts
hold very different books (tech-heavy vs industrials) with very different
outcomes: pooling them hides that. The report draws party-cohort decay
curves and the ranking carries a party column.

Name matching is deliberately conservative:
  * only records with a Senate term ending in the STOCK Act era
    (>= 2012) are candidates, the historical file goes back to 1789 and
    contains e.g. an 1815 'Roberts' who must never match Pat Roberts;
  * a filer matches only when the last name matches AND a first-name token
    (or nickname, or initial) corroborates, AND exactly one candidate
    survives. Ambiguity -> unmatched, counted, never guessed.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import httpx

BASE = "https://raw.githubusercontent.com/unitedstates/congress-legislators/main"
FILES = ("legislators-historical.yaml", "legislators-current.yaml")
ERA_START = "2012-01-01"   # STOCK Act signed April 2012; disclosures begin

_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}

# Common US-politics diminutives: 'joe' is NOT a prefix of 'joseph', so
# prefix rules alone miss half the Senate. Kept small and unambiguous.
_DIMINUTIVES = {
    "joe": "joseph", "tom": "thomas", "bill": "william", "bob": "robert",
    "jim": "james", "dick": "richard", "rick": "richard", "dan": "daniel",
    "mike": "michael", "chuck": "charles", "ted": "theodore", "pat": "patrick",
    "chris": "christopher", "tim": "timothy", "rob": "robert", "ben": "benjamin",
    "steve": "steven", "greg": "gregory", "ed": "edward", "tony": "anthony",
}

# Curated factual aliases where no rule can bridge legal vs common name
# (e.g. Ted Cruz's legal name is Rafael Edward Cruz). Keys are normalized
# filer strings as they appear in the senate data.
KNOWN_ALIASES = {
    "rafael e cruz": "ted cruz",
}


def _canon(token: str) -> set[str]:
    out = {token}
    if token in _DIMINUTIVES:
        out.add(_DIMINUTIVES[token])
    for dim, formal in _DIMINUTIVES.items():
        if formal == token:
            out.add(dim)
    return out


def _norm_tokens(name: str) -> list[str]:
    s = re.sub(r"[^\w\s]", " ", name.lower())
    return [t for t in s.split() if t and t not in _SUFFIXES]


class LegislatorIndex:
    """last-name -> [{first, nickname, party, state, bioguide}] for
    era-relevant legislators (both chambers by default, House members trade
    too, and their party powers the cohort analysis)."""

    def __init__(self, records: list[dict], chambers: tuple[str, ...] = ("sen", "rep")):
        self.by_last: dict[str, list[dict]] = {}
        for rec in records:
            terms = [t for t in rec.get("terms", []) if t.get("type") in chambers]
            if not terms or max(t.get("end", "") for t in terms) < ERA_START:
                continue
            last_term = max(terms, key=lambda t: t.get("end", ""))
            nm = rec.get("name", {})
            entry = {
                "first": (nm.get("first") or "").lower(),
                "nickname": (nm.get("nickname") or "").lower(),
                "middle": (nm.get("middle") or "").lower(),
                "official": nm.get("official_full") or f"{nm.get('first','')} {nm.get('last','')}",
                "party": last_term.get("party"),
                "state": last_term.get("state"),
                "bioguide": (rec.get("id") or {}).get("bioguide"),
            }
            last = (nm.get("last") or "").lower()
            if last:
                self.by_last.setdefault(last, []).append(entry)
                # compound surnames ('Van Hollen') are one dataset token but
                # two filer-name tokens: index the final word too
                if " " in last:
                    self.by_last.setdefault(last.split()[-1], []).append(entry)

    def match(self, filer_name: str) -> dict | None:
        tokens = _norm_tokens(filer_name)
        tokens = _norm_tokens(KNOWN_ALIASES.get(" ".join(tokens), " ".join(tokens)))
        if len(tokens) < 2:
            return None
        candidates = self.by_last.get(tokens[-1], [])
        fore = tokens[:-1]

        def corroborated(c: dict) -> bool:
            names = {c["first"], c["nickname"], c["middle"]} - {""}
            for t in fore:
                if _canon(t) & set().union(*(_canon(n) for n in names)):
                    return True  # exact or diminutive/formal pair
                # initial ('a mitchell mcconnell' -> 'a' vs 'addison')
                if len(t) == 1 and any(n.startswith(t) for n in names):
                    return True
                # shared 4-char stem: Jacky/Jacklyn, Rob/Robert, safe only
                # because the last name already matched and a unique hit is
                # still required
                if len(t) >= 4 and any(n[:4] == t[:4] for n in names if len(n) >= 4):
                    return True
            return False

        hits = [c for c in candidates if corroborated(c)]
        return hits[0] if len(hits) == 1 else None


def load_index(cache_dir: str | Path = "data/raw", refresh: bool = False) -> LegislatorIndex:
    """Fetch (with on-disk cache) and parse both YAML files."""
    import yaml

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    for fname in FILES:
        cache = cache_dir / fname
        if refresh or not cache.exists():
            resp = httpx.get(f"{BASE}/{fname}", timeout=120, follow_redirects=True)
            resp.raise_for_status()
            cache.write_bytes(resp.content)
        # explicit utf-8: the source YAML carries accented names, and Windows
        # would otherwise decode as cp1252 and raise UnicodeDecodeError
        records.extend(yaml.safe_load(cache.read_text(encoding="utf-8")))
    return LegislatorIndex(records)


def enrich_congress_filers(conn, index: LegislatorIndex) -> dict:
    """Merge party/state/bioguide into filers.meta for kind='congress'.
    Returns match statistics; unmatched filers are listed, not guessed."""
    matched = 0
    unmatched: list[str] = []
    for row in conn.execute("SELECT filer_id, name, meta FROM filers WHERE kind='congress'"):
        hit = index.match(row["name"])
        if hit is None:
            unmatched.append(row["name"])
            continue
        meta = json.loads(row["meta"] or "{}")
        meta.update({"party": hit["party"], "state": hit["state"],
                     "bioguide": hit["bioguide"]})
        conn.execute("UPDATE filers SET meta=? WHERE filer_id=?",
                     (json.dumps(meta), row["filer_id"]))
        matched += 1
    conn.commit()
    return {"matched": matched, "unmatched": len(unmatched),
            "unmatched_names": sorted(unmatched)}
