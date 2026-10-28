"""Committee jurisdiction vs traded sector, the last untested hypothesis.

THE CLAIM. Ziobrowski et al. (2004, 2011) argued congressional portfolios beat
the market, and the mechanism usually proposed is committee jurisdiction: a
member of Armed Services plausibly learns something about defence procurement
before the public does, in a way they do not about supermarkets. Every other
angle this project has tested pools all filers and all trades together, which
would wash out an effect concentrated in a minority of trades.

So: split each disclosed buy by whether the traded company's sector falls
inside the buyer's own committee jurisdiction, and compare.

TWO THINGS MAKE THIS HARD, AND BOTH ARE HANDLED EXPLICITLY.

1. MEMBERSHIP IS A SNAPSHOT. Upstream publishes current committee membership
   only. Applying today's roster to a 2016 trade is look-ahead. `jurisdiction`
   therefore restricts the test to trades filed on/after the snapshot's
   Congress convened, and reports the surviving sample so an underpowered
   answer is visible as underpowered rather than quietly asserted.

2. THE COMMITTEE->SECTOR MAP IS JUDGEMENT, NOT DATA. There is no authoritative
   crosswalk from a committee's prose jurisdiction to GICS sectors, so the map
   below is hand-authored from each committee's published remit. It is a
   modelling choice and is stated as one. It is deliberately CONSERVATIVE:
   only sectors a committee plainly legislates over are included, because a
   generous map would mark most trades as "in jurisdiction" and destroy the
   contrast the test depends on. Appropriations and Budget are deliberately
   EXCLUDED entirely: they touch everything, so including them would make
   their members match every sector and manufacture a result.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# yfinance sector vocabulary (what `sectors-refresh` stores).
SECTORS = {
    "Technology", "Financial Services", "Healthcare", "Consumer Cyclical",
    "Consumer Defensive", "Energy", "Industrials", "Basic Materials",
    "Real Estate", "Utilities", "Communication Services",
}

# Committee (thomas_id) -> sectors it plainly legislates over.
# House ids start H, Senate ids start S. Subcommittees roll up to their parent
# in `member_committees`, so only top-level ids appear here.
COMMITTEE_SECTORS: dict[str, set[str]] = {
    # --- House ---
    "HSAG": {"Consumer Defensive", "Basic Materials"},      # Agriculture
    "HSAS": {"Industrials"},                                # Armed Services
    "HSBA": {"Financial Services", "Real Estate"},          # Financial Services
    "HSED": {"Healthcare"},                                 # Education & Workforce
    "HSIF": {"Energy", "Utilities", "Healthcare",           # Energy & Commerce
             "Communication Services", "Technology"},       #   very broad remit
    "HSFA": set(),                                          # Foreign Affairs
    "HSGO": set(),                                          # Oversight
    "HSHM": {"Industrials", "Technology"},                  # Homeland Security
    "HSJU": {"Technology", "Communication Services"},       # Judiciary (antitrust)
    "HSII": {"Energy", "Basic Materials", "Utilities"},     # Natural Resources
    "HSPW": {"Industrials", "Basic Materials"},             # Transportation & Infra
    "HSSY": {"Technology", "Industrials"},                  # Science, Space & Tech
    "HSSM": set(),                                          # Small Business
    "HSVR": {"Healthcare"},                                 # Veterans' Affairs
    "HSWM": {"Financial Services", "Healthcare"},           # Ways & Means
    "HLIG": {"Industrials", "Technology"},                  # Intelligence
    # --- Senate ---
    "SSAF": {"Consumer Defensive", "Basic Materials"},      # Agriculture
    "SSAS": {"Industrials"},                                # Armed Services
    "SSBK": {"Financial Services", "Real Estate"},          # Banking & Housing
    "SSCM": {"Communication Services", "Technology",        # Commerce, Science
             "Industrials", "Consumer Cyclical"},
    "SSEG": {"Energy", "Utilities", "Basic Materials"},     # Energy & Nat. Resources
    "SSEV": {"Utilities", "Basic Materials", "Industrials"},  # Environment & Public Works
    "SSFI": {"Financial Services", "Healthcare"},           # Finance
    "SSFR": set(),                                          # Foreign Relations
    "SSGA": set(),                                          # Homeland Sec & Gov Affairs
    "SSHR": {"Healthcare"},                                 # Health, Ed, Labor, Pensions
    "SSJU": {"Technology", "Communication Services"},       # Judiciary
    "SSSB": set(),                                          # Small Business
    "SSVA": {"Healthcare"},                                 # Veterans' Affairs
    "SLIN": {"Industrials", "Technology"},                  # Intelligence
}

# Excluded on purpose: their remit spans the whole economy, so including them
# would mark nearly every trade in-jurisdiction and dissolve the comparison.
BROAD_EXCLUDED = {"HSAP", "SSAP", "HSBU", "SSBU"}  # Appropriations, Budget

# 119th Congress convened 2025-01-03. A current-membership snapshot is only a
# fair description of assignments from that date on.
CURRENT_CONGRESS_START = "2025-01-03"


def jurisdiction_sectors(committee_ids: set[str]) -> set[str]:
    """Union of sectors covered by a member's committees (broad ones dropped)."""
    out: set[str] = set()
    for cid in committee_ids:
        if cid in BROAD_EXCLUDED:
            continue
        out |= COMMITTEE_SECTORS.get(cid, set())
    return out


def build_pit_jurisdictions(conn):
    """Point-in-time jurisdiction lookup: (filer_id, date) -> sectors.

    Returns (lookup_fn, n_snapshots). The function resolves each trade against
    the committee roster in force on its OWN filing date, which is what removes
    the look-ahead that forced the current-Congress restriction. Snapshots are
    loaded once and bisected per trade rather than queried per row.
    """
    import json

    from copytrade.ingest.committees import load_snapshots, snapshot_for

    snapshots = load_snapshots(conn)
    bio_of: dict[int, str] = {}
    for row in conn.execute("SELECT filer_id, meta FROM filers WHERE kind='congress'"):
        try:
            meta = json.loads(row["meta"]) if row["meta"] else {}
        except (TypeError, ValueError):
            meta = {}
        if meta.get("bioguide"):
            bio_of[row["filer_id"]] = meta["bioguide"]

    cache: dict[tuple[str, str], frozenset] = {}

    def lookup(filer_id: int, on_date: str) -> set[str] | None:
        bio = bio_of.get(filer_id)
        if not bio:
            return None
        key = (bio, on_date[:7])          # month resolution is plenty
        if key not in cache:
            roster = snapshot_for(snapshots, on_date)
            cache[key] = frozenset(jurisdiction_sectors(roster.get(bio, set())))
        return set(cache[key]) or None

    return lookup, len(snapshots)


def build_member_jurisdictions(conn, as_of: str | None = None) -> dict[int, set[str]]:
    """{filer_id: sectors in that member's committee jurisdiction}.

    Filers with no bioguide, no seat, or only broad-committee seats map to an
    empty set and are excluded from the test by the caller, never silently
    treated as "no jurisdiction match", which would pad the control group.
    """
    import json

    from copytrade.ingest.committees import member_committees

    out: dict[int, set[str]] = {}
    for row in conn.execute("SELECT filer_id, meta FROM filers WHERE kind='congress'"):
        try:
            meta = json.loads(row["meta"]) if row["meta"] else {}
        except (TypeError, ValueError):
            meta = {}
        bio = meta.get("bioguide")
        if not bio:
            continue
        sectors = jurisdiction_sectors(member_committees(conn, bio, as_of))
        if sectors:
            out[row["filer_id"]] = sectors
    return out


def load_sector_map(conn) -> dict[str, str]:
    return {r[0]: r[1] for r in conn.execute(
        "SELECT ticker, sector FROM sector_map WHERE sector IS NOT NULL AND sector != ''")}


def classify_pit(per_trade: pd.DataFrame, lookup, sectors: dict[str, str]
                 ) -> pd.DataFrame:
    """Label each priced buy in/out of jurisdiction using the roster in force on
    ITS OWN filing date.

    No date restriction is needed: `lookup` returns None for a trade that
    predates every snapshot, so such trades are excluded rather than scored
    with a roster that did not exist yet.
    """
    df = per_trade.copy()
    df["sector"] = df["ticker"].map(sectors)
    dates = pd.to_datetime(df["entry_date"]).dt.strftime("%Y-%m-%d")
    df["juris"] = [lookup(f, d) for f, d in zip(df["filer_id"], dates)]
    df = df[df["sector"].notna() & df["juris"].notna()].copy()
    if df.empty:
        return df
    df["in_jurisdiction"] = [bool(s in j) for s, j in zip(df["sector"], df["juris"])]
    return df


def eligibility_pit(per_trade: pd.DataFrame, lookup,
                    sectors: dict[str, str]) -> dict:
    """Where the sample went, without a date cut-off."""
    dates = pd.to_datetime(per_trade["entry_date"]).dt.strftime("%Y-%m-%d")
    has_j = [lookup(f, d) is not None for f, d in zip(per_trade["filer_id"], dates)]
    has_s = per_trade["ticker"].map(sectors).notna()
    return {
        "total_priced_buys": int(len(per_trade)),
        "with_sector": int(has_s.sum()),
        "with_jurisdiction": int(sum(has_j)),
        "eligible": int((pd.Series(has_j).values & has_s.values).sum()),
        "mode": "point-in-time (roster in force on each filing date)",
    }


def classify(per_trade: pd.DataFrame, jurisdictions: dict[int, set[str]],
             sectors: dict[str, str], since: str = CURRENT_CONGRESS_START
             ) -> pd.DataFrame:
    """Label each priced buy in/out of the buyer's committee jurisdiction.

    Returns the eligible subset only: trades filed on/after `since` (so the
    membership snapshot applies), by a filer with a known jurisdiction, in a
    ticker with a known sector. Everything dropped is counted by
    `eligibility_report` so the attrition is visible.
    """
    df = per_trade.copy()
    df["sector"] = df["ticker"].map(sectors)
    df["juris"] = df["filer_id"].map(jurisdictions)
    ok = (
        (pd.to_datetime(df["entry_date"]) >= pd.Timestamp(since))
        & df["sector"].notna()
        & df["juris"].notna()
    )
    df = df[ok].copy()
    if df.empty:
        return df
    df["in_jurisdiction"] = [
        bool(s in j) for s, j in zip(df["sector"], df["juris"])
    ]
    return df


def eligibility_report(per_trade: pd.DataFrame, jurisdictions: dict[int, set[str]],
                       sectors: dict[str, str],
                       since: str = CURRENT_CONGRESS_START) -> dict:
    """Where the sample went. Attrition here is large and must be disclosed:
    the snapshot restriction alone discards most of the history."""
    n = len(per_trade)
    dt = pd.to_datetime(per_trade["entry_date"])
    after = int((dt >= pd.Timestamp(since)).sum())
    sub = per_trade[dt >= pd.Timestamp(since)]
    with_sector = int(sub["ticker"].map(sectors).notna().sum())
    with_juris = int(sub["filer_id"].map(jurisdictions).notna().sum())
    both = int((sub["ticker"].map(sectors).notna()
                & sub["filer_id"].map(jurisdictions).notna()).sum())
    return {
        "total_priced_buys": n,
        "since": since,
        "after_snapshot_date": after,
        "of_those_with_sector": with_sector,
        "of_those_with_jurisdiction": with_juris,
        "eligible": both,
        "attrition_note": (
            "membership is a CURRENT snapshot, so trades before it are excluded "
            "rather than scored with a roster that did not apply to them"),
    }


def compare(labelled: pd.DataFrame, hcol: str, horizon: int) -> dict:
    """In-jurisdiction vs out, with an overlap-aware t on the difference.

    The paired quantity is the DIFFERENCE in mean excess return between the two
    groups within each period, so market conditions cancel, a raw comparison
    of two groups spanning different months would mostly measure when each
    group happened to trade.

    THE PAIRING PERIOD IS A NON-OVERLAPPING BLOCK, NOT A MONTH. Pairing by
    calendar month and running a one-sample t over the monthly differences
    treats 129 months as 129 independent draws, but at a 63-day horizon a
    position opened in January and one opened in February share ~2/3 of their
    holding window. That is precisely the overlap `block_test` exists to
    correct, and applying it to the two group means while leaving the headline
    difference on monthly pairing was inconsistent, the group t's were honest
    and the number the verdict actually quotes was not. Blocks are
    `horizon/21` months long, so each paired difference is an independent
    observation.

    `diff_t_month` retains the old monthly-paired figure for comparability with
    earlier write-ups. It is reported, never used as the bar.
    """
    from copytrade.rank.timescale import block_test

    if labelled.empty:
        return {"error": "no eligible trades"}
    ins = labelled[labelled["in_jurisdiction"]]
    out = labelled[~labelled["in_jurisdiction"]]

    def stats(g):
        s = g[hcol].dropna()
        if s.empty:
            return {"n": 0, "mean": None, "median": None, "hit": None, "t": None}
        t, p, nb = block_test(s, g.loc[s.index, "entry_date"], horizon)
        return {"n": int(len(s)), "mean": float(s.mean()),
                "median": float(s.median()), "hit": float((s > 0).mean()),
                "t": t, "p": p, "blocks": nb}

    a, b = stats(ins), stats(out)

    def _paired(period_months: int):
        """Mean/t/p/n of the in-minus-out difference, pairing within
        `period_months`-long non-overlapping windows."""
        d = _paired_differences(labelled, hcol, period_months)
        if d is None or len(d) < 3:
            return None, None, None, 0 if d is None else int(len(d))
        from scipy import stats as st
        tt, pp = st.ttest_1samp(d, 0.0)
        return float(d.mean()), float(tt), float(pp), int(len(d))

    block_months = max(1, int(round(horizon / 21)))
    diff_mean = diff_t = diff_p = None
    n_blocks = n_months = 0
    if a["n"] and b["n"]:
        diff_mean, diff_t, diff_p, n_blocks = _paired(block_months)
        _, t_month, p_month, n_months = _paired(1)
    else:
        t_month = p_month = None
    return {
        "in_jurisdiction": a, "out_of_jurisdiction": b,
        "diff_mean": diff_mean, "diff_t": diff_t, "diff_p": diff_p,
        "paired_blocks": n_blocks, "paired_block_months": block_months,
        # legacy monthly pairing: overlapping at any horizon past ~21d, kept
        # only so older figures remain reconcilable
        "diff_t_month": t_month, "diff_p_month": p_month,
        "paired_months": n_months,
    }


def _paired_differences(labelled: pd.DataFrame, hcol: str,
                        period_months: int) -> pd.Series | None:
    """In-minus-out mean difference per non-overlapping period, dropping
    periods where either group is absent (an unpaired period carries no
    information about the contrast)."""
    months = pd.to_datetime(labelled["entry_date"]).dt.to_period("M")
    origin = months.min()
    block = (months - origin).apply(lambda x: x.n) // period_months
    g = (labelled.assign(_b=block)
         .groupby(["_b", "in_jurisdiction"])[hcol].mean().unstack())
    if not {True, False} <= set(g.columns):
        return None
    return (g[True] - g[False]).dropna()
