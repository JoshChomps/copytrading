"""Canadian federal disclosures, Conflict of Interest and Ethics Commissioner.

READ THIS before expecting US-style signals. Canada has NO STOCK Act:

  * There is no transaction-level, timestamped, amount-bearing PTR. MPs file
    a Disclosure Summary (assets/liabilities > $10,000, income > $1,000) and
    must report a MATERIAL CHANGE, an acquisition or disposal exceeding
    $10,000: within 60 DAYS, to the CIEC public registry. That is a coarser,
    slower signal than a US PTR (which is ~30-45 days and carries the exact
    security, date range and amount band).
  * REPORTING PUBLIC OFFICE HOLDERS (ministers, parliamentary secretaries and
    senior staff) are BARRED from holding controlled assets (publicly traded
    securities): they must divest or blind-trust within 120 days. So the
    highest-profile Canadians have, by law, nothing to copy.
  * The registry is HTML/PDF per member, no bulk API. Parsing yields
    material-change *declarations*, not clean buys/sells with prices.

What this source does, honestly: scrape the public-registry material-change
declarations into the SAME schema (as instrument='ca_material_change',
amount band from the >$10k threshold, filing_date = declaration date,
transaction_date = stated change date when present). These records are
STORED and shown but are NOT fed into the priced backtest by default, they
lack the precision the event study needs, and pretending otherwise is exactly
the fake-signal trap the rest of this codebase avoids. Third-party trackers
(integrityindex.ca, media FOIs) are richer starting points than the raw
registry and can be pointed at via `registry_url`/a local JSON dump.

Network: ciec-ccie.parl.gc.ca is 403 from the dev sandbox; this is a
documented, best-effort implementation to validate off-sandbox.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import httpx

from copytrade.scrape.base import ScrapeResult

REGISTRY = "https://ciec-ccie.parl.gc.ca/en/public-registry"
CA_MATERIAL_THRESHOLD = 10_000.0


@dataclass
class CaDisclosure:
    member: str
    party: str | None
    security: str | None
    ticker: str | None
    change_type: str | None      # acquired | disposed | None
    change_date: str | None      # ISO, when stated
    declared_date: str           # ISO, the point-in-time anchor
    source: str = "ca_ciec"

    def to_trade_kwargs(self, filer_id: int) -> dict:
        side = {"acquired": "buy", "disposed": "sell"}.get((self.change_type or "").lower())
        return dict(
            filer_id=filer_id, source=self.source,
            source_uid=f"ca:{self.member}:{self.security}:{self.declared_date}:{self.change_type}",
            ticker=self.ticker, side=side, instrument="ca_material_change",
            transaction_date=self.change_date or self.declared_date,
            # material change reportable within 60 days; anchor at declaration
            filing_datetime=f"{self.declared_date} 17:00:00",
            amount_low=CA_MATERIAL_THRESHOLD, amount_high=None,
        )


def _iso(s: str | None) -> str | None:
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%B %d, %Y", "%b %d, %Y"):
        try:
            return datetime.strptime((s or "").strip(), fmt).strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            continue
    return None


def parse_ciec_json(records: list[dict]) -> list[CaDisclosure]:
    """Normalize a list of declaration dicts (from a scrape or a third-party
    export like integrityindex). Expected keys are lenient."""
    out = []
    for r in records:
        declared = _iso(r.get("declared_date") or r.get("date") or r.get("published"))
        if not declared:
            continue
        out.append(CaDisclosure(
            member=(r.get("member") or r.get("name") or "").strip().title(),
            party=r.get("party"),
            security=r.get("security") or r.get("asset"),
            ticker=(r.get("ticker") or "").upper() or None,
            change_type=r.get("change_type") or r.get("type"),
            change_date=_iso(r.get("change_date") or r.get("transaction_date")),
            declared_date=declared,
        ))
    return out


def store_ca_disclosures(conn, disclosures: list[CaDisclosure]) -> tuple[int, int]:
    from copytrade.store import db as store

    inserted = deduped = 0
    for d in disclosures:
        if not d.member:
            deduped += 1
            continue
        filer_id = store.upsert_filer(
            conn, d.member, "congress",
            {"country": "CA", "party": d.party, "chamber": "House of Commons"},
        )
        new = store.insert_trade(conn, **d.to_trade_kwargs(filer_id))
        inserted += int(new)
        deduped += int(not new)
    conn.commit()
    return inserted, deduped


async def scrape_material_changes(conn, settings, json_url: str = "") -> ScrapeResult:
    """Pull material-change declarations. Prefers a configured JSON feed
    (third-party tracker / local export); the raw HTML registry parser is a
    documented stub because the registry has no stable machine format."""
    res = ScrapeResult(source="ca_ciec")
    url = json_url or getattr(settings, "canada_ciec_json_url", "")
    if not url:
        return ScrapeResult(
            source="ca_ciec",
            note="no CA feed configured (canada_ciec_json_url). Canada has no "
                 "STOCK Act: see docs/SCRAPING.md §Canada; point this at a "
                 "third-party tracker export to enable.",
        )
    ua = {"User-Agent": getattr(settings, "edgar_user_agent", "copytrade-research")}
    async with httpx.AsyncClient(headers=ua, timeout=60, follow_redirects=True) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        payload = resp.json()
    records = payload if isinstance(payload, list) else payload.get("data", [])
    disclosures = parse_ciec_json(records)
    ins, dedup = store_ca_disclosures(conn, disclosures)
    res.inserted, res.deduped = ins, dedup
    res.note = (f"{len(disclosures)} CA material-change declarations "
                "(instrument='ca_material_change'; excluded from priced backtest)")
    return res
