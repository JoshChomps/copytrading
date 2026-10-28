"""13F-HR ingestion: quarterly information tables -> holdings -> QoQ deltas.

13F caveats baked in (plan §7): longs only, no timing, window-dressing risk.
The cloning report restricts to low-turnover, concentrated managers. The
point-in-time anchor for a quarter's holdings is the filing acceptance time
(up to 45 days after quarter end), never the period date.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass

import httpx


@dataclass
class Holding:
    cusip: str
    value: float      # dollars (modern filings report USD, older ones $000s)
    shares: float
    name: str | None = None
    put_call: str | None = None   # 'Put' | 'Call' | None, the leveraged bets

    @property
    def instrument(self) -> str:
        pc = (self.put_call or "").lower()
        if pc == "put":
            return "put_option"
        if pc == "call":
            return "call_option"
        return "stock"


def _strip_ns(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def parse_13f_infotable(xml_text: str) -> list[Holding]:
    root = ET.fromstring(xml_text)
    holdings: list[Holding] = []
    for node in root.iter():
        if _strip_ns(node.tag) != "infoTable":
            continue
        fields = {_strip_ns(c.tag): c for c in node}
        cusip = (fields.get("cusip").text or "").strip() if fields.get("cusip") is not None else ""
        value_node = fields.get("value")
        shrs = None
        sap = fields.get("shrsOrPrnAmt")
        if sap is not None:
            for c in sap:
                if _strip_ns(c.tag) == "sshPrnamt" and c.text:
                    shrs = float(c.text)
        name_node = fields.get("nameOfIssuer")
        pc_node = fields.get("putCall")
        put_call = pc_node.text.strip() if pc_node is not None and pc_node.text else None
        if cusip and value_node is not None and value_node.text:
            holdings.append(
                Holding(
                    cusip=cusip.upper(),
                    value=float(value_node.text),
                    shares=shrs or 0.0,
                    name=name_node.text.strip() if name_node is not None and name_node.text else None,
                    put_call=put_call,
                )
            )
    return holdings


def store_13f(conn, filer_id: int, period: str, holdings: list[Holding]) -> int:
    rows = [(filer_id, period, h.cusip, h.put_call or "", h.value, h.shares) for h in holdings]
    conn.executemany(
        "INSERT INTO holdings_13f(filer_id, period, cusip, put_call, value, shares) "
        "VALUES (?,?,?,?,?,?) ON CONFLICT(filer_id, period, cusip, put_call) "
        "DO UPDATE SET value=excluded.value, shares=excluded.shares",
        rows,
    )
    conn.commit()
    return len(rows)


def quarterly_deltas(conn, filer_id: int, period_prev: str, period_curr: str,
                     filing_datetime: str) -> tuple[int, int]:
    """Turn holdings changes between two quarters into buy/sell 'trades'.

    Point-in-time anchor = the CURRENT quarter filing's acceptance time.
    Returns (inserted, skipped_unmapped_cusip).
    """
    from copytrade.store import db as store

    def load(period):
        # key by (cusip, put_call) so a stock delta and an option delta on the
        # same name stay distinct: the whole point of tracking fund options
        return {
            (r["cusip"], r["put_call"] or ""): (r["shares"], r["value"])
            for r in conn.execute(
                "SELECT cusip, put_call, shares, value FROM holdings_13f "
                "WHERE filer_id=? AND period=?",
                (filer_id, period),
            )
        }

    prev, curr = load(period_prev), load(period_curr)
    cusip_map = {r["cusip"]: r["ticker"] for r in conn.execute("SELECT cusip, ticker FROM cusip_map")}

    inserted = unmapped = 0
    for key in set(prev) | set(curr):
        cusip, put_call = key
        d_shares = (curr.get(key, (0, 0))[0] or 0) - (prev.get(key, (0, 0))[0] or 0)
        if d_shares == 0:
            continue
        ticker = cusip_map.get(cusip)
        if not ticker:
            unmapped += 1
            continue
        value = abs(d_shares) / max(curr.get(key, prev.get(key))[0] or 1, 1) * (
            curr.get(key, prev.get(key))[1] or 0
        )
        instrument = {"put": "put_option", "call": "call_option"}.get(
            put_call.lower(), "stock")
        new = store.insert_trade(
            conn,
            filer_id=filer_id,
            source="edgar_13f",
            source_uid=f"13f:{filer_id}:{period_curr}:{cusip}:{put_call}",
            ticker=ticker,
            cusip=cusip,
            side="buy" if d_shares > 0 else "sell",
            instrument=instrument,
            transaction_date=period_curr,   # quarter end (timing unknown)
            filing_datetime=filing_datetime,
            shares=abs(d_shares),
            amount_low=value or None,
            amount_high=value or None,
        )
        inserted += int(new)
    conn.commit()
    return inserted, unmapped


async def backfill_13f_cik(conn, cik: str, quarters: int, user_agent: str,
                           rps: float = 8.0, raw_dir: str = "data/raw",
                           openfigi_api_key: str = "") -> dict:
    """Load a manager's recent 13F-HR filings and derive QoQ delta trades.

    Discovery via the data.sec.gov submissions JSON (which carries the
    acceptanceDateTime used as the point-in-time anchor). Amendments
    (13F-HR/A) are skipped: restatement handling is out of scope and
    silently merging them would corrupt the deltas.
    """
    from copytrade.ingest.edgar import EdgarClient, FilingRef
    from copytrade.store import db as store

    client = EdgarClient(user_agent, rps, raw_dir)
    try:
        subs = await client.fetch_submissions(cik)
        name = subs.get("name") or f"CIK {cik}"
        recent = subs.get("filings", {}).get("recent", {})
        rows = [
            {
                "accession": acc,
                "acceptance": (recent["acceptanceDateTime"][i] or "")
                              .replace("T", " ").rstrip("Z")[:19],
                "period": recent["reportDate"][i],
            }
            for i, (form, acc) in enumerate(zip(recent.get("form", []),
                                                recent.get("accessionNumber", [])))
            if form == "13F-HR"
        ]
        rows.sort(key=lambda r: r["period"])
        rows = rows[-quarters:]
        if len(rows) < 2:
            return {"filer": name, "filings_found": len(rows),
                    "error": "need >= 2 quarters for deltas"}

        filer_id = store.upsert_filer(conn, name, "fund", {"cik": cik})
        loaded = []
        all_cusips: set[str] = set()
        for r in rows:
            xml = await client.fetch_13f_xml(FilingRef(accession=r["accession"], cik=cik))
            if xml is None:
                continue
            holdings = parse_13f_infotable(xml)
            store_13f(conn, filer_id, r["period"], holdings)
            all_cusips.update(h.cusip for h in holdings)
            loaded.append(r)

        unmapped = [c for c in sorted(all_cusips) if conn.execute(
            "SELECT 1 FROM cusip_map WHERE cusip=?", (c,)).fetchone() is None]
        if unmapped:
            mapping = await openfigi_cusip_to_ticker(unmapped, openfigi_api_key)
            conn.executemany(
                "INSERT OR REPLACE INTO cusip_map(cusip, ticker) VALUES (?,?)",
                list(mapping.items()),
            )
            conn.commit()

        inserted = skipped = 0
        for prev, curr in zip(loaded, loaded[1:]):
            i, u = quarterly_deltas(conn, filer_id, prev["period"], curr["period"],
                                    curr["acceptance"] or f"{curr['period']} 17:00:00")
            inserted += i
            skipped += u
        return {"filer": name, "filings_loaded": len(loaded),
                "delta_trades_inserted": inserted,
                "skipped_unmapped_cusip": skipped,
                "cusips_newly_mapped": len(unmapped)}
    finally:
        await client.aclose()


async def openfigi_cusip_to_ticker(cusips: list[str], api_key: str = "") -> dict[str, str]:
    """Free OpenFIGI mapping fallback for CUSIP->ticker (plan §2.1)."""
    url = "https://api.openfigi.com/v3/mapping"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-OPENFIGI-APIKEY"] = api_key
    out: dict[str, str] = {}
    async with httpx.AsyncClient(timeout=30) as client:
        for i in range(0, len(cusips), 10):  # free tier: 10 jobs/request
            batch = cusips[i : i + 10]
            resp = await client.post(
                url, headers=headers,
                json=[{"idType": "ID_CUSIP", "idValue": c} for c in batch],
            )
            if resp.status_code != 200:
                break
            for cusip, result in zip(batch, resp.json()):
                data = result.get("data") or []
                if data and data[0].get("ticker"):
                    out[cusip] = data[0]["ticker"]
    return out
