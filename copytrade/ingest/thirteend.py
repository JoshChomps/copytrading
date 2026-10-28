"""SEC Schedule 13D (activist >5% stakes), an event-driven channel.

A 13D is filed within 5 business days of an investor crossing 5% of a company
WITH intent to influence (activism). The announcement is a documented
event-study anomaly (target pops on disclosure). Unlike copying Congress, the
signal is FAST and the event is discrete.

Source: EDGAR quarterly master.idx (pipe-delimited: CIK|Company|Form|Date|File).
A 13D is filed on the SUBJECT company's EDGAR record, so the index CIK is the
target; we map it to a ticker via SEC's company_tickers.json and ingest each
INITIAL 'SC 13D' (not '/A' amendments) as a filing-anchored buy of the target.
The activist's identity is inside the document, not the index, irrelevant for
the target's forward-return event study, which is all we need here.
"""

from __future__ import annotations

import json

MASTER_IDX = "https://www.sec.gov/Archives/edgar/full-index/{year}/QTR{q}/master.idx"
COMPANY_TICKERS = "https://www.sec.gov/files/company_tickers.json"


def parse_master_idx(text: str, forms=("SC 13D",)) -> list[dict]:
    """Pipe-delimited master.idx -> rows for the requested form types (exact
    match, so 'SC 13D' excludes 'SC 13D/A' amendments)."""
    want = set(forms)
    out = []
    for line in text.splitlines():
        parts = line.split("|")
        if len(parts) != 5:
            continue
        cik, company, form, date, fname = (p.strip() for p in parts)
        if form in want and cik.isdigit():
            out.append({"cik": int(cik), "company": company, "form": form,
                        "date": date, "file": fname})
    return out


def parse_company_tickers(text: str) -> dict[int, str]:
    """SEC company_tickers.json -> {cik:int -> TICKER}. First ticker per CIK."""
    data = json.loads(text)
    rows = data.values() if isinstance(data, dict) else data
    out: dict[int, str] = {}
    for r in rows:
        cik = r.get("cik_str")
        tkr = (r.get("ticker") or "").strip().upper()
        if cik is not None and tkr and int(cik) not in out:
            out[int(cik)] = tkr
    return out


async def refresh_13d(conn, settings, years: list[int]) -> dict:
    """Download quarterly master.idx for `years`, keep initial SC 13D filings,
    map subject CIK -> ticker, and ingest each as a filing-anchored buy under a
    single synthetic 'SC 13D activists' filer. Idempotent via source_uid."""
    import httpx

    from copytrade.store import db as store

    ua = {"User-Agent": getattr(settings, "edgar_user_agent", "copytrade-research")}
    inserted = deduped = skipped = seen = 0
    async with httpx.AsyncClient(headers=ua, timeout=120, follow_redirects=True) as client:
        ct = await client.get(COMPANY_TICKERS)
        ct.raise_for_status()
        cik2t = parse_company_tickers(ct.text)
        filer_id = store.upsert_filer(conn, "SC 13D activists", "fund", {"channel": "13d"})
        for year in years:
            for q in (1, 2, 3, 4):
                try:
                    r = await client.get(MASTER_IDX.format(year=year, q=q))
                    if r.status_code != 200:
                        continue
                    rows = parse_master_idx(r.text)
                except httpx.HTTPError:
                    continue
                for row in rows:
                    seen += 1
                    ticker = cik2t.get(row["cik"])
                    if not ticker or not row["date"]:
                        skipped += 1
                        continue
                    new = store.insert_trade(
                        conn, filer_id=filer_id, source="edgar_13d",
                        source_uid=f"{row['cik']}:{row['date']}:{row['file']}",
                        ticker=ticker, side="buy", instrument="stock",
                        transaction_date=row["date"],
                        filing_datetime=f"{row['date']} 17:00:00",
                    )
                    inserted += int(new)
                    deduped += int(not new)
        conn.commit()
    return {"filings_seen": seen, "inserted": inserted, "deduped": deduped,
            "skipped_no_ticker": skipped, "tickers_mapped": len(cik2t)}
