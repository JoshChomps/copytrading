"""SEC DERA 'Insider Transactions Data Sets', bulk Form 3/4/5 backfill.

https://www.sec.gov/data-research/sec-markets-data/insider-transactions-data-sets
publishes quarterly ZIPs of every ownership filing as TSVs. One download
replaces thousands of per-filing EDGAR requests, which makes it the right
tool for backfilling YEARS of insider history (the live poller stays on the
EDGAR Atom feed for minute-level acceptance timestamps).

Anchor honesty: the data sets carry FILING_DATE (day precision) but not the
acceptance datetime, so trades are anchored at 17:00 ET on the filing date
the same conservative end-of-day convention used for congress PTRs. The live
EDGAR path remains strictly better for recent data; this is for history.

Tables used (joined on ACCESSION_NUMBER):
  SUBMISSION.tsv      -> ISSUERTRADINGSYMBOL, FILING_DATE
  REPORTINGOWNER.tsv  -> RPTOWNERNAME (first owner listed wins on multi-owner)
  NONDERIV_TRANS.tsv  -> TRANS_DATE, TRANS_CODE, TRANS_SHARES, TRANS_PRICEPERSHARE

All parsing is header-driven (column positions are not assumed) and rows
missing an anchor/symbol/date are counted and skipped, never guessed.

Validation status: the TSV layout follows the published DERA spec, but
sec.gov is unreachable from this development sandbox, so run one quarter
with `--limit 100` and eyeball `orders-list`/`trades` before trusting a full
backfill. Cross-source dedup keeps the earlier anchor when the same trade
also arrives via the live EDGAR path.
"""

from __future__ import annotations

import csv
import io
import zipfile
from datetime import datetime
from pathlib import Path

# SEC has moved these files across several paths over the years (dera ->
# structureddata -> data-research). Try each known pattern; the first that
# returns a ZIP wins. All are {year}q{quarter}_form345.zip.
DERA_URLS = (
    "https://www.sec.gov/files/structureddata/data/insider-transactions-data-sets/{year}q{quarter}_form345.zip",
    "https://www.sec.gov/files/dera/data/insider-transactions-data-sets/{year}q{quarter}_form345.zip",
    "https://www.sec.gov/data-research/sec-markets-data/insider-transactions-data-sets/{year}q{quarter}_form345.zip",
)
DERA_URL = DERA_URLS[0]  # back-compat for callers/tests referencing a single URL

_DATE_FORMATS = ("%d-%b-%Y", "%Y-%m-%d", "%m/%d/%Y")  # 01-FEB-2024 is DERA's usual


def parse_dera_date(s: str | None) -> str | None:
    if not s:
        return None
    s = s.strip()
    for fmt in _DATE_FORMATS:
        try:
            dt = datetime.strptime(s, fmt)  # %b is case-insensitive
        except ValueError:
            continue
        # reject implausible years: a 2-digit-year string like "22-10-12"
        # parses as year 22 under %Y and would round-trip to a garbage date
        if 1990 <= dt.year <= datetime.now().year + 1:
            return dt.strftime("%Y-%m-%d")
    return None


def _num(s: str | None) -> float | None:
    if s is None or not s.strip():
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _read_tsv(zf: zipfile.ZipFile, name: str) -> csv.DictReader | None:
    for info in zf.infolist():
        if info.filename.upper().endswith(name.upper()):
            text = zf.read(info).decode("utf-8", errors="replace")
            return csv.DictReader(io.StringIO(text), delimiter="\t")
    return None


def ingest_zip(conn, zip_path: str | Path, limit: int = 0) -> dict:
    """Parse one quarterly form345 ZIP into filers/trades. Returns counts."""
    from copytrade.store import db as store

    with zipfile.ZipFile(zip_path) as zf:
        subs = _read_tsv(zf, "SUBMISSION.tsv")
        owners = _read_tsv(zf, "REPORTINGOWNER.tsv")
        trans = _read_tsv(zf, "NONDERIV_TRANS.tsv")
        if subs is None or trans is None:
            raise ValueError(
                f"{zip_path}: SUBMISSION.tsv/NONDERIV_TRANS.tsv not found, "
                "is this a DERA form345 archive?"
            )

        sub_by_acc: dict[str, dict] = {}
        for r in subs:
            acc = (r.get("ACCESSION_NUMBER") or "").strip()
            if acc:
                sub_by_acc[acc] = {
                    "symbol": (r.get("ISSUERTRADINGSYMBOL") or "").strip().upper(),
                    "filing_date": parse_dera_date(r.get("FILING_DATE")),
                    "issuer": (r.get("ISSUERNAME") or "").strip(),
                }

        owner_by_acc: dict[str, str] = {}
        if owners is not None:
            for r in owners:
                acc = (r.get("ACCESSION_NUMBER") or "").strip()
                name = (r.get("RPTOWNERNAME") or "").strip()
                if acc and name and acc not in owner_by_acc:  # first owner wins
                    owner_by_acc[acc] = name

        inserted = deduped = skipped = 0
        filer_ids: dict[str, int] = {}
        n = 0
        for r in trans:
            if limit and n >= limit:
                break
            n += 1
            acc = (r.get("ACCESSION_NUMBER") or "").strip()
            sub = sub_by_acc.get(acc)
            owner = owner_by_acc.get(acc)
            tx_date = parse_dera_date(r.get("TRANS_DATE"))
            code = (r.get("TRANS_CODE") or "").strip().upper()
            if not sub or not sub["symbol"] or not sub["filing_date"] or not owner or not tx_date:
                skipped += 1
                continue
            shares = _num(r.get("TRANS_SHARES"))
            price = _num(r.get("TRANS_PRICEPERSHARE"))
            side = "buy" if code == "P" else "sell" if code == "S" else None
            if owner not in filer_ids:
                filer_ids[owner] = store.upsert_filer(
                    conn, owner, "insider", {"issuer": sub["issuer"]} if sub["issuer"] else None
                )
            amount = (shares or 0) * (price or 0) or None
            new = store.insert_trade(
                conn,
                filer_id=filer_ids[owner],
                source="dera_form345",
                source_uid=f"{acc}:{(r.get('NONDERIV_TRANS_SK') or n)}",
                ticker=sub["symbol"],
                side=side,
                tx_code=code,
                instrument="stock",
                transaction_date=tx_date,
                filing_datetime=f"{sub['filing_date']} 17:00:00",
                shares=shares,
                price=price,
                amount_low=amount,
                amount_high=amount,
            )
            inserted += int(new)
            deduped += int(not new)
        conn.commit()
    return {"transactions_seen": n, "inserted": inserted,
            "deduped": deduped, "skipped_unusable": skipped}


async def download_quarter(year: int, quarter: int, dest_dir: str | Path,
                           user_agent: str) -> Path:
    """Fetch one quarterly archive (SEC fair-access UA required)."""
    import httpx

    dest = Path(dest_dir) / f"{year}q{quarter}_form345.zip"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return dest
    last_err: Exception | None = None
    async with httpx.AsyncClient(
        headers={"User-Agent": user_agent}, timeout=300, follow_redirects=True
    ) as client:
        for tmpl in DERA_URLS:
            url = tmpl.format(year=year, quarter=quarter)
            try:
                resp = await client.get(url)
                if resp.status_code == 200 and resp.content[:2] == b"PK":  # a real ZIP
                    dest.write_bytes(resp.content)
                    return dest
                last_err = ValueError(f"{url} -> HTTP {resp.status_code}")
            except httpx.HTTPError as e:
                last_err = e
    raise RuntimeError(f"could not fetch {year}q{quarter} form345 from any known SEC path: {last_err}")
