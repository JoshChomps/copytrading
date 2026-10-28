"""News parsers + DERA bulk ingester.

Fixtures here are SCHEMA-SHAPED (built to the providers' documented response
formats) because finnhub.io / gdeltproject.org / sec.gov are unreachable from
the dev sandbox: they validate parsing logic, not provider availability.
Validate against one live pull before trusting a production backfill
(README data-sources table says the same).
"""

import io
import zipfile

from copytrade.ingest.form345_bulk import ingest_zip, parse_dera_date
from copytrade.ingest.news import (
    NewsItem,
    latest_news,
    parse_finnhub_news,
    parse_gdelt_articles,
    store_news,
)


def test_parse_finnhub_news_shape():
    payload = [
        {"datetime": 1704222000, "headline": "Boeing wins order", "url": "https://x/1",
         "source": "Reuters"},
        {"datetime": None, "headline": "bad row", "url": "https://x/2"},  # skipped
    ]
    items = parse_finnhub_news(payload, "ba")
    assert len(items) == 1
    it = items[0]
    assert it.ticker == "BA" and it.provider == "finnhub"
    assert it.published_at.startswith("2024-01-02")


def test_parse_gdelt_articles_shape():
    payload = {"articles": [
        {"url": "https://y/1", "title": "Apple ships record units",
         "seendate": "20240102T130500Z", "domain": "apnews.com"},
        {"url": "https://y/2", "title": "", "seendate": "20240102T130500Z"},  # skipped
        {"url": "https://y/3", "title": "bad date", "seendate": "nope"},      # skipped
    ]}
    items = parse_gdelt_articles(payload, "AAPL")
    assert len(items) == 1
    assert items[0].published_at == "2024-01-02 13:05:00"
    assert items[0].provider == "gdelt"


def test_store_news_dedup(conn):
    it = NewsItem(ticker="BA", headline="h", url="https://x/1",
                  published_at="2024-01-02 00:00:00", provider="finnhub")
    assert store_news(conn, [it]) == (1, 0)
    assert store_news(conn, [it]) == (0, 1)
    heads = latest_news(conn, "ba", 5)
    assert len(heads) == 1 and heads[0]["headline"] == "h"


def test_parse_dera_dates():
    assert parse_dera_date("01-FEB-2024") == "2024-02-01"
    assert parse_dera_date("2024-02-01") == "2024-02-01"
    assert parse_dera_date("02/01/2024") == "2024-02-01"
    assert parse_dera_date("") is None
    assert parse_dera_date("garbage") is None


def _dera_zip() -> io.BytesIO:
    sub = ("ACCESSION_NUMBER\tFILING_DATE\tPERIOD_OF_REPORT\tISSUERNAME\tISSUERTRADINGSYMBOL\n"
           "0001-24-000001\t05-FEB-2024\t01-FEB-2024\tACME CORP\tACME\n"
           "0001-24-000002\t06-FEB-2024\t02-FEB-2024\tNOSYM INC\t\n")
    own = ("ACCESSION_NUMBER\tRPTOWNERCIK\tRPTOWNERNAME\n"
           "0001-24-000001\t123\tDOE JANE\n"
           "0001-24-000002\t456\tROE RICHARD\n")
    trans = ("ACCESSION_NUMBER\tNONDERIV_TRANS_SK\tTRANS_DATE\tTRANS_CODE\tTRANS_SHARES\tTRANS_PRICEPERSHARE\n"
             "0001-24-000001\t1\t01-FEB-2024\tP\t100\t50.5\n"
             "0001-24-000001\t2\t01-FEB-2024\tS\t40\t51.0\n"
             "0001-24-000001\t3\t01-FEB-2024\tM\t10\t0\n"        # option exercise: stored unsided
             "0001-24-000002\t4\t02-FEB-2024\tP\t5\t1.0\n")      # no symbol: skipped
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("SUBMISSION.tsv", sub)
        zf.writestr("REPORTINGOWNER.tsv", own)
        zf.writestr("NONDERIV_TRANS.tsv", trans)
    buf.seek(0)
    return buf


def test_dera_ingest(conn, tmp_path):
    p = tmp_path / "2024q1_form345.zip"
    p.write_bytes(_dera_zip().read())
    res = ingest_zip(conn, p)
    assert res == {"transactions_seen": 4, "inserted": 3, "deduped": 0,
                   "skipped_unusable": 1}
    rows = conn.execute(
        "SELECT ticker, side, tx_code, filing_datetime FROM trades ORDER BY source_uid"
    ).fetchall()
    assert [tuple(r)[:3] for r in rows] == [
        ("ACME", "buy", "P"), ("ACME", "sell", "S"), ("ACME", None, "M"),
    ]
    assert all(r["filing_datetime"] == "2024-02-05 17:00:00" for r in rows)
    # idempotent re-run
    assert ingest_zip(conn, p)["inserted"] == 0
