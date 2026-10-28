"""Senate Stock Watcher parser tests against REAL fixture files (copied
verbatim from the data repo, no synthetic records)."""

import json

from copytrade.ingest.congress import store_congress_trades
from copytrade.ingest.senate_watcher import (
    clean_ticker,
    parse_amount,
    parse_report_payload,
)
from tests.conftest import FIXTURES

SW = FIXTURES / "senate_watcher"


def load(name):
    return json.loads((SW / name).read_text())


def test_clean_ticker_variants():
    assert clean_ticker('<a href="https://finance.yahoo.com/q?s=BA" target="_blank">BA</a>') == "BA"
    assert clean_ticker("AAPL") == "AAPL"
    assert clean_ticker("--") is None
    assert clean_ticker("") is None
    assert clean_ticker(None) is None
    assert clean_ticker('<a href="...">BRK.B</a>') == "BRK.B"


def test_parse_amount_all_real_buckets():
    # every bucket string observed in the real dataset
    assert parse_amount("$1,001 - $15,000") == (1001, 15000)
    assert parse_amount("$50,001 - $100,000") == (50001, 100000)
    assert parse_amount("$25,000,001 - $50,000,000") == (25000001, 50000000)
    assert parse_amount("Over $50,000,000") == (50000000, None)
    assert parse_amount(None) == (None, None)
    assert parse_amount("--") == (None, None)


def test_real_report_with_transactions():
    trades, skipped = parse_report_payload(load("transaction_report_for_12_31_2020.json"))
    # Pat Roberts' PTR: 17 rows, all Stock, tickers HTML-wrapped
    assert len(trades) + skipped == 17
    assert all(t.filing_date == "2020-12-31" for t in trades)
    assert any(t.ticker == "BA" and t.side == "buy" for t in trades)
    assert all(t.source == "senate_watcher" for t in trades)
    assert all(t.transaction_date < t.filing_date for t in trades)  # lag exists


def test_real_report_empty_transactions_and_paper_filing():
    # 10/08/2019 file: paper filing with transactions: [] plus a '--' ticker row
    trades, skipped = parse_report_payload(load("transaction_report_for_10_08_2019.json"))
    assert all(t.ticker for t in trades)  # no empty tickers slip through
    assert skipped >= 1  # the '--' non-ticker asset was skipped, counted


def test_real_report_over_50m_bucket():
    trades, skipped = parse_report_payload(load("transaction_report_for_10_26_2015.json"))
    # Inhofe 'Over $50,000,000' row has ticker '--' -> unusable, skipped
    assert skipped >= 1
    assert all(t.amount_low is not None for t in trades)


def test_ingest_and_amendment_dedup(conn):
    trades, _ = parse_report_payload(load("transaction_report_for_12_31_2020.json"))
    ins1, dup1 = store_congress_trades(conn, trades)
    assert ins1 == len(trades)
    # amended PTR re-reports the identical rows a week later: same uid -> dedup
    re_reported, _ = parse_report_payload(load("transaction_report_for_12_31_2020.json"))
    for t in re_reported:
        t.filing_date = "2021-01-07"
    ins2, dup2 = store_congress_trades(conn, re_reported)
    assert ins2 == 0 and dup2 == len(re_reported)
    # earliest receipt was kept
    row = conn.execute(
        "SELECT MIN(filing_datetime), MAX(filing_datetime) FROM trades"
    ).fetchone()
    assert row[0].startswith("2020-12-31") and row[1].startswith("2020-12-31")


def test_same_ptr_duplicate_rows_preserved():
    # two identical rows within ONE filing must both be stored (occurrence
    # counter): a real double purchase, not an amendment
    entry = load("transaction_report_for_12_31_2020.json")[0]
    tx = [t for t in entry["transactions"] if "BA" in t["ticker"]][0]
    entry2 = {**entry, "transactions": [tx, dict(tx)]}
    trades, _ = parse_report_payload([entry2])
    assert len(trades) == 2
    assert trades[0].source_uid != trades[1].source_uid


def test_clean_ticker_rejects_option_symbols_and_raw_html():
    """Regression: an HTML-wrapped OPTION symbol (SPY160219P00180000) slipped
    through as raw markup and reached the price fetcher in CI run #4."""
    assert clean_ticker(
        '<a href="https://finance.yahoo.com/q?s=SPY160219P00180000" target="_blank">SPY160219P00180000</a>'
    ) is None
    assert clean_ticker("<div>unclosed") is None
    assert clean_ticker("SPY160219P00180000") is None   # bare option symbol
    assert clean_ticker("BA") == "BA"                   # sanity: real tickers survive
