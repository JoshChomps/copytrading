"""Golden-file tests for every parser (plan §1: real captured payload shapes)."""

import asyncio
import json

from copytrade.ingest.congress import (
    JsonFileCongress,
    _instrument,
    parse_finnhub_payload,
    store_congress_trades,
)
from copytrade.ingest.edgar import parse_atom_feed, parse_form_idx
from copytrade.ingest.form4 import parse_form4, store_form4
from copytrade.ingest.thirteenf import parse_13f_infotable
from tests.conftest import FIXTURES


def test_form4_parser_golden():
    filing = parse_form4((FIXTURES / "form4_sample.xml").read_text())
    assert filing.issuer_symbol == "AAPL"
    assert filing.issuer_cik == "0000320193"
    assert filing.owner_name == "DOE JANE"
    assert filing.relationship["officer"] is True
    assert filing.relationship["officer_title"] == "Chief Financial Officer"
    assert len(filing.transactions) == 3

    p, s, m = filing.transactions
    assert (p.code, p.side, p.shares, p.price, p.date) == ("P", "buy", 1000.0, 182.50, "2024-05-01")
    assert (s.code, s.side, s.shares) == ("S", "sell", 250.0)
    # M (option exercise) is stored but never sided -> excluded from backtests
    assert (m.code, m.side) == ("M", None)


def test_form4_store_idempotent(conn):
    filing = parse_form4((FIXTURES / "form4_sample.xml").read_text())
    n1 = store_form4(conn, filing, "0001214128-24-000123", "2024-05-01 16:35:12")
    n2 = store_form4(conn, filing, "0001214128-24-000123", "2024-05-01 16:35:12")
    assert n1 == 3 and n2 == 0  # rerun inserts nothing
    sided = conn.execute("SELECT COUNT(*) FROM trades WHERE side IS NOT NULL").fetchone()[0]
    assert sided == 2


def test_congress_parser_golden():
    payload = json.loads((FIXTURES / "congress_sample.json").read_text())
    trades, skipped = parse_finnhub_payload(payload)
    assert skipped == 1  # record without filing date is counted, never backdated
    assert len(trades) == 2
    buy = next(t for t in trades if t.side == "buy")
    assert (buy.ticker, buy.amount_low, buy.amount_high) == ("AAPL", 15001, 50000)
    assert buy.filing_date == "2023-03-15" and buy.transaction_date == "2023-02-10"
    opt = next(t for t in trades if t.side == "sell")
    # "Apple Inc Call Options" -> call_option (direction preserved for the
    # derivatives study); still excluded from the stock-only priced backtest
    assert opt.instrument == "call_option"


def test_congress_instrument_preserves_option_direction():
    # call vs put must survive so the derivatives study signs the underlying
    # correctly (long call bullish, long put bearish), same taxonomy as House
    assert _instrument("Apple Inc") == "stock"
    assert _instrument("NVIDIA Corp - Call Options") == "call_option"
    assert _instrument("SPY Put Options") == "put_option"
    assert _instrument("Some Warrant option") == "option"
    assert _instrument(None) == "stock"


def test_congress_store_and_dedup(conn):
    src = JsonFileCongress(FIXTURES / "congress_sample.json")
    trades, _ = asyncio.run(src.fetch())
    ins1, _ = store_congress_trades(conn, trades)
    ins2, dup2 = store_congress_trades(conn, trades)
    assert ins1 == 2 and ins2 == 0 and dup2 == 2
    row = conn.execute("SELECT filing_datetime FROM trades WHERE side='buy'").fetchone()
    assert row[0] == "2023-03-15 17:00:00"  # date-only PTR anchored to end of day


def test_form_idx_parser():
    refs = parse_form_idx((FIXTURES / "form_idx_sample.idx").read_text(), form_type="4")
    assert [r.accession for r in refs] == ["0001214128-24-000123", "0000777777-24-000456"]
    assert refs[0].cik == "1214128"
    assert refs[0].directory_url().endswith("/1214128/000121412824000123")


def test_atom_feed_parser():
    refs = parse_atom_feed((FIXTURES / "atom_sample.xml").read_text())
    assert len(refs) == 2  # duplicate accession collapsed
    assert refs[0].accession == "0001214128-24-000123"


def test_13f_infotable_parser():
    holdings = parse_13f_infotable((FIXTURES / "infotable_13f.xml").read_text())
    assert len(holdings) == 2
    aapl = next(h for h in holdings if h.cusip == "037833100")
    assert aapl.shares == 5_000_000 and aapl.value == 915_555_000


def test_load_universe_skips_comments(tmp_path):
    """Regression: the '#' header of config/universe.txt was sent to Finnhub
    as a ticker and its 403 sank a CI backtest run."""
    from copytrade.ingest.congress import load_universe

    f = tmp_path / "universe.txt"
    f.write_text("# comment header\n\nAAPL\n  # indented comment\nmsft \n")
    assert load_universe(f) == ["AAPL", "MSFT"]
    assert load_universe(tmp_path / "missing.txt") == []


def test_finnhub_per_symbol_errors_do_not_abort():
    import asyncio

    import httpx

    from copytrade.ingest.congress import FinnhubCongress

    def handler(request: httpx.Request) -> httpx.Response:
        sym = request.url.params["symbol"]
        if sym == "BAD":
            return httpx.Response(403)
        return httpx.Response(200, json={"symbol": sym, "data": [{
            "name": "Sen X", "symbol": sym, "transactionDate": "2026-01-05",
            "filingDate": "2026-01-20", "transactionType": "Purchase",
            "amountFrom": 1001, "amountTo": 15000,
        }]})

    src = FinnhubCongress("key", rps=1000, transport=httpx.MockTransport(handler))
    trades, skipped = asyncio.run(src.fetch(["BAD", "AAPL"], "2026-01-01", "2026-02-01"))
    assert len(trades) == 1 and trades[0].ticker == "AAPL"
    assert src.errors == {"BAD": "http 403"}
    assert not src.all_failed_uniformly  # AAPL succeeded


def test_finnhub_uniform_failure_detected():
    import asyncio

    import httpx

    from copytrade.ingest.congress import FinnhubCongress

    src = FinnhubCongress("key", rps=1000,
                          transport=httpx.MockTransport(lambda r: httpx.Response(403)))
    trades, _ = asyncio.run(src.fetch(["A", "B"], "2026-01-01", "2026-02-01"))
    assert trades == [] and src.all_failed_uniformly
