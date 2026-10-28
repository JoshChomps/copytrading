"""Scraping engine: pure parsers (spec-accurate fixtures, the live hosts are
proxy-blocked) + scheduler mechanics (dedup, backoff, heartbeat) with fakes."""

import asyncio

import pytest

from copytrade.ingest.canada_ciec import parse_ciec_json, store_ca_disclosures
from copytrade.ingest.house_clerk import parse_fd_index, parse_ptr_pdf_text
from copytrade.ingest.senate_efd import parse_efd_report_html
from copytrade.scrape.base import BaseSource, ScrapeResult
from copytrade.scrape.registry import build_sources
from copytrade.scrape.scheduler import scrape_once
from copytrade.store import db as store


# ---- House Clerk FD.xml index (documented schema) ------------------------

FD_XML = """<?xml version="1.0"?>
<FinancialDisclosure>
  <Member><Prefix>Hon.</Prefix><Last>Pelosi</Last><First>Nancy</First>
    <FilingType>P</FilingType><StateDst>CA11</StateDst><Year>2021</Year>
    <FilingDate>2/1/2021</FilingDate><DocID>20200001</DocID></Member>
  <Member><Last>Green</Last><First>Mark</First>
    <FilingType>O</FilingType><StateDst>TN07</StateDst><Year>2021</Year>
    <FilingDate>5/15/2021</FilingDate><DocID>20200002</DocID></Member>
  <Member><Last>NoDoc</Last><First>Skip</First><FilingType>P</FilingType></Member>
</FinancialDisclosure>"""


def test_parse_fd_index():
    filings = parse_fd_index(FD_XML)
    assert len(filings) == 2                       # the DocID-less row is dropped
    p = filings[0]
    assert p.doc_id == "20200001" and p.is_ptr and p.filing_date == "2021-02-01"
    assert p.name == "Nancy Pelosi"
    assert filings[1].is_ptr is False              # FilingType O is not a PTR


def test_parse_ptr_pdf_text():
    text = (
        "Periodic Transaction Report\n"
        "SP  Apple Inc. (AAPL) [ST] P 01/05/2021 01/20/2021 $1,001 - $15,000\n"
        "JT  Boeing Co (BA) [ST] Sale 02/10/2021 02/20/2021 $15,001 - $50,000\n"
        "Some Fund with no ticker Purchase 03/01/2021 $1,001 - $15,000\n"
    )
    trades, skipped = parse_ptr_pdf_text(text, "Nancy Pelosi", "2021-02-01")
    assert len(trades) == 2
    assert trades[0].ticker == "AAPL" and trades[0].side == "buy"
    assert trades[1].ticker == "BA" and trades[1].side == "sell"
    assert all(t.filing_date == "2021-02-01" and t.chamber == "House" for t in trades)


def test_ptr_scanned_pdf_yields_nothing_not_garbage():
    trades, skipped = parse_ptr_pdf_text("", "X", "2021-01-01")
    assert trades == [] and skipped == 0


# ---- Senate eFD electronic HTML table ------------------------------------

def test_parse_efd_report_html():
    html = """<table><tr><td>1</td><td>01/05/2021</td><td>Self</td>
      <td>AAPL</td><td>Apple Inc</td><td>Purchase</td><td>$1,001 - $15,000</td></tr>
      <tr><td>2</td><td>01/06/2021</td><td>Spouse</td>
      <td>--</td><td>Some LLC</td><td>Purchase</td><td>$1,001 - $15,000</td></tr></table>"""
    trades, skipped = parse_efd_report_html(html, "Jon Ossoff", "2021-02-01")
    assert len(trades) == 1 and trades[0].ticker == "AAPL" and trades[0].side == "buy"
    assert skipped == 1                             # the '--' ticker row


# ---- Canada CIEC ----------------------------------------------------------

def test_parse_and_store_ca_disclosures(conn):
    recs = [
        {"member": "jane doe", "party": "Liberal", "security": "Suncor",
         "ticker": "SU", "type": "acquired", "change_date": "2024-01-10",
         "declared_date": "2024-02-20"},
        {"member": "no date", "declared_date": None},   # dropped
    ]
    ds = parse_ciec_json(recs)
    assert len(ds) == 1 and ds[0].ticker == "SU"
    ins, dedup = store_ca_disclosures(conn, ds)
    assert ins == 1
    row = conn.execute("SELECT instrument, side, filing_datetime FROM trades").fetchone()
    assert row["instrument"] == "ca_material_change"      # excluded from backtest
    assert row["side"] == "buy"
    assert row["filing_datetime"].startswith("2024-02-20")
    # re-store is idempotent
    assert store_ca_disclosures(conn, ds)[0] == 0


def test_ca_records_excluded_from_backtest_frame(conn):
    store_ca_disclosures(conn, parse_ciec_json([
        {"member": "x", "ticker": "SU", "type": "acquired", "declared_date": "2024-02-20"}]))
    stock = store.trades_frame(conn, instruments=["stock"])
    assert stock.empty                                    # ca_material_change filtered out


# ---- scheduler mechanics --------------------------------------------------

class _FakeSource(BaseSource):
    name = "fake"
    kind = "congress"
    cadence_secs = 1
    authoritative = True

    def __init__(self, boom=False):
        self.boom = boom
        self.calls = 0

    async def run_once(self, conn, settings):
        self.calls += 1
        if self.boom:
            raise RuntimeError("kaboom")
        return ScrapeResult(source=self.name, inserted=2)


def test_scrape_once_records_heartbeat_and_counts(conn):
    class S:
        webhook_url = ""
        webhook_format = "discord"
        reports_dir = "reports"
    src = _FakeSource()
    results = asyncio.run(scrape_once(conn, S(), sources=[src]))
    assert results[0].inserted == 2 and src.calls == 1
    hb = store.get_state(conn, "scrape:fake:heartbeat")
    assert hb and '"ok": true' in hb


def test_scrape_once_isolates_source_failure(conn):
    class S:
        webhook_url = ""
        webhook_format = "discord"
        reports_dir = "reports"
    results = asyncio.run(scrape_once(conn, S(), sources=[_FakeSource(boom=True)]))
    assert results[0].error and "kaboom" in results[0].error
    hb = store.get_state(conn, "scrape:fake:heartbeat")
    assert '"ok": false' in hb


def test_registry_auto_skips_finnhub_without_key():
    class S:
        finnhub_api_key = ""
        scrape_sources = ["auto"]
    names = [s.name for s in build_sources(S())]
    assert "edgar_form4" in names and "house_clerk" in names
    assert "finnhub_congress" not in names        # prereq (key) not met

    class S2(S):
        finnhub_api_key = "k"
    assert "finnhub_congress" in [s.name for s in build_sources(S2())]


def test_scrape_result_add():
    a = ScrapeResult("x", inserted=1, deduped=2, alertable=[{"a": 1}])
    b = ScrapeResult("x", inserted=3, skipped=1, alertable=[{"b": 2}])
    c = a + b
    assert c.inserted == 4 and c.deduped == 2 and c.skipped == 1
    assert len(c.alertable) == 2


def test_house_ptr_detects_options():
    from copytrade.ingest.house_clerk import house_instrument, parse_ptr_pdf_text

    assert house_instrument(" [ST] ") == "stock"
    assert house_instrument("NVIDIA Corp - Call options [OP]") == "call_option"
    assert house_instrument("Put options on SPY [OP]") == "put_option"
    assert house_instrument("Apple Inc option [OP]") == "option"

    text = (
        "P1 NVIDIA Corporation - Call (NVDA) [OP] P 06/01/2024 06/15/2024 $250,001 - $500,000\n"
        "SP Apple Inc. (AAPL) [ST] P 01/05/2024 01/20/2024 $1,001 - $15,000\n"
    )
    trades, _ = parse_ptr_pdf_text(text, "Nancy Pelosi", "2024-06-20")
    by_ticker = {t.ticker: t.instrument for t in trades}
    assert by_ticker["NVDA"] == "call_option"   # the leveraged bet is now captured
    assert by_ticker["AAPL"] == "stock"


# ---- House PTR: the block-structure regression ---------------------------
#
# The parser used to anchor on the ticker and scan FORWARD for the type code.
# In the modern Clerk layout the asset name wraps, so the ticker sits on the
# line BELOW the type and dates, and forward of a ticker the next standalone
# capital letter is the `S` of the "Filing Status" row. Every wrapped row was
# therefore read as a Sale, with the following row's date and amount attached.
#
# Measured damage: House sells ran ~46% of trades in 2015 and 92% in 2025-26
# while the Senate feed held ~50% throughout. Verified against Clerk doc
# 20030803 (Cleo Fields, filed 2025-08-06), which contains 55 Purchases and one
# Sale and which the old parser reported as 1 buy and 55 sells.

_MODERN_BLOCK = (
    "ID Owner Asset Transaction Date Notification Amount Cap.\n"
    "Type Date Gains >\n"
    "$200?\n"
    "Taiwan Semiconductor P 07/17/2025 07/17/2025 $100,001 -\n"
    "Manufacturing Company Ltd. (TSM) [ST] $250,000\n"
    "F      S     : New\n"
    "S          O : Morgan Stanley - E*TRADE #2\n"
    "Arm Holdings plc - American P 07/16/2025 07/16/2025 $50,001 -\n"
    "Depositary Shares (ARM) [ST] $100,000\n"
    "F      S     : New\n"
    "S          O : Morgan Stanley - E*TRADE #2\n"
)


def test_wrapped_asset_name_keeps_its_own_side_date_and_amount():
    """The exact row that was stored as `sell 2025-07-10 $15,001+`."""
    trades, _ = parse_ptr_pdf_text(_MODERN_BLOCK, "Cleo Fields", "2025-08-06")
    by = {t.ticker: t for t in trades}
    assert set(by) == {"TSM", "ARM"}
    tsm = by["TSM"]
    assert tsm.side == "buy", "the document says P (Purchase)"
    assert tsm.transaction_date == "2025-07-17"
    assert (tsm.amount_low, tsm.amount_high) == (100_001.0, 250_000.0)
    arm = by["ARM"]
    assert arm.side == "buy" and arm.transaction_date == "2025-07-16"
    assert (arm.amount_low, arm.amount_high) == (50_001.0, 100_000.0)


def test_filing_status_row_is_not_read_as_a_sale():
    """`F      S     : New` contains a standalone S. It is metadata, not a Sale
    , this single confusion produced the 92% sell rate."""
    trades, _ = parse_ptr_pdf_text(_MODERN_BLOCK, "X", "2025-08-06")
    assert [t.side for t in trades] == ["buy", "buy"]


def test_a_row_never_borrows_the_next_rows_fields():
    """Two adjacent rows with different sides, dates and amounts must stay
    separate: the old parser slid every field down by one row."""
    text = (
        "Alpha Corp P 01/05/2025 01/06/2025 $1,001 - $15,000\n"
        "Holdings Inc. (AAA) [ST]\n"
        "F      S     : New\n"
        "Beta Corp S 03/09/2025 03/10/2025 $250,001 -\n"
        "Industries Ltd. (BBB) [ST] $500,000\n"
        "F      S     : New\n"
    )
    trades, _ = parse_ptr_pdf_text(text, "X", "2025-04-01")
    by = {t.ticker: t for t in trades}
    assert by["AAA"].side == "buy" and by["AAA"].transaction_date == "2025-01-05"
    assert by["AAA"].amount_high == 15_000.0
    assert by["BBB"].side == "sell" and by["BBB"].transaction_date == "2025-03-09"
    assert by["BBB"].amount_low == 250_001.0


def test_the_old_single_line_layout_still_parses():
    """2019-era rows put the ticker BEFORE the type on one line. The fix must
    not regress the format it already handled."""
    text = "JT 2U, Inc. (TWOU) [ST] S 09/10/2019 09/16/2019 $50,001 - $100,000 gfedc\n"
    trades, _ = parse_ptr_pdf_text(text, "X", "2019-09-20")
    assert len(trades) == 1
    t = trades[0]
    assert (t.ticker, t.side, t.transaction_date) == ("TWOU", "sell", "2019-09-10")
    assert (t.amount_low, t.amount_high) == (50_001.0, 100_000.0)


def test_exchange_is_neither_a_buy_nor_a_sell():
    """'E' is an Exchange. Calling it either direction would put a fictitious
    directional bet in the sample."""
    text = ("Some Corp E 05/01/2025 05/02/2025 $1,001 - $15,000\n"
            "Holdings (EEE) [ST]\n")
    trades, skipped = parse_ptr_pdf_text(text, "X", "2025-05-10")
    assert trades == []
    assert skipped == 1


def test_rows_without_a_ticker_are_skipped_not_guessed():
    """Treasury notes and unsymboled funds are genuinely unpriceable."""
    text = ("SP U.S. Treasury Note due 02/28/2027 P 11/06/2023 11/07/2023 Over\n"
            "[GS] $1,000,000\n")
    trades, skipped = parse_ptr_pdf_text(text, "X", "2023-11-20")
    assert trades == [] and skipped == 1


def test_partial_sale_is_recorded_as_a_sale():
    text = ("Gamma Corp S (partial) 02/03/2025 02/04/2025 $15,001 - $50,000\n"
            "Holdings (GGG) [ST]\n")
    trades, _ = parse_ptr_pdf_text(text, "X", "2025-03-01")
    assert len(trades) == 1
    assert trades[0].side == "sell"
    assert "partial" in trades[0].raw["type"]


def test_option_words_do_not_leak_between_adjacent_rows():
    """`asset_text` now spans exactly one block, so a Call on one row cannot
    reclassify the next row's plain stock."""
    text = (
        "NVIDIA Corporation - Call P 06/01/2024 06/15/2024 $250,001 -\n"
        "options (NVDA) [OP] $500,000\n"
        "F      S     : New\n"
        "Apple Inc. - Common Stock (AAPL) P 01/05/2024 01/20/2024 $1,001 - $15,000\n"
        "[ST]\n"
    )
    trades, _ = parse_ptr_pdf_text(text, "X", "2024-06-20")
    by = {t.ticker: t.instrument for t in trades}
    assert by["NVDA"] == "call_option"
    assert by["AAPL"] == "stock", "the Call above must not leak downward"


# ---- House PTR: wrapped amount buckets ------------------------------------
#
# The Clerk wraps the amount across the block's two lines AND drops checkbox
# glyphs into the gap. A contiguous "$lo - $hi" pattern cannot span 'gfedc', so
# the high bound was silently lost: 10.5% of stored House rows had a NULL
# amount_high, and since the $1,001 / $15,001 / $50,001 / $100,001 / $250,001
# tiers are never open-ended those were losses, not "Over $X" rows.
# amount_for_basis then falls back to hi = lo, sizing a $15,001-$50,000 trade
# at $15,001, 54% light, which feeds the "large" conviction subset,
# walk-forward weights and the cost notional.

def test_wrapped_amount_survives_checkbox_glyphs():
    """Verbatim layout from Clerk doc 20012476 (2019)."""
    text = ("JT 2U, Inc. (TWOU) [ST] S 09/10/2019 09/16/2019 $50,001 - gfedc\n"
            "$100,000\n"
            "FILING STATUS: New\n")
    trades, _ = parse_ptr_pdf_text(text, "X", "2019-09-20")
    assert len(trades) == 1
    assert (trades[0].amount_low, trades[0].amount_high) == (50_001.0, 100_000.0)


def test_open_ended_bucket_keeps_a_null_high_bound():
    """'Over $X' is genuinely unbounded, it must NOT gain a fabricated high."""
    from copytrade.ingest.house_clerk import block_amount

    assert block_amount(" Spouse/DC Over $1,000,000") == (1_000_000.0, None)
    assert block_amount("Over $50,000,000") == (50_000_000.0, None)


def test_a_lower_second_figure_is_not_taken_as_the_high_bound():
    """A bucket ascends; a stray smaller amount must not invert it."""
    from copytrade.ingest.house_clerk import block_amount

    assert block_amount("$50,001 - $100,000") == (50_001.0, 100_000.0)
    assert block_amount("$50,001 $12") == (50_001.0, None)
    assert block_amount("") == (None, None)


# ---- House PTR: the unpadded-date vintage --------------------------------
#
# Third defect in the same family as the two above: a pattern fitted to the
# modern layout and never checked against the older one. §7.1 was "validated
# against Clerk documents 2019-2025", and the 2015-2017 renderer does not
# zero-pad the DAY: "05/5/2015", not "05/05/2015". Against a `\d{2}` spine
# those transactions do not mis-parse, they are INVISIBLE: the spine matches
# nothing, the document yields zero rows, and the run reports success.
#
# The tell was rows-per-document by year: 0.96 in 2015 and 1.09 in 2016 against
# 5.26 in 2020. On a random sample of real Clerk PDFs, accepting unpadded
# month/day recovers +9% of 2015 transactions, +32% of 2017 and +600% of 2016
# (4 found where 28 exist). 2018 onward is byte-identical, which is exactly why
# nothing downstream ever looked wrong.

def test_unpadded_day_still_parses():
    """Verbatim layout from Clerk doc 20003185 (Brad Ashford, filed 2015-05-29).
    A `\\d{2}` spine finds no transaction at all in this line."""
    text = ("SP Union Pacific Corporation (UNP) S 05/5/2015 05/5/2015 "
            "$15,001 - $50,000\n"
            "FILINg STATUS: New\n")
    trades, _ = parse_ptr_pdf_text(text, "Brad Ashford", "2015-05-29")
    assert len(trades) == 1, "the unpadded day made this row invisible"
    t = trades[0]
    assert (t.ticker, t.side, t.transaction_date) == ("UNP", "sell", "2015-05-05")
    assert (t.amount_low, t.amount_high) == (15_001.0, 50_000.0)


def test_unpadded_month_and_single_digit_both_dates():
    """Padding is inconsistent within the vintage, month, day, or both."""
    text = ("Apple Inc. (AAPL) [ST] P 1/5/2016 1/12/2016 $1,001 - $15,000\n"
            "Microsoft Corp (MSFT) [ST] P 3/7/2016 03/14/2016 $1,001 - $15,000\n")
    trades, _ = parse_ptr_pdf_text(text, "X", "2016-03-20")
    by = {t.ticker: t for t in trades}
    assert set(by) == {"AAPL", "MSFT"}
    assert by["AAPL"].transaction_date == "2016-01-05"
    assert by["MSFT"].transaction_date == "2016-03-07"


def test_relaxed_padding_cannot_bite_into_a_longer_digit_run():
    """The specific risk `\\d{1,2}` opens: a two-digit-max component could in
    principle match the TAIL of a longer number and manufacture a date.

    `123/45/2020` must not be read as `23/45/2020`. What prevents it is that
    the spine is anchored to the type code, `\\s+` pins the first date group
    immediately after it, so the engine cannot slide forward to find a match.
    That anchoring is the invariant under test; the discriminating positive
    case (unpadded dates that ONLY the relaxed pattern accepts) is
    `test_unpadded_day_still_parses`.
    """
    text = "Holdings (AAA) [ST] P 123/45/2020 6/7/2020 $1,001 - $15,000\n"
    trades, _ = parse_ptr_pdf_text(text, "X", "2020-08-01")
    assert trades == [], "a 3-digit component is not a date to slice up"


def test_dates_without_a_type_code_are_not_a_transaction():
    """Prose and signature lines carry dates too. The type code is what makes a
    date pair a transaction."""
    text = ("SP U.S. Treasury Note due 2/28/2027 maturing 3/1/2027\n"
            "FILINg STATUS: New\n"
            "Digitally Signed: Hon. Someone , 05/29/2015\n")
    trades, _ = parse_ptr_pdf_text(text, "X", "2015-06-01")
    assert trades == [], "no type code owns those dates"


# ---- House PTR scrape orchestration --------------------------------------
#
# These cover the two failures that actually destroyed a rebuild, neither of
# which any parser test could catch, because both live in the loop rather than
# in the parsing:
#
#   * `_extract_pdf_text` sat inside a `try` that caught only httpx.HTTPError.
#     pdfplumber raises other things on a truncated PDF, so ONE bad document
#     propagated out and abandoned every remaining year.
#   * the resume cursor was written once, after all years. Trades commit per
#     document, so a crash kept the rows and lost the cursor, which is how the
#     2021 pass came to hold 1,422 rows against 0 recorded DocIDs, and why
#     2022-2026 were never attempted at all.

def _fd_zip(doc_ids: list[str], year: int = 2021) -> bytes:
    """A real FD.zip: one <Member> PTR row per doc id, zipped as <YEAR>FD.xml."""
    import io
    import zipfile

    rows = "".join(
        f"<Member><Last>Filer{i}</Last><First>A</First><FilingType>P</FilingType>"
        f"<StateDst>CA01</StateDst><Year>{year}</Year>"
        f"<FilingDate>2/1/{year}</FilingDate><DocID>{d}</DocID></Member>"
        for i, d in enumerate(doc_ids)
    )
    xml = f'<?xml version="1.0"?><FinancialDisclosure>{rows}</FinancialDisclosure>'
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(f"{year}FD.xml", xml)
    return buf.getvalue()


class _FakeResponse:
    def __init__(self, content: bytes, status_code: int = 200):
        self.content = content
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code != 200:
            raise RuntimeError(f"HTTP {self.status_code}")


def _fake_client_factory(zip_bytes: bytes):
    """Stands in for httpx.AsyncClient: the index URL yields the zip, every
    PDF URL yields bytes naming its own doc id."""
    class _FakeClient:
        def __init__(self, *a, **kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def get(self, url, *a, **kw):
            if url.endswith("FD.zip"):
                return _FakeResponse(zip_bytes)
            doc = url.rsplit("/", 1)[-1].replace(".pdf", "")
            return _FakeResponse(f"PDF:{doc}".encode())

    return _FakeClient


_ONE_BUY = ("Apple Inc. (AAPL) [ST] P 01/05/2021 01/20/2021 $1,001 - $15,000\n")


def test_one_unreadable_pdf_does_not_abandon_the_rest_of_the_run(conn, monkeypatch):
    """The 2017/2021 failure: a single malformed PDF killed everything after
    it. Document 2 raises; documents 1, 3, 4 and 5 must still be ingested."""
    import copytrade.ingest.house_clerk as hc

    ids = [f"1000000{i}" for i in range(1, 6)]
    monkeypatch.setattr(hc.httpx, "AsyncClient", _fake_client_factory(_fd_zip(ids)))

    def extract(pdf_bytes: bytes) -> str:
        if b"10000002" in pdf_bytes:
            raise ValueError("truncated PDF, pdfplumber cannot open this")
        return _ONE_BUY

    monkeypatch.setattr(hc, "_extract_pdf_text", extract)

    class S:
        edgar_user_agent = "test"

    res = asyncio.run(hc.scrape_recent_ptrs(conn, S(), years=[2021]))

    assert res.inserted == 4, "the four readable documents must all be ingested"
    assert "1 PDFs unreadable" in (res.note or "")
    assert res.error is None, "a run that ingested 4 of 5 documents did not fail"
    seen = store.get_state(conn, "scrape:house_clerk:seen_docids") or ""
    assert "10000002" not in seen, "an unreadable doc must be retried next run"
    assert sum(i in seen for i in ids) == 4


def test_a_blanket_non_200_is_counted_not_silently_skipped(conn, monkeypatch):
    """A 403/429 across the board is the likeliest way this scrape fails
    wholesale, and it arrives as a STATUS CODE, not an exception. Leaving that
    path uncounted made the run report a tidy zero, §7.4's 'the provider is
    refusing' wearing the face of 'nothing new today'."""
    import copytrade.ingest.house_clerk as hc

    ids = [f"5000000{i}" for i in range(1, 6)]
    Client = _fake_client_factory(_fd_zip(ids))

    class _Refusing(Client):
        async def get(self, url, *a, **kw):
            r = await Client.get(self, url, *a, **kw)
            if not url.endswith("FD.zip"):
                r.status_code = 403
            return r

    monkeypatch.setattr(hc.httpx, "AsyncClient", _Refusing)
    monkeypatch.setattr(hc, "_extract_pdf_text", lambda b: _ONE_BUY)

    class S:
        edgar_user_agent = "test"

    res = asyncio.run(hc.scrape_recent_ptrs(conn, S(), years=[2021]))

    assert res.inserted == 0
    assert "non-200" in (res.note or ""), res.note
    assert res.degraded and "DEGRADED" in res.degraded, (
        "a run where every document was refused must say so"
    )


def test_a_degraded_run_is_not_reported_as_a_failed_one(conn, monkeypatch):
    """`error` means the run failed: the scheduler skips alert dispatch on it
    and counts it toward disabling the source after MAX_CONSECUTIVE_FAILURES.
    An incomplete-but-working run must not borrow that field, or a stretch of
    flaky fetches silently switches off the only post-2021 congressional feed.
    """
    import copytrade.ingest.house_clerk as hc

    ids = [f"6000000{i}" for i in range(1, 6)]
    Client = _fake_client_factory(_fd_zip(ids))

    class _Refusing(Client):
        async def get(self, url, *a, **kw):
            r = await Client.get(self, url, *a, **kw)
            if not url.endswith("FD.zip"):
                r.status_code = 503
            return r

    monkeypatch.setattr(hc.httpx, "AsyncClient", _Refusing)
    monkeypatch.setattr(hc, "_extract_pdf_text", lambda b: _ONE_BUY)

    class S:
        edgar_user_agent = "test"

    res = asyncio.run(hc.scrape_recent_ptrs(conn, S(), years=[2021]))
    assert res.degraded, "the run is degraded"
    assert res.error is None, "but it did not FAIL — error drives the disable path"


def test_a_year_with_an_unreadable_index_survives_the_summary_note(conn, monkeypatch):
    """The 'no XML in FD.zip' note used to be ASSIGNED to res.note and was then
    overwritten by the OCR/unreadable summary built at the end of the run.

    Reproducing that needs TWO years: one whose index cannot be read, and one
    that goes on to produce a summary note. With only the broken year there is
    nothing to overwrite it and the bug hides, which is exactly what a
    single-year version of this test failed to catch.
    """
    import io
    import zipfile

    import copytrade.ingest.house_clerk as hc

    empty = io.BytesIO()
    with zipfile.ZipFile(empty, "w") as zf:
        zf.writestr("readme.txt", "no xml here")
    good = _fd_zip(["70000001"], year=2021)

    class _MixedYears:
        def __init__(self, *a, **kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def get(self, url, *a, **kw):
            if url.endswith("2020FD.zip"):
                return _FakeResponse(empty.getvalue())     # index unreadable
            if url.endswith("FD.zip"):
                return _FakeResponse(good)
            return _FakeResponse(b"PDF")

    monkeypatch.setattr(hc.httpx, "AsyncClient", _MixedYears)
    # 2021's one document is a scan -> produces the summary note that used to
    # clobber 2020's
    monkeypatch.setattr(hc, "_extract_pdf_text", lambda b: "")

    class S:
        edgar_user_agent = "test"

    res = asyncio.run(hc.scrape_recent_ptrs(conn, S(), years=[2020, 2021]))
    assert "need OCR" in (res.note or ""), res.note
    assert "2020: no XML in FD.zip" in (res.note or ""), (
        "the skipped year must survive alongside the summary, not be replaced "
        f"by it: got {res.note!r}"
    )


def test_the_resume_cursor_survives_a_crash_partway_through(conn, monkeypatch):
    """Trades commit per document, so the cursor must too. Before checkpointing
    it was written only after every year finished, and a crash discarded the
    whole run's progress while leaving its rows behind."""
    import copytrade.ingest.house_clerk as hc

    monkeypatch.setattr(hc, "CHECKPOINT_EVERY", 2)
    ids = [f"2000000{i}" for i in range(1, 6)]
    monkeypatch.setattr(hc.httpx, "AsyncClient", _fake_client_factory(_fd_zip(ids)))

    calls = {"n": 0}

    def extract(pdf_bytes: bytes) -> str:
        calls["n"] += 1
        if calls["n"] > 4:
            raise KeyboardInterrupt("operator stops the run")
        return _ONE_BUY

    monkeypatch.setattr(hc, "_extract_pdf_text", extract)

    class S:
        edgar_user_agent = "test"

    with pytest.raises(KeyboardInterrupt):
        asyncio.run(hc.scrape_recent_ptrs(conn, S(), years=[2021]))

    # the crash must NOT have cost the documents already processed
    seen = store.get_state(conn, "scrape:house_clerk:seen_docids") or ""
    assert sum(i in seen for i in ids) >= 4, (
        "documents processed before the crash must survive in the cursor, "
        f"got {seen!r}"
    )


def test_a_second_run_skips_what_the_cursor_already_recorded(conn, monkeypatch):
    """Resume must actually resume, re-fetching thousands of already-ingested
    PDFs and deduping them all is the expensive way to do nothing."""
    import copytrade.ingest.house_clerk as hc

    ids = [f"3000000{i}" for i in range(1, 4)]
    monkeypatch.setattr(hc.httpx, "AsyncClient", _fake_client_factory(_fd_zip(ids)))
    monkeypatch.setattr(hc, "_extract_pdf_text", lambda b: _ONE_BUY)

    class S:
        edgar_user_agent = "test"

    first = asyncio.run(hc.scrape_recent_ptrs(conn, S(), years=[2021]))
    assert first.inserted == 3

    second = asyncio.run(hc.scrape_recent_ptrs(conn, S(), years=[2021]))
    assert second.inserted == 0 and second.deduped == 0, (
        "the second run should not have fetched anything at all"
    )


def test_reparse_clears_both_the_cursor_and_the_existing_house_rows(conn, monkeypatch):
    """--reparse exists for parser changes, so leaving rows behind would mean
    the old parser's output silently surviving the fix (dedup keys ignore the
    fields a re-parse corrects)."""
    import copytrade.ingest.house_clerk as hc

    ids = [f"4000000{i}" for i in range(1, 4)]
    monkeypatch.setattr(hc.httpx, "AsyncClient", _fake_client_factory(_fd_zip(ids)))
    monkeypatch.setattr(hc, "_extract_pdf_text", lambda b: _ONE_BUY)

    class S:
        edgar_user_agent = "test"

    asyncio.run(hc.scrape_recent_ptrs(conn, S(), years=[2021]))
    before = conn.execute("select count(*) from trades where source='house_clerk'").fetchone()[0]
    assert before == 3

    again = asyncio.run(hc.scrape_recent_ptrs(conn, S(), years=[2021], reparse=True))
    after = conn.execute("select count(*) from trades where source='house_clerk'").fetchone()[0]
    assert again.inserted == 3 and after == 3, "rebuilt, not doubled"


# ---- House PTR: a swallowed error must not read as a clean run ------------
#
# Per-document errors are caught so one corrupt PDF cannot abandon a 12-year
# scrape (that is how the 2017 and 2021 passes were lost). The cost is that a
# real outage stops looking like one, §7.4's "0 new" that actually meant "the
# provider is refusing". A run that could not read most of what it asked for
# has to say so where the caller looks.

def test_a_mostly_failed_run_is_reported_as_degraded():
    from copytrade.ingest.house_clerk import degradation_error

    err = degradation_error(attempted=300, unreadable=280)
    assert err and err.startswith("DEGRADED")
    assert "280 of 300" in err


def test_a_healthy_run_reports_no_error():
    from copytrade.ingest.house_clerk import degradation_error

    # 733 of 7,612 PTRs are image-only scans, but those are counted as
    # ocr_needed, not failures: the real rebuild had 0 unreadable.
    assert degradation_error(attempted=7612, unreadable=0) is None
    assert degradation_error(attempted=800, unreadable=40) is None
    assert degradation_error(attempted=0, unreadable=0) is None


def test_a_tiny_run_is_not_judged_on_its_failure_rate():
    """2 of 3 unreadable is noise; the rate only means something at volume."""
    from copytrade.ingest.house_clerk import degradation_error

    assert degradation_error(attempted=3, unreadable=2) is None


def test_a_small_run_where_EVERYTHING_failed_still_reports():
    """The scheduler polls incrementally, so `attempted` is usually a handful
    of new DocIDs and never reaches the rate threshold. A rate is meaningless
    at n=3, but 'every single document we asked for failed' is not."""
    from copytrade.ingest.house_clerk import degradation_error

    assert degradation_error(attempted=3, unreadable=3)
    assert degradation_error(attempted=1, unreadable=1) is None, "n=1 is noise"


def test_two_transactions_on_one_line_are_both_parsed():
    """When the PDF's columns collapse, two rows share a text line. Anchoring
    with search() instead of finditer() silently dropped the second, and the
    skipped counter never mentioned it."""
    text = ("Alpha (AAA) [ST] P 01/05/2025 01/06/2025 $1,001 - $15,000 "
            "Beta (BBB) [ST] S 03/09/2025 03/10/2025 $250,001 - $500,000\n")
    trades, _ = parse_ptr_pdf_text(text, "X", "2025-04-01")
    by = {t.ticker: t for t in trades}
    assert set(by) == {"AAA", "BBB"}
    assert by["AAA"].side == "buy" and by["AAA"].transaction_date == "2025-01-05"
    assert (by["AAA"].amount_low, by["AAA"].amount_high) == (1_001.0, 15_000.0)
    assert by["BBB"].side == "sell" and by["BBB"].transaction_date == "2025-03-09"
    assert (by["BBB"].amount_low, by["BBB"].amount_high) == (250_001.0, 500_000.0)


def test_continuation_attaches_to_the_last_block_on_a_shared_line():
    """Only the final transaction on a line owns the lines beneath it."""
    text = ("Alpha (AAA) [ST] P 01/05/2025 01/06/2025 $1,001 - $15,000 "
            "Beta Corp P 03/09/2025 03/10/2025 $250,001 -\n"
            "Industries Ltd. (BBB) [ST] $500,000\n")
    trades, _ = parse_ptr_pdf_text(text, "X", "2025-04-01")
    by = {t.ticker: t for t in trades}
    assert by["AAA"].amount_high == 15_000.0, "first block must not absorb the wrap"
    assert (by["BBB"].amount_low, by["BBB"].amount_high) == (250_001.0, 500_000.0)
