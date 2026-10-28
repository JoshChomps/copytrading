"""US House PTR scraper, the authoritative, complete House feed.

Two-stage, exactly as the Office of the Clerk publishes it:
  1. annual index: GET .../public_disc/financial-pdfs/<YEAR>FD.zip, which
     contains <YEAR>FD.xml: one <Member> row per filing with FilingType,
     name, state/district, filing date and DocID. FilingType 'P' = Periodic
     Transaction Report (the tradeable signal). This is the backbone: it
     lists EVERY filer, so unlike a symbol-keyed aggregator it has no blind
     spots.
  2. per-PTR detail: GET .../public_disc/ptr-pdfs/<YEAR>/<DocID>.pdf and
     extract the transaction table (ticker, date, Purchase/Sale, amount
     bucket).

Honest status: the FD.xml index parser is exact to the published schema and
unit-tested. PDF extraction is the genuinely hard, format-drifting part
electronic PTRs have a text layer this parser reads; older/scanned PTRs need
OCR (not wired: flagged per-filing, never guessed). And sec-style fair
access does not apply here, but we still throttle and send a contact UA.
disclosures-clerk.house.gov is unreachable from this dev sandbox (403), so
run one real fetch and eyeball `orders-list`/trades before trusting a bulk
scrape: see docs/SCRAPING.md.
"""

from __future__ import annotations

import io
import re
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass
from datetime import date, datetime

import httpx

from copytrade.ingest.congress import CongressTrade, store_congress_trades
from copytrade.ingest.senate_watcher import clean_ticker
from copytrade.scrape.base import ScrapeResult

INDEX_URL = "https://disclosures-clerk.house.gov/public_disc/financial-pdfs/{year}FD.zip"
PTR_PDF_URL = "https://disclosures-clerk.house.gov/public_disc/ptr-pdfs/{year}/{doc_id}.pdf"
UA = "copytrade-research (contact via COPYTRADE_EDGAR_USER_AGENT)"

#: Persist the resume cursor every N processed documents.
#: Measured on the 2026-08-13 full rebuild: 6,879 documents in 27.6 minutes =
#: **4.1 docs/sec**, so 100 documents is ~24 seconds of re-fetching lost to a
#: crash, across 68 cursor writes for the whole 12-year run. Each write
#: serialises the entire DocID set (~62KB at the end), so the total written is
#: a couple of MB against 7,612 PDF downloads, the ratio that matters, and the
#: reason this is not simply every document.
CHECKPOINT_EVERY = 100


@dataclass
class HouseFiling:
    doc_id: str
    first: str
    last: str
    filing_type: str
    state_dst: str
    year: str
    filing_date: str        # ISO
    is_ptr: bool

    @property
    def name(self) -> str:
        return f"{self.first} {self.last}".strip().title()


def _iso(mdy: str | None) -> str | None:
    if not mdy:
        return None
    for fmt in ("%m/%d/%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(mdy.strip(), fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


def parse_fd_index(xml_text: str) -> list[HouseFiling]:
    """Parse <YEAR>FD.xml -> filings. Tolerant of missing optional tags."""
    root = ET.fromstring(xml_text)
    out = []
    for m in root.iter("Member"):
        def g(tag):
            e = m.find(tag)
            return e.text.strip() if e is not None and e.text else ""
        ftype = g("FilingType")
        doc_id = g("DocID")
        if not doc_id:
            continue
        out.append(HouseFiling(
            doc_id=doc_id, first=g("First"), last=g("Last"),
            filing_type=ftype, state_dst=g("StateDst"), year=g("Year"),
            filing_date=_iso(g("FilingDate")) or "",
            is_ptr=ftype.upper() == "P",
        ))
    return out


# --- PDF transaction extraction -------------------------------------------
#
# A House PTR transaction is a BLOCK of lines, not a line. The asset name wraps,
# and the ticker frequently lands on a different line from the type and dates:
#
#   Arm Holdings plc - American        P 07/16/2025 07/16/2025 $50,001 -
#   Depositary Shares (ARM) [ST]                              $100,000
#   F      S     : New
#   S          O : Morgan Stanley - E*TRADE #2
#
# THE BUG THIS REPLACES. The previous regex anchored on the ticker and then
# scanned FORWARD (with re.DOTALL) for the type code. Forward of a ticker, the
# very next standalone capital letter is usually the `S` in the "Filing Status"
# line above: so the parser read "Sale", then kept scanning and took the NEXT
# transaction's date and amount as well.
#
# The damage was systematic and grew with the format. House sells ran ~46% of
# trades in 2015 and 92% in 2025-26, while the Senate feed held ~50% throughout;
# congressional trading did not change, the PDF layout did. Verified against
# Clerk doc 20030803: the document contains 55 Purchases and one Sale, and the
# old parser reported 1 buy and 55 sells. Its TSM row
#   "Taiwan Semiconductor P 07/17/2025 07/17/2025 $100,001 - / ... (TSM) $250,000"
# , was stored as `sell, 2025-07-10, $15,001+`: side, date and amount all wrong.
#
# So: anchor on the SPINE (type + both dates), which is unambiguous and always
# on one line, then attach the ticker and the rest of the amount from the
# continuation lines that belong to that same block.

#: The one unambiguous landmark in a transaction block: the type code followed
#: by the transaction and notification dates. `E` is Exchange, which the House
#: uses and which is neither a buy nor a sell.
#: Longest-first alternation: `Purchase` must be tried before `P`, or the
#: leftmost-first engine matches the `P` and then fails the trailing `\b`.
#: Real Clerk PDFs use the single letters; the spelled-out forms are accepted
#: because they cost nothing and older/third-party renderings do use them.
#:
#: The month and day are `\d{1,2}`, NOT `\d{2}`. The 2015-2017 Clerk renderer
#: does not zero-pad the day, so doc 20003185 reads
#:   "SP Union Pacific Corporation (UNP) S 05/5/2015 05/5/2015 $15,001 - $50,000"
#: and a `\d{2}` spine matches nothing at all, the transaction is not
#: mis-parsed, it is invisible, and the document silently contributes zero rows.
#: Same family as the ticker-anchored bug above: a pattern fitted to the modern
#: layout, never checked against the older one (§7.1 was "validated 2019-2025").
#: Measured on a random sample of Clerk PDFs, relaxing the padding recovers
#: +9% of 2015 transactions, +32% of 2017, and +600% of 2016 (4 found -> 28
#: real); 2018 onward is byte-identical, which is why it went unnoticed.
#: `_iso` already accepts unpadded input, `%m/%d/%Y` parses "05/5/2015", so
#: the regex was the only thing rejecting these rows.
_SPINE = re.compile(
    r"\b(Purchase|Sale|Exchange|P|S|E)\b\s*(\(partial\))?\s+"
    r"(\d{1,2}/\d{1,2}/\d{4})\s+"
    r"(\d{1,2}/\d{1,2}/\d{4})",
    re.IGNORECASE,
)

#: Ticker in parentheses, e.g. "(TSM)" or "(BRK.B)".
_TICKER = re.compile(r"\(([A-Z][A-Z.\-]{0,9})\)")

#: Individual dollar figures inside a block's amount text.
#:
#: Deliberately NOT a single "$lo - $hi" pattern. The Clerk wraps the bucket
#: across the block's two lines and drops checkbox glyphs into the gap:
#:
#:   JT 2U, Inc. (TWOU) [ST] S 09/10/2019 09/16/2019 $50,001 - gfedc
#:   $100,000
#:
#: A contiguous `\$[\d,]+\s*-\s*\$[\d,]+` cannot span 'gfedc', so it matched
#: only "$50,001" and the high bound was silently lost, 10.5% of stored House
#: rows had a NULL `amount_high` this way, and since the $1,001 / $15,001 /
#: $50,001 / $100,001 / $250,001 tiers are never open-ended those were losses,
#: not genuine "Over $X" rows. `amount_for_basis` then falls back to hi = lo,
#: so a $15,001-$50,000 trade was sized at $15,001, 54% light, which feeds
#: the "large" conviction subset, walk-forward weights and the cost notional.
_MONEY = re.compile(r"\$\s?([\d,]+)")

#: Per-transaction metadata rows that follow the block and must never be read
#: as part of it, the `S` in "F S : New" is precisely what the old parser
#: mistook for a Sale. The Clerk renders these labels two ways depending on
#: vintage: spelled out ("FILING STATUS: New", 2019) and as spaced-out glyphs
#: ("F      S     : New", 2025), so both spellings are matched.
_META = re.compile(
    r"^\s*(?:[A-Z]\s+[A-Z]?\s*[A-Z]?\s*:"
    r"|FILING\s+STATUS\s*:|SUBHOLDING\s+OWNER\s*:|DESCRIPTION\s*:|LOCATION\s*:)",
    re.IGNORECASE)

#: Repeated page/table headers.
_HEADER = re.compile(r"^\s*(ID\s+Owner\s+Asset|Type\s+Date|\$\d+\?|Filing ID|"
                     r"Clerk of the House|Name:|Status:|State/District:)", re.IGNORECASE)


#: A run must read at least this many PTRs before its failure RATE means
#: anything: 2 of 3 unreadable is noise, 200 of 300 is an outage. At the 25%
#: fraction below, 20 attempts means 6 failures are needed to trip it.
_DEGRADED_MIN_ATTEMPTS = 20
#: Fraction of attempted documents that may fail before the run is DEGRADED.
#:
#: Calibration: the 2026-08-13 full rebuild attempted 7,612 PTRs and recorded
#: **0 HTTP failures and 0 PDF failures**, the observed baseline is zero. The
#: 733 image-only scans are counted as `ocr_needed`, deliberately NOT as
#: failures, so they cannot drag this toward firing.
#:
#: 25% therefore sits 25 points above a zero baseline, which looks permissive
#: until you note what it gates: `degraded` means "do not trust this run at
#: all". Every non-zero failure count ALREADY appears in `res.note` with exact
#: numbers, at any rate, so nothing below this threshold is silent, the choice
#: here is only about when to escalate from "here are the counts" to "these
#: counts are a floor, re-run before rebuilding panels".
_DEGRADED_FRACTION = 0.25
#: A run small enough that no RATE is meaningful can still say something: if
#: every document it tried failed, that is a signal at n=3 even though 1 of 3
#: would not be. Without this the guard is inert on the small incremental polls
#: the scheduler runs, which may never reach _DEGRADED_MIN_ATTEMPTS.
#: This cannot false-fire on a quiet day: a scanned PTR returns empty text and
#: is counted as `ocr_needed`, so reaching 3-of-3 requires three genuine HTTP
#: or PDF errors.
_DEGRADED_MIN_TOTAL_FAILURE = 3


def degradation_error(attempted: int, unreadable: int) -> str | None:
    """The message for a run that could not read most of what it asked for.

    Catching per-document errors buys resilience and costs visibility: an
    outage or a blanket 403 otherwise looks like a tidy run that simply found
    little. That is the §7.4 failure mode, a status reading "done" when it
    means "the provider is refusing".

    The caller puts this in `ScrapeResult.degraded`, NOT `error`: the run
    worked, it is just incomplete, and `error` would stop the scheduler
    dispatching alerts for what it did ingest and count the run toward
    disabling the source.
    """
    if attempted <= 0 or unreadable <= 0:
        return None
    everything_failed = (unreadable == attempted
                         and attempted >= _DEGRADED_MIN_TOTAL_FAILURE)
    over_rate = (attempted >= _DEGRADED_MIN_ATTEMPTS
                 and unreadable > attempted * _DEGRADED_FRACTION)
    if not (everything_failed or over_rate):
        return None
    return (
        f"DEGRADED: {unreadable} of {attempted} PTRs unreadable "
        f"({100 * unreadable // attempted}%), treat these counts as a floor, "
        "not a complete scrape, and re-run before rebuilding panels"
    )


def house_instrument(asset_text: str) -> str:
    """Detect option type from a House PTR asset description / type code.
    This is what finally captures a member's LEVERAGED bets (e.g. Pelosi's
    calls) instead of mislabeling them 'stock'."""
    t = asset_text.lower()
    is_opt = ("option" in t or "[op]" in t or "(op)" in t
              or "call" in t or "put" in t)
    if not is_opt:
        return "stock"
    if "call" in t:
        return "call_option"
    if "put" in t:
        return "put_option"
    return "option"


def block_amount(amount_text: str) -> tuple[float | None, float | None]:
    """(low, high) from a block's amount text, tolerant of wrapping and noise.

    Collects every dollar figure in the block rather than requiring a
    contiguous "$lo - $hi", see `_MONEY` for the layout that defeats the
    contiguous form. An "Over" bucket is genuinely open-ended and keeps a None
    high bound. A second figure is only accepted as the high bound when it
    exceeds the first, so a stray amount ahead of the bucket cannot invert it.
    """
    nums = [float(n.replace(",", "")) for n in _MONEY.findall(amount_text or "")]
    if not nums:
        return None, None
    if re.search(r"\bover\b", amount_text or "", re.IGNORECASE):
        return nums[0], None
    if len(nums) >= 2 and nums[1] > nums[0]:
        return nums[0], nums[1]
    return nums[0], None


def _blocks(lines: list[str]) -> list[tuple[re.Match, str, str]]:
    """Group the text layer into transaction blocks.

    Yields (spine_match, amount_text, asset_text) per transaction.

    A line may carry MORE THAN ONE spine when the PDF's columns collapse, so
    every match is walked rather than just the first: field slices are bounded
    by the neighbouring matches on the same line, and only the last match on a
    line takes the continuation lines below it.
    """
    out = []
    for i, line in enumerate(lines):
        if _HEADER.match(line):
            continue
        matches = list(_SPINE.finditer(line))
        if not matches:
            continue
        for k, m in enumerate(matches):
            prev_end = matches[k - 1].end() if k else 0
            next_start = matches[k + 1].start() if k + 1 < len(matches) else len(line)
            asset_text = line[prev_end:m.start()]
            amount_text = line[m.end():next_start]
            if k + 1 < len(matches):
                out.append((m, amount_text, asset_text))
                continue
            # Continuation lines belong to the last block on this line, until
            # the next spine, a metadata row, or a repeated header.
            for nxt in lines[i + 1: i + 4]:
                if _SPINE.search(nxt) or _META.match(nxt) or _HEADER.match(nxt):
                    break
                if not nxt.strip():
                    continue
                # on a continuation line the amount starts at the first '$'
                cut = nxt.find("$")
                if cut == -1:
                    asset_text += " " + nxt
                else:
                    asset_text += " " + nxt[:cut]
                    amount_text += " " + nxt[cut:]
            out.append((m, amount_text, asset_text))
    return out


def parse_ptr_pdf_text(text: str, member: str, filing_date: str) -> tuple[list[CongressTrade], int]:
    """Extract transactions from a PTR PDF's text layer. Returns
    (trades, skipped). Scanned PTRs yield empty text -> (`[]`, 0); the caller
    counts those as needing OCR.

    Parsing is anchored on the transaction SPINE (type code + transaction date +
    notification date), which is always on a single line, rather than on the
    ticker: the asset name wraps and routinely pushes the ticker onto the next
    line, so a ticker-anchored scan reads the following row's fields. See the
    module comment above `_SPINE` for what that cost.
    """
    trades: list[CongressTrade] = []
    skipped = 0
    # pdfplumber renders the form's label glyphs as NULs ("F\x00\x00\x00\x00\x00
    # S\x00\x00\x00\x00\x00: New"), which stops the metadata rows from matching
    # `_META` and lets broker names leak into the asset text used for option
    # detection. Normalise them to spaces first.
    lines = text.replace("\x00", " ").splitlines()
    for m, amount_text, asset_text in _blocks(lines):
        tick = _TICKER.search(asset_text)
        ticker = clean_ticker(tick.group(1)) if tick else None
        tx_iso = _iso(m.group(3))
        lo, hi = block_amount(amount_text)
        if not ticker or not tx_iso:
            skipped += 1
            continue
        code = m.group(1).upper()[0]
        # 'E' is an Exchange: neither a buy nor a sell, and calling it either
        # would put a fictitious directional bet in the sample.
        side = {"P": "buy", "S": "sell"}.get(code)
        if side is None:
            skipped += 1
            continue
        partial = bool(m.group(2))
        trades.append(CongressTrade(
            name=member, chamber="House", ticker=ticker, side=side,
            transaction_date=tx_iso, filing_date=filing_date,
            amount_low=lo, amount_high=hi,
            # option words can sit before the ticker ('NVIDIA Corp - Call
            # options (NVDA)') or after it (the '[OP]' asset-type code), and
            # `asset_text` now spans exactly this block's lines and no others.
            instrument=house_instrument(asset_text),
            source="house_clerk",
            raw={"type": code + (" (partial)" if partial else ""),
                 "asset": asset_text.strip()[:80]},
        ))
    return trades, skipped


def _extract_pdf_text(pdf_bytes: bytes) -> str:
    """Best-effort text layer. Uses pdfplumber if installed; empty string
    (=> caller flags OCR-needed) otherwise, rather than a hard dependency."""
    try:
        import pdfplumber  # optional
    except ImportError:
        return ""
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        return "\n".join(page.extract_text() or "" for page in pdf.pages)


async def scrape_recent_ptrs(conn, settings, years: list[int] | None = None,
                             reparse: bool = False) -> ScrapeResult:
    """Download the current year's index, fetch NEW PTR PDFs, store trades.
    Resume state = the set of DocIDs already processed (in poll_state).
    reparse=True clears that state so every PTR is re-fetched and re-parsed
    needed after a parser change (e.g. adding option detection) to rebuild
    cached trades. Dedup at the DB layer keeps it idempotent.

    Raises RuntimeError without pdfplumber. The guard lives HERE, at the single
    entry point, rather than in the CLI command: the scrape scheduler
    (`scrape/sources.py::HouseClerkSource`) reaches this function directly and
    `scrape_sources` defaults to ["auto"], so a CLI-only guard left the
    scheduler path silently ingesting nothing.
    """
    from copytrade.store import db as store

    try:
        import pdfplumber  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "pdfplumber is required for House PTR ingestion, without a PDF "
            "text layer every filing is skipped as 'needs OCR' and the run "
            "ingests nothing. Install it with: pip install pdfplumber==0.11.4"
        ) from e

    if reparse:
        # dedup would keep the stale instrument (source_uid ignores it), so a
        # true re-parse must clear both the seen-DocID cursor AND the existing
        # house rows, then rebuild them with the corrected parser
        store.set_state(conn, "scrape:house_clerk:seen_docids", "")
        conn.execute("DELETE FROM trades WHERE source='house_clerk'")
        conn.commit()
    years = years or [date.today().year]
    res = ScrapeResult(source="house_clerk")
    seen_key = "scrape:house_clerk:seen_docids"
    seen = set((store.get_state(conn, seen_key) or "").split(",")) - {""}
    ua = {"User-Agent": getattr(settings, "edgar_user_agent", UA)}
    ocr_needed = 0
    http_failed = 0
    pdf_failed = 0
    attempted = 0
    notes: list[str] = []

    # The cursor used to be written ONCE, after every year had finished. Trades
    # commit per document, so an exception anywhere in the run left the rows in
    # place and the cursor untouched, the next run then re-fetched thousands of
    # already-ingested PDFs and deduped them all. That is not hypothetical: the
    # 2021 pass stored 1,422 rows against 0 recorded DocIDs, and 2017 died
    # outright. Checkpoint often enough that a crash costs minutes, not a year;
    # `set_state` already commits, so this is the whole of it.
    def checkpoint() -> None:
        store.set_state(conn, seen_key, ",".join(sorted(seen)))

    async with httpx.AsyncClient(headers=ua, timeout=120, follow_redirects=True) as client:
        for year in years:
            zresp = await client.get(INDEX_URL.format(year=year))
            zresp.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(zresp.content)) as zf:
                xml_name = next((n for n in zf.namelist() if n.upper().endswith(".XML")), None)
                if not xml_name:
                    # Collected, not assigned: a bare `res.note = ...` here was
                    # overwritten by the OCR/unreadable summary at the end, so a
                    # year whose index could not be read vanished from the
                    # report entirely: the same silent year-loss this function
                    # was hardened against.
                    notes.append(f"{year}: no XML in FD.zip, YEAR SKIPPED")
                    continue
                filings = parse_fd_index(zf.read(xml_name).decode("utf-8", "replace"))
            for f in filings:
                if not f.is_ptr or f.doc_id in seen or not f.filing_date:
                    continue
                attempted += 1
                try:
                    presp = await client.get(PTR_PDF_URL.format(year=f.year or year, doc_id=f.doc_id))
                    if presp.status_code != 200:
                        # COUNTED, not skipped silently. A blanket 403/429 is
                        # the likeliest way this scrape fails wholesale, and it
                        # arrives as a status code rather than an exception
                        # so leaving this path uncounted made the degradation
                        # guard blind to the exact outage it exists for.
                        http_failed += 1
                        continue
                    text = _extract_pdf_text(presp.content)
                # A truncated or malformed PDF makes pdfplumber raise, and that
                # exception is NOT an httpx one, it used to propagate out of
                # the whole scrape and abandon every remaining year. One corrupt
                # document out of ~7,600 must cost that document, nothing more.
                # KeyboardInterrupt and SystemExit derive from BaseException, so
                # an operator can still stop the run.
                except Exception:
                    pdf_failed += 1
                    continue
                if not text:
                    ocr_needed += 1
                    continue
                # Mark seen only AFTER a text layer was actually extracted.
                # Marking on entry meant a run without pdfplumber (or during a
                # transient fetch failure) recorded every DocID as processed
                # while ingesting nothing: installing the dependency later
                # recovered none of them, because the resume cursor said they
                # were done. A missing dependency must cost a retry, not data.
                seen.add(f.doc_id)
                trades, skipped = parse_ptr_pdf_text(text, f.name, f.filing_date)
                ins, dedup = store_congress_trades(conn, trades)
                res.inserted += ins
                res.deduped += dedup
                res.skipped += skipped
                # this DocID was unseen, so its buys are new to us; `trades[:ins]`
                # would mis-slice when a row deduped against another source
                for t in trades:
                    if t.side == "buy":
                        res.alertable.append({
                            "filer_name": t.name, "kind": "congress", "side": t.side,
                            "ticker": t.ticker, "filing_datetime": t.filing_date,
                            "transaction_date": t.transaction_date,
                            "amount_low": t.amount_low, "amount_high": t.amount_high,
                        })
                if len(seen) % CHECKPOINT_EVERY == 0:
                    checkpoint()
    # Two checkpoint sites, not three: the per-N one bounds what a crash can
    # cost, and this one catches the tail (including a year that `continue`d
    # past its documents). A per-year call in between was covered by both.
    checkpoint()

    if ocr_needed:
        notes.append(f"{ocr_needed} scanned PTRs need OCR (no text layer), not ingested")
    if http_failed:
        notes.append(f"{http_failed} PDFs returned a non-200 status, will retry next run")
    if pdf_failed:
        notes.append(f"{pdf_failed} PDFs unreadable (fetch or PDF error), will retry next run")
    if notes:
        res.note = "; ".join(notes)
    # `degraded`, never `error`: see ScrapeResult.degraded. Both failure modes
    # count: the provider refusing and the file being corrupt are equally
    # "we asked for a document and did not get one".
    res.degraded = degradation_error(attempted, http_failed + pdf_failed)
    return res
