"""US Senate PTR scraper, live path to complement the (frozen) watcher.

The senate_watcher ingester gives real 2013-2021 history. For anything newer
you must scrape efdsearch.senate.gov directly. That site is deliberately
unfriendly to scrapers:
  * you must GET /search/home/ first to accept an agreement and receive a
    CSRF token + session cookie;
  * the report list is a POST to /search/report/data/ with that token;
  * each report is HTML (electronic filings) or a PDF (paper filings, need
    OCR, same as the House).

This module implements that handshake. It is UNVALIDATED from the dev sandbox
(efdsearch is 403 through the proxy), so treat it as a documented, best-effort
implementation to run and verify off-sandbox, the CSRF/agreement flow is the
part most likely to have drifted. For reproducible backtests prefer
`ingest-senate` (the watcher); use this for the live tail.
"""

from __future__ import annotations

from datetime import date, datetime

import httpx

from copytrade.ingest.congress import CongressTrade, store_congress_trades
from copytrade.ingest.senate_watcher import clean_ticker, parse_amount
from copytrade.scrape.base import ScrapeResult

HOME = "https://efdsearch.senate.gov/search/home/"
REPORTS = "https://efdsearch.senate.gov/search/report/data/"
UA = "copytrade-research (contact via COPYTRADE_EDGAR_USER_AGENT)"


async def _get_with_retry(client: httpx.AsyncClient, url: str, attempts: int = 4) -> httpx.Response:
    """efdsearch 503s aggressively for non-browser traffic and under load
    retry with backoff before concluding the site is unreachable (observed:
    a first-contact 503 from a GitHub runner)."""
    import asyncio

    last: httpx.Response | None = None
    for i in range(attempts):
        r = await client.get(url)
        if r.status_code < 500:
            return r
        last = r
        await asyncio.sleep(2 ** (i + 1))
    last.raise_for_status()
    return last  # pragma: no cover


async def _session(client: httpx.AsyncClient) -> str:
    """Accept the agreement, return the CSRF token for subsequent POSTs."""
    import re

    r = await _get_with_retry(client, HOME)
    r.raise_for_status()
    m = re.search(r'name="csrfmiddlewaretoken" value="([^"]+)"', r.text)
    if not m:
        raise RuntimeError("efdsearch: CSRF token not found, page layout changed")
    token = m.group(1)
    await client.post(HOME, data={
        "csrfmiddlewaretoken": token, "prohibition_agreement": "1",
    }, headers={"Referer": HOME})
    return token


def _iso(mdY: str | None) -> str | None:
    for fmt in ("%m/%d/%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime((mdY or "").strip(), fmt).strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            continue
    return None


async def scrape_recent_ptrs(conn, settings, days: int = 14) -> ScrapeResult:
    """Fetch the recent PTR list. Electronic reports are parsed for
    transactions; paper (PDF) reports are counted as OCR-needed."""
    from copytrade.store import db as store

    res = ScrapeResult(source="senate_efd")
    # efdsearch's WAF rejects bare-bones clients; look like a normal browser
    # session (still identifiable via the contact email in From)
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) copytrade-research",
        "From": getattr(settings, "edgar_user_agent", UA),
        "Accept": "text/html,application/json;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }
    frm = (date.today().replace(day=1)).strftime("%m/%d/%Y")
    async with httpx.AsyncClient(headers=headers, timeout=60, follow_redirects=True) as client:
        token = await _session(client)
        resp = await client.post(REPORTS, data={
            "csrfmiddlewaretoken": token, "start": "0", "length": "100",
            "report_types": "[11]",           # Periodic Transaction Report
            "filer_types": "[]", "submitted_start_date": frm,
            "submitted_end_date": "", "candidate_state": "", "senator_state": "",
            "office_id": "", "first_name": "", "last_name": "",
        }, headers={"Referer": HOME})
        resp.raise_for_status()
        rows = resp.json().get("data", [])
    # rows: [first, last, office, report_link_html, submitted_date]
    seen_key = "scrape:senate_efd:seen"
    seen = set((store.get_state(conn, seen_key) or "").split(",")) - {""}
    paper = 0
    for row in rows:
        link_html = row[3] if len(row) > 3 else ""
        filing_iso = _iso(row[4]) if len(row) > 4 else None
        uid = link_html  # the report URL is unique per filing
        if uid in seen or not filing_iso:
            continue
        seen.add(uid)
        if "/paper/" in link_html or ".pdf" in link_html.lower():
            paper += 1
            continue
        # electronic report parsing (fetch + table parse) is the remaining
        # work; the handshake above is the hard part. Flagged, not faked.
        res.skipped += 1
    store.set_state(conn, seen_key, ",".join(sorted(seen)))
    res.note = (f"{len(rows)} recent PTRs; {paper} paper (OCR needed); "
                "electronic-report table parse pending live validation")
    return res


# electronic-report HTML table -> trades (pure, unit-testable)
def parse_efd_report_html(html: str, member: str, filing_date: str) -> tuple[list[CongressTrade], int]:
    """Parse an electronic PTR HTML table. Columns:
    #, Transaction Date, Owner, Ticker, Asset Name, Type, Amount, ..."""
    import re

    trades, skipped = [], 0
    for row in re.findall(r"<tr[^>]*>(.*?)</tr>", html, re.DOTALL | re.IGNORECASE):
        cells = [re.sub(r"<[^>]+>", " ", c).strip()
                 for c in re.findall(r"<td[^>]*>(.*?)</td>", row, re.DOTALL | re.IGNORECASE)]
        if len(cells) < 7:
            continue
        tx_date, _owner, ticker_raw, _asset, ttype, amount = cells[1], cells[2], cells[3], cells[4], cells[5], cells[6]
        ticker = clean_ticker(ticker_raw)
        tx_iso = _iso(tx_date)
        if not ticker or not tx_iso:
            skipped += 1
            continue
        lo, hi = parse_amount(amount)
        side = "buy" if ttype.lower().startswith("purchase") else "sell" if "sale" in ttype.lower() else None
        trades.append(CongressTrade(
            name=member, chamber="Senate", ticker=ticker, side=side,
            transaction_date=tx_iso, filing_date=filing_date,
            amount_low=lo, amount_high=hi, instrument="stock", source="senate_efd",
        ))
    return trades, skipped
