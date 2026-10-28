"""Concrete sources, each wrapping an existing ingester behind the Source
interface. Adding "other significant ones" = adding a class here.

Coverage map (see docs/SCRAPING.md for the full table):

  authoritative (backbone: complete, source-of-record):
    Form4Source        US corporate insiders   EDGAR Atom feed        ~3 min
    HouseClerkSource   US House PTRs            disclosures-clerk XML  ~6 h
    SenateEFDSource    US Senate PTRs           efdsearch (or watcher) ~6 h
    CanadaCIECSource   Canadian MPs            CIEC public registry   daily

  aggregator (enrichment / watchlist, may be symbol-scoped or partial):
    FinnhubCongressSource  US House+Senate      Finnhub (symbol-keyed) ~1 h
"""

from __future__ import annotations

from datetime import date, timedelta

from copytrade.scrape.base import BaseSource, ScrapeResult


class Form4Source(BaseSource):
    name = "edgar_form4"
    kind = "insider"
    cadence_secs = 180
    authoritative = True

    async def run_once(self, conn, settings) -> ScrapeResult:
        from copytrade.ingest.edgar import EdgarClient
        from copytrade.ingest.form4 import parse_form4, store_form4
        from copytrade.store import db as store

        client = EdgarClient(settings.edgar_user_agent, settings.edgar_rps, settings.raw_dir)
        res = ScrapeResult(source=self.name)
        try:
            refs = await client.recent_form4_accessions()
            last_seen = store.get_state(conn, self.state_key())
            new_last = refs[0].accession if refs else last_seen
            for ref in refs:
                if ref.accession == last_seen:
                    break
                xml = await client.fetch_filing_xml(ref)
                if not xml:
                    continue
                accepted = await client.fetch_acceptance_datetime(ref) or f"{date.today()} 17:00:00"
                filing = parse_form4(xml)
                n = store_form4(conn, filing, ref.accession, accepted)
                res.inserted += n
                if n and filing.owner_name:
                    for tx in filing.transactions:
                        if tx.side:
                            res.alertable.append({
                                "filer_name": filing.owner_name, "kind": "insider",
                                "side": tx.side, "ticker": filing.issuer_symbol,
                                "filing_datetime": accepted, "transaction_date": tx.date,
                                "amount_low": (tx.shares or 0) * (tx.price or 0),
                                "amount_high": (tx.shares or 0) * (tx.price or 0),
                            })
            if new_last:
                store.set_state(conn, self.state_key(), new_last)
        finally:
            await client.aclose()
        return res


class FinnhubCongressSource(BaseSource):
    name = "finnhub_congress"
    kind = "congress"
    cadence_secs = 3600
    authoritative = False   # symbol-keyed: only finds trades in the watchlist

    async def run_once(self, conn, settings) -> ScrapeResult:
        from copytrade.ingest.congress import FinnhubCongress, store_congress_trades
        from copytrade.store import db as store

        res = ScrapeResult(source=self.name)
        if not settings.finnhub_api_key:
            return ScrapeResult(source=self.name, note="no Finnhub key, skipped")
        symbols = _universe(settings)
        if not symbols:
            return ScrapeResult(source=self.name, note="empty universe, skipped")
        source = FinnhubCongress(settings.finnhub_api_key)
        frm = (date.today() - timedelta(days=14)).isoformat()
        trades, _ = await source.fetch(symbols, frm, date.today().isoformat())
        for t in trades:
            filer = store.resolve_alias(conn, t.name)
            inserted, deduped = store_congress_trades(conn, [t])
            res.inserted += inserted
            res.deduped += deduped
            if inserted:
                res.alertable.append({
                    "filer_name": filer, "kind": "congress", "side": t.side,
                    "ticker": t.ticker, "filing_datetime": t.filing_date,
                    "transaction_date": t.transaction_date,
                    "amount_low": t.amount_low, "amount_high": t.amount_high,
                })
        return res


class HouseClerkSource(BaseSource):
    name = "house_clerk"
    kind = "congress"
    cadence_secs = 6 * 3600
    authoritative = True

    async def run_once(self, conn, settings) -> ScrapeResult:
        from copytrade.ingest.house_clerk import scrape_recent_ptrs

        return await scrape_recent_ptrs(conn, settings)


class SenateEFDSource(BaseSource):
    name = "senate_efd"
    kind = "congress"
    cadence_secs = 6 * 3600
    authoritative = True

    async def run_once(self, conn, settings) -> ScrapeResult:
        from copytrade.ingest.senate_efd import scrape_recent_ptrs

        return await scrape_recent_ptrs(conn, settings)


class CanadaCIECSource(BaseSource):
    name = "canada_ciec"
    kind = "congress_ca"
    cadence_secs = 24 * 3600
    authoritative = True

    async def run_once(self, conn, settings) -> ScrapeResult:
        from copytrade.ingest.canada_ciec import scrape_material_changes

        return await scrape_material_changes(conn, settings)


def _universe(settings) -> list[str]:
    from copytrade.ingest.congress import load_universe

    return load_universe(settings.congress_universe_file)
