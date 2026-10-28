"""SEC EDGAR client: fair-access compliant (declared User-Agent with contact
email, global rate limit <= 8 req/s, exponential backoff on 403/429/5xx).

Every raw payload is cached to data/raw/<accession>.xml before parsing so the
pipeline is reprocessable without re-hitting EDGAR.
"""

from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import httpx

SEC_WWW = "https://www.sec.gov"
RETRY_STATUSES = {403, 429, 500, 502, 503, 504}
BACKOFFS = [2, 4, 8, 16]


class RateLimiter:
    """Global token spacing: at most `rps` requests per second across tasks."""

    def __init__(self, rps: float):
        self._min_interval = 1.0 / max(rps, 0.1)
        self._lock = asyncio.Lock()
        self._next_at = 0.0

    async def wait(self) -> None:
        async with self._lock:
            now = time.monotonic()
            delay = self._next_at - now
            self._next_at = max(now, self._next_at) + self._min_interval
        if delay > 0:
            await asyncio.sleep(delay)


class EdgarClient:
    def __init__(self, user_agent: str, rps: float = 8.0, raw_dir: str | Path = "data/raw"):
        if "@" not in user_agent:
            raise ValueError("EDGAR User-Agent must include a contact email (fair access rules)")
        self._limiter = RateLimiter(rps)
        self._headers = {"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"}
        self._client = httpx.AsyncClient(headers=self._headers, timeout=30, follow_redirects=True)
        self.raw_dir = Path(raw_dir)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def get_text(self, url: str) -> str:
        for attempt, backoff in enumerate(BACKOFFS + [None]):
            await self._limiter.wait()
            try:
                resp = await self._client.get(url)
            except httpx.TransportError:
                if backoff is None:
                    raise
                await asyncio.sleep(backoff)
                continue
            if resp.status_code == 200:
                return resp.text
            if resp.status_code in RETRY_STATUSES and backoff is not None:
                await asyncio.sleep(backoff)
                continue
            resp.raise_for_status()
        raise RuntimeError(f"unreachable: {url}")

    # ------------------------------------------------------------ discovery

    async def recent_form4_accessions(self, count: int = 100) -> list["FilingRef"]:
        """Poll the near-real-time Atom feed of newly accepted Form 4s."""
        url = (
            f"{SEC_WWW}/cgi-bin/browse-edgar?action=getcurrent&type=4"
            f"&company=&dateb=&owner=include&count={count}&output=atom"
        )
        return parse_atom_feed(await self.get_text(url))

    async def daily_form4_accessions(self, day: date) -> list["FilingRef"]:
        """Backfill discovery via the daily form index (form.YYYYMMDD.idx)."""
        q = (day.month - 1) // 3 + 1
        url = f"{SEC_WWW}/Archives/edgar/daily-index/{day.year}/QTR{q}/form.{day:%Y%m%d}.idx"
        try:
            text = await self.get_text(url)
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:  # weekend/holiday
                return []
            raise
        return parse_form_idx(text, form_type="4")

    # ------------------------------------------------------------ fetching

    async def fetch_filing_xml(self, ref: "FilingRef") -> str | None:
        """Fetch (and cache) the ownershipDocument XML for a filing."""
        cache = self.raw_dir / f"{ref.accession}.xml"
        if cache.exists():
            return cache.read_text(encoding="utf-8")
        index = await self.get_text(f"{ref.directory_url()}/index.json")
        for name in re.findall(r'"name"\s*:\s*"([^"]+\.xml)"', index):
            xml = await self.get_text(f"{ref.directory_url()}/{name}")
            if "<ownershipDocument" in xml:
                self.raw_dir.mkdir(parents=True, exist_ok=True)
                cache.write_text(xml, encoding="utf-8")
                return xml
        return None

    async def fetch_13f_xml(self, ref: "FilingRef") -> str | None:
        """Fetch (and cache) the informationTable XML for a 13F-HR filing."""
        cache = self.raw_dir / f"{ref.accession}-13f.xml"
        if cache.exists():
            return cache.read_text(encoding="utf-8")
        index = await self.get_text(f"{ref.directory_url()}/index.json")
        for name in re.findall(r'"name"\s*:\s*"([^"]+\.xml)"', index):
            xml = await self.get_text(f"{ref.directory_url()}/{name}")
            if "informationTable" in xml or "infoTable" in xml:
                self.raw_dir.mkdir(parents=True, exist_ok=True)
                cache.write_text(xml, encoding="utf-8")
                return xml
        return None

    async def fetch_submissions(self, cik: str) -> dict:
        """data.sec.gov submissions JSON: filing history incl. acceptance
        datetimes and report dates, the 13F discovery path."""
        import json as _json

        url = f"https://data.sec.gov/submissions/CIK{int(cik):010d}.json"
        return _json.loads(await self.get_text(url))

    async def fetch_acceptance_datetime(self, ref: "FilingRef") -> str | None:
        """ACCEPTANCE-DATETIME from the filing's index-headers page, this is
        the point-in-time anchor (first public availability)."""
        url = f"{ref.directory_url()}/{ref.accession}-index-headers.html"
        try:
            text = await self.get_text(url)
        except httpx.HTTPStatusError:
            return None
        m = re.search(r"ACCEPTANCE-DATETIME[^\d]*(\d{14})", text)
        if not m:
            return None
        s = m.group(1)
        return f"{s[0:4]}-{s[4:6]}-{s[6:8]} {s[8:10]}:{s[10:12]}:{s[12:14]}"


@dataclass(frozen=True)
class FilingRef:
    accession: str  # dashed form: 0001234567-24-000123
    cik: str

    def directory_url(self) -> str:
        return (
            f"{SEC_WWW}/Archives/edgar/data/{int(self.cik)}/"
            f"{self.accession.replace('-', '')}"
        )


def parse_atom_feed(xml_text: str) -> list[FilingRef]:
    """Entries look like .../Archives/edgar/data/CIK/ACCESSION-index.htm."""
    refs = []
    for m in re.finditer(
        r"Archives/edgar/data/(\d+)/(\d{10}-\d{2}-\d{6})-index", xml_text
    ):
        refs.append(FilingRef(accession=m.group(2), cik=m.group(1)))
    seen: set[str] = set()
    out = []
    for r in refs:
        if r.accession not in seen:
            seen.add(r.accession)
            out.append(r)
    return out


def parse_form_idx(text: str, form_type: str = "4") -> list[FilingRef]:
    """form.idx: fixed-width columns Form Type | Company | CIK | Date | File Name."""
    refs = []
    for line in text.splitlines():
        if not line.startswith(f"{form_type} ") and not line.startswith(f"{form_type}\t"):
            continue
        m = re.search(r"edgar/data/(\d+)/(\d{10})-(\d{2})-(\d{6})\.txt", line)
        if m:
            refs.append(
                FilingRef(accession=f"{m.group(2)}-{m.group(3)}-{m.group(4)}", cik=m.group(1))
            )
    return refs


def backfill_days(end: date, days: int) -> list[date]:
    return [end - timedelta(days=i) for i in range(days)]
