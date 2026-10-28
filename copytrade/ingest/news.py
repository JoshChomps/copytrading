"""Company-news ingestion: real providers only, used as CONTEXT, not signal.

Providers (both real, both optional):
  * Finnhub /company-news: needs the same free API key the congress
    aggregator uses; solid per-symbol coverage.
  * GDELT DOC 2.0 API: keyless, full-text news index since 2017; noisier
    (free-text query on company/ticker) but requires nothing.

What news is used for HERE: enriching live alerts and the dashboard so a
human can sanity-check a filing before acting on it ("did this insider buy
before or after the guidance cut?"). It is deliberately NOT a backtest
factor: free providers offer no survivorship-clean historical archive keyed
to tickers, and bolting a partial one onto the event study would be the kind
of quiet data corruption this project refuses. If a paid archive (RavenPack,
Tiingo news, GDELT BigQuery export) is added later, it plugs in via the same
`news` table with its provenance recorded per row.

Network reality (measured 2026-07): finnhub.io and api.gdeltproject.org are
both blocked from restricted sandboxes; both work from a normal machine.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone

import httpx

FINNHUB_NEWS_URL = "https://finnhub.io/api/v1/company-news"
GDELT_DOC_URL = "https://api.gdeltproject.org/api/v2/doc/doc"


@dataclass
class NewsItem:
    ticker: str
    headline: str
    url: str
    published_at: str          # ISO UTC
    source: str | None = None  # outlet
    provider: str = "unknown"  # finnhub | gdelt

    @property
    def uid(self) -> str:
        return hashlib.sha1(f"{self.ticker}|{self.url}".encode()).hexdigest()[:20]


def parse_finnhub_news(payload: list[dict], ticker: str) -> list[NewsItem]:
    """Finnhub company-news: [{datetime: epoch, headline, url, source}, ...]"""
    items = []
    for rec in payload or []:
        url = rec.get("url")
        headline = (rec.get("headline") or "").strip()
        ts = rec.get("datetime")
        if not url or not headline or not ts:
            continue
        items.append(NewsItem(
            ticker=ticker.upper(), headline=headline, url=url,
            published_at=datetime.fromtimestamp(int(ts), tz=timezone.utc)
                                 .strftime("%Y-%m-%d %H:%M:%S"),
            source=rec.get("source"), provider="finnhub",
        ))
    return items


def parse_gdelt_articles(payload: dict, ticker: str) -> list[NewsItem]:
    """GDELT DOC 2.0 artlist: {articles: [{url, title, seendate: YYYYMMDDTHHMMSSZ,
    domain}, ...]}"""
    items = []
    for rec in (payload or {}).get("articles", []):
        url = rec.get("url")
        title = (rec.get("title") or "").strip()
        seen = rec.get("seendate", "")
        if not url or not title or len(seen) < 15:
            continue
        try:
            published = datetime.strptime(seen, "%Y%m%dT%H%M%SZ").strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            continue
        items.append(NewsItem(
            ticker=ticker.upper(), headline=title, url=url,
            published_at=published, source=rec.get("domain"), provider="gdelt",
        ))
    return items


async def fetch_finnhub(client: httpx.AsyncClient, api_key: str, ticker: str,
                        from_date: str, to_date: str) -> list[NewsItem]:
    resp = await client.get(FINNHUB_NEWS_URL, params={
        "symbol": ticker, "from": from_date, "to": to_date, "token": api_key,
    })
    resp.raise_for_status()
    return parse_finnhub_news(resp.json(), ticker)


async def fetch_gdelt(client: httpx.AsyncClient, ticker: str, company_name: str | None = None,
                      max_records: int = 30) -> list[NewsItem]:
    """Keyless fallback. Query quotes the company name when known, a bare
    ticker like 'A' or 'ALL' is useless as full-text search."""
    q = f'"{company_name}"' if company_name else f'"{ticker}" stock'
    resp = await client.get(GDELT_DOC_URL, params={
        "query": q, "mode": "artlist", "format": "json",
        "maxrecords": max_records, "sort": "datedesc",
    })
    resp.raise_for_status()
    return parse_gdelt_articles(resp.json(), ticker)


def store_news(conn, items: list[NewsItem]) -> tuple[int, int]:
    inserted = deduped = 0
    for it in items:
        cur = conn.execute(
            "INSERT OR IGNORE INTO news(uid, ticker, headline, url, published_at, source, provider) "
            "VALUES (?,?,?,?,?,?,?)",
            (it.uid, it.ticker, it.headline, it.url, it.published_at, it.source, it.provider),
        )
        inserted += cur.rowcount
        deduped += 1 - cur.rowcount
    conn.commit()
    return inserted, deduped


def latest_news(conn, ticker: str, limit: int = 5) -> list[dict]:
    return [dict(r) for r in conn.execute(
        "SELECT headline, url, published_at, source, provider FROM news "
        "WHERE ticker=? ORDER BY published_at DESC LIMIT ?",
        (ticker.upper(), limit),
    )]


async def refresh_news(conn, tickers: list[str], api_key: str = "",
                       from_date: str = "", to_date: str = "") -> dict:
    """Refresh news for tickers: Finnhub when a key exists, GDELT otherwise.
    Best-effort per ticker; failures are counted and reported, never fatal."""
    inserted = deduped = failed = 0
    provider = "finnhub" if api_key else "gdelt"
    async with httpx.AsyncClient(timeout=30) as client:
        for t in tickers:
            try:
                if api_key:
                    items = await fetch_finnhub(client, api_key, t, from_date, to_date)
                else:
                    items = await fetch_gdelt(client, t)
                i, d = store_news(conn, items)
                inserted += i
                deduped += d
            except (httpx.HTTPError, ValueError):
                failed += 1
    return {"provider": provider, "tickers": len(tickers),
            "inserted": inserted, "deduped": deduped, "failed": failed}
