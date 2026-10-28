"""Senate Stock Watcher ingestion, real PTR data with real disclosure dates.

Source: github.com/timothycarambat/senate-stock-watcher-data. The per-day
report files (data/transaction_report_for_MM_DD_YYYY.json) carry
`date_recieved` (sic): the date efdsearch.senate.gov received the PTR. That
is the point-in-time anchor; the aggregate/all_transactions.json file does
NOT have it and is therefore useless for honest backtesting.

Coverage & staleness (verified 2026-07): ~Dec 2012 → Mar 2021, ~8.3k
transactions, 66 senators. The repo stopped updating in March 2021, so this
source is for BACKTESTING; live signals need Finnhub/QuiverQuant or scraping
efdsearch directly (see docs/CANADA.md "Live data obstacles").

Real-world quirks handled here (all observed in the actual files):
  * tickers wrapped in HTML: '<a href="...">BA</a>', or '--' / '' for
    non-ticker assets
  * amounts as bucket strings: '$1,001 - $15,000' ... 'Over $50,000,000'
  * types: Purchase | Sale (Full) | Sale (Partial) | Exchange
  * asset_type: Stock | Stock Option | Municipal Security | ... (only Stock
    becomes instrument='stock'; everything else is stored, flagged, excluded
    from backtests)
  * amended PTRs re-report identical rows with a later date_recieved, dedup
    keeps the EARLIEST receipt (files are ingested in chronological order);
    identical duplicate rows within one PTR are preserved via an occurrence
    counter. Amendments that change the amount bucket cannot be linked and
    remain as separate rows (documented limitation).
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path

from copytrade.ingest.congress import CongressTrade, store_congress_trades

REPO_URL = "https://github.com/timothycarambat/senate-stock-watcher-data.git"
REPORT_GLOB = "data/transaction_report_for_*.json"

_TICKER_HTML = re.compile(r">([^<>]{1,40})<")
_TICKER_OK = re.compile(r"[A-Z][A-Z.\-]{0,9}$")
_AMOUNT = re.compile(r"\$([\d,]+)")


def clean_ticker(raw: str | None) -> str | None:
    """'<a href=...>BA</a>' -> 'BA'; '--', '', 'N/A' -> None.

    The extracted text must still LOOK like a ticker: real senate filings
    contain HTML-wrapped OPTION symbols ('SPY160219P00180000') which
    previously slipped through as raw HTML and polluted the price fetch
    anything that fails the ticker shape is rejected, not passed along."""
    if not raw:
        return None
    s = raw.strip()
    m = _TICKER_HTML.search(s)
    if m:
        s = m.group(1)
    elif "<" in s or ">" in s:
        return None  # HTML we couldn't extract from, never return markup
    s = s.strip().upper()
    if not s or s in ("--", "N/A") or not _TICKER_OK.fullmatch(s):
        return None
    return s


def parse_amount(raw: str | None) -> tuple[float | None, float | None]:
    """'$1,001 - $15,000' -> (1001, 15000); 'Over $50,000,000' -> (5e7, None)."""
    if not raw:
        return None, None
    nums = [float(n.replace(",", "")) for n in _AMOUNT.findall(raw)]
    if not nums:
        return None, None
    if raw.strip().lower().startswith("over") or len(nums) == 1:
        return nums[0], (None if len(nums) == 1 else nums[1])
    return nums[0], nums[1]


def _side(tx_type: str | None) -> str | None:
    t = (tx_type or "").lower()
    if "purchase" in t:
        return "buy"
    if "sale" in t:
        return "sell"
    return None  # Exchange / unknown: stored unsided, never a signal


def _instrument(asset_type: str | None) -> str:
    a = (asset_type or "").strip().lower()
    if a == "stock":
        return "stock"
    if not a:
        return "unknown"
    return a.replace(" ", "_")  # stock_option, municipal_security, ...


def _mdY_to_iso(s: str | None) -> str | None:
    if not s:
        return None
    try:
        return datetime.strptime(s.strip(), "%m/%d/%Y").strftime("%Y-%m-%d")
    except ValueError:
        return None


def parse_report_payload(payload: list[dict]) -> tuple[list[CongressTrade], int]:
    """One per-day report file -> normalized trades.

    Skips (counted, never guessed): missing date_recieved (no point-in-time
    anchor), missing/unmappable ticker, missing transaction date.
    """
    trades: list[CongressTrade] = []
    skipped = 0
    for entry in payload:
        filing = _mdY_to_iso(entry.get("date_recieved"))
        name = " ".join(
            p for p in (entry.get("first_name", "").strip(), entry.get("last_name", "").strip()) if p
        ).title()
        occurrence: dict[str, int] = {}
        for tx in entry.get("transactions", []):
            ticker = clean_ticker(tx.get("ticker"))
            tx_date = _mdY_to_iso(tx.get("transaction_date"))
            if not filing or not name or not ticker or not tx_date:
                skipped += 1
                continue
            lo, hi = parse_amount(tx.get("amount"))
            side = _side(tx.get("type"))
            key = json.dumps([name, ticker, side, tx_date, lo, hi], sort_keys=True)
            k = occurrence.get(key, 0)
            occurrence[key] = k + 1
            trades.append(
                CongressTrade(
                    name=name,
                    chamber="Senate",
                    ticker=ticker,
                    side=side,
                    transaction_date=tx_date,
                    filing_date=filing,
                    amount_low=lo,
                    amount_high=hi,
                    instrument=_instrument(tx.get("asset_type")),
                    source="senate_watcher",
                    raw=tx,
                    # uid excludes filing date -> amended re-reports dedup to
                    # the earliest receipt; `k` preserves true same-PTR dupes
                    uid=hashlib.sha1(key.encode()).hexdigest()[:18] + f":{k}",
                )
            )
    return trades, skipped


def _report_date(path: Path):
    # transaction_report_for_MM_DD_YYYY.json -> sortable (YYYY, MM, DD)
    m = re.search(r"for_(\d{2})_(\d{2})_(\d{4})", path.name)
    return (m.group(3), m.group(1), m.group(2)) if m else ("9999", "99", "99")


def clone_or_pull(dest: str | Path, repo_url: str = REPO_URL) -> Path:
    """Shallow-clone (or update) the data repo. Requires outbound git to
    github.com: this is the one congress source that works even from
    restricted environments where sec.gov/finnhub are blocked."""
    dest = Path(dest)
    if (dest / ".git").exists():
        subprocess.run(["git", "-C", str(dest), "pull", "--ff-only", "--depth", "1"],
                       check=True, capture_output=True, text=True, timeout=300)
    else:
        dest.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--depth", "1", repo_url, str(dest)],
                       check=True, capture_output=True, text=True, timeout=600)
    return dest


def ingest_repo(conn, repo_dir: str | Path) -> dict:
    """Parse every per-day report chronologically and store. Returns counts."""
    repo_dir = Path(repo_dir)
    files = sorted(repo_dir.glob(REPORT_GLOB), key=_report_date)
    if not files:
        raise FileNotFoundError(f"no {REPORT_GLOB} under {repo_dir}, wrong repo?")
    inserted = deduped = skipped = 0
    latest: str | None = None
    for f in files:
        trades, s = parse_report_payload(json.loads(f.read_text(encoding="utf-8")))
        skipped += s
        i, d = store_congress_trades(conn, trades)
        inserted += i
        deduped += d
        for t in trades:
            if latest is None or t.filing_date > latest:
                latest = t.filing_date
    return {
        "files": len(files),
        "inserted": inserted,
        "deduped": deduped,
        "skipped_unusable": skipped,
        "latest_filing": latest,
        "staleness_warning": (
            f"newest filing is {latest}; this dataset stopped updating in 2021, "
            "backtest-grade only, see docs/CANADA.md for live options"
            if latest and latest < "2025-01-01" else None
        ),
    }
