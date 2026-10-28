"""CFTC Commitments of Traders (COT), futures positioning of 'powerful people'.

No individual discloses a futures position, so the honest analogue of
"copy the smart money's futures bets" is the CFTC's weekly COT report: for
each futures market it breaks open interest into trader CATEGORIES
  * non-commercial  = large speculators (hedge funds, CTAs, managed money)
  * commercial      = hedgers (producers/users)
  * non-reportable  = small traders
The large-spec net position (long − short) is the closest public read on
where leveraged professionals are positioned.

Source: the CFTC publishes the legacy 'Futures Only' report as a fixed-format
text/CSV. This parser targets the legacy short-format CSV columns; validate
one live pull before trusting a bulk load (cftc.gov is unreachable from the
dev sandbox). Positioning is CONTEXT/framing, it is not fed into any priced
backtest (a weekly aggregate category is not a copyable trade).
"""

from __future__ import annotations

import csv
import io
import zipfile
from dataclasses import dataclass

import httpx

# The CFTC reorganized its file paths; the old deacot.txt now 404s. The stable
# current source is the annual legacy Futures-Only archive ZIP (contains
# annual.txt / annualof.txt in the fixed positional layout this parser reads).
# Verify the year on first live run, cftc.gov is unreachable from the dev
# sandbox, so this URL is a best-effort default like the other blocked sources.
LATEST_CSV = "https://www.cftc.gov/files/dea/history/deacot2026.zip"


@dataclass
class CotRow:
    report_date: str          # ISO
    market: str
    market_code: str | None
    noncomm_long: float | None
    noncomm_short: float | None
    comm_long: float | None
    comm_short: float | None
    nonrept_long: float | None
    nonrept_short: float | None
    open_interest: float | None

    @property
    def spec_net(self) -> float | None:
        if self.noncomm_long is None or self.noncomm_short is None:
            return None
        return self.noncomm_long - self.noncomm_short


def _f(v) -> float | None:
    try:
        return float(str(v).replace(",", "").strip())
    except (ValueError, AttributeError):
        return None


def _iso(yymmdd: str | None) -> str | None:
    """COT 'As of Date in Form YYYY-MM-DD' or YYMMDD -> ISO."""
    if not yymmdd:
        return None
    s = str(yymmdd).strip()
    if len(s) == 10 and s[4] == "-":
        return s
    if len(s) == 6 and s.isdigit():
        return f"20{s[0:2]}-{s[2:4]}-{s[4:6]}"
    return None


# Legacy short-format column indices (stable positional layout):
#  0 Market and Exchange Names
#  1 As of Date in Form YYMMDD
#  2 As of Date in Form YYYY-MM-DD
#  3 CFTC Contract Market Code
#  7 Open Interest (All)
#  8 Noncommercial Positions-Long (All)
#  9 Noncommercial Positions-Short (All)
# 11 Commercial Positions-Long (All)
# 12 Commercial Positions-Short (All)
# 15 Nonreportable Positions-Long (All)
# 16 Nonreportable Positions-Short (All)
def parse_cot_csv(text: str) -> list[CotRow]:
    rows: list[CotRow] = []
    reader = csv.reader(io.StringIO(text))
    for rec in reader:
        if len(rec) < 17:
            continue
        date = _iso(rec[2]) or _iso(rec[1])
        market = (rec[0] or "").strip().strip('"')
        if not date or not market or market.lower().startswith("market"):
            continue  # header/garbage
        rows.append(CotRow(
            report_date=date, market=market, market_code=(rec[3] or "").strip() or None,
            open_interest=_f(rec[7]),
            noncomm_long=_f(rec[8]), noncomm_short=_f(rec[9]),
            comm_long=_f(rec[11]), comm_short=_f(rec[12]),
            nonrept_long=_f(rec[15]), nonrept_short=_f(rec[16]),
        ))
    return rows


def store_cot(conn, rows: list[CotRow]) -> int:
    conn.executemany(
        "INSERT INTO cot_positioning(report_date, market, market_code, noncomm_long, "
        "noncomm_short, comm_long, comm_short, nonrept_long, nonrept_short, open_interest) "
        "VALUES (?,?,?,?,?,?,?,?,?,?) ON CONFLICT(report_date, market) DO UPDATE SET "
        "noncomm_long=excluded.noncomm_long, noncomm_short=excluded.noncomm_short, "
        "open_interest=excluded.open_interest",
        [(r.report_date, r.market, r.market_code, r.noncomm_long, r.noncomm_short,
          r.comm_long, r.comm_short, r.nonrept_long, r.nonrept_short, r.open_interest)
         for r in rows],
    )
    conn.commit()
    return len(rows)


def refresh_cot(conn, url: str = LATEST_CSV, markets: list[str] | None = None) -> dict:
    """Fetch the latest COT report and store it (optionally filtered to
    markets whose name contains one of `markets`, e.g. ['S&P 500', 'GOLD'])."""
    resp = httpx.get(url, timeout=60, follow_redirects=True,
                     headers={"User-Agent": "copytrade-research"})
    resp.raise_for_status()
    if url.lower().endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            txt = next((n for n in zf.namelist() if n.lower().endswith(".txt")), None)
            text = zf.read(txt).decode("utf-8", "replace") if txt else ""
    else:
        text = resp.text
    rows = parse_cot_csv(text)
    if markets:
        low = [m.lower() for m in markets]
        rows = [r for r in rows if any(m in r.market.lower() for m in low)]
    stored = store_cot(conn, rows)
    return {"rows": stored, "markets": len({r.market for r in rows}),
            "latest": max((r.report_date for r in rows), default=None)}
