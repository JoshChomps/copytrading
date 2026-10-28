"""Daily USD/CAD FX rates, real data, two real sources, no synthesis.

Primary: Bank of Canada Valet API (official noon/indicative rates, FXUSDCAD).
Fallback: the frictionless-data mirror of the US Fed H.10 release
(github.com/datasets/exchange-rates, data/daily.csv), verified live
(updated within the week as of 2026-07) and reachable even from
git/raw-github-only environments where bankofcanada.ca is blocked.

Rates are stored in fx_rates(date, pair, rate) with pair='USDCAD'
(CAD per 1 USD) and the source recorded per row.
"""

from __future__ import annotations

import csv
import io

import httpx

BOC_URL = (
    "https://www.bankofcanada.ca/valet/observations/FXUSDCAD/json?start_date={start}"
)
FED_MIRROR_URL = (
    "https://raw.githubusercontent.com/datasets/exchange-rates/master/data/daily.csv"
)


def fetch_boc(start: str, timeout: float = 30.0) -> list[tuple[str, float, str]]:
    """Bank of Canada Valet: [(YYYY-MM-DD, rate, 'boc_valet'), ...]."""
    resp = httpx.get(BOC_URL.format(start=start), timeout=timeout)
    resp.raise_for_status()
    out = []
    for obs in resp.json().get("observations", []):
        v = obs.get("FXUSDCAD", {}).get("v")
        if v is not None:
            out.append((obs["d"], float(v), "boc_valet"))
    return out


def fetch_fed_mirror(start: str, timeout: float = 60.0) -> list[tuple[str, float, str]]:
    """Fed H.10 mirror CSV (Date,Country,Exchange rate), Canada rows only."""
    resp = httpx.get(FED_MIRROR_URL, timeout=timeout, follow_redirects=True)
    resp.raise_for_status()
    out = []
    for row in csv.reader(io.StringIO(resp.text)):
        if len(row) >= 3 and row[1] == "Canada" and row[0] >= start:
            try:
                out.append((row[0], float(row[2]), "fed_h10_mirror"))
            except ValueError:
                continue
    return out


def refresh_usdcad(conn, start: str = "2012-01-01") -> dict:
    """Try BoC first (official), fall back to the Fed mirror. Stores rows,
    returns provenance + coverage so reports can disclose the source."""
    rows: list[tuple[str, float, str]] = []
    errors: dict[str, str] = {}
    try:
        rows = fetch_boc(start)
    except Exception as e:  # noqa: BLE001, provenance matters more than the type
        errors["boc_valet"] = str(e)
    if not rows:
        try:
            rows = fetch_fed_mirror(start)
        except Exception as e:  # noqa: BLE001
            errors["fed_h10_mirror"] = str(e)
    if not rows:
        raise RuntimeError(f"no FX source reachable: {errors}")
    conn.executemany(
        "INSERT INTO fx_rates(date, pair, rate, source) VALUES (?,?,?,?) "
        "ON CONFLICT(date, pair) DO UPDATE SET rate=excluded.rate, source=excluded.source",
        [(d, "USDCAD", r, s) for d, r, s in rows],
    )
    conn.commit()
    return {
        "rows": len(rows),
        "source": rows[-1][2],
        "first": rows[0][0],
        "last": rows[-1][0],
        "source_errors": errors or None,
    }
