"""Market data: daily adjusted closes with a three-provider fallback chain.

    1. yfinance: keyless, batch, dividend+split adjusted. Primary, but the
                   flakiest from datacenter IPs (CI runners get throttled).
    2. Tiingo: with COPYTRADE_TIINGO_API_KEY: per-symbol REST, `adjClose`
                   is dividend+split adjusted (same quality as yfinance).
    3. stooq: keyless CSV of last resort. SPLIT-adjusted only (no
                   dividend adjustment), so total returns are slightly
                   understated for payers; each stooq-sourced ticker is
                   reported in the refresh summary rather than silently mixed.

Delisted-ticker handling: tickers no provider can price are flagged
unpriceable and the count is disclosed in every report (survivorship-bias
disclosure).
"""

from __future__ import annotations

import csv
import io

import pandas as pd

from copytrade.store import db as store

# Latches once Tiingo returns 429. The free-tier allocation is hourly and
# ACCOUNT-wide, so after the first throttle every later chunk in the same
# run is guaranteed to fail too, retrying just burns the remaining passes.
# Reset to retry within the same process.
_TIINGO_THROTTLED = {"hit": False}

TIINGO_URL = "https://api.tiingo.com/tiingo/daily/{ticker}/prices"
STOOQ_URL = "https://stooq.com/q/d/l/?s={symbol}&i=d&d1={d1}&d2={d2}"


def fetch_yfinance(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Long DataFrame (ticker, date, adj_close). Missing tickers just absent."""
    import yfinance as yf

    if not tickers:
        return _empty()
    # Query the provider's symbol but key the result by the DISCLOSED ticker,
    # so class shares (BRK.B -> BRK-B) resolve without the rest of the pipeline
    # having to know about provider-specific spellings.
    sym_to_orig = {provider_symbol(t): t for t in tickers}
    data = yf.download(
        tickers=list(sym_to_orig), start=start, end=end,
        auto_adjust=True, progress=False, group_by="ticker", threads=True,
    )
    frames = []
    if isinstance(data.columns, pd.MultiIndex):
        have = set(data.columns.get_level_values(0))
        for sym, orig in sym_to_orig.items():
            if sym in have:
                s = data[sym]["Close"].dropna()
                if len(s):
                    frames.append(pd.DataFrame({"ticker": orig, "date": s.index,
                                                "adj_close": s.values}))
    elif "Close" in data.columns and len(sym_to_orig) == 1:
        s = data["Close"].dropna()
        frames.append(pd.DataFrame({"ticker": next(iter(sym_to_orig.values())),
                                    "date": s.index, "adj_close": s.values}))
    return pd.concat(frames, ignore_index=True) if frames else _empty()


def parse_tiingo_payload(payload: list[dict], ticker: str) -> pd.DataFrame:
    """Tiingo daily prices: [{date: ISO, adjClose: float, ...}, ...]."""
    rows = [
        {"ticker": ticker, "date": r["date"][:10], "adj_close": float(r["adjClose"])}
        for r in (payload or [])
        if r.get("date") and r.get("adjClose") is not None
    ]
    return pd.DataFrame(rows) if rows else _empty()


def fetch_tiingo(tickers: list[str], start: str, end: str, api_key: str) -> pd.DataFrame:
    import httpx

    frames = []
    # A 429 (free-tier hourly allocation) or 401 (bad key) is NOT the same
    # thing as "this ticker does not exist", but the response looks identical
    # to the caller once it is swallowed. Recording them separately keeps a
    # throttled run from being written up as a set of delistings, the same
    # failure mode that had yfinance dropping live mega-caps into the
    # unpriceable pile.
    fetch_tiingo.last_errors = {}
    with httpx.Client(timeout=30, headers={"Authorization": f"Token {api_key}"}) as client:
        for t in tickers:
            try:
                # request the provider's symbol, store under the DISCLOSED one
                resp = client.get(TIINGO_URL.format(ticker=provider_symbol(t).lower()),
                                  params={"startDate": start, "endDate": end})
                if resp.status_code != 200:
                    if resp.status_code in (401, 403, 429):
                        fetch_tiingo.last_errors[t] = resp.status_code
                        if resp.status_code == 429:
                            # the allocation is per-hour and account-wide, so
                            # every remaining ticker will fail too
                            break
                    continue
                df = parse_tiingo_payload(resp.json(), t)
                if len(df):
                    frames.append(df)
            except httpx.HTTPError:
                continue
    return pd.concat(frames, ignore_index=True) if frames else _empty()


# Suffixes that mark something which is NOT an ordinary US common share, so a
# dot-to-dash rewrite would fetch the wrong instrument (or nothing) rather than
# fixing the symbol. Warrants, units, foreign listings and preferreds.
_NON_EQUITY_SUFFIXES = {
    "W", "WS", "U", "V", "PL", "P", "SW", "AS", "F", "L", "TO", "HK", "DE",
}


def provider_symbol(ticker: str) -> str:
    """Normalise a disclosed ticker for yfinance / Tiingo.

    US class shares are disclosed with a dot (BRK.B, BF.B, HEI.A) but both
    providers key them with a dash (BRK-B). Without this, every class share
    silently 404s and lands in the unpriceable pile alongside genuine
    delistings: Berkshire B was missing from the price matrix for exactly this
    reason.

    Only single-letter class suffixes are rewritten. `RDS.A` looks identical in
    shape but is a foreign ADR line, and `ETWO.W` / `PCPL.U` are warrants and
    units, so anything in `_NON_EQUITY_SUFFIXES` is left alone to fail honestly
    rather than being mapped onto a different security.
    """
    if "." not in ticker:
        return ticker
    head, _, tail = ticker.rpartition(".")
    if tail.upper() in _NON_EQUITY_SUFFIXES:
        return ticker
    if len(tail) == 1 and tail.isalpha():
        return f"{head}-{tail}".upper()
    return ticker


def stooq_symbol(ticker: str) -> str:
    """US tickers on stooq: lowercase, dots to hyphens, '.us' suffix
    (BRK.B -> brk-b.us)."""
    return ticker.lower().replace(".", "-") + ".us"


def parse_stooq_csv(text: str, ticker: str) -> pd.DataFrame:
    """Stooq daily CSV: Date,Open,High,Low,Close,Volume (Close split-adjusted)."""
    rows = []
    for r in csv.DictReader(io.StringIO(text)):
        d, c = r.get("Date"), r.get("Close")
        if not d or not c:
            continue
        try:
            rows.append({"ticker": ticker, "date": d, "adj_close": float(c)})
        except ValueError:
            continue
    return pd.DataFrame(rows) if rows else _empty()


def fetch_stooq(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    import httpx

    d1, d2 = start.replace("-", ""), end.replace("-", "")
    frames = []
    with httpx.Client(timeout=30, follow_redirects=True) as client:
        for t in tickers:
            try:
                resp = client.get(STOOQ_URL.format(symbol=stooq_symbol(t), d1=d1, d2=d2))
                if resp.status_code != 200 or resp.text.startswith("No data"):
                    continue
                df = parse_stooq_csv(resp.text, t)
                if len(df):
                    frames.append(df)
            except httpx.HTTPError:
                continue
    return pd.concat(frames, ignore_index=True) if frames else _empty()


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=["ticker", "date", "adj_close"])


def fetch_daily_closes(tickers: list[str], start: str, end: str,
                       tiingo_api_key: str = "") -> tuple[pd.DataFrame, dict]:
    """Provider chain; returns (long df, provenance {provider: [tickers]}).

    `_TIINGO_THROTTLED` latches once the account-wide hourly allocation is
    hit, so later chunks skip Tiingo instead of collecting guaranteed 429s.
    """
    provenance: dict[str, list[str]] = {}
    got: set[str] = set()
    frames = []

    def take(df: pd.DataFrame, provider: str) -> None:
        if not len(df):
            return
        new = sorted(set(df["ticker"]) - got)
        if new:
            frames.append(df[df["ticker"].isin(new)])
            got.update(new)
            provenance[provider] = new

    try:
        take(fetch_yfinance(tickers, start, end), "yfinance")
    except Exception as e:  # noqa: BLE001, a dead primary must not kill the chain
        provenance["yfinance_error"] = [str(e)[:200]]
    missing = [t for t in tickers if t not in got]
    # The Tiingo allocation is hourly and ACCOUNT-wide, so once one chunk is
    # throttled every later chunk in the same run is too. Skipping them keeps a
    # long backfill from spending its remaining passes on guaranteed 429s.
    if missing and tiingo_api_key and not _TIINGO_THROTTLED["hit"]:
        take(fetch_tiingo(missing, start, end, tiingo_api_key), "tiingo")
        errs = getattr(fetch_tiingo, "last_errors", {}) or {}
        if 429 in errs.values():
            _TIINGO_THROTTLED["hit"] = True
        if errs:
            codes = sorted(set(errs.values()))
            provenance["tiingo_error"] = [
                f"HTTP {c} on {sum(1 for v in errs.values() if v == c)} ticker(s)"
                + (": free-tier hourly allocation exhausted; these are THROTTLED,"
                   " not delisted, and will resolve on a later run"
                   if c == 429 else " — check the API key")
                for c in codes
            ]
        missing = [t for t in tickers if t not in got]
    if missing:
        take(fetch_stooq(missing, start, end), "stooq_split_adjusted_only")
    df = pd.concat(frames, ignore_index=True) if frames else _empty()
    return df, provenance


def _fetch_in_chunks(conn, missing: list[str], start: str, end: str,
                     tiingo_api_key: str = "", chunk: int = 120,
                     passes: int = 3, pause: float = 2.0) -> tuple[int, dict]:
    """Fetch `missing` in small batches, upserting as we go, then retry what
    is still absent on later passes with a growing pause.

    Repeated passes matter because a rate-limit rejection is transient and
    indistinguishable from a genuine delisting in the response, the only way
    to tell them apart is to ask again later. Whatever is still missing after
    the final pass is treated as genuinely unpriceable.
    """
    import time

    total_rows = 0
    provenance: dict[str, list[str]] = {}
    todo = list(missing)
    for p in range(passes):
        if not todo:
            break
        still: list[str] = []
        n_batches = (len(todo) + chunk - 1) // chunk
        for b, i in enumerate(range(0, len(todo), chunk)):
            batch = todo[i:i + chunk]
            failed = False
            try:
                df, prov = fetch_daily_closes(batch, start, end, tiingo_api_key)
            except Exception as e:  # noqa: BLE001, one bad batch must not end the run
                provenance.setdefault("batch_error", []).append(str(e)[:120])
                still.extend(batch)
                df, prov, failed = None, {}, True
            if not failed:
                # Merge provenance even when the frame is EMPTY. Doing it only
                # on success discarded `yfinance_error` precisely when a chunk
                # returned nothing, so a provider outage was reported as an
                # empty result and those tickers surfaced in `unpriceable`
                # which report.py publishes as a delisting/survivorship caveat.
                for k, v in prov.items():
                    provenance.setdefault(k, []).extend(v)
                if len(df):
                    total_rows += store.upsert_prices(conn, df)
                    got = set(df["ticker"])
                else:
                    got = set()
                still.extend([t for t in batch if t not in got])
            # Back off on the FAILURE path too, that is the path the backoff
            # exists for. Sleeping only after a success meant a rate-limited
            # run re-fired every remaining chunk instantly and burned all three
            # passes against the same wall. Skip the sleep after the last chunk
            # of the last pass, where it only delays the return.
            last = (p == passes - 1) and (b == n_batches - 1)
            if not last:
                time.sleep(pause)
        todo = still
        pause *= 2  # back off harder before each retry pass
    if todo:
        provenance["unresolved_after_retries"] = todo
    return total_rows, provenance


def refresh_prices(conn, start: str, end: str, extra_tickers: list[str] = ("SPY", "TLT"),
                   tiingo_api_key: str = "") -> dict:
    """Ensure every traded ticker + benchmarks have cached history."""
    traded = [
        r[0] for r in conn.execute(
            "SELECT DISTINCT ticker FROM trades WHERE ticker IS NOT NULL AND instrument='stock'"
        )
    ]
    wanted = sorted(set(traded) | set(extra_tickers))
    have = {
        r[0] for r in conn.execute(
            "SELECT ticker FROM prices GROUP BY ticker HAVING COUNT(*) > 50"
        )
    }
    missing = [t for t in wanted if t not in have]
    fetched_rows = 0
    provenance: dict = {}
    if missing:
        # Chunked, with backoff. yf.download() fires every ticker in one
        # threaded burst; at universe scale (2000+ names) Yahoo rate-limits
        # partway through and returns NOTHING for the rest of the batch. That
        # failure is silent, the tickers simply come back absent, and it
        # drops live mega-caps, not just delisted ones, which would bias every
        # downstream result. Chunking keeps each request small enough to
        # survive, and results are committed per chunk so a late failure never
        # discards earlier work.
        fetched_rows, provenance = _fetch_in_chunks(
            conn, missing, start, end, tiingo_api_key
        )
    priced = {
        r[0] for r in conn.execute("SELECT DISTINCT ticker FROM prices")
    }
    unpriceable = sorted(set(traded) - priced)
    return {
        "wanted": len(wanted),
        "fetched_rows": fetched_rows,
        "providers": {k: (v if k.endswith("error") else len(v)) for k, v in provenance.items()},
        "stooq_dividend_caveat": provenance.get("stooq_split_adjusted_only"),
        "unpriceable": unpriceable,
    }
