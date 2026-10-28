"""SQLite access layer + migrations runner. Single file DB, WAL mode.

All writes are idempotent: filers upsert on (name, kind), trades are
INSERT OR IGNORE on UNIQUE(source, source_uid), prices upsert on
(ticker, date). Re-running any backfill is safe.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd

TRADE_COLS = (
    "filer_id source source_uid ticker cusip side tx_code instrument "
    "transaction_date filing_datetime amount_low amount_high shares price raw_ref"
).split()


def connect(db_path: str | Path) -> sqlite3.Connection:
    path = Path(db_path)
    if path.name != ":memory:":
        path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    # WAL allows concurrent readers, but two writers still contend, e.g. a
    # long ingest running alongside `prices` or `poll`. Wait for the lock
    # instead of failing the whole job on a transient conflict.
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


def run_migrations(conn: sqlite3.Connection, migrations_dir: str | Path = "migrations") -> list[str]:
    conn.execute(
        "CREATE TABLE IF NOT EXISTS schema_migrations "
        "(id TEXT PRIMARY KEY, applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
    )
    applied = {r[0] for r in conn.execute("SELECT id FROM schema_migrations")}
    ran = []
    for f in sorted(Path(migrations_dir).glob("*.sql")):
        if f.name in applied:
            continue
        conn.executescript(f.read_text(encoding="utf-8"))
        conn.execute("INSERT INTO schema_migrations(id) VALUES (?)", (f.name,))
        conn.commit()
        ran.append(f.name)
    return ran


# ---------------------------------------------------------------- filers ----

def resolve_alias(conn: sqlite3.Connection, name: str) -> str:
    """House data has messy name variants; map through the manual alias table."""
    row = conn.execute(
        "SELECT canonical FROM filers_aliases WHERE alias=?", (name.strip(),)
    ).fetchone()
    return row[0] if row else name.strip()


def upsert_filer(conn: sqlite3.Connection, name: str, kind: str, meta: dict | None = None) -> int:
    name = resolve_alias(conn, name)
    row = conn.execute(
        "SELECT filer_id FROM filers WHERE name=? AND kind=?", (name, kind)
    ).fetchone()
    if row:
        if meta:
            conn.execute(
                "UPDATE filers SET meta=? WHERE filer_id=?", (json.dumps(meta), row[0])
            )
        return row[0]
    cur = conn.execute(
        "INSERT INTO filers(name, kind, meta) VALUES (?,?,?)",
        (name, kind, json.dumps(meta or {})),
    )
    return cur.lastrowid


# ---------------------------------------------------------------- trades ----

def sanitize_transaction_date(transaction_date, filing_datetime):
    """Return (clean_transaction_date, rejection_reason_or_None).

    A PTR discloses a trade that has ALREADY happened, so a transaction date
    after its own filing timestamp is impossible. The House Clerk PDFs are OCR
    of scanned forms and produce exactly that: 3031-04-30 filed 2021-05-03,
    2220-04-07 filed 2022-05-04, and ~78 quieter day/month typos that land a
    few days past the filing.

    Such a date is nulled, never the whole trade. `filing_datetime` is the
    point-in-time execution anchor and is unaffected, so the trade stays
    copyable and stays in every backtest; it only drops out of the
    disclosure-lag and zero-lag decay statistics, which are the two things
    that actually read `transaction_date`.

    Long POSITIVE lags are explicitly NOT rejected. 603 trades in the live
    sample are disclosed more than two years late; those are genuine amended
    or delinquent filings, and lateness is the phenomenon this project
    measures rather than a defect to clean away.
    """
    if transaction_date is None or filing_datetime is None:
        return transaction_date, None
    td = pd.to_datetime(transaction_date, errors="coerce")
    fd = pd.to_datetime(filing_datetime, errors="coerce")
    if pd.isna(td):
        # Unparseable text is left as-is: `trades_frame` coerces it to NaT and
        # drops it from date-derived statistics anyway, and preserving the raw
        # string keeps the parser bug diagnosable from the DB.
        return transaction_date, None
    if pd.isna(fd):
        return transaction_date, None
    if td.normalize() > fd.normalize():
        return None, "after_filing"
    return transaction_date, None


def insert_trade(conn: sqlite3.Connection, **kw) -> bool:
    """Insert one normalized trade; returns True if it was new.

    Dedup: UNIQUE(source, source_uid). Cross-source dedup for congress
    records (same trade from two aggregators): match on
    (filer_id, ticker, side, transaction_date, amount_low) and keep the
    EARLIEST filing_datetime (updating in place if the new record is earlier).

    An impossible transaction date (after its own filing) is nulled here rather
    than in each parser: every source funnels through this function, so one
    guard covers House Clerk PDFs, the Senate feed, DERA bulk and the live
    scrapers alike. See `sanitize_transaction_date`.
    """
    vals = {c: kw.get(c) for c in TRADE_COLS}
    if vals["source_uid"] is None or vals["filing_datetime"] is None:
        raise ValueError("source_uid and filing_datetime are required")
    vals["transaction_date"], _ = sanitize_transaction_date(
        vals["transaction_date"], vals["filing_datetime"])

    dup = conn.execute(
        "SELECT trade_id, filing_datetime, source FROM trades WHERE filer_id=? AND ticker IS ? "
        "AND side IS ? AND transaction_date IS ? AND amount_low IS ? AND source != ?",
        (vals["filer_id"], vals["ticker"], vals["side"], vals["transaction_date"],
         vals["amount_low"], vals["source"]),
    ).fetchone()
    if dup is not None:
        if str(vals["filing_datetime"]) < str(dup["filing_datetime"]):
            conn.execute(
                "UPDATE trades SET filing_datetime=? WHERE trade_id=?",
                (vals["filing_datetime"], dup["trade_id"]),
            )
        return False

    cols = ",".join(TRADE_COLS)
    marks = ",".join("?" * len(TRADE_COLS))
    cur = conn.execute(
        f"INSERT OR IGNORE INTO trades ({cols}) VALUES ({marks})",
        [vals[c] for c in TRADE_COLS],
    )
    return cur.rowcount > 0


def trades_frame(
    conn: sqlite3.Connection,
    kinds: list[str] | None = None,
    sides: list[str] | None = None,
    instruments: list[str] | None = ("stock",),
    since: str | None = None,
) -> pd.DataFrame:
    """Trades joined to filers, parsed dates, ready for the backtester.

    since: ISO date lower bound on filing_datetime (the point-in-time anchor),
    so a recent-era slice keeps the same look-ahead discipline, a trade
    enters the window by when it became ACTIONABLE, never by transaction date.
    """
    q = (
        "SELECT t.*, f.name AS filer_name, f.kind AS filer_kind, "
        "json_extract(f.meta, '$.party') AS filer_party "
        "FROM trades t JOIN filers f USING(filer_id) WHERE 1=1"
    )
    args: list = []
    for col, values in (("f.kind", kinds), ("t.side", sides), ("t.instrument", instruments)):
        if values:
            q += f" AND {col} IN ({','.join('?' * len(values))})"
            args.extend(values)
    if since:
        q += " AND t.filing_datetime >= ?"
        args.append(since)
    df = pd.read_sql_query(q, conn, params=args)
    if df.empty:
        return df
    # errors="coerce": a stray malformed date (e.g. a 2-digit-year DERA row)
    # becomes NaT instead of crashing the whole frame; rows without a usable
    # filing anchor can't be executed point-in-time, so they're dropped.
    df["transaction_date"] = pd.to_datetime(df["transaction_date"], errors="coerce")
    df["filing_datetime"] = pd.to_datetime(df["filing_datetime"], errors="coerce")
    df = df[df["filing_datetime"].notna()].reset_index(drop=True)
    if df.empty:
        return df
    df["amount_mid"] = (df["amount_low"].fillna(0) + df["amount_high"].fillna(0)) / 2
    exact = df["shares"].notna() & df["price"].notna()
    if exact.any():  # all-False .loc assignment raises under pandas 3
        df.loc[exact, "amount_mid"] = df.loc[exact, "shares"] * df.loc[exact, "price"]
    return df


# ---------------------------------------------------------------- prices ----

def upsert_prices(conn: sqlite3.Connection, df: pd.DataFrame) -> int:
    """df columns: ticker, date, adj_close.

    Non-positive closes are REJECTED, not stored. An adjusted close of zero or
    less is not a price, it is a provider error, yfinance emits them for some
    thinly-traded foreign OTC ADRs. They are catastrophic downstream because
    returns are computed as p_t/p_0 - 1: a negative p_t flips the sign of the
    ratio and yields impossible values (one ticker produced a -587,200% return
    and single-handedly dragged the whole-sample mean to -310%).
    """
    rows = [
        (str(r.ticker), pd.Timestamp(r.date).strftime("%Y-%m-%d"), float(r.adj_close))
        for r in df.itertuples()
        if pd.notna(r.adj_close) and float(r.adj_close) > 0
    ]
    conn.executemany(
        "INSERT INTO prices(ticker, date, adj_close) VALUES (?,?,?) "
        "ON CONFLICT(ticker, date) DO UPDATE SET adj_close=excluded.adj_close",
        rows,
    )
    conn.commit()
    return len(rows)


def load_price_matrix(conn: sqlite3.Connection, tickers: list[str] | None = None,
                      include_quarantined: bool = False) -> pd.DataFrame:
    """Wide DataFrame of adjusted closes: index=date, columns=ticker.

    Quarantined tickers are dropped here rather than in each analysis, because
    this is the one function every analysis reads prices through, the same
    single-choke-point argument that governs `execution_position`. A series
    that is positive but not credible (an unadjusted reverse split, a sub-penny
    artifact, a multi-year gap read as one session) corrupts any statistic
    computed over it, and the offenders are already known by name: see
    `store.price_qa`. Their trades then land in the existing "unpriceable"
    bucket, which every report counts and discloses.

    `include_quarantined=True` is for inspecting the excluded series itself.
    """
    q = "SELECT ticker, date, adj_close FROM prices"
    args: list = []
    if tickers:
        q += f" WHERE ticker IN ({','.join('?' * len(tickers))})"
        args = list(tickers)
    df = pd.read_sql_query(q, conn, params=args)
    if df.empty:
        return pd.DataFrame()
    if not include_quarantined:
        from copytrade.store.price_qa import quarantined_tickers

        bad = quarantined_tickers(conn)
        if bad:
            df = df[~df["ticker"].isin(bad)]
            if df.empty:
                return pd.DataFrame()
    df["date"] = pd.to_datetime(df["date"])
    return df.pivot(index="date", columns="ticker", values="adj_close").sort_index()


# ----------------------------------------------------------------- state ----

def get_state(conn: sqlite3.Connection, key: str, default: str | None = None) -> str | None:
    row = conn.execute("SELECT value FROM poll_state WHERE key=?", (key,)).fetchone()
    return row[0] if row else default


def set_state(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO poll_state(key, value) VALUES (?,?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (key, value),
    )
    conn.commit()
