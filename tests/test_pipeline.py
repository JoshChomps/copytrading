"""End-to-end on a small synthetic dataset with KNOWN ground truth:
skilled fast-filers must out-rank unskilled ones, the walk-forward must run,
and 13F deltas must produce sided trades."""

import pandas as pd

from copytrade.backtest.costs import get_cost_model
from copytrade.backtest.engine import walk_forward
from copytrade.demo.synthetic import generate
from copytrade.ingest.thirteenf import parse_13f_infotable, quarterly_deltas, store_13f
from copytrade.rank.rank import build_ranking
from copytrade.store import db as store
from tests.conftest import FIXTURES

ZERO = get_cost_model("zero")


def _seed(conn):
    return generate(conn, seed=7, start="2019-01-01", end="2022-12-31",
                    n_tickers=12, congress_trades_per_year=16, insider_trades_per_year=16)


def test_demo_generate_and_rank(conn):
    summary = _seed(conn)
    assert summary["trades"] > 500

    trades = store.trades_frame(conn, instruments=["stock"])
    prices = store.load_price_matrix(conn)
    bench = prices["SPY"].dropna()
    ranking, diag = build_ranking(trades, prices, bench, ZERO,
                                  horizon=63, min_trades_rank=30)
    assert diag["unpriceable"] > 0  # the delisted-ticker trades are disclosed
    assert not ranking.empty

    # ground truth: insiders CXO-00..03 are skilled AND file within 2 days;
    # CXO-06..09 have zero skill. Skilled must out-rank unskilled on average.
    skilled = ranking[ranking["filer"].str.contains("CXO-0[0-3]")]["lag_alpha_63d"].mean()
    unskilled = ranking[ranking["filer"].str.contains("CXO-0[6-9]")]["lag_alpha_63d"].mean()
    assert skilled > unskilled
    assert skilled > 0.01  # 2-day lag preserves most of the injected 6%

    # lag destruction: slow-filing skilled congress keeps less than skilled insiders
    slow_congress = ranking[ranking["filer"].str.contains("Demo-01|Demo-02")]
    if not slow_congress.empty:
        assert slow_congress["lag_alpha_63d"].mean() < skilled


def test_walk_forward_runs_and_reports(conn):
    _seed(conn)
    trades = store.trades_frame(conn, instruments=["stock"])
    prices = store.load_price_matrix(conn)
    bench = prices["SPY"].dropna()
    wf = walk_forward(trades, prices, bench, ZERO, horizon=63, top_n=5,
                      min_trades=3, max_weight=0.05)
    assert wf is not None
    assert len(wf.equity) > 200
    assert wf.equity.iloc[0] == 1.0
    assert not wf.taxlots.empty
    assert set(wf.taxlots.columns) >= {"ticker", "entry_date", "exit_date", "pnl"}
    assert wf.picks  # at least one test year got picks


def test_13f_deltas(conn):
    filer_id = store.upsert_filer(conn, "Demo Fund LP", "fund", {"cik": "0000000000"})
    holdings = parse_13f_infotable((FIXTURES / "infotable_13f.xml").read_text())
    store_13f(conn, filer_id, "2024-03-31", holdings)
    # next quarter: AAPL trimmed, MSFT added to, new position appears unmapped
    q2 = [h for h in holdings]
    q2[0].shares = 4_000_000   # AAPL -1M
    q2[1].shares = 1_500_000   # MSFT +0.5M
    store_13f(conn, filer_id, "2024-06-30", q2)
    conn.execute("INSERT INTO cusip_map(cusip, ticker) VALUES ('037833100','AAPL'), ('594918104','MSFT')")

    inserted, unmapped = quarterly_deltas(conn, filer_id, "2024-03-31", "2024-06-30",
                                          "2024-08-14 16:00:00")
    assert inserted == 2 and unmapped == 0
    rows = {r["ticker"]: r["side"] for r in conn.execute(
        "SELECT ticker, side FROM trades WHERE source='edgar_13f'")}
    assert rows == {"AAPL": "sell", "MSFT": "buy"}
    anchor = conn.execute(
        "SELECT filing_datetime FROM trades WHERE source='edgar_13f' LIMIT 1").fetchone()[0]
    assert anchor == "2024-08-14 16:00:00"  # anchored at filing, not quarter end


def test_migrations_idempotent(conn):
    from tests.conftest import ROOT
    assert store.run_migrations(conn, ROOT / "migrations") == []  # second run: no-op


def test_trades_frame_without_exact_amounts(conn):
    """Regression: an all-False shares&price mask (every senate row) crashed
    trades_frame under pandas 3."""
    from copytrade.store import db as store

    f = store.upsert_filer(conn, "Sen. NoShares", "congress", None)
    store.insert_trade(conn, filer_id=f, source="senate_watcher", source_uid="x1",
                       ticker="BA", side="buy", instrument="stock",
                       transaction_date="2020-01-02",
                       filing_datetime="2020-01-10 17:00:00",
                       amount_low=1001.0, amount_high=15000.0)
    df = store.trades_frame(conn)
    assert len(df) == 1
    assert df["amount_mid"].iloc[0] == (1001.0 + 15000.0) / 2


def test_trades_frame_since_filters_on_filing_date(conn):
    """Recent-era slice must window by filing_datetime (the actionable anchor),
    not transaction_date: so the point-in-time discipline is preserved."""
    from copytrade.store import db as store

    f = store.upsert_filer(conn, "Sen. Window", "congress", None)
    store.insert_trade(conn, filer_id=f, source="senate_watcher", source_uid="old",
                       ticker="AA", side="buy", instrument="stock",
                       transaction_date="2019-12-01", filing_datetime="2020-01-10 17:00:00",
                       amount_low=1001.0, amount_high=15000.0)
    store.insert_trade(conn, filer_id=f, source="senate_watcher", source_uid="new",
                       ticker="BB", side="buy", instrument="stock",
                       transaction_date="2023-12-01", filing_datetime="2024-01-10 17:00:00",
                       amount_low=1001.0, amount_high=15000.0)
    assert len(store.trades_frame(conn)) == 2
    recent = store.trades_frame(conn, since="2023-01-01")
    assert len(recent) == 1 and recent["ticker"].iloc[0] == "BB"


def test_unprefixed_credentials_are_bridged_from_dotenv(tmp_path, monkeypatch):
    """Alpaca keys deliberately carry no COPYTRADE_ prefix (vendor convention),
    but pydantic-settings only reads PREFIXED keys out of .env, so without an
    explicit bridge they sit in the file while broker/paper.py and the CLI read
    os.environ and report them missing."""
    import os

    from copytrade.config import load_dotenv_unprefixed

    env = tmp_path / ".env"
    env.write_text(
        "# comment\n"
        "COPYTRADE_TIINGO_API_KEY=prefixed-should-be-ignored-here\n"
        "ALPACA_API_KEY_ID=abc123\n"
        'ALPACA_API_SECRET_KEY="quoted-secret"\n'
        "NOT_ALLOWED=nope\n",
        encoding="utf-8")
    for k in ("ALPACA_API_KEY_ID", "ALPACA_API_SECRET_KEY", "NOT_ALLOWED"):
        monkeypatch.delenv(k, raising=False)

    loaded = load_dotenv_unprefixed(env)
    assert set(loaded) == {"ALPACA_API_KEY_ID", "ALPACA_API_SECRET_KEY"}
    assert os.environ["ALPACA_API_KEY_ID"] == "abc123"
    assert os.environ["ALPACA_API_SECRET_KEY"] == "quoted-secret", "quotes stripped"
    assert "NOT_ALLOWED" not in os.environ, "only the allow-listed names are exported"


def test_real_environment_wins_over_dotenv(tmp_path, monkeypatch):
    """CI secrets and shell exports must never be clobbered by a stale local
    .env: the bridge fills gaps only."""
    import os

    from copytrade.config import load_dotenv_unprefixed

    env = tmp_path / ".env"
    env.write_text("ALPACA_API_KEY_ID=from-file\n", encoding="utf-8")
    monkeypatch.setenv("ALPACA_API_KEY_ID", "from-shell")
    loaded = load_dotenv_unprefixed(env)
    assert loaded == []
    assert os.environ["ALPACA_API_KEY_ID"] == "from-shell"


def test_dotenv_bridge_tolerates_a_missing_file(tmp_path):
    from copytrade.config import load_dotenv_unprefixed

    assert load_dotenv_unprefixed(tmp_path / "nope.env") == []
