"""Price-chain + Quiver parsers (schema-shaped fixtures, live hosts are
proxy-blocked in the dev sandbox; validate each with one live pull)."""

import pandas as pd

from copytrade.ingest.market import (
    fetch_daily_closes,
    parse_stooq_csv,
    parse_tiingo_payload,
    stooq_symbol,
)
from copytrade.ingest.quiver import parse_quiver_payload


def test_stooq_symbol_mapping():
    assert stooq_symbol("AAPL") == "aapl.us"
    assert stooq_symbol("BRK.B") == "brk-b.us"


def test_parse_stooq_csv():
    text = "Date,Open,High,Low,Close,Volume\n2024-01-02,100,101,99,100.5,123\n2024-01-03,,,,bad,1\n"
    df = parse_stooq_csv(text, "AAPL")
    assert len(df) == 1
    assert df.iloc[0]["adj_close"] == 100.5 and df.iloc[0]["ticker"] == "AAPL"
    assert parse_stooq_csv("No data", "X").empty


def test_parse_tiingo_payload():
    payload = [
        {"date": "2024-01-02T00:00:00.000Z", "adjClose": 187.6, "close": 188.0},
        {"date": None, "adjClose": 1.0},
    ]
    df = parse_tiingo_payload(payload, "AAPL")
    assert len(df) == 1 and df.iloc[0]["date"] == "2024-01-02"


def test_fetch_chain_falls_back_and_records_provenance(monkeypatch):
    import copytrade.ingest.market as m

    monkeypatch.setattr(m, "fetch_yfinance", lambda t, s, e: pd.DataFrame(
        [{"ticker": "AAPL", "date": "2024-01-02", "adj_close": 1.0}]))
    monkeypatch.setattr(m, "fetch_tiingo", lambda t, s, e, k: pd.DataFrame(
        [{"ticker": "MSFT", "date": "2024-01-02", "adj_close": 2.0}]))
    monkeypatch.setattr(m, "fetch_stooq", lambda t, s, e: pd.DataFrame(
        [{"ticker": "GE", "date": "2024-01-02", "adj_close": 3.0}]))
    df, prov = fetch_daily_closes(["AAPL", "MSFT", "GE"], "2024-01-01", "2024-02-01",
                                  tiingo_api_key="k")
    assert set(df["ticker"]) == {"AAPL", "MSFT", "GE"}
    assert prov["yfinance"] == ["AAPL"]
    assert prov["tiingo"] == ["MSFT"]
    assert prov["stooq_split_adjusted_only"] == ["GE"]


def test_fetch_chain_survives_dead_primary(monkeypatch):
    import copytrade.ingest.market as m

    def boom(t, s, e):
        raise RuntimeError("rate limited")
    monkeypatch.setattr(m, "fetch_yfinance", boom)
    monkeypatch.setattr(m, "fetch_stooq", lambda t, s, e: pd.DataFrame(
        [{"ticker": "SPY", "date": "2024-01-02", "adj_close": 470.0}]))
    df, prov = fetch_daily_closes(["SPY"], "2024-01-01", "2024-02-01")
    assert len(df) == 1
    assert "yfinance_error" in prov


def test_parse_quiver_payload():
    payload = [
        {"Representative": "NANCY PELOSI", "ReportDate": "2026-07-01",
         "TransactionDate": "2026-06-20", "Ticker": "NVDA",
         "Transaction": "Purchase", "Range": "$1,000,001 - $5,000,000",
         "House": "Representatives", "Party": "D"},
        {"Representative": "Some Member", "Ticker": "AAPL",
         "TransactionDate": "2026-06-20", "Transaction": "Sale (Full)"},  # no ReportDate
    ]
    trades, skipped = parse_quiver_payload(payload)
    assert len(trades) == 1 and skipped == 1
    t = trades[0]
    assert t.name == "Nancy Pelosi" and t.chamber == "House" and t.side == "buy"
    assert (t.amount_low, t.amount_high) == (1000001, 5000000)
    assert t.filing_date == "2026-07-01" and t.source == "quiver_congress"


def test_registry_quiver_prereq():
    from copytrade.scrape.registry import build_sources

    class S:
        finnhub_api_key = ""
        quiver_api_key = ""
        scrape_sources = ["auto"]
    assert "quiver_congress" not in [s.name for s in build_sources(S())]

    class S2(S):
        quiver_api_key = "k"
    assert "quiver_congress" in [s.name for s in build_sources(S2())]


def test_upsert_rejects_non_positive_prices(tmp_path):
    """An adj_close <= 0 is a provider error, not a price. Storing one is
    catastrophic: returns are p_t/p_0 - 1, so a negative price flips the sign
    of the ratio. Four foreign OTC ADRs once dragged the whole-sample mean
    excess return to -310% this way."""
    import pandas as pd

    from pathlib import Path

    from copytrade.store import db as store

    conn = store.connect(tmp_path / "p.db")
    store.run_migrations(conn, Path(__file__).resolve().parents[1] / "migrations")
    df = pd.DataFrame({
        "ticker": ["GOOD", "NEG", "ZERO", "GOOD"],
        "date": ["2020-01-02", "2020-01-02", "2020-01-02", "2020-01-03"],
        "adj_close": [10.0, -2.83, 0.0, 11.0],
    })
    n = store.upsert_prices(conn, df)
    assert n == 2, "only the two positive prices may be stored"
    stored = {r[0] for r in conn.execute("SELECT DISTINCT ticker FROM prices")}
    assert stored == {"GOOD"}


def test_event_study_truncates_at_a_bad_price_instead_of_dropping_the_trade():
    """A non-positive price MID-PATH is end-of-data, not grounds to reject.

    Rejecting outright made a short horizon's SAMPLE depend on prices far in
    the future: with the full sweep grid a corrupt tick ~500 days out silently
    removed the trade from the 1-day cell too. So horizons that complete
    BEFORE the bad print must still be priced, and only those that would read
    through it return NaN.
    """
    import numpy as np
    import pandas as pd

    from copytrade.backtest.costs import get_cost_model
    from copytrade.backtest.engine import run_event_study

    cal = pd.bdate_range("2020-01-01", periods=40)
    bench = pd.Series(np.linspace(100, 110, len(cal)), index=cal)
    prices = pd.DataFrame({
        "OK": np.linspace(50, 55, len(cal)),
        "BAD": np.concatenate([np.full(20, 20.0), np.full(len(cal) - 20, -5.0)]),
    }, index=cal)
    trades = pd.DataFrame({
        "trade_id": [1, 2],
        "filer_id": [1, 1],
        "filer_name": ["X", "X"],
        "filer_kind": ["congress", "congress"],
        "side": ["buy", "buy"],
        "ticker": ["OK", "BAD"],
        "instrument": ["stock", "stock"],
        "filing_datetime": pd.to_datetime(["2020-01-06", "2020-01-06"]),
        "transaction_date": pd.to_datetime(["2020-01-02", "2020-01-02"]),
        "amount_low": [1000.0, 1000.0],
        "amount_high": [15000.0, 15000.0],
    })
    es = run_event_study(trades, prices, bench, get_cost_model("zero"), [5, 21],
                         anchor="filing")
    bad = es.per_trade[es.per_trade["ticker"] == "BAD"]
    assert len(bad) == 1, "the trade must be kept, not dropped"
    # entry is ~index 4; h5 completes before the bad print at index 20
    assert pd.notna(bad["h5"].iloc[0]), "a horizon ending before the bad tick is valid"
    # h21 would have to read through it
    assert pd.isna(bad["h21"].iloc[0]), "a horizon reading through the bad tick is NaN"
    assert es.per_trade["h5"].abs().max() < 1.0, "no impossible returns survive"


def test_event_study_rejects_a_non_positive_ENTRY_price():
    """An entry price of zero or less is still fatal, there is no return
    without an entry price. Only MID-PATH corruption is treated as truncation."""
    import numpy as np
    import pandas as pd

    from copytrade.backtest.costs import get_cost_model
    from copytrade.backtest.engine import run_event_study

    cal = pd.bdate_range("2020-01-01", periods=40)
    bench = pd.Series(np.linspace(100, 110, len(cal)), index=cal)
    # ENTRY price itself is non-positive from the very start
    prices = pd.DataFrame({
        "OK": np.linspace(50, 55, len(cal)),
        "DEAD": np.full(len(cal), -5.0),
    }, index=cal)
    trades = pd.DataFrame({
        "trade_id": [1, 2],
        "filer_id": [1, 1],
        "filer_name": ["X", "X"],
        "filer_kind": ["congress", "congress"],
        "side": ["buy", "buy"],
        "ticker": ["OK", "DEAD"],
        "instrument": ["stock", "stock"],
        "filing_datetime": pd.to_datetime(["2020-01-06", "2020-01-06"]),
        "transaction_date": pd.to_datetime(["2020-01-02", "2020-01-02"]),
        "amount_low": [1000.0, 1000.0],
        "amount_high": [15000.0, 15000.0],
    })
    es = run_event_study(trades, prices, bench, get_cost_model("zero"), [21],
                         anchor="filing")
    assert list(es.per_trade["ticker"]) == ["OK"], "DEAD must be excluded"
    assert es.unpriceable == 1
    assert es.per_trade["h21"].abs().max() < 1.0, "no impossible returns survive"


def test_provider_symbol_rewrites_class_shares_only():
    """US class shares are disclosed with a dot (BRK.B) but yfinance/Tiingo key
    them with a dash. Without the rewrite every class share 404s and lands in
    the unpriceable pile next to genuine delistings, Berkshire B was missing
    from the price matrix for exactly this reason."""
    from copytrade.ingest.market import provider_symbol

    assert provider_symbol("BRK.B") == "BRK-B"
    assert provider_symbol("BF.B") == "BF-B"
    assert provider_symbol("HEI.A") == "HEI-A"
    assert provider_symbol("AAPL") == "AAPL"


def test_provider_symbol_leaves_non_equity_lines_alone():
    """Same shape, different instrument: warrants, units, foreign listings and
    preferreds must fail honestly rather than being mapped onto a DIFFERENT
    security that happens to exist under the rewritten symbol."""
    from copytrade.ingest.market import provider_symbol

    for t in ("ETWO.W", "PCPL.U", "INSW.V", "SBUX.SW", "RDSA.AS", "ADT.F", "BAC.PL"):
        assert provider_symbol(t) == t, t


def test_yfinance_maps_provider_symbols_back_to_the_disclosed_ticker(monkeypatch):
    """The DB is keyed by the DISCLOSED ticker, so a provider-symbol round trip
    must not leak into stored rows."""
    import numpy as np
    import pandas as pd

    import copytrade.ingest.market as m

    idx = pd.to_datetime(["2024-01-02", "2024-01-03"])
    frame = pd.DataFrame(
        np.array([[100.0], [101.0]]),
        index=idx,
        columns=pd.MultiIndex.from_tuples([("BRK-B", "Close")]),
    )

    class FakeYF:
        @staticmethod
        def download(**kwargs):
            assert "BRK-B" in kwargs["tickers"], "must query the provider symbol"
            return frame

    monkeypatch.setitem(__import__("sys").modules, "yfinance", FakeYF)
    out = m.fetch_yfinance(["BRK.B"], "2024-01-01", "2024-02-01")
    assert list(out["ticker"].unique()) == ["BRK.B"], "stored under the disclosed ticker"
    assert len(out) == 2


def test_tiingo_throttle_is_recorded_not_mistaken_for_a_delisting(monkeypatch):
    """A 429 means the free-tier hourly allocation ran out. Swallowed silently
    it is indistinguishable from 'ticker does not exist', so a throttled run
    gets written up as a set of delistings, and report.py publishes that as a
    survivorship caveat."""
    import httpx

    import copytrade.ingest.market as m

    class FakeResp:
        status_code = 429
        text = "rate limited"

        def json(self):
            return {}

    class FakeClient:
        def __init__(self, *a, **k): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def get(self, *a, **k): return FakeResp()

    monkeypatch.setattr(httpx, "Client", FakeClient)
    out = m.fetch_tiingo(["BK", "CTRA", "EXAS"], "2024-01-01", "2024-02-01", "key")
    assert out.empty
    assert m.fetch_tiingo.last_errors, "the throttle must be recorded"
    assert set(m.fetch_tiingo.last_errors.values()) == {429}
    # account-wide limit: stop hammering after the first 429
    assert len(m.fetch_tiingo.last_errors) == 1


def test_fetch_chain_surfaces_the_tiingo_throttle_in_provenance(monkeypatch):
    import pandas as pd

    import copytrade.ingest.market as m

    monkeypatch.setattr(m, "fetch_yfinance", lambda t, s, e: m._empty())
    def fake_tiingo(tickers, s, e, k):
        fake_tiingo.last_errors = {tickers[0]: 429}
        m.fetch_tiingo.last_errors = fake_tiingo.last_errors
        return m._empty()
    monkeypatch.setattr(m, "fetch_tiingo", fake_tiingo)
    monkeypatch.setattr(m, "fetch_stooq", lambda t, s, e: m._empty())

    _, prov = m.fetch_daily_closes(["BK"], "2024-01-01", "2024-02-01", "key")
    assert "tiingo_error" in prov
    assert "THROTTLED" in prov["tiingo_error"][0]
