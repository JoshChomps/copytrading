import numpy as np
import pandas as pd

from copytrade.ingest.cftc import parse_cot_csv, store_cot
from copytrade.ingest.thirteenf import parse_13f_infotable
from copytrade.rank.derivatives import (
    classify_option,
    cot_snapshot,
    instrument_framing,
    option_direction_study,
    stock_vs_option_by_filer,
)


def test_classify_option_direction():
    assert classify_option("call_option", None) == ("call", 1)      # long call = bullish
    assert classify_option("put_option", None) == ("put", -1)       # long put = bearish
    assert classify_option("option", "Sold Put on AAPL") == ("put", 1)   # short put = bullish
    assert classify_option("option", "Written Call") == ("call", -1)     # short call = bearish
    assert classify_option("stock_option", "AAPL options") == ("option", 1)  # ambiguous -> long


def test_instrument_framing_buckets():
    trades = pd.DataFrame({
        "trade_id": [1, 2, 3, 4],
        "filer_id": [1, 1, 2, 2],
        "ticker": ["AAPL", "AAPL", "BA", "SU"],
        "instrument": ["stock", "stock_option", "put_option", "ca_material_change"],
    })
    fr = instrument_framing(trades).set_index("cls")
    assert fr.loc["stock", "trades"] == 1
    assert fr.loc["option", "trades"] == 2                # stock_option + put_option
    assert fr.loc["canada_material_change", "trades"] == 1


def test_option_direction_study_signs_underlying_move():
    cal = pd.bdate_range("2020-01-01", periods=200)
    px = pd.DataFrame({"XX": np.linspace(100, 200, len(cal)),   # doubles
                       "SPY": np.linspace(100, 105, len(cal))}, index=cal)
    bench = px["SPY"]
    trades = pd.DataFrame({
        "trade_id": [1, 2], "filer_id": [1, 1], "filer_name": ["F1", "F1"],
        "filer_kind": ["congress", "congress"], "ticker": ["XX", "XX"],
        "instrument": ["call_option", "put_option"],
        "filing_datetime": [pd.Timestamp("2020-01-02 17:00")] * 2,
    })
    trades["asset_description"] = None
    study = option_direction_study(trades, px, bench, horizon=100)
    by = study.set_index("opt_kind")["underlying_excess"]
    # underlying rose ~+48% excess: bullish call gets +, bearish put gets -
    assert by["call"] > 0.3
    assert by["put"] < -0.3


def test_stock_vs_option_framing():
    opt = pd.DataFrame({
        "filer_id": [1, 1, 1], "filer": ["Pelosi"] * 3,
        "underlying_excess": [0.3, 0.4, -0.1],
    })
    stock = pd.DataFrame({"filer": ["Pelosi"], "mean_return": [0.05]})
    framed = stock_vs_option_by_filer(stock, opt, min_each=3)
    assert len(framed) == 1
    row = framed.iloc[0]
    assert row["opt_n"] == 3
    assert row["stock_mean_return"] == 0.05
    assert row["opt_dir_excess"] > row["stock_mean_return"]   # edge is leverage, not picking


# ---- CFTC COT ----

COT_CSV = (
    '"E-MINI S&P 500 - CHICAGO MERCANTILE EXCHANGE","250124","2025-01-24","13874A",'
    '"","","",2000000,120000,180000,"",900000,700000,"","",50000,80000\n'
    '"GOLD - COMMODITY EXCHANGE INC.","250124","2025-01-24","088691",'
    '"","","",500000,250000,90000,"",100000,120000,"","",20000,15000\n'
)


def test_parse_cot_csv():
    rows = parse_cot_csv(COT_CSV)
    assert len(rows) == 2
    sp = rows[0]
    assert "S&P 500" in sp.market and sp.report_date == "2025-01-24"
    assert sp.noncomm_long == 120000 and sp.noncomm_short == 180000
    assert sp.spec_net == -60000            # large specs net SHORT the S&P


def test_store_and_snapshot_cot(conn):
    store_cot(conn, parse_cot_csv(COT_CSV))
    snap = cot_snapshot(conn, markets=("S&P 500", "GOLD"))
    assert len(snap) == 2
    sp = snap.set_index("market").filter(like="S&P", axis=0)
    assert float(sp["spec_net"].iloc[0]) == -60000


def test_13f_putcall_parsing():
    xml = """<informationTable xmlns="http://www.sec.gov/edgar/document/thirteenf/informationtable">
      <infoTable><nameOfIssuer>APPLE</nameOfIssuer><cusip>037833100</cusip>
        <value>1000</value><shrsOrPrnAmt><sshPrnamt>100</sshPrnamt></shrsOrPrnAmt></infoTable>
      <infoTable><nameOfIssuer>TESLA</nameOfIssuer><cusip>88160R101</cusip>
        <value>2000</value><shrsOrPrnAmt><sshPrnamt>50</sshPrnamt></shrsOrPrnAmt>
        <putCall>Put</putCall></infoTable>
    </informationTable>"""
    hs = parse_13f_infotable(xml)
    assert len(hs) == 2
    assert hs[0].instrument == "stock"
    assert hs[1].put_call == "Put" and hs[1].instrument == "put_option"


# ---- disclosure lag ----

def test_lag_decomposition():
    from copytrade.rank.lag import fastest_filers, lag_by_source, reporting_lag_days

    trades = pd.DataFrame({
        "filer_id": [1]*3 + [2]*3,
        "filer_name": ["Fast Fred"]*3 + ["Slow Sue"]*3,
        "source": ["edgar_form4"]*3 + ["senate_watcher"]*3,
        "transaction_date": ["2024-01-01"]*6,
        "filing_datetime": ["2024-01-03 17:00"]*3 + ["2024-02-14 17:00"]*3,  # 2d vs 44d
        "instrument": ["stock"]*6,
    })
    lags = reporting_lag_days(trades)
    assert abs(lags.iloc[0] - 2) < 0.5 and abs(lags.iloc[3] - 44) < 1.5
    by = lag_by_source(trades).set_index("source")
    assert by.loc["edgar_form4", "pct_le_2d"] == 1.0
    assert by.loc["senate_watcher", "pct_le_2d"] == 0.0
    assert by.loc["edgar_form4", "median_days"] < by.loc["senate_watcher", "median_days"]
    fast = fastest_filers(trades, min_trades=3)
    assert fast.iloc[0]["filer_name"] == "Fast Fred"


def test_13f_option_deltas_stay_distinct_from_stock(conn):
    """A fund holding the same cusip as stock AND puts must produce two
    distinct delta trades: the whole point of tracking fund options."""
    from copytrade.ingest.thirteenf import Holding, quarterly_deltas, store_13f
    from copytrade.store import db as store

    fid = store.upsert_filer(conn, "Scion (Burry)", "fund", {"cik": "0001649339"})
    conn.execute("INSERT INTO cusip_map(cusip, ticker) VALUES (?,?)", ("88160R101", "TSLA"))
    # Q1: 100 shares of stock, no puts
    store_13f(conn, fid, "2024-03-31", [Holding("88160R101", 10000, 100, "TSLA")])
    # Q2: same stock, NEW 50-contract put position on the same name
    store_13f(conn, fid, "2024-06-30", [
        Holding("88160R101", 10000, 100, "TSLA"),
        Holding("88160R101", 5000, 50, "TSLA", put_call="Put"),
    ])
    conn.commit()
    ins, _ = quarterly_deltas(conn, fid, "2024-03-31", "2024-06-30", "2024-08-14 16:00:00")
    assert ins == 1                                  # only the put position changed
    row = conn.execute("SELECT ticker, side, instrument FROM trades WHERE source='edgar_13f'").fetchone()
    assert row["ticker"] == "TSLA" and row["side"] == "buy" and row["instrument"] == "put_option"
