import numpy as np
import pandas as pd

from copytrade.backtest.currency import load_usdcad, to_cad


def _fx(dates, rates):
    return pd.Series(dict(zip(pd.to_datetime(dates), rates))).sort_index()


def test_to_cad_compounds_asset_and_currency_returns():
    idx = pd.to_datetime(["2020-01-01", "2020-06-01", "2020-12-01"])
    eq = pd.Series([100.0, 110.0, 120.0], index=idx)      # +20% in USD
    fx = _fx(idx, [1.30, 1.35, 1.43])                     # +10% USD/CAD
    cad, info = to_cad(eq, fx)
    assert np.isclose(info["usd_return"], 0.20)
    assert np.isclose(info["fx_return"], 0.10)
    # CAD wealth compounds both legs: 1.20 * 1.10 - 1
    assert np.isclose(info["cad_return"], 0.32)
    assert np.isclose(info["fx_contribution"], 0.12)
    assert np.isclose(cad.iloc[0], 130.0)


def test_a_falling_usd_can_erase_a_usd_gain():
    # The whole reason this module exists: a USD win is not automatically a
    # CAD win for a Canadian investor.
    idx = pd.to_datetime(["2020-01-01", "2021-01-01"])
    eq = pd.Series([100.0, 115.0], index=idx)             # +15% USD
    fx = _fx(idx, [1.40, 1.20])                           # USD/CAD -14.3%
    _, info = to_cad(eq, fx)
    assert info["usd_return"] > 0
    assert info["cad_return"] < 0, "FX can flip the sign of the realised return"


def test_rates_are_forward_filled_but_never_back_filled():
    # Equity starts BEFORE the first FX observation. Those early days must be
    # dropped, not filled with a fabricated backwards rate.
    idx = pd.to_datetime(["2016-01-01", "2017-01-03", "2017-01-04", "2017-01-05"])
    eq = pd.Series([100.0, 101.0, 102.0, 103.0], index=idx)
    fx = _fx(["2017-01-03", "2017-01-05"], [1.34, 1.32])   # 01-04 missing
    cad, info = to_cad(eq, fx)
    assert info["dropped_days"] == 1, "the pre-FX day must be dropped"
    assert cad.index[0] == pd.Timestamp("2017-01-03")
    # 2017-01-04 has no print -> carries 01-03's rate forward (correct: the
    # rate genuinely did not change on a non-publication day)
    assert np.isclose(cad.loc[pd.Timestamp("2017-01-04")], 102.0 * 1.34)


def test_empty_inputs_return_empty_without_raising():
    idx = pd.to_datetime(["2020-01-01", "2020-02-01"])
    eq = pd.Series([100.0, 101.0], index=idx)
    cad, info = to_cad(eq, pd.Series(dtype=float))
    assert cad.empty and info == {}
    cad2, info2 = to_cad(pd.Series(dtype=float), _fx(idx, [1.3, 1.3]))
    assert cad2.empty and info2 == {}


def test_no_overlap_returns_empty():
    eq = pd.Series([100.0, 101.0], index=pd.to_datetime(["2010-01-01", "2010-02-01"]))
    fx = _fx(["2020-01-01", "2020-02-01"], [1.3, 1.35])
    cad, info = to_cad(eq, fx)
    assert cad.empty and info == {}


def test_load_usdcad_reads_only_the_usdcad_pair(tmp_path):
    from pathlib import Path

    from copytrade.store import db as store

    conn = store.connect(tmp_path / "t.db")
    store.run_migrations(conn, Path(__file__).resolve().parents[1] / "migrations")
    conn.executemany(
        "INSERT INTO fx_rates(date,pair,rate,source) VALUES (?,?,?,?)",
        [("2020-01-02", "USDCAD", 1.30, "test"),
         ("2020-01-03", "USDCAD", 1.31, "test"),
         ("2020-01-02", "EURCAD", 1.45, "test")],
    )
    conn.commit()
    s = load_usdcad(conn)
    assert len(s) == 2
    assert s.index.is_monotonic_increasing
    assert np.isclose(s.iloc[1], 1.31)
