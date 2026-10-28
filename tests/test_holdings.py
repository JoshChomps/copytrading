"""Disclosed net positions must never overstate what a PTR can support.

A Periodic Transaction Report discloses TRANSACTIONS, not holdings, and every
size is a RANGE. Consumer trackers render this stream as a "portfolio"; the
tests below pin the three ways that would be wrong, invisible pre-window
positions, fabricated scalar sizes, and a net-zero band read as a closed
position.
"""

from __future__ import annotations

import pandas as pd
import pytest

from copytrade.rank.holdings import filer_positions, filer_summary, most_held


def _trades(rows):
    return pd.DataFrame([
        {"filer_name": f, "ticker": t, "side": s, "instrument": "stock",
         "amount_low": lo, "amount_high": hi,
         "filing_datetime": pd.Timestamp(d)}
        for f, t, s, lo, hi, d in rows
    ])


def test_a_single_purchase_is_accumulating():
    pos = filer_positions(_trades([
        ("A", "TSM", "buy", 100_001.0, 250_000.0, "2025-08-06")]))
    r = pos.iloc[0]
    assert r["status"] == "accumulating (no disclosed sale)"
    assert (r["net_low"], r["net_high"]) == (100_001.0, 250_000.0)


def test_the_net_is_a_band_not_a_number():
    """Buy $100k-250k, sell $1k-15k. The honest net spans most-bought-least-sold
    to least-bought-most-sold; collapsing it to one figure invents precision."""
    pos = filer_positions(_trades([
        ("A", "X", "buy", 100_001.0, 250_000.0, "2025-01-10"),
        ("A", "X", "sell", 1_001.0, 15_000.0, "2025-02-10"),
    ]))
    r = pos.iloc[0]
    assert r["net_low"] == pytest.approx(100_001.0 - 15_000.0)
    assert r["net_high"] == pytest.approx(250_000.0 - 1_001.0)
    assert r["net_low"] < r["net_high"], "a range, never a point"


def test_matching_bands_are_ambiguous_not_closed():
    """Selling the same bucket you bought is consistent with a full exit AND
    with a 90% remaining stake. Calling it closed would be a guess."""
    pos = filer_positions(_trades([
        ("A", "X", "buy", 1_001.0, 15_000.0, "2025-01-10"),
        ("A", "X", "sell", 1_001.0, 15_000.0, "2025-02-10"),
    ]))
    assert pos.iloc[0]["status"] == "ambiguous (bands overlap)"


def test_a_clear_net_buyer_is_labelled_as_one():
    pos = filer_positions(_trades([
        ("A", "X", "buy", 250_001.0, 500_000.0, "2025-01-10"),
        ("A", "X", "sell", 1_001.0, 15_000.0, "2025-02-10"),
    ]))
    assert pos.iloc[0]["status"] == "net buyer"


def test_a_clear_net_seller_is_labelled_as_one():
    pos = filer_positions(_trades([
        ("A", "X", "buy", 1_001.0, 15_000.0, "2025-01-10"),
        ("A", "X", "sell", 250_001.0, 500_000.0, "2025-02-10"),
    ]))
    assert pos.iloc[0]["status"] == "net seller"


def test_sales_with_no_disclosed_purchase_say_so():
    """The pre-window blind spot, named explicitly: the buy happened before we
    were looking, so the position is real but its opening is invisible."""
    pos = filer_positions(_trades([
        ("A", "X", "sell", 15_001.0, 50_000.0, "2025-02-10")]))
    assert pos.iloc[0]["status"] == "reducing (no disclosed purchase in window)"


def test_open_ended_top_bucket_does_not_collapse_to_zero():
    """'Over $50,000,000' and the $X+ tiers have a NULL high bound. Treating
    that as worth nothing would understate exactly the largest trades."""
    pos = filer_positions(_trades([
        ("A", "X", "buy", 50_000_001.0, None, "2025-01-10")]))
    r = pos.iloc[0]
    assert r["buy_high"] == 50_000_001.0
    assert r["net_high"] == 50_000_001.0


def test_positions_are_split_per_filer_and_ticker():
    pos = filer_positions(_trades([
        ("A", "X", "buy", 1_001.0, 15_000.0, "2025-01-10"),
        ("A", "Y", "buy", 1_001.0, 15_000.0, "2025-01-10"),
        ("B", "X", "buy", 1_001.0, 15_000.0, "2025-01-10"),
    ]))
    assert len(pos) == 3
    assert set(zip(pos["filer"], pos["ticker"])) == {("A", "X"), ("A", "Y"), ("B", "X")}


def test_filter_by_filer_is_case_insensitive():
    t = _trades([("Cleo Fields", "TSM", "buy", 1_001.0, 15_000.0, "2025-08-06"),
                 ("Other Member", "AAPL", "buy", 1_001.0, 15_000.0, "2025-08-06")])
    pos = filer_positions(t, "cleo fields")
    assert list(pos["ticker"]) == ["TSM"]


def test_most_held_counts_filers_not_holders():
    """A member who bought before the window and never traded again is invisible
    here: the column is filers with a net-buy DISCLOSURE."""
    rows = [(f, "HOT", "buy", 1_001.0, 15_000.0, "2025-01-10") for f in "ABC"]
    rows += [("D", "COLD", "buy", 1_001.0, 15_000.0, "2025-01-10")]
    crowded = most_held(filer_positions(_trades(rows)), min_filers=3)
    assert list(crowded["ticker"]) == ["HOT"]
    assert crowded.iloc[0]["filers"] == 3


def test_most_held_excludes_net_sellers():
    rows = [(f, "X", "buy", 1_001.0, 15_000.0, "2025-01-10") for f in "AB"]
    rows += [("C", "X", "sell", 250_001.0, 500_000.0, "2025-02-10")]
    crowded = most_held(filer_positions(_trades(rows)), min_filers=1)
    assert crowded.iloc[0]["filers"] == 2, "the net seller must not be counted long"


def test_summary_nets_across_a_filers_whole_book():
    pos = filer_positions(_trades([
        ("A", "X", "buy", 100_001.0, 250_000.0, "2025-01-10"),
        ("A", "Y", "sell", 15_001.0, 50_000.0, "2025-03-10"),
    ]))
    s = filer_summary(pos)
    assert s.iloc[0]["tickers"] == 2
    assert s.iloc[0]["net_low"] == pytest.approx(100_001.0 - 50_000.0)
    assert s.iloc[0]["last_disclosed"] == "2025-03-10"


def test_empty_input_returns_empty_frames_not_errors():
    empty = pd.DataFrame(columns=["filer_name", "ticker", "side", "amount_low",
                                  "amount_high", "filing_datetime", "instrument"])
    assert filer_positions(empty).empty
    assert filer_summary(pd.DataFrame()).empty
    assert most_held(pd.DataFrame()).empty


def test_rows_without_a_ticker_are_dropped():
    t = _trades([("A", "X", "buy", 1_001.0, 15_000.0, "2025-01-10")])
    t.loc[len(t)] = {"filer_name": "A", "ticker": None, "side": "buy",
                     "instrument": "stock", "amount_low": 1_001.0,
                     "amount_high": 15_000.0,
                     "filing_datetime": pd.Timestamp("2025-01-10")}
    assert len(filer_positions(t)) == 1
