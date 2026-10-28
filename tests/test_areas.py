import pandas as pd

from copytrade.rank.areas import filer_focus, sector_attribution, ticker_leaders

# one row per priced buy, with a net forward-return column h63
PT = pd.DataFrame({
    "trade_id": [1, 2, 3, 4, 5, 6],
    "filer_id": [1, 1, 2, 2, 3, 3],
    "filer_name": ["Alice", "Alice", "Bob", "Bob", "Cara", "Cara"],
    "ticker": ["NVDA", "AAPL", "NVDA", "LMT", "LMT", "XOM"],
    "amount": [10000, 10000, 10000, 10000, 10000, 10000],
    "h63": [0.20, 0.10, 0.30, -0.05, -0.10, 0.02],
})
SMAP = pd.DataFrame({
    "ticker": ["NVDA", "AAPL", "LMT", "XOM"],
    "sector": ["Technology", "Technology", "Industrials", "Energy"],
})


def test_sector_attribution_ranks_by_return():
    out = sector_attribution(PT, SMAP, "h63", min_buys=1).set_index("sector")
    # Technology: NVDA+0.20, AAPL+0.10, NVDA+0.30 -> mean 0.20; Industrials LMT -0.075; Energy +0.02
    assert out.loc["Technology", "n_buys"] == 3
    assert abs(out.loc["Technology", "mean_ret"] - 0.20) < 1e-9
    assert out.index[0] == "Technology"                 # best area on top
    assert out.loc["Technology", "n_filers"] == 2       # Alice + Bob


def test_sector_attribution_unmapped_falls_back_unknown():
    pt = PT.assign(ticker=["ZZZZ"] * 6)
    out = sector_attribution(pt, SMAP, "h63", min_buys=1)
    assert list(out["sector"]) == ["Unknown"]           # never guessed


def test_ticker_leaders_best_worst_crowded():
    lead = ticker_leaders(PT, "h63", min_buys=2, top=5)
    assert lead["best"].iloc[0]["ticker"] == "NVDA"     # +0.20/+0.30 mean 0.25
    assert lead["most_traded"].iloc[0]["ticker"] in ("NVDA", "LMT")  # each traded twice
    # LMT is the worst 2+ sample (-0.05/-0.10)
    assert lead["worst"].iloc[0]["ticker"] == "LMT"


def test_filer_focus_shows_pursuit_and_outcome():
    ff = filer_focus(PT, SMAP, "h63", "Alice", top=8)
    assert set(ff["tickers"]["ticker"]) == {"NVDA", "AAPL"}
    assert ff["sectors"].iloc[0]["sector"] == "Technology"   # Alice is all tech
    assert abs(ff["sectors"].iloc[0]["mean_ret"] - 0.15) < 1e-9
