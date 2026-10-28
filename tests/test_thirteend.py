from copytrade.ingest.thirteend import parse_company_tickers, parse_master_idx

MASTER = """Description:           Master Index of EDGAR Dissemination Feed
CIK|Company Name|Form Type|Date Filed|Filename
320193|APPLE INC|SC 13D|2024-03-15|edgar/data/320193/000.txt
1045810|NVIDIA CORP|SC 13D/A|2024-03-16|edgar/data/1045810/001.txt
789019|MICROSOFT CORP|SC 13D|2024-04-01|edgar/data/789019/002.txt
1000|SOME CO|10-K|2024-04-01|edgar/data/1000/003.txt
"""


def test_parse_master_idx_keeps_initial_13d_only():
    rows = parse_master_idx(MASTER, forms=("SC 13D",))
    # AAPL + MSFT initial 13D; NVDA is a /A amendment (excluded); 10-K excluded
    assert {r["cik"] for r in rows} == {320193, 789019}
    assert all(r["form"] == "SC 13D" for r in rows)
    assert rows[0]["date"] == "2024-03-15"


def test_parse_company_tickers_maps_cik_to_ticker():
    js = ('{"0":{"cik_str":320193,"ticker":"AAPL","title":"Apple Inc."},'
          '"1":{"cik_str":789019,"ticker":"MSFT","title":"Microsoft"}}')
    m = parse_company_tickers(js)
    assert m[320193] == "AAPL" and m[789019] == "MSFT"


def test_master_idx_maps_to_tickers_end_to_end():
    rows = parse_master_idx(MASTER, forms=("SC 13D",))
    m = {320193: "AAPL", 789019: "MSFT"}
    tickers = sorted(m[r["cik"]] for r in rows if r["cik"] in m)
    assert tickers == ["AAPL", "MSFT"]
