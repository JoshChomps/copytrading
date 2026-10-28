import pandas as pd

from copytrade.rank.insiders import (
    abnormal_size_ids,
    cluster_buy_ids,
    routine_filer_ids,
    subset_stats,
)


def test_abnormal_size_ids_flags_outsized_bets():
    # filer 1 usually buys ~$10k, then makes one ~$50k bet -> abnormal.
    # filer 2 has too little history to judge -> never flagged.
    buys = pd.DataFrame({
        "trade_id": [1, 2, 3, 4, 5],
        "filer_id": [1, 1, 1, 1, 2],
        "amount_mid": [10000.0, 11000.0, 9000.0, 50000.0, 99000.0],
    })
    ids = abnormal_size_ids(buys, mult=2.0, min_history=3)
    assert ids == {4}                       # only filer 1's outsized bet; filer 2 lacks history


def _buys(rows):
    df = pd.DataFrame(rows)
    df["filing_datetime"] = pd.to_datetime(df["filing_datetime"])
    return df


def test_cluster_buy_ids_flags_consensus():
    # AAA: 3 distinct insiders within 7 days -> all 3 are cluster buys.
    # BBB: 1 insider buying 3x -> NOT a cluster (needs distinct insiders).
    buys = _buys([
        {"trade_id": 1, "filer_id": 10, "ticker": "AAA", "filing_datetime": "2024-01-01"},
        {"trade_id": 2, "filer_id": 11, "ticker": "AAA", "filing_datetime": "2024-01-03"},
        {"trade_id": 3, "filer_id": 12, "ticker": "AAA", "filing_datetime": "2024-01-05"},
        {"trade_id": 4, "filer_id": 20, "ticker": "BBB", "filing_datetime": "2024-01-01"},
        {"trade_id": 5, "filer_id": 20, "ticker": "BBB", "filing_datetime": "2024-01-02"},
        {"trade_id": 6, "filer_id": 20, "ticker": "BBB", "filing_datetime": "2024-01-03"},
    ])
    ids = cluster_buy_ids(buys, window_days=7, min_insiders=3)
    assert ids == {3}                      # only the 3rd AAA buy sees all 3 insiders in-window
    # a wider requirement of 2 insiders flags the 2nd and 3rd AAA buys
    ids2 = cluster_buy_ids(buys, window_days=7, min_insiders=2)
    assert ids2 == {2, 3} and 4 not in ids2 and 5 not in ids2


def test_cluster_respects_window():
    buys = _buys([
        {"trade_id": 1, "filer_id": 10, "ticker": "AAA", "filing_datetime": "2024-01-01"},
        {"trade_id": 2, "filer_id": 11, "ticker": "AAA", "filing_datetime": "2024-01-02"},
        {"trade_id": 3, "filer_id": 12, "ticker": "AAA", "filing_datetime": "2024-03-01"},  # 2 months later
    ])
    # window too short to gather 3 distinct insiders together
    assert cluster_buy_ids(buys, window_days=7, min_insiders=3) == set()


def test_routine_filer_ids_cohen_malloy():
    # filer 1 trades every January for 3 years -> routine; filer 2 trades once -> opportunistic
    trades = pd.DataFrame({
        "filer_id": [1, 1, 1, 2],
        "transaction_date": pd.to_datetime(
            ["2021-01-15", "2022-01-20", "2023-01-10", "2023-06-01"]),
    })
    routine = routine_filer_ids(trades, min_years=3)
    assert routine == {1}


def test_subset_stats_basic():
    pt = pd.DataFrame({
        "h63": [0.10, -0.05, 0.20, None],
        "entry_date": pd.to_datetime(["2024-01-02", "2024-02-02", "2024-03-04", "2024-04-01"]),
    })
    s = subset_stats(pt, "h63")
    assert s["n"] == 3                              # NaN row dropped
    assert abs(s["mean_excess"] - (0.10 - 0.05 + 0.20) / 3) < 1e-9
    assert abs(s["hit_rate"] - 2 / 3) < 1e-9


def test_abnormal_size_ids_pit_uses_only_prior_trades():
    """Point-in-time: a buy is 'abnormal' only versus the filer's EARLIER buys.
    The full-sample variant judged a 2016 trade against 2026 data, measured on
    the live congress sample, 364 trades (18.5% of the subset) were selected
    only because of information from the future."""
    import pandas as pd

    from copytrade.rank.insiders import abnormal_size_ids, abnormal_size_ids_pit

    # filer buys small four times, then huge, then small again.
    buys = pd.DataFrame({
        "trade_id": [1, 2, 3, 4, 5, 6],
        "filer_id": [1] * 6,
        "amount_mid": [10_000.0, 10_000.0, 10_000.0, 10_000.0, 90_000.0, 10_000.0],
        "filing_datetime": pd.to_datetime([
            "2020-01-01", "2020-02-01", "2020-03-01",
            "2020-04-01", "2020-05-01", "2020-06-01"]),
    })
    pit = abnormal_size_ids_pit(buys, mult=2.0, min_history=3)
    assert pit == {5}, "only the outsized buy, judged against its own past"

    # the whole-sample rule sees the 90k when judging EARLIER trades: the
    # median rises, which is information trade 4 could not have had
    full = abnormal_size_ids(buys, mult=2.0, min_history=3)
    assert 5 in full


def test_abnormal_pit_never_flags_before_min_history():
    import pandas as pd

    from copytrade.rank.insiders import abnormal_size_ids_pit

    buys = pd.DataFrame({
        "trade_id": [1, 2],
        "filer_id": [1, 1],
        "amount_mid": [1_000.0, 500_000.0],
        "filing_datetime": pd.to_datetime(["2020-01-01", "2020-02-01"]),
    })
    assert abnormal_size_ids_pit(buys, min_history=3) == set()


def test_top_quartile_mask_handles_tied_bucket_midpoints():
    """Congress amounts are disclosed RANGES stored as bucket midpoints, so 74%
    of the live sample shares one value. `>=` swept the whole mode in and
    selected 98.5% of trades, a 'large' subset that was ALL trades relabelled,
    which also padded the multiple-testing family with duplicate cells."""
    import pandas as pd

    from copytrade.rank.insiders import top_quartile_mask

    # 80% sit on one bucket midpoint, as on the live sample (74% at 8000.5),
    # so the 75th percentile lands ON the mode
    amounts = pd.Series([8000.5] * 80 + [50_000.0] * 14 + [1_000_000.0] * 6)
    mask, info = top_quartile_mask(amounts)
    assert info["threshold"] == 8000.5
    assert mask.sum() == 20, "strict > excludes the tied mode"
    # `>=` would have selected all 100, the bug this guards
    assert (amounts >= info["threshold"]).sum() == 100
    assert 0.10 <= info["fraction"] <= 0.30
    assert not info["degenerate"]

    # a genuinely degenerate case must be flagged for the caller to surface
    mask2, info2 = top_quartile_mask(pd.Series([8000.5] * 99 + [50_000.0]))
    assert info2["degenerate"] and mask2.sum() == 1


def test_top_quartile_mask_is_a_real_quartile_on_continuous_amounts():
    import numpy as np
    import pandas as pd

    from copytrade.rank.insiders import top_quartile_mask

    amounts = pd.Series(np.linspace(1_000, 500_000, 400))
    mask, info = top_quartile_mask(amounts)
    assert 0.24 <= info["fraction"] <= 0.26
    assert not info["degenerate"]
