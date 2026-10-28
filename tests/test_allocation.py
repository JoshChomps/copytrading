import pandas as pd

from copytrade.accounts.allocation import allocate_sleeves


def _ranking(rows):
    return pd.DataFrame(rows, columns=[
        "filer_id", "filer", "turnover_class", "copyable",
    ])


def _trades(filer_rows):
    # filer_id, filing dates spanning some period
    recs = []
    for filer_id, dates in filer_rows.items():
        for d in dates:
            recs.append({"filer_id": filer_id, "filing_datetime": pd.Timestamp(d)})
    return pd.DataFrame(recs)


def test_slow_filer_goes_tfsa_fast_filer_goes_margin():
    ranking = _ranking([
        (1, "Slow Sam", "long", True),
        (2, "Rapid Rita", "short", True),
    ])
    trades = _trades({
        1: [f"2023-{m:02d}-01" for m in range(1, 13)],          # 12/yr
        2: [f"2023-01-{d:02d}" for d in range(1, 29)] * 3,      # ~84/yr
    })
    plan = allocate_sleeves(ranking, trades, tfsa_room=50_000, taxable_capital=80_000)
    by = plan.assignments.set_index("filer")["sleeve"]
    assert by["Slow Sam"] == "tfsa"
    assert by["Rapid Rita"] == "margin"
    assert "business-income" in plan.assignments.set_index("filer").loc["Rapid Rita", "reason"]
    assert plan.tfsa_deployed > 0 and plan.margin_deployed > 0


def test_high_frequency_slow_turnover_still_excluded_from_tfsa():
    # long holds but files constantly -> too many orders for a TFSA pattern
    ranking = _ranking([(1, "Busy Bob", "long", True)])
    trades = _trades({1: [f"2023-01-{d:02d}" for d in range(1, 29)] * 2})  # ~56/yr
    plan = allocate_sleeves(ranking, trades)
    assert plan.assignments.iloc[0]["sleeve"] == "margin"


def test_no_copyable_filers_means_copy_nothing():
    ranking = _ranking([(1, "Lucky Lou", "short", False)])
    plan = allocate_sleeves(ranking, _trades({1: ["2023-01-01"]}))
    assert plan.assignments.empty
    assert plan.tfsa_deployed == 0.0
    assert any("copy nothing" in n for n in plan.notes)


def test_empty_ranking():
    plan = allocate_sleeves(pd.DataFrame(), pd.DataFrame())
    assert plan.assignments.empty


def test_weights_capped_and_within_capital():
    ranking = _ranking([(i, f"F{i}", "long", True) for i in range(1, 4)])
    trades = _trades({i: ["2022-06-01", "2023-06-01"] for i in range(1, 4)})
    plan = allocate_sleeves(ranking, trades, tfsa_room=30_000, taxable_capital=0.0,
                            max_weight=0.10)
    w = plan.assignments["weight_cad"]
    assert (w <= 30_000 * 0.10 + 1e-9).all()          # cap binds (3 filers, 10% cap)
    assert plan.tfsa_deployed <= 30_000
