import numpy as np
import pandas as pd

from copytrade.rank.persistence import (
    filer_window_table,
    persistence_pairs,
    persistence_report,
)


def _trades(spec, hcol="h63"):
    """spec: {filer: {year: [returns]}} -> per-trade frame."""
    rows = []
    for filer, years in spec.items():
        for year, rets in years.items():
            for i, r in enumerate(rets):
                rows.append({
                    "filer_name": filer,
                    "entry_date": pd.Timestamp(f"{year}-0{(i % 9) + 1}-15"),
                    hcol: r,
                })
    return pd.DataFrame(rows)


def test_window_table_drops_thin_filers():
    pt = _trades({"A": {2020: [0.01] * 6}, "B": {2020: [0.02] * 2}})
    tbl = filer_window_table(pt, "h63", freq="Y", min_trades=5)
    assert set(tbl["filer_name"]) == {"A"}, "B has too few trades to rank"


def test_pairs_only_include_filers_present_in_both_windows():
    pt = _trades({
        "A": {2020: [0.01] * 6, 2021: [0.01] * 6},
        "B": {2020: [0.02] * 6},                      # vanishes in 2021
        "C": {2021: [0.03] * 6},                      # appears only in 2021
        "D": {2020: [0.0] * 6, 2021: [0.0] * 6},
        "E": {2020: [0.01] * 6, 2021: [0.02] * 6},
        "F": {2020: [0.02] * 6, 2021: [0.01] * 6},
        "G": {2020: [0.03] * 6, 2021: [0.03] * 6},
    })
    tbl = filer_window_table(pt, "h63", freq="Y", min_trades=5)
    pairs = persistence_pairs(tbl)
    assert len(pairs) == 1
    _, _, merged = pairs[0]
    # B (gone by 2021) and C (absent in 2020) cannot be tested for persistence
    assert set(merged["filer_name"]) == {"A", "D", "E", "F", "G"}


def test_pair_is_dropped_when_too_few_filers_to_rank():
    # 4 common filers cannot be split into meaningful terciles -> no pair
    pt = _trades({
        "A": {2020: [0.01] * 6, 2021: [0.01] * 6},
        "D": {2020: [0.0] * 6, 2021: [0.0] * 6},
        "E": {2020: [0.01] * 6, 2021: [0.02] * 6},
        "F": {2020: [0.02] * 6, 2021: [0.01] * 6},
    })
    tbl = filer_window_table(pt, "h63", freq="Y", min_trades=5)
    assert persistence_pairs(tbl) == []


def _persistent_spec(years, n_filers=9, seed=0, persistent=True):
    """Filers with fixed skill (persistent) or reshuffled each year (not)."""
    rng = np.random.default_rng(seed)
    skills = np.linspace(-0.04, 0.04, n_filers)
    spec = {}
    for i in range(n_filers):
        spec[f"F{i}"] = {}
    for y in years:
        order = skills if persistent else rng.permutation(skills)
        for i in range(n_filers):
            spec[f"F{i}"][y] = list(order[i] + rng.normal(0, 0.01, 8))
    return spec


def test_detects_persistence_when_skill_is_real_and_fixed():
    pt = _trades(_persistent_spec(range(2014, 2026), persistent=True, seed=1))
    rep = persistence_report(pt, "h63", freq="Y", min_trades=5)
    assert rep["pairs"] >= 5
    assert rep["mean_spearman"] > 0.5
    assert rep["t_spread"] > 2, "fixed skill must show a significant tercile spread"


def test_reports_no_persistence_when_ranking_is_luck():
    # Skill reshuffled every year = past rank carries no information.
    pt = _trades(_persistent_spec(range(2014, 2026), persistent=False, seed=2))
    rep = persistence_report(pt, "h63", freq="Y", min_trades=5)
    assert rep["pairs"] >= 5
    assert abs(rep["mean_spearman"]) < 0.35
    assert abs(rep["t_spread"]) < 2, "reshuffled luck must not look persistent"


def test_report_handles_insufficient_data_without_crashing():
    pt = _trades({"A": {2020: [0.01] * 6}})
    rep = persistence_report(pt, "h63", freq="Y", min_trades=5)
    assert rep["pairs"] == 0
    assert rep["detail"].empty


def test_spread_is_computed_out_of_sample():
    # Rank must come from TRAIN, measurement from TEST. Construct a case where
    # the train-best filer is the test-worst: the spread must be negative,
    # which an accidentally in-sample implementation could never produce.
    spec = {}
    for i in range(9):
        spec[f"F{i}"] = {2020: [0.05 - 0.01 * i] * 8,      # F0 best in train
                         2021: [0.01 * i] * 8}             # F0 worst in test
    pt = _trades(spec)
    rep = persistence_report(pt, "h63", freq="Y", min_trades=5)
    assert rep["pairs"] == 1
    assert rep["detail"]["spread"].iloc[0] < 0
    assert rep["detail"]["spearman"].iloc[0] < 0
