"""Regression: _write_readme must not crash when a filer's party is NaN.

CI run #7 (House 2021-2026 + senate) died with 'float object is not
subscriptable': House filers are un-enriched so their party column is NaN
(a float), and `(NaN or '')[:3]` blew up because NaN is truthy in Python."""

from types import SimpleNamespace

import numpy as np
import pandas as pd

from copytrade.backtest.costs import get_cost_model
from copytrade.report.report import _write_readme


def test_write_readme_survives_nan_party(tmp_path):
    settings = SimpleNamespace(
        amount_basis="mid", horizon=63, min_trades_rank=30,
        same_day_close=False, same_day_cutoff="15:30",
    )
    track = pd.DataFrame([
        # a real (enriched) senator and an un-enriched House filer (party NaN)
        {"filer": "Ron L Wyden", "kind": "congress", "party": "Democrat",
         "n_buys_priced": 10, "n_closed": 3, "win_rate": 0.9, "mean_return": 0.2,
         "total_pnl_per_10k": 20000.0, "annualized_per_pick": 0.25,
         "avg_holding_days": 240.0, "confidence": "medium"},
        {"filer": "Nancy Pelosi", "kind": "congress", "party": np.nan,
         "n_buys_priced": 5, "n_closed": 2, "win_rate": 0.8, "mean_return": 0.3,
         "total_pnl_per_10k": 15000.0, "annualized_per_pick": 0.35,
         "avg_holding_days": 200.0, "confidence": "low"},
    ])
    empty = pd.DataFrame()
    _write_readme(tmp_path, settings, get_cost_model("ibkr_ca"), {}, empty, empty,
                  {}, {}, 0, None, track=track)
    text = (tmp_path / "README.md").read_text()
    assert "Nancy Pelosi" in text        # NaN-party row rendered, no crash
    assert "| Dem |" in text             # enriched party still truncated to 3
