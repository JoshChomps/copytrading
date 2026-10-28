"""Filer scoring: lag-adjusted alpha, significance, turnover class.

Multiple-hypothesis discipline (plan §7): a filer is only flagged 'copyable'
with clustered t-stat > 2 AND n_trades >= min_trades_rank, ranking hundreds
of filers guarantees lucky outliers otherwise.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from copytrade.backtest import metrics
from copytrade.backtest.costs import CostModel
from copytrade.backtest.engine import run_event_study
from copytrade.rank.signals import routine_filers


def build_ranking(
    trades: pd.DataFrame,
    prices: pd.DataFrame,
    bench: pd.Series,
    cost_model: CostModel,
    horizon: int = 63,
    min_trades_rank: int = 30,
    same_day_close: bool = False,
    cutoff: str = "15:30",
    amount_basis: str = "mid",
    trade_notional: float = 10_000.0,
) -> tuple[pd.DataFrame, dict]:
    """Returns (ranking df, diagnostics). Buys drive the score; the decay gap
    column is the per-filer alpha destroyed by the disclosure lag."""
    buys = trades[trades["side"] == "buy"]
    common = dict(
        prices=prices, bench=bench, cost_model=cost_model, horizons=[horizon],
        same_day_close=same_day_close, cutoff=cutoff,
        amount_basis=amount_basis, trade_notional=trade_notional,
    )
    es_filing = run_event_study(buys, anchor="filing", **common)
    es_tx = run_event_study(buys, anchor="transaction", **common)

    diagnostics = {
        "trades_total": int(es_filing.total),
        "unpriceable": int(es_filing.unpriceable),
        "priced": int(len(es_filing.per_trade)),
    }
    if es_filing.per_trade.empty:
        return pd.DataFrame(), diagnostics

    col = f"h{horizon}"
    rows = []
    tx_by_filer = (
        es_tx.per_trade.groupby("filer_id")[col].mean()
        if not es_tx.per_trade.empty else pd.Series(dtype=float)
    )
    routine = routine_filers(trades)
    for (filer_id, name, kind), g in es_filing.per_trade.groupby(
        ["filer_id", "filer_name", "kind"]
    ):
        vals = g[col].dropna()
        if vals.empty:
            continue
        t, n_clusters = metrics.clustered_tstat(g[col], g["entry_date"])
        ftrades = trades[trades["filer_id"] == filer_id]
        alpha_filing = float(vals.mean())
        alpha_tx = float(tx_by_filer.get(filer_id, np.nan))
        party = ftrades["filer_party"].iloc[0] if "filer_party" in ftrades and len(ftrades) else None
        rows.append({
            "filer_id": int(filer_id),
            "filer": name,
            "kind": kind,
            "party": party,
            "n_trades": int(len(vals)),
            f"lag_alpha_{horizon}d": alpha_filing,
            f"zero_lag_alpha_{horizon}d": alpha_tx,
            "alpha_lost_to_lag": (alpha_tx - alpha_filing) if not np.isnan(alpha_tx) else np.nan,
            "tstat": t,
            "n_month_clusters": n_clusters,
            "hit_rate": metrics.hit_rate(g[col]),
            "median_lag_days": float(g["lag_days"].median()),
            "turnover_class": metrics.turnover_class(ftrades),
            # Cohen-Frazzini-Malloy: same-calendar-month filers are calendar-
            # driven; their trades historically carry little information
            "routine": bool(routine.get(filer_id, False)),
            "copyable": bool(len(vals) >= min_trades_rank and not np.isnan(t) and t > 2),
        })
    ranking = pd.DataFrame(rows).sort_values(
        f"lag_alpha_{horizon}d", ascending=False
    ).reset_index(drop=True)
    return ranking, diagnostics
