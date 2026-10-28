"""Report generation: ranking.csv, decay plots, walk-forward equity vs the
low-effort benchmarks, amount-range sensitivity, taxlots.csv, README.md.

Everything regenerates deterministically from the DB (plan P5 acceptance).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from copytrade.backtest import benchmarks as bm
from copytrade.backtest import metrics
from copytrade.backtest.costs import get_cost_model
from copytrade.backtest.engine import run_event_study, walk_forward
from copytrade.rank.rank import build_ranking
from copytrade.store import db as store

# Reference palette (dataviz skill): categorical slots in fixed order + chrome
SERIES = ["#2a78d6", "#008300", "#e87ba4", "#eda100", "#1baf7a", "#eb6834"]
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"


def _fig(figsize=(9, 5)):
    fig, ax = plt.subplots(figsize=figsize, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=9)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_color(MUTED)
    return fig, ax


def _finish(fig, ax, title: str, path: Path, ylabel: str = ""):
    ax.set_title(title, color=INK, fontsize=12, loc="left", pad=12)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    leg = ax.get_legend()
    if leg:
        leg.get_frame().set_facecolor(SURFACE)
        leg.get_frame().set_edgecolor(GRID)
        for text in leg.get_texts():
            text.set_color(INK2)
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


DECAY_LABELS = ("anchored at transaction (zero-lag, hypothetical)",
                "anchored at filing (copyable)",
                "alpha lost to disclosure lag")


def plot_decay(curve_tx: np.ndarray, curve_filing: np.ndarray, title: str, path: Path,
               labels: tuple[str, str, str] = DECAY_LABELS):
    """The project's key chart: mean cumulative excess return in event time,
    two cohorts/anchors compared. The shaded gap is the difference."""
    fig, ax = _fig()
    x = np.arange(len(curve_tx))
    ax.axhline(0, color=BASELINE, linewidth=1)
    ax.fill_between(x, curve_tx * 100, curve_filing * 100, color=GRID, alpha=0.6,
                    label=labels[2])
    ax.plot(x, curve_tx * 100, color=SERIES[0], linewidth=2, label=labels[0])
    ax.plot(x, curve_filing * 100, color=SERIES[1], linewidth=2, label=labels[1])
    for curve, color in ((curve_tx, SERIES[0]), (curve_filing, SERIES[1])):
        if np.isfinite(curve[-1]):
            ax.annotate(f"{curve[-1] * 100:+.1f}%", (x[-1], curve[-1] * 100),
                        xytext=(6, 0), textcoords="offset points",
                        color=INK, fontsize=9, va="center")
    ax.set_xlabel("trading days after signal", color=INK2, fontsize=9)
    ax.legend(fontsize=8, loc="upper left")
    _finish(fig, ax, title, path, ylabel="mean cumulative excess return vs SPY (%)")


def plot_equity(curves: pd.DataFrame, title: str, path: Path):
    fig, ax = _fig()
    ax.axhline(1.0, color=BASELINE, linewidth=1)
    for i, col in enumerate(curves.columns):
        s = curves[col].dropna()
        color = SERIES[i % len(SERIES)]
        ax.plot(s.index, s.values, color=color, linewidth=2, label=col)
        ax.annotate(f" {col}  {s.iloc[-1]:.2f}x", (s.index[-1], s.iloc[-1]),
                    color=INK, fontsize=8, va="center")
    ax.legend(fontsize=8, loc="upper left")
    _finish(fig, ax, title, path, ylabel="growth of 1.0 (net of costs)")


def horizon_table(es_per_trade: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    """Pooled per-horizon stats, buys and sells separately (plan §4.2)."""
    rows = []
    for side, g in es_per_trade.groupby("side"):
        for h in horizons:
            col = f"h{h}"
            if col not in g:
                continue
            vals = g[col].dropna()
            if vals.empty:
                continue
            t, ncl = metrics.clustered_tstat(g[col], g["entry_date"])
            rows.append({
                "side": side, "horizon_td": h, "n": len(vals),
                "mean_excess": vals.mean(), "median_excess": vals.median(),
                "hit_rate": metrics.hit_rate(vals), "tstat_month_clustered": t,
            })
    return pd.DataFrame(rows)


def generate_report(conn, settings, out_dir: str | Path | None = None) -> dict:
    out = Path(out_dir or settings.reports_dir)
    out.mkdir(parents=True, exist_ok=True)
    cost = get_cost_model(settings.cost_model)

    trades = store.trades_frame(conn, instruments=["stock"])
    if trades.empty:
        raise RuntimeError("no trades in DB, run a backfill or `demo` first")
    prices = store.load_price_matrix(conn)
    if settings.benchmark not in prices.columns:
        raise RuntimeError(f"benchmark {settings.benchmark} has no cached prices, run `prices`")
    bench = prices[settings.benchmark].dropna()

    common = dict(
        same_day_close=settings.same_day_close, cutoff=settings.same_day_cutoff,
        amount_basis=settings.amount_basis, trade_notional=settings.trade_notional,
    )

    # ---- ranking -----------------------------------------------------------
    ranking, diag = build_ranking(
        trades, prices, bench, cost, horizon=settings.horizon,
        min_trades_rank=settings.min_trades_rank, **common,
    )
    ranking.to_csv(out / "ranking.csv", index=False)

    # ---- pooled decay curves per cohort ------------------------------------
    buys = trades[trades["side"] == "buy"]
    max_h = min(126, max(settings.horizons))
    es_f = run_event_study(buys, prices, bench, cost, settings.horizons,
                           anchor="filing", **common)
    es_t = run_event_study(buys, prices, bench, cost, settings.horizons,
                           anchor="transaction", **common)
    for kind in sorted(buys["filer_kind"].dropna().unique()):
        mf = (es_f.per_trade["kind"] == kind).values if not es_f.per_trade.empty else None
        mt = (es_t.per_trade["kind"] == kind).values if not es_t.per_trade.empty else None
        if mf is None or mt is None or mf.sum() == 0:
            continue
        plot_decay(
            es_t.decay_curve(mt)[: max_h + 1], es_f.decay_curve(mf)[: max_h + 1],
            f"Signal decay: {kind} buys (n={int(mf.sum())}, gross excess vs SPY)",
            out / f"decay_{kind}.png",
        )

    # ---- per-filer decay for the top 20 ------------------------------------
    top_filers = ranking.head(20) if not ranking.empty else pd.DataFrame()
    for r in top_filers.itertuples():
        mf = (es_f.per_trade["filer_id"] == r.filer_id).values
        mt = (es_t.per_trade["filer_id"] == r.filer_id).values
        if mf.sum() < 5:
            continue
        plot_decay(
            es_t.decay_curve(mt)[: max_h + 1], es_f.decay_curve(mf)[: max_h + 1],
            f"Signal decay: {r.filer} (n={int(mf.sum())})",
            out / f"decay_filer_{r.filer_id}.png",
        )

    # ---- cluster buys vs solo buys (consensus amplification check) ----------
    from copytrade.rank.signals import cluster_sizes

    cluster_stats = None
    if not es_f.per_trade.empty:
        pt_f = es_f.per_trade
        buys_priced = buys.set_index("trade_id").loc[
            pt_f["trade_id"], ["ticker", "side", "filer_id", "filing_datetime"]
        ].reset_index()
        csize = cluster_sizes(buys_priced).values
        is_cluster = csize >= 2
        if is_cluster.any() and (~is_cluster).any():
            plot_decay(
                es_f.decay_curve(pd.Series(is_cluster))[: max_h + 1],
                es_f.decay_curve(pd.Series(~is_cluster))[: max_h + 1],
                f"Cluster buys (≥2 filers/10d, n={int(is_cluster.sum())}) vs "
                f"solo buys (n={int((~is_cluster).sum())}), filing-anchored",
                out / "decay_cluster_vs_solo.png",
                labels=("cluster buys (≥2 distinct filers/10d)",
                        "solo buys", "consensus premium"),
            )
            col = f"h{settings.horizon}"
            cluster_stats = {
                "cluster_n": int(is_cluster.sum()),
                "solo_n": int((~is_cluster).sum()),
                "cluster_mean": float(pt_f.loc[is_cluster, col].mean()),
                "solo_mean": float(pt_f.loc[~is_cluster, col].mean()),
            }

    # ---- cohort decays: party (NANC/KRUZ-style) and disclosure speed --------
    if not es_f.per_trade.empty:
        pt_f = es_f.per_trade
        if "filer_party" in buys.columns:
            party_of = buys.set_index("trade_id")["filer_party"]
            parties = pt_f["trade_id"].map(party_of)
            dem = (parties == "Democrat").values
            rep = (parties == "Republican").values
            if dem.sum() >= 20 and rep.sum() >= 20:
                plot_decay(
                    es_f.decay_curve(pd.Series(dem))[: max_h + 1],
                    es_f.decay_curve(pd.Series(rep))[: max_h + 1],
                    f"Party cohorts: Democrat buys (n={int(dem.sum())}) vs "
                    f"Republican buys (n={int(rep.sum())}), filing-anchored",
                    out / "decay_party.png",
                    labels=("Democrat buys", "Republican buys", "party gap"),
                )
        fast = (pt_f["lag_days"] <= 15).values
        slow = (pt_f["lag_days"] > 15).values
        if fast.sum() >= 20 and slow.sum() >= 20:
            plot_decay(
                es_f.decay_curve(pd.Series(fast))[: max_h + 1],
                es_f.decay_curve(pd.Series(slow))[: max_h + 1],
                f"Disclosure speed: filed ≤15 bd after trade (n={int(fast.sum())}) vs "
                f">15 bd (n={int(slow.sum())})",
                out / "decay_lag_speed.png",
                labels=("fast filers (≤15 bd lag)", "slow filers (>15 bd lag)",
                        "cost of slow disclosure"),
            )

    # ---- pooled horizon stats table (buys AND sells) ------------------------
    es_all = run_event_study(trades[trades["side"].isin(["buy", "sell"])],
                             prices, bench, cost, settings.horizons,
                             anchor="filing", **common)
    htable = horizon_table(es_all.per_trade, settings.horizons)
    htable.to_csv(out / "horizon_stats.csv", index=False)

    # ---- walk-forward portfolio vs low-effort benchmarks --------------------
    wf_args = dict(
        horizon=settings.horizon, top_n=settings.top_n,
        min_trades=settings.min_trades_walkforward, max_weight=settings.max_weight,
        same_day_close=settings.same_day_close, cutoff=settings.same_day_cutoff,
        trade_notional=settings.trade_notional,
    )
    wf = walk_forward(trades, prices, bench, cost,
                      amount_basis=settings.amount_basis, **wf_args)
    stats: dict[str, dict] = {}
    outperf_p = None
    if wf is not None:
        curves = bm.benchmark_suite(
            prices, cost, start=wf.equity.index[0], end=wf.equity.index[-1],
            spy=settings.benchmark,
        )
        curves.insert(0, f"CopyTrade top-{settings.top_n}", wf.equity)
        # copy-the-exit variant: same selection, but positions close when the
        # filer files a sell (fixed horizon stays the backstop)
        wf_exit = walk_forward(trades, prices, bench, cost,
                               amount_basis=settings.amount_basis,
                               exit_on_filer_sell=True, **wf_args)
        if wf_exit is not None:
            curves["CopyTrade + copy-the-exit"] = wf_exit.equity
        # 'congress index' baseline: copy EVERYONE, no selection, if top-N
        # can't beat this, the ranking layer adds nothing
        wf_all = walk_forward(trades, prices, bench, cost,
                              amount_basis=settings.amount_basis,
                              copy_all=True, **wf_args)
        if wf_all is not None:
            curves["Copy ALL filers (no selection)"] = (
                wf_all.equity.reindex(curves.index).ffill()
            )
        plot_equity(curves, "Walk-forward: copy the top filers vs low-effort strategies",
                    out / "walkforward_equity.png")
        for col in curves.columns:
            stats[col] = metrics.perf_stats(curves[col].dropna())
        bench_curve = f"{settings.benchmark} buy & hold"
        if bench_curve in curves.columns:
            outperf_p = metrics.bootstrap_outperf_pvalue(
                wf.equity.pct_change().dropna(),
                curves[bench_curve].pct_change().dropna(),
            )
        wf.taxlots.to_csv(out / "taxlots.csv", index=False)
        picks = wf.picks
    else:
        curves = pd.DataFrame()
        picks = {}

    # ---- per-filer track record (profits, NO significance gate) -------------
    from copytrade.rank.profit import filer_track_records

    track = filer_track_records(
        trades, prices, bench, cost,
        same_day_close=settings.same_day_close, cutoff=settings.same_day_cutoff,
    )
    track.drop(columns=["positions"], errors="ignore").to_csv(
        out / "track_records.csv", index=False)

    # ---- TFSA / margin sleeve allocation (Canadian resident) ----------------
    from copytrade.accounts.allocation import allocate_sleeves

    plan = allocate_sleeves(ranking, trades)
    plan.assignments.to_csv(out / "allocation.csv", index=False)

    # ---- congress amount-range sensitivity (low/mid/high) -------------------
    sens = {}
    for basis in ("low", "mid", "high"):
        w = walk_forward(trades, prices, bench, cost, amount_basis=basis, **wf_args)
        if w is not None:
            sens[f"amount={basis}"] = w.equity
    if sens:
        plot_equity(pd.DataFrame(sens),
                    "Sensitivity: congress amount ranges taken at low / mid / high",
                    out / "sensitivity_amounts.png")

    # ---- README --------------------------------------------------------------
    _write_readme(out, settings, cost, diag, ranking, htable, stats, picks,
                  es_f.unpriceable, wf, plan, cluster_stats, outperf_p, track)
    return {
        "ranking_rows": len(ranking),
        "diagnostics": diag,
        "stats": stats,
        "picks": picks,
        "files": sorted(p.name for p in out.iterdir()),
    }


def _write_readme(out, settings, cost, diag, ranking, htable, stats, picks,
                  unpriceable, wf, plan=None, cluster_stats=None, outperf_p=None,
                  track=None):
    def pct(x):
        return f"{x:+.2%}" if pd.notna(x) else "n/a"

    lines = [
        "# CopyTrade report",
        "",
        "Generated deterministically from the local DB. Point-in-time discipline:",
        "every backtested entry executes at the first market close AFTER the filing",
        "timestamp: never at the transaction date.",
        "",
        f"- cost model: **{cost.name}** (spread/slippage {cost.spread_slippage_bps}bps/side, "
        f"commission ${cost.commission_per_trade}/trade + ${cost.commission_per_share}/share "
        f"(min ${cost.commission_min}), FX {cost.fx_bps}bps/side, SEC fee on sells: {cost.sec_fee_on_sells})",
        f"- execution: {'same-day close if filed before ' + settings.same_day_cutoff + ' ET, else ' if settings.same_day_close else ''}next trading day close after filing",
        f"- congress amounts: ranges, taken at **{settings.amount_basis}** here; see sensitivity_amounts.png",
        "",
        "## Data",
        f"- trades considered: {diag.get('trades_total', 0)} buys "
        f"({diag.get('priced', 0)} priced, **{diag.get('unpriceable', 0)} unpriceable/delisted, "
        "survivorship caveat: unpriceable names are EXCLUDED, results are upward-biased "
        "to the extent losers delist**)",
        "",
        "## Walk-forward portfolio (net of costs)",
        "",
    ]
    if stats:
        lines.append("| strategy | total | CAGR | vol | Sharpe | Sortino | max DD |")
        lines.append("|---|---|---|---|---|---|---|")
        for name, s in stats.items():
            if not s:
                continue
            sortino = f"{s['sortino']:.2f}" if pd.notna(s.get("sortino")) else "n/a"
            lines.append(
                f"| {name} | {pct(s['total_return'])} | {pct(s['cagr'])} | "
                f"{s['vol']:.1%} | {s['sharpe']:.2f} | {sortino} | {s['max_drawdown']:.1%} |"
            )
        if outperf_p is not None and pd.notna(outperf_p):
            lines.append("")
            lines.append(
                f"_Bootstrap p-value that the top-N excess over the benchmark is luck: "
                f"**{outperf_p:.3f}** (2000 resamples of daily excess returns; "
                "small = harder to attribute to noise; selection bias is already "
                "handled by the walk-forward)._"
            )
        if wf is not None and (wf.skipped_full or wf.stale_exits):
            lines.append("")
            if wf.skipped_full:
                lines.append(f"_{wf.skipped_full} signals skipped because the book was fully invested._")
            if wf.stale_exits:
                lines.append(f"_{wf.stale_exits} exits priced off a forward-filled close "
                             "(no real print that day, possible delisting; treat those fills skeptically)._")
    else:
        lines.append("_Not enough history for a walk-forward run (needs 2 training years)._")
    if picks:
        lines += ["", "### Picks per test year", ""]
        for y, names in picks.items():
            lines.append(f"- **{y}**: {', '.join(names) if names else '(none qualified)'}")

    if cluster_stats:
        lines += [
            "", "## Cluster buys vs solo buys", "",
            f"≥2 distinct filers buying the same ticker within 10 days "
            f"(n={cluster_stats['cluster_n']}) vs solo buys (n={cluster_stats['solo_n']}), "
            f"net {settings.horizon}d excess: **{pct(cluster_stats['cluster_mean'])}** vs "
            f"**{pct(cluster_stats['solo_mean'])}**, see decay_cluster_vs_solo.png. "
            "Consensus buying is the classic amplifier of the insider signal; if the "
            "premium is absent here, weight cluster alerts accordingly.",
        ]
    if not ranking.empty and "routine" in ranking.columns and ranking["routine"].any():
        lines += [
            "",
            f"_{int(ranking['routine'].sum())} of {len(ranking)} ranked filers are "
            "flagged **routine** (same-calendar-month filers, Cohen-Frazzini-Malloy "
            "2012): their trades are calendar-driven and historically carry little "
            "information; discount their alpha._",
        ]

    if track is not None and not track.empty:
        lines += [
            "", "## Filer track records, profits at any frequency", "",
            "No significance gate (unlike 'copyable'): a filer who trades a few "
            "times a year but picks well surfaces here. `confidence` flags the "
            "sample size; returns are net position returns, point-in-time "
            "(entry AND exit at the first close after the filing).",
            "",
            "| filer | kind | party | buys | closed | win | mean/pick | $ per 10k/pick | ann/pick | hold(bd) | conf |",
            "|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for r in track.head(25).itertuples():
            # party is NaN (float) for un-enriched filers (e.g. House members)
            # `NaN or ''` returns NaN (NaN is truthy), so guard with notna
            party3 = str(r.party)[:3] if pd.notna(r.party) else ""
            lines.append(
                f"| {r.filer} | {r.kind} | {party3} | {r.n_buys_priced} | "
                f"{r.n_closed} | {r.win_rate:.0%} | {pct(r.mean_return)} | "
                f"${r.total_pnl_per_10k:,.0f} | {pct(r.annualized_per_pick)} | "
                f"{r.avg_holding_days:.0f} | {r.confidence} |"
            )

    if not htable.empty:
        lines += ["", "## Pooled horizon stats (net, filing-anchored)", "",
                  "| side | horizon | n | mean | median | hit | t (clustered) |", "|---|---|---|---|---|---|---|"]
        for r in htable.itertuples():
            lines.append(
                f"| {r.side} | {r.horizon_td}d | {r.n} | {pct(r.mean_excess)} | "
                f"{pct(r.median_excess)} | {r.hit_rate:.0%} | {r.tstat_month_clustered:.2f} |"
            )

    lines += [
        "",
        "## Methodology caveats (read before acting on anything)",
        "",
        "- **Congress amounts are ranges, not sizes.** Backtests use the "
        f"{settings.amount_basis} of the range; the sensitivity plot shows low/mid/high.",
        "- **Options and non-stock instruments are excluded** from backtests (stored, flagged).",
        "- **Only Form 4 codes P (buy) and S (sell) are treated as signals**; awards, "
        "gifts, option exercises are stored but unsided.",
        "- **Survivorship**: delisted/unmappable tickers are excluded and counted above.",
        "- **Adjusted close is a total-return proxy**; SPY adjusted close is the benchmark.",
        "- **Multiple-hypothesis risk**: with hundreds of filers, lucky streaks are "
        "expected. 'Copyable' requires month-clustered t>2 and n>="
        f"{settings.min_trades_rank}; portfolio results are walk-forward.",
        "- **Decay curves are gross** of costs (information geometry); tables and "
        "equity curves are net.",
        "- **13F cloning** sees quarter-end longs only, no shorts, no timing, "
        "window-dressing risk.",
        "- **Canadian residents**: allocation.csv splits copyable filers into "
        "TFSA vs margin sleeves. Frequent short-hold trading inside a TFSA risks "
        "CRA reassessment as business income (folio S3-F9-C1); US dividends in a "
        "TFSA lose 15% to unrecoverable withholding. taxlots.csv gives per-lot "
        "P&L for ACB/capital-gains records.",
    ]
    if plan is not None and plan.notes:
        lines += ["", "### Account-sleeve notes", ""]
        lines += [f"- {n}" for n in plan.notes]
    # utf-8: the report body contains →, ², and ≥, cp1252 would raise on Windows
    (out / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
