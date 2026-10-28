"""Precomputed analysis payloads for the dashboard.

WHY A CACHE LAYER. The analyses this dashboard exists to show are slow: the
timescale sweep runs an event study over 12 horizons, `leverage` runs a
thousands-of-paths bootstrap per candidate per leverage rung, and `indicators`
builds a feature matrix plus a matched control. Computing any of them inside a
request handler would mean a page load that takes minutes and a browser that
times out first.

So each analysis is computed by an explicit build step (`copytrade
dashboard-build`) and written to `reports/dashboard/<name>.json`. The API then
serves whatever is on disk, with the build timestamp attached so the UI can
show how stale it is rather than pretending the numbers are live.

Every payload carries a `verdict` string. That is deliberate: the whole point
of this project is that the headline numbers are easy to misread, and a table
without its interpretation invites exactly the misreadings the review found
(a "large trades" subset that was really all trades, a CAD uplift that was
really a shorter window, a 3x leverage row that was really a liquidation).
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timedelta, timezone
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

from copytrade.backtest.costs import get_cost_model
from copytrade.config import load_settings
from copytrade.store import db as store


def cache_dir() -> Path:
    d = Path(load_settings().reports_dir) / "dashboard"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _clean(obj):
    """JSON-safe: NaN/Inf are not valid JSON and silently become nulls that the
    front end renders as 0, which is how a missing statistic turns into a
    fabricated one."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return None if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    if obj is None or isinstance(obj, (str, int)):
        return obj
    return str(obj)


def save(name: str, payload: dict) -> Path:
    payload = dict(payload)
    payload["built_at"] = datetime.now(timezone.utc).isoformat()
    p = cache_dir() / f"{name}.json"
    p.write_text(json.dumps(_clean(payload), indent=1), encoding="utf-8")
    return p


def load(name: str) -> dict | None:
    p = cache_dir() / f"{name}.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except ValueError:
        return None


def was_built_since(name: str, when: datetime) -> bool:
    """Did `name` get written by a run that started at `when`?

    Read from the payload's own `built_at` rather than the file mtime, because
    the mtime says when the bytes were touched and `built_at` says when the
    numbers were computed: and a panel copied, synced or restored carries a
    fresh mtime over stale contents.
    """
    payload = load(name)
    if not payload:
        return False
    stamp = payload.get("built_at")
    if not stamp:
        return False
    try:
        built = datetime.fromisoformat(stamp)
    except ValueError:
        return False
    if built.tzinfo is None:
        built = built.replace(tzinfo=timezone.utc)
    # `save` stamps built_at AFTER `when` is taken, so in the normal path no
    # slack is needed. One second absorbs a clock that steps backwards mid-build
    # (an NTP correction) without opening a hole worth caring about: the most it
    # can wave through is a panel written in the second before the run began.
    return built >= when - timedelta(seconds=1)


# ----------------------------------------------------------------- builders --

def _base():
    settings = load_settings()
    conn = store.connect(settings.db_path)
    store.run_migrations(conn)
    return conn, settings


# --------------------------------------------------------- shared workload --
#
# `dashboard-build` runs ~20 builders back to back and most of them need the
# same three expensive objects: the price matrix (7.6M rows), the copy-ALL
# walk-forward basket, and the filing-anchored event study over congress buys.
# Rebuilding those per panel turned a build into something that ran for the
# better part of an hour. They are pure functions of the DB, so memoize them
# for the life of the process, `dashboard-build` is a one-shot command, and
# the API never calls a builder at all (it serves the JSON from disk).

_SHARED: dict = {}


def _shared(key, factory):
    if key not in _SHARED:
        _SHARED[key] = factory()
    return _SHARED[key]


def reset_shared() -> None:
    """Drop the memo. Tests build against several different databases in one
    process, and a cached price matrix from the previous one would silently
    answer for the next."""
    _SHARED.clear()


def _db_key() -> str:
    """Every memo key carries the database it was computed from.

    Keying on the name alone would be wrong the moment a second database is in
    play: which is exactly what the test suite does, building panels against
    an empty DB right after a populated one. A cached price matrix answering
    for the wrong database is the quietest possible way to fabricate results.
    """
    return str(load_settings().db_path)


def _ctx():
    """(conn, settings, cost, prices, bench), the common preamble."""
    def build():
        conn, settings = _base()
        cost = get_cost_model(settings.cost_model)
        prices = store.load_price_matrix(conn)
        bench = (prices[settings.benchmark].dropna()
                 if settings.benchmark in prices.columns else pd.Series(dtype=float))
        return conn, settings, cost, prices, bench
    return _shared(("ctx", _db_key()), build)


def _trades(**kw):
    conn, *_ = _ctx()
    key = ("trades", _db_key(),
           tuple(sorted((k, tuple(v) if isinstance(v, list) else v)
                        for k, v in kw.items())))
    return _shared(key, lambda: store.trades_frame(conn, **kw))


def _congress_buys():
    return _trades(kinds=["congress"], sides=["buy"], instruments=["stock"])


def _event_study(horizons, anchor="filing", kinds=("congress",), sides=("buy",)):
    """Memoized `run_event_study`. Keyed on everything that changes the result."""
    from copytrade.backtest.engine import run_event_study

    _, settings, cost, prices, bench = _ctx()
    key = ("es", _db_key(), tuple(sorted(horizons)), anchor,
           tuple(kinds or ()), tuple(sides or ()))

    def build():
        tr = _trades(kinds=list(kinds) if kinds else None,
                     sides=list(sides) if sides else None, instruments=["stock"])
        if tr.empty or bench.empty:
            return None
        return run_event_study(
            tr, prices, bench, cost, list(horizons), anchor=anchor,
            same_day_close=settings.same_day_close, cutoff=settings.same_day_cutoff,
            amount_basis=settings.amount_basis, trade_notional=settings.trade_notional)
    return _shared(key, build)


def _copy_all():
    """The copy-everyone walk-forward basket, the object almost every headline
    number is computed from."""
    from copytrade.backtest.engine import walk_forward

    _, settings, cost, prices, bench = _ctx()

    def build():
        trades = _trades(instruments=["stock"])
        if trades.empty or bench.empty:
            return None
        return walk_forward(
            trades, prices, bench, cost, copy_all=True, min_trades=1,
            horizon=settings.horizon, max_weight=settings.max_weight,
            trade_notional=settings.trade_notional,
            same_day_close=settings.same_day_close, cutoff=settings.same_day_cutoff)
    return _shared(("copy_all", _db_key()), build)


def build_health() -> dict:
    """Data coverage and freshness. First panel on the dashboard because every
    other number is worthless if the underlying sample is stale or thin."""
    conn, settings = _base()
    q = lambda s: conn.execute(s).fetchone()[0]  # noqa: E731

    sources = [{
        "source": r[0], "trades": r[1],
        "first": str(r[2])[:10], "last": str(r[3])[:10],
    } for r in conn.execute(
        "SELECT source, COUNT(*), MIN(filing_datetime), MAX(filing_datetime) "
        "FROM trades GROUP BY source ORDER BY 2 DESC")]

    # `priced` must be the INTERSECTION with the traded universe, not the size
    # of the price table. `prices` also holds benchmarks (SPY/QQQ/VTI/IWM/SCHD/
    # RSP/TLT) and tickers that were only ever traded as options, none of which
    # are in the instrument='stock' denominator, so counting the table made
    # `priced` and `missing` answer different questions and the panel published
    # an impossible aggregate: 2,990 priced + 631 missing = 3,621 against a
    # universe of 3,335. §2.8: an impossible aggregate is a data bug.
    #
    # It was not cosmetic. The inflated ratio (89.7% against a true 81.1%)
    # cleared the `cov < 0.85` gate below, so the FIRST panel on the dashboard
    # reported verdict "ok" with no notes at a coverage level it exists to
    # flag: §7.4's failure mode, a status reading "done" that means "the
    # provider is refusing".
    _TRADED_UNIVERSE = ("SELECT DISTINCT ticker FROM trades "
                        "WHERE ticker IS NOT NULL AND instrument='stock'")
    traded = q(f"SELECT COUNT(*) FROM ({_TRADED_UNIVERSE})")
    priced = q(f"SELECT COUNT(*) FROM ({_TRADED_UNIVERSE} "
               "INTERSECT SELECT DISTINCT ticker FROM prices)")
    missing = q(f"SELECT COUNT(*) FROM ({_TRADED_UNIVERSE} "
                "EXCEPT SELECT DISTINCT ticker FROM prices)")

    benchmarks = []
    for t in ("SPY", "QQQ", "VTI", "IWM", "SCHD", "RSP", "TLT"):
        r = conn.execute("SELECT COUNT(*), MIN(date), MAX(date) FROM prices "
                         "WHERE ticker=?", (t,)).fetchone()
        benchmarks.append({"ticker": t, "rows": r[0],
                           "first": r[1], "last": r[2], "ok": bool(r[0] > 100)})

    fx = conn.execute("SELECT COUNT(*), MIN(date), MAX(date) FROM fx_rates "
                      "WHERE pair='USDCAD'").fetchone()

    stale_bench = [b["ticker"] for b in benchmarks if not b["ok"]]
    verdict = "ok"
    notes = []
    if stale_bench:
        verdict = "degraded"
        notes.append(f"benchmark(s) missing prices: {', '.join(stale_bench)}, "
                     f"every scorecard row that uses them is invalid")
    cov = priced / traded if traded else 0
    if cov < 0.85:
        verdict = "degraded" if verdict == "ok" else verdict
        notes.append(f"only {cov:.0%} of traded tickers are priced; the rest are "
                     f"a mix of genuine delistings and provider gaps")
    return {
        "verdict": verdict, "notes": notes,
        "trades": q("SELECT COUNT(*) FROM trades"),
        "filers": q("SELECT COUNT(*) FROM filers"),
        "sources": sources,
        "traded_tickers": traded, "priced_tickers": priced,
        "missing_prices": missing, "coverage": cov,
        "price_rows": q("SELECT COUNT(*) FROM prices"),
        "benchmarks": benchmarks,
        "fx": {"rows": fx[0], "first": fx[1], "last": fx[2]},
        "cost_model": settings.cost_model,
    }


def build_scorecard(cad: bool = False, ma_window: int = 200) -> dict:
    """Full metrics battery: copy basket, regime-filtered, and the ETFs a real
    investor would otherwise buy."""
    from copytrade.backtest import benchmarks as bm
    from copytrade.backtest import metrics
    from copytrade.backtest.currency import load_usdcad, to_cad
    from copytrade.backtest.engine import walk_forward
    from copytrade.backtest.regime import regime_equity

    conn, settings = _base()
    cost = get_cost_model(settings.cost_model)
    trades = store.trades_frame(conn, instruments=["stock"])
    prices = store.load_price_matrix(conn)
    if trades.empty or settings.benchmark not in prices.columns:
        return {"error": "no trades or benchmark prices"}
    bench = prices[settings.benchmark].dropna()
    wf = walk_forward(trades, prices, bench, cost, copy_all=True, min_trades=1,
                      horizon=settings.horizon, max_weight=settings.max_weight,
                      trade_notional=settings.trade_notional,
                      same_day_close=settings.same_day_close,
                      cutoff=settings.same_day_cutoff)
    if wf is None or wf.equity.empty:
        return {"error": "copy-all produced no equity curve"}
    basket = wf.equity.dropna()
    span0, span1 = basket.index[0], basket.index[-1]

    curves = {"Copy ALL": basket}
    reg, reg_info = regime_equity(basket, bench, ma_window=ma_window)
    if not reg.empty:
        curves[f"Copy ALL|regime{ma_window}"] = reg
    etfs = bm.etf_buy_hold(prices, cost, start=span0, end=span1)
    curves.update(etfs)
    for t in ("QQQ", "SPY"):
        if t in etfs:
            r, _ = regime_equity(etfs[t], bench, ma_window=ma_window)
            if not r.empty:
                curves[f"{t}|regime{ma_window}"] = r

    fx_note = None
    if cad:
        usdcad = load_usdcad(conn)
        if not usdcad.empty:
            conv, infos = {}, {}
            for k, v in curves.items():
                c, i = to_cad(v, usdcad)
                if not c.empty:
                    conv[k], infos[k] = c, i
            if conv:
                curves = conv
                ref = next(iter(infos.values()))
                span0, span1 = ref["cad_span"]
                bench_cad, binfo = to_cad(bench.loc[span0:span1], usdcad)
                bench = bench_cad
                usd_same = prices[settings.benchmark].dropna().loc[span0:span1]
                yrs = max((span1 - span0).days / 365.25, 1e-9)
                fx_note = {
                    "fx_return": binfo["fx_return"],
                    "usd_cagr_same_window": float(
                        (usd_same.iloc[-1] / usd_same.iloc[0]) ** (1 / yrs) - 1),
                    "dropped_days": ref["dropped_days"],
                }

    bench_rets = bench.loc[span0:span1].pct_change().dropna()
    rows = []
    for name, eq in curves.items():
        st = metrics.full_metrics(eq, bench_rets=bench_rets)
        if st:
            rows.append({"name": name, **st})
    return {
        "currency": "CAD" if (cad and fx_note) else "USD",
        "span": [str(span0.date()), str(span1.date())],
        "cost_model": settings.cost_model,
        "regime": reg_info, "fx": fx_note, "rows": rows,
        "verdict": _scorecard_verdict(rows),
    }


def _scorecard_verdict(rows: list[dict]) -> str:
    by = {r["name"]: r for r in rows}
    copy = by.get("Copy ALL")
    if not copy:
        return ""
    beats = [n for n, r in by.items()
             if n != "Copy ALL" and (r.get("total_return") or -9) > (copy.get("total_return") or 0)]
    a = copy.get("alpha_annualized")
    alpha_txt = (f"Jensen alpha {a:+.1%}" if a is not None else "alpha n/a")
    return (f"Copy ALL returns {copy.get('total_return', 0):+.0%} with {alpha_txt} "
            f"at beta {copy.get('beta', float('nan')):.2f}. "
            + (f"{len(beats)} other holdings beat it outright: {', '.join(beats[:5])}."
               if beats else "It leads the table."))


def build_timescale() -> dict:
    """The {subset x horizon} sweep, FDR-corrected."""
    from copytrade.backtest.engine import run_event_study
    from copytrade.rank.insiders import (
        abnormal_size_ids_pit,
        cluster_buy_ids,
        top_quartile_mask,
    )
    from copytrade.rank.timescale import ALL_HORIZONS, band_summary, horizon_sweep

    conn, settings = _base()
    buys = store.trades_frame(conn, kinds=["congress"], sides=["buy"],
                              instruments=["stock"])
    prices = store.load_price_matrix(conn)
    if buys.empty or settings.benchmark not in prices.columns:
        return {"error": "no data"}
    bench = prices[settings.benchmark].dropna()
    es = run_event_study(buys, prices, bench, get_cost_model(settings.cost_model),
                         list(ALL_HORIZONS), anchor="filing",
                         same_day_close=settings.same_day_close,
                         cutoff=settings.same_day_cutoff,
                         amount_basis=settings.amount_basis,
                         trade_notional=settings.trade_notional)
    pt = es.per_trade
    if pt.empty:
        return {"error": "no priceable buys"}
    big, big_info = top_quartile_mask(pt["amount"])
    subsets = {
        "ALL buys": pd.Series(True, index=pt.index),
        "cluster (>=3/30d)": pt["trade_id"].isin(cluster_buy_ids(buys, 30, 3)),
        "large ($ top quartile)": big,
        "abnormal (>=2x own median)": pt["trade_id"].isin(
            abnormal_size_ids_pit(buys, 2.0)),
    }
    grid = horizon_sweep(pt, subsets, list(ALL_HORIZONS))
    surv = grid[grid["survives_fdr"]]
    return {
        "n_buys": int(len(buys)), "n_priceable": int(len(pt)),
        "benchmark": settings.benchmark, "cost_model": settings.cost_model,
        "quartile_info": big_info,
        "grid": grid.to_dict(orient="records"),
        "bands": band_summary(grid).to_dict(orient="records"),
        "n_survivors": int(len(surv)),
        "verdict": (
            f"{len(surv)} of {int(grid['p'].notna().sum())} cells survive FDR"
            + (f", all in the {surv['band'].mode().iloc[0]} band and "
               f"{'negative' if (surv['mean'] < 0).all() else 'mixed'}, "
               f"the robust effect is the transaction cost, not a signal."
               if len(surv) else ", no horizon carries an edge.")),
    }


def build_exits() -> dict:
    from copytrade.backtest.engine import run_event_study
    from copytrade.rank.exits import rule_grid

    conn, settings = _base()
    h = settings.horizon
    buys = store.trades_frame(conn, kinds=["congress"], sides=["buy"],
                              instruments=["stock"])
    prices = store.load_price_matrix(conn)
    if buys.empty or settings.benchmark not in prices.columns:
        return {"error": "no data"}
    bench = prices[settings.benchmark].dropna()
    es = run_event_study(buys, prices, bench, get_cost_model(settings.cost_model),
                         [h], anchor="filing",
                         same_day_close=settings.same_day_close,
                         cutoff=settings.same_day_cutoff,
                         amount_basis=settings.amount_basis,
                         trade_notional=settings.trade_notional)
    if es.per_trade.empty:
        return {"error": "no priceable buys"}
    grid = rule_grid(es.paths, es.per_trade, baseline_h=h)
    winners = grid[grid["beats_baseline"]]
    return {
        "baseline_h": h, "rows": grid.to_dict(orient="records"),
        "n_winners": int(len(winners)),
        "verdict": ("No exit rule beats the fixed-horizon baseline after FDR, "
                    "exit engineering is not where the edge is."
                    if winners.empty else
                    f"{len(winners)} rule(s) beat the baseline after correction."),
    }


def build_leverage(paths: int = 1200, ma_window: int = 200) -> dict:
    from copytrade.backtest import benchmarks as bm
    from copytrade.backtest.engine import walk_forward
    from copytrade.backtest.leverage import (
        DEFAULT_FINANCING,
        DEFAULT_MAINTENANCE,
        MAINTENANCE_GRID,
        leverage_scan,
        max_safe_leverage,
    )
    from copytrade.backtest.regime import regime_equity

    conn, settings = _base()
    cost = get_cost_model(settings.cost_model)
    trades = store.trades_frame(conn, instruments=["stock"])
    prices = store.load_price_matrix(conn)
    if trades.empty or settings.benchmark not in prices.columns:
        return {"error": "no data"}
    bench = prices[settings.benchmark].dropna()
    wf = walk_forward(trades, prices, bench, cost, copy_all=True, min_trades=1,
                      horizon=settings.horizon, max_weight=settings.max_weight,
                      trade_notional=settings.trade_notional,
                      same_day_close=settings.same_day_close,
                      cutoff=settings.same_day_cutoff)
    if wf is None or wf.equity.empty:
        return {"error": "no equity curve"}
    basket = wf.equity.dropna()
    span0, span1 = basket.index[0], basket.index[-1]

    cands = {"Copy ALL": basket}
    r, _ = regime_equity(basket, bench, ma_window=ma_window)
    if not r.empty:
        cands[f"Copy ALL|regime{ma_window}"] = r
    for t, eq in bm.etf_buy_hold(prices, cost, start=span0, end=span1).items():
        if t in ("SPY", "QQQ"):
            cands[t] = eq
            rr, _ = regime_equity(eq, bench, ma_window=ma_window)
            if not rr.empty:
                cands[f"{t}|regime{ma_window}"] = rr

    out = []
    for name, eq in cands.items():
        rets = eq.pct_change().dropna()
        scan = leverage_scan(rets, financing_rate=DEFAULT_FINANCING,
                             maintenance=DEFAULT_MAINTENANCE, n_paths=paths)
        if scan.empty:
            continue
        sens = {}
        for m in MAINTENANCE_GRID:
            sc = leverage_scan(rets, financing_rate=DEFAULT_FINANCING,
                               maintenance=m, n_paths=max(400, paths // 4))
            if not sc.empty:
                sens[f"{m:.0%}"] = max_safe_leverage(sc)
        out.append({
            "name": name,
            "max_safe": max_safe_leverage(scan),
            "sensitivity": sens,
            "rows": scan.to_dict(orient="records"),
        })
    best = max(out, key=lambda c: c["max_safe"] and
               max(r["realised_total_return"] for r in c["rows"]
                   if r["leverage"] <= c["max_safe"]), default=None)
    return {
        "financing": DEFAULT_FINANCING, "maintenance": DEFAULT_MAINTENANCE,
        "paths": paths, "candidates": out,
        "verdict": (
            "Sizing must key off the bootstrapped margin-call probability, not "
            "the realised drawdown: a realised path that survived is one draw, "
            "and rows marked liquidated stopped participating from that day."
            + (f" Highest survivable return here: {best['name']} at "
               f"{best['max_safe']:.2f}x." if best else "")),
        "caveats": ["intraday gaps through the maintenance level",
                    "brokers raising maintenance in stress",
                    "borrow availability", "tax cost of a forced sale"],
    }


def build_persistence() -> dict:
    from copytrade.backtest.engine import run_event_study
    from copytrade.rank.persistence import persistence_report

    conn, settings = _base()
    h = settings.horizon
    buys = store.trades_frame(conn, kinds=["congress"], sides=["buy"],
                              instruments=["stock"])
    prices = store.load_price_matrix(conn)
    if buys.empty or settings.benchmark not in prices.columns:
        return {"error": "no data"}
    bench = prices[settings.benchmark].dropna()
    es = run_event_study(buys, prices, bench, get_cost_model(settings.cost_model),
                         [h], anchor="filing",
                         same_day_close=settings.same_day_close,
                         cutoff=settings.same_day_cutoff,
                         amount_basis=settings.amount_basis,
                         trade_notional=settings.trade_notional)
    if es.per_trade.empty:
        return {"error": "no priceable buys"}
    rep = persistence_report(es.per_trade, f"h{h}", freq="Y", min_trades=5)
    if rep["pairs"] == 0:
        return {"error": "insufficient data", "verdict": "not enough windows"}
    persists = abs(rep.get("t_spread") or 0) > 2 and (rep.get("mean_spread") or 0) > 0
    detail = rep["detail"]
    return {
        "horizon": h, "pairs": rep["pairs"],
        "mean_spearman": rep["mean_spearman"], "t_spearman": rep["t_spearman"],
        "mean_spread": rep["mean_spread"], "t_spread": rep["t_spread"],
        "pairs_positive": rep["pairs_positive"], "sign_p": rep["sign_p"],
        "detail": detail.to_dict(orient="records"),
        "persists": bool(persists),
        "verdict": ("Past filer edge predicts future edge, selection is worth "
                    "building on." if persists else
                    "No significant persistence: a filer's past edge does not "
                    "predict their next period. There is no stable 'who' to "
                    "copy, so copy-everyone is the correct design."),
    }


def build_jurisdiction() -> dict:
    """Committee jurisdiction vs traded sector, point-in-time where possible."""
    from copytrade.backtest.engine import run_event_study
    from copytrade.rank.jurisdiction import (
        build_pit_jurisdictions,
        classify_pit,
        compare,
        eligibility_pit,
        load_sector_map,
    )

    conn, settings = _base()
    h = settings.horizon
    buys = store.trades_frame(conn, kinds=["congress"], sides=["buy"],
                              instruments=["stock"])
    prices = store.load_price_matrix(conn)
    if buys.empty or settings.benchmark not in prices.columns:
        return {"error": "no data"}
    lookup, n_snapshots = build_pit_jurisdictions(conn)
    sectors = load_sector_map(conn)
    if n_snapshots < 2:
        return {"error": "run ingest-committees-history for point-in-time rosters"}
    if not sectors:
        return {"error": "sector_map is empty, run sectors-refresh"}

    bench = prices[settings.benchmark].dropna()
    es = run_event_study(buys, prices, bench, get_cost_model(settings.cost_model),
                         [h], anchor="filing",
                         same_day_close=settings.same_day_close,
                         cutoff=settings.same_day_cutoff,
                         amount_basis=settings.amount_basis,
                         trade_notional=settings.trade_notional)
    if es.per_trade.empty:
        return {"error": "no priceable buys"}

    elig = eligibility_pit(es.per_trade, lookup, sectors)
    lab = classify_pit(es.per_trade, lookup, sectors)
    if lab.empty:
        return {"error": "no eligible trades"}
    res = compare(lab, f"h{h}", h)
    n_in = res["in_jurisdiction"]["n"]
    blocks = min(res["in_jurisdiction"].get("blocks") or 0,
                 res["out_of_jurisdiction"].get("blocks") or 0)
    underpowered = n_in < 200 or blocks < 8 or res["paired_months"] < 24
    sig = (res["diff_t"] is not None and abs(res["diff_t"]) > 2
           and (res["diff_mean"] or 0) > 0)
    return {
        "horizon": h, "snapshots": n_snapshots, "sectors_mapped": len(sectors),
        "eligibility": elig, "underpowered": underpowered, **res,
        "verdict": (
            "Jurisdiction effect detected: treat as a lead to replicate."
            if sig else
            (f"UNDERPOWERED, not a null: only {n_in} in-jurisdiction trades over "
             f"{blocks} independent blocks." if underpowered else
             "No jurisdiction effect. Members' trades inside their own "
             "committee's remit do not beat their trades outside it, tested "
             "point-in-time against the roster in force on each filing date.")),
    }


# ------------------------------------------------- panels formerly computed --
#                                                    live by /api/analytics
#
# These four used to be built inside the request handler. On the live sample
# (41k trades, 7.6M price rows) that request never returned, the walk-forward
# alone runs an event study and then a day-by-day portfolio simulation over
# ~3,000 trading days, so the browser timed out and the four sections at the
# bottom of the dashboard were permanently blank. They are ordinary precomputed
# panels now, like everything under FINDINGS.

def _series(s: pd.Series) -> dict:
    s = s.dropna()
    return {"dates": [d.strftime("%Y-%m-%d") for d in s.index],
            "values": [float(v) for v in s.values]}


def build_walkforward() -> dict:
    """Copy-the-top-N walk-forward against the low-effort alternatives.

    Reports the copy-ALL basket alongside top-N deliberately: if selection
    cannot beat copying everyone, the ranking layer adds nothing, and that
    comparison is the entire point of the panel.
    """
    from copytrade.backtest import benchmarks as bm
    from copytrade.backtest import metrics
    from copytrade.backtest.engine import walk_forward

    conn, settings, cost, prices, bench = _ctx()
    trades = _trades(instruments=["stock"])
    if trades.empty or bench.empty:
        return {"error": "no trades or benchmark prices"}
    wf_all = _copy_all()
    if wf_all is None or wf_all.equity.empty:
        return {"error": "copy-all produced no equity curve"}
    span0, span1 = wf_all.equity.index[0], wf_all.equity.index[-1]
    common = dict(horizon=settings.horizon, max_weight=settings.max_weight,
                  trade_notional=settings.trade_notional,
                  same_day_close=settings.same_day_close,
                  cutoff=settings.same_day_cutoff)
    wf_top = walk_forward(trades, prices, bench, cost, top_n=settings.top_n,
                          min_trades=settings.min_trades_walkforward, **common)
    wf_exit = walk_forward(trades, prices, bench, cost, copy_all=True, min_trades=1,
                           exit_on_filer_sell=True, **common)

    curves: dict[str, pd.Series] = {"Copy ALL": wf_all.equity}
    if wf_top is not None and not wf_top.equity.empty:
        curves[f"Copy top-{settings.top_n}"] = wf_top.equity
    if wf_exit is not None and not wf_exit.equity.empty:
        curves["Copy ALL + copy the exit"] = wf_exit.equity
    suite = bm.benchmark_suite(prices, cost, start=span0, end=span1,
                               spy=settings.benchmark)
    for col in suite.columns:
        curves[col] = suite[col].dropna()

    bench_rets = bench.loc[span0:span1].pct_change().dropna()
    spy_curve = curves.get(f"{settings.benchmark} buy & hold")
    rows, series = [], {}
    for name, eq in curves.items():
        eq = eq.dropna()
        if eq.empty:
            continue
        st = metrics.full_metrics(eq, bench_rets=bench_rets)
        if not st:
            continue
        p_boot = None
        if spy_curve is not None and name != f"{settings.benchmark} buy & hold":
            p_boot = metrics.bootstrap_outperf_pvalue(
                eq.pct_change().dropna(), spy_curve.pct_change().dropna())
        rows.append({"name": name, "bootstrap_p": p_boot, **st})
        series[name] = _series(eq)

    best = max(rows, key=lambda r: r.get("total_return") or -9, default=None)
    top_row = next((r for r in rows if r["name"].startswith("Copy top-")), None)
    all_row = next((r for r in rows if r["name"] == "Copy ALL"), None)
    verdict = ""
    if best:
        verdict = f"{best['name']} leads on total return ({best['total_return']:+.0%})."
    if top_row and all_row:
        gap = (top_row["total_return"] or 0) - (all_row["total_return"] or 0)
        verdict += (f" Selecting the top {settings.top_n} filers "
                    f"{'beats' if gap > 0 else 'LOSES to'} copying everyone by "
                    f"{abs(gap):.0%}: selection "
                    f"{'adds' if gap > 0 else 'destroys'} value.")

    cap = _capacity(wf_all, settings)
    if cap["signals_skipped_pct"] > 0.2:
        verdict += (
            f" CAPACITY: only {cap['signals_taken']:,} of {cap['signals_available']:,} "
            f"signals ({1 - cap['signals_skipped_pct']:.0%}) actually fit in the "
            f"portfolio: see the capacity note.")
    return {
        "span": [str(span0.date()), str(span1.date())],
        "cost_model": settings.cost_model, "top_n": settings.top_n,
        "horizon": settings.horizon,
        "series": series, "rows": rows,
        "picks": {str(y): names for y, names in (wf_top.picks if wf_top else {}).items()},
        "diagnostics": {
            "skipped_full": int(wf_all.skipped_full),
            "stale_exits": int(wf_all.stale_exits),
            "taxlots": int(len(wf_all.taxlots)),
        },
        "capacity": cap,
        "verdict": verdict,
    }


def _capacity(wf, settings) -> dict:
    """How much of "copy everyone" the portfolio could actually absorb.

    THIS IS NOT A ROUNDING DETAIL. The simulator sizes each position at
    `max_weight` (5%) of the book, so at most ~20 can be open at once. Congress
    files ~1,200 priceable buys a year and each is held 63 trading days, which
    implies ~300 concurrent positions, fifteen times the capacity. The
    remainder are skipped for want of cash.

    So the curve labelled "Copy ALL" is really "copy whatever arrived while
    cash was free", and the trades it took were selected by ARRIVAL ORDER, a
    rule nobody chose and nothing justifies. It is not look-ahead (arrival
    order is point-in-time), and the event-study evidence, the timescale grid,
    the conviction subsets, the jurisdiction split, is unaffected, because
    those measure per-trade excess return and never build a portfolio. But any
    statement about the BASKET's total return inherits it, and the honest
    reading of the scorecard gap is "signal plus capacity", not "signal".
    """
    horizon = settings.horizon
    taken = int(len(wf.taxlots)) if wf.taxlots is not None else 0
    skipped = int(wf.skipped_full)
    avail = taken + skipped
    days = max(len(wf.equity), 1)
    years = days / 252
    return {
        "signals_available": avail,
        "signals_taken": taken,
        "signals_skipped": skipped,
        "signals_skipped_pct": (skipped / avail) if avail else 0.0,
        "max_weight": settings.max_weight,
        "max_concurrent_positions": int(1 / settings.max_weight) if settings.max_weight else None,
        "implied_concurrent_positions": round(
            (avail / years) * (horizon / 252), 1) if years else None,
        "note": (
            f"Positions are sized at {settings.max_weight:.0%} of the book, so at "
            f"most ~{int(1/settings.max_weight)} can be open at once. Holding "
            f"{avail:,} signals for {horizon} trading days each implies roughly "
            f"{round((avail/years)*(horizon/252)) if years else '?'} concurrent "
            f"positions, which is far beyond that. The basket therefore takes "
            f"signals in ARRIVAL ORDER until cash runs out, point-in-time, but "
            f"not the 'copy everyone' the label implies. Per-trade evidence "
            f"(timescale, conviction, jurisdiction) is unaffected; basket "
            f"returns are not."),
    }


def build_decay() -> dict:
    """The chart the whole project is named after: mean cumulative excess
    return anchored at the transaction (zero-lag, impossible) versus at the
    filing (what you can legally copy). The vertical gap is the alpha the
    disclosure lag destroys."""
    _, settings, _, _, bench = _ctx()
    max_h = 126
    es_f = _event_study([max_h], anchor="filing", kinds=None, sides=("buy",))
    es_t = _event_study([max_h], anchor="transaction", kinds=None, sides=("buy",))
    if es_f is None or es_f.per_trade.empty:
        return {"error": "no priceable buys"}
    out = {}
    for kind in sorted(es_f.per_trade["kind"].dropna().unique()):
        mf = (es_f.per_trade["kind"] == kind).values
        if mf.sum() == 0:
            continue
        filing = es_f.decay_curve(mf)
        rec = {"n": int(mf.sum()),
               "filing": [None if np.isnan(v) else float(v) for v in filing]}
        if es_t is not None and not es_t.per_trade.empty:
            mt = (es_t.per_trade["kind"] == kind).values
            if mt.sum():
                tx = es_t.decay_curve(mt)
                rec["n_transaction"] = int(mt.sum())
                rec["transaction"] = [None if np.isnan(v) else float(v) for v in tx]
                gap = tx[min(settings.horizon, len(tx) - 1)] - filing[
                    min(settings.horizon, len(filing) - 1)]
                rec["gap_at_horizon"] = None if np.isnan(gap) else float(gap)
        out[kind] = rec
    if not out:
        return {"error": "no decay curves"}
    gaps = [v["gap_at_horizon"] for v in out.values()
            if v.get("gap_at_horizon") is not None]
    return {
        "horizon": settings.horizon, "max_h": max_h, "cohorts": out,
        "note": ("Curves are GROSS of costs, the lag gap is an information "
                 "effect, and mixing the cost shift into it would confuse two "
                 "different subtractions."),
        "verdict": (
            f"At {settings.horizon}d the zero-lag anchor sits "
            f"{max(gaps):+.2%} above the filing anchor, that gap is alpha "
            f"destroyed by the disclosure lag, and it is not recoverable: the "
            f"lag is statutory." if gaps else ""),
    }


def build_ranking() -> dict:
    """Per-filer leaderboard. Explicitly NOT a buy list, `persistence` shows
    the ordering does not survive out of sample, so this exists to be read
    against that null, not acted on."""
    from copytrade.rank.rank import build_ranking as _rank

    conn, settings, cost, prices, bench = _ctx()
    trades = _trades(instruments=["stock"])
    if trades.empty or bench.empty:
        return {"error": "no trades or benchmark prices"}
    ranking, diag = _rank(
        trades, prices, bench, cost, horizon=settings.horizon,
        min_trades_rank=settings.min_trades_rank,
        same_day_close=settings.same_day_close, cutoff=settings.same_day_cutoff,
        amount_basis=settings.amount_basis, trade_notional=settings.trade_notional)
    if ranking.empty:
        return {"error": "no ranking rows"}
    n_copyable = int(ranking["copyable"].sum()) if "copyable" in ranking else 0
    return {
        "horizon": settings.horizon, "min_trades_rank": settings.min_trades_rank,
        "diagnostics": diag,
        "rows": ranking.to_dict(orient="records"),
        "n_copyable": n_copyable,
        "verdict": (
            f"{n_copyable} of {len(ranking)} filers clear the copyable bar "
            f"(month-clustered t>2 with n>={settings.min_trades_rank}). "
            f"Ranking hundreds of filers guarantees lucky outliers, and the "
            f"out-of-sample persistence test finds no stable skill, read this "
            f"table as a description, not a shortlist."),
    }


def build_allocation() -> dict:
    """TFSA vs margin sleeve assignment for a Canadian resident."""
    from copytrade.accounts.allocation import allocate_sleeves

    r = build_ranking()
    if r.get("error"):
        return {"error": r["error"]}
    conn, *_ = _ctx()
    trades = _trades(instruments=["stock"])
    plan = allocate_sleeves(pd.DataFrame(r["rows"]), trades)
    return {
        "assignments": plan.assignments.to_dict(orient="records"),
        "tfsa_deployed": plan.tfsa_deployed,
        "margin_deployed": plan.margin_deployed,
        "notes": plan.notes,
        "verdict": (
            f"{len(plan.assignments)} copyable filers assigned: "
            f"${plan.tfsa_deployed:,.0f} TFSA / ${plan.margin_deployed:,.0f} margin. "
            f"CRA can reassess frequent trading in a TFSA as business income "
            f"(folio S3-F9-C1; Ahamed v The King, TCC 2023), so high-turnover "
            f"filers are gated out of the TFSA sleeve."),
    }


# ------------------------------------------------------------- new analyses --

def build_netedge() -> dict:
    """The FAIR test of a tradeable edge, not 'beat SPY'.

    Three parts: a standalone absolute scorecard, the BETA-NEUTRAL long/short
    (long the basket, short beta x SPY, so market direction is hedged and only
    signal remains), and consensus CAR. Part two is the decisive one: if the
    basket has alpha independent of market beta it shows up here, and nowhere
    else in the project isolates it as cleanly.
    """
    from copytrade.backtest import benchmarks as bm
    from copytrade.backtest import metrics
    from copytrade.backtest.engine import run_event_study, walk_forward
    from copytrade.rank.factors import long_short_returns
    from copytrade.rank.insiders import cluster_buy_ids, subset_stats

    conn, settings, cost, prices, bench = _ctx()
    trades = _trades(instruments=["stock"])
    wf_all = _copy_all()
    if wf_all is None or wf_all.equity.empty:
        return {"error": "copy-all produced no equity curve"}
    span = wf_all.equity.index
    wf_top = walk_forward(trades, prices, bench, cost, top_n=settings.top_n,
                          min_trades=settings.min_trades_walkforward,
                          horizon=settings.horizon, max_weight=settings.max_weight,
                          trade_notional=settings.trade_notional,
                          same_day_close=settings.same_day_close,
                          cutoff=settings.same_day_cutoff)

    curves = {"Copy ALL (congress index)": wf_all.equity}
    if wf_top is not None and not wf_top.equity.empty:
        curves[f"Copy top-{settings.top_n}"] = wf_top.equity
    curves[f"{settings.benchmark} buy & hold"] = bm.buy_and_hold(
        bench.loc[span[0]:span[-1]], cost)
    if "TLT" in prices.columns and prices["TLT"].notna().sum() > 50:
        curves["60/40 SPY/TLT"] = bm.sixty_forty(
            bench.loc[span[0]:span[-1]], prices["TLT"], cost)
    absolute = [{"name": n, **metrics.perf_stats(eq.dropna())}
                for n, eq in curves.items() if metrics.perf_stats(eq.dropna())]

    basket_m = wf_all.equity.resample("ME").last().pct_change().dropna()
    spy_m = bench.loc[span[0]:span[-1]].resample("ME").last().pct_change().dropna()
    hedges = []
    for label, fixed in (("OLS beta-neutral", None), ("dollar-neutral (beta=1)", 1.0)):
        ls, b = long_short_returns(basket_m, spy_m, beta=fixed)
        if ls.empty or ls.std() == 0:
            continue
        n = len(ls)
        hedges.append({
            "hedge": label, "beta": float(b),
            "alpha_annualized": float((1 + ls.mean()) ** 12 - 1),
            "sharpe": float(ls.mean() / ls.std() * np.sqrt(12)),
            "t": float(ls.mean() / (ls.std() / np.sqrt(n))),
            "n_months": int(n),
        })

    car = []
    buys = _congress_buys()
    cluster_ids = cluster_buy_ids(buys, 45, 3)
    consensus = buys[buys["trade_id"].isin(cluster_ids)]
    hs = [21, 63, 126]
    if not consensus.empty:
        es = run_event_study(consensus, prices, bench, cost, hs, anchor="filing",
                             same_day_close=settings.same_day_close,
                             cutoff=settings.same_day_cutoff,
                             trade_notional=settings.trade_notional)
        for h in hs:
            s = subset_stats(es.per_trade, f"h{h}")
            if s["n"]:
                car.append({"horizon": h, **s})

    best_t = max((abs(h["t"]) for h in hedges), default=float("nan"))
    return {
        "span": [str(span[0].date()), str(span[-1].date())],
        "cost_model": settings.cost_model,
        "absolute": absolute, "hedges": hedges,
        "consensus": {"n_buys": int(len(consensus)),
                      "n_tickers": int(consensus["ticker"].nunique()) if not consensus.empty else 0,
                      "min_filers": 3, "window_days": 45, "rows": car},
        "verdict": (
            f"Beta-neutral alpha is not significant (best |t| = {best_t:.2f}, "
            f"bar is 2). Hedging the market out leaves nothing, the basket's "
            f"return is market exposure, not signal."
            if best_t == best_t and best_t <= 2 else
            f"Beta-neutral alpha reaches |t| = {best_t:.2f}, investigate."),
    }


def build_factors() -> dict:
    """Fama-French attribution: is the basket skill, or a known size/value tilt?"""
    from copytrade.ingest.famafrench import fetch_ff_factors
    from copytrade.rank.factors import factor_regression, single_benchmark

    conn, settings, cost, prices, bench = _ctx()
    wf = _copy_all()
    if wf is None or wf.equity.empty:
        return {"error": "copy-all produced no equity curve"}
    basket_m = wf.equity.resample("ME").last().pct_change().dropna()

    ff_error, reg = None, {}
    try:
        ff = fetch_ff_factors(getattr(settings, "edgar_user_agent", "copytrade-research"))
    except Exception as e:  # noqa: BLE001, offline builds still produce the panel
        ff, ff_error = None, str(e)[:200]
    if ff is not None and not ff.empty:
        names = ("Mkt-RF", "SMB", "HML", "Mom") if "Mom" in ff.columns else ("Mkt-RF", "SMB", "HML")
        reg = factor_regression(basket_m, ff, factors=names) or {}

    singles = []
    for label in (settings.benchmark, "RSP", "IWM", "QQQ"):
        if label in prices.columns:
            b_m = prices[label].dropna().resample("ME").last().pct_change().dropna()
            sb = single_benchmark(basket_m, b_m)
            if sb:
                singles.append({"benchmark": label, **sb})

    verdict = ""
    if reg:
        a = reg.get("alpha", {})
        smb = reg.get("SMB", {})
        sig = a.get("p", 1) < 0.05
        verdict = (f"Fama-French alpha is {reg['alpha_annualized']:+.1%}/yr "
                   f"(t={a.get('t', float('nan')):.2f}), "
                   f"{'SIGNIFICANT' if sig else 'not significant'}. ")
        if smb:
            verdict += (f"SMB loading {smb['coef']*100:+.1f}% "
                        f"(t={smb['t']:.2f}); R2={reg['r2']:.2f} means the market "
                        f"factor explains most of the variance.")
    elif ff_error:
        verdict = ("Fama-French factors could not be downloaded, so only the "
                   "single-benchmark regressions below are available.")
    return {
        "span": [str(wf.equity.index[0].date()), str(wf.equity.index[-1].date())],
        "n_months": int(len(basket_m)),
        "regression": reg, "ff_error": ff_error, "singles": singles,
        "verdict": verdict,
    }


def build_conviction() -> dict:
    """Does filtering to 'high conviction' congress buys beat copying everyone?"""
    from copytrade.rank.insiders import (
        abnormal_size_ids_pit,
        cluster_buy_ids,
        subset_stats,
        top_quartile_mask,
    )

    _, settings, _, _, _ = _ctx()
    h = settings.horizon
    es = _event_study([h])
    if es is None or es.per_trade.empty:
        return {"error": "no priceable congress buys"}
    pt = es.per_trade
    buys = _congress_buys()
    big, big_info = top_quartile_mask(pt["amount"])
    subsets = {
        "ALL congress buys": pd.Series(True, index=pt.index),
        "cluster (>=3 members / 30d)": pt["trade_id"].isin(cluster_buy_ids(buys, 30, 3)),
        "large ($ top quartile)": big,
        "abnormal (>=2x own median)": pt["trade_id"].isin(abnormal_size_ids_pit(buys, 2.0)),
    }
    subsets["cluster & large"] = subsets["cluster (>=3 members / 30d)"] & big
    rows = []
    for label, mask in subsets.items():
        s = subset_stats(pt[mask], f"h{h}")
        if s["n"]:
            rows.append({"subset": label, **s})
    best = max(rows, key=lambda r: abs(r.get("tstat") or 0), default=None)
    return {
        "horizon": h, "quartile_info": big_info, "rows": rows,
        "verdict": (
            f"Nothing clears |t|>2 (best is {best['subset']} at "
            f"t={best['tstat']:.2f}). Filtering to high-conviction buys does "
            f"not rescue the signal."
            if best and abs(best.get("tstat") or 0) <= 2 else
            (f"{best['subset']} reaches t={best['tstat']:.2f}, but this is one "
             f"of 5 subsets searched; check it against the FDR-corrected "
             f"timescale grid before believing it." if best else "")),
        "caveat": ("These t-stats cluster by month only. The timescale panel "
                   "runs the same subsets with overlap-aware blocks AND an "
                   "FDR correction across the whole grid, that is the bar."),
    }


def build_lag() -> dict:
    """Disclosure-lag decomposition, why you cannot front-run the filing."""
    from copytrade.rank.lag import fastest_filers, lag_by_source, reporting_lag_days

    trades = _trades(instruments=["stock"])
    if trades.empty:
        return {"error": "no trades"}
    by_source = lag_by_source(trades)
    fastest = fastest_filers(trades, min_trades=10, top=20)
    lag = reporting_lag_days(trades).dropna()
    lag = lag[lag >= 0]
    edges = [0, 2, 5, 10, 15, 20, 30, 45, 60, 90, 180, 365, 100000]
    labels = ["0-2", "3-5", "6-10", "11-15", "16-20", "21-30", "31-45",
              "46-60", "61-90", "91-180", "181-365", ">365"]
    hist = pd.cut(lag, bins=edges, labels=labels, include_lowest=True,
                  right=True).value_counts().reindex(labels).fillna(0)
    return {
        "by_source": by_source.to_dict(orient="records") if not by_source.empty else [],
        "fastest": fastest.to_dict(orient="records") if not fastest.empty else [],
        "histogram": [{"bucket": b, "n": int(v)} for b, v in hist.items()],
        "n": int(len(lag)),
        "median_days": float(lag.median()),
        "pct_within_45d": float((lag <= 45).mean()),
        "statutory": [
            {"channel": "Congress (STOCK Act PTR)", "limit": "45 days",
             "note": "filer-controlled within the window, the binding constraint"},
            {"channel": "SEC Form 4 (insiders)", "limit": "2 business days",
             "note": "structurally tiny"},
            {"channel": "13F (funds)", "limit": "45 days after quarter-end",
             "note": "quarter-end longs only"},
            {"channel": "13D (activists)", "limit": "5 business days", "note": ""},
        ],
        "verdict": (
            f"Median disclosure lag is {lag.median():.0f} days and "
            f"{(lag <= 45).mean():.0%} of filings land inside the 45-day "
            f"statutory window. The disclosure IS the first public signal, "
            f"funds and the NANC/KRUZ ETFs act the same second you can, so "
            f"there is no legal way to get the data faster."),
    }


def build_areas() -> dict:
    """Sector attribution: what areas do these people pursue, and where are
    their picks actually good?"""
    from copytrade.rank.areas import sector_attribution, ticker_leaders

    conn, settings, _, _, _ = _ctx()
    h = settings.horizon
    es = _event_study([h])
    if es is None or es.per_trade.empty:
        return {"error": "no priceable congress buys"}
    smap = pd.read_sql_query(
        "SELECT ticker, sector, industry FROM sector_map "
        "WHERE sector IS NOT NULL AND sector != ''", conn)
    sectors = sector_attribution(es.per_trade, smap, f"h{h}", min_buys=20)
    leaders = ticker_leaders(es.per_trade, f"h{h}", min_buys=8, top=15)
    mapped = int(es.per_trade["ticker"].isin(set(smap["ticker"])).sum())
    return {
        "horizon": h,
        "sectors_mapped": int(len(smap)),
        "trades_with_sector": mapped,
        "trades_total": int(len(es.per_trade)),
        "sectors": sectors.to_dict(orient="records") if not sectors.empty else [],
        "best": leaders["best"].to_dict(orient="records"),
        "worst": leaders["worst"].to_dict(orient="records"),
        "most_traded": leaders["most_traded"].to_dict(orient="records"),
        "verdict": (
            f"{mapped:,} of {len(es.per_trade):,} priced buys have a mapped "
            f"sector. Per-sector means are descriptive only, with 11 sectors "
            f"searched, the spread between best and worst is what noise looks "
            f"like at this sample size, and no sector survived correction in "
            f"the timescale grid."),
    }


def build_profits() -> dict:
    """Per-filer track records with NO significance gate.

    The ranking panel gates on t>2 and n>=30, which structurally hides a filer
    who makes three high-conviction bets a year. This answers the plain
    question "how have this filer's picks actually done?" and labels the
    sample size honestly instead of filtering on it.
    """
    from copytrade.rank.profit import filer_track_records

    conn, settings, cost, prices, bench = _ctx()
    trades = _trades(instruments=["stock"])
    if trades.empty or bench.empty:
        return {"error": "no trades or benchmark prices"}
    tr = filer_track_records(trades, prices, bench, cost,
                             same_day_close=settings.same_day_close,
                             cutoff=settings.same_day_cutoff, min_buys=3)
    if tr.empty:
        return {"error": "no track records"}
    by_conf = tr["confidence"].value_counts().to_dict()
    return {
        "rows": tr.to_dict(orient="records"),
        "n_filers": int(len(tr)),
        "by_confidence": {k: int(v) for k, v in by_conf.items()},
        "verdict": (
            f"{len(tr)} filers with a priced record. Sorted by dollars from "
            f"staking $10k on every pick, positions overlap in time, so that "
            f"total is ADDITIVE, never compounded. `confidence` is sample size, "
            f"not skill: an 'anecdotal' filer at the top of this table is the "
            f"most likely place in the whole project to mistake luck for edge."),
    }


def build_timeline() -> dict:
    """Year by year: which calendar years did following the crowd help?"""
    from copytrade.backtest import benchmarks as bm
    from copytrade.backtest.metrics import annual_returns

    conn, settings, cost, prices, bench = _ctx()
    wf = _copy_all()
    if wf is None or wf.equity.empty:
        return {"error": "copy-all produced no equity curve"}
    span0, span1 = wf.equity.index[0], wf.equity.index[-1]
    basket = annual_returns(wf.equity)
    spy = annual_returns(bm.buy_and_hold(bench.loc[span0:span1], cost))
    etfs = {t: annual_returns(eq) for t, eq in
            bm.etf_buy_hold(prices, cost, start=span0, end=span1).items()}
    years = sorted(set(basket.index) | set(spy.index))
    rows = []
    for y in years:
        b = basket.get(y)
        s = spy.get(y)
        row = {"year": int(y), "copy_all": None if b is None else float(b),
               "spy": None if s is None else float(s),
               "excess": None if (b is None or s is None) else float(b - s)}
        for t, ser in etfs.items():
            v = ser.get(y)
            row[t.lower()] = None if v is None else float(v)
        rows.append(row)
    wins = [r for r in rows if r["excess"] is not None and r["excess"] > 0]
    return {
        "span": [str(span0.date()), str(span1.date())],
        "rows": rows,
        "years": len(rows), "years_beating_spy": len(wins),
        "verdict": (
            f"The copy basket beat SPY in {len(wins)} of {len(rows)} calendar "
            f"years. Year-by-year sign counting is not a significance test, "
            f"read it as texture alongside the FDR-corrected grid, and note "
            f"that a coin flip produces about half."),
    }


def build_costs() -> dict:
    """What the broker choice alone does to the answer.

    For a Canadian this is not a footnote: the FX line on a retail schedule is
    ~150bps EACH WAY, which is an order of magnitude above the trading edge
    every other panel is arguing about. The same trades are repriced under
    each model so the sensitivity is visible rather than asserted.
    """
    from copytrade.backtest.costs import PRESETS
    from copytrade.rank.timescale import block_test

    conn, settings, _, prices, bench = _ctx()
    h = settings.horizon
    buys = _congress_buys()
    if buys.empty or bench.empty:
        return {"error": "no data"}
    # ONE event study, repriced seven ways. The gross excess paths do not
    # depend on the cost model, the model only ever subtracts, so running
    # `run_event_study` per preset repeated the same minutes-long computation
    # seven times to produce an identical gross column each round.
    es = _event_study([h])
    if es is None or es.per_trade.empty:
        return {"error": "no priceable buys"}
    base = es.per_trade
    gross_all = base[f"h{h}"] + base["roundtrip_cost"]
    shares = base["amount"] / base["entry_price"].replace(0, np.nan)
    rows = []
    for name in sorted(PRESETS):
        model = PRESETS[name]
        rt = np.array([model.roundtrip_frac(a, sh)
                       for a, sh in zip(base["amount"], shares)])
        net = gross_all - rt
        s = net.dropna()
        if s.empty:
            continue
        gross = gross_all.loc[s.index]
        # averaged over the SAME rows the reported net is, so the two columns
        # describe one sample rather than two
        rt_used = pd.Series(rt, index=base.index).loc[s.index]
        t, p, nb = block_test(s, base.loc[s.index, "entry_date"], h)
        rows.append({
            "cost_model": name,
            "roundtrip_bps": float(rt_used.mean() * 10_000),
            "fx_bps_per_side": model.fx_bps,
            "spread_bps_per_side": model.spread_slippage_bps,
            "mean_gross": float(gross.mean()),
            "mean_net": float(s.mean()),
            "t_net": t, "p_net": p, "blocks": nb,
            "n": int(len(s)),
            "is_default": name == settings.cost_model,
        })
    rows.sort(key=lambda r: r["roundtrip_bps"])
    gross_ref = rows[0]["mean_gross"] if rows else None
    return {
        "horizon": h, "default": settings.cost_model, "rows": rows,
        "gross_mean": gross_ref,
        "verdict": (
            f"Gross excess return is {gross_ref:+.2%} at {h}d regardless of "
            f"broker: the cost model only ever subtracts. Round-trip cost "
            f"ranges from {rows[0]['roundtrip_bps']:.0f}bps ({rows[0]['cost_model']}) "
            f"to {rows[-1]['roundtrip_bps']:.0f}bps ({rows[-1]['cost_model']}), "
            f"and on the retail FX schedules the drag alone exceeds anything "
            f"the signal produces." if rows else ""),
    }


def build_holdings() -> dict:
    """Disclosed net positions: the "what do they hold?" view, honestly bounded.

    Added after comparing this project against consumer trackers (InsiderWave,
    Quiver, Unusual Whales), all of which lead with a portfolio screen. The
    question readers actually arrive with is phrased that way, and this project
    could not answer it at all. See `rank/holdings.py` for why the answer is a
    disclosed NET BAND rather than a portfolio: a PTR discloses transactions,
    not holdings, so positions opened before the window are invisible and every
    size is a range.
    """
    from copytrade.rank.holdings import filer_positions, filer_summary, most_held

    # STOCK ONLY. An option's disclosed amount is a PREMIUM, not a position
    # size, so summing a $250k call into the same band as a $250k share
    # purchase, and letting an option sale net against a stock buy, would
    # misstate the disclosed stake in a way the band notation cannot express.
    # Derivatives are counted separately below.
    trades = _trades(instruments=["stock"])
    if trades.empty:
        return {"error": "no trades"}
    pos = filer_positions(trades)
    if pos.empty:
        return {"error": "no positions"}
    summary = filer_summary(pos, top=30)
    crowded = most_held(pos, min_filers=3, top=25)
    # The per-filer detail dominates the payload and every panel is fetched on
    # page load, so ship the most active filers' largest positions only.
    keep = set(summary["filer"])
    detail = (pos[pos["filer"].isin(keep)]
              .sort_values(["filer", "net_high"], ascending=[True, False])
              .groupby("filer", sort=False).head(40))
    derivs = _trades(instruments=["call_option", "put_option", "option",
                                  "stock_option"])
    return {
        "n_filers": int(pos["filer"].nunique()),
        "n_positions": int(len(pos)),
        "summary": summary.to_dict(orient="records"),
        "crowded": crowded.to_dict(orient="records"),
        "positions": detail.to_dict(orient="records"),
        "status_counts": {k: int(v) for k, v in pos["status"].value_counts().items()},
        "positions_shown": int(len(detail)),
        "derivative_disclosures": int(len(derivs)),
        "caveat": (
            "A PTR discloses TRANSACTIONS, not holdings. Positions opened before "
            "this window are invisible, a member who bought Apple in 2019 and "
            "never traded it again has no row here. Sizes are disclosed RANGES, "
            "so a netted position is a BAND that widens with every transaction, "
            "never a number. A net-zero band is not a closed position. Common "
            "stock only: an option's disclosed amount is a premium, not a "
            f"position size, so the {len(derivs)} derivative disclosures are "
            "counted but never netted against shares."),
        "verdict": (
            f"{int(pos['filer'].nunique())} filers across {int(len(pos)):,} "
            f"filer-ticker pairs. This is disclosed net FLOW, not a portfolio: "
            f"treat it as direction and attention, not position size."),
    }


def build_quality() -> dict:
    """Data-integrity checks that run against the DB, not against bookkeeping.

    Every number on this dashboard inherits whatever is wrong here, and two
    bugs of exactly this shape (non-positive prices stored as prices; an
    uncapped per-share commission) once produced a whole-sample mean below
    -100% for a long-only book. An impossible aggregate is a data bug, never a
    finding, so the checks that would have caught them run every build.
    """
    conn, settings, _, prices, _ = _ctx()
    q = lambda s: conn.execute(s).fetchone()[0]  # noqa: E731

    checks = []

    def check(name, value, ok, detail):
        checks.append({"check": name, "value": value, "ok": bool(ok), "detail": detail})

    n_future = q("SELECT COUNT(*) FROM trades WHERE transaction_date IS NOT NULL "
                 "AND filing_datetime IS NOT NULL "
                 "AND date(transaction_date) > date(filing_datetime)")
    check("transaction date after its own filing", n_future, n_future == 0,
          "Impossible: a PTR discloses a trade that already happened. Nulled by "
          "migration 008 and blocked at ingest by store.db.sanitize_transaction_date.")

    n_nonpos = q("SELECT COUNT(*) FROM prices WHERE adj_close <= 0")
    check("non-positive adjusted closes", n_nonpos, n_nonpos == 0,
          "Returns are p_t/p_0-1, so a negative price flips the ratio's sign. "
          "Rejected at write time by store.db.upsert_prices.")

    # The <=0 guard above was a PARTIAL fix: it removed the negative ticks and
    # left the rest of the same corrupted series in place. DAIUF still ran
    # 5.1e-07 to 45.31 and reached the dashboard as the best-performing name in
    # the sample at +251% mean. This screen is the other half.
    unscreened = 0
    screen_failures: list[dict] = []
    screen_review: list[dict] = []
    try:
        from copytrade.store.price_qa import screen_all

        n_quar = q("SELECT COUNT(*) FROM price_quarantine WHERE released=0")
        # ONE pass over the 7.6M-row price table for both lists, and the
        # already-recorded set hoisted out of the comprehension, it used to be
        # re-queried once per finding.
        screen_failures, screen_review = screen_all(conn)
        recorded = {r[0] for r in conn.execute("SELECT ticker FROM price_quarantine")}
        unscreened = sum(1 for f in screen_failures if f["ticker"] not in recorded)
    except Exception:  # noqa: BLE001, table arrives with migration 010
        n_quar = 0
    check("implausible price series left unscreened", unscreened, unscreened == 0,
          f"{n_quar} ticker(s) are quarantined and excluded at load_price_matrix "
          f"(sub-penny artifacts, unadjusted reverse splits, series oscillating "
          f"between two scales, multi-year gaps read as one session). A non-zero "
          f"count here means `python -m copytrade prices-qa` has not been re-run "
          f"since new prices landed.")

    n_nofiling = q("SELECT COUNT(*) FROM trades WHERE filing_datetime IS NULL")
    check("trades with no filing anchor", n_nofiling, n_nofiling == 0,
          "The filing timestamp is the point-in-time execution anchor; a trade "
          "without one cannot be executed and is dropped by trades_frame.")

    n_nulltx = q("SELECT COUNT(*) FROM trades WHERE transaction_date IS NULL")
    checks.append({
        "check": "transaction date missing (informational)", "value": n_nulltx,
        "ok": True,
        "detail": "Excluded from disclosure-lag and zero-lag decay statistics "
                  "only. These trades remain in every backtest, the filing "
                  "anchor is what execution uses."})

    n_dupe = q("SELECT COUNT(*) FROM (SELECT source, source_uid FROM trades "
               "GROUP BY source, source_uid HAVING COUNT(*) > 1)")
    check("duplicate (source, source_uid)", n_dupe, n_dupe == 0,
          "Enforced by a UNIQUE index; a breach would double-count trades.")

    fut = q("SELECT COUNT(*) FROM prices WHERE date > date('now', '+2 day')")
    check("price rows dated in the future", fut, fut == 0,
          "A future-dated close would leak look-ahead into every event study.")

    bench_missing = [t for t in ("SPY", "QQQ", "VTI", "IWM", "SCHD", "RSP", "TLT")
                     if t not in prices.columns or prices[t].notna().sum() < 100]
    check("benchmarks priced", len(bench_missing), not bench_missing,
          "yfinance silently drops tickers mid-run under rate limiting. Every "
          "scorecard row using a missing benchmark would be invalid."
          + (f" Missing: {', '.join(bench_missing)}." if bench_missing else ""))

    # sector coverage is a POWER limit, not a correctness bug
    mapped = q("SELECT COUNT(*) FROM sector_map WHERE sector IS NOT NULL AND sector != ''")
    traded = q("SELECT COUNT(DISTINCT ticker) FROM trades "
               "WHERE instrument='stock' AND ticker IS NOT NULL")
    throttled = 0
    try:
        throttled = q("SELECT COUNT(*) FROM sector_attempts WHERE last_status='throttled'")
    except Exception:  # noqa: BLE001, table arrives with migration 009
        pass

    failed = n_fail = 0
    try:
        n_fail = q("SELECT COUNT(*) FROM sector_attempts")
        failed = q("SELECT COUNT(*) FROM sector_attempts WHERE attempts >= 3")
    except Exception:  # noqa: BLE001
        pass

    # `review` comes from the single `screen_all` pass above, not a second scan.
    quarantine, review = [], screen_review
    try:
        quarantine = [dict(r) for r in conn.execute(
            "SELECT ticker, reason, detail, detected_at FROM price_quarantine "
            "WHERE released=0 ORDER BY ticker")]
        for row in quarantine:
            try:
                detail = json.loads(row["detail"] or "{}")
            except ValueError:
                detail = {}
            # Flattened rather than nested: the UI table gave four columns the
            # same `detail` key, which collided in the sort handler and made
            # those headers look clickable while doing nothing.
            row["detail"] = detail
            row["min"] = detail.get("min")
            row["max"] = detail.get("max")
            row["range_ratio"] = detail.get("range_ratio")
            row["sessions_over_100pct"] = detail.get("sessions_over_100pct")
    except Exception:  # noqa: BLE001, table arrives with migration 010
        pass
    quar_trades = 0
    if quarantine:
        names = [r["ticker"] for r in quarantine]
        marks = ",".join("?" * len(names))
        quar_trades = conn.execute(
            f"SELECT COUNT(*) FROM trades WHERE ticker IN ({marks}) "
            "AND instrument='stock'", names).fetchone()[0]

    n_bad = sum(1 for c in checks if not c["ok"])
    return {
        "checks": checks,
        "n_failed": n_bad,
        "quarantine": quarantine,
        "quarantine_trades": quar_trades,
        "review": review,
        "coverage": {
            "sectors_mapped": mapped, "tickers_traded": traded,
            "sector_coverage": (mapped / traded) if traded else 0,
            "sector_lookups_attempted": n_fail,
            "sector_lookups_exhausted": failed,
            "sector_lookups_throttled": throttled,
        },
        "verdict": (
            "All integrity checks pass. Remaining gaps are coverage limits "
            "(they cost statistical power), not correctness bugs."
            if n_bad == 0 else
            f"{n_bad} integrity check(s) FAILED, treat every number on this "
            f"dashboard as suspect until they are resolved."),
    }


BUILDERS = {
    "health": build_health,
    "quality": build_quality,
    "jurisdiction": build_jurisdiction,
    "scorecard": build_scorecard,
    # The CAD scorecard is the same builder with a different argument, but it
    # used to be invoked as a special case AFTER the build loop, which made it
    # the one panel that could go stale without anything noticing: a run that
    # died on the last builder skipped it silently, `--only scorecard_cad` was
    # rejected as unknown, and both the API and the API index carried their own
    # `+ ["scorecard_cad"]` patches. Registering it here is what makes it
    # buildable, selectable and verifiable like every other panel.
    "scorecard_cad": partial(build_scorecard, cad=True),
    "timescale": build_timescale,
    "exits": build_exits,
    "leverage": build_leverage,
    "persistence": build_persistence,
    "netedge": build_netedge,
    "factors": build_factors,
    "conviction": build_conviction,
    "costs": build_costs,
    "lag": build_lag,
    "areas": build_areas,
    "decay": build_decay,
    "walkforward": build_walkforward,
    "timeline": build_timeline,
    "ranking": build_ranking,
    "profits": build_profits,
    "holdings": build_holdings,
    "allocation": build_allocation,
}
