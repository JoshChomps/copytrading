"""FastAPI backend serving the dashboard + JSON API.

Analytics (ranking, decay curves, walk-forward equity vs benchmarks) are
computed from the DB on first request and cached in-process; POST /api/refresh
recomputes after new data lands. Static report PNGs are served from reports/.
"""

from __future__ import annotations

import math
from pathlib import Path
from threading import Lock

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from copytrade.backtest import benchmarks as bm
from copytrade.backtest import metrics
from copytrade.backtest.costs import PRESETS, get_cost_model
from copytrade.backtest.engine import run_event_study, walk_forward
from copytrade.config import load_settings
from copytrade.rank.rank import build_ranking
from copytrade.store import db as store

settings = load_settings()
app = FastAPI(title="CopyTrade", version="0.1.0")

_cache: dict = {}
_lock = Lock()

STATIC_DIR = Path(__file__).parent / "static"


def _conn():
    conn = store.connect(settings.db_path)
    store.run_migrations(conn)
    return conn


def _clean(obj):
    """JSON-safe: numpy scalars -> python, NaN/inf -> None."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, np.ndarray)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return None if (math.isnan(obj) or math.isinf(obj)) else float(obj)
    if isinstance(obj, (pd.Timestamp,)):
        return obj.strftime("%Y-%m-%d")
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def _series_json(s: pd.Series) -> dict:
    s = s.dropna()
    return {
        "dates": [d.strftime("%Y-%m-%d") for d in s.index],
        "values": _clean(list(s.values)),
    }


def compute_analytics(cost_model_name: str | None = None, force: bool = False) -> dict:
    key = cost_model_name or settings.cost_model
    with _lock:
        if not force and key in _cache:
            return _cache[key]
        conn = _conn()
        cost = get_cost_model(key)
        trades = store.trades_frame(conn, instruments=["stock"])
        if trades.empty:
            raise HTTPException(409, "no trades in DB, run `python -m copytrade demo` or a backfill")
        prices = store.load_price_matrix(conn)
        if settings.benchmark not in prices.columns:
            raise HTTPException(409, f"no {settings.benchmark} prices, run `python -m copytrade prices`")
        bench = prices[settings.benchmark].dropna()
        common = dict(
            same_day_close=settings.same_day_close, cutoff=settings.same_day_cutoff,
            amount_basis=settings.amount_basis, trade_notional=settings.trade_notional,
        )

        ranking, diag = build_ranking(
            trades, prices, bench, cost, horizon=settings.horizon,
            min_trades_rank=settings.min_trades_rank, **common,
        )

        buys = trades[trades["side"] == "buy"]
        max_h = min(126, max(settings.horizons))
        es_f = run_event_study(buys, prices, bench, cost, [settings.horizon],
                               anchor="filing", **common)
        es_t = run_event_study(buys, prices, bench, cost, [settings.horizon],
                               anchor="transaction", **common)
        decay = {}
        for kind in sorted(buys["filer_kind"].dropna().unique()):
            mf = (es_f.per_trade["kind"] == kind).values if not es_f.per_trade.empty else np.array([])
            mt = (es_t.per_trade["kind"] == kind).values if not es_t.per_trade.empty else np.array([])
            if mf.size == 0 or mf.sum() == 0:
                continue
            decay[kind] = {
                "n": int(mf.sum()),
                "filing": _clean(list(es_f.decay_curve(mf)[: max_h + 1])),
                "transaction": _clean(list(es_t.decay_curve(mt)[: max_h + 1])),
            }

        wf = walk_forward(
            trades, prices, bench, cost,
            horizon=settings.horizon, top_n=settings.top_n,
            min_trades=settings.min_trades_walkforward, max_weight=settings.max_weight,
            trade_notional=settings.trade_notional, **{
                k: common[k] for k in ("same_day_close", "cutoff", "amount_basis")
            },
        )
        equity: dict = {}
        stats: dict = {}
        picks: dict = {}
        if wf is not None:
            curves = bm.benchmark_suite(
                prices, cost, start=wf.equity.index[0], end=wf.equity.index[-1],
                spy=settings.benchmark,
            )
            curves.insert(0, f"CopyTrade top-{settings.top_n}", wf.equity)
            for col in curves.columns:
                s = curves[col].dropna()
                equity[col] = _series_json(s)
                stats[col] = _clean(metrics.perf_stats(s))
            picks = {str(y): names for y, names in wf.picks.items()}

        result = {
            "cost_model": key,
            "horizon": settings.horizon,
            "diagnostics": _clean(diag),
            "ranking": _clean(ranking.to_dict(orient="records")),
            "decay": decay,
            "equity": equity,
            "stats": stats,
            "picks": picks,
        }
        _cache[key] = result
        return result


@app.get("/api/summary")
def summary():
    conn = _conn()
    by_kind = {r["kind"]: r["n"] for r in conn.execute(
        "SELECT kind, COUNT(*) n FROM filers GROUP BY kind")}
    by_side = {str(r["side"]): r["n"] for r in conn.execute(
        "SELECT side, COUNT(*) n FROM trades GROUP BY side")}
    n_trades = conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
    n_prices = conn.execute("SELECT COUNT(*) FROM prices").fetchone()[0]
    n_tickers = conn.execute("SELECT COUNT(DISTINCT ticker) FROM prices").fetchone()[0]
    return {
        "filers_by_kind": by_kind, "trades_by_side": by_side,
        "trades": n_trades, "price_rows": n_prices, "priced_tickers": n_tickers,
        "demo_data": store.get_state(conn, "demo") == "true",
        "cost_models": sorted(PRESETS),
        "settings": {
            "cost_model": settings.cost_model, "horizon": settings.horizon,
            "top_n": settings.top_n, "benchmark": settings.benchmark,
            "amount_basis": settings.amount_basis, "max_weight": settings.max_weight,
        },
    }


@app.get("/api/analytics")
def analytics(cost_model: str | None = None, live: bool = False):
    """Ranking + decay + walk-forward in one payload.

    THIS IS THE SLOW PATH and the dashboard no longer uses it. On the live
    sample (41k trades, 7.6M price rows) an uncached call runs an event study
    and then a day-by-day portfolio simulation across ~3,000 trading days; it
    did not return inside five minutes, which is why the walk-forward, decay,
    ranking and sleeve sections of the dashboard used to sit permanently
    empty. Those are precomputed panels now (`dashboard-build`), and this
    endpoint serves the same figures from disk unless `live=true` forces a
    recompute: kept because it is a documented public JSON API and because
    the demo dataset is small enough for it to be genuinely live.
    """
    if cost_model and cost_model not in PRESETS:
        raise HTTPException(400, f"unknown cost model; options: {sorted(PRESETS)}")
    if not live and not cost_model:
        merged = _from_panels()
        if merged is not None:
            return merged
    return compute_analytics(cost_model)


def _from_panels() -> dict | None:
    """Assemble the legacy /api/analytics shape out of precomputed panels.

    Returns None when they have not been built, so the caller can fall back to
    computing rather than reporting an empty dataset as an answer.
    """
    from copytrade.web import analyses as an

    rank_p, wf_p, decay_p = an.load("ranking"), an.load("walkforward"), an.load("decay")
    if not rank_p or rank_p.get("error") or not wf_p or wf_p.get("error"):
        return None
    equity = {k: {"dates": v["dates"], "values": v["values"]}
              for k, v in (wf_p.get("series") or {}).items()}
    stats = {r["name"]: {k: v for k, v in r.items() if k != "name"}
             for r in (wf_p.get("rows") or [])}
    decay = {}
    for kind, d in ((decay_p or {}).get("cohorts") or {}).items():
        decay[kind] = {"n": d.get("n"), "filing": d.get("filing"),
                       "transaction": d.get("transaction")}
    return {
        "cost_model": rank_p.get("cost_model") or settings.cost_model,
        "horizon": rank_p.get("horizon", settings.horizon),
        "diagnostics": rank_p.get("diagnostics", {}),
        "ranking": rank_p.get("rows", []),
        "decay": decay,
        "equity": equity,
        "stats": stats,
        "picks": wf_p.get("picks", {}),
        "source": "precomputed",
        # The OLDEST of the three, not the ranking's own. `dashboard-build
        # --only` can refresh one panel and leave the others behind, and
        # reporting the freshest stamp would present a silently mixed-vintage
        # payload as coherent.
        "built_at": min((p.get("built_at") for p in (rank_p, wf_p, decay_p)
                         if p and p.get("built_at")), default=None),
        "panel_built_at": {"ranking": rank_p.get("built_at"),
                           "walkforward": wf_p.get("built_at"),
                           "decay": (decay_p or {}).get("built_at")},
    }


@app.post("/api/refresh")
def refresh():
    _cache.clear()
    return compute_analytics(force=True)["diagnostics"]


@app.get("/api/allocation")
def allocation(tfsa_room: float = 109_000.0, taxable_capital: float = 100_000.0):
    """TFSA vs margin sleeve plan (Canadian resident; see docs/CANADA.md).

    Reads the precomputed ranking when it exists, building the ranking inside
    this handler is the same minutes-long walk-forward that made /api/analytics
    unusable. The sleeve maths itself is instant; only its input was slow.
    """
    from copytrade.accounts.allocation import allocate_sleeves
    from copytrade.web import analyses as an

    panel = an.load("ranking")
    if panel and not panel.get("error") and panel.get("rows"):
        ranking = pd.DataFrame(panel["rows"])
    else:
        ranking = pd.DataFrame(compute_analytics()["ranking"])
    conn = _conn()
    trades = store.trades_frame(conn, instruments=["stock"])
    plan = allocate_sleeves(ranking, trades, tfsa_room, taxable_capital)
    return _clean({
        "assignments": plan.assignments.to_dict(orient="records"),
        "tfsa_deployed": plan.tfsa_deployed,
        "margin_deployed": plan.margin_deployed,
        "notes": plan.notes,
    })


@app.get("/api/news/{ticker}")
def news(ticker: str, limit: int = 10):
    """Recent stored headlines for a ticker (context, not signal
    refresh via `python -m copytrade news-refresh`)."""
    from copytrade.ingest.news import latest_news

    return _clean(latest_news(_conn(), ticker, min(limit, 50)))


@app.get("/api/trades/recent")
def recent_trades(limit: int = 50):
    conn = _conn()
    rows = conn.execute(
        "SELECT t.trade_id, f.name filer, f.kind, t.ticker, t.side, t.instrument, "
        "t.transaction_date, t.filing_datetime, t.amount_low, t.amount_high, t.source "
        "FROM trades t JOIN filers f USING(filer_id) "
        "ORDER BY t.filing_datetime DESC LIMIT ?", (min(limit, 500),),
    ).fetchall()
    return _clean([dict(r) for r in rows])


@app.get("/api/filers/{filer_id}/trades")
def filer_trades(filer_id: int, limit: int = 200):
    conn = _conn()
    rows = conn.execute(
        "SELECT t.*, f.name filer, f.kind FROM trades t JOIN filers f USING(filer_id) "
        "WHERE filer_id=? ORDER BY filing_datetime DESC LIMIT ?", (filer_id, limit),
    ).fetchall()
    if not rows:
        raise HTTPException(404, "no trades for filer")
    return _clean([dict(r) for r in rows])


@app.get("/api/analysis/{name}")
def analysis(name: str):
    """Serve a PRECOMPUTED analysis payload.

    These are read from disk rather than computed per request: the timescale
    sweep, the leverage bootstrap and the feature/control build each take
    minutes, so computing them in a handler would time the browser out. Build
    them with `python -m copytrade dashboard-build`.
    """
    from copytrade.web import analyses as an

    if name not in an.BUILDERS:
        raise HTTPException(404, f"unknown analysis; options: {sorted(an.BUILDERS)}")
    payload = an.load(name)
    if payload is None:
        raise HTTPException(
            404, f"'{name}' has not been built yet, run "
                 f"`python -m copytrade dashboard-build`")
    return payload


@app.get("/api/analysis")
def analysis_index():
    """What has been built, and how stale it is."""
    from copytrade.web import analyses as an

    out = {}
    for name in an.BUILDERS:
        payload = an.load(name)
        out[name] = {"built": payload is not None,
                     "built_at": (payload or {}).get("built_at"),
                     "error": (payload or {}).get("error")}
    return out


app.mount("/reports", StaticFiles(directory=settings.reports_dir, check_dir=False), name="reports")


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")
