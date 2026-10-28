"""Deterministic synthetic dataset for offline demos and tests.

Simulates a market plus filers with KNOWN skill and disclosure lags, so the
pipeline's conclusions can be sanity-checked against ground truth:

  * skilled INSIDERS file within 2 business days -> most alpha survives the lag
  * skilled CONGRESS members file ~30 calendar days late -> the exponential
    decay of their edge means most alpha is gone by the time it is public
  * a couple of fast-filing congress members retain part of their edge

Alpha is injected as front-loaded drift (half-life ~10 trading days) after
each skilled buy, and the same total is subtracted market-wide so the average
stock has zero net injected alpha (no free lunch from the simulation itself).
All names are fictional and clearly marked as demo data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from copytrade.store import db as store

EVENT_DAYS = 60
HALF_LIFE = 10.0
CONGRESS_BUCKETS = [
    (1_001, 15_000), (15_001, 50_000), (50_001, 100_000),
    (100_001, 250_000), (250_001, 500_000),
]
BUCKET_P = [0.4, 0.3, 0.15, 0.1, 0.05]


def _decay_weights(total: float) -> np.ndarray:
    w = 0.5 ** (np.arange(1, EVENT_DAYS + 1) / HALF_LIFE)
    return w / w.sum() * total


def generate(
    conn,
    seed: int = 42,
    start: str = "2019-01-01",
    end: str = "2024-12-31",
    n_tickers: int = 40,
    congress_trades_per_year: int = 24,
    insider_trades_per_year: int = 18,
) -> dict:
    rng = np.random.default_rng(seed)
    cal = pd.bdate_range(start, end)
    n_days = len(cal)
    tickers = [f"DEMO{i:02d}" for i in range(n_tickers)]

    # ------------------------------------------------------------- filers --
    # (name, kind, skill = total injected alpha per buy, filing lag profile)
    filers = []
    for i in range(12):
        skill = 0.06 if i < 3 else (0.03 if i < 5 else 0.0)
        lag_mean = 5 if i in (0, 3) else 30  # two fast filers among the skilled
        chamber = "Senate" if i % 2 == 0 else "House"
        filers.append({
            "name": f"{'Sen.' if chamber == 'Senate' else 'Rep.'} Demo-{i:02d}",
            "kind": "congress", "skill": skill, "lag_mean": lag_mean,
            "lag_kind": "calendar", "hold_bd": int(rng.choice([45, 120, 300])),
            "meta": {"chamber": chamber, "demo": True},
            "trades_per_year": congress_trades_per_year,
        })
    for i in range(10):
        skill = 0.06 if i < 4 else (0.03 if i < 6 else 0.0)
        filers.append({
            "name": f"Insider CXO-{i:02d} (demo)",
            "kind": "insider", "skill": skill, "lag_mean": 2,
            "lag_kind": "business", "hold_bd": int(rng.choice([45, 120, 300])),
            "meta": {"officer": True, "officer_title": "Chief Demo Officer", "demo": True},
            "trades_per_year": insider_trades_per_year,
        })

    # ---------------------------------------------------------- schedule --
    years = n_days / 252
    events = []  # (filer_idx, day_idx, ticker_idx)
    for fi, f in enumerate(filers):
        n_buys = int(f["trades_per_year"] * years)
        days = rng.integers(0, n_days - 5, size=n_buys)
        ticks = rng.integers(0, n_tickers, size=n_buys)
        for d, tk in zip(days, ticks):
            events.append((fi, int(d), int(tk)))

    # ------------------------------------------------------------- market --
    mkt = rng.normal(0.00035, 0.010, n_days)
    idio = rng.normal(0.0, 0.010, (n_days, n_tickers))
    drift = np.zeros((n_days, n_tickers))
    for fi, d, tk in events:
        skill = filers[fi]["skill"]
        if skill <= 0:
            continue
        w = _decay_weights(skill)
        end_i = min(d + 1 + EVENT_DAYS, n_days)
        drift[d + 1 : end_i, tk] += w[: end_i - d - 1]
    drift -= drift.sum() / (n_days * n_tickers)  # zero net injected alpha

    ticker_rets = mkt[:, None] + idio + drift
    prices = 50.0 * np.cumprod(1 + ticker_rets, axis=0)
    spy = 100.0 * np.cumprod(1 + mkt)
    tlt = 100.0 * np.cumprod(1 + rng.normal(0.00008, 0.007, n_days))

    frames = [pd.DataFrame({"ticker": t, "date": cal, "adj_close": prices[:, j]})
              for j, t in enumerate(tickers)]
    frames.append(pd.DataFrame({"ticker": "SPY", "date": cal, "adj_close": spy}))
    frames.append(pd.DataFrame({"ticker": "TLT", "date": cal, "adj_close": tlt}))
    store.upsert_prices(conn, pd.concat(frames, ignore_index=True))

    # ------------------------------------------------------------- trades --
    filer_ids = {fi: store.upsert_filer(conn, f["name"], f["kind"], f["meta"])
                 for fi, f in enumerate(filers)}
    n_trades = 0

    def filing_dt(f, day_idx: int) -> str | None:
        if f["lag_kind"] == "business":
            j = day_idx + max(1, int(round(rng.normal(f["lag_mean"], 0.5))))
            if j >= n_days:
                return None
            t = "16:35:00" if rng.random() < 0.9 else "10:30:00"
            return f"{cal[j].date()} {t}"
        lag = int(np.clip(rng.normal(f["lag_mean"], 5), 2, 44))
        dt = cal[day_idx] + pd.Timedelta(days=lag)
        if dt > cal[-1]:
            return None
        return f"{dt.date()} 17:00:00"

    for n, (fi, d, tk) in enumerate(events):
        f = filers[fi]
        legs = [("buy", d)]
        sell_day = d + f["hold_bd"] + int(rng.integers(-10, 10))
        if 0 < sell_day < n_days:
            legs.append(("sell", sell_day))
        for side, day_idx in legs:
            fdt = filing_dt(f, day_idx)
            if fdt is None:
                continue
            kw = dict(
                filer_id=filer_ids[fi], source="demo",
                source_uid=f"demo:{n}:{side}", ticker=tickers[tk], side=side,
                instrument="stock", transaction_date=str(cal[day_idx].date()),
                filing_datetime=fdt,
            )
            if f["kind"] == "congress":
                lo, hi = CONGRESS_BUCKETS[rng.choice(len(CONGRESS_BUCKETS), p=BUCKET_P)]
                kw.update(amount_low=float(lo), amount_high=float(hi))
            else:
                px = float(prices[day_idx, tk])
                shares = float(np.round(rng.lognormal(mean=6.5, sigma=0.8)))
                kw.update(shares=shares, price=px,
                          amount_low=shares * px, amount_high=shares * px,
                          tx_code="P" if side == "buy" else "S")
            n_trades += int(store.insert_trade(conn, **kw))

    # a few trades on a delisted ticker: exercises the unpriceable disclosure
    ghost = store.upsert_filer(conn, "Rep. Ghost (demo)", "congress",
                               {"chamber": "House", "demo": True})
    for i in range(8):
        store.insert_trade(
            conn, filer_id=ghost, source="demo", source_uid=f"demo:ghost:{i}",
            ticker="GONE", side="buy", instrument="stock",
            transaction_date=str(cal[100 + i * 30].date()),
            filing_datetime=f"{(cal[100 + i * 30] + pd.Timedelta(days=25)).date()} 17:00:00",
            amount_low=15_001.0, amount_high=50_000.0,
        )
    conn.commit()
    store.set_state(conn, "demo", "true")

    return {
        "filers": len(filers) + 1,
        "trades": n_trades + 8,
        "tickers": n_tickers + 2,
        "days": n_days,
        "start": start,
        "end": end,
    }
