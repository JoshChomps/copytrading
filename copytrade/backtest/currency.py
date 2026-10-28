"""CAD-denominated reporting: the return a Canadian actually realises.

Every equity curve in this project is computed in USD, because the assets are
US-listed. But the investor is a BC resident who spends CAD, judges the result
in CAD, and pays CRA in CAD. Those are not the same number: holding a USD asset
is implicitly also a long-USD/short-CAD position, and USD/CAD is volatile
enough over a decade to move the ranking of two strategies.

This is a DIFFERENT thing from the FX cost already in `costs.py`. That charges
the spread paid to convert currency at entry and exit (150bps retail, ~3bps
IBKR): a transaction cost. What is modelled here is the translation effect:
the mark-to-market of the currency exposure you carry for the whole holding
period, whether or not you ever convert. The cost model cannot capture it
because it is not a fee, and it does not net to zero over a sample.

`fx_rates` has existed since migration 002 with a comment saying it is "for
CAD-denominated reporting", but nothing ever consumed it. This wires it up.

CAVEAT ON HISTORY. The Bank of Canada Valet series available here starts
2017-01-03, while the backtest span starts earlier. Curves are therefore
reported in CAD only over the overlapping window, and `cad_span` tells the
caller what was actually covered rather than silently forward-filling a rate
backwards: that would fabricate FX history, which the project's no-fabricated-
data rule forbids.
"""

from __future__ import annotations

import pandas as pd


def load_usdcad(conn) -> pd.Series:
    """Daily USD/CAD (CAD per 1 USD) as a date-indexed Series."""
    rows = conn.execute(
        "SELECT date, rate FROM fx_rates WHERE pair='USDCAD' ORDER BY date"
    ).fetchall()
    if not rows:
        return pd.Series(dtype=float)
    s = pd.Series({pd.Timestamp(r[0]): float(r[1]) for r in rows}).sort_index()
    return s.rename("USDCAD")


def to_cad(equity_usd: pd.Series, usdcad: pd.Series,
           max_stale_days: int = 7) -> tuple[pd.Series, dict]:
    """Convert a USD equity curve to CAD.

    A Canadian's CAD wealth is the USD curve times the spot rate, so the CAD
    return compounds the asset return with the currency return. The series is
    forward-filled onto the equity index for non-trading days (a stale rate on
    a holiday is correct, the rate genuinely did not print), but NOT
    back-filled before the first observation: those dates are dropped.

    Returns (cad_curve, info) where info reports the covered span and the FX
    contribution, so the caller can state how much of the CAD result was the
    currency rather than the strategy.
    """
    eq = equity_usd.dropna()
    if eq.empty or usdcad.empty:
        return pd.Series(dtype=float), {}
    # Bounded ffill. An unlimited ffill fabricates the TAIL exactly as
    # back-filling would fabricate the head, and `fx-refresh` is a manual
    # command, so a stale fx_rates table is the EXPECTED state. Carrying the
    # last rate forward for years silently multiplies the whole curve by a
    # frozen number and then reports that frozen rate's move as the realised
    # currency contribution. A few days covers real gaps (holidays); anything
    # past the last observation is dropped and counted.
    last_obs = usdcad.index.max()
    fx = (usdcad.reindex(eq.index.union(usdcad.index))
          .ffill(limit=max_stale_days)
          .reindex(eq.index))
    fx = fx.where(eq.index <= last_obs)
    covered = eq.index[fx.notna()]
    if len(covered) < 2:
        return pd.Series(dtype=float), {}
    eq_c = eq.loc[covered]
    fx_c = fx.loc[covered]
    cad = (eq_c * fx_c).rename("cad")
    usd_ret = float(eq_c.iloc[-1] / eq_c.iloc[0] - 1)
    fx_ret = float(fx_c.iloc[-1] / fx_c.iloc[0] - 1)
    cad_ret = float(cad.iloc[-1] / cad.iloc[0] - 1)
    return cad, {
        "cad_span": (covered[0], covered[-1]),
        "dropped_days": int(len(eq) - len(covered)),
        "fx_last_observation": last_obs,
        "dropped_after_last_fx": int((eq.index > last_obs).sum()),
        "usd_return": usd_ret,
        "fx_return": fx_ret,
        "cad_return": cad_ret,
        # what the currency added/subtracted, in return-space
        "fx_contribution": cad_ret - usd_ret,
    }
