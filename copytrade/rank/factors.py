"""Factor attribution of the congressional basket.

Regress the basket's monthly returns on Fama-French factors to answer the
question every earlier stage kept raising: is the basket's mild edge real
alpha, or just a known SIZE/value/breadth tilt? A positive, significant SMB
(small-minus-big) beta with an insignificant alpha means "it's the size
factor, not skill." Also regresses the basket on an equal-weight benchmark
(RSP) to see how much a single breadth proxy explains.

OLS by numpy (no statsmodels dependency); t-stats from classical standard
errors, p-values from the t-distribution.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def ols(y: np.ndarray, X: np.ndarray) -> dict:
    """Ordinary least squares. X must already include an intercept column."""
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    n, k = X.shape
    dof = max(n - k, 1)
    sigma2 = float(resid @ resid) / dof
    cov = sigma2 * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    t = beta / se
    tss = float(((y - y.mean()) ** 2).sum()) or 1.0
    r2 = 1.0 - float(resid @ resid) / tss
    p = 2 * (1 - stats.t.cdf(np.abs(t), dof))
    return {"beta": beta, "se": se, "t": t, "p": p, "r2": r2, "n": n}


def factor_regression(basket_ret: pd.Series, ff: pd.DataFrame,
                      factors=("Mkt-RF", "SMB", "HML")) -> dict:
    """Regress basket EXCESS return (basket - RF) on the FF factors. Returns
    per-term {coef, t, p}, annualized alpha, R2, n_months."""
    factors = [f for f in factors if f in ff.columns]
    df = pd.concat([basket_ret.rename("r"), ff], axis=1).dropna(subset=["r", "RF", *factors])
    if len(df) < len(factors) + 3:
        return {}
    y = (df["r"] - df["RF"]).to_numpy()
    X = np.column_stack([np.ones(len(df))] + [df[f].to_numpy() for f in factors])
    r = ols(y, X)
    names = ["alpha", *factors]
    out = {nm: {"coef": float(b), "t": float(tt), "p": float(pp)}
           for nm, b, tt, pp in zip(names, r["beta"], r["t"], r["p"])}
    out["alpha_annualized"] = float((1 + r["beta"][0]) ** 12 - 1)
    out["r2"] = float(r["r2"])
    out["n_months"] = int(r["n"])
    return out


def long_short_returns(basket_ret: pd.Series, bench_ret: pd.Series,
                       beta: float | None = None) -> tuple[pd.Series, float]:
    """Beta-neutral long/short return series: long the basket, short beta x the
    benchmark, so market direction is hedged out and only the SIGNAL (alpha)
    remains. If beta is None it's the OLS slope of basket on benchmark. Returns
    (long_short_series, beta)."""
    df = pd.concat([basket_ret.rename("r"), bench_ret.rename("b")], axis=1).dropna()
    if df.empty:
        return pd.Series(dtype=float), float("nan")
    if beta is None:
        var = float(df["b"].var(ddof=1))
        beta = float(df["r"].cov(df["b"]) / var) if var > 0 else 1.0
    return (df["r"] - beta * df["b"]).rename("ls"), beta


def single_benchmark(basket_ret: pd.Series, bench_ret: pd.Series) -> dict:
    """Regress the basket on ONE benchmark's return (e.g. RSP or SPY): alpha,
    beta, R2. High R2 + ~zero alpha => that benchmark explains the basket."""
    df = pd.concat([basket_ret.rename("r"), bench_ret.rename("b")], axis=1).dropna()
    if len(df) < 6:
        return {}
    y = df["r"].to_numpy()
    X = np.column_stack([np.ones(len(df)), df["b"].to_numpy()])
    r = ols(y, X)
    return {"alpha": float(r["beta"][0]), "alpha_t": float(r["t"][0]),
            "alpha_annualized": float((1 + r["beta"][0]) ** 12 - 1),
            "beta": float(r["beta"][1]), "beta_t": float(r["t"][1]),
            "r2": float(r["r2"]), "n": int(r["n"])}
