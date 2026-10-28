import numpy as np
import pandas as pd

from copytrade.ingest.famafrench import parse_ff_csv
from copytrade.rank.factors import (
    factor_regression,
    long_short_returns,
    ols,
    single_benchmark,
)


def test_long_short_hedges_out_beta():
    import numpy as np
    idx = pd.date_range("2015-01-31", periods=48, freq="ME")
    rng = np.random.default_rng(3)
    b = pd.Series(rng.normal(scale=0.03, size=48), index=idx)
    # basket = 1.5*bench + small constant alpha; long-short should recover ~alpha, beta~1.5
    basket = 1.5 * b + 0.002
    ls, beta = long_short_returns(basket, b)
    assert abs(beta - 1.5) < 0.05
    assert abs(ls.mean() - 0.002) < 5e-4          # residual is the alpha, beta removed
    assert ls.std() < b.std()                     # market risk hedged out


def test_ols_recovers_known_coefficients():
    rng = np.random.default_rng(0)
    n = 400
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = 0.5 + 2.0 * x1 - 1.0 * x2 + rng.normal(scale=0.01, size=n)
    X = np.column_stack([np.ones(n), x1, x2])
    r = ols(y, X)
    assert abs(r["beta"][0] - 0.5) < 0.05
    assert abs(r["beta"][1] - 2.0) < 0.05
    assert abs(r["beta"][2] + 1.0) < 0.05
    assert r["r2"] > 0.99


def test_factor_regression_detects_size_beta():
    # basket excess return is exactly 1.3*SMB (a pure size bet, zero alpha)
    idx = pd.date_range("2015-01-31", periods=60, freq="ME")
    rng = np.random.default_rng(1)
    smb = rng.normal(scale=0.03, size=60)
    ff = pd.DataFrame({
        "Mkt-RF": rng.normal(scale=0.04, size=60),
        "SMB": smb,
        "HML": rng.normal(scale=0.03, size=60),
        "RF": np.full(60, 0.001),
    }, index=idx)
    noise = rng.normal(scale=0.004, size=60)             # realistic residual
    basket = pd.Series(ff["RF"].to_numpy() + 1.3 * smb + noise, index=idx)
    reg = factor_regression(basket, ff, factors=("Mkt-RF", "SMB", "HML"))
    assert abs(reg["SMB"]["coef"] - 1.3) < 0.05          # size beta recovered
    assert reg["SMB"]["p"] < 0.01                        # and significant
    assert abs(reg["alpha"]["coef"]) < 0.003             # ~zero alpha
    assert reg["alpha"]["p"] > 0.05                      # insignificant alpha (with noise)


def test_single_benchmark_perfect_tracking():
    idx = pd.date_range("2015-01-31", periods=40, freq="ME")
    rng = np.random.default_rng(2)
    b = pd.Series(rng.normal(scale=0.03, size=40), index=idx)
    basket = b * 1.0                                     # identical to benchmark
    sb = single_benchmark(basket, b)
    assert abs(sb["beta"] - 1.0) < 1e-6 and sb["r2"] > 0.999
    assert abs(sb["alpha"]) < 1e-6


def test_parse_ff_csv_reads_monthly_section_only():
    text = (
        "This file was created by ...\n\n"
        ",Mkt-RF,SMB,HML,RF\n"
        "202401, 1.00,-0.50, 0.20, 0.40\n"
        "202402, 2.00, 0.50,-0.10, 0.42\n"
        "\n Annual Factors: January-December \n"
        ",Mkt-RF,SMB,HML,RF\n"
        "2024, 12.0, 3.0, 1.0, 5.0\n"        # annual row (4-digit) must be ignored
    )
    df = parse_ff_csv(text, ["Mkt-RF", "SMB", "HML", "RF"])
    assert len(df) == 2                                  # only the two monthly rows
    assert abs(df["Mkt-RF"].iloc[0] - 0.01) < 1e-9       # percent -> fraction
