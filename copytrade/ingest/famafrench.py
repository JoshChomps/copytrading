"""Fama-French factor returns from Ken French's data library.

Monthly FF3 (Mkt-RF, SMB, HML, RF) + momentum (Mom), used to regress the
congressional basket's returns on known factors, the decisive test of whether
its mild edge is just the SIZE/value tilt (positive SMB beta, ~zero alpha)
rather than skill. Keyless public CSVs; values are in PERCENT and converted to
fractions here.
"""

from __future__ import annotations

import io
import zipfile

import pandas as pd

FF3_URL = ("https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
           "F-F_Research_Data_Factors_CSV.zip")
MOM_URL = ("https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
           "F-F_Momentum_Factor_CSV.zip")


def parse_ff_csv(text: str, cols: list[str]) -> pd.DataFrame:
    """Parse the monthly section of a Ken French CSV: rows keyed by a 6-digit
    YYYYMM, stopping before the annual section (whose keys are 4-digit years).
    Returns a DataFrame indexed by month-end date, values as fractions."""
    rows = []
    for line in text.splitlines():
        parts = [p.strip() for p in line.strip().split(",")]
        if len(parts) >= 1 + len(cols) and len(parts[0]) == 6 and parts[0].isdigit():
            try:
                vals = [float(x) for x in parts[1:1 + len(cols)]]
            except ValueError:
                continue
            rows.append([parts[0]] + vals)
    if not rows:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(rows, columns=["ym"] + cols)
    df["date"] = pd.to_datetime(df["ym"], format="%Y%m") + pd.offsets.MonthEnd(0)
    for c in cols:
        df[c] = df[c] / 100.0
    return df.set_index("date")[cols]


def _read_zip_csv(content: bytes, cols: list[str]) -> pd.DataFrame:
    z = zipfile.ZipFile(io.BytesIO(content))
    name = next(n for n in z.namelist() if n.upper().endswith(".CSV"))
    return parse_ff_csv(z.read(name).decode("latin-1"), cols)


def fetch_ff_factors(user_agent: str = "copytrade-research") -> pd.DataFrame:
    """Monthly FF3 + RF, joined with momentum when reachable."""
    import httpx

    with httpx.Client(timeout=90, follow_redirects=True,
                      headers={"User-Agent": user_agent}) as c:
        resp = c.get(FF3_URL)
        resp.raise_for_status()
        ff = _read_zip_csv(resp.content, ["Mkt-RF", "SMB", "HML", "RF"])
        try:
            mom = _read_zip_csv(c.get(MOM_URL).content, ["Mom"])
            ff = ff.join(mom, how="left")
        except Exception:  # noqa: BLE001, momentum is optional
            pass
    return ff
