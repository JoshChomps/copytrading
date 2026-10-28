"""TFSA vs non-registered sleeve allocation for a Canadian resident.

The strategy's later functional step (user requirement): split the copy-trade
book so the TFSA compounds tax-free where that is actually safe, and the
high-turnover copying lives in a non-registered (margin) account.

Why the split matters (facts, not advice, verify against CRA guidance):

  * CRA can treat frequent, short-hold trading inside a TFSA as *carrying on
    a business* (Income Tax Folio S3-F9-C1 factors: frequency, holding
    period, knowledge, time spent). Business income in a TFSA is TAXABLE and
    reassessments are retroactive with interest. Patterns that draw audits:
    hundreds of trades/year, holds of days, options, finance-industry
    knowledge. There is NO safe trade-count bright line.
  * A non-registered account has none of that risk: short-term profits are
    just taxed (100% as business income or 50% inclusion as capital gains,
    facts-dependent), and losses are actually usable there.
  * US-listed dividends inside a TFSA suffer 15% US withholding that is
    unrecoverable (the Canada-US treaty pension exemption covers RRSPs, not
    TFSAs): so the TFSA sleeve prefers low-yield names; an RRSP is the
    better wrapper for high-yield US holdings.
  * TFSA contribution room 2026: $7,000 new; $109,000 cumulative if eligible
    (18+, resident) every year since 2009 and never contributed. Verify your
    personal number in CRA My Account, over-contributions cost 1%/month.

Mapping to this codebase: `turnover_class` (from FIFO-matched holds of the
FILER's own trades) + observed filing frequency decide the sleeve. Copying a
filer whose median hold is short would itself generate short holds in YOUR
account, so those filers are assigned to the margin sleeve regardless of
alpha.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TFSA_ROOM_2026_MAX = 109_000.0   # cumulative since 2009, verify in CRA My Account
TFSA_MAX_TRADES_PER_YEAR = 24    # conservative guideline, NOT a CRA rule; the
                                 # folio test is multi-factor with no bright line
TFSA_MIN_TURNOVER = ("medium", "long")  # median hold >= 63 trading days


@dataclass
class SleevePlan:
    assignments: pd.DataFrame   # filer, sleeve, weight_cad, trades_per_year, reason
    tfsa_deployed: float
    margin_deployed: float
    notes: list[str]


def _trades_per_year(trades: pd.DataFrame) -> pd.Series:
    """Observed filings/year per filer_id (signal rate you would be copying)."""
    if trades.empty:
        return pd.Series(dtype=float)
    g = trades.dropna(subset=["filing_datetime"]).groupby("filer_id")["filing_datetime"]
    span_days = (g.max() - g.min()).dt.days.clip(lower=365)
    return (g.count() / (span_days / 365.25)).rename("trades_per_year")


def allocate_sleeves(
    ranking: pd.DataFrame,
    trades: pd.DataFrame,
    tfsa_room: float = TFSA_ROOM_2026_MAX,
    taxable_capital: float = 100_000.0,
    max_weight: float = 0.10,
    max_tfsa_trades_per_year: float = TFSA_MAX_TRADES_PER_YEAR,
) -> SleevePlan:
    """Assign each copyable filer to a TFSA or margin sleeve.

    TFSA sleeve: turnover_class in {medium, long} AND copying them implies
    <= max_tfsa_trades_per_year orders. Everything else copyable -> margin.
    Weights: equal-weight within a sleeve, capped at max_weight of that
    sleeve's capital. Filers that are not `copyable` are excluded entirely
    allocating to statistically unproven filers is how multiple-hypothesis
    luck becomes losses.
    """
    notes = [
        "TFSA business-income risk is fact-specific (CRA folio S3-F9-C1); the "
        f"{max_tfsa_trades_per_year:.0f} trades/yr gate here is a conservative guideline, not a rule.",
        "US dividends in a TFSA lose 15% to unrecoverable withholding; prefer "
        "low-yield US names there, or hold high-yielders in an RRSP instead.",
        "Margin sleeve gains are fully taxable (capital gains or business "
        "income); keep records: taxlots.csv gives per-lot P&L for ACB.",
        "Superficial-loss rule: repurchasing within 30 days denies the loss, "
        "relevant when the same filer flips in and out of one name.",
    ]
    cols = ["filer", "sleeve", "weight_cad", "trades_per_year", "turnover_class", "reason"]
    if ranking.empty or "copyable" not in ranking.columns:
        return SleevePlan(pd.DataFrame(columns=cols), 0.0, 0.0,
                          ["no ranking available — run the pipeline first"] + notes)

    eligible = ranking[ranking["copyable"]].copy()
    if eligible.empty:
        return SleevePlan(
            pd.DataFrame(columns=cols), 0.0, 0.0,
            ["no filer passes the significance gate (clustered t>2, n>=min), "
             "copy nothing; buy the benchmark instead"] + notes,
        )

    tpy = _trades_per_year(trades)
    eligible["trades_per_year"] = eligible["filer_id"].map(tpy).fillna(0.0)

    rows = []
    for r in eligible.itertuples():
        slow = r.turnover_class in TFSA_MIN_TURNOVER
        calm = r.trades_per_year <= max_tfsa_trades_per_year
        if slow and calm:
            sleeve, reason = "tfsa", (
                f"median hold {r.turnover_class}, ~{r.trades_per_year:.0f} filings/yr, "
                "compatible with buy-and-hold character"
            )
        else:
            why = [] if slow else [f"short median hold ({r.turnover_class})"]
            if not calm:
                why.append(f"~{r.trades_per_year:.0f} filings/yr > {max_tfsa_trades_per_year:.0f}")
            sleeve, reason = "margin", ", ".join(why) + ", CRA business-income risk in a TFSA"
        rows.append({
            "filer": r.filer, "sleeve": sleeve, "trades_per_year": r.trades_per_year,
            "turnover_class": r.turnover_class, "reason": reason,
        })
    df = pd.DataFrame(rows)
    df["weight_cad"] = 0.0

    for sleeve, capital in (("tfsa", tfsa_room), ("margin", taxable_capital)):
        m = df["sleeve"] == sleeve
        n = int(m.sum())
        if n:
            df.loc[m, "weight_cad"] = round(min(capital / n, capital * max_weight), 2)

    tfsa_deployed = float(df.loc[df["sleeve"] == "tfsa", "weight_cad"].sum())
    margin_deployed = float(df.loc[df["sleeve"] == "margin", "weight_cad"].sum())
    if tfsa_deployed == 0.0:
        notes.insert(0, "no filer qualified for the TFSA sleeve, deploy TFSA room "
                        "into the benchmark (e.g. broad-index ETF) rather than leaving it idle")
    return SleevePlan(df[cols], tfsa_deployed, margin_deployed, notes)
