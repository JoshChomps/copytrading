"""Transaction cost models. Every backtested return is NET of these.

Components per side (entry or exit):
  * spread/slippage proxy (bps of notional)
  * commission: min(max_pct*notional,
                    max(min_commission, per_trade + per_share*shares + pct*notional))
  * FX conversion (bps): a CAD-based investor buying USD assets pays FX on
    both conversion legs; this line item usually dominates for Canadians on
    zero-commission platforms (plan §7)
  * SEC section 31 fee on SELLS only (currently $27.80 per $1M sold)

Presets reflect real platforms for a Canadian resident so "does the edge
survive costs?" is answered against the brokerage you would actually use:
  * ibkr_ca: Interactive Brokers Canada, Fixed pricing, US stocks:
                  USD 0.005/share (min USD 1, max 1% of trade value), FX at
                  interbank spot +0.5bp (min USD 2 order fee), modeled as
                  ~3bps effective on retail-size conversions. Cheapest FX in
                  Canada and the only major CA broker with a real trading API.
  * questrade: $0 commissions on stocks (since Feb 2025); CAD<->USD
                  conversion ~150bps each way unless you Norbert's-gambit
                  (DLR/DLR.U), which cuts FX to roughly spread + two
                  commissions. TFSA/RRSP-friendly.
  * wealthsimple: $0 commissions; 1.5% FX each way on USD trades from a CAD
                  account (core tier). No public trading API, included so the
                  fee drag is visible, not as an execution target.
  * alpaca: US, commission-free, API-first; slippage only. The
                  paper-trading target for Phase 8 (paper trading needs no
                  residency; live accounts for Canadians are limited).
  * zero: frictionless, for sensitivity comparison only.

The max-1%-of-value commission cap (IBKR) IS modeled, via
`CostModel.commission_max_pct`. An earlier revision of this docstring claimed
the cap was deliberately omitted because "it only binds on sub-$100 orders"
that reasoning was wrong and cost the project a full set of results. With
`per_share = 0.005` the cap binds whenever `0.005 * (notional / price) >
0.01 * notional`, i.e. for ANY order in a stock priced under $0.50, at any
size. Uncapped, a $10k order in a $0.01 stock is 1,000,000 shares = $5,000 of
commission per side (a 100% round trip), and the penny-stock tail dragged the
whole-sample mean excess return to -280%. Do not remove the cap.
"""

from __future__ import annotations

from dataclasses import dataclass

SEC_FEE_RATE_SELL = 27.80 / 1_000_000  # section 31, per $ sold


@dataclass(frozen=True)
class CostModel:
    name: str
    spread_slippage_bps: float = 10.0
    commission_per_trade: float = 0.0
    commission_per_share: float = 0.0
    commission_pct: float = 0.0
    commission_min: float = 0.0
    commission_max_pct: float = 0.0   # 0 = uncapped
    fx_bps: float = 0.0
    sec_fee_on_sells: bool = True

    def commission(self, notional: float, shares: float) -> float:
        c = (
            self.commission_per_trade
            + self.commission_per_share * shares
            + self.commission_pct * notional
        )
        if c > 0 or self.commission_min > 0:
            c = max(c, self.commission_min)
        # Per-share commissions must be capped or low-priced stocks produce
        # absurd costs: at $0.01/share a $10k order is 1,000,000 shares, which
        # uncapped bills 100% of notional per side. Real brokers cap exactly
        # for this reason (IBKR: 1% of trade value). Without the cap the
        # penny-stock tail dragged the whole-sample mean excess return to
        # -280%, swamping every genuine result.
        if self.commission_max_pct > 0:
            c = min(c, notional * self.commission_max_pct)
        return c

    def cost_per_side(self, notional: float, shares: float, side: str) -> float:
        """Dollar cost of one side (side is the direction of THIS leg:
        'buy' for entries, 'sell' for exits)."""
        cost = notional * (self.spread_slippage_bps + self.fx_bps) / 10_000
        cost += self.commission(notional, shares)
        if side == "sell" and self.sec_fee_on_sells:
            cost += notional * SEC_FEE_RATE_SELL
        return cost

    def roundtrip_frac(self, notional: float, shares: float) -> float:
        """Total round-trip (buy then sell) cost as a fraction of notional."""
        if notional <= 0:
            return 0.0
        total = self.cost_per_side(notional, shares, "buy") + self.cost_per_side(
            notional, shares, "sell"
        )
        return total / notional

    def side_frac(self, notional: float, shares: float, side: str) -> float:
        if notional <= 0:
            return 0.0
        return self.cost_per_side(notional, shares, side) / notional


PRESETS: dict[str, CostModel] = {
    "zero": CostModel(
        name="zero", spread_slippage_bps=0.0, sec_fee_on_sells=False
    ),
    "default10bps": CostModel(name="default10bps", spread_slippage_bps=10.0),
    "alpaca": CostModel(
        name="alpaca", spread_slippage_bps=10.0
    ),
    "ibkr_ca": CostModel(
        name="ibkr_ca",
        spread_slippage_bps=10.0,
        commission_per_share=0.005,
        commission_min=1.0,
        commission_max_pct=0.01,   # IBKR caps at 1% of trade value
        fx_bps=3.0,
    ),
    "questrade": CostModel(
        name="questrade", spread_slippage_bps=10.0, fx_bps=150.0
    ),
    "questrade_gambit": CostModel(
        # Norbert's gambit via DLR/DLR.U: FX collapses to the ETF spread
        # (~2-4bps) amortized over the converted amount; commissions are $0.
        name="questrade_gambit", spread_slippage_bps=10.0, fx_bps=4.0
    ),
    "wealthsimple": CostModel(
        name="wealthsimple", spread_slippage_bps=10.0, fx_bps=150.0
    ),
}


def get_cost_model(name: str) -> CostModel:
    try:
        return PRESETS[name]
    except KeyError:
        raise ValueError(f"unknown cost model {name!r}; options: {sorted(PRESETS)}") from None
