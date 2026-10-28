import pytest

from copytrade.backtest.costs import SEC_FEE_RATE_SELL, CostModel, get_cost_model


def test_spread_only():
    m = CostModel(name="x", spread_slippage_bps=10, sec_fee_on_sells=False)
    assert m.cost_per_side(10_000, 100, "buy") == pytest.approx(10.0)
    assert m.roundtrip_frac(10_000, 100) == pytest.approx(0.002)


def test_min_commission_binds():
    m = CostModel(name="x", spread_slippage_bps=0, commission_per_share=0.005,
                  commission_min=1.0, sec_fee_on_sells=False)
    # 100 shares -> $0.50 raw, min $1 binds
    assert m.commission(10_000, 100) == pytest.approx(1.0)
    # 1000 shares -> $5 raw, min doesn't bind
    assert m.commission(10_000, 1000) == pytest.approx(5.0)


def test_sec_fee_sells_only():
    m = CostModel(name="x", spread_slippage_bps=0)
    assert m.cost_per_side(1_000_000, 1, "sell") == pytest.approx(1_000_000 * SEC_FEE_RATE_SELL)
    assert m.cost_per_side(1_000_000, 1, "buy") == 0.0


def test_fx_both_legs():
    m = CostModel(name="x", spread_slippage_bps=0, fx_bps=15, sec_fee_on_sells=False)
    assert m.roundtrip_frac(10_000, 10) == pytest.approx(0.0030)  # 15bps each way


def test_presets_exist():
    for name in ("zero", "alpaca", "ibkr_ca", "questrade", "questrade_gambit",
                 "wealthsimple", "default10bps"):
        assert get_cost_model(name).name == name
    with pytest.raises(ValueError):
        get_cost_model("robinhood_ca")


def test_ibkr_math():
    m = get_cost_model("ibkr_ca")
    # $10k of a $100 stock = 100 shares: 10bps slip + 3bps fx + max(0.5,1)=1 commission
    cost = m.cost_per_side(10_000, 100, "buy")
    assert cost == pytest.approx(10.0 + 3.0 + 1.0)


def test_fx_dominates_on_zero_commission_canadian_brokers():
    # the whole Canadian point: Questrade/Wealthsimple 'free' trades cost ~30x
    # IBKR's FX on a USD round trip
    ws = get_cost_model("wealthsimple").roundtrip_frac(10_000, 100)
    ib = get_cost_model("ibkr_ca").roundtrip_frac(10_000, 100)
    assert ws > 0.030 and ib < 0.005


def test_per_share_commission_is_capped_at_one_percent():
    """IBKR caps commission at 1% of trade value. The docstring said so; the
    code did not implement it. Uncapped, a $10k order in a $0.01 stock is
    1,000,000 shares at $0.005 = $5,000 per side (100% of notional), and the
    penny-stock tail dragged the whole-sample mean excess return to -280%."""
    from copytrade.backtest.costs import get_cost_model

    c = get_cost_model("ibkr_ca")
    notional = 10_000.0
    # a normal $50 stock is nowhere near the cap
    assert c.roundtrip_frac(notional, notional / 50) < 0.005
    # a penny stock is capped, not explosive
    penny = c.roundtrip_frac(notional, notional / 0.01)
    assert penny < 0.025, f"capped round trip should be ~2.3%, got {penny:.2%}"
    # and an absurdly low price cannot exceed it either
    assert c.roundtrip_frac(notional, notional / 0.0005) <= penny + 1e-9


def test_commission_cap_overrides_the_minimum_on_tiny_orders():
    from copytrade.backtest.costs import CostModel

    m = CostModel(name="t", commission_per_share=0.005, commission_min=1.0,
                  commission_max_pct=0.01, spread_slippage_bps=0.0,
                  sec_fee_on_sells=False)
    # Tiny order: the 1% cap WINS over the $1 floor, which is IBKR's actual
    # rule (the cap is the outer bound). The old test name claimed the floor
    # was preserved while asserting 0.5, half of it.
    assert m.commission(50.0, 10) == 0.5
    # normal order: neither bound binds
    assert m.commission(10_000.0, 200) == 1.0
