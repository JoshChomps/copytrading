import numpy as np
import pandas as pd

from copytrade.rank.exits import apply_exit, rule_grid, run_rule


def test_fixed_horizon_takes_the_last_day():
    path = np.array([0.0, 0.01, 0.02, 0.03, 0.04])
    val, held = apply_exit(path, max_h=4)
    assert val == 0.04 and held == 4


def test_max_h_caps_the_hold():
    path = np.array([0.0, 0.01, 0.02, 0.03, 0.04])
    val, held = apply_exit(path, max_h=2)
    assert val == 0.02 and held == 2


def test_trailing_stop_fires_after_a_giveback():
    # runs up to +10%, then gives back 6% -> a 5% trail must fire at that point
    path = np.array([0.0, 0.05, 0.10, 0.04, 0.20])
    val, held = apply_exit(path, trail=0.05)
    assert held == 3 and np.isclose(val, 0.04)
    # a 6% trail also fires at day 3 (0.04 <= 0.10 - 0.06)
    assert apply_exit(path, trail=0.06)[1] == 3
    # an 8% trail does NOT fire: threshold is 0.02 and the path only fell to
    # 0.04, so the position rides the giveback and captures the later run-up
    assert np.isclose(apply_exit(path, trail=0.08)[0], 0.20)


def test_take_profit_and_stop_loss_fire_on_first_touch():
    path = np.array([0.0, 0.05, 0.12, -0.30])
    assert apply_exit(path, take_profit=0.10)[1] == 2
    down = np.array([0.0, -0.05, -0.12, 0.30])
    assert apply_exit(down, stop_loss=0.10)[1] == 2


def test_earliest_trigger_wins():
    # take-profit at day 1, trailing stop would fire later -> TP wins
    path = np.array([0.0, 0.25, 0.24, 0.05])
    val, held = apply_exit(path, trail=0.05, take_profit=0.20)
    assert held == 1 and np.isclose(val, 0.25)


def test_rule_cannot_fire_on_the_entry_day():
    # day 0 is the entry close; a stop must not trigger before the position exists
    path = np.array([0.0, 0.05, 0.10])
    val, held = apply_exit(path, stop_loss=0.0)
    assert held != 0 and np.isclose(val, 0.10)


def test_unelapsed_horizon_is_nan_not_a_short_hold():
    """THE baseline-integrity guard. run_event_study writes NaN into h{max_h}
    when the window has not elapsed; apply_exit must agree, or the "fixed 63d
    baseline" quietly averages in 3-56 day holds. On the live sample those 55
    recent trades averaged +4.91% against -0.35% for the rest, moving the
    printed baseline from -0.3509% to -0.33%."""
    path = np.array([0.0, 0.03, 0.07, np.nan, np.nan])
    val, held = apply_exit(path, max_h=4)
    assert np.isnan(val) and held == 0


def test_require_full_can_be_disabled_to_close_at_the_last_print():
    # opting out gives the old behaviour: close at the last real observation
    path = np.array([0.0, 0.03, 0.07, np.nan, np.nan])
    val, held = apply_exit(path, max_h=4, require_full=False)
    assert held == 2 and np.isclose(val, 0.07)


def test_full_window_is_unaffected_by_require_full():
    path = np.array([0.0, 0.01, 0.02, 0.03, 0.04])
    assert apply_exit(path, max_h=4) == apply_exit(path, max_h=4, require_full=False)


def test_rule_grid_baseline_matches_the_engine_universe():
    """The baseline must contain exactly the trades whose horizon elapsed."""
    paths, per_trade = _fixture(n=100)
    paths[:20, 40:] = np.nan          # 20 trades never reach h=63
    grid = rule_grid(paths, per_trade, baseline_h=63)
    base = grid[grid["rule"].str.contains("baseline")].iloc[0]
    assert base["n"] == 80, "trades short of the horizon must not enter the baseline"
    assert np.isclose(base["mean_hold"], 63.0)


def test_rule_labels_do_not_collide_for_close_parameters():
    """Keying the grid on a percent-formatted label made 0.10 and 0.104 collide,
    silently dropping a rule AND shrinking the BH family below the grid actually
    searched."""
    paths, per_trade = _fixture()
    grid = rule_grid(paths, per_trade, baseline_h=63, trails=(0.10, 0.104),
                     take_profits=(), stop_losses=())
    trail_rules = [r for r in grid["rule"] if r.startswith("trail")]
    assert len(trail_rules) == 2, f"both trails must survive, got {trail_rules}"


def test_all_nan_path_is_unpriceable():
    val, held = apply_exit(np.array([np.nan, np.nan]))
    assert np.isnan(val) and held == 0


def _fixture(n=240, seed=0):
    rng = np.random.default_rng(seed)
    H = 63
    paths = np.cumsum(rng.normal(0, 0.01, size=(n, H + 1)), axis=1)
    paths[:, 0] = 0.0
    dates = pd.to_datetime("2020-01-15") + pd.to_timedelta(
        rng.integers(0, 1400, n), unit="D")
    per_trade = pd.DataFrame({
        "roundtrip_cost": np.full(n, 0.002),
        "entry_date": dates,
    })
    return paths, per_trade


def test_run_rule_subtracts_round_trip_costs():
    paths, per_trade = _fixture()
    r = run_rule(paths, per_trade, max_h=63)
    gross = paths[:, 63]
    assert np.allclose(r["net_excess"].to_numpy(), gross - 0.002)


def test_grid_contains_baseline_and_corrects_across_rules():
    paths, per_trade = _fixture()
    grid = rule_grid(paths, per_trade, baseline_h=63)
    assert (grid["rule"].str.contains("baseline")).sum() == 1
    # baseline has no paired comparison against itself
    base = grid[grid["rule"].str.contains("baseline")].iloc[0]
    assert pd.isna(base["t_vs_base"])
    # q-values are never more significant than the raw paired p-values
    tested = grid.dropna(subset=["p_vs_base", "q_fdr"])
    assert (tested["q_fdr"] >= tested["p_vs_base"] - 1e-12).all()


def test_no_rule_beats_baseline_on_pure_noise():
    paths, per_trade = _fixture(seed=11)
    grid = rule_grid(paths, per_trade, baseline_h=63)
    assert grid["beats_baseline"].sum() == 0


def test_trailing_stop_shortens_average_hold():
    paths, per_trade = _fixture()
    grid = rule_grid(paths, per_trade, baseline_h=63).set_index("rule")
    base_hold = grid.loc["fixed 63d (baseline)", "mean_hold"]
    trail_hold = grid.loc["trail 5%", "mean_hold"]
    assert trail_hold < base_hold
