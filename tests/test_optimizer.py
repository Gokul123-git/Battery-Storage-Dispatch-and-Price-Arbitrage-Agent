"""Optimizer: physics, efficiency, degradation, reserve, WAIT, no forced trades."""
from __future__ import annotations

import numpy as np
import pytest

from src.config import BatteryConfig, RiskConfig
from src.optimizer import (check_feasible, check_no_simultaneous, classify,
                           optimize_dispatch)
from src.risk import reserve_floor

FLAT = np.full(24, 50.0)
SPREAD = np.array([10.0] * 12 + [200.0] * 12)


def solve(prices, battery, risk, soc_mwh, **kw):
    return optimize_dispatch(prices, prices, battery, risk, soc_mwh, **kw)


def test_soc_power_and_capacity_limits_hold(battery, risk):
    dp = solve(SPREAD, battery, risk, soc_mwh=50.0)
    p = dp.plan
    assert check_feasible(p, battery, dp.reserve_mwh) == []
    assert p["charge_mw"].max() <= battery.power_mw + 1e-6
    assert p["discharge_mw"].max() <= battery.power_mw + 1e-6
    assert p["soc_mwh"].max() <= battery.e_max + 1e-6
    assert p["soc_mwh"].min() >= battery.e_min - 1e-6


def test_soc_dynamics_match_the_efficiency_model(battery, risk):
    dp = solve(SPREAD, battery, risk, soc_mwh=50.0)
    p = dp.plan
    prev = 50.0
    for _, row in p.iterrows():
        expect = (prev + row["charge_mw"] * battery.eta_charge
                  - row["discharge_mw"] / battery.eta_discharge)
        assert row["soc_mwh"] == pytest.approx(expect, abs=1e-5)
        prev = row["soc_mwh"]


def test_no_simultaneous_charge_and_discharge(battery, risk):
    for prices in (FLAT, SPREAD, np.random.default_rng(0).normal(50, 40, 24)):
        dp = solve(prices, battery, risk, soc_mwh=50.0)
        assert check_no_simultaneous(dp.plan) == 0


def test_flat_prices_produce_no_trading(battery, risk):
    """No spread -> no cycles. The system must not force a trade every hour."""
    dp = solve(FLAT, battery, risk, soc_mwh=50.0)
    assert dp.plan["charge_mw"].sum() == pytest.approx(0.0, abs=1e-4)
    assert dp.plan["discharge_mw"].sum() == pytest.approx(0.0, abs=1e-4)
    assert (dp.plan["action"] == "WAIT").all()


def test_spread_below_breakeven_is_refused(battery, risk):
    """100 -> 105 does not pay for efficiency losses plus degradation."""
    prices = np.array([100.0] * 12 + [105.0] * 12)
    assert battery.breakeven_sell_price(100.0) > 105.0
    dp = solve(prices, battery, risk, soc_mwh=50.0, terminal_value=False)
    assert dp.plan["charge_mw"].sum() == pytest.approx(0.0, abs=1e-4)


def test_wide_spread_is_taken(battery, risk):
    dp = solve(SPREAD, battery, risk, soc_mwh=50.0)
    assert dp.plan["charge_mw"].sum() > 1.0
    assert dp.plan["discharge_mw"].sum() > 1.0
    assert "CHARGE" in set(dp.plan["action"]) and "DISCHARGE" in set(dp.plan["action"])


def test_degradation_cost_is_in_the_objective_and_suppresses_cycling(battery, risk):
    mild = np.array([40.0] * 12 + [70.0] * 12)
    cheap = BatteryConfig(**{**battery.to_dict(), "cycle_cost_per_mwh": 0.0})
    dear = BatteryConfig(**{**battery.to_dict(), "cycle_cost_per_mwh": 25.0})
    t_cheap = solve(mild, cheap, risk, 50.0).throughput
    t_dear = solve(mild, dear, risk, 50.0).throughput
    assert t_cheap > t_dear, "higher degradation cost must reduce throughput"
    dp = solve(mild, dear, risk, 50.0)
    thr = dp.plan["charge_mw"] + dp.plan["discharge_mw"]
    assert np.allclose(dp.plan["degradation_cost"], 25.0 * thr)


def test_adaptive_reserve_is_a_binding_constraint(battery):
    """Raising the reserve must raise the realised SOC floor in the solution."""
    low = RiskConfig(risk_aversion=0.0, soc_reserve_frac=0.0)
    high = RiskConfig(risk_aversion=0.0, soc_reserve_frac=0.30)
    a = solve(SPREAD, battery, low, soc_mwh=80.0)
    b = solve(SPREAD, battery, high, soc_mwh=80.0)
    assert b.reserve_mwh > a.reserve_mwh
    assert b.plan["soc_mwh"].min() >= b.reserve_mwh - 1e-6
    assert b.plan["soc_mwh"].min() > a.plan["soc_mwh"].min()
    assert b.throughput <= a.throughput + 1e-6


def test_reserve_never_makes_the_problem_infeasible(battery):
    """Starting below a freshly raised reserve must still solve."""
    risk = RiskConfig(risk_aversion=0.0, soc_reserve_frac=0.40)
    dp = solve(SPREAD, battery, risk, soc_mwh=battery.e_min)
    assert dp.status == "Optimal"


def test_negative_prices_make_charging_profitable(battery, risk):
    prices = np.array([-50.0] * 6 + [60.0] * 18)
    dp = solve(prices, battery, risk, soc_mwh=20.0)
    assert dp.plan["charge_mw"].iloc[:6].sum() > 1.0


def test_throughput_cap_binds(battery):
    tight = RiskConfig(risk_aversion=0.0, soc_reserve_frac=0.0, max_daily_cycles=0.25)
    loose = RiskConfig(risk_aversion=0.0, soc_reserve_frac=0.0, max_daily_cycles=4.0)
    assert solve(SPREAD, battery, tight, 50.0).throughput < \
           solve(SPREAD, battery, loose, 50.0).throughput


def test_risk_adjusted_prices_reduce_trading(battery):
    """lambda * sigma widens the required spread -> fewer cycles."""
    mild = np.array([40.0] * 12 + [95.0] * 12)      # spread just clears breakeven
    sigma = np.full(24, 25.0)
    neutral = optimize_dispatch(mild, mild, battery,
                                RiskConfig(risk_aversion=0.0), 50.0,
                                terminal_value=False)
    averse = optimize_dispatch(mild + sigma, mild - sigma, battery,
                               RiskConfig(risk_aversion=1.0), 50.0,
                               terminal_value=False)
    assert averse.throughput < neutral.throughput


def test_final_soc_constraint(battery, risk):
    dp = solve(SPREAD, battery, risk, soc_mwh=50.0, final_soc_mwh=80.0)
    assert dp.plan["soc_mwh"].iloc[-1] >= 80.0 - 1e-5


def test_classify_actions():
    assert classify(5.0, 0.0) == "CHARGE"
    assert classify(0.0, 5.0) == "DISCHARGE"
    assert classify(0.0, 0.0) == "WAIT"
    assert classify(1e-9, 1e-9) == "WAIT"


def test_bad_inputs_raise(battery, risk):
    with pytest.raises(ValueError):
        optimize_dispatch([], [], battery, risk, 50.0)
    with pytest.raises(ValueError):
        optimize_dispatch([np.nan] * 24, [50.0] * 24, battery, risk, 50.0)


def test_solve_time_is_fast_enough(battery, risk):
    dp = optimize_dispatch(np.random.default_rng(1).normal(50, 30, 48),
                           np.random.default_rng(1).normal(50, 30, 48),
                           battery, risk, 50.0)
    assert dp.solve_seconds < 5.0
    assert dp.n_vars == 3 * 48
