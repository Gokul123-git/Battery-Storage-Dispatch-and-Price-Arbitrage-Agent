"""Risk engine: monotone in uncertainty/volatility, and it reaches the LP."""
from __future__ import annotations

import numpy as np
import pytest

from src.config import RiskConfig
from src.optimizer import optimize_dispatch
from src.risk import (adjust_prices, apply_risk_state, compute_risk_state,
                      reserve_floor, risk_notes, throughput_cap)

PRICES = np.array([20.0] * 12 + [90.0] * 12)


def test_conservative_prices_bracket_the_forecast(battery):
    r = RiskConfig(risk_aversion=1.0)
    adj = adjust_prices(PRICES, np.full(24, 10.0), r)
    assert (adj.buy >= adj.forecast).all() and (adj.sell <= adj.forecast).all()
    assert np.allclose(adj.buy - adj.sell, 2 * 10.0)
    neutral = adjust_prices(PRICES, np.full(24, 10.0), RiskConfig(risk_aversion=0.0))
    assert np.allclose(neutral.buy, neutral.sell)


def test_price_shock_preserves_sign_of_negative_prices():
    p = np.array([-40.0, 50.0])
    adj = adjust_prices(p, np.zeros(2), RiskConfig(risk_aversion=0.0), price_shock=0.5)
    assert adj.forecast[0] < 0


def test_higher_uncertainty_raises_the_reserve(battery):
    r = RiskConfig(risk_aversion=0.5)
    calm = compute_risk_state(np.full(24, 2.0), PRICES, battery, r)
    wild = compute_risk_state(np.full(24, 40.0), PRICES, battery, r)
    assert wild.risk_score > calm.risk_score
    assert wild.reserve_frac > calm.reserve_frac
    assert wild.reserve_mwh > calm.reserve_mwh


def test_higher_volatility_raises_the_reserve(battery):
    r = RiskConfig(risk_aversion=0.5)
    sigma = np.full(24, 8.0)
    calm = compute_risk_state(sigma, np.full(48, 50.0), battery, r)
    wild = compute_risk_state(sigma, np.tile([5.0, 150.0], 24), battery, r)
    assert wild.risk_score > calm.risk_score
    assert wild.reserve_frac >= calm.reserve_frac


def test_risk_score_is_bounded_and_levelled(battery):
    r = RiskConfig(risk_aversion=3.0)
    s = compute_risk_state(np.full(24, 1e6), np.tile([-500.0, 900.0], 24), battery, r)
    assert 0.0 <= s.risk_score <= 1.0
    assert s.level in {"LOW", "MODERATE", "ELEVATED", "HIGH"}


def test_scenario_floor_is_respected(battery):
    r = RiskConfig(risk_aversion=0.0, soc_reserve_frac=0.0)
    s = compute_risk_state(np.full(24, 1.0), PRICES, battery, r,
                           scenario_reserve_floor=0.25)
    assert s.reserve_frac >= 0.25


def test_reserve_reaches_the_optimizer_and_changes_dispatch(battery):
    """End-to-end: uncertainty -> risk state -> RiskConfig -> LP SOC floor."""
    base = RiskConfig(risk_aversion=0.5, soc_reserve_frac=0.0)
    calm = apply_risk_state(base, compute_risk_state(np.full(24, 1.0), PRICES, battery, base))
    wild = apply_risk_state(base, compute_risk_state(np.full(24, 45.0),
                                                     np.tile([0.0, 160.0], 24),
                                                     battery, base))
    a = optimize_dispatch(PRICES, PRICES, battery, calm, 80.0)
    b = optimize_dispatch(PRICES, PRICES, battery, wild, 80.0)
    assert b.reserve_mwh > a.reserve_mwh
    assert b.plan["soc_mwh"].min() >= b.reserve_mwh - 1e-6
    assert b.plan["soc_mwh"].min() > a.plan["soc_mwh"].min()


def test_reserve_floor_cannot_swallow_the_whole_band(battery):
    r = RiskConfig(risk_aversion=0.0, soc_reserve_frac=0.9)
    f = reserve_floor(battery, r)
    assert f < battery.e_max
    assert f <= battery.e_min + 0.8 * battery.usable_mwh + 1e-9


def test_enforce_reserve_false_disables_it(battery):
    r = RiskConfig(risk_aversion=0.5, soc_reserve_frac=0.2, enforce_reserve=False)
    s = compute_risk_state(np.full(24, 30.0), PRICES, battery, r)
    assert s.reserve_frac == 0.0
    assert reserve_floor(battery, r) == battery.e_min


def test_throughput_cap_scales_with_window(battery):
    r = RiskConfig(max_daily_cycles=2.0)
    assert throughput_cap(battery, r, 48) == pytest.approx(
        2 * throughput_cap(battery, r, 24))


def test_risk_notes_are_number_backed(battery):
    r = RiskConfig(risk_aversion=0.5, soc_reserve_frac=0.1)
    notes = risk_notes(adjust_prices(PRICES, np.full(24, 10.0), r), battery, r, 24)
    assert notes and any("breakeven" in n for n in notes)
    assert any("reserve" in n.lower() for n in notes)
