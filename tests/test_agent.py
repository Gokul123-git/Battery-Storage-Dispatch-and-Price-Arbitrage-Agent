"""Scenario agent: it may tune policy, it may never touch physics or dispatch."""
from __future__ import annotations

import numpy as np
import pytest

from src.agent import (ALLOWED_POLICY_KEYS, LOCKED_KEYS, PolicyViolation,
                       node_policy_selection, node_scenario_analysis,
                       run_agent, sanitize_policy)
from src.config import SCENARIO_ORDER, SCENARIOS
from src.pipeline import resolve_policy


# ---------------------------------------------------------------- boundary
def test_agent_may_set_policy_parameters():
    ok, rej = sanitize_policy({"risk_aversion": 1.2, "soc_reserve_frac": 0.2,
                               "horizon": 48, "cycle_cost_multiplier": 1.5})
    assert ok == {"risk_aversion": 1.2, "soc_reserve_frac": 0.2,
                  "horizon": 48, "cycle_cost_multiplier": 1.5}
    assert rej == []


@pytest.mark.parametrize("key", sorted(LOCKED_KEYS))
def test_agent_cannot_touch_physics_or_dispatch(key):
    ok, rej = sanitize_policy({key: 999})
    assert ok == {}, f"{key} must never reach the optimiser"
    assert any(key in r for r in rej)


def test_strict_mode_raises_on_forbidden_key():
    with pytest.raises(PolicyViolation):
        sanitize_policy({"charge_mw": 25}, strict=True)


def test_out_of_range_policy_is_clipped_not_obeyed():
    ok, rej = sanitize_policy({"risk_aversion": 99, "soc_reserve_frac": 0.95,
                               "horizon": 5000})
    lo, hi = ALLOWED_POLICY_KEYS["risk_aversion"]
    assert ok["risk_aversion"] == hi
    assert ok["soc_reserve_frac"] == ALLOWED_POLICY_KEYS["soc_reserve_frac"][1]
    assert ok["horizon"] == ALLOWED_POLICY_KEYS["horizon"][1]
    assert len(rej) == 3


def test_non_numeric_policy_is_refused():
    ok, rej = sanitize_policy({"risk_aversion": "very high"})
    assert ok == {} and rej


def test_pipeline_rebuilds_configs_and_ignores_unknown_keys():
    """Even if a forbidden key slipped through, resolve_policy cannot apply it."""
    sc, risk = resolve_policy("normal", overrides={"capacity_mwh": 5_000,
                                                   "charge_mw": 25,
                                                   "risk_aversion": 1.1})
    assert not hasattr(sc, "capacity_mwh") and not hasattr(sc, "charge_mw")
    assert risk.risk_aversion == pytest.approx(1.1)


def test_unknown_scenario_is_rejected_by_the_pipeline():
    from src.config import ConfigError
    with pytest.raises(ConfigError):
        resolve_policy("teleport_to_mars")


def test_unknown_scenario_falls_back_in_the_agent():
    st = node_policy_selection({"scenario": "nonsense", "trace": []})
    assert st["scenario"] == "normal"
    assert st["policy"]["risk_aversion"] > 0


# ---------------------------------------------------------------- behaviour
def test_scenario_analysis_reads_real_market_numbers():
    st = node_scenario_analysis({"trace": []})
    a = st["analysis"]
    assert a["window_hours"] > 0
    assert np.isfinite(a["price_mean"]) and a["price_std"] > 0
    assert 0.0 <= a["renewable_ratio"] <= 5.0


def test_scenarios_change_policy_parameters():
    seen = {}
    for name in SCENARIO_ORDER:
        st = node_policy_selection(node_scenario_analysis(
            {"scenario": name, "trace": []}))
        seen[name] = st["policy"]
    assert len({(p["risk_aversion"], p["soc_reserve_frac"], p["price_shock"])
                for p in seen.values()}) > 1, "scenarios must differ materially"
    assert seen["conservative"]["risk_aversion"] > seen["high_renewable"]["risk_aversion"]


@pytest.mark.slow
def test_agent_runs_end_to_end_and_respects_constraints():
    st = run_agent("normal", horizon=24, backtest_hours=12, with_open_loop=False)
    r = st["result"]
    assert r is not None, st.get("error")
    assert r["current_action"] in {"CHARGE", "DISCHARGE", "WAIT"}
    plan = r["dispatch"]
    b = r["battery"]
    assert plan["charge_mw"].max() <= b["power_mw"] + 1e-6
    assert plan["discharge_mw"].max() <= b["power_mw"] + 1e-6
    assert plan["soc_mwh"].max() <= b["soc_max"] * b["capacity_mwh"] + 1e-6
    assert st["trace"], "agent must expose its reasoning trace"
    assert r["operator_plan"]["narrative"]


@pytest.mark.slow
def test_agent_override_of_dispatch_cannot_change_the_plan():
    """Ask the agent to force a 25 MW charge; the LP must still decide."""
    honest = run_agent("normal", horizon=24, backtest_hours=0)
    forced = run_agent("normal", horizon=24, backtest_hours=0,
                       override={"charge_mw": 25, "soc_mwh": 99,
                                 "force_action": "CHARGE"})
    assert forced["rejected"], "forbidden keys must be reported as rejected"
    a, b = honest["result"]["dispatch"], forced["result"]["dispatch"]
    assert np.allclose(a["charge_mw"], b["charge_mw"])
    assert np.allclose(a["soc_mwh"], b["soc_mwh"])
    assert honest["result"]["current_action"] == forced["result"]["current_action"]
