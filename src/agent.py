"""LangGraph scenario agent -- decision SUPPORT, never the controller.

    Scenario Input -> Scenario Analysis -> Policy Selection -> Optimizer Tool
    -> Result Analysis -> Risk Analysis -> Operator Plan

BOUNDARY (enforced in code, not by convention)
----------------------------------------------
The agent may ONLY emit values for the keys in ``ALLOWED_POLICY_KEYS`` -- the
policy knobs: risk aversion, scenario reserve floor, horizon, degradation
sensitivity, forecast stress, volatility assumption. ``sanitize_policy`` drops
anything else (SOC, capacity, power, charge/discharge quantities, dispatch
overrides, reserve in MWh...) and clips the survivors into valid ranges.

The only way this module can produce a dispatch is ``pipeline.run_dispatch``,
which rebuilds ``BatteryConfig``/``RiskConfig`` from scratch and re-solves the
LP. There is no code path from agent output to a charge/discharge number.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .config import SCENARIO_ORDER, SCENARIOS

# ---------------------------------------------------------------------------
# THE BOUNDARY
# ---------------------------------------------------------------------------
ALLOWED_POLICY_KEYS: dict[str, tuple[float, float]] = {
    "risk_aversion": (0.0, 3.0),
    "soc_reserve_frac": (0.0, 0.40),
    "horizon": (6, 48),
    "cycle_cost_multiplier": (0.5, 3.0),
    "price_shock": (-0.60, 0.60),
    "volatility_multiplier": (0.5, 3.0),
}

# Anything resembling physics or a dispatch instruction is explicitly refused.
LOCKED_KEYS = {
    "capacity_mwh", "power_mw", "soc_min", "soc_max", "soc", "soc_mwh",
    "round_trip_efficiency", "eta_charge", "eta_discharge", "charge_mw",
    "discharge_mw", "dispatch", "action", "reserve_mwh", "force_action",
    "cycle_cost_per_mwh", "enforce_reserve", "max_daily_cycles",
}


class PolicyViolation(ValueError):
    """Raised when a caller insists on an out-of-bounds policy key."""


def sanitize_policy(proposed: dict | None, strict: bool = False) -> tuple[dict, list[str]]:
    """Return ``(accepted_policy, rejected_notes)``.

    This is the single gate between agent reasoning and the optimiser.
    """
    accepted: dict[str, Any] = {}
    rejected: list[str] = []
    for k, v in (proposed or {}).items():
        if k in LOCKED_KEYS:
            msg = f"REJECTED '{k}': physical/dispatch parameters are optimiser-owned."
            if strict:
                raise PolicyViolation(msg)
            rejected.append(msg)
            continue
        if k not in ALLOWED_POLICY_KEYS:
            msg = f"REJECTED '{k}': not a policy parameter."
            if strict:
                raise PolicyViolation(msg)
            rejected.append(msg)
            continue
        lo, hi = ALLOWED_POLICY_KEYS[k]
        try:
            val = float(v)
        except (TypeError, ValueError):
            rejected.append(f"REJECTED '{k}': not numeric.")
            continue
        if not np.isfinite(val):
            rejected.append(f"REJECTED '{k}': not finite.")
            continue
        clipped = float(np.clip(val, lo, hi))
        if abs(clipped - val) > 1e-9:
            rejected.append(f"CLIPPED '{k}' {val:g} -> {clipped:g} (allowed {lo}-{hi}).")
        accepted[k] = int(clipped) if k == "horizon" else clipped
    return accepted, rejected


# ---------------------------------------------------------------------------
# NODES  (plain functions; LangGraph wires them, and they also run standalone)
# ---------------------------------------------------------------------------
def node_scenario_analysis(state: dict) -> dict:
    """Measure the recent market so the policy choice is evidence-based."""
    from .pipeline import get_data
    df = get_data(state.get("data_path"))
    tail = df.tail(state.get("lookback", 168))
    p = tail["price_per_mwh"].to_numpy(float)
    ren = float((tail["renewable_mw"] / tail["load_mw"].replace(0, np.nan)).mean())
    analysis = {
        "window_hours": int(len(tail)),
        "price_mean": float(np.mean(p)),
        "price_std": float(np.std(p)),
        "price_min": float(np.min(p)),
        "price_max": float(np.max(p)),
        "negative_hours": int((p < 0).sum()),
        "spike_hours": int((p > np.mean(p) + 3 * np.std(p)).sum()),
        "renewable_ratio": 0.0 if not np.isfinite(ren) else ren,
    }
    analysis["volatility_ratio"] = (analysis["price_std"] /
                                    max(abs(analysis["price_mean"]), 10.0))
    state["analysis"] = analysis
    state.setdefault("trace", []).append(
        f"Scenario analysis: last {analysis['window_hours']}h mean "
        f"{analysis['price_mean']:,.1f}/MWh, std {analysis['price_std']:,.1f}, "
        f"{analysis['negative_hours']} negative hours, "
        f"{analysis['spike_hours']} spike hours, renewable share "
        f"{analysis['renewable_ratio']:.0%}.")
    return state


def node_policy_selection(state: dict) -> dict:
    """Pick policy knobs from the named scenario, then adapt to the evidence."""
    name = str(state.get("scenario", "normal")).lower()
    sc = SCENARIOS.get(name)
    if sc is None:
        state.setdefault("trace", []).append(
            f"Unknown scenario '{name}' -> falling back to 'normal'.")
        name, sc = "normal", SCENARIOS["normal"]
    state["scenario"] = name

    proposed = {
        "risk_aversion": sc.risk_aversion,
        "soc_reserve_frac": sc.soc_reserve_frac,
        "horizon": state.get("horizon") or sc.horizon,
        "cycle_cost_multiplier": sc.cycle_cost_multiplier,
        "price_shock": sc.price_shock,
        "volatility_multiplier": sc.volatility_multiplier,
    }
    a = state.get("analysis", {})
    # Evidence-driven adaptation -- still only policy knobs.
    if a.get("volatility_ratio", 0) > 0.8:
        proposed["risk_aversion"] *= 1.25
        proposed["soc_reserve_frac"] = max(proposed["soc_reserve_frac"], 0.10)
        state.setdefault("trace", []).append(
            "Realised volatility is high -> raising risk aversion 25% and "
            "requiring at least a 10% scenario reserve.")
    if a.get("negative_hours", 0) > 0.10 * max(a.get("window_hours", 1), 1):
        proposed["cycle_cost_multiplier"] *= 0.9
        state.setdefault("trace", []).append(
            "Frequent negative prices -> slightly reducing degradation "
            "sensitivity so paid-to-charge hours are taken.")
    if state.get("risk_level") is not None:
        proposed["risk_aversion"] = float(state["risk_level"]) * 2.0

    accepted, rejected = sanitize_policy({**proposed, **(state.get("override") or {})})
    state["policy"] = accepted
    state["rejected"] = rejected
    state.setdefault("trace", []).extend(rejected)
    state["trace"].append(
        "Policy selected: " + ", ".join(f"{k}={v:g}" for k, v in accepted.items()))
    return state


def node_optimizer_tool(state: dict) -> dict:
    """The ONLY way the agent can obtain a dispatch: re-solve the LP."""
    from .pipeline import run_dispatch
    policy = dict(state.get("policy") or {})
    horizon = int(policy.pop("horizon", state.get("horizon") or 24))
    try:
        state["result"] = run_dispatch(
            horizon=horizon,
            scenario=state.get("scenario", "normal"),
            policy_overrides=policy,
            backtest_hours=int(state.get("backtest_hours", 72)),
            with_backtest=bool(state.get("with_backtest", True)),
            with_open_loop=bool(state.get("with_open_loop", True)),
            data_path=state.get("data_path"),
            use_llm=bool(state.get("use_llm", False)),
            battery=state.get("battery"),
        )
        state["trace"].append(
            f"Optimiser solved {horizon}h horizon in "
            f"{state['result']['solve_time'] * 1000:.0f} ms "
            f"({state['result']['solver']['n_vars']} vars, "
            f"{state['result']['solver']['n_constraints']} constraints).")
    except Exception as exc:                 # never crash the demo
        state["result"] = None
        state["error"] = f"{type(exc).__name__}: {exc}"
        state["trace"].append(f"Optimiser call failed: {state['error']}")
    return state


def node_result_analysis(state: dict) -> dict:
    r = state.get("result")
    if not r:
        return state
    bt = r.get("backtest") or {}
    state["result_summary"] = {
        "action": r["current_action"],
        "soc_pct": r["current_soc_pct"],
        "expected_profit": r["expected_profit"],
        "net_profit": bt.get("net_profit"),
        "regret_pct": r.get("regret_pct"),
        "equivalent_cycles": bt.get("equivalent_cycles"),
        "wait_share": (bt.get("wait_events", 0) / bt["steps"]) if bt.get("steps") else None,
    }
    state["trace"].append(
        f"Result: action {r['current_action']}, horizon value "
        f"{r['expected_profit']:,.0f}, closed-loop net "
        f"{bt.get('net_profit', float('nan')):,.0f} over {bt.get('steps', 0)}h.")
    return state


def node_risk_analysis(state: dict) -> dict:
    r = state.get("result")
    if not r:
        return state
    state["risk_summary"] = {
        "risk_score": r["risk_score"],
        "risk_level": r["risk_level"],
        "adaptive_reserve": r["adaptive_reserve"],
        "adaptive_reserve_mwh": r["adaptive_reserve_mwh"],
        "drivers": r["risk_state"].get("drivers", []),
        "notes": r["risk_notes"],
    }
    state["trace"].append(
        f"Risk {r['risk_level']} (score {r['risk_score']:.2f}) -> adaptive reserve "
        f"{r['adaptive_reserve']:.0%} of capacity "
        f"({r['adaptive_reserve_mwh']:,.1f} MWh) enforced as SOC lower bound.")
    return state


def node_operator_plan(state: dict) -> dict:
    r = state.get("result")
    if not r:
        state["operator_plan"] = {
            "narrative": "No plan: the optimiser could not be run. "
                         f"{state.get('error', '')}".strip(),
            "recommended_action": "WAIT",
            "source": "fallback",
        }
        return state
    op = dict(r["operator_plan"])
    op["scenario_rationale"] = " ".join(state.get("trace", [])[:3])
    op["agent_trace"] = list(state.get("trace", []))
    op["rejected_policy"] = list(state.get("rejected", []))
    state["operator_plan"] = op
    return state


NODES = [
    ("scenario_analysis", node_scenario_analysis),
    ("policy_selection", node_policy_selection),
    ("optimizer_tool", node_optimizer_tool),
    ("result_analysis", node_result_analysis),
    ("risk_analysis", node_risk_analysis),
    ("operator_plan", node_operator_plan),
]


# ---------------------------------------------------------------------------
def build_graph():
    """Compile the LangGraph state machine (None if LangGraph is unavailable)."""
    try:
        from langgraph.graph import END, StateGraph
    except Exception:
        return None
    try:
        g = StateGraph(dict)
        for name, fn in NODES:
            g.add_node(name, fn)
        g.set_entry_point(NODES[0][0])
        for (a, _), (b, _) in zip(NODES, NODES[1:]):
            g.add_edge(a, b)
        g.add_edge(NODES[-1][0], END)
        return g.compile()
    except Exception:
        return None


def run_agent(
    scenario: str = "normal",
    horizon: int | None = None,
    risk_level: float | None = None,
    *,
    override: dict | None = None,
    backtest_hours: int = 72,
    with_backtest: bool = True,
    with_open_loop: bool = True,
    data_path=None,
    use_llm: bool = False,
    use_graph: bool = True,
    battery=None,
) -> dict:
    """Run the scenario agent. Falls back to sequential nodes without LangGraph."""
    state: dict = {
        "scenario": scenario, "horizon": horizon, "risk_level": risk_level,
        "override": override, "backtest_hours": backtest_hours,
        "with_backtest": with_backtest, "with_open_loop": with_open_loop,
        "data_path": data_path, "use_llm": use_llm, "battery": battery,
        "trace": [],
    }
    graph = build_graph() if use_graph else None
    if graph is not None:
        try:
            out = graph.invoke(state)
            out["engine"] = "langgraph"
            return out
        except Exception as exc:
            state.setdefault("trace", []).append(
                f"LangGraph execution failed ({exc}); running nodes sequentially.")
    for _, fn in NODES:
        state = fn(state)
    state["engine"] = "sequential_fallback"
    return state


def what_if(scenarios: list[str] | None = None, horizon: int = 24,
            backtest_hours: int = 48, **kw) -> pd.DataFrame:
    """Run several scenarios and tabulate how the DISPATCH actually changed."""
    rows = []
    for name in (scenarios or list(SCENARIO_ORDER)):
        st = run_agent(name, horizon=horizon, backtest_hours=backtest_hours, **kw)
        r = st.get("result")
        if not r:
            rows.append({"scenario": name, "error": st.get("error", "failed")})
            continue
        bt = r.get("backtest") or {}
        rows.append({
            "scenario": name,
            "risk_aversion": r["policy"]["risk_aversion"],
            "action": r["current_action"],
            "risk_level": r["risk_level"],
            "reserve_%cap": 100 * r["adaptive_reserve"],
            "horizon_value": r["expected_profit"],
            "net_profit": bt.get("net_profit"),
            "cycles": bt.get("equivalent_cycles"),
            "regret_%": r.get("regret_pct"),
        })
    return pd.DataFrame(rows)
