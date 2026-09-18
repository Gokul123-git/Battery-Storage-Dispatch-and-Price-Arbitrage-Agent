"""Operator-facing narrative, generated from computed numbers only.

Every sentence here is a template filled from values produced by the
forecaster, the risk engine, the LP and the backtest. Nothing is invented. If
an LLM is configured it is allowed to REWRITE this text for readability, and
the deterministic version is always kept alongside it, so the demo works with
no API key at all.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from .config import BatteryConfig

ACTION_VERB = {
    "CHARGE": "CHARGE (buy energy now)",
    "DISCHARGE": "DISCHARGE (sell energy now)",
    "WAIT": "WAIT (no economically viable arbitrage this interval)",
}


def build_operator_plan(
    *,
    battery: BatteryConfig,
    plan: pd.DataFrame,
    action: str,
    soc_mwh: float,
    current_price: float,
    forecast: np.ndarray,
    sigma: np.ndarray,
    risk_state,
    scenario_name: str,
    scenario_desc: str,
    risk_notes: list[str],
    expected_profit: float,
    horizon: int,
    warnings: list[str] | None = None,
) -> dict:
    """Return the structured operator plan (sections + a joined narrative)."""
    f = np.asarray(forecast, float)
    lo_i, hi_i = int(np.argmin(f)), int(np.argmax(f))
    ts = pd.to_datetime(plan["ts"])
    lo_t, hi_t = ts.iloc[lo_i], ts.iloc[hi_i]
    spread = float(f[hi_i] - f[lo_i])
    breakeven = battery.breakeven_spread(float(f[lo_i]))
    soc_pct = 100.0 * soc_mwh / battery.capacity_mwh

    charge_h = int((plan["charge_mw"] > 1e-3).sum())
    dis_h = int((plan["discharge_mw"] > 1e-3).sum())
    wait_h = int(horizon - charge_h - dis_h)

    outlook = (
        f"Over the next {horizon}h the forecast ranges from "
        f"{f[lo_i]:,.1f}/MWh at {lo_t:%a %H:%M} to {f[hi_i]:,.1f}/MWh at "
        f"{hi_t:%a %H:%M}, a spread of {spread:,.1f}/MWh against a round-trip "
        f"breakeven of {breakeven:,.1f}/MWh. "
        + ("The spread clears breakeven, so cycling is economic."
           if spread > breakeven else
           "The spread does not clear breakeven, so most hours should idle.")
    )
    forecast_summary = (
        f"Mean forecast {float(np.mean(f)):,.1f}/MWh, 1-sigma error "
        f"{float(np.mean(sigma)):,.1f}/MWh (from validation residuals, widening "
        f"with lead time). Cheapest hours cluster around {lo_t:%H:%M}, richest "
        f"around {hi_t:%H:%M}."
    )
    battery_status = (
        f"SOC {soc_mwh:,.1f} MWh ({soc_pct:.1f}% of {battery.capacity_mwh:,.0f} MWh), "
        f"power limit {battery.power_mw:,.0f} MW, round-trip efficiency "
        f"{battery.round_trip_efficiency:.0%}, degradation "
        f"{battery.cycle_cost_per_mwh:,.2f}/MWh of throughput."
    )
    economics = (
        f"Planned horizon value {expected_profit:,.0f} (risk-adjusted prices): "
        f"{charge_h}h charging, {dis_h}h discharging, {wait_h}h idle. "
        f"Current spot {current_price:,.1f}/MWh."
    )
    traj = (
        f"SOC is planned to move from {soc_pct:.1f}% to "
        f"{float(plan['soc_pct'].iloc[-1]):.1f}%, with a low of "
        f"{float(plan['soc_pct'].min()):.1f}% and a high of "
        f"{float(plan['soc_pct'].max()):.1f}%."
    )
    risk_section = (
        f"Risk level {risk_state.level} (score {risk_state.risk_score:.2f}). "
        f"Adaptive reserve {risk_state.reserve_frac:.0%} of capacity above the "
        f"{battery.soc_min:.0%} floor, i.e. SOC is held at or above "
        f"{risk_state.reserve_soc_pct:.1f}% ({risk_state.reserve_mwh:,.1f} MWh) "
        f"inside the optimiser."
    )
    degradation = (
        f"Every MWh moved costs {battery.cycle_cost_per_mwh:,.2f} in throughput "
        f"degradation, charged on both the charge and the discharge leg. The "
        f"optimiser therefore requires at least {breakeven:,.1f}/MWh of spread "
        "before it will cycle."
    )
    why = _why(action, f, lo_i, hi_i, spread, breakeven, risk_state, soc_pct, battery)

    sections = {
        "market_outlook": outlook,
        "forecast_summary": forecast_summary,
        "battery_status": battery_status,
        "recommended_action": ACTION_VERB.get(action, action),
        "economic_opportunity": economics,
        "soc_trajectory": traj,
        "risk": risk_section,
        "adaptive_reserve": (
            f"{risk_state.reserve_frac:.0%} of capacity "
            f"({risk_state.reserve_mwh:,.1f} MWh floor)"),
        "degradation": degradation,
        "scenario": f"{scenario_name}: {scenario_desc}",
        "why": why,
        "warnings": list(warnings or []) + list(risk_notes),
    }
    sections["narrative"] = "\n\n".join([
        f"RECOMMENDED ACTION: {sections['recommended_action']}",
        why, outlook, forecast_summary, battery_status, economics, traj,
        risk_section, degradation,
    ])
    sections["source"] = "deterministic_template"
    return sections


def _why(action, f, lo_i, hi_i, spread, breakeven, risk_state, soc_pct, battery) -> str:
    if action == "CHARGE":
        return (
            f"Charging now: the current interval is among the cheapest in the "
            f"horizon and the {spread:,.1f}/MWh spread to the "
            f"{f[hi_i]:,.1f}/MWh peak more than covers the {breakeven:,.1f}/MWh "
            "round-trip cost of efficiency losses plus degradation."
        )
    if action == "DISCHARGE":
        return (
            f"Discharging now: the current interval is near the horizon peak "
            f"({f[hi_i]:,.1f}/MWh) and selling still leaves SOC at or above the "
            f"{risk_state.reserve_soc_pct:.1f}% adaptive reserve."
        )
    if spread <= breakeven:
        return (
            f"Waiting: the {spread:,.1f}/MWh forecast spread is below the "
            f"{breakeven:,.1f}/MWh needed to pay for efficiency losses and "
            "degradation. Cycling here would consume battery life for no margin."
        )
    return (
        f"Waiting: a better spread exists later in the horizon, and at risk "
        f"level {risk_state.level} the optimiser prefers to hold SOC "
        f"({soc_pct:.1f}%) for that opportunity rather than trade this interval."
    )


# -- optional LLM rewrite ---------------------------------------------------
def llm_polish(sections: dict, model: str = "claude-sonnet-5") -> dict:
    """Ask Claude to rewrite the narrative. Numbers are supplied, never asked for.

    Silently returns the deterministic text if no key / no package / any error.
    """
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        return sections
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=key)
        msg = client.messages.create(
            model=model,
            max_tokens=700,
            system=(
                "You are a grid battery trading desk assistant. Rewrite the "
                "supplied dispatch briefing as a concise operator note of at "
                "most 180 words. Use ONLY the numbers given. Never invent or "
                "round away figures, never add recommendations of your own."
            ),
            messages=[{"role": "user", "content": sections["narrative"]}],
        )
        text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
        if text.strip():
            out = dict(sections)
            out["narrative_llm"] = text.strip()
            out["source"] = f"llm:{model}"
            return out
    except Exception as exc:
        sections = dict(sections)
        sections["llm_error"] = str(exc)
    return sections
