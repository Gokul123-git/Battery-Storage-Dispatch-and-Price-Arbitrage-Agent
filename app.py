"""GridFlex AI -- operator dashboard.

Every number on this page comes from ``src.pipeline.run_dispatch`` (via the
scenario agent). There are no hardcoded KPIs: change a control and the
forecast, risk score, reserve, LP and backtest are all recomputed.

    streamlit run app.py
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

from src.agent import run_agent, sanitize_policy          # noqa: E402
from src.config import DEFAULT_BATTERY, SCENARIO_ORDER, BatteryConfig  # noqa: E402
from src.data_loader import describe                      # noqa: E402
from src.pipeline import get_data                         # noqa: E402
from src.storage import recent_runs                       # noqa: E402

st.set_page_config(page_title="GridFlex AI", page_icon="B", layout="wide")

ACCENT = {"CHARGE": "#2e86de", "DISCHARGE": "#e17055", "WAIT": "#7f8c8d"}
LABEL = {"normal": "Normal", "high_volatility": "High Volatility",
         "low_renewable": "Low Renewable", "high_renewable": "High Renewable",
         "price_spike": "Price Spike", "conservative": "Conservative"}


@st.cache_data(show_spinner=False, ttl=3600)
def compute(scenario: str, horizon: int, risk_level: float, backtest_hours: int,
            capacity: float, power: float, rt_eff: float, cycle_cost: float,
            use_llm: bool) -> dict:
    battery = BatteryConfig(
        capacity_mwh=capacity, power_mw=power, round_trip_efficiency=rt_eff,
        soc_min=DEFAULT_BATTERY.soc_min, soc_max=DEFAULT_BATTERY.soc_max,
        cycle_cost_per_mwh=cycle_cost).validate()
    state = run_agent(scenario, horizon=horizon, risk_level=risk_level,
                      backtest_hours=backtest_hours, with_backtest=backtest_hours > 0,
                      with_open_loop=True, use_llm=use_llm, battery=battery)
    return {"result": state.get("result"), "trace": state.get("trace", []),
            "engine": state.get("engine"), "error": state.get("error"),
            "operator_plan": state.get("operator_plan", {}),
            "rejected": state.get("rejected", [])}


@st.cache_data(show_spinner=False, ttl=3600)
def data_summary() -> dict:
    return describe(get_data())


def kpi(col, label, value, help_=None):
    col.metric(label, value, help=help_)


# ============================== SIDEBAR ====================================
st.sidebar.title("Controls")
scenario = st.sidebar.selectbox("Scenario", SCENARIO_ORDER,
                                format_func=lambda s: LABEL.get(s, s))
horizon = st.sidebar.select_slider("Optimisation horizon (h)", [24, 36, 48], value=24)
risk_level = st.sidebar.slider("Operator risk level", 0.0, 1.0, 0.25, 0.05,
                               help="0 = risk-neutral, 1 = maximally defensive. "
                                    "Scales the optimiser's risk aversion lambda.")
backtest_hours = st.sidebar.select_slider(
    "Closed-loop backtest window (h)", [0, 24, 48, 72, 168], value=48,
    help="Hours of rolling closed-loop simulation ending at 'now'. 0 = plan only.")

st.sidebar.markdown("**Battery**")
capacity = st.sidebar.number_input("Capacity (MWh)", 10.0, 1000.0,
                                   DEFAULT_BATTERY.capacity_mwh, 10.0)
power = st.sidebar.number_input("Power (MW)", 1.0, 500.0, DEFAULT_BATTERY.power_mw, 5.0)
rt_eff = st.sidebar.slider("Round-trip efficiency", 0.60, 1.0,
                           DEFAULT_BATTERY.round_trip_efficiency, 0.01)
cycle_cost = st.sidebar.slider("Degradation cost (/MWh throughput)", 0.0, 30.0,
                               DEFAULT_BATTERY.cycle_cost_per_mwh, 0.5)
use_llm = st.sidebar.checkbox("LLM narrative (needs ANTHROPIC_API_KEY)", False)
if st.sidebar.button("Clear caches"):
    st.cache_data.clear()
    st.rerun()

# ============================== HEADER =====================================
st.title("GridFlex AI")
st.caption("Risk-Aware, Health-Aware Battery Trading & Dispatch  -  "
           "forecast -> risk -> adaptive reserve -> LP dispatch -> closed-loop backtest")

with st.spinner("Forecasting, scoring risk, solving the LP and backtesting..."):
    try:
        out = compute(scenario, horizon, risk_level, backtest_hours,
                      capacity, power, rt_eff, cycle_cost, use_llm)
    except Exception as exc:
        st.error(f"Pipeline failed: {type(exc).__name__}: {exc}")
        st.stop()

r = out.get("result")
if not r:
    st.error(f"No dispatch produced. {out.get('error', '')}")
    st.stop()

plan = r["dispatch"]
hist = r["history"]
bt = r.get("backtest") or {}
bench = r.get("benchmark") or {}
op = out.get("operator_plan") or r["operator_plan"]

for w in r.get("warnings", []):
    st.warning(w)

# ======================= CURRENT BATTERY STATUS ============================
st.subheader("Current status")
c = st.columns(6)
action = r["current_action"]
c[0].markdown(
    f"<div style='font-size:0.8rem;color:#888'>RECOMMENDED ACTION</div>"
    f"<div style='font-size:1.9rem;font-weight:700;color:{ACCENT.get(action, '#333')}'>"
    f"{action}</div>", unsafe_allow_html=True)
kpi(c[1], "State of charge", f"{r['current_soc_pct']:.1f}%",
    f"{r['current_soc']:,.1f} of {r['battery']['capacity_mwh']:,.0f} MWh")
kpi(c[2], "Current price", f"{r['current_price']:,.1f}", f"as of {r['now_ts']}")
kpi(c[3], "Risk level", f"{r['risk_level']}", f"score {r['risk_score']:.2f}")
kpi(c[4], "Adaptive reserve", f"{r['adaptive_reserve']:.0%}",
    f"SOC floor {r['adaptive_reserve_pct']:.1f}% = {r['adaptive_reserve_mwh']:,.1f} MWh")
kpi(c[5], "Breakeven spread", f"{r['breakeven_spread']:,.1f}",
    "Spread needed to cover efficiency losses + degradation")

# ============================ PRICE FORECAST ===============================
left, right = st.columns([3, 2])
with left:
    st.subheader("Price forecast")
    f = r["forecast"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist["ts"], y=hist["price_per_mwh"], name="Actual (history)",
                             line=dict(color="#34495e", width=1.6)))
    up = f["forecast_price"] + 1.96 * f["sigma"]
    dn = f["forecast_price"] - 1.96 * f["sigma"]
    fig.add_trace(go.Scatter(x=list(f["ts"]) + list(f["ts"][::-1]),
                             y=list(up) + list(dn[::-1]), fill="toself",
                             fillcolor="rgba(46,134,222,0.15)", line=dict(width=0),
                             hoverinfo="skip", name="95% band (validation residuals)"))
    fig.add_trace(go.Scatter(x=f["ts"], y=f["forecast_price"], name="Forecast",
                             line=dict(color="#2e86de", width=2.4, dash="dot")))
    if f["actual_price"].notna().any():
        fig.add_trace(go.Scatter(x=f["ts"], y=f["actual_price"], name="Actual (realised)",
                                 line=dict(color="#95a5a6", width=1.2)))
    fig.add_vline(x=pd.to_datetime(r["now_ts"]), line_dash="dash", line_color="#c0392b")
    fig.add_hline(y=0, line_color="#bbb", line_width=1)
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10),
                      legend=dict(orientation="h", y=1.12),
                      yaxis_title="Price /MWh")
    st.plotly_chart(fig, width='stretch')

with right:
    st.subheader("Battery dispatch plan")
    fig2 = make_subplots(specs=[[{"secondary_y": True}]])
    fig2.add_trace(go.Bar(x=plan["ts"], y=plan["charge_mw"], name="Charge (MW)",
                          marker_color=ACCENT["CHARGE"]), secondary_y=False)
    fig2.add_trace(go.Bar(x=plan["ts"], y=-plan["discharge_mw"], name="Discharge (MW)",
                          marker_color=ACCENT["DISCHARGE"]), secondary_y=False)
    fig2.add_trace(go.Scatter(x=plan["ts"], y=plan["soc_pct"], name="SOC %",
                              line=dict(color="#2d3436", width=2.2)), secondary_y=True)
    fig2.add_hline(y=r["adaptive_reserve_pct"], line_dash="dot", line_color="#c0392b",
                   annotation_text="adaptive reserve", secondary_y=True)
    fig2.update_layout(height=340, barmode="relative",
                       margin=dict(l=10, r=10, t=10, b=10),
                       legend=dict(orientation="h", y=1.12))
    fig2.update_yaxes(title_text="MW", secondary_y=False)
    fig2.update_yaxes(title_text="SOC %", range=[0, 100], secondary_y=True)
    st.plotly_chart(fig2, width='stretch')

# ============================== PLAN TABLE =================================
st.subheader(f"{horizon}-hour operating plan")
tbl = plan[["ts", "actual_price", "forecast_price", "sigma", "charge_mw",
            "discharge_mw", "soc_mwh", "soc_pct", "action", "expected_profit"]].copy()
tbl.insert(len(tbl.columns) - 1, "risk", r["risk_level"])
tbl.columns = ["Timestamp", "Price (realised)", "Forecast", "+/-1s", "Charge MW",
               "Discharge MW", "SOC MWh", "SOC %", "Action", "Risk", "Exp. profit"]
st.dataframe(tbl.style.format({
    "Price (realised)": "{:,.1f}", "Forecast": "{:,.1f}", "+/-1s": "{:,.1f}",
    "Charge MW": "{:,.2f}", "Discharge MW": "{:,.2f}", "SOC MWh": "{:,.1f}",
    "SOC %": "{:,.1f}", "Exp. profit": "{:,.0f}"}),
    width='stretch', height=280)

# ========================= ECONOMICS / BENCHMARK ===========================
st.subheader("Closed-loop economics")
if bt:
    e = st.columns(6)
    kpi(e[0], "Net profit", f"{bt['net_profit']:,.0f}",
        f"{bt['steps']}h closed loop ending {bt['end']}")
    kpi(e[1], "Discharge revenue", f"{bt['revenue']:,.0f}")
    kpi(e[2], "Charging cost", f"{bt['charging_cost']:,.0f}")
    kpi(e[3], "Degradation cost", f"{bt['degradation_cost']:,.0f}")
    kpi(e[4], "Throughput", f"{bt['throughput_mwh']:,.0f} MWh")
    kpi(e[5], "Equivalent cycles", f"{bt['equivalent_cycles']:.2f}")

    b = st.columns(6)
    kpi(b[0], "Closed-loop profit", f"{bench.get('closed_loop_profit', float('nan')):,.0f}")
    kpi(b[1], "Open-loop profit", f"{bench.get('open_loop_profit', float('nan')):,.0f}",
        "Plan once per horizon, never re-forecast")
    kpi(b[2], "Perfect foresight", f"{bench.get('perfect_foresight_profit', float('nan')):,.0f}",
        "Benchmark only: knows the actual future prices")
    kpi(b[3], "Regret", f"{bench.get('regret', float('nan')):,.0f}")
    kpi(b[4], "Regret %", f"{bench.get('regret_pct', float('nan')):.1f}%")
    kpi(b[5], "Value of feedback", f"{bench.get('closed_vs_open', float('nan')):,.0f}",
        "Closed-loop minus open-loop profit")

    a = st.columns(6)
    kpi(a[0], "CHARGE / DISCHARGE / WAIT",
        f"{bt['charge_events']} / {bt['discharge_events']} / {bt['wait_events']}")
    kpi(a[1], "Forecast MAE", f"{bt['forecast_mae']:,.2f}")
    kpi(a[2], "Forecast RMSE", f"{bt['forecast_rmse']:,.2f}")
    kpi(a[3], "Dispatch-weighted MAE", f"{bt['dispatch_weighted_mae']:,.2f}",
        "Error in the hours where we actually traded -- the error that costs money")
    kpi(a[4], "Avg LP solve", f"{bt['avg_solve_seconds'] * 1000:,.0f} ms",
        f"{r['solver']['n_vars']} vars / {r['solver']['n_constraints']} constraints, "
        f"max {bt['max_solve_seconds'] * 1000:,.0f} ms")
    kpi(a[5], "Backtest runtime", f"{bt['runtime_seconds']:,.1f} s",
        f"{bt['solver_calls']} solver calls, {bt['reserve_violations']} reserve breaches")

    g1, g2 = st.columns([3, 2])
    with g1:
        log = r["backtest_log"]
        fig3 = make_subplots(specs=[[{"secondary_y": True}]])
        fig3.add_trace(go.Scatter(x=log["ts"], y=log["price"], name="Actual price",
                                  line=dict(color="#34495e", width=1.4)))
        fig3.add_trace(go.Scatter(x=log["ts"], y=log["forecast"], name="Forecast (t+1)",
                                  line=dict(color="#2e86de", width=1.2, dash="dot")))
        fig3.add_trace(go.Bar(x=log["ts"], y=log["charge_mw"], name="Charge",
                              marker_color=ACCENT["CHARGE"], opacity=0.55),
                       secondary_y=True)
        fig3.add_trace(go.Bar(x=log["ts"], y=-log["discharge_mw"], name="Discharge",
                              marker_color=ACCENT["DISCHARGE"], opacity=0.55),
                       secondary_y=True)
        fig3.update_layout(height=320, barmode="relative",
                           title="Closed-loop execution (realised prices)",
                           margin=dict(l=10, r=10, t=40, b=10),
                           legend=dict(orientation="h", y=1.12))
        fig3.update_yaxes(title_text="Price /MWh", secondary_y=False)
        fig3.update_yaxes(title_text="MW", secondary_y=True)
        st.plotly_chart(fig3, width='stretch')
    with g2:
        ei = r.get("error_impact")
        if isinstance(ei, pd.DataFrame) and not ei.empty:
            st.markdown("**Where forecast error cost money**")
            st.dataframe(ei.style.format({
                "mean_abs_error": "{:,.1f}", "traded_mwh": "{:,.0f}",
                "profit": "{:,.0f}", "profit_per_mwh": "{:,.2f}"}),
                width='stretch', height=250)
            st.caption("Hours bucketed by absolute forecast error. Error is only "
                       "expensive where the battery actually traded.")
else:
    st.info("Backtest window set to 0 - showing the forward plan only.")

# ================================ RISK =====================================
st.subheader("Risk engine")
rs = r["risk_state"]
q = st.columns(5)
kpi(q[0], "Forecast uncertainty", f"{rs['mean_sigma']:,.1f} /MWh", "mean 1-sigma over the horizon")
kpi(q[1], "Market volatility", f"{rs['price_volatility']:,.1f} /MWh", "trailing 168h realised std")
kpi(q[2], "Risk score", f"{rs['risk_score']:.2f}",
    f"{rs['uncertainty_norm']:.2f} uncertainty + {rs['volatility_norm']:.2f} volatility (weighted)")
kpi(q[3], "Adaptive reserve", f"{rs['reserve_frac']:.0%}", "extra SOC above the hard floor")
kpi(q[4], "Risk aversion lambda", f"{rs['risk_aversion']:.2f}",
    "Shifts optimiser prices by lambda*sigma against us on both legs")
st.markdown("**Why the reserve is at this level**")
for d in rs.get("drivers", []):
    st.markdown(f"- {d}")
with st.expander("Risk notes (deterministic, number-backed)"):
    for n in r["risk_notes"]:
        st.markdown(f"- {n}")
rt = r.get("risk_trace")
if isinstance(rt, pd.DataFrame) and not rt.empty:
    figr = go.Figure()
    figr.add_trace(go.Scatter(x=rt["ts"], y=rt["risk_score"], name="Risk score",
                              line=dict(color="#c0392b")))
    figr.add_trace(go.Scatter(x=rt["ts"], y=rt["reserve_frac"], name="Reserve (frac of capacity)",
                              line=dict(color="#2e86de")))
    figr.update_layout(height=240, margin=dict(l=10, r=10, t=30, b=10),
                       title="Risk score and adaptive reserve through the backtest",
                       legend=dict(orientation="h", y=1.2))
    st.plotly_chart(figr, width='stretch')

# ============================ OPERATOR PLAN ================================
st.subheader("Operator plan")
p1, p2 = st.columns([3, 2])
with p1:
    st.markdown(f"**Recommended action:** :blue[{op.get('recommended_action', action)}]")
    st.markdown(f"**Why:** {op.get('why', '')}")
    st.markdown(f"**Market outlook:** {op.get('market_outlook', '')}")
    st.markdown(f"**Forecast:** {op.get('forecast_summary', '')}")
    st.markdown(f"**Battery:** {op.get('battery_status', '')}")
    st.markdown(f"**SOC trajectory:** {op.get('soc_trajectory', '')}")
    st.markdown(f"**Degradation:** {op.get('degradation', '')}")
    if op.get("narrative_llm"):
        with st.expander("LLM-written operator note"):
            st.write(op["narrative_llm"])
    st.caption(f"Narrative source: {op.get('source', 'deterministic_template')}")
with p2:
    st.markdown(f"**Scenario:** {op.get('scenario', scenario)}")
    st.markdown(f"**Risk:** {op.get('risk', '')}")
    st.markdown("**Warnings**")
    ws = op.get("warnings", [])
    st.markdown("\n".join(f"- {w}" for w in ws) if ws else "- none")
    with st.expander("Agent trace (LangGraph)"):
        st.caption(f"engine: {out.get('engine')}")
        for line in out.get("trace", []):
            st.markdown(f"- {line}")
        rej = out.get("rejected", [])
        if rej:
            st.markdown("**Policy requests refused at the optimiser boundary**")
            for line in rej:
                st.markdown(f"- {line}")

# ======================== AGENT BOUNDARY DEMO ==============================
with st.expander("Agent boundary: try to make the agent override the battery"):
    st.caption("The agent may only set policy knobs. Physical/dispatch keys are "
               "rejected by src.agent.sanitize_policy before the optimiser is called.")
    demo = st.text_input("Proposed policy (key=value, comma separated)",
                         "risk_aversion=1.4, soc_mwh=95, charge_mw=25, horizon=48")
    proposed = {}
    for part in demo.split(","):
        if "=" in part:
            k, v = part.split("=", 1)
            proposed[k.strip()] = v.strip()
    ok, rej = sanitize_policy(proposed)
    st.write({"accepted (reach the LP)": ok, "refused / clipped": rej})

# ============================== DATA / RUNS ================================
with st.expander("Dataset and run log"):
    st.write(data_summary())
    runs = recent_runs(10)
    if not runs.empty:
        st.markdown("**Recent runs (SQLite)**")
        st.dataframe(runs, width='stretch')

st.caption("Prototype / simulation. Not connected to a physical BESS or a live "
           "market. See README for deployment requirements.")
