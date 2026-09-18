"""Closed-loop (and open-loop) rolling simulation of the whole control stack.

CLOSED LOOP (``replan_every=1``) -- the honest evaluation
--------------------------------------------------------
For every decision hour i:

  1. Slice history up to i only. Future prices are blanked inside
     ``PriceForecaster.predict_window``, so nothing after i can reach a feature.
  2. Forecast i+1 .. i+H and take the per-hour error scale with it.
  3. Recompute the risk score and the adaptive SOC reserve from that error
     scale plus trailing realised volatility.
  4. Solve the dispatch LP over the H-hour horizon.
  5. EXECUTE ONLY THE FIRST INTERVAL (hour i+1).
  6. Settle it at the ACTUAL price of hour i+1, update SOC, step forward.

OPEN LOOP (``replan_every=H``) -- the comparison
------------------------------------------------
Identical, except the plan produced at the start of each block is followed for
H hours without re-forecasting. Execution is still clipped to the battery's
physical limits, so the open-loop run stays feasible; the gap in profit is the
value of re-optimising on new information.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import BatteryConfig, RiskConfig
from .optimizer import InfeasibleDispatch, classify, optimize_dispatch
from .risk import RiskState, adjust_prices, apply_risk_state, compute_risk_state

HISTORY_WINDOW = 720          # hours of history handed to the feature builder
VOL_WINDOW = 168              # trailing hours used for realised volatility
TOL = 1e-4


@dataclass
class BacktestResult:
    """Per-hour execution log plus aggregated economics and diagnostics."""

    log: pd.DataFrame
    metrics: dict
    risk_trace: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def net_profit(self) -> float:
        return float(self.metrics.get("net_profit", 0.0))


def run_backtest(
    df: pd.DataFrame,
    forecaster,
    battery: BatteryConfig,
    risk: RiskConfig,
    *,
    start_idx: int,
    steps: int,
    horizon: int = 24,
    soc0_frac: float | None = None,
    replan_every: int = 1,
    adaptive_reserve: bool = True,
    price_shock: float = 0.0,
    volatility_multiplier: float = 1.0,
    cycle_cost_multiplier: float = 1.0,
    progress=None,
) -> BacktestResult:
    """Roll the controller forward ``steps`` hours from ``start_idx``."""
    battery.validate()
    risk.validate()
    if steps < 1:
        raise ValueError("steps must be >= 1")
    if start_idx < HISTORY_WINDOW // 2:
        raise ValueError("start_idx too early: not enough history for features")
    if start_idx + steps + horizon > len(df):
        steps = max(len(df) - start_idx - horizon - 1, 0)
        if steps < 1:
            raise ValueError("not enough data after start_idx for this horizon")

    price = df["price_per_mwh"].to_numpy(float)
    ts = df["ts"]
    soc = float(battery.capacity_mwh * (soc0_frac if soc0_frac is not None
                                        else (battery.soc_min + battery.soc_max) / 2))

    eta_c, eta_d = battery.eta_charge, battery.eta_discharge
    cc = battery.cycle_cost_per_mwh * float(cycle_cost_multiplier)

    rows: list[dict] = []
    risk_rows: list[dict] = []
    solve_times: list[float] = []
    plan_df: pd.DataFrame | None = None
    plan_pos = 0
    reserve_mwh = battery.e_min
    state: RiskState | None = None
    risk_eff = risk
    failures = 0
    t_start = time.perf_counter()

    for k in range(steps):
        i = start_idx + k

        if plan_df is None or plan_pos >= replan_every or plan_pos >= len(plan_df):
            lo = max(0, i - HISTORY_WINDOW)
            window = df.iloc[lo:i + 1 + horizon]
            local_idx = i - lo
            fc = forecaster.predict_window(window, local_idx, horizon=horizon)

            recent = price[max(0, i - VOL_WINDOW):i + 1]
            if adaptive_reserve:
                state = compute_risk_state(
                    fc.error_scale, recent, battery, risk,
                    volatility_multiplier=volatility_multiplier,
                    scenario_reserve_floor=risk.soc_reserve_frac,
                )
                risk_eff = apply_risk_state(risk, state)
            else:
                state = compute_risk_state(
                    fc.error_scale, recent, battery, risk,
                    volatility_multiplier=volatility_multiplier,
                    scenario_reserve_floor=risk.soc_reserve_frac,
                    base_frac=risk.soc_reserve_frac, max_frac=risk.soc_reserve_frac,
                )
                risk_eff = risk

            prices = adjust_prices(fc.forecast, fc.error_scale, risk_eff,
                                   price_shock=price_shock)
            try:
                dp = optimize_dispatch(
                    prices.buy, prices.sell, battery, risk_eff, soc,
                    ts=fc.ts, cycle_cost_multiplier=cycle_cost_multiplier,
                )
                plan_df = dp.plan
                reserve_mwh = dp.reserve_mwh
                solve_times.append(dp.solve_seconds)
            except (InfeasibleDispatch, ValueError):
                # Never crash the simulation: hold position for this block.
                failures += 1
                plan_df = pd.DataFrame({
                    "ts": fc.ts, "charge_mw": 0.0, "discharge_mw": 0.0,
                    "sell_price": fc.forecast, "buy_price": fc.forecast,
                })
            plan_pos = 0
            fc_head = float(fc.forecast[0])
            sigma_head = float(fc.error_scale[0])
        else:
            fc_head = float(plan_df["sell_price"].iloc[plan_pos])
            sigma_head = float("nan")

        # ---- execute exactly one interval: hour i+1 ---------------------
        c = float(plan_df["charge_mw"].iloc[plan_pos])
        d = float(plan_df["discharge_mw"].iloc[plan_pos])

        # Physical clipping (matters for open loop following a stale plan).
        c = min(c, battery.power_mw, max((battery.e_max - soc) / eta_c, 0.0))
        d = min(d, battery.power_mw, max((soc - reserve_mwh) * eta_d, 0.0))
        if c > TOL and d > TOL:            # defensive: should never happen
            c, d = (c, 0.0) if c >= d else (0.0, d)

        p_actual = float(price[i + 1])
        soc_prev = soc
        soc = float(np.clip(soc + c * eta_c - d / eta_d, 0.0, battery.capacity_mwh))

        revenue = d * p_actual
        cost = c * p_actual
        deg = cc * (c + d)
        rows.append({
            "ts": ts.iloc[i + 1],
            "decision_ts": ts.iloc[i],
            "price": p_actual,
            "forecast": fc_head,
            "forecast_error": fc_head - p_actual,
            "sigma": sigma_head,
            "charge_mw": c,
            "discharge_mw": d,
            "action": classify(c, d),
            "soc_start_mwh": soc_prev,
            "soc_mwh": soc,
            "soc_pct": 100.0 * soc / battery.capacity_mwh,
            "reserve_mwh": reserve_mwh,
            "revenue": revenue,
            "cost": cost,
            "degradation_cost": deg,
            "profit": revenue - cost - deg,
            "risk_score": state.risk_score if state else np.nan,
            "risk_level": state.level if state else "n/a",
        })
        if state is not None and plan_pos == 0:
            risk_rows.append({"ts": ts.iloc[i], **{
                k2: v for k2, v in state.to_dict().items() if k2 != "drivers"}})

        plan_pos += 1
        if progress is not None and (k % 10 == 0 or k == steps - 1):
            try:
                progress((k + 1) / steps)
            except Exception:
                pass

    log = pd.DataFrame(rows)
    runtime = time.perf_counter() - t_start
    metrics = summarise(log, battery, runtime, solve_times, failures,
                        replan_every=replan_every, horizon=horizon)
    return BacktestResult(log=log, metrics=metrics,
                          risk_trace=pd.DataFrame(risk_rows))


def summarise(log: pd.DataFrame, battery: BatteryConfig, runtime: float,
              solve_times, failures: int, *, replan_every: int, horizon: int) -> dict:
    """Aggregate the execution log into the KPIs the dashboard reports."""
    if log.empty:
        return {"net_profit": 0.0, "steps": 0}
    thr = float((log["charge_mw"] + log["discharge_mw"]).sum())
    err = log["forecast_error"].to_numpy(float)
    w = (log["charge_mw"] + log["discharge_mw"]).to_numpy(float)
    dispatch_weighted_mae = (float(np.sum(np.abs(err) * w) / np.sum(w))
                             if np.sum(w) > TOL else 0.0)
    # A violation is *crossing* the floor by discharging. Hours that are below
    # the floor only because risk RAISED the floor are tracked separately: the
    # LP relaxes to the current SOC in that case and refills instead of failing.
    below = log["soc_mwh"] < log["reserve_mwh"] - 1e-3
    was_above = log["soc_start_mwh"] >= log["reserve_mwh"] - 1e-3
    viol = int((below & was_above).sum())
    below_after_raise = int((below & ~was_above).sum())
    return {
        "steps": int(len(log)),
        "start": str(log["ts"].iloc[0]),
        "end": str(log["ts"].iloc[-1]),
        "revenue": float(log["revenue"].sum()),
        "charging_cost": float(log["cost"].sum()),
        "degradation_cost": float(log["degradation_cost"].sum()),
        "net_profit": float(log["profit"].sum()),
        "throughput_mwh": thr,
        "equivalent_cycles": thr / (2.0 * battery.usable_mwh),
        "charge_events": int((log["action"] == "CHARGE").sum()),
        "discharge_events": int((log["action"] == "DISCHARGE").sum()),
        "wait_events": int((log["action"] == "WAIT").sum()),
        "avg_soc_pct": float(log["soc_pct"].mean()),
        "min_soc_pct": float(log["soc_pct"].min()),
        "max_soc_pct": float(log["soc_pct"].max()),
        "reserve_violations": viol,
        "hours_below_raised_reserve": below_after_raise,
        "forecast_mae": float(np.mean(np.abs(err))),
        "forecast_rmse": float(np.sqrt(np.mean(err ** 2))),
        "forecast_bias": float(np.mean(err)),
        "dispatch_weighted_mae": dispatch_weighted_mae,
        "solver_calls": int(len(solve_times)),
        "avg_solve_seconds": float(np.mean(solve_times)) if len(solve_times) else 0.0,
        "max_solve_seconds": float(np.max(solve_times)) if len(solve_times) else 0.0,
        "solver_failures": int(failures),
        "runtime_seconds": float(runtime),
        "replan_every": int(replan_every),
        "horizon": int(horizon),
        "mode": "closed_loop" if replan_every == 1 else "open_loop",
    }


def closed_loop(df, forecaster, battery, risk, **kw) -> BacktestResult:
    kw.pop("replan_every", None)
    return run_backtest(df, forecaster, battery, risk, replan_every=1, **kw)


def open_loop(df, forecaster, battery, risk, *, horizon: int = 24, **kw) -> BacktestResult:
    """Plan once per horizon and follow it blind -- the 'no feedback' control."""
    kw.pop("replan_every", None)
    return run_backtest(df, forecaster, battery, risk, horizon=horizon,
                        replan_every=horizon, **kw)
