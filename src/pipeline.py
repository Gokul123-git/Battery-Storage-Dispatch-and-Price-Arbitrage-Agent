"""THE integration layer. Everything (dashboard, agent, tests, CLI) calls this.

    from src.pipeline import run_dispatch
    res = run_dispatch(horizon=24, scenario="normal", risk_level=0.5)

``run_dispatch`` wires the full chain: data -> features -> forecast ->
uncertainty -> risk score -> adaptive reserve -> LP -> action -> closed-loop
backtest -> open-loop comparison -> perfect-foresight benchmark -> regret ->
operator plan. No business logic lives in the UI.
"""
from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from . import benchmark as bm
from . import storage
from .backtest import VOL_WINDOW, run_backtest
from .config import (DEFAULT_BATTERY, MODEL_DIR, SCENARIOS, BatteryConfig,
                     ConfigError, RiskConfig, ScenarioConfig)
from .data_loader import load_market_data
from .forecaster import PriceForecaster
from .operator_plan import build_operator_plan, llm_polish
from .optimizer import (InfeasibleDispatch, check_feasible,
                        check_no_simultaneous, optimize_dispatch)
from .risk import (adjust_prices, apply_risk_state, compute_risk_state,
                   risk_notes)

_DATA_CACHE: dict = {}
_MODEL_CACHE: dict = {}
_RESULT_CACHE: dict = {}

MAX_HORIZON = 48


# --------------------------------------------------------------------------
def get_data(path: str | Path | None = None, refresh: bool = False) -> pd.DataFrame:
    """Load (and cache) the market series; generates synthetic data if absent."""
    key = str(path)
    if refresh or key not in _DATA_CACHE:
        _DATA_CACHE[key] = load_market_data(path)
    return _DATA_CACHE[key]


def get_forecaster(df: pd.DataFrame, horizon: int, refresh: bool = False) -> PriceForecaster:
    """Train once per horizon and reuse; persisted to ``models/``.

    The model is fitted on the chronologically FIRST 70% of the series, so any
    decision made in the recent tail is genuinely out of sample.
    """
    horizon = int(horizon)
    key = (horizon, len(df), str(df["ts"].iloc[-1]))
    if not refresh and key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    path = MODEL_DIR / f"forecaster_h{horizon}_{len(df)}.joblib"
    if not refresh and path.exists():
        try:
            fc = PriceForecaster.load(path)
            _MODEL_CACHE[key] = fc
            return fc
        except Exception:
            pass

    fc = PriceForecaster(horizon=horizon).fit(df)
    try:
        fc.save(path)
    except Exception:
        pass
    _MODEL_CACHE[key] = fc
    return fc


def resolve_policy(scenario: str = "normal", risk_level: float | None = None,
                   overrides: dict | None = None) -> tuple[ScenarioConfig, RiskConfig]:
    """Scenario name + optional operator/agent tuning -> validated configs.

    ``overrides`` may only touch POLICY knobs (see ``ALLOWED_POLICY_KEYS`` in
    :mod:`src.agent`); battery physics is never reachable from here.
    """
    sc = SCENARIOS.get(str(scenario).lower())
    if sc is None:
        raise ConfigError(
            f"unknown scenario '{scenario}'. Known: {sorted(SCENARIOS)}")
    sc = replace(sc)
    for k, v in (overrides or {}).items():
        if hasattr(sc, k) and k != "name":
            setattr(sc, k, v)
    if risk_level is not None:
        sc.risk_aversion = float(np.clip(risk_level, 0.0, 1.0)) * 2.0
    sc.horizon = int(np.clip(sc.horizon, 1, MAX_HORIZON))

    risk = RiskConfig(
        risk_aversion=max(float(sc.risk_aversion), 0.0),
        soc_reserve_frac=float(np.clip(sc.soc_reserve_frac, 0.0, 0.5)),
        max_daily_cycles=2.0,
        enforce_reserve=True,
    ).validate()
    return sc, risk


# --------------------------------------------------------------------------
def run_dispatch(
    horizon: int = 24,
    scenario: str = "normal",
    risk_level: float | None = None,
    *,
    battery: BatteryConfig | None = None,
    policy_overrides: dict | None = None,
    data_path: str | Path | None = None,
    decision_idx: int | None = None,
    soc_frac: float | None = None,
    backtest_hours: int = 72,
    with_backtest: bool = True,
    with_open_loop: bool = True,
    use_llm: bool = False,
    log_to_db: bool = True,
    cache: bool = True,
    progress=None,
) -> dict:
    """Run the full stack and return one structured result dictionary."""
    t0 = time.perf_counter()
    battery = (battery or DEFAULT_BATTERY).validate()
    sc, risk = resolve_policy(scenario, risk_level, policy_overrides)
    horizon = int(np.clip(horizon or sc.horizon, 1, MAX_HORIZON))
    sc.horizon = horizon

    ck = (horizon, sc.name, round(sc.risk_aversion, 4), round(sc.soc_reserve_frac, 4),
          round(sc.price_shock, 4), round(sc.volatility_multiplier, 4),
          round(sc.cycle_cost_multiplier, 4), backtest_hours, with_backtest,
          with_open_loop, soc_frac, decision_idx, str(data_path),
          battery.capacity_mwh, battery.power_mw, battery.cycle_cost_per_mwh,
          battery.round_trip_efficiency, use_llm)
    if cache and ck in _RESULT_CACHE:
        return _RESULT_CACHE[ck]

    warnings: list[str] = []
    df = get_data(data_path)
    # One model per horizon: features are shifted by the horizon, so a 24h
    # model is strictly better at 24h than a 48h model truncated.
    forecaster = get_forecaster(df, horizon)

    steps = int(max(backtest_hours, 0)) if with_backtest else 0
    if decision_idx is None:
        decision_idx = len(df) - horizon - 2
    decision_idx = int(np.clip(decision_idx, 24 * 30, len(df) - horizon - 2))
    bt_start = decision_idx - steps
    if with_backtest and bt_start < 24 * 30:
        steps = max(decision_idx - 24 * 30, 0)
        bt_start = decision_idx - steps
        warnings.append("Backtest window shortened: not enough history.")

    soc0 = battery.capacity_mwh * (soc_frac if soc_frac is not None
                                   else (battery.soc_min + battery.soc_max) / 2)

    # ---- 1. closed-loop backtest up to 'now' ---------------------------
    closed = open_res = None
    soc_now = soc0
    if with_backtest and steps > 0:
        closed = run_backtest(
            df, forecaster, battery, risk, start_idx=bt_start, steps=steps,
            horizon=horizon, soc0_frac=(soc0 / battery.capacity_mwh),
            replan_every=1, adaptive_reserve=True, price_shock=sc.price_shock,
            volatility_multiplier=sc.volatility_multiplier,
            cycle_cost_multiplier=sc.cycle_cost_multiplier, progress=progress,
        )
        soc_now = float(closed.log["soc_mwh"].iloc[-1])
        if closed.metrics["solver_failures"]:
            warnings.append(
                f"{closed.metrics['solver_failures']} solver failure(s) during the "
                "backtest were handled by holding position.")
        if closed.metrics["reserve_violations"]:
            warnings.append(
                f"{closed.metrics['reserve_violations']} hour(s) discharged through "
                "the adaptive reserve floor - investigate before trusting the run.")
        if closed.metrics.get("hours_below_raised_reserve"):
            warnings.append(
                f"{closed.metrics['hours_below_raised_reserve']} hour(s) sat below the "
                "reserve because risk RAISED the floor above the current SOC; the "
                "optimiser refills rather than reporting infeasible.")
        if with_open_loop:
            open_res = run_backtest(
                df, forecaster, battery, risk, start_idx=bt_start, steps=steps,
                horizon=horizon, soc0_frac=(soc0 / battery.capacity_mwh),
                replan_every=horizon, adaptive_reserve=True,
                price_shock=sc.price_shock,
                volatility_multiplier=sc.volatility_multiplier,
                cycle_cost_multiplier=sc.cycle_cost_multiplier,
            )

    # ---- 2. the live decision at 'now' ---------------------------------
    lo = max(0, decision_idx - 720)
    window = df.iloc[lo:decision_idx + 1 + horizon]
    fc = forecaster.predict_window(window, decision_idx - lo, horizon=horizon)
    if not np.isfinite(fc.forecast).all():
        warnings.append("Forecast contained NaN; non-finite hours replaced by the "
                        "trailing realised mean.")
        fill = float(np.nanmean(df["price_per_mwh"].iloc[decision_idx - 168:decision_idx + 1]))
        fc.forecast = np.nan_to_num(fc.forecast, nan=fill, posinf=fill, neginf=fill)

    recent = df["price_per_mwh"].to_numpy(float)[max(0, decision_idx - VOL_WINDOW):decision_idx + 1]
    state = compute_risk_state(
        fc.error_scale, recent, battery, risk,
        volatility_multiplier=sc.volatility_multiplier,
        scenario_reserve_floor=risk.soc_reserve_frac,
    )
    risk_eff = apply_risk_state(risk, state)
    prices = adjust_prices(fc.forecast, fc.error_scale, risk_eff,
                           price_shock=sc.price_shock)

    try:
        dp = optimize_dispatch(
            prices.buy, prices.sell, battery, risk_eff, soc_now, ts=fc.ts,
            cycle_cost_multiplier=sc.cycle_cost_multiplier)
    except InfeasibleDispatch as exc:
        warnings.append(f"Optimiser infeasible ({exc}); retried without the reserve.")
        relaxed = RiskConfig(risk_aversion=risk_eff.risk_aversion,
                             soc_reserve_frac=0.0,
                             max_daily_cycles=risk_eff.max_daily_cycles,
                             enforce_reserve=False)
        dp = optimize_dispatch(prices.buy, prices.sell, battery, relaxed, soc_now,
                               ts=fc.ts,
                               cycle_cost_multiplier=sc.cycle_cost_multiplier)

    viol = check_feasible(dp.plan, battery, dp.reserve_mwh)
    if viol:
        warnings.append("Plan constraint check: " + "; ".join(viol))
    if check_no_simultaneous(dp.plan):
        warnings.append("Simultaneous charge+discharge detected in the plan.")

    plan = dp.plan.copy()
    plan["forecast_price"] = fc.forecast
    plan["sigma"] = fc.error_scale
    plan["actual_price"] = fc.actual if fc.actual is not None else np.nan
    plan["risk_level"] = state.level

    # ---- 3. benchmark + regret -----------------------------------------
    bench: dict = {}
    pf_plan = None
    if closed is not None:
        try:
            pf_plan = bm.perfect_foresight(
                df, battery, start_idx=bt_start, steps=steps,
                soc0_frac=(soc0 / battery.capacity_mwh),
                cycle_cost_multiplier=sc.cycle_cost_multiplier)
            bench = bm.compare(closed, pf_plan, open_res)
        except (InfeasibleDispatch, ValueError) as exc:
            warnings.append(f"Benchmark unavailable: {exc}")

    # ---- 4. forecast quality -------------------------------------------
    fmet = forecaster.metrics.to_dict() if forecaster.metrics else {}
    base_met = (forecaster.baseline_metrics.to_dict()
                if forecaster.baseline_metrics else {})
    forecast_mae = (closed.metrics["forecast_mae"] if closed is not None
                    else float(fmet.get("mae", np.nan)))
    forecast_rmse = (closed.metrics["forecast_rmse"] if closed is not None
                     else float(fmet.get("rmse", np.nan)))

    # ---- 5. operator plan ----------------------------------------------
    notes = risk_notes(prices, battery, risk_eff, horizon)
    op = build_operator_plan(
        battery=battery, plan=plan, action=dp.current_action, soc_mwh=soc_now,
        current_price=float(df["price_per_mwh"].iloc[decision_idx]),
        forecast=fc.forecast, sigma=fc.error_scale, risk_state=state,
        scenario_name=sc.name, scenario_desc=sc.description, risk_notes=notes,
        expected_profit=dp.expected_profit, horizon=horizon, warnings=warnings,
    )
    if use_llm:
        op = llm_polish(op)

    result = {
        "generated_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "now_ts": str(df["ts"].iloc[decision_idx]),
        "decision_idx": int(decision_idx),
        "scenario": sc.name,
        "scenario_description": sc.description,
        "policy": {"risk_aversion": sc.risk_aversion,
                   "scenario_reserve_frac": sc.soc_reserve_frac,
                   "horizon": horizon, "price_shock": sc.price_shock,
                   "volatility_multiplier": sc.volatility_multiplier,
                   "cycle_cost_multiplier": sc.cycle_cost_multiplier},
        "battery": battery.to_dict(),
        "history": df.iloc[max(0, decision_idx - 168):decision_idx + 1][
            ["ts", "price_per_mwh", "load_mw", "renewable_mw"]].reset_index(drop=True),
        "forecast": plan[["ts", "forecast_price", "sigma", "actual_price"]].copy(),
        "dispatch": plan,
        "current_action": dp.current_action,
        "current_price": float(df["price_per_mwh"].iloc[decision_idx]),
        "current_soc": soc_now,
        "current_soc_pct": 100.0 * soc_now / battery.capacity_mwh,
        "adaptive_reserve": state.reserve_frac,
        "adaptive_reserve_mwh": dp.reserve_mwh,
        "adaptive_reserve_pct": state.reserve_soc_pct,
        "risk_score": state.risk_score,
        "risk_level": state.level,
        "risk_state": state.to_dict(),
        "risk_notes": notes,
        "expected_profit": dp.expected_profit,
        "breakeven_spread": battery.breakeven_spread(float(np.min(fc.forecast))),
        "solve_time": dp.solve_seconds,
        "solver": dp.to_dict(),
        "forecast_model": forecaster.name,
        "forecast_metrics_holdout": fmet,
        "baseline_metrics_holdout": base_met,
        "forecast_mae": forecast_mae,
        "forecast_rmse": forecast_rmse,
        "backtest": closed.metrics if closed is not None else {},
        "backtest_log": closed.log if closed is not None else pd.DataFrame(),
        "risk_trace": closed.risk_trace if closed is not None else pd.DataFrame(),
        "open_loop": open_res.metrics if open_res is not None else {},
        "benchmark": bench,
        "perfect_foresight_plan": pf_plan.plan if pf_plan is not None else pd.DataFrame(),
        "net_profit": float(closed.metrics["net_profit"]) if closed is not None else np.nan,
        "degradation_cost": float(closed.metrics["degradation_cost"]) if closed is not None else np.nan,
        "throughput": float(closed.metrics["throughput_mwh"]) if closed is not None else np.nan,
        "perfect_foresight_profit": bench.get("perfect_foresight_profit", np.nan),
        "regret": bench.get("regret", np.nan),
        "regret_pct": bench.get("regret_pct", np.nan),
        "error_impact": bm.error_impact(closed.log) if closed is not None else pd.DataFrame(),
        "operator_plan": op,
        "warnings": warnings,
        "pipeline_seconds": time.perf_counter() - t0,
    }

    if log_to_db and closed is not None:
        result["run_id"] = storage.log_run(
            sc.name, horizon, sc.risk_aversion / 2.0,
            {**closed.metrics, **bench}, closed.log)

    if cache:
        _RESULT_CACHE[ck] = result
    return result


def clear_caches() -> None:
    _DATA_CACHE.clear()
    _MODEL_CACHE.clear()
    _RESULT_CACHE.clear()


if __name__ == "__main__":       # smoke run: python -m src.pipeline
    r = run_dispatch(horizon=24, scenario="normal", backtest_hours=48)
    print(f"now={r['now_ts']} action={r['current_action']} "
          f"soc={r['current_soc_pct']:.1f}% risk={r['risk_level']} "
          f"reserve={r['adaptive_reserve']:.0%}")
    print(f"net_profit={r['net_profit']:,.0f} pf={r['perfect_foresight_profit']:,.0f} "
          f"regret={r['regret']:,.0f} ({r['regret_pct']:.1f}%) "
          f"mae={r['forecast_mae']:.2f} solve={r['solve_time']*1000:.0f}ms "
          f"total={r['pipeline_seconds']:.1f}s")
