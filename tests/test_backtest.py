"""Closed-loop backtest + perfect-foresight benchmark + regret."""
from __future__ import annotations

import numpy as np
import pytest

from src.backtest import run_backtest
from src.benchmark import compare, error_impact, perfect_foresight, regret
from src.config import RiskConfig

STEPS = 36
HORIZON = 24


def _start(df):
    return len(df) - HORIZON - STEPS - 2


def test_executes_one_interval_per_step_with_actual_prices(market, oracle, battery, risk):
    i = _start(market)
    res = run_backtest(market, oracle, battery, risk, start_idx=i, steps=STEPS,
                       horizon=HORIZON, soc0_frac=0.5, adaptive_reserve=False)
    log = res.log
    assert len(log) == STEPS
    # One row per consecutive hour, each settled at that hour's ACTUAL price.
    expected_ts = market["ts"].iloc[i + 1:i + 1 + STEPS].reset_index(drop=True)
    assert (log["ts"].reset_index(drop=True) == expected_ts).all()
    actual = market["price_per_mwh"].iloc[i + 1:i + 1 + STEPS].to_numpy()
    assert np.allclose(log["price"].to_numpy(), actual)


def test_soc_updates_with_efficiency_and_stays_in_bounds(market, oracle, battery, risk):
    res = run_backtest(market, oracle, battery, risk, start_idx=_start(market),
                       steps=STEPS, horizon=HORIZON, soc0_frac=0.5,
                       adaptive_reserve=False)
    log = res.log
    expect = (log["soc_start_mwh"] + log["charge_mw"] * battery.eta_charge
              - log["discharge_mw"] / battery.eta_discharge)
    assert np.allclose(log["soc_mwh"], expect, atol=1e-6)
    assert log["soc_mwh"].max() <= battery.e_max + 1e-6
    assert log["soc_mwh"].min() >= 0.0
    assert not ((log["charge_mw"] > 1e-4) & (log["discharge_mw"] > 1e-4)).any()


def test_realised_profit_matches_the_log(market, oracle, battery, risk):
    res = run_backtest(market, oracle, battery, risk, start_idx=_start(market),
                       steps=STEPS, horizon=HORIZON, soc0_frac=0.5,
                       adaptive_reserve=False)
    m, log = res.metrics, res.log
    assert m["net_profit"] == pytest.approx(
        m["revenue"] - m["charging_cost"] - m["degradation_cost"])
    assert m["revenue"] == pytest.approx((log["discharge_mw"] * log["price"]).sum())
    assert m["degradation_cost"] == pytest.approx(
        battery.cycle_cost_per_mwh * (log["charge_mw"] + log["discharge_mw"]).sum())
    assert m["charge_events"] + m["discharge_events"] + m["wait_events"] == m["steps"]


def test_metrics_cover_the_required_kpis(market, oracle, battery, risk):
    m = run_backtest(market, oracle, battery, risk, start_idx=_start(market),
                     steps=STEPS, horizon=HORIZON, soc0_frac=0.5).metrics
    for k in ("revenue", "charging_cost", "degradation_cost", "net_profit",
              "throughput_mwh", "equivalent_cycles", "charge_events",
              "discharge_events", "wait_events", "avg_soc_pct", "min_soc_pct",
              "max_soc_pct", "reserve_violations", "forecast_mae", "forecast_rmse",
              "runtime_seconds", "avg_solve_seconds"):
        assert k in m, f"missing KPI {k}"


def test_perfect_foresight_uses_only_future_actual_prices(market, battery):
    i = _start(market)
    pf = perfect_foresight(market, battery, start_idx=i, steps=STEPS, soc0_frac=0.5)
    actual = market["price_per_mwh"].iloc[i + 1:i + 1 + STEPS].to_numpy()
    assert np.allclose(pf.plan["sell_price"].to_numpy(), actual)
    assert np.allclose(pf.plan["buy_price"].to_numpy(), actual)
    assert len(pf.plan) == STEPS


def test_perfect_foresight_is_an_upper_bound(market, oracle, battery, risk):
    """Even an ORACLE closed loop cannot beat full-window perfect foresight."""
    i = _start(market)
    res = run_backtest(market, oracle, battery,
                       RiskConfig(risk_aversion=0.0, soc_reserve_frac=0.0,
                                  max_daily_cycles=999.0, enforce_reserve=False),
                       start_idx=i, steps=STEPS, horizon=HORIZON, soc0_frac=0.5,
                       adaptive_reserve=False)
    pf = perfect_foresight(market, battery, start_idx=i, steps=STEPS, soc0_frac=0.5)
    pf_profit = float(pf.plan["expected_profit"].sum())
    assert res.metrics["net_profit"] <= pf_profit + 1e-6
    # ... and the machinery should capture most of it when the forecast is perfect.
    assert res.metrics["net_profit"] >= 0.7 * pf_profit


def test_regret_arithmetic():
    out = regret(80.0, 100.0)
    assert out["regret"] == pytest.approx(20.0)
    assert out["regret_pct"] == pytest.approx(20.0)
    assert out["capture_rate_pct"] == pytest.approx(80.0)
    neg = regret(-50.0, -100.0)          # regret stays well-defined if PF loses money
    assert neg["regret"] == pytest.approx(-50.0)


def test_open_loop_is_not_better_than_closed_loop_on_average(market, oracle, battery, risk):
    i = _start(market)
    kw = dict(start_idx=i, steps=STEPS, horizon=HORIZON, soc0_frac=0.5,
              adaptive_reserve=False)
    closed = run_backtest(market, oracle, battery, risk, replan_every=1, **kw)
    open_ = run_backtest(market, oracle, battery, risk, replan_every=HORIZON, **kw)
    assert open_.metrics["mode"] == "open_loop"
    assert closed.metrics["mode"] == "closed_loop"
    pf = perfect_foresight(market, battery, start_idx=i, steps=STEPS, soc0_frac=0.5)
    cmp_ = compare(closed, pf, open_)
    assert set(("regret", "regret_pct", "open_loop_profit")) <= set(cmp_)
    assert cmp_["closed_loop_profit"] >= cmp_["open_loop_profit"] - 1e-6


def test_error_impact_table(market, oracle, battery, risk):
    res = run_backtest(market, oracle, battery, risk, start_idx=_start(market),
                       steps=STEPS, horizon=HORIZON, soc0_frac=0.5)
    ei = error_impact(res.log)
    assert ei.empty or {"hours", "mean_abs_error", "profit"} <= set(ei.columns)


def test_backtest_refuses_impossible_windows(market, oracle, battery, risk):
    with pytest.raises(ValueError):
        run_backtest(market, oracle, battery, risk, start_idx=10, steps=5)
