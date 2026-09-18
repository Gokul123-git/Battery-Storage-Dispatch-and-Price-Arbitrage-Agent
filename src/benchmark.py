"""Perfect-foresight benchmark and regret analysis.

The benchmark solves ONE LP over the whole evaluation window using the ACTUAL
realised prices. It is an upper bound on what any causal controller could have
earned with the same battery, and it is used for scoring only -- the live
controller (``src.pipeline`` / ``src.backtest``) never calls it.

    regret      = perfect_foresight_profit - closed_loop_profit
    regret_pct  = regret / |perfect_foresight_profit| * 100

Absolute profit mostly measures how volatile the market happened to be.
Regret measures how much value the forecast + decision stack gave away.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import BatteryConfig, RiskConfig
from .optimizer import DispatchPlan, optimize_dispatch


def perfect_foresight(
    df: pd.DataFrame,
    battery: BatteryConfig,
    *,
    start_idx: int,
    steps: int,
    soc0_frac: float | None = None,
    cycle_cost_multiplier: float = 1.0,
) -> DispatchPlan:
    """Optimal dispatch over hours ``start_idx+1 .. start_idx+steps``.

    Risk-neutral by construction (there is nothing to be uncertain about) but
    it pays the SAME efficiency losses and the SAME throughput degradation
    cost, so the comparison isolates information, not physics.
    """
    lo, hi = start_idx + 1, start_idx + 1 + steps
    if hi > len(df):
        raise ValueError("benchmark window extends beyond the dataset")
    prices = df["price_per_mwh"].to_numpy(float)[lo:hi]
    soc0 = battery.capacity_mwh * (soc0_frac if soc0_frac is not None
                                   else (battery.soc_min + battery.soc_max) / 2)
    risk = RiskConfig(risk_aversion=0.0, soc_reserve_frac=0.0,
                      max_daily_cycles=999.0, enforce_reserve=False)
    return optimize_dispatch(
        prices, prices, battery, risk, soc0,
        ts=df["ts"].iloc[lo:hi].reset_index(drop=True),
        cycle_cost_multiplier=cycle_cost_multiplier,
    )


def regret(closed_loop_profit: float, perfect_profit: float) -> dict:
    """Absolute and percentage value lost to imperfect information."""
    r = float(perfect_profit) - float(closed_loop_profit)
    denom = abs(float(perfect_profit))
    return {
        "closed_loop_profit": float(closed_loop_profit),
        "perfect_foresight_profit": float(perfect_profit),
        "regret": r,
        "regret_pct": (r / denom * 100.0) if denom > 1e-9 else float("nan"),
        "capture_rate_pct": (float(closed_loop_profit) / denom * 100.0)
        if denom > 1e-9 else float("nan"),
    }


def compare(closed, perfect_plan: DispatchPlan, open_loop=None) -> dict:
    """Bundle closed-loop / open-loop / perfect-foresight into one report."""
    pf_profit = float(perfect_plan.plan["expected_profit"].sum())
    out = regret(closed.metrics["net_profit"], pf_profit)
    out["perfect_foresight_throughput"] = perfect_plan.throughput
    if open_loop is not None:
        ol = float(open_loop.metrics["net_profit"])
        out["open_loop_profit"] = ol
        out["closed_vs_open"] = out["closed_loop_profit"] - ol
        out["open_loop_regret"] = pf_profit - ol
        out["open_loop_regret_pct"] = (
            (pf_profit - ol) / abs(pf_profit) * 100.0 if abs(pf_profit) > 1e-9
            else float("nan")
        )
        out["feedback_value_pct"] = (
            (out["closed_loop_profit"] - ol) / abs(pf_profit) * 100.0
            if abs(pf_profit) > 1e-9 else float("nan")
        )
    return out


def error_impact(log: pd.DataFrame, bins: int = 4) -> pd.DataFrame:
    """Where forecast error actually cost money.

    Buckets executed hours by absolute forecast error and reports realised
    profit per bucket. This is the evidence for 'error near a discharge
    opportunity is expensive; error in a flat hour is not'.
    """
    if log.empty:
        return pd.DataFrame()
    d = log.copy()
    d["abs_error"] = d["forecast_error"].abs()
    d["traded_mwh"] = d["charge_mw"] + d["discharge_mw"]
    try:
        d["bucket"] = pd.qcut(d["abs_error"], bins, duplicates="drop")
    except ValueError:
        return pd.DataFrame()
    g = (d.groupby("bucket", observed=True)
         .agg(hours=("profit", "size"),
              mean_abs_error=("abs_error", "mean"),
              traded_mwh=("traded_mwh", "sum"),
              profit=("profit", "sum"),
              profit_per_mwh=("profit", "sum"))
         .reset_index())
    g["profit_per_mwh"] = np.where(g["traded_mwh"] > 1e-6,
                                   g["profit"] / g["traded_mwh"], np.nan)
    g["bucket"] = g["bucket"].astype(str)
    return g
