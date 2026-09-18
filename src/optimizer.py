"""Battery dispatch optimiser: a pure linear programme solved with PuLP/CBC.

FORMULATION (per interval t = 0..T-1, hourly so 1 MW == 1 MWh)
--------------------------------------------------------------
decision vars   c[t] >= 0   grid-side charge power      (MW)
                d[t] >= 0   grid-side discharge power   (MW)
                e[t]        stored energy at end of t   (MWh)

dynamics        e[t] = e[t-1] + c[t]*eta_c - d[t]/eta_d
bounds          floor <= e[t] <= e_max            (floor = adaptive reserve)
                0 <= c[t] <= P,  0 <= d[t] <= P
throughput      sum_t (c[t] + d[t]) <= cap        (degradation exposure cap)

objective       max  sum_t [ sell[t]*d[t] - buy[t]*c[t] - cc*(c[t]+d[t]) ]
                     + tv * e[T-1]                (terminal energy value)

NOTES
-----
* No binaries. Simultaneous charge+discharge is unprofitable whenever
  eta_c*eta_d < 1 or cc > 0 (every round trip loses energy *and* pays
  degradation twice), so exclusivity is implied rather than imposed. A cheap
  post-solve assertion verifies it instead of paying MIP cost. See
  ``check_no_simultaneous``.
* ``cc`` is the throughput-based marginal degradation cost, charged on BOTH
  legs, which is what makes the optimiser refuse thin spreads.
* ``tv`` (terminal value) prices the energy left in the battery at the end of
  the horizon so a finite horizon does not force a myopic end-of-window dump.
  It is a *conservative* price (40th percentile of the horizon's sell prices),
  never a future actual price.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import pulp

from .config import BatteryConfig, RiskConfig
from .risk import RiskAdjustedPrices, reserve_floor, throughput_cap

TOL = 1e-6
CHARGE, DISCHARGE, WAIT = "CHARGE", "DISCHARGE", "WAIT"


class InfeasibleDispatch(RuntimeError):
    """Raised when the LP has no feasible dispatch at all."""


@dataclass
class DispatchPlan:
    """Optimiser output: a per-interval plan plus solver diagnostics."""

    plan: pd.DataFrame            # ts, charge_mw, discharge_mw, soc_mwh, action, ...
    status: str
    objective: float
    solve_seconds: float
    n_vars: int
    n_constraints: int
    reserve_mwh: float
    horizon: int
    meta: dict = field(default_factory=dict)

    @property
    def current_action(self) -> str:
        return str(self.plan["action"].iloc[0]) if len(self.plan) else WAIT

    @property
    def first_charge(self) -> float:
        return float(self.plan["charge_mw"].iloc[0]) if len(self.plan) else 0.0

    @property
    def first_discharge(self) -> float:
        return float(self.plan["discharge_mw"].iloc[0]) if len(self.plan) else 0.0

    @property
    def expected_profit(self) -> float:
        return float(self.plan["expected_profit"].sum()) if len(self.plan) else 0.0

    @property
    def throughput(self) -> float:
        if not len(self.plan):
            return 0.0
        return float((self.plan["charge_mw"] + self.plan["discharge_mw"]).sum())

    def to_dict(self) -> dict:
        return {
            "status": self.status, "objective": self.objective,
            "solve_seconds": self.solve_seconds, "n_vars": self.n_vars,
            "n_constraints": self.n_constraints, "reserve_mwh": self.reserve_mwh,
            "horizon": self.horizon, "current_action": self.current_action,
            "expected_profit": self.expected_profit, "throughput": self.throughput,
        }


def classify(charge: float, discharge: float, tol: float = 1e-3) -> str:
    """CHARGE / DISCHARGE / WAIT. WAIT is a first-class outcome."""
    if charge > tol and charge >= discharge:
        return CHARGE
    if discharge > tol:
        return DISCHARGE
    return WAIT


def optimize_dispatch(
    buy_prices,
    sell_prices,
    battery: BatteryConfig,
    risk: RiskConfig,
    soc_mwh: float,
    *,
    ts=None,
    cycle_cost_multiplier: float = 1.0,
    final_soc_mwh: float | None = None,
    terminal_value: bool = True,
    msg: bool = False,
) -> DispatchPlan:
    """Solve the dispatch LP over ``len(buy_prices)`` intervals.

    ``buy_prices``/``sell_prices`` are the RISK-ADJUSTED prices from
    :mod:`src.risk` (buy >= forecast, sell <= forecast). Passing the raw
    forecast for both gives the risk-neutral problem; passing actual future
    prices gives the perfect-foresight benchmark (benchmark use only).
    """
    battery.validate()
    risk.validate()
    buy = np.asarray(buy_prices, float)
    sell = np.asarray(sell_prices, float)
    if buy.shape != sell.shape:
        raise ValueError("buy_prices and sell_prices must have the same length")
    T = len(buy)
    if T == 0:
        raise ValueError("empty horizon")
    if not np.isfinite(buy).all() or not np.isfinite(sell).all():
        raise ValueError("prices contain NaN/inf")

    cc = max(battery.cycle_cost_per_mwh * float(cycle_cost_multiplier), 0.0)
    floor = reserve_floor(battery, risk)
    e0 = float(np.clip(soc_mwh, 0.0, battery.capacity_mwh))
    # A battery that starts below the reserve must be allowed to recover: relax
    # the floor to where it actually is rather than declaring infeasibility.
    eff_floor = min(floor, e0)
    cap = throughput_cap(battery, risk, T)

    prob = pulp.LpProblem("gridflex_dispatch", pulp.LpMaximize)
    c = [pulp.LpVariable(f"c_{t}", 0, battery.power_mw) for t in range(T)]
    d = [pulp.LpVariable(f"d_{t}", 0, battery.power_mw) for t in range(T)]
    e = [pulp.LpVariable(f"e_{t}", eff_floor, battery.e_max) for t in range(T)]

    eta_c, eta_d = battery.eta_charge, battery.eta_discharge
    prev = e0
    for t in range(T):
        prob += e[t] == prev + c[t] * eta_c - d[t] * (1.0 / eta_d), f"soc_{t}"
        prev = e[t]
    prob += pulp.lpSum(c) + pulp.lpSum(d) <= cap, "throughput_cap"
    if final_soc_mwh is not None:
        prob += e[T - 1] >= float(np.clip(final_soc_mwh, eff_floor, battery.e_max)), "final_soc"

    tv = 0.0
    if terminal_value and final_soc_mwh is None:
        # Conservative marginal value of energy carried past the horizon.
        tv = float(np.percentile(sell, 40)) * eta_d
        tv = max(tv, 0.0)

    prob += (
        pulp.lpSum(sell[t] * d[t] for t in range(T))
        - pulp.lpSum(buy[t] * c[t] for t in range(T))
        - pulp.lpSum(cc * (c[t] + d[t]) for t in range(T))
        + tv * e[T - 1]
    )

    t0 = time.perf_counter()
    prob.solve(pulp.PULP_CBC_CMD(msg=msg))
    solve_s = time.perf_counter() - t0
    status = pulp.LpStatus[prob.status]
    if status != "Optimal":
        raise InfeasibleDispatch(f"solver status={status}")

    cv = np.array([max(v.value() or 0.0, 0.0) for v in c])
    dv = np.array([max(v.value() or 0.0, 0.0) for v in d])
    ev = np.array([v.value() or 0.0 for v in e])
    cv[cv < TOL] = 0.0
    dv[dv < TOL] = 0.0

    deg = cc * (cv + dv)
    profit = sell * dv - buy * cv - deg
    plan = pd.DataFrame(
        {
            "ts": ts if ts is not None else np.arange(T),
            "buy_price": buy,
            "sell_price": sell,
            "charge_mw": cv,
            "discharge_mw": dv,
            "soc_mwh": ev,
            "soc_pct": 100.0 * ev / battery.capacity_mwh,
            "degradation_cost": deg,
            "expected_profit": profit,
            "action": [classify(a, b) for a, b in zip(cv, dv)],
        }
    )
    return DispatchPlan(
        plan=plan,
        status=status,
        objective=float(pulp.value(prob.objective) or 0.0),
        solve_seconds=solve_s,
        n_vars=len(prob.variables()),
        n_constraints=len(prob.constraints),
        reserve_mwh=floor,
        horizon=T,
        meta={"cycle_cost": cc, "terminal_value": tv, "throughput_cap": cap,
              "effective_floor_mwh": eff_floor, "soc_start_mwh": e0},
    )


def optimize_from_risk(
    prices: RiskAdjustedPrices,
    battery: BatteryConfig,
    risk: RiskConfig,
    soc_mwh: float,
    **kw,
) -> DispatchPlan:
    """Convenience wrapper: solve directly from a :class:`RiskAdjustedPrices`."""
    return optimize_dispatch(prices.buy, prices.sell, battery, risk, soc_mwh, **kw)


# -- post-solve verification ------------------------------------------------
def check_no_simultaneous(plan: pd.DataFrame, tol: float = 1e-4) -> int:
    """Count intervals charging and discharging at once (expected: 0)."""
    both = (plan["charge_mw"] > tol) & (plan["discharge_mw"] > tol)
    return int(both.sum())


def check_feasible(plan: pd.DataFrame, battery: BatteryConfig, floor: float,
                   tol: float = 1e-4) -> list[str]:
    """Return a list of physical-constraint violations (empty == valid plan)."""
    v: list[str] = []
    if (plan["charge_mw"] > battery.power_mw + tol).any():
        v.append("charge exceeds power limit")
    if (plan["discharge_mw"] > battery.power_mw + tol).any():
        v.append("discharge exceeds power limit")
    if (plan["soc_mwh"] > battery.e_max + tol).any():
        v.append("SOC above soc_max")
    if (plan["soc_mwh"] > battery.capacity_mwh + tol).any():
        v.append("SOC above capacity")
    if (plan["soc_mwh"] < min(floor, plan["soc_mwh"].iloc[0]) - tol).any():
        v.append("SOC below reserve floor")
    if check_no_simultaneous(plan):
        v.append("simultaneous charge and discharge")
    return v
