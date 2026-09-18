"""Risk layer: turns forecast uncertainty into optimiser-visible conservatism.

Three *real* risk mechanisms, all of which change the LP itself -- none of this
lives only in the narrative:

1. **Conservative price adjustment (price-space robustness).**
   The optimiser never sees the raw point forecast. It sees a downside-shifted
   price pair derived from the forecaster's per-hour error scale:

       p_sell[t] = f[t] - lambda * sigma[t]     (revenue assumed worse)
       p_buy[t]  = f[t] + lambda * sigma[t]     (cost assumed worse)

   This is the LP dual of a box-uncertainty robust programme: because revenue
   is linear and increasing in ``discharge`` and cost linear and increasing in
   ``charge``, the worst case over the box ``f +/- lambda*sigma`` is attained at
   exactly these corners. So maximising with (p_sell, p_buy) *is* maximising the
   worst-case profit over that uncertainty set. Raising ``lambda`` widens the
   apparent spread requirement and suppresses marginal cycles.

2. **SOC reserve.** Raises the effective floor on stored energy so the battery
   always retains energy to sell into an unforecast scarcity spike.

3. **Throughput / cycle cap.** Limits equivalent full cycles per rolling day,
   bounding degradation exposure on any single bad day.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import BatteryConfig, RiskConfig


@dataclass
class RiskAdjustedPrices:
    """Prices the optimiser is allowed to see, plus provenance for the UI."""

    buy: np.ndarray            # price charged when buying (>= forecast)
    sell: np.ndarray           # price earned when selling (<= forecast)
    forecast: np.ndarray       # original point forecast
    sigma: np.ndarray          # per-step error scale used
    risk_aversion: float

    @property
    def penalty(self) -> np.ndarray:
        """Per-hour one-sided conservatism applied, in currency/MWh."""
        return self.risk_aversion * self.sigma

    @property
    def mean_penalty(self) -> float:
        return float(np.mean(self.penalty))


def adjust_prices(
    forecast: np.ndarray,
    error_scale: np.ndarray,
    risk: RiskConfig,
    price_shock: float = 0.0,
) -> RiskAdjustedPrices:
    """Build the conservative buy/sell price vectors for the optimiser.

    ``price_shock`` is a scenario stress (e.g. +0.35 = assume a 35% richer
    market) applied to the point forecast *before* the risk shift.
    """
    f = np.asarray(forecast, float).copy()
    s = np.abs(np.asarray(error_scale, float))
    if f.shape != s.shape:
        raise ValueError("forecast and error_scale must have the same shape")
    if price_shock:
        # Scale magnitude while preserving sign -- negative prices stay negative.
        f = f * (1.0 + price_shock)

    lam = max(float(risk.risk_aversion), 0.0)
    return RiskAdjustedPrices(
        buy=f + lam * s,
        sell=f - lam * s,
        forecast=f,
        sigma=s,
        risk_aversion=lam,
    )


def reserve_floor(battery: BatteryConfig, risk: RiskConfig) -> float:
    """Effective lower bound on stored energy (MWh), including the reserve.

    Clipped so the reserve can never exceed the usable band, which would make
    the LP infeasible for a battery that starts below the floor.
    """
    if not risk.enforce_reserve or risk.soc_reserve_frac <= 0:
        return battery.e_min
    extra = risk.soc_reserve_frac * battery.capacity_mwh
    # Leave at least 20% of the usable band free to trade in.
    max_extra = 0.80 * battery.usable_mwh
    return battery.e_min + min(extra, max_extra)


def throughput_cap(battery: BatteryConfig, risk: RiskConfig, hours: int) -> float:
    """Max grid-side throughput (MWh) allowed over a window of ``hours``.

    One "equivalent full cycle" is charging *and* discharging the usable band,
    i.e. 2 x usable_mwh of grid-side throughput.
    """
    cycles = max(float(risk.max_daily_cycles), 0.0)
    return cycles * 2.0 * battery.usable_mwh * (hours / 24.0)


def risk_notes(
    prices: RiskAdjustedPrices,
    battery: BatteryConfig,
    risk: RiskConfig,
    horizon: int,
) -> list[str]:
    """Deterministic, number-backed risk statements. No LLM involved."""
    notes: list[str] = []
    spread = float(np.max(prices.forecast) - np.min(prices.forecast))
    breakeven = battery.breakeven_spread(float(np.min(prices.forecast)))
    notes.append(
        f"Forecast spread over the horizon is {spread:,.1f}/MWh against a "
        f"round-trip breakeven of {breakeven:,.1f}/MWh "
        f"({'adequate' if spread > breakeven else 'THIN - cycles may not pay'})."
    )
    notes.append(
        f"Forecast uncertainty averages +/-{float(np.mean(prices.sigma)):,.1f}/MWh "
        f"(1 sigma); risk aversion {prices.risk_aversion:.2f} shifts optimiser "
        f"prices by {prices.mean_penalty:,.1f}/MWh one-sided."
    )
    if prices.mean_penalty > 0.5 * spread:
        notes.append(
            "Risk penalty exceeds half the forecast spread - the optimiser will "
            "trade very defensively and may hold idle."
        )
    floor = reserve_floor(battery, risk)
    if floor > battery.e_min:
        notes.append(
            f"SOC reserve holds {floor - battery.e_min:,.1f} MWh above the "
            f"{battery.soc_min:.0%} floor as spike insurance, reducing tradable "
            f"energy to {battery.e_max - floor:,.1f} MWh."
        )
    cap = throughput_cap(battery, risk, horizon)
    notes.append(
        f"Throughput capped at {cap:,.1f} MWh over {horizon}h "
        f"({risk.max_daily_cycles:.2f} equivalent full cycles/day)."
    )
    if float(np.min(prices.forecast)) < 0:
        notes.append(
            f"Negative prices forecast (min {float(np.min(prices.forecast)):,.1f}"
            "/MWh): charging is paid for. Confirm the offtake contract permits "
            "consuming at negative prices."
        )
    return notes


# ---------------------------------------------------------------------------
# ADAPTIVE RESERVE ENGINE
# ---------------------------------------------------------------------------
# The reserve is NOT a hardcoded number. It is computed from two observable
# quantities and mapped linearly into a configurable band:
#
#   u = mean(sigma) / price_scale            normalised forecast uncertainty
#   v = std(recent prices) / price_scale     normalised market volatility
#   risk_score = clip(w_u*u + w_v*v, 0, 1) * scenario_multiplier
#   reserve_frac = base + (max - base) * risk_score        (extra above soc_min)
#
# ``reserve_frac`` is written into RiskConfig.soc_reserve_frac, which
# ``reserve_floor`` turns into the LP's lower bound on e[t]. Risk therefore
# changes the feasible set of the optimiser, not just the dashboard text.

RESERVE_BASE_FRAC = 0.05     # extra above soc_min at zero risk
RESERVE_MAX_FRAC = 0.25      # extra above soc_min at maximum risk
UNCERTAINTY_WEIGHT = 0.6
VOLATILITY_WEIGHT = 0.4
PRICE_SCALE_FLOOR = 10.0     # currency/MWh, keeps the ratio finite near zero


@dataclass
class RiskState:
    """Everything the risk engine computed, with its inputs kept for the UI."""

    risk_score: float               # 0..1
    uncertainty_norm: float
    volatility_norm: float
    mean_sigma: float               # currency/MWh, 1-sigma forecast error
    price_volatility: float         # currency/MWh, recent realised std
    price_scale: float
    reserve_frac: float             # extra fraction of capacity above soc_min
    reserve_soc_pct: float          # resulting absolute SOC floor, % of capacity
    reserve_mwh: float
    risk_aversion: float
    level: str                      # LOW / MODERATE / ELEVATED / HIGH
    drivers: list[str]

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        d["drivers"] = list(self.drivers)
        return d


def _level(score: float) -> str:
    if score < 0.25:
        return "LOW"
    if score < 0.50:
        return "MODERATE"
    if score < 0.75:
        return "ELEVATED"
    return "HIGH"


def compute_risk_state(
    error_scale,
    recent_prices,
    battery: BatteryConfig,
    risk: RiskConfig,
    *,
    volatility_multiplier: float = 1.0,
    scenario_reserve_floor: float = 0.0,
    base_frac: float = RESERVE_BASE_FRAC,
    max_frac: float = RESERVE_MAX_FRAC,
) -> RiskState:
    """Derive risk score and adaptive SOC reserve from measured quantities.

    ``error_scale`` comes from the forecaster's validation residuals (never
    invented); ``recent_prices`` is the trailing observed price window.
    ``scenario_reserve_floor`` lets a scenario demand *at least* that much
    reserve, but the computed value wins when it is stricter.
    """
    sigma = np.abs(np.asarray(error_scale, float))
    sigma = sigma[np.isfinite(sigma)]
    prices = np.asarray(recent_prices, float)
    prices = prices[np.isfinite(prices)]

    mean_sigma = float(np.mean(sigma)) if sigma.size else 0.0
    vol = float(np.std(prices)) * float(volatility_multiplier) if prices.size else 0.0
    # Scale by the market's own level so the score is unit-free and portable.
    price_scale = max(float(np.mean(np.abs(prices))) if prices.size else 0.0,
                      PRICE_SCALE_FLOOR)

    u = mean_sigma / price_scale
    v = vol / price_scale
    # u ~ 0.35 and v ~ 0.75 are already stressed markets -> normalise to ~1.
    u_n = float(np.clip(u / 0.35, 0.0, 1.0))
    v_n = float(np.clip(v / 0.75, 0.0, 1.0))

    score = UNCERTAINTY_WEIGHT * u_n + VOLATILITY_WEIGHT * v_n
    # Risk aversion tilts the posture: lambda=0.5 is neutral by construction.
    score *= float(np.clip(0.5 + risk.risk_aversion / 2.0, 0.4, 1.6))
    score = float(np.clip(score, 0.0, 1.0))

    frac = base_frac + (max_frac - base_frac) * score
    frac = max(frac, float(scenario_reserve_floor))
    if not risk.enforce_reserve:
        frac = 0.0

    probe = RiskConfig(
        risk_aversion=risk.risk_aversion,
        soc_reserve_frac=frac,
        max_daily_cycles=risk.max_daily_cycles,
        enforce_reserve=risk.enforce_reserve,
    ).validate()
    floor_mwh = reserve_floor(battery, probe)

    drivers = [
        f"Forecast uncertainty {mean_sigma:,.1f}/MWh vs price scale "
        f"{price_scale:,.1f}/MWh -> normalised {u_n:.2f}",
        f"Recent price volatility {vol:,.1f}/MWh -> normalised {v_n:.2f}",
        f"Risk aversion {risk.risk_aversion:.2f} scales the blended score",
    ]
    if scenario_reserve_floor > 0:
        drivers.append(
            f"Scenario imposes a reserve floor of {scenario_reserve_floor:.0%} of capacity"
        )

    return RiskState(
        risk_score=score,
        uncertainty_norm=u_n,
        volatility_norm=v_n,
        mean_sigma=mean_sigma,
        price_volatility=vol,
        price_scale=price_scale,
        reserve_frac=frac,
        reserve_soc_pct=100.0 * floor_mwh / battery.capacity_mwh,
        reserve_mwh=floor_mwh,
        risk_aversion=float(risk.risk_aversion),
        level=_level(score),
        drivers=drivers,
    )


def apply_risk_state(risk: RiskConfig, state: RiskState) -> RiskConfig:
    """Return a copy of ``risk`` whose reserve is the adaptive one."""
    return RiskConfig(
        risk_aversion=risk.risk_aversion,
        soc_reserve_frac=state.reserve_frac,
        max_daily_cycles=risk.max_daily_cycles,
        enforce_reserve=risk.enforce_reserve,
    ).validate()
