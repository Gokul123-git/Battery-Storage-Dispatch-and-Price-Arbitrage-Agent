"""Central configuration: battery spec, risk spec, paths, defaults."""
from __future__ import annotations

from dataclasses import dataclass, asdict, field, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
LOG_DIR = ROOT / "logs"
for _d in (DATA_DIR, MODEL_DIR, LOG_DIR):
    _d.mkdir(exist_ok=True)

MARKET_PARQUET = DATA_DIR / "market_prices.parquet"
RUNS_DB = LOG_DIR / "runs.sqlite"

SEED = 42


class ConfigError(ValueError):
    """Raised when a configuration is physically/economically invalid."""


@dataclass
class BatteryConfig:
    """Physical + economic specification of the storage asset.

    Energy units are MWh, power units MW. With hourly intervals 1 MW of
    sustained power moves exactly 1 MWh, so the LP treats charge/discharge
    decision variables as MW == MWh per interval.
    """

    capacity_mwh: float = 100.0
    power_mw: float = 25.0
    round_trip_efficiency: float = 0.90
    soc_min: float = 0.10          # fraction of capacity
    soc_max: float = 0.90          # fraction of capacity
    cycle_cost_per_mwh: float = 5.0
    calendar_life_yr: float = 15.0

    def validate(self) -> "BatteryConfig":
        if self.capacity_mwh <= 0:
            raise ConfigError("capacity_mwh must be > 0")
        if self.power_mw <= 0:
            raise ConfigError("power_mw must be > 0")
        if not 0 < self.round_trip_efficiency <= 1:
            raise ConfigError("round_trip_efficiency must be in (0, 1]")
        if not 0 <= self.soc_min < self.soc_max <= 1:
            raise ConfigError("require 0 <= soc_min < soc_max <= 1")
        if self.cycle_cost_per_mwh < 0:
            raise ConfigError("cycle_cost_per_mwh must be >= 0")
        if self.calendar_life_yr <= 0:
            raise ConfigError("calendar_life_yr must be > 0")
        return self

    # --- derived quantities -------------------------------------------------
    @property
    def eta_charge(self) -> float:
        """One-way charging efficiency (sqrt split of round-trip)."""
        return self.round_trip_efficiency ** 0.5

    @property
    def eta_discharge(self) -> float:
        """One-way discharging efficiency (sqrt split of round-trip)."""
        return self.round_trip_efficiency ** 0.5

    @property
    def e_min(self) -> float:
        return self.soc_min * self.capacity_mwh

    @property
    def e_max(self) -> float:
        return self.soc_max * self.capacity_mwh

    @property
    def usable_mwh(self) -> float:
        return self.e_max - self.e_min

    def breakeven_sell_price(self, buy_price: float) -> float:
        """Minimum sell price that makes a charge/discharge pair profitable.

        Buy q MWh from the grid at ``buy_price`` -> store q*eta_c -> deliver
        q*eta_c*eta_d = q*rt MWh back to the grid. Degradation is charged on
        grid-side throughput of *both* legs, so:

            q*rt*(p_sell - cc) >= q*(buy_price + cc)
            p_sell >= (buy_price + cc)/rt + cc
        """
        rt = max(self.round_trip_efficiency, 1e-9)
        cc = self.cycle_cost_per_mwh
        return (buy_price + cc) / rt + cc

    def breakeven_spread(self, buy_price: float = 50.0) -> float:
        """Spread (currency/MWh) required at ``buy_price`` to break even."""
        return self.breakeven_sell_price(buy_price) - buy_price

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RiskConfig:
    """Risk posture applied *inside* the optimisation, not in narrative only.

    risk_aversion (lambda) scales a conservative price adjustment derived from
    the forecaster's per-hour error scale; soc_reserve_frac forces extra stored
    energy to be held back as a hedge against price spikes.
    max_daily_throughput_cycles caps equivalent full cycles per rolling day.
    """

    risk_aversion: float = 0.5          # 0 = risk-neutral
    soc_reserve_frac: float = 0.0       # extra fraction of capacity held above e_min
    max_daily_cycles: float = 2.0       # equivalent full cycles per 24h window
    enforce_reserve: bool = True

    def validate(self) -> "RiskConfig":
        if self.risk_aversion < 0:
            raise ConfigError("risk_aversion must be >= 0")
        if not 0 <= self.soc_reserve_frac < 1:
            raise ConfigError("soc_reserve_frac must be in [0, 1)")
        if self.max_daily_cycles <= 0:
            raise ConfigError("max_daily_cycles must be > 0")
        return self

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ScenarioConfig:
    """A named operating scenario the agent is allowed to select/tune."""

    name: str = "normal"
    description: str = "Normal market conditions."
    risk_aversion: float = 0.5
    soc_reserve_frac: float = 0.0
    horizon: int = 24
    cycle_cost_multiplier: float = 1.0
    price_shock: float = 0.0       # multiplicative stress applied to forecast
    volatility_multiplier: float = 1.0


SCENARIOS: dict[str, ScenarioConfig] = {
    "normal": ScenarioConfig(
        name="normal",
        description="Normal market. Balanced risk, standard degradation sensitivity.",
        risk_aversion=0.5, soc_reserve_frac=0.0, horizon=24,
    ),
    "high_volatility": ScenarioConfig(
        name="high_volatility",
        description="Elevated volatility. Wider forecast error, hold reserve for spikes.",
        risk_aversion=1.2, soc_reserve_frac=0.10, horizon=24,
        volatility_multiplier=1.8,
    ),
    "low_price_high_renewable": ScenarioConfig(
        name="low_price_high_renewable",
        description="Renewable surplus, depressed/negative midday prices. Charge aggressively.",
        risk_aversion=0.2, soc_reserve_frac=0.0, horizon=48,
        price_shock=-0.25, cycle_cost_multiplier=0.8,
    ),
    "low_renewable": ScenarioConfig(
        name="low_renewable",
        description="Renewable drought: thermal sets price, levels and spikes both higher.",
        risk_aversion=0.7, soc_reserve_frac=0.10, horizon=24,
        price_shock=0.20, volatility_multiplier=1.3,
    ),
    "price_spike": ScenarioConfig(
        name="price_spike",
        description="Scarcity event expected. Preserve energy for the peak.",
        risk_aversion=0.8, soc_reserve_frac=0.15, horizon=24,
        price_shock=0.35, volatility_multiplier=1.5,
    ),
    "conservative": ScenarioConfig(
        name="conservative",
        description="Capital-preservation mode. Few, high-confidence cycles only.",
        risk_aversion=2.0, soc_reserve_frac=0.20, horizon=24,
        cycle_cost_multiplier=1.5,
    ),
}

# Operator-facing alias: "high renewable" is the low-price surplus regime.
SCENARIOS["high_renewable"] = replace(
    SCENARIOS["low_price_high_renewable"], name="high_renewable")

# Order used by the dashboard's scenario selector.
SCENARIO_ORDER = ["normal", "high_volatility", "low_renewable",
                  "high_renewable", "price_spike", "conservative"]

DEFAULT_BATTERY = BatteryConfig()
DEFAULT_RISK = RiskConfig()
