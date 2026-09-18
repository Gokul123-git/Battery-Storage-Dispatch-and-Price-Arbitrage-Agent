"""Shared fixtures. Kept deliberately small so the suite runs in seconds."""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

from src.config import BatteryConfig, RiskConfig          # noqa: E402
from src.data_generator import generate_market_data       # noqa: E402
from src.forecaster import ForecastResult                 # noqa: E402


@pytest.fixture(scope="session")
def market() -> pd.DataFrame:
    """~4 months of deterministic synthetic data (fast, still realistic)."""
    return generate_market_data(years=0.33, seed=7)


@pytest.fixture
def battery() -> BatteryConfig:
    return BatteryConfig(capacity_mwh=100, power_mw=25, round_trip_efficiency=0.90,
                         soc_min=0.10, soc_max=0.90, cycle_cost_per_mwh=5.0)


@pytest.fixture
def risk() -> RiskConfig:
    return RiskConfig(risk_aversion=0.0, soc_reserve_frac=0.0, max_daily_cycles=2.0)


class OracleForecaster:
    """Test double: returns the true future. Used to validate the MACHINERY."""

    name = "oracle"

    def predict_window(self, df, decision_idx, horizon=24):
        s, e = decision_idx + 1, decision_idx + 1 + horizon
        a = df["price_per_mwh"].iloc[s:e].to_numpy(float)
        return ForecastResult(ts=df["ts"].iloc[s:e].reset_index(drop=True),
                              forecast=a, error_scale=np.full(len(a), 1e-6),
                              actual=a, model_name="oracle")


@pytest.fixture
def oracle() -> OracleForecaster:
    return OracleForecaster()
