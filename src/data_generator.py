"""Deterministic synthetic hourly power-market generator.

Produces a realistic-looking but fully reproducible series with diurnal and
weekly shape, fuel-cost drift, GARCH-like volatility clustering, scarcity
spikes and genuine negative-price events driven by renewable surplus.

Negative prices are NEVER clipped -- they are a real market phenomenon and the
battery earns money by consuming during them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import MARKET_PARQUET, SEED

SCHEMA = ["ts", "price_per_mwh", "load_mw", "renewable_mw"]


def generate_market_data(
    years: float = 2.0,
    start: str = "2023-01-01",
    seed: int = SEED,
    freq: str = "h",
) -> pd.DataFrame:
    """Generate ~`years` of hourly market data. Deterministic given `seed`."""
    rng = np.random.default_rng(seed)
    n = int(round(years * 365.25 * 24))
    ts = pd.date_range(start=start, periods=n, freq=freq)

    hour = ts.hour.to_numpy()
    dow = ts.dayofweek.to_numpy()
    doy = ts.dayofyear.to_numpy()
    t = np.arange(n)

    # ---- demand ---------------------------------------------------------
    # Twin-peak diurnal load shape (morning ramp + evening peak).
    diurnal = (
        1.0
        + 0.16 * np.sin(2 * np.pi * (hour - 7) / 24)
        + 0.10 * np.sin(4 * np.pi * (hour - 3) / 24)
        + 0.12 * np.exp(-0.5 * ((hour - 19) / 2.0) ** 2)
    )
    weekly = np.where(dow >= 5, 0.88, 1.0)                  # weekend dip
    seasonal = 1.0 + 0.14 * np.cos(2 * np.pi * (doy - 15) / 365.25)  # winter peak
    load_noise = rng.normal(0, 0.03, n)
    load_mw = 9000.0 * diurnal * weekly * seasonal * (1.0 + load_noise)

    # ---- renewables -----------------------------------------------------
    # Solar: bell curve over daylight, seasonally scaled, with cloud persistence.
    solar_shape = np.clip(np.sin(np.pi * (hour - 6) / 12), 0, None) ** 1.4
    solar_season = 0.6 + 0.6 * np.cos(2 * np.pi * (doy - 172) / 365.25)  # summer peak
    cloud = _ar1(rng, n, phi=0.92, sigma=0.30, mean=1.0).clip(0.15, 1.45)
    solar = 3800.0 * solar_shape * solar_season * cloud

    # Wind: slow AR(1) process, mildly anti-correlated with solar season.
    wind = 2200.0 * _ar1(rng, n, phi=0.985, sigma=0.16, mean=1.0).clip(0.02, 3.0)
    wind *= 1.0 + 0.18 * np.cos(2 * np.pi * (doy - 15) / 365.25)
    renewable_mw = np.clip(solar + wind, 0, None)

    # ---- price ----------------------------------------------------------
    # Residual (net) load is the true economic driver of price.
    net_load = load_mw - renewable_mw
    nl_ref = float(np.median(net_load))
    nl_scale = float(np.std(net_load)) or 1.0
    z = (net_load - nl_ref) / nl_scale

    fuel_drift = 42.0 + 11.0 * _ar1(rng, n, phi=0.9995, sigma=0.010, mean=0.0)
    fuel_drift += 4.0 * np.cos(2 * np.pi * (doy - 20) / 365.25)

    # Convex supply stack: price rises steeply when residual load is high.
    stack = 17.0 * z + 7.5 * np.clip(z, 0, None) ** 2.2
    hour_premium = 5.0 * np.sin(2 * np.pi * (hour - 8) / 24) + 4.0 * np.exp(
        -0.5 * ((hour - 19) / 1.8) ** 2
    )

    # Volatility clustering (GARCH-ish): shocks beget shocks.
    vol = _garch_vol(rng, n, omega=0.6, alpha=0.12, beta=0.85, base=4.0)
    noise = rng.standard_normal(n) * vol

    price = fuel_drift + stack + hour_premium + noise

    # Scarcity spikes: more likely when residual load is already stressed.
    spike_p = 0.0006 + 0.010 * _sigmoid(2.2 * (z - 1.3))
    spikes = rng.random(n) < spike_p
    price[spikes] += rng.gamma(shape=2.2, scale=95.0, size=int(spikes.sum()))

    # Negative-price events: renewable surplus + must-run inflexibility.
    surplus = np.clip(renewable_mw / np.maximum(load_mw, 1.0) - 0.42, 0, None)
    neg_p = 0.012 + 0.55 * surplus
    negs = (rng.random(n) < neg_p) & (surplus > 0)
    price[negs] -= rng.gamma(shape=2.0, scale=32.0, size=int(negs.sum()))

    # Floor at the market's technical bid floor -- NOT at zero.
    price = np.maximum(price, -120.0)

    df = pd.DataFrame(
        {
            "ts": ts,
            "price_per_mwh": np.round(price, 2),
            "load_mw": np.round(load_mw, 1),
            "renewable_mw": np.round(renewable_mw, 1),
        }
    )
    return df[SCHEMA]


# --------------------------------------------------------------------------
def _ar1(rng, n: int, phi: float, sigma: float, mean: float = 0.0) -> np.ndarray:
    """Stationary AR(1) path, initialised at its unconditional distribution."""
    eps = rng.normal(0, sigma, n)
    out = np.empty(n)
    out[0] = mean + eps[0] / np.sqrt(max(1 - phi**2, 1e-6))
    for i in range(1, n):
        out[i] = mean + phi * (out[i - 1] - mean) + eps[i]
    return out


def _garch_vol(rng, n: int, omega: float, alpha: float, beta: float, base: float) -> np.ndarray:
    """GARCH(1,1) conditional standard deviation path."""
    var = np.empty(n)
    var[0] = base**2
    e2 = 0.0
    for i in range(1, n):
        var[i] = omega + alpha * e2 + beta * var[i - 1]
        e2 = var[i] * rng.standard_normal() ** 2
    return np.sqrt(var)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def write_market_data(df: pd.DataFrame | None = None, path=MARKET_PARQUET, **kw) -> str:
    if df is None:
        df = generate_market_data(**kw)
    df.to_parquet(path, index=False)
    return str(path)


if __name__ == "__main__":
    d = generate_market_data()
    print(write_market_data(d))
    print(d.describe())
