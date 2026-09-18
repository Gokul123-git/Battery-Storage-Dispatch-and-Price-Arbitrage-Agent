"""Leakage-safe feature engineering for hourly price forecasting.

LEAKAGE RULE
------------
A model that predicts price at ``t + h`` may only use information observable at
or before ``t``. Every lag/rolling feature here is therefore shifted by the
forecast horizon ``h`` *before* being paired with the target, so the feature
row for target time ``T`` contains nothing newer than ``T - h``.

Exogenous load/renewable values are treated as *day-ahead schedules* (they are
published before the market clears in real systems), so they are available for
the target hour itself. This is stated explicitly rather than assumed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PRICE_LAGS = (1, 2, 3, 24, 25, 48, 168)
ROLL_WINDOWS = (24, 168)

CALENDAR_FEATURES = [
    "hour", "day_of_week", "month", "is_weekend",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
]
EXOG_FEATURES = ["load_mw", "renewable_mw", "renewable_ratio", "net_load_mw"]


def build_features(df: pd.DataFrame, horizon: int = 24) -> pd.DataFrame:
    """Return the frame with calendar, lag, rolling and exogenous features.

    ``horizon`` is the number of hours ahead being predicted; all
    price-derived features are shifted by it to prevent look-ahead.
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    out = df.copy().sort_values("ts").reset_index(drop=True)
    ts = out["ts"].dt

    # ---- calendar (known arbitrarily far ahead) --------------------------
    out["hour"] = ts.hour
    out["day_of_week"] = ts.dayofweek
    out["month"] = ts.month
    out["is_weekend"] = (out["day_of_week"] >= 5).astype(int)
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)
    out["dow_sin"] = np.sin(2 * np.pi * out["day_of_week"] / 7)
    out["dow_cos"] = np.cos(2 * np.pi * out["day_of_week"] / 7)

    # ---- exogenous (day-ahead schedules, available for the target hour) --
    out["net_load_mw"] = out["load_mw"] - out["renewable_mw"]
    out["renewable_ratio"] = out["renewable_mw"] / out["load_mw"].replace(0, np.nan)
    out["renewable_ratio"] = out["renewable_ratio"].fillna(0.0)

    # ---- price history, shifted by the horizon ---------------------------
    # shift(h) => value at T-h is the most recent price the model may see.
    base = out["price_per_mwh"].shift(horizon)
    for lag in PRICE_LAGS:
        out[f"lag_{lag}"] = out["price_per_mwh"].shift(horizon + lag - 1)
    for w in ROLL_WINDOWS:
        out[f"rolling_mean_{w}"] = base.rolling(w, min_periods=max(2, w // 4)).mean()
        out[f"rolling_std_{w}"] = base.rolling(w, min_periods=max(2, w // 4)).std()
    out["rolling_min_24"] = base.rolling(24, min_periods=6).min()
    out["rolling_max_24"] = base.rolling(24, min_periods=6).max()
    out["price_spread_24"] = out["rolling_max_24"] - out["rolling_min_24"]

    # Same-hour-yesterday / last-week levels relative to recent mean.
    out["lag_24_dev"] = out["lag_24"] - out["rolling_mean_24"]
    out["lag_168_dev"] = out["lag_168"] - out["rolling_mean_168"]

    return out


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Model input columns: everything engineered, never the raw target."""
    banned = {"ts", "price_per_mwh"}
    return [c for c in df.columns if c not in banned]


def make_supervised(
    df: pd.DataFrame, horizon: int = 24
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Return ``(X, y, ts)`` aligned and stripped of warm-up rows."""
    feat = build_features(df, horizon=horizon)
    cols = feature_columns(feat)
    feat = feat.dropna(subset=cols)
    return feat[cols].reset_index(drop=True), \
        feat["price_per_mwh"].reset_index(drop=True), \
        feat["ts"].reset_index(drop=True)


def chronological_split(
    n: int, train: float = 0.70, val: float = 0.15
) -> tuple[slice, slice, slice]:
    """Time-ordered split. NEVER shuffled -- this is a time series."""
    if not 0 < train < 1 or not 0 <= val < 1 or train + val >= 1:
        raise ValueError("require train>0, val>=0 and train+val<1")
    i_tr = int(n * train)
    i_va = int(n * (train + val))
    return slice(0, i_tr), slice(i_tr, i_va), slice(i_va, n)
