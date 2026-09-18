"""Price forecasting: gradient-boosted model + seasonal naive baseline.

Interface contract (teammates may swap the internals freely):

    model = PriceForecaster(horizon=24).fit(df)
    fc    = model.predict_window(df, decision_idx)   ->  ForecastResult

``ForecastResult`` carries point forecasts *and* a per-hour error scale, which
the risk module turns into conservative prices for the optimiser.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .config import MODEL_DIR, SEED
from .features import build_features, chronological_split, make_supervised

try:  # LightGBM preferred; sklearn is a fully functional fallback.
    import lightgbm as lgb
    HAS_LGBM = True
except Exception:  # pragma: no cover - environment dependent
    HAS_LGBM = False
from sklearn.ensemble import HistGradientBoostingRegressor


@dataclass
class ForecastResult:
    """Point forecast for a horizon plus its uncertainty band."""

    ts: pd.Series                 # target timestamps
    forecast: np.ndarray          # predicted price_per_mwh
    error_scale: np.ndarray       # per-step 1-sigma error estimate
    actual: np.ndarray | None = None
    model_name: str = "gbm"

    @property
    def horizon(self) -> int:
        return len(self.forecast)

    def to_frame(self) -> pd.DataFrame:
        d = {"ts": self.ts.values, "forecast_price": self.forecast,
             "error_scale": self.error_scale}
        if self.actual is not None:
            d["actual_price"] = self.actual
        return pd.DataFrame(d)


@dataclass
class ForecastMetrics:
    mae: float
    rmse: float
    bias: float                    # mean error (pred - actual)
    mape_robust: float
    by_hour: pd.DataFrame = field(repr=False, default_factory=pd.DataFrame)
    n: int = 0

    def to_dict(self) -> dict:
        return {"mae": self.mae, "rmse": self.rmse, "bias": self.bias,
                "mape_robust": self.mape_robust, "n": self.n}


def evaluate(y_true, y_pred, hours=None) -> ForecastMetrics:
    """MAE / RMSE / bias, plus a per-hour breakdown when hours are supplied."""
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    err = y_pred - y_true
    # Robust MAPE: prices cross zero, so scale by a floor rather than |price|.
    denom = np.maximum(np.abs(y_true), 10.0)
    by_hour = pd.DataFrame()
    if hours is not None:
        by_hour = (
            pd.DataFrame({"hour": hours, "err": err, "abs_err": np.abs(err)})
            .groupby("hour")
            .agg(mae=("abs_err", "mean"), bias=("err", "mean"), n=("err", "size"))
            .reset_index()
        )
    return ForecastMetrics(
        mae=float(np.mean(np.abs(err))),
        rmse=float(np.sqrt(np.mean(err ** 2))),
        bias=float(np.mean(err)),
        mape_robust=float(np.mean(np.abs(err) / denom) * 100),
        by_hour=by_hour,
        n=int(len(err)),
    )


class SeasonalNaive:
    """Baseline: price at the same hour one day (or one week) ago."""

    def __init__(self, lag: int = 24):
        self.lag = lag
        self.name = f"seasonal_naive_{lag}h"

    def fit(self, df: pd.DataFrame) -> "SeasonalNaive":
        return self

    def predict_window(self, df: pd.DataFrame, decision_idx: int,
                       horizon: int = 24) -> ForecastResult:
        start, end = decision_idx + 1, decision_idx + 1 + horizon
        src = np.arange(start, end) - self.lag
        src = np.clip(src, 0, len(df) - 1)
        pred = df["price_per_mwh"].to_numpy()[src]
        return ForecastResult(
            ts=df["ts"].iloc[start:end].reset_index(drop=True),
            forecast=pred,
            error_scale=np.full(horizon, float(np.std(pred)) or 10.0),
            actual=df["price_per_mwh"].iloc[start:end].to_numpy(),
            model_name=self.name,
        )


class PriceForecaster:
    """Gradient-boosted regressor over leakage-safe features.

    One model is trained per horizon setting. The model is *direct* multi-step:
    features are shifted by ``horizon`` so a single regressor predicts every
    step of the horizon from information available at decision time.
    """

    def __init__(self, horizon: int = 24, seed: int = SEED, n_estimators: int = 300):
        self.horizon = int(horizon)
        self.seed = seed
        self.n_estimators = n_estimators
        self.model = None
        self.cols: list[str] = []
        self.metrics: ForecastMetrics | None = None
        self.baseline_metrics: ForecastMetrics | None = None
        self.error_by_hour: np.ndarray = np.full(24, 10.0)
        self.residual_sigma: float = 10.0
        self.train_seconds: float = 0.0
        self.name = "lightgbm" if HAS_LGBM else "sklearn_hgb"

    # -- training ---------------------------------------------------------
    def fit(self, df: pd.DataFrame, train: float = 0.70, val: float = 0.15) -> "PriceForecaster":
        t0 = time.perf_counter()
        X, y, ts = make_supervised(df, horizon=self.horizon)
        self.cols = list(X.columns)
        s_tr, s_va, s_te = chronological_split(len(y), train, val)

        Xtr, ytr = X.iloc[s_tr], y.iloc[s_tr]
        Xva, yva = X.iloc[s_va], y.iloc[s_va]
        Xte, yte = X.iloc[s_te], y.iloc[s_te]

        if HAS_LGBM:
            self.model = lgb.LGBMRegressor(
                objective="l1",              # robust to price spikes
                n_estimators=self.n_estimators,
                learning_rate=0.06,
                num_leaves=48,
                min_child_samples=30,
                subsample=0.85, subsample_freq=1,
                colsample_bytree=0.85,
                random_state=self.seed,
                n_jobs=-1,
                verbose=-1,
            )
            self.model.fit(
                Xtr, ytr,
                eval_set=[(Xva, yva)],
                eval_metric="l1",
                callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(0)],
            )
        else:
            self.model = HistGradientBoostingRegressor(
                loss="absolute_error",
                max_iter=self.n_estimators,
                learning_rate=0.06,
                max_leaf_nodes=48,
                min_samples_leaf=30,
                random_state=self.seed,
                early_stopping=True,
                validation_fraction=0.15,
            )
            self.model.fit(Xtr, ytr)

        # -- honest out-of-sample evaluation on the held-out tail ----------
        pred_te = self.model.predict(Xte)
        hours_te = Xte["hour"].to_numpy()
        self.metrics = evaluate(yte.to_numpy(), pred_te, hours_te)

        # Seasonal-naive baseline over the identical test window.
        price_by_ts = df.set_index("ts")["price_per_mwh"]
        target_ts = pd.DatetimeIndex(ts.iloc[s_te])
        bp = price_by_ts.reindex(target_ts - pd.Timedelta(hours=24)).to_numpy()
        mask = ~np.isnan(bp)
        self.baseline_metrics = evaluate(yte.to_numpy()[mask], bp[mask], hours_te[mask])

        # -- uncertainty model: per-hour absolute error on validation ------
        pred_va = self.model.predict(Xva)
        res = pred_va - yva.to_numpy()
        self.residual_sigma = float(np.std(res))
        eb = pd.DataFrame({"hour": Xva["hour"].to_numpy(), "e": np.abs(res)})
        per_hour = eb.groupby("hour")["e"].mean()
        # Convert MAE -> sigma-like scale (for a normal, sigma ~ 1.2533*MAE).
        self.error_by_hour = np.array(
            [float(per_hour.get(h, self.residual_sigma)) * 1.2533 for h in range(24)]
        )
        self.train_seconds = time.perf_counter() - t0
        return self

    # -- inference --------------------------------------------------------
    def predict_window(
        self, df: pd.DataFrame, decision_idx: int, horizon: int | None = None
    ) -> ForecastResult:
        """Forecast hours ``decision_idx+1 .. decision_idx+horizon``.

        Leakage guard: the frame is truncated at ``decision_idx`` and all future
        prices are blanked to NaN before features are built, so no future price
        can reach the feature matrix even through a rolling window.
        """
        if self.model is None:
            raise RuntimeError("PriceForecaster.fit() must be called first")
        h = int(horizon or self.horizon)
        if h > self.horizon:
            raise ValueError(
                f"requested horizon {h} exceeds trained horizon {self.horizon}; "
                "train a model with horizon >= requested"
            )
        start, end = decision_idx + 1, decision_idx + 1 + h
        if end > len(df):
            raise IndexError("forecast window extends beyond the dataset")

        sub = df.iloc[:end].copy()
        sub.iloc[start:, sub.columns.get_loc("price_per_mwh")] = np.nan
        feat = build_features(sub, horizon=self.horizon)
        win = feat.iloc[start:end]
        X = win[self.cols].astype(float)
        X = X.fillna(X.median(numeric_only=True)).fillna(0.0)

        pred = np.asarray(self.model.predict(X), float)
        hours = win["hour"].to_numpy().astype(int)
        scale = self.error_by_hour[hours]
        # Uncertainty grows with lead time (random-walk-ish widening).
        scale = scale * np.sqrt(1.0 + np.arange(h) / max(h, 1))

        return ForecastResult(
            ts=df["ts"].iloc[start:end].reset_index(drop=True),
            forecast=pred,
            error_scale=scale,
            actual=df["price_per_mwh"].iloc[start:end].to_numpy(),
            model_name=self.name,
        )

    # -- persistence ------------------------------------------------------
    def save(self, path: str | Path | None = None) -> str:
        import joblib
        path = Path(path or MODEL_DIR / f"forecaster_h{self.horizon}.joblib")
        joblib.dump(self, path)
        return str(path)

    @staticmethod
    def load(path: str | Path) -> "PriceForecaster":
        import joblib
        return joblib.load(path)
