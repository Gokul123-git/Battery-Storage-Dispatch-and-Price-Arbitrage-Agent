"""Forecasting: leakage safety, chronological splits, shapes, metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features import build_features, chronological_split, make_supervised
from src.forecaster import PriceForecaster, SeasonalNaive, evaluate


def test_chronological_split_is_ordered_and_disjoint():
    tr, va, te = chronological_split(1000, 0.7, 0.15)
    assert tr.start == 0 and tr.stop == 700
    assert va.start == 700 and va.stop == 850
    assert te.start == 850 and te.stop == 1000


def test_features_never_use_the_future(market):
    """A lag/rolling feature for target T must not move when T's own price does."""
    h = 24
    a = build_features(market, horizon=h)
    poisoned = market.copy()
    i = 500
    poisoned.loc[i:, "price_per_mwh"] += 500.0       # corrupt the future only
    b = build_features(poisoned, horizon=h)
    cols = [c for c in a.columns if c.startswith(("lag_", "rolling_", "price_spread"))]
    # Rows strictly before the corruption must be identical.
    pd.testing.assert_frame_equal(a.loc[: i - 1, cols], b.loc[: i - 1, cols])
    # And the row at i itself may only see prices up to i-h.
    assert np.allclose(a.loc[i, cols].to_numpy(dtype=float),
                       b.loc[i, cols].to_numpy(dtype=float), equal_nan=True)


def test_supervised_alignment_has_no_nans(market):
    X, y, ts = make_supervised(market, horizon=24)
    assert len(X) == len(y) == len(ts) and len(X) > 100
    assert not X.isna().any().any()
    assert "price_per_mwh" not in X.columns


def test_prediction_window_length_and_timestamps(market):
    m = PriceForecaster(horizon=24, n_estimators=60).fit(market)
    idx = len(market) - 30
    fc = m.predict_window(market, idx, horizon=24)
    assert fc.horizon == 24 and len(fc.ts) == 24
    assert fc.ts.iloc[0] == market["ts"].iloc[idx + 1]
    assert np.isfinite(fc.forecast).all()
    assert (fc.error_scale > 0).all()


def test_forecast_is_invariant_to_future_prices(market):
    """The controller's forecast must not change if the future is rewritten."""
    m = PriceForecaster(horizon=24, n_estimators=60).fit(market)
    idx = len(market) - 40
    base = m.predict_window(market, idx, horizon=24).forecast
    tampered = market.copy()
    tampered.loc[idx + 1:, "price_per_mwh"] = 9999.0
    after = m.predict_window(tampered, idx, horizon=24).forecast
    assert np.allclose(base, after), "future prices leaked into the forecast"


def test_metrics_are_sane_and_beat_or_match_baseline(market):
    m = PriceForecaster(horizon=24, n_estimators=120).fit(market)
    assert m.metrics.mae > 0 and m.metrics.rmse >= m.metrics.mae
    assert np.isfinite(m.residual_sigma) and m.residual_sigma > 0
    assert len(m.error_by_hour) == 24 and (m.error_by_hour > 0).all()
    assert m.metrics.mae <= m.baseline_metrics.mae * 1.05


def test_seasonal_naive_baseline_runs(market):
    b = SeasonalNaive(24)
    fc = b.predict_window(market, len(market) - 30, horizon=24)
    assert fc.horizon == 24 and np.isfinite(fc.forecast).all()


def test_evaluate_math():
    m = evaluate([0.0, 10.0], [1.0, 8.0])
    assert m.mae == pytest.approx(1.5)
    assert m.rmse == pytest.approx(np.sqrt((1 + 4) / 2))
    assert m.bias == pytest.approx(-0.5)


def test_requesting_longer_horizon_than_trained_is_refused(market):
    m = PriceForecaster(horizon=24, n_estimators=40).fit(market)
    with pytest.raises(ValueError):
        m.predict_window(market, len(market) - 60, horizon=48)
