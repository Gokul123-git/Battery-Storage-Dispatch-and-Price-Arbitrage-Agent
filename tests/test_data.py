"""Data contract: schema, timestamps, plausibility, negative prices kept."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_generator import SCHEMA, generate_market_data
from src.data_loader import DataError, normalise


def test_schema_and_dtypes(market):
    assert list(market.columns) == SCHEMA
    assert pd.api.types.is_datetime64_any_dtype(market["ts"])
    assert market["price_per_mwh"].notna().all()


def test_hourly_grid_is_regular_and_unique(market):
    d = market["ts"].diff().dropna().unique()
    assert len(d) == 1 and d[0] == np.timedelta64(1, "h")
    assert market["ts"].is_unique and market["ts"].is_monotonic_increasing


def test_two_year_generation_length():
    df = generate_market_data(years=2.0)
    assert 17_000 <= len(df) <= 17_600        # ~2 * 365.25 * 24


def test_negative_prices_exist_and_are_not_clipped(market):
    assert (market["price_per_mwh"] < 0).any(), "negative price events must occur"
    assert market["price_per_mwh"].min() < -1.0


def test_no_impossible_values(market):
    assert (market["load_mw"] > 0).all()
    assert (market["renewable_mw"] >= 0).all()
    assert np.isfinite(market[["price_per_mwh", "load_mw", "renewable_mw"]]
                       .to_numpy()).all()


def test_generator_is_deterministic():
    a = generate_market_data(years=0.1, seed=3)
    b = generate_market_data(years=0.1, seed=3)
    pd.testing.assert_frame_equal(a, b)


def test_diurnal_and_weekly_structure(market):
    by_hour = market.groupby(market["ts"].dt.hour)["price_per_mwh"].mean()
    assert by_hour.max() - by_hour.min() > 5, "no diurnal price shape"
    wk = market.groupby(market["ts"].dt.dayofweek >= 5)["load_mw"].mean()
    assert wk[True] < wk[False], "weekend load should be lower"


def test_renewables_depress_price(market):
    ratio = market["renewable_mw"] / market["load_mw"]
    hi = market.loc[ratio > ratio.quantile(0.8), "price_per_mwh"].mean()
    lo = market.loc[ratio < ratio.quantile(0.2), "price_per_mwh"].mean()
    assert hi < lo, "high renewable hours must be cheaper"


def test_loader_maps_aliases_and_keeps_negatives(market, tmp_path):
    src = market.rename(columns={"ts": "datetime", "price_per_mwh": "lmp",
                                 "load_mw": "demand", "renewable_mw": "vre"})
    out = normalise(src)
    assert list(out.columns) == SCHEMA
    assert out["price_per_mwh"].min() == market["price_per_mwh"].min()


def test_loader_rejects_unusable_frame():
    try:
        normalise(pd.DataFrame({"foo": [1, 2, 3]}))
    except DataError:
        return
    raise AssertionError("expected DataError for a frame with no timestamp")
