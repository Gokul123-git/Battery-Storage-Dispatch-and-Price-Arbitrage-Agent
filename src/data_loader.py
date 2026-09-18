"""Load market data from Parquet/CSV, or fall back to the synthetic generator.

The loader is deliberately forgiving about external datasets: it maps common
column aliases onto the canonical schema, parses timestamps, enforces a regular
hourly grid and handles missing values. The synthetic generator is the default
so the demo runs with zero setup.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import MARKET_PARQUET
from .data_generator import SCHEMA, generate_market_data, write_market_data

# Common aliases seen in public ISO/market exports -> canonical name.
ALIASES: dict[str, str] = {
    "timestamp": "ts", "datetime": "ts", "date": "ts", "time": "ts",
    "period": "ts", "settlement_date": "ts", "interval_start": "ts",
    "price": "price_per_mwh", "lmp": "price_per_mwh", "rrp": "price_per_mwh",
    "spot_price": "price_per_mwh", "da_price": "price_per_mwh",
    "price_eur_mwh": "price_per_mwh", "price_usd_mwh": "price_per_mwh",
    "clearing_price": "price_per_mwh", "energy_price": "price_per_mwh",
    "load": "load_mw", "demand": "load_mw", "demand_mw": "load_mw",
    "total_load": "load_mw", "system_load": "load_mw", "consumption": "load_mw",
    "renewable": "renewable_mw", "vre": "renewable_mw", "res": "renewable_mw",
    "wind_solar": "renewable_mw", "renewables_mw": "renewable_mw",
    "generation_renewable": "renewable_mw",
}


class DataError(ValueError):
    """Raised when an external dataset cannot be mapped onto the schema."""


def load_market_data(
    path: str | Path | None = None,
    *,
    fallback_to_synthetic: bool = True,
    column_map: dict[str, str] | None = None,
    **gen_kwargs,
) -> pd.DataFrame:
    """Return a clean hourly DataFrame with the canonical schema.

    Parameters
    ----------
    path
        Parquet or CSV file. Defaults to ``data/market_prices.parquet``.
    column_map
        Explicit ``{source_column: canonical_name}`` override, applied before
        the built-in alias table. Use this for datasets we cannot guess.
    """
    path = Path(path) if path is not None else MARKET_PARQUET

    if not path.exists():
        if not fallback_to_synthetic:
            raise DataError(f"Dataset not found: {path}")
        write_market_data(path=path, **gen_kwargs)

    raw = _read_any(path)
    return normalise(raw, column_map=column_map)


def _read_any(path: Path) -> pd.DataFrame:
    suf = path.suffix.lower()
    if suf in (".parquet", ".pq"):
        return pd.read_parquet(path)
    if suf in (".csv", ".txt"):
        return pd.read_csv(path)
    if suf in (".xlsx", ".xls"):
        return pd.read_excel(path)
    raise DataError(f"Unsupported file type '{suf}'. Use .parquet, .csv or .xlsx")


def normalise(df: pd.DataFrame, column_map: dict[str, str] | None = None) -> pd.DataFrame:
    """Map aliases, coerce dtypes, enforce an hourly grid, fill gaps."""
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

    if column_map:
        df = df.rename(columns={k.strip().lower().replace(" ", "_"): v
                                for k, v in column_map.items()})
    df = df.rename(columns={k: v for k, v in ALIASES.items()
                            if k in df.columns and v not in df.columns})

    if "ts" not in df.columns:
        # Fall back to a DatetimeIndex if the file was saved with one.
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index().rename(columns={df.index.name or "index": "ts"})
        else:
            raise DataError(
                "No timestamp column found. Expected one of: "
                f"{sorted(k for k, v in ALIASES.items() if v == 'ts')}. "
                f"Got: {list(df.columns)}"
            )
    if "price_per_mwh" not in df.columns:
        raise DataError(
            "No price column found. Expected one of: "
            f"{sorted(k for k, v in ALIASES.items() if v == 'price_per_mwh')}. "
            f"Got: {list(df.columns)}"
        )

    df["ts"] = pd.to_datetime(df["ts"], errors="coerce", utc=False)
    if getattr(df["ts"].dtype, "tz", None) is not None:
        df["ts"] = df["ts"].dt.tz_localize(None)
    df = df.dropna(subset=["ts"]).sort_values("ts")
    df = df.drop_duplicates(subset="ts", keep="last")

    df["price_per_mwh"] = pd.to_numeric(df["price_per_mwh"], errors="coerce")
    # NOTE: negative prices are valid market outcomes and are preserved.

    # Optional exogenous columns -- synthesise neutral stand-ins if absent so
    # downstream feature engineering never has to branch.
    if "load_mw" not in df.columns:
        df["load_mw"] = np.nan
    if "renewable_mw" not in df.columns:
        df["renewable_mw"] = np.nan
    for c in ("load_mw", "renewable_mw"):
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = _regular_hourly_grid(df)

    # Fill gaps causally (forward), then backfill only the leading edge.
    df["price_per_mwh"] = df["price_per_mwh"].ffill().bfill()
    for c in ("load_mw", "renewable_mw"):
        if df[c].notna().any():
            df[c] = df[c].ffill().bfill()
        else:
            df[c] = 0.0

    if df["price_per_mwh"].isna().all():
        raise DataError("Price column is entirely empty after parsing.")
    if len(df) < 24 * 30:
        raise DataError(f"Need at least 30 days of hourly data, got {len(df)} rows.")

    return df[SCHEMA].reset_index(drop=True)


def _regular_hourly_grid(df: pd.DataFrame) -> pd.DataFrame:
    """Resample onto a gap-free hourly index (mean-aggregating sub-hourly data)."""
    s = df.set_index("ts")
    step = s.index.to_series().diff().dropna()
    if not step.empty and step.median() < pd.Timedelta("1h"):
        s = s.resample("h").mean()       # sub-hourly (e.g. 5/15 min) -> hourly
    else:
        s = s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="h"))
    return s.rename_axis("ts").reset_index()


def describe(df: pd.DataFrame) -> dict:
    """Summary used by the dashboard's data panel."""
    p = df["price_per_mwh"]
    return {
        "rows": len(df),
        "start": str(df["ts"].iloc[0]),
        "end": str(df["ts"].iloc[-1]),
        "price_mean": float(p.mean()),
        "price_std": float(p.std()),
        "price_min": float(p.min()),
        "price_max": float(p.max()),
        "negative_hours": int((p < 0).sum()),
        "negative_pct": float(100 * (p < 0).mean()),
    }
