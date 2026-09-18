"""Persistence, used where each technology actually fits.

* Parquet  -> the market time series (columnar, compressed) -- see data_loader.
* SQLite   -> run headers and per-hour dispatch logs (small, transactional,
              append-only operational records).
* DuckDB   -> ad-hoc analytics over those logs, without an ETL step.

Nothing here is required for the numerics to run; every call degrades to a
no-op warning if the database cannot be opened.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager

import pandas as pd

from .config import RUNS_DB

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS runs (
    run_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at   TEXT NOT NULL,
    scenario     TEXT,
    horizon      INTEGER,
    risk_level   REAL,
    net_profit   REAL,
    regret       REAL,
    regret_pct   REAL,
    metrics_json TEXT
);
CREATE TABLE IF NOT EXISTS dispatch_log (
    run_id     INTEGER,
    ts         TEXT,
    price      REAL,
    forecast   REAL,
    charge_mw  REAL,
    discharge_mw REAL,
    soc_mwh    REAL,
    reserve_mwh REAL,
    action     TEXT,
    risk_score REAL,
    profit     REAL
);
CREATE INDEX IF NOT EXISTS idx_dispatch_run ON dispatch_log(run_id);
"""


@contextmanager
def connect(path=RUNS_DB):
    con = sqlite3.connect(str(path))
    try:
        con.executescript(SCHEMA_SQL)
        yield con
        con.commit()
    finally:
        con.close()


def log_run(scenario: str, horizon: int, risk_level: float, metrics: dict,
            log: pd.DataFrame | None = None, path=RUNS_DB) -> int | None:
    """Persist a run header plus its dispatch log. Returns the run id."""
    try:
        with connect(path) as con:
            cur = con.execute(
                "INSERT INTO runs (created_at, scenario, horizon, risk_level, "
                "net_profit, regret, regret_pct, metrics_json) "
                "VALUES (datetime('now'), ?, ?, ?, ?, ?, ?, ?)",
                (scenario, int(horizon), float(risk_level),
                 float(metrics.get("net_profit", 0.0) or 0.0),
                 float(metrics.get("regret", 0.0) or 0.0),
                 float(metrics.get("regret_pct", 0.0) or 0.0),
                 json.dumps(metrics, default=str)),
            )
            run_id = int(cur.lastrowid)
            if log is not None and not log.empty:
                cols = ["ts", "price", "forecast", "charge_mw", "discharge_mw",
                        "soc_mwh", "reserve_mwh", "action", "risk_score", "profit"]
                d = log.reindex(columns=cols).copy()
                d["ts"] = d["ts"].astype(str)
                d.insert(0, "run_id", run_id)
                con.executemany(
                    "INSERT INTO dispatch_log VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    d.itertuples(index=False, name=None),
                )
            return run_id
    except Exception as exc:                      # storage is never fatal
        print(f"[storage] run not logged: {exc}")
        return None


def recent_runs(limit: int = 20, path=RUNS_DB) -> pd.DataFrame:
    try:
        with connect(path) as con:
            return pd.read_sql_query(
                "SELECT run_id, created_at, scenario, horizon, risk_level, "
                "net_profit, regret, regret_pct FROM runs "
                "ORDER BY run_id DESC LIMIT ?", con, params=(limit,))
    except Exception:
        return pd.DataFrame()


def query(sql: str, path=RUNS_DB) -> pd.DataFrame:
    """Analytics over the SQLite logs via DuckDB (falls back to sqlite3)."""
    try:
        import duckdb
        con = duckdb.connect()
        con.execute("INSTALL sqlite; LOAD sqlite;")
        con.execute(f"ATTACH '{path}' AS runsdb (TYPE sqlite);")
        con.execute("USE runsdb;")
        return con.execute(sql).df()
    except Exception:
        try:
            with connect(path) as con:
                return pd.read_sql_query(sql, con)
        except Exception as exc:
            print(f"[storage] query failed: {exc}")
            return pd.DataFrame()
