"""SQLite store for model metadata, probe runs, raw measurements, and fitted ceilings."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "suite.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS models (
    hf_id           TEXT PRIMARY KEY,
    weights_gb      REAL,
    n_layers        INTEGER,
    growing_layers  INTEGER,
    kv_heads        INTEGER,
    head_dim        INTEGER,
    hidden_size     INTEGER,
    max_context     INTEGER,
    cache_type      TEXT,
    can_quantize_kv INTEGER,
    layer_types     TEXT,
    updated_at      TEXT
);

CREATE TABLE IF NOT EXISTS probe_runs (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    hf_id              TEXT NOT NULL,
    kv_bits            INTEGER,          -- NULL = fp16
    kv_group_size      INTEGER,
    quantized_kv_start INTEGER,
    mlx_version        TEXT,
    wall_gb            REAL,
    safe_threshold_gb  REAL,
    created_at         TEXT,
    FOREIGN KEY (hf_id) REFERENCES models(hf_id)
);

CREATE TABLE IF NOT EXISTS measurements (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id        INTEGER NOT NULL,
    context       INTEGER NOT NULL,
    mlx_peak_gb   REAL,
    mlx_true_gb   REAL,
    os_wired_gb   REAL,
    status        TEXT,                  -- ok | skipped | error
    note          TEXT,
    FOREIGN KEY (run_id) REFERENCES probe_runs(id)
);

CREATE TABLE IF NOT EXISTS fits (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id           INTEGER NOT NULL,
    model_base_gb    REAL,               -- DELTA intercept: model's own wired footprint at c->0 (invariant)
    slope_gb_per_k   REAL,               -- GB per 1000 tokens
    r2               REAL,
    ref_baseline_gb  REAL,               -- live system baseline used for the ceiling calc
    threshold_gb     REAL,
    safe_ceiling_ctx INTEGER,            -- context where (ref_baseline + model_base + slope*c) hits threshold
    hard_wall_ctx    INTEGER,            -- ... hits the 17.18 wall
    n_points         INTEGER,
    created_at       TEXT,
    FOREIGN KEY (run_id) REFERENCES probe_runs(id)
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    con.executescript(SCHEMA)
    return con


def upsert_model(con: sqlite3.Connection, info: dict) -> None:
    cols = ["hf_id", "weights_gb", "n_layers", "growing_layers", "kv_heads", "head_dim",
            "hidden_size", "max_context", "cache_type", "can_quantize_kv", "layer_types"]
    row = {k: info.get(k) for k in cols}
    row["can_quantize_kv"] = int(bool(row["can_quantize_kv"]))
    if isinstance(row["layer_types"], dict):
        row["layer_types"] = json.dumps(row["layer_types"])
    row["updated_at"] = _now()
    placeholders = ", ".join(f":{k}" for k in row)
    updates = ", ".join(f"{k}=excluded.{k}" for k in row if k != "hf_id")
    con.execute(
        f"INSERT INTO models ({', '.join(row)}) VALUES ({placeholders}) "
        f"ON CONFLICT(hf_id) DO UPDATE SET {updates}",
        row,
    )
    con.commit()


def start_run(con: sqlite3.Connection, hf_id: str, *, kv_bits, kv_group_size,
              quantized_kv_start, mlx_version, wall_gb, safe_threshold_gb) -> int:
    cur = con.execute(
        "INSERT INTO probe_runs (hf_id, kv_bits, kv_group_size, quantized_kv_start, "
        "mlx_version, wall_gb, safe_threshold_gb, created_at) "
        "VALUES (?,?,?,?,?,?,?,?)",
        (hf_id, kv_bits, kv_group_size, quantized_kv_start, mlx_version,
         wall_gb, safe_threshold_gb, _now()),
    )
    con.commit()
    return cur.lastrowid


def add_measurement(con: sqlite3.Connection, run_id: int, context: int, *,
                    mlx_peak_gb=None, mlx_true_gb=None, os_wired_gb=None,
                    status="ok", note=None) -> None:
    con.execute(
        "INSERT INTO measurements (run_id, context, mlx_peak_gb, mlx_true_gb, "
        "os_wired_gb, status, note) VALUES (?,?,?,?,?,?,?)",
        (run_id, context, mlx_peak_gb, mlx_true_gb, os_wired_gb, status, note),
    )
    con.commit()


def latest_fit(con: sqlite3.Connection, hf_id: str) -> dict | None:
    """Most recent fitted curve for a model, or None if never characterized."""
    row = con.execute(
        "SELECT f.* FROM fits f JOIN probe_runs r ON f.run_id = r.id "
        "WHERE r.hf_id = ? ORDER BY f.id DESC LIMIT 1",
        (hf_id,),
    ).fetchone()
    return dict(row) if row else None


def save_fit(con: sqlite3.Connection, run_id: int, fit: dict) -> None:
    fit = {**fit, "run_id": run_id, "created_at": _now()}
    keys = ["run_id", "model_base_gb", "slope_gb_per_k", "r2", "ref_baseline_gb",
            "threshold_gb", "safe_ceiling_ctx", "hard_wall_ctx", "n_points", "created_at"]
    con.execute(
        f"INSERT INTO fits ({', '.join(keys)}) VALUES ({', '.join('?' for _ in keys)})",
        tuple(fit.get(k) for k in keys),
    )
    con.commit()
