# ModernBERT Embeddings Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a crash-safe 2D (batch × seq_len) OS-wired memory-surface benchmark for ModernBERT embeddings to wmx-suite (issue #14).

**Architecture:** Mirrors the LLM probe layering — a single-cell isolated worker (`probe_worker_embeddings.py`) measures one `(batch, seq)` cell with a background OS-wired sampler; a parent orchestrator (`embeddings_probe.py`) does the per-row ramp with a predictive-skip gate (fit governed by real measured peaks, analytic prior cold-start only), spawning one subprocess per cell. DB tables, a CLI `benchmark-embeddings` command, and a web dashboard follow the existing Kokoro conventions.

**Tech Stack:** Python, MLX (`mlx.core`), `mlx-embeddings` (ModernBERT), SQLite, Flask, pytest.

**Spec:** `docs/superpowers/specs/2026-06-14-embeddings-benchmark-design.md`

**Conventions (all tasks):**
- Use `uv` only (`uv run pytest`, `uv add`); never `--break-system-packages`.
- Tests must be hardware-free: no real model load, no live `vm_stat`/memory probing, no production DB (`data/suite.db`). Patch on shared module objects.
- New pytest-style tests use `monkeypatch` + `tmp_path`; isolate the DB with `monkeypatch.setattr(db, "DB_PATH", tmp_path / "suite.db")`.
- Commit to `main` after each task.

---

## File Structure

- Create `wmx_suite/probe_worker_embeddings.py` — single-cell measurement worker.
- Create `wmx_suite/embeddings_probe.py` — sweep orchestrator (gate, fit, traversal, persistence).
- Modify `wmx_suite/db.py` — add `embeddings_runs` / `embeddings_measurements` tables + 5 functions.
- Modify `wmx_suite/cli.py` — add `cmd_benchmark_embeddings` + `benchmark-embeddings` subparser.
- Modify `wmx_suite/web/app.py` — add `/embeddings` + `/embeddings/run/<id>` routes.
- Create `wmx_suite/web/templates/embeddings_dashboard.html`, `embeddings_run.html`.
- Modify `wmx_suite/web/templates/base.html` — add nav entry.
- Modify `pyproject.toml` — add `mlx-embeddings>=0.1.0`.
- Create `tests/test_embeddings_benchmark.py` — DB, worker, orchestrator, CLI tests.
- Modify `tests/test_web.py` — add route smoke test (if that's where web tests live).

---

## Task 1: Add the `mlx-embeddings` dependency

**Files:**
- Modify: `pyproject.toml` (the `[project] dependencies` list, currently lines ~10-14)

- [ ] **Step 1: Add the dependency via uv**

Run:
```bash
uv add "mlx-embeddings>=0.1.0"
```
Expected: `pyproject.toml` gains `"mlx-embeddings>=0.1.0"` in `dependencies`, and `uv.lock` updates. (`mlx-embeddings` is already installed in the venv from earlier research, so resolution should be fast.)

- [ ] **Step 2: Verify the import works (no model load)**

Run:
```bash
uv run python -c "import mlx_embeddings; print('load' in dir(mlx_embeddings))"
```
Expected: `True`

- [ ] **Step 3: Verify the existing suite still imports and passes**

Run:
```bash
uv run python -m compileall -q wmx_suite tests && uv run pytest -q
```
Expected: compile OK; all existing tests pass.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "build: add mlx-embeddings dependency for ModernBERT benchmark (#14)"
```

---

## Task 2: Database schema + functions

**Files:**
- Modify: `wmx_suite/db.py` — append two `CREATE TABLE` blocks to `SCHEMA` (before the closing `"""` at line ~200); add 5 functions (after the kokoro_baseline functions, ~line 654).
- Test: `tests/test_embeddings_benchmark.py` (create)

- [ ] **Step 1: Write the failing test**

Create `tests/test_embeddings_benchmark.py`:
```python
from pathlib import Path

from wmx_suite import db


def test_embeddings_db_lifecycle(monkeypatch, tmp_path):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "suite.db")
    con = db.connect()

    run_id = db.start_embeddings_run(con, "mlx-community/test-modernbert", "0.31.2")
    assert isinstance(run_id, int)

    db.add_embeddings_measurement(
        con, run_id, batch_size=2, seq_len=512,
        os_wired_gb=4.5, peak_gb=3.1, throughput_tps=12345.0, latency_ms=82.9,
    )
    db.add_embeddings_measurement(
        con, run_id, batch_size=4, seq_len=128,
        os_wired_gb=4.8, peak_gb=3.3, throughput_tps=22222.0, latency_ms=44.1,
    )

    runs = db.get_all_embeddings_runs(con)
    assert len(runs) == 1
    assert runs[0]["model_id"] == "mlx-community/test-modernbert"
    assert runs[0]["mlx_version"] == "0.31.2"

    rows = db.get_embeddings_measurements(con, run_id)
    assert len(rows) == 2
    # ordered by batch_size, seq_len
    assert (rows[0]["batch_size"], rows[0]["seq_len"]) == (2, 512)
    assert (rows[1]["batch_size"], rows[1]["seq_len"]) == (4, 128)
    assert rows[0]["throughput_tps"] == 12345.0

    latest = db.get_latest_embeddings_run(con)
    assert latest["id"] == run_id

    # FK cascade delete
    con.execute("DELETE FROM embeddings_runs WHERE id = ?", (run_id,))
    con.commit()
    assert db.get_embeddings_measurements(con, run_id) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_embeddings_benchmark.py::test_embeddings_db_lifecycle -q`
Expected: FAIL (`AttributeError: module 'wmx_suite.db' has no attribute 'start_embeddings_run'`).

- [ ] **Step 3: Add the schema tables**

In `wmx_suite/db.py`, inside the `SCHEMA` string, immediately before the closing `"""` (after the `system_profiles` table, line ~199), add:
```sql

CREATE TABLE IF NOT EXISTS embeddings_runs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id    TEXT NOT NULL,
    mlx_version TEXT,
    created_at  TEXT
);
CREATE TABLE IF NOT EXISTS embeddings_measurements (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id         INTEGER NOT NULL,
    batch_size     INTEGER NOT NULL,
    seq_len        INTEGER NOT NULL,
    os_wired_gb    REAL,
    peak_gb        REAL,
    throughput_tps REAL,
    latency_ms     REAL,
    FOREIGN KEY (run_id) REFERENCES embeddings_runs(id) ON DELETE CASCADE
);
```

- [ ] **Step 4: Add the functions**

At the end of `wmx_suite/db.py` (after the kokoro_baseline functions), add:
```python
# --- Embeddings (ModernBERT 2D batch x seq) ---
def start_embeddings_run(con: sqlite3.Connection, model_id: str, mlx_version: str | None) -> int:
    cur = con.execute(
        "INSERT INTO embeddings_runs (model_id, mlx_version, created_at) VALUES (?, ?, ?)",
        (model_id, mlx_version, _now()),
    )
    con.commit()
    return cur.lastrowid


def add_embeddings_measurement(
    con: sqlite3.Connection,
    run_id: int,
    batch_size: int,
    seq_len: int,
    os_wired_gb: float,
    peak_gb: float,
    throughput_tps: float,
    latency_ms: float,
) -> None:
    con.execute(
        "INSERT INTO embeddings_measurements "
        "(run_id, batch_size, seq_len, os_wired_gb, peak_gb, throughput_tps, latency_ms) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (run_id, batch_size, seq_len, os_wired_gb, peak_gb, throughput_tps, latency_ms),
    )
    con.commit()


def get_all_embeddings_runs(con: sqlite3.Connection) -> list[dict]:
    rows = con.execute(
        "SELECT id, model_id, mlx_version, created_at FROM embeddings_runs ORDER BY id DESC"
    ).fetchall()
    return [dict(row) for row in rows]


def get_embeddings_measurements(con: sqlite3.Connection, run_id: int) -> list[dict]:
    rows = con.execute(
        "SELECT id, run_id, batch_size, seq_len, os_wired_gb, peak_gb, throughput_tps, latency_ms "
        "FROM embeddings_measurements WHERE run_id = ? ORDER BY batch_size ASC, seq_len ASC",
        (run_id,),
    ).fetchall()
    return [dict(row) for row in rows]


def get_latest_embeddings_run(con: sqlite3.Connection) -> dict | None:
    row = con.execute(
        "SELECT id, model_id, mlx_version, created_at FROM embeddings_runs ORDER BY id DESC LIMIT 1"
    ).fetchone()
    return dict(row) if row else None
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_embeddings_benchmark.py::test_embeddings_db_lifecycle -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add wmx_suite/db.py tests/test_embeddings_benchmark.py
git commit -m "feat(db): embeddings_runs/measurements tables + accessors (#14)"
```

---

## Task 3: The single-cell worker

**Files:**
- Create: `wmx_suite/probe_worker_embeddings.py`
- Test: `tests/test_embeddings_benchmark.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_embeddings_benchmark.py`:
```python
import json
import sys
from types import SimpleNamespace

import pytest


class _FakeArray:
    """Stand-in for an mx.array; only needs a .dtype attribute for the dtype lookup."""
    def __init__(self, dtype="bf16"):
        self.dtype = dtype


def _install_fake_worker_env(monkeypatch, *, wired_now, threshold, wired_series):
    """Patch mlx_embeddings/mx/system on the worker's module objects.

    wired_series: list of floats the background sampler / reads will return in order,
    then the last value repeats. Returns the captured stdout lines list.
    """
    from wmx_suite import probe_worker_embeddings as w

    # --- system ---
    limits = SimpleNamespace(
        wired_now_gb=wired_now,
        safe_threshold_gb=lambda margin=2.0: threshold,
    )
    monkeypatch.setattr(w.system, "read_limits", lambda: limits)
    series = list(wired_series)
    def fake_wired_gb():
        return series[0] if len(series) == 1 else series.pop(0)
    monkeypatch.setattr(w.system, "wired_gb", fake_wired_gb)

    # --- mlx.core ---
    monkeypatch.setattr(w.mx, "clear_cache", lambda: None)
    monkeypatch.setattr(w.mx, "reset_peak_memory", lambda: None)
    monkeypatch.setattr(w.mx, "get_peak_memory", lambda: int(3.0 * 1e9))
    monkeypatch.setattr(w.mx, "eval", lambda *a, **k: None)
    monkeypatch.setattr(w.mx, "zeros", lambda shape, dtype=None: _FakeArray())
    monkeypatch.setattr(w.mx, "ones", lambda shape, dtype=None: _FakeArray())
    monkeypatch.setattr(w.mx, "int32", "int32", raising=False)

    return w


class _FakeModel:
    """Callable stand-in: exposes model.model.embeddings.tok_embeddings.weight.dtype and
    returns an output object with .last_hidden_state from __call__."""
    def __init__(self):
        self.model = SimpleNamespace(
            embeddings=SimpleNamespace(
                tok_embeddings=SimpleNamespace(weight=_FakeArray(dtype="bf16"))
            )
        )

    def __call__(self, *args, **kwargs):
        return SimpleNamespace(last_hidden_state=_FakeArray())


def _fake_model():
    return _FakeModel()


def test_worker_happy_path(monkeypatch, capsys):
    w = _install_fake_worker_env(
        monkeypatch, wired_now=3.0, threshold=15.18,
        wired_series=[3.0, 4.0, 5.5, 5.0],  # sampler high-water should capture 5.5
    )
    load_calls = []
    def fake_load(model_id):
        load_calls.append(model_id)
        return _fake_model(), object()
    monkeypatch.setattr(w.mlx_embeddings, "load", fake_load)

    monkeypatch.setattr(sys, "argv", [
        "probe_worker_embeddings", "--model", "m", "--batch", "2", "--seq", "128",
        "--repeats", "2", "--margin", "2.0",
    ])
    w.main()

    out = capsys.readouterr().out
    line = next(l for l in out.splitlines() if l.startswith("{"))
    data = json.loads(line)
    assert data["status"] == "rung_done"
    assert data["batch"] == 2 and data["seq"] == 128
    assert load_calls == ["m"]
    # throughput = batch*seq/compute_time ; latency_ms = compute_time*1000 ; both > 0
    assert data["throughput_tps"] > 0 and data["latency_ms"] >= 0
    assert "os_wired_gb" in data and "peak_gb" in data


def test_worker_preflight_refusal_never_loads(monkeypatch, capsys):
    w = _install_fake_worker_env(
        monkeypatch, wired_now=15.0, threshold=15.18,  # 15.0 + MODEL_WEIGHT_EST >= 15.18
        wired_series=[15.0],
    )
    load_calls = []
    monkeypatch.setattr(w.mlx_embeddings, "load",
                        lambda mid: load_calls.append(mid) or (_fake_model(), object()))

    monkeypatch.setattr(sys, "argv", [
        "probe_worker_embeddings", "--model", "m", "--batch", "1", "--seq", "128",
        "--repeats", "1", "--margin", "2.0",
    ])
    with pytest.raises(SystemExit) as ei:
        w.main()
    assert ei.value.code == 0
    out = capsys.readouterr().out
    data = json.loads(next(l for l in out.splitlines() if l.startswith("{")))
    assert data["status"] == "error"
    assert load_calls == []  # model NEVER loaded — RULE #1 guard
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_embeddings_benchmark.py -q -k worker`
Expected: FAIL (module `wmx_suite.probe_worker_embeddings` does not exist).

- [ ] **Step 3: Implement the worker**

Create `wmx_suite/probe_worker_embeddings.py`:
```python
"""Single isolated memory measurement for one ModernBERT embeddings (batch, seq) cell.

Run as a subprocess — one fresh process per cell — so wired-memory residue from a previous
cell never contaminates the high-water reading. Prints one JSON line.

Usage:
    python -m wmx_suite.probe_worker_embeddings --model <id> --batch B --seq S \
        [--repeats N] [--margin GB]

Import convention: modules are imported (not their members) so tests can patch
`mlx_embeddings.load`, `mx.*`, and `system.*` on these shared module objects.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import threading
import time

import mlx.core as mx

from . import system

# Headroom (GB) the worker reserves for loading the model weights before allocating
# activations. Conservative for a ModernBERT-base bf16 (~0.3 GB weights + overhead).
MODEL_WEIGHT_EST_GB = 0.6

try:
    import mlx_embeddings
except ImportError:  # pragma: no cover - exercised via patched import in tests
    mlx_embeddings = None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--batch", type=int, required=True)
    ap.add_argument("--seq", type=int, required=True)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--margin", type=float, default=2.0)
    args = ap.parse_args()

    limits = system.read_limits()
    threshold = limits.safe_threshold_gb(args.margin)
    if limits.wired_now_gb + MODEL_WEIGHT_EST_GB >= threshold:
        print(json.dumps({
            "status": "error",
            "note": (f"Pre-flight aborted: wired {limits.wired_now_gb:.2f} GB + "
                     f"weight headroom {MODEL_WEIGHT_EST_GB} GB >= threshold "
                     f"{threshold:.2f} GB. Model not loaded."),
        }), flush=True)
        sys.exit(0)

    if mlx_embeddings is None:
        print(json.dumps({
            "status": "error",
            "note": "mlx-embeddings not installed; add it to run the embeddings benchmark.",
        }), flush=True)
        sys.exit(1)

    model, _tokenizer = mlx_embeddings.load(args.model)
    embed_dtype = model.model.embeddings.tok_embeddings.weight.dtype
    input_ids = mx.zeros((args.batch, args.seq), dtype=mx.int32)
    attention_mask = mx.ones((args.batch, args.seq), dtype=embed_dtype)

    # Background OS-wired sampler: MLX may free per-layer buffers mid-forward, so a single
    # post-eval read can miss the true high-water that gates LARGER cells.
    hi = [0.0]
    stop = [False]

    def sampler():
        while not stop[0]:
            hi[0] = max(hi[0], system.wired_gb())
            time.sleep(0.05)

    t = threading.Thread(target=sampler, daemon=True)
    t.start()

    # Warmup (compile Metal graphs); not measured.
    out = model(input_ids, attention_mask=attention_mask)
    mx.eval(out.last_hidden_state)
    mx.clear_cache()

    compute_times = []
    peaks = []
    for _ in range(max(1, args.repeats)):
        mx.clear_cache()
        mx.reset_peak_memory()
        t0 = time.perf_counter()
        out = model(input_ids, attention_mask=attention_mask)
        mx.eval(out.last_hidden_state)
        compute_times.append(time.perf_counter() - t0)
        peaks.append(mx.get_peak_memory() / 1e9)

    stop[0] = True

    compute_time = statistics.median(compute_times)
    throughput_tps = (args.batch * args.seq) / compute_time if compute_time > 0 else 0.0
    print(json.dumps({
        "status": "rung_done",
        "batch": args.batch,
        "seq": args.seq,
        "os_wired_gb": round(hi[0], 3),            # sampler high-water (max)
        "peak_gb": round(max(peaks), 3),           # max across repeats (conservative)
        "compute_time": round(compute_time, 4),
        "throughput_tps": round(throughput_tps, 2),
        "latency_ms": round(compute_time * 1000, 3),
    }), flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_embeddings_benchmark.py -q -k worker`
Expected: PASS (both worker tests).

- [ ] **Step 5: Mutation check — confirm the pre-flight guard is real**

Temporarily change `limits.wired_now_gb + MODEL_WEIGHT_EST_GB >= threshold` to
`limits.wired_now_gb >= threshold` and rerun `test_worker_preflight_refusal_never_loads`.
Expected: FAIL (15.0 < 15.18 so the model would load). Then revert the change and confirm PASS.

- [ ] **Step 6: Commit**

```bash
git add wmx_suite/probe_worker_embeddings.py tests/test_embeddings_benchmark.py
git commit -m "feat(embeddings): single-cell worker with sampler + pre-flight gate (#14)"
```

---

## Task 4: The sweep orchestrator

**Files:**
- Create: `wmx_suite/embeddings_probe.py`
- Test: `tests/test_embeddings_benchmark.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_embeddings_benchmark.py`:
```python
def test_fit_recovers_known_coeffs():
    from wmx_suite import embeddings_probe as ep
    # delta = 1e-6*x1 + 2e-8*x2 exactly
    pts = []
    for b, s in [(1, 128), (1, 256), (1, 512), (2, 512), (4, 256)]:
        x1, x2 = b * s, b * s * s
        pts.append((x1, x2, 1e-6 * x1 + 2e-8 * x2))
    a, b = ep._fit_ab(pts)
    assert a == pytest.approx(1e-6, rel=1e-3)
    assert b == pytest.approx(2e-8, rel=1e-3)


def test_cold_start_gate_uses_nonzero_model_base(monkeypatch):
    """With no measured cells and high host pressure, the first cell must be gated by the
    non-zero model_base seed (a model_base=0 impl would wrongly spawn it)."""
    from wmx_suite import embeddings_probe as ep
    spawned = []
    # live_base high enough that live + seed >= threshold, but live alone < threshold
    monkeypatch.setattr(ep, "sample_settled_baseline",
                        lambda: 15.18 - ep.MODEL_BASE_SEED_GB + 0.01)
    monkeypatch.setattr(ep, "read_limits",
                        lambda: SimpleNamespace(safe_threshold_gb=lambda m=2.0: 15.18,
                                                wall_gb=17.18, wired_now_gb=3.0))
    monkeypatch.setattr(ep, "_run_cell",
                        lambda *a, **k: spawned.append(a) or {"status": "rung_done"})
    events = []
    summary = ep.sweep(con=None, run_id=1, model="m",
                       batches=[1], seqs=[128], repeats=1, margin_gb=2.0,
                       on_event=events.append, persist=False)
    assert spawned == []  # never spawned — RULE #1 cold-start guard
    assert any(e["event"] == "preflight_abort" for e in events)


def test_predictive_skip_does_not_spawn_unsafe_cell(monkeypatch):
    from wmx_suite import embeddings_probe as ep
    spawned = []

    def fake_run_cell(py, model, batch, seq, repeats, margin):
        spawned.append((batch, seq))
        # Return a measured delta that makes the fitted b large enough that the NEXT
        # seq is predicted to breach. os_wired - baseline = delta.
        x2 = batch * seq * seq
        delta = 5.0e-7 * x2  # strong quadratic signal
        return {"status": "rung_done", "batch": batch, "seq": seq,
                "os_wired_gb": 3.0 + delta, "peak_gb": 1.0,
                "throughput_tps": 1.0, "latency_ms": 1.0}

    monkeypatch.setattr(ep, "_run_cell", fake_run_cell)
    monkeypatch.setattr(ep, "sample_settled_baseline", lambda: 3.0)
    monkeypatch.setattr(ep, "read_limits",
                        lambda: SimpleNamespace(safe_threshold_gb=lambda m=2.0: 15.18,
                                                wall_gb=17.18, wired_now_gb=3.0))
    events = []
    ep.sweep(con=None, run_id=1, model="m",
             batches=[1], seqs=[512, 1024, 2048, 4096, 8192], repeats=1, margin_gb=2.0,
             on_event=events.append, persist=False)
    # Some large seq must have been skipped, and the largest spawned seq must be safe.
    skipped = [e for e in events if e["event"] == "row_skipped"]
    assert skipped, "expected at least one predictive skip"
    # The skipped seq was never spawned
    skipped_seqs = {e["seq"] for e in skipped}
    assert not (skipped_seqs & {s for (_, s) in spawned})


def test_monotonic_pruning_skips_larger_batch_same_seq(monkeypatch):
    from wmx_suite import embeddings_probe as ep
    spawned = []

    def fake_run_cell(py, model, batch, seq, repeats, margin):
        spawned.append((batch, seq))
        x2 = batch * seq * seq
        return {"status": "rung_done", "batch": batch, "seq": seq,
                "os_wired_gb": 3.0 + 5.0e-7 * x2, "peak_gb": 1.0,
                "throughput_tps": 1.0, "latency_ms": 1.0}

    monkeypatch.setattr(ep, "_run_cell", fake_run_cell)
    monkeypatch.setattr(ep, "sample_settled_baseline", lambda: 3.0)
    monkeypatch.setattr(ep, "read_limits",
                        lambda: SimpleNamespace(safe_threshold_gb=lambda m=2.0: 15.18,
                                                wall_gb=17.18, wired_now_gb=3.0))
    events = []
    ep.sweep(con=None, run_id=1, model="m",
             batches=[1, 32], seqs=[2048, 8192], repeats=1, margin_gb=2.0,
             on_event=events.append, persist=False)
    # If (1, 8192) is unsafe, (32, 8192) must never be spawned.
    if (1, 8192) not in spawned:
        assert (32, 8192) not in spawned
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_embeddings_benchmark.py -q -k "fit or cold_start or predictive or monotonic"`
Expected: FAIL (module `wmx_suite.embeddings_probe` does not exist).

- [ ] **Step 3: Implement the orchestrator**

Create `wmx_suite/embeddings_probe.py`:
```python
"""Safe 2D (batch x seq_len) OS-wired memory-surface benchmark for encoder embeddings.

Mirrors probe.py's "extrapolate, never crash" approach for a NON-causal encoder:
  * one isolated subprocess per grid cell (probe_worker_embeddings)
  * the predictive gate runs in THIS (parent) process BEFORE spawning each cell
  * per-row ramp; skip the rest of a row once a cell is predicted to breach the wall
  * the gate trusts a fit of REAL measured high-water peaks; a conservative analytic prior
    is used only for cold start (before MIN_FIT_POINTS cells exist)
"""
from __future__ import annotations

import json
import subprocess
import sys

from . import config, db
from .system import read_limits, sample_settled_baseline

DEFAULT_BATCHES = [1, 2, 4, 8, 16, 32]
DEFAULT_SEQS = [128, 256, 512, 1024, 2048, 4096, 8192]

MIN_FIT_POINTS = 3
PRED_SAFETY = 1.25
MODEL_BASE_SEED_GB = 1.0  # weight-residency seed so cold-start/pre-flight aren't zero

# ModernBERT-base architecture constants (verified from config.json).
NUM_LAYERS = 22
HIDDEN_SIZE = 768
NUM_HEADS = 12

# Cold-start OVER-estimate (sum over all layers as if global): a safe upper bound that only
# ever gates the first tiny cells, since the sweep ramps seq from the smallest value.
A_COLD = NUM_LAYERS * HIDDEN_SIZE * 2 / 1e9   # GB per (batch*seq)
B_COLD = NUM_LAYERS * NUM_HEADS * 2 / 1e9     # GB per (batch*seq^2)

# Physical one-layer FLOOR (lower bound: >=1 layer's attention + residual resident at peak).
A_FLOOR = HIDDEN_SIZE * 2 / 1e9
B_FLOOR = NUM_HEADS * 2 / 1e9


def _default_event(_event: dict) -> None:
    pass


def _fit_ab(points: list[tuple[float, float, float]]) -> tuple[float, float]:
    """Least-squares delta = a*x1 + b*x2 through the origin (no intercept).

    points: (x1=batch*seq, x2=batch*seq^2, delta). Returns (a, b); (0,0) if singular.
    """
    s11 = s12 = s22 = sd1 = sd2 = 0.0
    for x1, x2, d in points:
        s11 += x1 * x1
        s12 += x1 * x2
        s22 += x2 * x2
        sd1 += x1 * d
        sd2 += x2 * d
    det = s11 * s22 - s12 * s12
    if abs(det) < 1e-30:
        return 0.0, 0.0
    a = (sd1 * s22 - sd2 * s12) / det
    b = (s11 * sd2 - s12 * sd1) / det
    return a, b


def _coeffs(points: list[tuple[float, float, float]]) -> tuple[float, float]:
    """Gate coefficients (a, b): cold over-estimate before MIN_FIT_POINTS, else fit
    clamped to the one-layer physical floor."""
    if len(points) < MIN_FIT_POINTS:
        return A_COLD, B_COLD
    a_fit, b_fit = _fit_ab(points)
    return max(A_FLOOR, max(0.0, a_fit)), max(B_FLOOR, max(0.0, b_fit))


def _run_cell(py: str, model: str, batch: int, seq: int, repeats: int, margin: float) -> dict:
    cmd = [py, "-m", "wmx_suite.probe_worker_embeddings",
           "--model", model, "--batch", str(batch), "--seq", str(seq),
           "--repeats", str(repeats), "--margin", str(margin)]
    out = subprocess.run(cmd, capture_output=True, text=True)
    line = next((l for l in out.stdout.splitlines() if l.startswith("{")), None)
    if not line:
        return {"status": "error",
                "note": f"no result (stderr: {out.stderr.strip()[-200:]})"}
    return json.loads(line)


def sweep(con, run_id: int, model: str, batches=None, seqs=None, repeats: int = 3,
          margin_gb: float | None = None, *, on_event=None, persist: bool = True) -> dict:
    batches = batches or DEFAULT_BATCHES
    seqs = sorted(seqs or DEFAULT_SEQS)
    margin = config.margin_gb(margin_gb)
    on_event = on_event or _default_event
    py = sys.executable

    limits = read_limits()
    threshold = limits.safe_threshold_gb(margin)

    points: list[tuple[float, float, float]] = []  # (x1, x2, delta)
    model_base = MODEL_BASE_SEED_GB
    smallest_unsafe_seq: float = float("inf")  # for monotonic pruning across batches
    n_measured = 0
    n_skipped = 0

    # Pre-flight: if even a tiny cell can't fit given current pressure, abort the sweep.
    live = sample_settled_baseline()
    if live + model_base >= threshold:
        on_event({"event": "preflight_abort",
                  "note": (f"host pressure {live:.2f} GB + model seed {model_base} GB "
                           f">= threshold {threshold:.2f} GB")})
        return {"model": model, "run_id": run_id, "n_cells_measured": 0,
                "n_cells_skipped": 0, "aborted": True}

    for batch in batches:
        for seq in seqs:
            # Monotonic pruning: memory grows with batch at fixed seq.
            if seq >= smallest_unsafe_seq:
                on_event({"event": "row_skipped", "batch": batch, "seq": seq,
                          "predicted_gb": None})
                n_skipped += 1
                continue

            live_base = sample_settled_baseline()
            a, b = _coeffs(points)
            x1, x2 = batch * seq, batch * seq * seq
            predicted = live_base + model_base + PRED_SAFETY * (a * x1 + b * x2)
            if predicted >= threshold:
                on_event({"event": "row_skipped", "batch": batch, "seq": seq,
                          "predicted_gb": round(predicted, 3)})
                n_skipped += 1
                smallest_unsafe_seq = min(smallest_unsafe_seq, seq)
                break  # skip the rest of this (ascending-seq) row

            result = _run_cell(py, model, batch, seq, repeats, margin)
            if result.get("status") != "rung_done":
                on_event({"event": "error", "batch": batch, "seq": seq,
                          "note": result.get("note", "worker error")})
                return {"model": model, "run_id": run_id,
                        "n_cells_measured": n_measured, "n_cells_skipped": n_skipped,
                        "error": result.get("note")}

            # Refit on the measured delta over this cell's own baseline.
            delta = max(0.0, result["os_wired_gb"] - live_base)
            points.append((x1, x2, delta))
            if n_measured == 0:
                # Seed model_base from the smallest cell's residency (delta is ~0 there,
                # so use the absolute minus live as a floor; keep the non-zero seed).
                model_base = max(MODEL_BASE_SEED_GB,
                                 result["os_wired_gb"] - live_base - (a * x1 + b * x2))
            n_measured += 1

            if persist and con is not None:
                db.add_embeddings_measurement(
                    con, run_id, batch_size=batch, seq_len=seq,
                    os_wired_gb=result["os_wired_gb"], peak_gb=result["peak_gb"],
                    throughput_tps=result["throughput_tps"], latency_ms=result["latency_ms"],
                )
            on_event({"event": "cell_done", "batch": batch, "seq": seq,
                      "os_wired_gb": result["os_wired_gb"], "peak_gb": result["peak_gb"],
                      "throughput_tps": result["throughput_tps"],
                      "latency_ms": result["latency_ms"]})

    return {"model": model, "run_id": run_id, "n_cells_measured": n_measured,
            "n_cells_skipped": n_skipped}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_embeddings_benchmark.py -q -k "fit or cold_start or predictive or monotonic"`
Expected: PASS (all four orchestrator tests).

- [ ] **Step 5: Mutation check — confirm the predictive gate is real**

Temporarily change the gate `if predicted >= threshold:` to `if False:` and rerun
`test_predictive_skip_does_not_spawn_unsafe_cell` and `test_cold_start_gate_uses_nonzero_model_base`.
Expected: both FAIL (unsafe cells get spawned / no preflight abort). Revert and confirm PASS.

- [ ] **Step 6: Commit**

```bash
git add wmx_suite/embeddings_probe.py tests/test_embeddings_benchmark.py
git commit -m "feat(embeddings): sweep orchestrator with predictive-skip gate + fit (#14)"
```

---

## Task 5: CLI command

**Files:**
- Modify: `wmx_suite/cli.py` — add `cmd_benchmark_embeddings` (near the other `cmd_benchmark_*`) and a subparser (near line ~1172).
- Test: `tests/test_embeddings_benchmark.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_embeddings_benchmark.py`:
```python
def test_cmd_benchmark_embeddings_persists_and_renders(monkeypatch, tmp_path, capsys):
    from wmx_suite import cli, db, embeddings_probe

    monkeypatch.setattr(db, "DB_PATH", tmp_path / "suite.db")

    # Fake sweep: emit canned events AND persist two measurements like the real one would.
    def fake_sweep(con, run_id, model, batches, seqs, repeats, margin_gb=None,
                   *, on_event=None, persist=True):
        for (bsz, seq) in [(1, 128), (2, 256)]:
            db.add_embeddings_measurement(con, run_id, batch_size=bsz, seq_len=seq,
                                          os_wired_gb=4.0, peak_gb=2.0,
                                          throughput_tps=100.0, latency_ms=5.0)
            on_event({"event": "cell_done", "batch": bsz, "seq": seq,
                      "os_wired_gb": 4.0, "peak_gb": 2.0,
                      "throughput_tps": 100.0, "latency_ms": 5.0})
        on_event({"event": "row_skipped", "batch": 2, "seq": 8192, "predicted_gb": 99.0})
        return {"model": model, "run_id": run_id, "n_cells_measured": 2, "n_cells_skipped": 1}

    monkeypatch.setattr(embeddings_probe, "sweep", fake_sweep)

    args = SimpleNamespace(model="mlx-community/test", batches="1,2", seqs="128,256,8192",
                           repeats=1, margin=None)
    cli.cmd_benchmark_embeddings(args)

    out = capsys.readouterr().out
    assert "SKIP" in out  # skipped cell rendered

    con = db.connect()
    latest = db.get_latest_embeddings_run(con)
    assert latest is not None
    rows = db.get_embeddings_measurements(con, latest["id"])
    assert len(rows) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_embeddings_benchmark.py::test_cmd_benchmark_embeddings_persists_and_renders -q`
Expected: FAIL (`cli` has no attribute `cmd_benchmark_embeddings`).

- [ ] **Step 3: Implement the CLI command**

In `wmx_suite/cli.py`, add this function next to the other `cmd_benchmark_*` functions:
```python
def cmd_benchmark_embeddings(args):
    """Run the ModernBERT embeddings 2D (batch x seq) memory-surface sweep and log it."""
    from . import embeddings_probe

    margin_val = _configured_margin(args.margin)
    batches = [int(x) for x in str(args.batches).split(",") if x.strip()]
    seqs = [int(x) for x in str(args.seqs).split(",") if x.strip()]

    print("============================================================")
    print(" ModernBERT Embeddings Memory Benchmark (batch x seq)")
    print("============================================================")
    print(f"  Model   : {args.model}")
    print(f"  Batches : {batches}")
    print(f"  Seqs    : {seqs}")
    print(f"  Repeats : {args.repeats}")
    print(f"  Margin  : {margin_val} GB")
    print("------------------------------------------------------------")

    import mlx.core as mx
    mlx_version = mx.__version__

    con = db.connect()
    run_id = db.start_embeddings_run(con, args.model, mlx_version)

    aborted = {"flag": False}

    def render(event):
        ev = event.get("event")
        if ev == "preflight_abort":
            print(f"  PRE-FLIGHT ABORT: {event['note']}")
            aborted["flag"] = True
        elif ev == "error":
            print(f"  ERROR at batch {event.get('batch')} seq {event.get('seq')}: "
                  f"{event.get('note')}")
            aborted["flag"] = True
        elif ev == "row_skipped":
            pred = event.get("predicted_gb")
            pred_s = f"{pred:.2f} GB" if pred is not None else "pruned"
            print(f"  SKIP  batch={event['batch']:<3} seq={event['seq']:<6} "
                  f"(predicted {pred_s})")
        elif ev == "cell_done":
            print(f"  OK    batch={event['batch']:<3} seq={event['seq']:<6} "
                  f"wired={event['os_wired_gb']:.2f}GB  peak={event['peak_gb']:.2f}GB  "
                  f"{event['throughput_tps']:.0f} tok/s  {event['latency_ms']:.1f}ms")

    summary = embeddings_probe.sweep(
        con, run_id, args.model, batches=batches, seqs=seqs,
        repeats=args.repeats, margin_gb=margin_val, on_event=render,
    )

    if aborted["flag"]:
        sys.exit(1)

    print("============================================================")
    print(f"  Done. Measured {summary['n_cells_measured']} cells, "
          f"skipped {summary['n_cells_skipped']}. Saved as Run ID: {run_id}")
    print("============================================================")
```

- [ ] **Step 4: Register the subparser**

In `wmx_suite/cli.py`, after the `benchmark-kokoro` subparser block (line ~1172), add:
```python
    p = sub.add_parser("benchmark-embeddings",
                       help="Benchmark ModernBERT embeddings batch x seq memory scaling")
    p.add_argument("--model", default="mlx-community/nomicai-modernbert-embed-base-bf16",
                   help="HuggingFace model ID or path")
    p.add_argument("--batches", default="1,2,4,8,16,32",
                   help="Comma-separated batch sizes to sweep")
    p.add_argument("--seqs", default="128,256,512,1024,2048,4096,8192",
                   help="Comma-separated sequence lengths to sweep")
    p.add_argument("--repeats", type=int, default=3,
                   help="Forward passes per cell (median timing, max memory)")
    p.add_argument("--margin", default=None,
                   help="safety cushion in GB (overrides WMX_SUITE_MARGIN_GB)")
    p.set_defaults(func=cmd_benchmark_embeddings)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_embeddings_benchmark.py::test_cmd_benchmark_embeddings_persists_and_renders -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add wmx_suite/cli.py tests/test_embeddings_benchmark.py
git commit -m "feat(cli): benchmark-embeddings command (#14)"
```

---

## Task 6: Web dashboard

**Files:**
- Modify: `wmx_suite/web/app.py` — add two routes inside `create_app()` (near the kokoro routes).
- Create: `wmx_suite/web/templates/embeddings_dashboard.html`, `wmx_suite/web/templates/embeddings_run.html`.
- Modify: `wmx_suite/web/templates/base.html` — add nav entry (after the Kokoro Baseline `<li>`, line ~61).
- Test: `tests/test_web.py` (append)

- [ ] **Step 1: Write the failing test**

First inspect how `tests/test_web.py` builds the client (look for `create_app` and DB setup), then append a test that mirrors that fixture. Example (adjust the app/client construction to match the file's existing pattern):
```python
def test_embeddings_routes(monkeypatch, tmp_path):
    from wmx_suite import db
    from wmx_suite.web.app import create_app

    monkeypatch.setattr(db, "DB_PATH", tmp_path / "suite.db")
    con = db.connect()
    run_id = db.start_embeddings_run(con, "mlx-community/test", "0.31.2")
    db.add_embeddings_measurement(con, run_id, batch_size=1, seq_len=128,
                                  os_wired_gb=4.0, peak_gb=2.0,
                                  throughput_tps=100.0, latency_ms=5.0)

    app = create_app()
    app.config.update(TESTING=True)
    client = app.test_client()

    assert client.get("/embeddings").status_code == 200
    assert client.get(f"/embeddings/run/{run_id}").status_code == 200
    assert client.get("/embeddings/run/99999").status_code == 404
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_web.py::test_embeddings_routes -q`
Expected: FAIL (404 on `/embeddings` — route not registered).

- [ ] **Step 3: Add the routes**

In `wmx_suite/web/app.py`, inside `create_app()` near the other kokoro routes, add:
```python
    @app.route("/embeddings")
    def embeddings_dashboard():
        con = get_db()
        runs = db.get_all_embeddings_runs(con)
        limits = system.read_limits()
        decorated = []
        for r in runs:
            m = db.get_embeddings_measurements(con, r["id"])
            decorated.append({
                "id": r["id"],
                "model_id": r["model_id"],
                "mlx_version": r["mlx_version"],
                "created_at": r["created_at"],
                "n_cells": len(m),
                "max_wired_gb": round(max((x["os_wired_gb"] for x in m), default=0.0), 2),
                "max_tps": round(max((x["throughput_tps"] for x in m), default=0.0), 0),
            })
        return render_template("embeddings_dashboard.html", runs=decorated, limits=limits)

    @app.route("/embeddings/run/<int:run_id>")
    def embeddings_run_detail(run_id):
        con = get_db()
        runs = db.get_all_embeddings_runs(con)
        run = next((r for r in runs if r["id"] == run_id), None)
        if not run:
            abort(404, description=f"Embeddings Run not found: {run_id}")
        measurements = db.get_embeddings_measurements(con, run_id)
        batches = sorted({m["batch_size"] for m in measurements})
        seqs = sorted({m["seq_len"] for m in measurements})
        grid = {(m["batch_size"], m["seq_len"]): m for m in measurements}
        limits = system.read_limits()
        return render_template("embeddings_run.html", run=run, measurements=measurements,
                               batches=batches, seqs=seqs, grid=grid, limits=limits)
```

- [ ] **Step 4: Create the dashboard template**

Create `wmx_suite/web/templates/embeddings_dashboard.html`:
```html
{% extends "base.html" %}
{% block content %}
<h1>ModernBERT Embeddings — Memory Surface</h1>
<p>Each run sweeps a batch × sequence-length grid, recording OS-wired memory and throughput.
Cells predicted to breach the wall are skipped, never run.</p>
{% if runs %}
<table class="data-table">
  <thead>
    <tr><th>Run</th><th>Model</th><th>MLX</th><th>When</th>
        <th>Cells</th><th>Max OS-wired (GB)</th><th>Max tok/s</th></tr>
  </thead>
  <tbody>
    {% for r in runs %}
    <tr>
      <td><a href="{{ url_for('embeddings_run_detail', run_id=r.id) }}">#{{ r.id }}</a></td>
      <td>{{ r.model_id }}</td>
      <td>{{ r.mlx_version }}</td>
      <td>{{ r.created_at }}</td>
      <td>{{ r.n_cells }}</td>
      <td>{{ r.max_wired_gb }}</td>
      <td>{{ r.max_tps }}</td>
    </tr>
    {% endfor %}
  </tbody>
</table>
{% else %}
<p>No embeddings runs yet. Run <code>uv run wmx-suite benchmark-embeddings</code>.</p>
{% endif %}
{% endblock %}
```

- [ ] **Step 5: Create the run-detail template**

Create `wmx_suite/web/templates/embeddings_run.html`:
```html
{% extends "base.html" %}
{% block content %}
<h1>Embeddings Run #{{ run.id }}</h1>
<p><strong>Model:</strong> {{ run.model_id }} &middot; <strong>MLX:</strong> {{ run.mlx_version }}
   &middot; <strong>When:</strong> {{ run.created_at }}</p>
<p>Crash wall: {{ "%.2f"|format(limits.wall_gb) }} GB. Blank cells were not reached;
   skipped cells were predicted to breach.</p>
<h2>OS-wired memory (GB) — rows = batch, cols = seq</h2>
<table class="data-table">
  <thead>
    <tr><th>batch \ seq</th>{% for s in seqs %}<th>{{ s }}</th>{% endfor %}</tr>
  </thead>
  <tbody>
    {% for b in batches %}
    <tr>
      <th>{{ b }}</th>
      {% for s in seqs %}
        {% set cell = grid.get((b, s)) %}
        {% if cell %}<td>{{ "%.2f"|format(cell.os_wired_gb) }}</td>
        {% else %}<td>&mdash;</td>{% endif %}
      {% endfor %}
    </tr>
    {% endfor %}
  </tbody>
</table>
<h2>All measurements</h2>
<table class="data-table">
  <thead><tr><th>batch</th><th>seq</th><th>OS-wired (GB)</th><th>MLX peak (GB)</th>
             <th>tok/s</th><th>latency (ms)</th></tr></thead>
  <tbody>
    {% for m in measurements %}
    <tr><td>{{ m.batch_size }}</td><td>{{ m.seq_len }}</td>
        <td>{{ "%.2f"|format(m.os_wired_gb) }}</td>
        <td>{{ "%.2f"|format(m.peak_gb) }}</td>
        <td>{{ "%.0f"|format(m.throughput_tps) }}</td>
        <td>{{ "%.1f"|format(m.latency_ms) }}</td></tr>
    {% endfor %}
  </tbody>
</table>
{% endblock %}
```

- [ ] **Step 6: Add the nav entry**

In `wmx_suite/web/templates/base.html`, after the Kokoro Baseline `<li>` (line ~61), add:
```html
            <li class="nav-item {% if request.path.startswith('/embeddings') %}active{% endif %}">
                <a href="{{ url_for('embeddings_dashboard') }}">
                    <span>🧬</span> Embeddings
                </a>
            </li>
```

- [ ] **Step 7: Run test to verify it passes**

Run: `uv run pytest tests/test_web.py::test_embeddings_routes -q`
Expected: PASS. (If `data-table` class or block name differs from the codebase, adjust the templates to match the existing template conventions — check `kokoro_cache_run.html`.)

- [ ] **Step 8: Commit**

```bash
git add wmx_suite/web/app.py wmx_suite/web/templates/embeddings_dashboard.html wmx_suite/web/templates/embeddings_run.html wmx_suite/web/templates/base.html tests/test_web.py
git commit -m "feat(web): embeddings dashboard + run-detail grid (#14)"
```

---

## Task 7: Full verification + issue close-out

**Files:** none (verification only)

- [ ] **Step 1: Run the full suite**

Run: `uv run pytest -q`
Expected: all tests pass (existing + new embeddings tests).

- [ ] **Step 2: Compile-check**

Run: `uv run python -m compileall -q wmx_suite tests`
Expected: clean (no output, exit 0).

- [ ] **Step 3: Smoke-check the CLI wiring (no model load)**

Run: `uv run wmx-suite benchmark-embeddings --help`
Expected: help text listing `--model`, `--batches`, `--seqs`, `--repeats`, `--margin`.

- [ ] **Step 4: Commit any final fixups, then update the issue**

If everything passes, the feature is complete. Reference issue #14 in the final commit and post a summary comment (do NOT auto-close until the user confirms a real hardware run, since these tests are hardware-free):
```bash
gh issue comment 14 --body "Implemented the ModernBERT embeddings batch×seq memory benchmark: single-cell worker with background OS-wired sampler + pre-flight gate, parent orchestrator with predictive-skip gate (fit-governed, analytic prior cold-start only), DB tables, \`benchmark-embeddings\` CLI, and web dashboard. All tests hardware-free and passing. Pending a real on-device sweep to validate the memory surface."
```

---

## Notes for the implementer

- **Real hardware run is separate.** Every test here is hardware-free by design. An actual `uv run wmx-suite benchmark-embeddings` (which loads ModernBERT and runs the grid) must only be done by/with the user per RULE #1; the worker + orchestrator gates are built to make it safe, but the first live run should be observed.
- **`_fit_ab` numerics:** features are raw counts (`seq²` up to ~6.7e7); the normal-equation sums are large but well within float range. The `det < 1e-30` guard handles the degenerate single-point / collinear case.
- **`model_base` seed for non-base models:** `MODEL_BASE_SEED_GB = 1.0` covers ModernBERT-base. A much larger `--model` is not weight-estimated up front; the worker's `MODEL_WEIGHT_EST_GB` pre-flight and the first measured cell (which replaces the seed) are the guards. This is a documented limitation, acceptable for the default base model.
