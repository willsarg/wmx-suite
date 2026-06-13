# `wmx-suite calibrate` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `wmx-suite calibrate` command that measures this machine's real cold-start `FIXED_OVERHEAD_GB` once and stores it in a per-machine profile, so the pre-flight base estimate for uncharacterized models is trustworthy on any Apple Silicon SKU.

**Architecture:** A new `profiles.py` module owns the machine key, a `system_profiles` SQLite table, and a single `cold_start_constants(con)` resolver that the two estimators (`probe.estimate_base_gb`, `launcher.plan`) call. `calibrate` reuses the existing safe-probe primitives (`_measure_rung`, `_linfit`) on the smallest cached causal model, fits the base intercept at context→0, and derives overhead as `intercept − 1.05·weights`, floored at the current default so calibration can only tighten the estimate. `run`/`health`/`system` surface a warning when no profile matches.

**Tech Stack:** Python 3, SQLite (`wmx_suite/db.py`), MLX (`mlx.core`), pytest. Run everything with `uv run`.

**Spec:** `docs/superpowers/specs/2026-06-13-calibrate-design.md`

---

## File structure

- **Create** `wmx_suite/profiles.py` — defaults, `machine_key()`, `cold_start_constants(con)`.
- **Create** `tests/test_profiles.py` — unit tests for the above.
- **Modify** `wmx_suite/system.py` — add `macos_major()`.
- **Modify** `wmx_suite/db.py` — add `system_profiles` table to `SCHEMA`; add `upsert_profile`/`get_profile`.
- **Modify** `wmx_suite/probe.py` — `estimate_base_gb` reads `cold_start_constants`; add `calibrate()` + `_pick_calibration_model()`; defaults sourced from `profiles`.
- **Modify** `wmx_suite/launcher.py` — `plan()` reads `cold_start_constants`, sets `p["cold_start_profile"]`; drop the `probe` constant import.
- **Modify** `wmx_suite/cli.py` — add `cmd_calibrate` + subparser; `run` warning; `health` banner; `system` profile line; import `profiles`.
- **Modify** tests `tests/test_system.py`, `tests/test_db.py`, `tests/test_probe_math.py`, `tests/test_launcher.py`, `tests/test_cli.py`.
- **Modify** `README.md`, `AGENTS.md` — document `calibrate`.

Build order is bottom-up so each task's dependencies already exist: system → db → profiles → estimators → calibrate → cli → docs.

---

### Task 1: `system.macos_major()`

**Files:**
- Modify: `wmx_suite/system.py`
- Test: `tests/test_system.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_system.py`:
```python
from wmx_suite import system


def test_macos_major_parses_version(monkeypatch):
    monkeypatch.setattr("platform.mac_ver", lambda: ("15.7.4", ("", "", ""), "arm64"))
    assert system.macos_major() == 15


def test_macos_major_returns_zero_when_undetectable(monkeypatch):
    monkeypatch.setattr("platform.mac_ver", lambda: ("", ("", "", ""), ""))
    assert system.macos_major() == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_system.py -k macos_major -v`
Expected: FAIL with `AttributeError: module 'wmx_suite.system' has no attribute 'macos_major'`.

- [ ] **Step 3: Implement `macos_major`**

Add to `wmx_suite/system.py` (after the imports / near `swap_free_gb`):
```python
def macos_major() -> int:
    """Major macOS version (e.g. 15), or 0 if undetectable.

    Part of the per-machine profile key: a major OS bump shifts the ambient
    wired baseline enough to invalidate a stored cold-start overhead.
    """
    import platform
    try:
        ver = platform.mac_ver()[0]
        return int(ver.split(".")[0]) if ver else 0
    except (ValueError, IndexError):
        return 0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_system.py -k macos_major -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add wmx_suite/system.py tests/test_system.py
git commit -m "feat(system): add macos_major() for the profile key"
```

---

### Task 2: `system_profiles` table + `upsert_profile`/`get_profile`

**Files:**
- Modify: `wmx_suite/db.py` (SCHEMA string near line 11–186; accessors after `upsert_model` ~line 217)
- Test: `tests/test_db.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_db.py`:
```python
def test_profile_roundtrip_and_upsert(monkeypatch, tmp_path):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "suite.db")
    con = db.connect()
    key = ("Apple M4 Pro", 25769803776, 15)

    assert db.get_profile(con, key) is None

    db.upsert_profile(
        con, key,
        resident_factor=1.05, fixed_overhead_gb=1.0,
        model_id="mlx-community/tiny", n_points=2, mlx_version="9.9",
    )
    row = db.get_profile(con, key)
    assert row["device_name"] == "Apple M4 Pro"
    assert row["total_ram_bytes"] == 25769803776
    assert row["macos_major"] == 15
    assert row["fixed_overhead_gb"] == 1.0
    assert row["model_id"] == "mlx-community/tiny"
    assert row["calibrated_at"]  # set by upsert

    # upsert replaces in place (same key -> one row, updated value)
    db.upsert_profile(
        con, key,
        resident_factor=1.05, fixed_overhead_gb=1.7,
        model_id="mlx-community/tiny", n_points=2, mlx_version="9.9",
    )
    assert db.get_profile(con, key)["fixed_overhead_gb"] == 1.7
    count = con.execute("SELECT COUNT(*) FROM system_profiles").fetchone()[0]
    assert count == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_db.py -k profile_roundtrip -v`
Expected: FAIL — `sqlite3.OperationalError: no such table: system_profiles` (or `AttributeError` on `get_profile`).

- [ ] **Step 3: Add the table to `SCHEMA`**

In `wmx_suite/db.py`, append this block inside the `SCHEMA = """ ... """` string (after the last `kokoro_baseline_measurements` table, before the closing `"""`):
```sql

CREATE TABLE IF NOT EXISTS system_profiles (
    device_name       TEXT NOT NULL,
    total_ram_bytes   INTEGER NOT NULL,
    macos_major       INTEGER NOT NULL,
    resident_factor   REAL NOT NULL,
    fixed_overhead_gb REAL NOT NULL,
    model_id          TEXT,
    n_points          INTEGER,
    mlx_version       TEXT,
    calibrated_at     TEXT,
    PRIMARY KEY (device_name, total_ram_bytes, macos_major)
);
```

- [ ] **Step 4: Add the accessors**

In `wmx_suite/db.py`, after `upsert_model` (around line 217), add:
```python
def upsert_profile(con: sqlite3.Connection, key: tuple[str, int, int], *,
                   resident_factor: float, fixed_overhead_gb: float,
                   model_id: str | None, n_points: int | None,
                   mlx_version: str | None) -> None:
    device_name, total_ram_bytes, macos_major = key
    con.execute(
        "INSERT INTO system_profiles (device_name, total_ram_bytes, macos_major, "
        "resident_factor, fixed_overhead_gb, model_id, n_points, mlx_version, calibrated_at) "
        "VALUES (?,?,?,?,?,?,?,?,?) "
        "ON CONFLICT(device_name, total_ram_bytes, macos_major) DO UPDATE SET "
        "resident_factor=excluded.resident_factor, "
        "fixed_overhead_gb=excluded.fixed_overhead_gb, model_id=excluded.model_id, "
        "n_points=excluded.n_points, mlx_version=excluded.mlx_version, "
        "calibrated_at=excluded.calibrated_at",
        (device_name, total_ram_bytes, macos_major, resident_factor, fixed_overhead_gb,
         model_id, n_points, mlx_version, _now()),
    )
    con.commit()


def get_profile(con: sqlite3.Connection, key: tuple[str, int, int]) -> dict | None:
    device_name, total_ram_bytes, macos_major = key
    row = con.execute(
        "SELECT device_name, total_ram_bytes, macos_major, resident_factor, "
        "fixed_overhead_gb, model_id, n_points, mlx_version, calibrated_at "
        "FROM system_profiles WHERE device_name=? AND total_ram_bytes=? AND macos_major=?",
        (device_name, total_ram_bytes, macos_major),
    ).fetchone()
    return dict(row) if row is not None else None
```
(`con.row_factory = sqlite3.Row` is already set in `connect()`, so `dict(row)` works.)

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_db.py -k profile_roundtrip -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add wmx_suite/db.py tests/test_db.py
git commit -m "feat(db): add system_profiles table + upsert/get accessors"
```

---

### Task 3: `profiles.py` — defaults, `machine_key`, `cold_start_constants`

**Files:**
- Create: `wmx_suite/profiles.py`
- Test: `tests/test_profiles.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_profiles.py`:
```python
from wmx_suite import db, profiles


def test_cold_start_constants_defaults_when_no_profile(monkeypatch, tmp_path):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "suite.db")
    monkeypatch.setattr(profiles, "machine_key", lambda: ("Apple M4 Pro", 1, 15))
    con = db.connect()
    factor, overhead, source = profiles.cold_start_constants(con)
    assert factor == profiles.DEFAULT_RESIDENT_FACTOR
    assert overhead == profiles.DEFAULT_FIXED_OVERHEAD_GB
    assert source == "default"


def test_cold_start_constants_uses_stored_profile(monkeypatch, tmp_path):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "suite.db")
    key = ("Apple M4 Pro", 1, 15)
    monkeypatch.setattr(profiles, "machine_key", lambda: key)
    con = db.connect()
    db.upsert_profile(con, key, resident_factor=1.05, fixed_overhead_gb=1.6,
                      model_id="m", n_points=2, mlx_version="9.9")
    factor, overhead, source = profiles.cold_start_constants(con)
    assert factor == 1.05
    assert overhead == 1.6
    assert source == "profile"


def test_machine_key_shape(monkeypatch):
    monkeypatch.setattr(profiles.system, "macos_major", lambda: 15)
    # mlx is installed; machine_key reads real device_info — just assert the shape/types.
    dev, ram, osv = profiles.machine_key()
    assert isinstance(dev, str)
    assert isinstance(ram, int) and ram >= 0
    assert osv == 15
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_profiles.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'wmx_suite.profiles'`.

- [ ] **Step 3: Create `profiles.py`**

Create `wmx_suite/profiles.py`:
```python
"""Per-machine cold-start constants for the pre-flight base estimate.

The crash wall, the ambient baseline, and per-model fits already adapt to the host
at runtime. The only M4-Pro-tuned part is the cold-start base estimate for models that
haven't been characterized yet. `wmx-suite calibrate` measures this machine's real
FIXED_OVERHEAD_GB once and stores it (keyed by chip + RAM + macOS major) so the estimate
is trustworthy on any Apple Silicon SKU. `characterize` remains the per-model mechanism.
"""
from __future__ import annotations

from . import db, system

# Loose priors, measured loosely on the M4 Pro (see probe.py). The resident factor is
# HELD FIXED (effective factor measured 0.88-1.10 across models); only the overhead is
# calibrated per machine, and never stored below this default (calibration only tightens).
DEFAULT_RESIDENT_FACTOR = 1.05
DEFAULT_FIXED_OVERHEAD_GB = 1.0


def machine_key() -> tuple[str, int, int]:
    """(device_name, total_ram_bytes, macos_major) identifying the current machine.

    total RAM is read in BYTES from mx.device_info()['memory_size'] (SystemLimits.total_gb
    is already divided by 1e9, so it can't supply a stable integer key).
    """
    import mlx.core as mx
    d = mx.device_info()
    return (str(d.get("device_name", "")), int(d.get("memory_size", 0)), system.macos_major())


def cold_start_constants(con) -> tuple[float, float, str]:
    """Return (resident_factor, fixed_overhead_gb, source).

    source == "profile" when a stored profile matches this machine, else "default".
    The factor is always the default (held fixed); only the overhead is profile-specific.
    """
    profile = db.get_profile(con, machine_key())
    if profile is not None:
        return float(profile["resident_factor"]), float(profile["fixed_overhead_gb"]), "profile"
    return DEFAULT_RESIDENT_FACTOR, DEFAULT_FIXED_OVERHEAD_GB, "default"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_profiles.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add wmx_suite/profiles.py tests/test_profiles.py
git commit -m "feat(profiles): machine key + cold_start_constants resolver"
```

---

### Task 4: `probe.estimate_base_gb` reads the profile

**Files:**
- Modify: `wmx_suite/probe.py` (constants ~33-34; `estimate_base_gb` ~73-76; caller in `characterize` ~154)
- Test: `tests/test_probe_math.py` (existing callers at lines ~71, ~76)

- [ ] **Step 1: Update the existing tests to the new signature and assert profile use**

In `tests/test_probe_math.py`, the two existing `estimate_base_gb(info, limits)` calls (around lines 71, 76) must pass a `con`. Replace those test bodies' calls and add a profile-aware test. First, read the file to see the exact `info`/`limits` fixtures it uses, then update the calls to `probe.estimate_base_gb(info, limits, con)` where `con` comes from an isolated DB. Add this new test:
```python
def test_estimate_base_gb_uses_profile_overhead(monkeypatch, tmp_path):
    from wmx_suite import db, profiles
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "suite.db")
    key = ("Apple M4 Pro", 1, 15)
    monkeypatch.setattr(profiles, "machine_key", lambda: key)
    con = db.connect()
    info = _model_info(weights_gb=4.0)          # reuse this file's helper
    limits = _limits(wired_now_gb=3.0)          # reuse this file's helper
    # default overhead 1.0:
    base_default = probe.estimate_base_gb(info, limits, con)
    # store a larger overhead, expect the estimate to rise by the delta:
    db.upsert_profile(con, key, resident_factor=1.05, fixed_overhead_gb=2.5,
                      model_id="m", n_points=2, mlx_version="9.9")
    base_profile = probe.estimate_base_gb(info, limits, con)
    assert round(base_profile - base_default, 3) == 1.5
```
NOTE: if the existing helpers are named differently (e.g. they build a `ModelInfo`/`SystemLimits` inline), mirror whatever the file already does instead of `_model_info`/`_limits`. Read the file first.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_probe_math.py -k estimate_base -v`
Expected: FAIL — `TypeError: estimate_base_gb() takes 2 positional arguments but 3 were given`.

- [ ] **Step 3: Source the defaults from `profiles` and make `estimate_base_gb` read the profile**

In `wmx_suite/probe.py`:

Add `profiles` to the existing relative import line `from . import config, db, models`:
```python
from . import config, db, models, profiles
```

Replace the constant definitions (lines ~33-34):
```python
RESIDENT_FACTOR = 1.05
FIXED_OVERHEAD_GB = 1.0
```
with aliases to the single source of truth (keeps these names available for any reader):
```python
RESIDENT_FACTOR = profiles.DEFAULT_RESIDENT_FACTOR
FIXED_OVERHEAD_GB = profiles.DEFAULT_FIXED_OVERHEAD_GB
```

Replace `estimate_base_gb` (lines ~73-76):
```python
def estimate_base_gb(info: models.ModelInfo, limits: SystemLimits, con) -> float:
    """Pre-flight guess of ABSOLUTE base (context->0) OS-wired footprint, before any probe."""
    factor, overhead, _ = profiles.cold_start_constants(con)
    os_baseline = max(limits.wired_now_gb, 2.5)
    return os_baseline + info.weights_gb * factor + overhead
```

In `characterize`, update the one internal caller (around line 154, `con` is created above it at ~line 135):
```python
    est = estimate_base_gb(info, limits, con)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_probe_math.py -v`
Expected: PASS (all in file, including the updated existing callers).

- [ ] **Step 5: Commit**

```bash
git add wmx_suite/probe.py tests/test_probe_math.py
git commit -m "feat(probe): estimate_base_gb reads per-machine cold-start profile"
```

---

### Task 5: `launcher.plan` reads the profile + exposes `cold_start_profile`

**Files:**
- Modify: `wmx_suite/launcher.py` (import line 16; estimated branch ~89-93; `p` dict ~97-106)
- Test: `tests/test_launcher.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_launcher.py` (mirror the file's existing approach for monkeypatching `models.describe`, `read_limits`, and `db`; read it first to reuse helpers):
```python
def test_plan_estimated_marks_cold_start_source(monkeypatch, tmp_path):
    from wmx_suite import db, launcher, models, profiles
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "suite.db")
    monkeypatch.setattr(profiles, "machine_key", lambda: ("Apple M4 Pro", 1, 15))
    # a quantizable, uncharacterized model with comfortable headroom:
    info = _model_info(weights_gb=2.0, can_quantize_kv=True, is_causal=True, max_context=32768)
    monkeypatch.setattr(models, "describe", lambda hf_id: info)
    monkeypatch.setattr(launcher, "read_limits", lambda: _limits(wall_gb=17.0, wired_now_gb=3.0))
    monkeypatch.setattr(launcher, "sample_settled_baseline", lambda: 3.0)

    p = launcher.plan("mlx-community/test")
    assert p["source"] == "estimated"
    assert p["cold_start_profile"] == "default"

    db.upsert_profile(db.connect(), ("Apple M4 Pro", 1, 15), resident_factor=1.05,
                      fixed_overhead_gb=1.0, model_id="m", n_points=2, mlx_version="9.9")
    p2 = launcher.plan("mlx-community/test")
    assert p2["cold_start_profile"] == "profile"
```
NOTE: use whatever `_model_info`/`_limits` helpers exist in `tests/test_launcher.py`; match their parameter names. If none exist, build `models.ModelInfo(...)` and `system.SystemLimits(...)` inline with all required fields.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_launcher.py -k cold_start_source -v`
Expected: FAIL — `KeyError: 'cold_start_profile'`.

- [ ] **Step 3: Implement**

In `wmx_suite/launcher.py`:

Remove the constant import (line 16):
```python
from .probe import FIXED_OVERHEAD_GB, RESIDENT_FACTOR
```
and add `profiles` to the existing relative import `from . import config, db, models`:
```python
from . import config, db, models, profiles
```

In `plan()`, before the `if fit and fit.get("slope_gb_per_k"):` block, initialize the source marker:
```python
    cold_source = None
```
Replace the estimated `else` branch (lines ~89-93):
```python
    else:
        model_base = info.weights_gb * RESIDENT_FACTOR + FIXED_OVERHEAD_GB
        slope = _estimated_slope_gb_per_k(info)
        source = "estimated"
        fit_stale = False
```
with:
```python
    else:
        factor, overhead, cold_source = profiles.cold_start_constants(con)
        model_base = info.weights_gb * factor + overhead
        slope = _estimated_slope_gb_per_k(info)
        source = "estimated"
        fit_stale = False
```
In the `p = {...}` dict (around lines 97-106), add a key:
```python
        "cold_start_profile": cold_source,
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_launcher.py -v`
Expected: PASS (whole file; the removed import must not be referenced elsewhere — `_estimated_slope_gb_per_k` uses only `PREFILL_SPIKE_MULT`).

- [ ] **Step 5: Commit**

```bash
git add wmx_suite/launcher.py tests/test_launcher.py
git commit -m "feat(launcher): plan reads cold-start profile, exposes cold_start_profile"
```

---

### Task 6: `probe.calibrate()` + `_pick_calibration_model()`

**Files:**
- Modify: `wmx_suite/probe.py` (add near the other module functions; reuse `_measure_rung`, `_linfit`, `read_limits`)
- Test: `tests/test_probe_math.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_probe_math.py`:
```python
def test_pick_calibration_model_smallest_causal(monkeypatch):
    from wmx_suite import models, probe
    monkeypatch.setattr(models, "scan_cache", lambda: ["org/big", "org/small", "org/notcausal"])

    def fake_describe(hf_id):
        table = {
            "org/big": _model_info(weights_gb=8.0, is_causal=True),
            "org/small": _model_info(weights_gb=0.5, is_causal=True),
            "org/notcausal": _model_info(weights_gb=0.1, is_causal=False),
        }
        return table[hf_id]
    monkeypatch.setattr(models, "describe", fake_describe)
    assert probe._pick_calibration_model() == "org/small"


def test_pick_calibration_model_errors_when_none(monkeypatch):
    import pytest
    from wmx_suite import models, probe
    monkeypatch.setattr(models, "scan_cache", lambda: [])
    with pytest.raises(SystemExit, match="no causal"):
        probe._pick_calibration_model()


def test_calibrate_solves_and_floors_overhead(monkeypatch, tmp_path):
    from wmx_suite import db, models, probe, profiles
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "suite.db")
    key = ("Apple M4 Pro", 1, 15)
    monkeypatch.setattr(profiles, "machine_key", lambda: key)
    info = _model_info(weights_gb=0.5, is_causal=True, can_quantize_kv=True, max_context=32768)
    monkeypatch.setattr(models, "describe", lambda hf_id: info)
    monkeypatch.setattr(probe, "read_limits", lambda: _limits(wall_gb=17.0, wired_now_gb=3.0))

    # Two rungs whose delta intercept at c->0 is 2.0 GB: delta = 2.0 + 0.1*ctx_k
    deltas = {512: 2.05, 2048: 2.20}
    def fake_measure(py, hf_id, ctx, kv_bits, repeats, *, verbose, log):
        return {"status": "ok", "context": ctx, "repeats": 3, "delta": deltas[ctx],
                "os_wired_gb": 5.0, "mlx_peak_gb": 4.0, "spread_gb": 0.1}
    monkeypatch.setattr(probe, "_measure_rung", fake_measure)

    result = probe.calibrate("org/tiny", verbose=False)
    # intercept ~2.0; measured overhead = 2.0 - 1.05*0.5 = 1.475; > floor 1.0 -> stored as-is
    assert abs(result["intercept_gb"] - 2.0) < 0.05
    assert abs(result["measured_overhead_gb"] - 1.475) < 0.05
    assert result["fixed_overhead_gb"] >= profiles.DEFAULT_FIXED_OVERHEAD_GB
    stored = db.get_profile(db.connect(), key)
    assert stored["fixed_overhead_gb"] == result["fixed_overhead_gb"]


def test_calibrate_floor_applies_when_residual_low(monkeypatch, tmp_path):
    from wmx_suite import db, models, probe, profiles
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "suite.db")
    monkeypatch.setattr(profiles, "machine_key", lambda: ("Apple M4 Pro", 1, 15))
    info = _model_info(weights_gb=0.5, is_causal=True, can_quantize_kv=True, max_context=32768)
    monkeypatch.setattr(models, "describe", lambda hf_id: info)
    monkeypatch.setattr(probe, "read_limits", lambda: _limits(wall_gb=17.0, wired_now_gb=3.0))
    deltas = {512: 0.30, 2048: 0.30}  # tiny base -> residual goes negative
    monkeypatch.setattr(probe, "_measure_rung",
                        lambda py, hf, ctx, kv, r, *, verbose, log:
                        {"status": "ok", "repeats": 3, "delta": deltas[ctx], "mlx_peak_gb": 1.0,
                         "os_wired_gb": 3.3, "spread_gb": 0.0})
    result = probe.calibrate("org/tiny", verbose=False)
    assert result["fixed_overhead_gb"] == profiles.DEFAULT_FIXED_OVERHEAD_GB  # floored
```
NOTE: ensure `_model_info`/`_limits` accept the kwargs used here (`weights_gb`, `is_causal`, `can_quantize_kv`, `max_context`, `wall_gb`, `wired_now_gb`). Extend the helpers if needed (sensible defaults), matching how the file already constructs `ModelInfo`/`SystemLimits`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_probe_math.py -k "calibrat or pick_calibration" -v`
Expected: FAIL — `AttributeError: module 'wmx_suite.probe' has no attribute '_pick_calibration_model'`.

- [ ] **Step 3: Implement `_pick_calibration_model` and `calibrate`**

In `wmx_suite/probe.py`, add a constant near `DEFAULT_RAMP`:
```python
CALIBRATE_RAMP = [512, 2048]  # two small safe rungs; the c->0 intercept gives the base
```
And add these functions (after `characterize`):
```python
def _pick_calibration_model() -> str:
    """Smallest causal mlx-community model in the HF cache (by on-disk weight size)."""
    candidates: list[tuple[float, str]] = []
    for hf_id in models.scan_cache():
        info = models.describe(hf_id)
        if info is None or not info.is_causal:
            continue
        candidates.append((info.weights_gb, hf_id))
    if not candidates:
        raise SystemExit(
            "[calibrate] no causal mlx-community model found in the HF cache. "
            "Download a small one (e.g. a 0.5-1.5B mlx-community model) or pass --model."
        )
    candidates.sort()
    return candidates[0][1]


def calibrate(model: str | None = None, *, margin_gb: float | None = None,
              repeats: int = DEFAULT_REPEATS, worker_python: str | None = None,
              verbose: bool = True) -> dict:
    """Measure this machine's cold-start FIXED_OVERHEAD_GB and store a per-machine profile.

    Fits the model's base delta-over-baseline at context->0 from two small safe rungs and
    derives overhead = intercept - DEFAULT_RESIDENT_FACTOR * weights, floored at the default
    so calibration only ever tightens (never loosens) the pre-flight estimate.
    """
    margin_gb = config.margin_gb(margin_gb)
    hf_id = model or _pick_calibration_model()
    info = models.describe(hf_id)
    if info is None:
        raise SystemExit(f"[calibrate] model not found in HF cache: {hf_id}")
    if not info.is_causal:
        raise SystemExit(f"[calibrate] {hf_id} is not a supported causal language model.")

    limits = read_limits()
    threshold = limits.safe_threshold_gb(margin_gb)
    kv_bits = 4 if info.can_quantize_kv else None
    py = worker_python or sys.executable

    def log(*a):
        if verbose:
            print(*a, flush=True)

    import mlx.core as mx
    con = db.connect()

    est = estimate_base_gb(info, limits, con)
    if est >= threshold:
        raise SystemExit(
            f"[calibrate] estimated base {est:.2f}GB >= threshold {threshold:.2f}GB — "
            f"machine too loaded or model too large to calibrate safely. Free memory or "
            f"pass a smaller --model."
        )

    log(f"# calibrate {hf_id}  weights={info.weights_gb}GB  "
        f"threshold={threshold:.2f}GB  kv_bits={kv_bits}")

    xs_k: list[float] = []
    ys: list[float] = []
    for ctx in CALIBRATE_RAMP:
        if ctx > (info.max_context or ctx):
            continue
        m = _measure_rung(py, hf_id, ctx, kv_bits, repeats, verbose=verbose, log=log)
        if m is None or m.get("status") != "ok":
            note = (m or {}).get("note", "no output")
            raise SystemExit(f"[calibrate] rung {ctx} failed: {note}")
        xs_k.append(ctx / 1000)
        ys.append(m["delta"])
        log(f"{ctx:>6}  delta={m['delta']:.3f}GB  (median of {m['repeats']})")

    if len(xs_k) < 2:
        raise SystemExit("[calibrate] need >=2 successful rungs to fit the base intercept.")

    intercept, _slope, _r2 = _linfit(xs_k, ys)
    measured_overhead = round(intercept - profiles.DEFAULT_RESIDENT_FACTOR * info.weights_gb, 3)
    fixed_overhead = max(profiles.DEFAULT_FIXED_OVERHEAD_GB, measured_overhead)

    key = profiles.machine_key()
    db.upsert_profile(con, key, resident_factor=profiles.DEFAULT_RESIDENT_FACTOR,
                      fixed_overhead_gb=fixed_overhead, model_id=hf_id,
                      n_points=len(xs_k), mlx_version=mx.__version__)
    log(f"# intercept(base@c->0)={intercept:.3f}GB  measured_overhead={measured_overhead:.3f}GB  "
        f"stored={fixed_overhead:.3f}GB (floor {profiles.DEFAULT_FIXED_OVERHEAD_GB})")
    return {
        "hf_id": hf_id, "machine_key": key, "intercept_gb": round(intercept, 3),
        "measured_overhead_gb": measured_overhead, "fixed_overhead_gb": fixed_overhead,
        "default_overhead_gb": profiles.DEFAULT_FIXED_OVERHEAD_GB, "n_points": len(xs_k),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_probe_math.py -v`
Expected: PASS (all in file).

- [ ] **Step 5: Commit**

```bash
git add wmx_suite/probe.py tests/test_probe_math.py
git commit -m "feat(probe): add calibrate() and smallest-model picker"
```

---

### Task 7: `cmd_calibrate` + subparser

**Files:**
- Modify: `wmx_suite/cli.py` (import line ~26; add `cmd_calibrate` near other handlers; register in `_main_argparse` ~1045-1056)
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_cli.py`:
```python
def test_cmd_calibrate_prints_summary(monkeypatch, capsys):
    from types import SimpleNamespace
    from wmx_suite import cli, probe
    monkeypatch.setattr(probe, "calibrate", lambda model, margin_gb=None: {
        "hf_id": "org/tiny", "machine_key": ("Apple M4 Pro", 25769803776, 15),
        "intercept_gb": 2.0, "measured_overhead_gb": 1.48, "fixed_overhead_gb": 1.48,
        "default_overhead_gb": 1.0, "n_points": 2,
    })
    cli.cmd_calibrate(SimpleNamespace(model="org/tiny", margin=None))
    out = capsys.readouterr().out
    assert "org/tiny" in out
    assert "Apple M4 Pro" in out
    assert "1.48" in out


def test_cmd_calibrate_propagates_no_model_error(monkeypatch):
    import pytest
    from types import SimpleNamespace
    from wmx_suite import cli, probe
    def boom(model, margin_gb=None):
        raise SystemExit("[calibrate] no causal mlx-community model found in the HF cache.")
    monkeypatch.setattr(probe, "calibrate", boom)
    with pytest.raises(SystemExit, match="no causal"):
        cli.cmd_calibrate(SimpleNamespace(model=None, margin=None))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cli.py -k cmd_calibrate -v`
Expected: FAIL — `AttributeError: module 'wmx_suite.cli' has no attribute 'cmd_calibrate'`.

- [ ] **Step 3: Implement the handler and register it**

In `wmx_suite/cli.py`, add `profiles` to the relative import (line ~26):
```python
from . import config, db, launcher, models, probe, profiles
```
Add the handler (near the other `cmd_*` functions):
```python
def cmd_calibrate(args):
    """Measure this machine's cold-start overhead and store a per-machine profile."""
    margin = _configured_margin(args.margin)
    result = probe.calibrate(args.model, margin_gb=margin)
    dev, ram, osv = result["machine_key"]
    print("=" * 60)
    print("  Calibrated cold-start overhead for this machine")
    print("=" * 60)
    print(f"  Machine    : {dev} / {ram / 1e9:.0f}GB / macOS {osv}")
    print(f"  Model used : {result['hf_id']} ({result['n_points']} rungs)")
    print(f"  Measured   : {result['measured_overhead_gb']:.2f} GB overhead "
          f"(default {result['default_overhead_gb']:.2f} GB)")
    print(f"  Stored     : {result['fixed_overhead_gb']:.2f} GB  (floored at default)")
    print("=" * 60)
```
Register it in `_main_argparse` (alongside the other `sub.add_parser(...)` calls, ~line 1045):
```python
    p = sub.add_parser("calibrate", help="measure this machine's cold-start overhead constant")
    p.add_argument("--model", default=None,
                   help="model to calibrate with (default: smallest cached causal model)")
    p.add_argument("--margin", default=None,
                   help="safety cushion in GB (overrides WMX_SUITE_MARGIN_GB)")
    p.set_defaults(func=cmd_calibrate)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_cli.py -k cmd_calibrate -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add wmx_suite/cli.py tests/test_cli.py
git commit -m "feat(cli): add wmx-suite calibrate command"
```

---

### Task 8: Warnings in `run`, `health`, and `system`

**Files:**
- Modify: `wmx_suite/cli.py` (`_run` after the source prints; `cmd_health` after `con = db.connect()`; `cmd_system` end)
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_cli.py`:
```python
def test_health_warns_when_no_profile(monkeypatch, tmp_path, capsys):
    from types import SimpleNamespace
    from wmx_suite import cli, db, profiles
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "suite.db")
    monkeypatch.setattr(profiles, "machine_key", lambda: ("Apple M4 Pro", 25769803776, 15))
    monkeypatch.setattr(cli, "read_limits", lambda: _limits(wall_gb=17.0, wired_now_gb=3.0))
    monkeypatch.setattr(cli, "sample_settled_baseline", lambda: 3.0)
    cli.cmd_health(SimpleNamespace(margin=None))
    out = capsys.readouterr().out
    assert "No calibration profile" in out
    assert "calibrate" in out


def test_system_reports_no_profile(monkeypatch, tmp_path, capsys):
    from types import SimpleNamespace
    from wmx_suite import cli, db, profiles
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "suite.db")
    monkeypatch.setattr(profiles, "machine_key", lambda: ("Apple M4 Pro", 25769803776, 15))
    monkeypatch.setattr(cli, "read_limits", lambda: _limits(wall_gb=17.0, wired_now_gb=3.0))
    cli.cmd_system(SimpleNamespace())
    out = capsys.readouterr().out
    assert "calibration profile" in out.lower()
```
NOTE: reuse `tests/test_cli.py`'s existing `_limits` helper if present; otherwise build a `SystemLimits`. The `cmd_system`/`cmd_health` functions reference module-level `read_limits`/`sample_settled_baseline` imported into `cli`, so monkeypatch them on `cli`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_cli.py -k "no_profile or reports_no_profile" -v`
Expected: FAIL — assertions not found in output.

- [ ] **Step 3: Implement the three surfaces**

In `wmx_suite/cli.py`:

**`cmd_health`** — right after `con = db.connect()` (the line before the `rows = con.execute(...)` query), add:
```python
    if db.get_profile(con, profiles.machine_key()) is None:
        dev, ram, osv = profiles.machine_key()
        print(f"\nNo calibration profile for {dev}/{ram / 1e9:.0f}GB/macOS {osv}; "
              "cold-start estimates use M4-Pro defaults. Run 'wmx-suite calibrate'.")
```

**`cmd_system`** — at the end of the function, add:
```python
    con = db.connect()
    prof = db.get_profile(con, profiles.machine_key())
    if prof:
        print(f"calibration profile : overhead {prof['fixed_overhead_gb']:.2f} GB "
              f"(model {prof['model_id']}, {prof['calibrated_at']})")
    else:
        print("calibration profile : none — using M4-Pro defaults; run 'wmx-suite calibrate'")
```

**`_run`** — after the `print(f"[run] source=...")` / `print(f"[run] live_base ...")` block (just before the `if p.get("refuse"):` check), add:
```python
    if p["source"] == "estimated" and p.get("cold_start_profile") == "default":
        print("[run] WARNING: using default cold-start constants tuned for Apple M4 Pro; "
              "run 'wmx-suite calibrate' to tune them for this machine.", file=sys.stderr)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_cli.py -k "no_profile or reports_no_profile" -v`
Expected: PASS.

- [ ] **Step 5: Run the full suite + compile**

Run: `uv run pytest -q && uv run python -m compileall -q wmx_suite tests`
Expected: all pass, compile clean.

- [ ] **Step 6: Commit**

```bash
git add wmx_suite/cli.py tests/test_cli.py
git commit -m "feat(cli): warn/report when no calibration profile exists"
```

---

### Task 9: Documentation

**Files:**
- Modify: `README.md`, `AGENTS.md`

- [ ] **Step 1: Update `AGENTS.md` command list**

In the `## Commands` fenced block in `AGENTS.md`, add a line after the `characterize` entry:
```
uv run wmx-suite calibrate                # seed this machine's cold-start overhead profile
```
And in the conventions/profile discussion, add one line:
> `calibrate` tunes only the cold-start estimate (the `FIXED_OVERHEAD_GB` term) per machine; it is floored at the default so it can only tighten the estimate. `characterize` remains the per-model per-machine adaptation mechanism.

- [ ] **Step 2: Update `README.md`**

Add `calibrate` to the command list/usage section in `README.md`, mirroring the existing entries' format, with a one-line description: "Measure this machine's cold-start memory overhead so pre-flight estimates are accurate on your Apple Silicon SKU (run once per machine; `characterize` still adapts per model)."

- [ ] **Step 3: Verify docs compile / no broken references**

Run: `uv run python -m compileall -q wmx_suite` (sanity) and visually confirm the command name matches `calibrate`.

- [ ] **Step 4: Commit**

```bash
git add README.md AGENTS.md
git commit -m "docs: document wmx-suite calibrate"
```

---

## Final verification

- [ ] Run `uv run pytest -q` — all tests pass.
- [ ] Run `uv run python -m compileall -q wmx_suite tests` — clean.
- [ ] Run `uv run wmx-suite system` — shows the "calibration profile: none ..." line.
- [ ] Run `uv run wmx-suite calibrate` — picks the smallest cached causal model, prints the summary, and a second `system`/`health` shows the profile (manual smoke test; loads a tiny model through the safe path — acceptable per the spec).
