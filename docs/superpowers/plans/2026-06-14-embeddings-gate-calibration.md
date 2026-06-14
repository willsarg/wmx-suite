# Calibrated Embeddings Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the embeddings memory gate stop over-skipping safe cells by fitting the memory model **with an intercept**, and **persist** the fitted coefficients per (machine, MLX, model) so later runs are accurate from the start — without weakening the never-crash guarantee (#21).

**Architecture:** Replace the through-origin 2-param fit in `embeddings_probe.py` with a 3-param `c + a·x1 + b·x2` fit (the intercept absorbs fixed model residency that was inflating the linear slope and blowing up extrapolation). Add an `embedding_profiles` table + accessors mirroring `system_profiles`; `sweep` loads a matching profile to seed the gate and auto-upserts the final fit after each run. Physical one-layer floors (`A_FLOOR`/`B_FLOOR`) and `PRED_SAFETY` remain the safety backstop.

**Tech Stack:** Python, SQLite, MLX, pytest.

**Spec:** `docs/superpowers/specs/2026-06-14-embeddings-gate-calibration-design.md`

**Conventions (all tasks):** `uv` only; tests hardware-free (no model load, no live memory probing, no prod DB — `monkeypatch.setattr(db, "DB_PATH", tmp_path/"suite.db")`); commit to `main`; commit trailer `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.

---

## File Structure
- Modify `wmx_suite/db.py` — add `embedding_profiles` table + `upsert_embedding_profile`/`get_embedding_profile` (5-tuple key).
- Modify `wmx_suite/profiles.py` — add `embedding_machine_key`, `embedding_coeffs`, `upsert_embedding_coeffs`.
- Modify `wmx_suite/embeddings_probe.py` — `_fit_ab`→`_fit_cab` (3-param) + `_det3`/`_solve3`; rewrite `_coeffs` policy; replace `_estimate_model_base` with the `max(SEED,c)` clamp; wire profile load/upsert + `mlx_version`/`ignore_profile` into `sweep`.
- Modify `wmx_suite/cli.py` — pass `mlx_version`, add `--ignore-profile`, print a profile note.
- Modify `tests/test_embeddings_benchmark.py` — fit, policy, regression, persistence, RULE #1 tests.

---

## Task 1: `embedding_profiles` table + accessors

**Files:**
- Modify: `wmx_suite/db.py` (append a `CREATE TABLE` block to `SCHEMA` before its closing `"""`; add 2 functions at end of file)
- Test: `tests/test_embeddings_benchmark.py`

- [ ] **Step 1: Write the failing test** (append to `tests/test_embeddings_benchmark.py`)
```python
def test_embedding_profile_roundtrip_and_key_mismatch(monkeypatch, tmp_path):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "suite.db")
    con = db.connect()
    key = ("Apple M4 Pro", 25769803776, 15, "0.31.2", "mlx-community/test-embed")
    assert db.get_embedding_profile(con, key) is None

    db.upsert_embedding_profile(con, key, coef_intercept_gb=1.07,
                                coef_linear=2.1e-5, coef_quad=6.6e-9, n_points=20)
    row = db.get_embedding_profile(con, key)
    assert row["coef_intercept_gb"] == 1.07
    assert row["coef_linear"] == 2.1e-5
    assert row["coef_quad"] == 6.6e-9
    assert row["n_points"] == 20
    assert row["created_at"]

    # upsert is idempotent on the 5-part key (updates, not duplicates)
    db.upsert_embedding_profile(con, key, coef_intercept_gb=2.0,
                                coef_linear=3.0e-5, coef_quad=7.0e-9, n_points=30)
    assert db.get_embedding_profile(con, key)["coef_intercept_gb"] == 2.0
    assert con.execute("SELECT COUNT(*) FROM embedding_profiles").fetchone()[0] == 1

    # a different mlx_version is a different key -> miss (staleness safety)
    stale = ("Apple M4 Pro", 25769803776, 15, "0.32.0", "mlx-community/test-embed")
    assert db.get_embedding_profile(con, stale) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_embeddings_benchmark.py::test_embedding_profile_roundtrip_and_key_mismatch -q`
Expected: FAIL (`module 'wmx_suite.db' has no attribute 'upsert_embedding_profile'`).

- [ ] **Step 3: Add the schema block** — in `wmx_suite/db.py`, inside `SCHEMA`, immediately before the closing `"""` (after the `embeddings_measurements` table), add:
```sql

CREATE TABLE IF NOT EXISTS embedding_profiles (
    device_name       TEXT NOT NULL,
    total_ram_bytes   INTEGER NOT NULL,
    macos_major       INTEGER NOT NULL,
    mlx_version       TEXT NOT NULL,
    model_id          TEXT NOT NULL,
    coef_intercept_gb REAL NOT NULL,
    coef_linear       REAL NOT NULL,
    coef_quad         REAL NOT NULL,
    n_points          INTEGER NOT NULL,
    created_at        TEXT,
    PRIMARY KEY (device_name, total_ram_bytes, macos_major, mlx_version, model_id)
);
```

- [ ] **Step 4: Add the functions** at the END of `wmx_suite/db.py`:
```python
# --- Embedding calibration profiles (per machine + MLX + model) ---
def upsert_embedding_profile(
    con: sqlite3.Connection,
    key: tuple[str, int, int, str, str],
    *,
    coef_intercept_gb: float,
    coef_linear: float,
    coef_quad: float,
    n_points: int,
) -> None:
    dev, ram, osv, mlxv, model_id = key
    con.execute(
        "INSERT INTO embedding_profiles "
        "(device_name, total_ram_bytes, macos_major, mlx_version, model_id, "
        " coef_intercept_gb, coef_linear, coef_quad, n_points, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
        "ON CONFLICT(device_name, total_ram_bytes, macos_major, mlx_version, model_id) "
        "DO UPDATE SET coef_intercept_gb=excluded.coef_intercept_gb, "
        "coef_linear=excluded.coef_linear, coef_quad=excluded.coef_quad, "
        "n_points=excluded.n_points, created_at=excluded.created_at",
        (dev, ram, osv, mlxv, model_id, coef_intercept_gb, coef_linear, coef_quad,
         n_points, _now()),
    )
    con.commit()


def get_embedding_profile(
    con: sqlite3.Connection, key: tuple[str, int, int, str, str]
) -> dict | None:
    dev, ram, osv, mlxv, model_id = key
    row = con.execute(
        "SELECT device_name, total_ram_bytes, macos_major, mlx_version, model_id, "
        "coef_intercept_gb, coef_linear, coef_quad, n_points, created_at "
        "FROM embedding_profiles WHERE device_name=? AND total_ram_bytes=? AND "
        "macos_major=? AND mlx_version=? AND model_id=?",
        (dev, ram, osv, mlxv, model_id),
    ).fetchone()
    return dict(row) if row else None
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_embeddings_benchmark.py::test_embedding_profile_roundtrip_and_key_mismatch -q`
Expected: PASS.

- [ ] **Step 6: Commit**
```bash
git add wmx_suite/db.py tests/test_embeddings_benchmark.py
git commit -m "feat(db): embedding_profiles table + upsert/get accessors (#21)"
```

---

## Task 2: `profiles.py` embedding accessors

**Files:**
- Modify: `wmx_suite/profiles.py` (add 3 functions; it already imports `db` and defines `machine_key`)
- Test: `tests/test_embeddings_benchmark.py`

- [ ] **Step 1: Write the failing test** (append)
```python
def test_profiles_embedding_coeffs_roundtrip(monkeypatch, tmp_path):
    from wmx_suite import profiles
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "suite.db")
    monkeypatch.setattr(profiles, "machine_key",
                        lambda: ("Apple M4 Pro", 25769803776, 15))
    con = db.connect()

    assert profiles.embedding_coeffs(con, "org/m", "0.31.2") is None

    profiles.upsert_embedding_coeffs(con, "org/m", "0.31.2",
                                     coef_intercept_gb=1.1, coef_linear=2e-5,
                                     coef_quad=6e-9, n_points=12)
    assert profiles.embedding_coeffs(con, "org/m", "0.31.2") == (1.1, 2e-5, 6e-9)
    # different model or mlx version -> miss
    assert profiles.embedding_coeffs(con, "org/other", "0.31.2") is None
    assert profiles.embedding_coeffs(con, "org/m", "9.9.9") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_embeddings_benchmark.py::test_profiles_embedding_coeffs_roundtrip -q`
Expected: FAIL (`module 'wmx_suite.profiles' has no attribute 'embedding_coeffs'`).

- [ ] **Step 3: Add the functions** at the END of `wmx_suite/profiles.py`:
```python
def embedding_machine_key(model_id: str, mlx_version: str) -> tuple[str, int, int, str, str]:
    """5-part key for an embedding profile: machine identity + MLX version + model."""
    dev, ram, osv = machine_key()
    return (dev, ram, osv, mlx_version, model_id)


def embedding_coeffs(con: sqlite3.Connection, model_id: str,
                     mlx_version: str) -> tuple[float, float, float] | None:
    """Stored (intercept_gb, linear, quad) gate coefficients for this machine+mlx+model,
    or None when there is no matching calibration profile (→ cold-start fallback)."""
    row = db.get_embedding_profile(con, embedding_machine_key(model_id, mlx_version))
    if row is None:
        return None
    return (float(row["coef_intercept_gb"]), float(row["coef_linear"]),
            float(row["coef_quad"]))


def upsert_embedding_coeffs(con: sqlite3.Connection, model_id: str, mlx_version: str, *,
                            coef_intercept_gb: float, coef_linear: float,
                            coef_quad: float, n_points: int) -> None:
    db.upsert_embedding_profile(
        con, embedding_machine_key(model_id, mlx_version),
        coef_intercept_gb=coef_intercept_gb, coef_linear=coef_linear,
        coef_quad=coef_quad, n_points=n_points,
    )
```
(`sqlite3` is already imported in `profiles.py`.)

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_embeddings_benchmark.py::test_profiles_embedding_coeffs_roundtrip -q`
Expected: PASS.

- [ ] **Step 5: Commit**
```bash
git add wmx_suite/profiles.py tests/test_embeddings_benchmark.py
git commit -m "feat(profiles): embedding coeff accessors over embedding_profiles (#21)"
```

---

## Task 3: 3-parameter fit (`_fit_cab`)

**Files:**
- Modify: `wmx_suite/embeddings_probe.py` (replace `_fit_ab` with `_fit_cab` + 3×3 solver helpers)
- Test: `tests/test_embeddings_benchmark.py`

- [ ] **Step 1: Write the failing test** (append)
```python
def test_fit_cab_recovers_known_coeffs_and_handles_singular():
    from wmx_suite import embeddings_probe as ep
    # delta = 1.0 + 1e-6*x1 + 2e-8*x2 exactly, over varied (batch, seq)
    pts = []
    for bsz, s in [(1, 128), (1, 512), (2, 256), (4, 512), (8, 1024)]:
        x1, x2 = bsz * s, bsz * s * s
        pts.append((x1, x2, 1.0 + 1e-6 * x1 + 2e-8 * x2))
    c, a, b = ep._fit_cab(pts)
    assert c == pytest.approx(1.0, abs=1e-6)
    assert a == pytest.approx(1e-6, rel=1e-3)
    assert b == pytest.approx(2e-8, rel=1e-3)

    # < 3 points -> None
    assert ep._fit_cab([(128.0, 16384.0, 1.0), (256.0, 65536.0, 1.1)]) is None
    # collinear (all identical feature rows) -> singular -> None
    same = [(128.0, 16384.0, 1.0)] * 5
    assert ep._fit_cab(same) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_embeddings_benchmark.py::test_fit_cab_recovers_known_coeffs_and_handles_singular -q`
Expected: FAIL (`_fit_cab` not defined).

- [ ] **Step 3: Replace `_fit_ab` in `wmx_suite/embeddings_probe.py`** with these three functions (delete the old `_fit_ab`):
```python
def _det3(m: list[list[float]]) -> float:
    return (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]))


def _solve3(mat: list[list[float]], rhs: list[float]) -> tuple[float, float, float] | None:
    """Solve a 3x3 linear system by Cramer's rule. Returns None if singular."""
    det = _det3(mat)
    if abs(det) < 1e-30:
        return None
    out = []
    for i in range(3):
        mi = [row[:] for row in mat]
        for j in range(3):
            mi[j][i] = rhs[j]
        out.append(_det3(mi) / det)
    return (out[0], out[1], out[2])


def _fit_cab(points: list[tuple[float, float, float]]) -> tuple[float, float, float] | None:
    """Least-squares delta = c + a*x1 + b*x2 (intercept + linear + quadratic).

    points: (x1=batch*seq, x2=batch*seq^2, delta). Returns (c, a, b), or None when there
    are <3 points or the normal-equation system is singular (e.g. a degenerate grid with
    only one distinct (x1, x2)). The intercept c keeps fixed model residency out of the
    slope terms, which is what prevents through-origin extrapolation blow-up.
    """
    n = len(points)
    if n < 3:
        return None
    sx1 = sx2 = sy = sx11 = sx12 = sx22 = sx1y = sx2y = 0.0
    for x1, x2, y in points:
        sx1 += x1
        sx2 += x2
        sy += y
        sx11 += x1 * x1
        sx12 += x1 * x2
        sx22 += x2 * x2
        sx1y += x1 * y
        sx2y += x2 * y
    mat = [[float(n), sx1, sx2],
           [sx1, sx11, sx12],
           [sx2, sx12, sx22]]
    rhs = [sy, sx1y, sx2y]
    return _solve3(mat, rhs)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_embeddings_benchmark.py::test_fit_cab_recovers_known_coeffs_and_handles_singular -q`
Expected: PASS.

NOTE: the old `test_fit_recovers_known_coeffs` (which called `_fit_ab`) will now fail because `_fit_ab` is gone. Delete that obsolete test in this step (it is superseded by the new `_fit_cab` test):
```bash
# remove the test function `def test_fit_recovers_known_coeffs(...)` from tests/test_embeddings_benchmark.py
```
Then `uv run pytest tests/test_embeddings_benchmark.py -q -k fit` should pass with only the new test.

- [ ] **Step 5: Commit**
```bash
git add wmx_suite/embeddings_probe.py tests/test_embeddings_benchmark.py
git commit -m "feat(embeddings): 3-param intercept fit replacing through-origin (#21)"
```

---

## Task 4: Gate policy + model_base clamp + regression test

**Files:**
- Modify: `wmx_suite/embeddings_probe.py` (rewrite `_coeffs`; remove `_estimate_model_base`; update the gate block in `sweep` to use `max(SEED, c)` and the floors)
- Test: `tests/test_embeddings_benchmark.py`

This task changes only the *within-run* behavior (stored profile still defaults to None; persistence wiring is Task 5).

- [ ] **Step 1: Write the failing tests** (append)
```python
# Real Run-2 os_wired values (GB), keyed (batch, seq). Surface is flat (~2.7 baseline).
_RUN2_OSWIRED = {
    (1, 128): 2.72, (1, 256): 3.50, (1, 512): 3.35, (1, 1024): 3.63, (1, 2048): 4.07, (1, 4096): 4.26,
    (2, 128): 3.29, (2, 256): 3.48, (2, 512): 3.63, (2, 1024): 4.23, (2, 2048): 4.10, (2, 4096): 4.47,
    (4, 128): 3.50, (4, 256): 3.62, (4, 512): 4.11, (4, 1024): 4.18, (4, 2048): 4.33, (4, 4096): 4.49,
    (8, 128): 3.61, (8, 256): 4.10, (8, 512): 4.18, (8, 1024): 4.29, (8, 2048): 4.20,
    (16, 128): 4.11, (16, 256): 4.16, (16, 512): 4.26, (16, 1024): 4.04, (16, 2048): 4.79,
    (32, 128): 4.17, (32, 256): 4.25, (32, 512): 3.99, (32, 1024): 4.50,
}


def _events_from_run2_grid(monkeypatch):
    """Drive the real sweep over the default grid with Run-2 os_wired values; unknown
    (previously-skipped) cells return a flat 4.6 GB consistent with the measured surface."""
    from wmx_suite import embeddings_probe as ep

    def fake_run_cell(py, model, batch, seq, repeats, margin):
        ow = _RUN2_OSWIRED.get((batch, seq), 4.6)
        return {"status": "rung_done", "batch": batch, "seq": seq,
                "os_wired_gb": ow, "peak_gb": 1.0, "throughput_tps": 1.0, "latency_ms": 1.0}

    monkeypatch.setattr(ep, "_run_cell", fake_run_cell)
    monkeypatch.setattr(ep, "sample_settled_baseline", lambda: 2.71)
    monkeypatch.setattr(ep, "read_limits",
                        lambda: SimpleNamespace(safe_threshold_gb=lambda m=2.0: 15.18,
                                                wall_gb=17.18, wired_now_gb=2.71))
    events = []
    ep.sweep(con=None, run_id=1, model="m", repeats=1, margin_gb=2.0,
             on_event=events.append, persist=False)
    return events


def test_intercept_gate_measures_previously_skipped_safe_cells(monkeypatch):
    events = _events_from_run2_grid(monkeypatch)
    measured = {(e["batch"], e["seq"]) for e in events if e["event"] == "cell_done"}
    skipped = {(e["batch"], e["seq"]) for e in events if e["event"] == "row_skipped"}
    # Cells the OLD through-origin gate wrongly skipped are now measured:
    assert (8, 4096) in measured
    assert (32, 2048) in measured
    # The true danger corner is still never measured:
    assert (32, 8192) not in measured
    assert (32, 8192) in skipped


def test_model_base_clamped_to_seed_on_negative_intercept(monkeypatch):
    from wmx_suite import embeddings_probe as ep
    # A stored profile with a NEGATIVE intercept must not drop model_base below the seed.
    spawned = []

    def fake_run_cell(py, model, batch, seq, repeats, margin):
        spawned.append((batch, seq))
        return {"status": "rung_done", "batch": batch, "seq": seq,
                "os_wired_gb": 2.8, "peak_gb": 1.0, "throughput_tps": 1.0, "latency_ms": 1.0}

    monkeypatch.setattr(ep, "_run_cell", fake_run_cell)
    monkeypatch.setattr(ep, "sample_settled_baseline", lambda: 2.71)
    monkeypatch.setattr(ep, "read_limits",
                        lambda: SimpleNamespace(safe_threshold_gb=lambda m=2.0: 15.18,
                                                wall_gb=17.18, wired_now_gb=2.71))
    # stored coeffs with negative intercept; floors clamp slopes, seed clamps model_base
    monkeypatch.setattr(ep, "_coeffs", lambda points, stored: (-5.0, 1e-9, 1e-12))
    # tiny cell: with model_base clamped to >=1.0, predicted ~= 2.71+1.0 = 3.71 < 15.18 -> runs
    ep.sweep(con=None, run_id=1, model="m", batches=[1], seqs=[128], repeats=1,
             margin_gb=2.0, on_event=lambda e: None, persist=False)
    assert spawned == [(1, 128)]  # ran (didn't crash on negative c, didn't falsely abort)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_embeddings_benchmark.py -q -k "intercept_gate or model_base_clamped"`
Expected: FAIL — `test_intercept_gate...` fails because the current through-origin gate still skips `(8,4096)`/`(32,2048)`; `test_model_base_clamped...` may error on `_coeffs` arity (current `_coeffs(points)` takes one arg).

- [ ] **Step 3: Rewrite `_coeffs` and remove `_estimate_model_base`** in `wmx_suite/embeddings_probe.py`. Replace the existing `_coeffs` and `_estimate_model_base` functions with:
```python
def _coeffs(points: list[tuple[float, float, float]],
            stored: tuple[float, float, float] | None) -> tuple[float, float, float]:
    """Raw gate coefficients (c, a, b), before floors/clamps. Priority:
      1. in-run 3-param fit once we have >= MIN_FIT_POINTS points (ground truth for the
         current machine state),
      2. a stored calibration profile (seeds the early cells),
      3. the cold sum-over-layers OVER-estimate (safe fallback when we can't trust a fit).
    """
    if len(points) >= MIN_FIT_POINTS:
        fit = _fit_cab(points)
        if fit is not None:
            return fit
    if stored is not None:
        return stored
    return (0.0, A_COLD, B_COLD)
```

- [ ] **Step 4: Bump `MIN_FIT_POINTS` and update the gate block** in `wmx_suite/embeddings_probe.py`. Change the constant:
```python
MIN_FIT_POINTS = 4  # 3-param fit needs >0 DoF; 4th measured cell (1,1024) is trivially safe
```
Then in `sweep`, replace the per-cell coefficient/prediction block (the lines computing `a, b = _coeffs(points)`, `model_base = _estimate_model_base(...)`, and `predicted = ...`) with:
```python
            live_base = sample_settled_baseline()
            c, a, b = _coeffs(points, stored)
            # model_base = fixed residency (fitted/stored intercept), clamped to the seed
            # and monotonic non-decreasing so a later fit can't shrink it unsafely.
            model_base = max(model_base, MODEL_BASE_SEED_GB, c)
            a = max(A_FLOOR, a)
            b = max(B_FLOOR, b)
            x1, x2 = batch * seq, batch * seq * seq
            predicted = live_base + model_base + PRED_SAFETY * (a * x1 + b * x2)
```
Also delete the now-unused `delta`-based model_base seeding: keep the line `points.append((x1, x2, delta))` where `delta = max(0.0, result["os_wired_gb"] - live_base)`, but remove any reference to `_estimate_model_base`. Add a module-scoped `stored` default at the top of `sweep` so this block resolves (full wiring lands in Task 5):
```python
    stored = None  # set from the calibration profile in Task 5
```
Place that line near the other pre-loop initializers (next to `model_base = MODEL_BASE_SEED_GB`).

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_embeddings_benchmark.py -q`
Expected: PASS (new tests pass; the previously-added `test_predictive_skip...`, `test_cold_start_gate...`, `test_monotonic_pruning...` still pass — they don't depend on the intercept and use the cold/over-estimate path or strong synthetic deltas). If `test_predictive_skip_does_not_spawn_unsafe_cell` or `test_monotonic_pruning...` reference `_run_cell`/`_coeffs` with the old 1-arg form, update those call sites in the tests to the 2-arg `_coeffs(points, None)` where applicable. (They monkeypatch `_run_cell`, not `_coeffs`, so they should be unaffected.)

- [ ] **Step 6: Mutation check** — temporarily change the gate `if predicted >= threshold:` to `if False:` and rerun `test_intercept_gate_measures_previously_skipped_safe_cells`; it must FAIL (now `(32,8192)` gets measured). Revert and confirm PASS.

- [ ] **Step 7: Commit**
```bash
git add wmx_suite/embeddings_probe.py tests/test_embeddings_benchmark.py
git commit -m "feat(embeddings): intercept-based gate policy + model_base seed clamp (#21)"
```

---

## Task 5: Profile load + auto-upsert in `sweep`

**Files:**
- Modify: `wmx_suite/embeddings_probe.py` (add `mlx_version`/`ignore_profile` params; load `stored`; upsert at end; import `profiles`)
- Test: `tests/test_embeddings_benchmark.py`

- [ ] **Step 1: Write the failing tests** (append)
```python
def test_sweep_autoupserts_profile_after_run(monkeypatch, tmp_path):
    from wmx_suite import embeddings_probe as ep, profiles
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "suite.db")
    monkeypatch.setattr(profiles, "machine_key",
                        lambda: ("Apple M4 Pro", 25769803776, 15))
    con = db.connect()

    def fake_run_cell(py, model, batch, seq, repeats, margin):
        return {"status": "rung_done", "batch": batch, "seq": seq,
                "os_wired_gb": 2.71 + 0.001 * batch * seq, "peak_gb": 1.0,
                "throughput_tps": 1.0, "latency_ms": 1.0}

    monkeypatch.setattr(ep, "_run_cell", fake_run_cell)
    monkeypatch.setattr(ep, "sample_settled_baseline", lambda: 2.71)
    monkeypatch.setattr(ep, "read_limits",
                        lambda: SimpleNamespace(safe_threshold_gb=lambda m=2.0: 15.18,
                                                wall_gb=17.18, wired_now_gb=2.71))
    run_id = db.start_embeddings_run(con, "org/m", "0.31.2")
    ep.sweep(con, run_id, "org/m", batches=[1], seqs=[128, 256, 512, 1024, 2048],
             repeats=1, margin_gb=2.0, mlx_version="0.31.2", on_event=lambda e: None)
    # >= MIN_FIT_POINTS measured -> a profile was upserted
    assert profiles.embedding_coeffs(con, "org/m", "0.31.2") is not None


def test_sweep_loads_stored_profile(monkeypatch, tmp_path):
    from wmx_suite import embeddings_probe as ep, profiles
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "suite.db")
    monkeypatch.setattr(profiles, "machine_key",
                        lambda: ("Apple M4 Pro", 25769803776, 15))
    con = db.connect()
    profiles.upsert_embedding_coeffs(con, "org/m", "0.31.2",
                                     coef_intercept_gb=1.1, coef_linear=2e-5,
                                     coef_quad=6e-9, n_points=20)

    seen = {}
    real_coeffs = ep._coeffs
    def spy_coeffs(points, stored):
        seen["stored"] = stored
        return real_coeffs(points, stored)
    monkeypatch.setattr(ep, "_coeffs", spy_coeffs)

    def fake_run_cell(py, model, batch, seq, repeats, margin):
        return {"status": "rung_done", "batch": batch, "seq": seq, "os_wired_gb": 2.8,
                "peak_gb": 1.0, "throughput_tps": 1.0, "latency_ms": 1.0}
    monkeypatch.setattr(ep, "_run_cell", fake_run_cell)
    monkeypatch.setattr(ep, "sample_settled_baseline", lambda: 2.71)
    monkeypatch.setattr(ep, "read_limits",
                        lambda: SimpleNamespace(safe_threshold_gb=lambda m=2.0: 15.18,
                                                wall_gb=17.18, wired_now_gb=2.71))
    run_id = db.start_embeddings_run(con, "org/m", "0.31.2")
    ep.sweep(con, run_id, "org/m", batches=[1], seqs=[128], repeats=1, margin_gb=2.0,
             mlx_version="0.31.2", on_event=lambda e: None)
    assert seen["stored"] == (1.1, 2e-5, 6e-9)  # the gate was seeded from the profile


def test_sweep_ignore_profile_skips_load_but_still_upserts(monkeypatch, tmp_path):
    from wmx_suite import embeddings_probe as ep, profiles
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "suite.db")
    monkeypatch.setattr(profiles, "machine_key",
                        lambda: ("Apple M4 Pro", 25769803776, 15))
    con = db.connect()
    profiles.upsert_embedding_coeffs(con, "org/m", "0.31.2",
                                     coef_intercept_gb=9.9, coef_linear=1.0,
                                     coef_quad=1.0, n_points=1)
    seen = {}
    real_coeffs = ep._coeffs
    monkeypatch.setattr(ep, "_coeffs",
                        lambda points, stored: seen.setdefault("stored", stored) or real_coeffs(points, stored))

    def fake_run_cell(py, model, batch, seq, repeats, margin):
        return {"status": "rung_done", "batch": batch, "seq": seq,
                "os_wired_gb": 2.71 + 0.001 * batch * seq, "peak_gb": 1.0,
                "throughput_tps": 1.0, "latency_ms": 1.0}
    monkeypatch.setattr(ep, "_run_cell", fake_run_cell)
    monkeypatch.setattr(ep, "sample_settled_baseline", lambda: 2.71)
    monkeypatch.setattr(ep, "read_limits",
                        lambda: SimpleNamespace(safe_threshold_gb=lambda m=2.0: 15.18,
                                                wall_gb=17.18, wired_now_gb=2.71))
    run_id = db.start_embeddings_run(con, "org/m", "0.31.2")
    ep.sweep(con, run_id, "org/m", batches=[1], seqs=[128, 256, 512, 1024, 2048],
             repeats=1, margin_gb=2.0, mlx_version="0.31.2", ignore_profile=True,
             on_event=lambda e: None)
    assert seen["stored"] is None  # did NOT load the stored profile
    # but it re-fit and overwrote it (n_points now reflects this run, not the seeded 1)
    assert db.get_embedding_profile(
        con, profiles.embedding_machine_key("org/m", "0.31.2"))["n_points"] >= 4
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_embeddings_benchmark.py -q -k "autoupserts or loads_stored or ignore_profile"`
Expected: FAIL (`sweep` has no `mlx_version`/`ignore_profile` kwargs; no profile written).

- [ ] **Step 3: Wire persistence into `sweep`.** In `wmx_suite/embeddings_probe.py`:
  1. Add the import: change `from . import config, db` to `from . import config, db, profiles`.
  2. Update the signature:
```python
def sweep(con, run_id: int, model: str, batches=None, seqs=None, repeats: int = 3,
          margin_gb: float | None = None, *, mlx_version: str | None = None,
          ignore_profile: bool = False, on_event=None, persist: bool = True) -> dict:
```
  3. Replace the `stored = None  # set from the calibration profile in Task 5` line (added in Task 4) with the real load:
```python
    stored = None
    if not ignore_profile and mlx_version is not None and con is not None:
        stored = profiles.embedding_coeffs(con, model, mlx_version)
```
  4. Before each of the existing `return` statements that report the summary (the normal end-of-sweep return and the early `error` return — NOT the `preflight_abort` return, which measured nothing), add the upsert. Simplest: just before the final `return {...}` at the end of `sweep`, add:
```python
    if (persist and con is not None and mlx_version is not None
            and len(points) >= MIN_FIT_POINTS):
        fit = _fit_cab(points)
        if fit is not None:
            c, a, b = fit
            profiles.upsert_embedding_coeffs(con, model, mlx_version,
                                             coef_intercept_gb=c, coef_linear=a,
                                             coef_quad=b, n_points=len(points))
```
  (The `preflight_abort` and `error` early returns measured `< MIN_FIT_POINTS` cells in practice; guarding on `len(points) >= MIN_FIT_POINTS` makes upsert-at-final-return correct and avoids writing a profile from a failed/aborted run.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_embeddings_benchmark.py -q -k "autoupserts or loads_stored or ignore_profile"`
Expected: PASS.

- [ ] **Step 5: RULE #1 re-check with profile seeding (mutation)** — confirm seeding can't spawn an unsafe cell. Temporarily edit `test_sweep_loads_stored_profile`-style data is not enough; instead run the existing `test_predictive_skip_does_not_spawn_unsafe_cell` and `test_intercept_gate_measures_previously_skipped_safe_cells` and confirm both still PASS. Then mutate the gate (`if predicted >= threshold:` → `if False:`) and confirm `test_intercept_gate...` FAILS. Revert.

- [ ] **Step 6: Commit**
```bash
git add wmx_suite/embeddings_probe.py tests/test_embeddings_benchmark.py
git commit -m "feat(embeddings): load+auto-upsert calibration profile in sweep (#21)"
```

---

## Task 6: CLI wiring (`--ignore-profile` + profile note)

**Files:**
- Modify: `wmx_suite/cli.py` (`cmd_benchmark_embeddings` + subparser)
- Test: `tests/test_embeddings_benchmark.py`

- [ ] **Step 1: Write the failing test** (append)
```python
def test_cmd_benchmark_embeddings_passes_mlx_version_and_ignore_flag(monkeypatch, tmp_path, capsys):
    from wmx_suite import cli, db, embeddings_probe
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "suite.db")
    captured = {}

    def fake_sweep(con, run_id, model, batches, seqs, repeats, margin_gb=None, *,
                   mlx_version=None, ignore_profile=False, on_event=None, persist=True):
        captured["mlx_version"] = mlx_version
        captured["ignore_profile"] = ignore_profile
        return {"model": model, "run_id": run_id, "n_cells_measured": 0, "n_cells_skipped": 0}

    monkeypatch.setattr(embeddings_probe, "sweep", fake_sweep)
    args = SimpleNamespace(model="org/m", batches="1", seqs="128", repeats=1,
                           margin=None, ignore_profile=True)
    cli.cmd_benchmark_embeddings(args)
    assert captured["ignore_profile"] is True
    assert captured["mlx_version"]  # a real mlx version string was passed through
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_embeddings_benchmark.py::test_cmd_benchmark_embeddings_passes_mlx_version_and_ignore_flag -q`
Expected: FAIL (`SimpleNamespace` has no `ignore_profile` consumed / `sweep` called without `mlx_version`).

- [ ] **Step 3: Update `cmd_benchmark_embeddings`** in `wmx_suite/cli.py`. After `run_id = db.start_embeddings_run(con, args.model, mlx_version)`, add a profile note, and change the `embeddings_probe.sweep(...)` call to pass the new kwargs:
```python
    ignore_profile = getattr(args, "ignore_profile", False)
    from . import profiles
    if not ignore_profile and profiles.embedding_coeffs(con, args.model, mlx_version):
        print("  calibration profile: loaded (seeding gate)")
    else:
        print("  calibration profile: none — cold start"
              + (" (--ignore-profile)" if ignore_profile else ""))

    summary = embeddings_probe.sweep(
        con, run_id, args.model, batches=batches, seqs=seqs,
        repeats=args.repeats, margin_gb=margin_val, mlx_version=mlx_version,
        ignore_profile=ignore_profile, on_event=render,
    )
```

- [ ] **Step 4: Register `--ignore-profile`** on the `benchmark-embeddings` subparser in `wmx_suite/cli.py` (add after the `--margin` line):
```python
    p.add_argument("--ignore-profile", action="store_true",
                   help="ignore any stored calibration profile (cold start); still re-fits")
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_embeddings_benchmark.py::test_cmd_benchmark_embeddings_passes_mlx_version_and_ignore_flag -q`
Expected: PASS. Also confirm `uv run wmx-suite benchmark-embeddings --help` lists `--ignore-profile`.

- [ ] **Step 6: Commit**
```bash
git add wmx_suite/cli.py tests/test_embeddings_benchmark.py
git commit -m "feat(cli): embeddings --ignore-profile + calibration-profile note (#21)"
```

---

## Task 7: Full verification

**Files:** none (verification only)

- [ ] **Step 1: Full suite**

Run: `uv run pytest -q`
Expected: all pass (existing + new). If any pre-existing embeddings test referenced the removed `_fit_ab`/`_estimate_model_base` or 1-arg `_coeffs`, fix the call site (these were updated in Tasks 3–4; confirm none remain).

- [ ] **Step 2: Compile check**

Run: `uv run python -m compileall -q wmx_suite tests`
Expected: clean.

- [ ] **Step 3: CLI smoke (no model load)**

Run: `uv run wmx-suite benchmark-embeddings --help`
Expected: help lists `--model/--batches/--seqs/--repeats/--margin/--ignore-profile`.

- [ ] **Step 4: Comment on the issue (do NOT auto-close; needs a real on-device run)**
```bash
gh issue comment 21 --body "Implemented: intercept (3-param) fit so fixed model residency no longer inflates the linear slope, + persisted per-(machine,MLX,model) calibration profile auto-upserted each run and loaded to seed the gate. Safety unchanged (in-parent gate, one-layer floors, PRED_SAFETY, per-cell live baseline). Regression test drives the real sweep over Run-2 data and asserts previously-skipped safe cells (b8×s4096, b32×s2048) are now measured while the danger corner (b32×s8192) is still skipped. Hardware-free tests pass; pending a real on-device run to confirm the gate now maps more of the surface."
```

---

## Notes for the implementer
- **First real run is separate and observed** (RULE #1): after merge, a live `uv run wmx-suite benchmark-embeddings` should be run with the user. The first run cold-starts (no profile yet), fits, and upserts; the *second* run will load that profile.
- **Why the regression test is faithful:** it monkeypatches `_run_cell` to replay the real Run-2 OS-wired values through the *actual* `sweep`/gate, so it exercises the real incremental fit. Under the old through-origin fit it fails (b8×s4096 skipped); under the intercept fit it passes.
- **`B_FLOOR` is the load-bearing safety backstop:** at `b32×s8192`, `B_FLOOR·x2 ≈ 51 GB` (3.4× threshold) regardless of any fitted/stored coefficient — the danger corner can never be spawned.
