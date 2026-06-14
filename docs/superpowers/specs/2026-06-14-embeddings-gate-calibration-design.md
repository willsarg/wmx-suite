# Calibrated Embeddings Gate — Design Spec

**Issue:** #21 — embeddings predictive-skip gate is over-conservative vs the measured surface
**Date:** 2026-06-14
**Status:** Approved design (validated by 3 sub-agent audits), pending implementation plan
**Builds on:** `docs/superpowers/specs/2026-06-14-embeddings-benchmark-design.md`

## 1. Problem

The first live runs validated RULE #1 (max OS-wired 4.79 GB vs 15.18 threshold; the gate
skipped the true danger corner), but the gate is **far too conservative**: it skipped
`b8×s4096` (predicted 18.27 GB) and `b32×s2048` (predicted 15.33 GB) when the entire
measured surface is flat (delta over baseline only 1.3–2.1 GB across all 32 cells).

**Root cause (validated against the real Run-2 data):** the fit `delta = a·x1 + b·x2` is
**through the origin** (no intercept), so the fixed ~1.1 GB model-residency offset is forced
into the slope terms. Because the live gate fits **incrementally on the cells measured so
far** (a small, low-`x` partial dataset), this inflates the **linear coefficient `a`**,
which then explodes when extrapolated to a larger cell. (The quadratic `b_fit` is actually
*below* `B_FLOOR`, so the one-layer floor already governs the `seq²` term — the over-
prediction is the linear term, not the quadratic.)

## 2. Goal

Add an **intercept** to the fit so fixed residency stops contaminating the slopes, and
**persist** the fitted coefficients per (machine, MLX, model) so later runs are accurate
from the start — while keeping the never-crash guarantee fully intact.

## 3. Decisions (locked during brainstorming)

- **Scope:** persisted per-machine calibration (subsumes the within-run intercept fix).
- **Trigger:** auto-fit + auto-upsert at the end of each successful run (no separate
  command); `--ignore-profile` bypasses *loading* (still upserts).
- **Profile validity:** key by `(device_name, total_ram_bytes, macos_major, mlx_version,
  model_id)`. Any mismatch (e.g. MLX upgrade) → no hit → conservative cold-start fallback.

## 4. The engine fix — intercept in the fit

Replace the 2-param through-origin fit with a **3-param fit with intercept**:
`delta = c + a·(batch·seq) + b·(batch·seq²)`.

- `_fit_ab(points)` → `_fit_cab(points)` returning `(c, a, b)` via 3×3 normal equations
  (or numpy-free Gaussian elimination), or `None` when **under-determined / singular**
  (fewer than 3 distinct feature rows, or zero determinant).
- **`MIN_FIT_POINTS = 4`** (raised from 3): a 3-param fit on exactly 3 points has zero
  degrees of freedom; 4 points gives 1 DoF for noise averaging. The 4th measured cell is
  `(1, 1024)`, trivially safe.
- **`model_base = max(MODEL_BASE_SEED_GB, c)`** — REQUIRED safety clamp. The fitted (or
  stored) intercept can be negative from noisy/flat data; the clamp guarantees the
  fixed-residency term in the gate is never below the physical seed. This replaces the old
  `_estimate_model_base` residual-max heuristic.
- Slopes are still clamped to the physical one-layer floors: `a = max(A_FLOOR, a_fit)`,
  `b = max(B_FLOOR, b_fit)` (`A_FLOOR = hidden·2/1e9`, `B_FLOOR = heads·2/1e9`).
- Cold-start (`< MIN_FIT_POINTS` measured AND no usable profile) still uses the
  `A_COLD/B_COLD` sum-over-layers **over-estimate** (never the floor) as the safe fallback.
- Prediction unchanged in shape:
  `predicted = live_base + model_base + PRED_SAFETY·(a·x1 + b·x2)`.

## 5. Persistence layer (mirrors `system_profiles`)

New table in `db.py` `SCHEMA` (auto-created by `executescript`; no migration needed):
```sql
CREATE TABLE IF NOT EXISTS embedding_profiles (
    device_name      TEXT NOT NULL,
    total_ram_bytes  INTEGER NOT NULL,
    macos_major      INTEGER NOT NULL,
    mlx_version      TEXT NOT NULL,
    model_id         TEXT NOT NULL,
    coef_intercept_gb REAL NOT NULL,
    coef_linear      REAL NOT NULL,
    coef_quad        REAL NOT NULL,
    n_points         INTEGER NOT NULL,
    created_at       TEXT,
    PRIMARY KEY (device_name, total_ram_bytes, macos_major, mlx_version, model_id)
);
```
New db functions (5-tuple key; `system_profiles`' helpers are 3-tuple-specific, so these
are separate, parallel functions):
- `upsert_embedding_profile(con, key5, *, coef_intercept_gb, coef_linear, coef_quad, n_points)` — `INSERT ... ON CONFLICT(<5 key cols>) DO UPDATE SET ...`.
- `get_embedding_profile(con, key5) -> dict | None`.

`profiles.py` accessor:
- `embedding_machine_key(model_id, mlx_version) -> tuple` = `machine_key()` (3) + `(mlx_version, model_id)`.
- `embedding_coeffs(con, model_id, mlx_version) -> tuple[float, float, float] | None` —
  returns stored `(c, a, b)` or `None`.

## 6. Sweep flow changes (`embeddings_probe.sweep`)

`sweep` already receives `model`; add the ability to read/write the profile. It needs
`mlx_version` (pass it in from the CLI, like `mlx_version` is obtained there today).

- **Start:** `stored = profiles.embedding_coeffs(con, model, mlx_version)` unless
  `ignore_profile`. If present, it seeds the gate coefficients `(c, a, b)` (clamped:
  `c→max(SEED,c)`, `a→max(A_FLOOR,a)`, `b→max(B_FLOOR,b)`) for cells evaluated before the
  in-run fit is available.
- **Coefficient selection per cell** (`_coeffs` becomes a small policy):
  1. `>= MIN_FIT_POINTS` in-run points → in-run 3-param fit (clamped). *In-run data always
     wins once available* — it is ground truth for the current machine state.
  2. else if `stored` present → stored coeffs (clamped).
  3. else → `A_COLD/B_COLD` cold over-estimate.
- **Ramp/gate/pruning unchanged:** ascending per-row ramp; gate evaluated in the parent
  before spawn; monotonic cross-batch pruning; `model_base` monotonic non-decreasing.
- **End:** if `>= MIN_FIT_POINTS` cells measured, fit final `(c, a, b)` and
  `upsert_embedding_profile`. (Upsert happens even when `ignore_profile` was used for
  loading — that flag only skips reading.)
- **CLI:** `cmd_benchmark_embeddings` passes `mlx_version` into `sweep` and exposes
  `--ignore-profile` (store_true). Print a one-line note when a profile is loaded
  (e.g. `using calibration profile (n=NN, fitted <date>)`) vs `no profile — cold start`.

## 7. Safety invariants (unchanged — defense in depth)

Never weakened, confirmed by the safety audit:
- Gate evaluated **in the parent before any spawn**; worker pre-flight is the backstop.
- `a,b` clamped to physical one-layer floors; `c`/model_base clamped to the seed.
  **`B_FLOOR` alone predicts 51.5 GB at `b32×s8192` (3.4× threshold)** — the quadratic
  backstop is mathematically undefeatable at large cells regardless of any fit/profile.
- `PRED_SAFETY = 1.25` on the compute term + 2 GB wall margin.
- Per-row ascending ramp means a run measures its own small cells before any large cell;
  the stored profile only governs the first few tiny cells, which are safe under any
  clamped coefficients.
- `live_base = sample_settled_baseline()` sampled **per cell** — current OS pressure
  always captured; the profile supplies only the stable scaling coefficients, never
  `live_base`. Higher external pressure tightens the gate (safe direction).
- Profile keyed by exact (machine, mlx_version, model); mismatch → cold fallback.
- No feedback-loop ratchet: floors prevent coefficient drift below physical minimum, and
  the upserted fit derives from max-sampler high-water data (biased high, not low).

## 8. Testing (hardware-free + mutation-checked)

1. **`_fit_cab` recovers known `(c, a, b)`** on synthetic data; under-determined/singular
   → returns `None`.
2. **Regression test proving the fix (incremental condition):** feed the real Run-2 deltas
   for the cells measured *before* `b8×s4096` in the ramp (i.e. b1–b4 all + b8 up to
   s2048), then assert: the through-origin model predicts `b8×s4096` as SKIP (reproduces
   the bug) while the intercept model predicts it SAFE — and BOTH still SKIP `b32×s8192`.
   This must use the incremental partial fit, not a full-32-cell fit.
3. **`model_base` clamp:** a stored/fitted `c < 0` yields `model_base == MODEL_BASE_SEED_GB`
   (negative-c clamp guard).
4. **Profile round-trip:** `upsert_embedding_profile`/`get_embedding_profile` 5-tuple key;
   a key with a different `mlx_version` returns `None` (→ cold fallback).
5. **RULE #1 with profile seeding (mutation-checked):** seeding from a stored profile still
   never spawns a cell predicted ≥ threshold; the worst corner is still skipped. Removing
   the gate must fail this test.
6. **Auto-upsert:** after a sweep with ≥ MIN_FIT_POINTS measured cells, a profile row
   exists with the fitted coeffs and correct key.
7. **`--ignore-profile`:** loading is skipped (cold start) but upsert still happens.
8. All existing embeddings + web safety tests stay green.

## 9. Out of scope (YAGNI)

- No worker, web-template, or default-grid changes (the gate simply stops over-skipping).
- No cross-model coefficient sharing; no profile pruning/expiry beyond key mismatch.
- No change to `system_profiles` or the LLM `calibrate` path.
