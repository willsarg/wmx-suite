# Design: `wmx-suite calibrate` — per-machine cold-start constants

Issue: #9. Status: design approved (sections 1–2), full spec below.

## Goal

Make the **cold-start base estimate** trustworthy on any Apple Silicon SKU, not
just the M4 Pro 24 GB it was tuned on. The estimate is the only M4-flavored part
of the safety path; the wall, the ambient baseline, and per-model fits already
adapt at runtime. `calibrate` measures this machine's real load-time overhead
once and stores it in a per-machine profile that the cold-start estimators read.

## Non-goals (explicitly out of scope)

- **Deriving `PREFILL_SPIKE_MULT`** — the prefill transient only appears at large
  context, is not measurable from small safe rungs, and is only a fallback
  heuristic that goes unused once a model is characterized. Stays a conservative
  hardcoded default.
- **Per-model curves** — that remains `characterize`'s job. `characterize` is the
  per-machine adaptation mechanism for any specific model.
- **The `2.5 GB` baseline floor** in `estimate_base_gb` — it guards the *live*
  sampled baseline, not a per-SKU constant; left hardcoded.
- **Auto-downloading a model**, and **the community profile registry** (a possible
  future build-on, not this work).

## Decisions (from brainstorming)

1. **Measurement method:** fix `RESIDENT_FACTOR` at its default (~1.05 — weights
   map ~1:1 to memory) and *measure only* `FIXED_OVERHEAD_GB`. A single tiny model
   gives one equation for the base at context→0; with the factor fixed, overhead
   is the residual. (Separating both would need ≥2 models — rejected as more setup
   and complexity for little gain on the genuinely-universal factor.)
2. **Profile key:** `(device_name, total_ram_bytes, macos_major)`. A change in chip,
   installed RAM, or macOS major version invalidates the profile and re-prompts
   calibration.
3. **Model choice:** auto-pick the smallest causal `mlx-community` model in the HF
   cache; `--model <id>` overrides; clear error if none cached; never auto-download.
4. **No-profile UX:** fall back to current M4-Pro defaults (no behavior change for
   existing users) and, when an *estimated* plan is produced with no matching
   profile, print a one-line warning suggesting `calibrate`. `characterize` still
   overrides everything.

## Architecture

```
wmx_suite/
  profiles.py        # NEW: machine key, system_profiles accessors,
                     #      cold_start_constants() resolver + default fallback
  db.py              # + system_profiles table (additive CREATE IF NOT EXISTS)
                     # + upsert_profile() / get_profile()
  system.py          # + macos_major() helper; total RAM already in device_info()
  probe.py           # + calibrate(); estimate_base_gb() reads cold_start_constants()
  launcher.py        # plan() reads cold_start_constants() instead of module consts
  cli.py             # + cmd_calibrate; run/health emit the no-profile warning
```

`profiles.py` is the single seam. Both estimators call one resolver, and the
warning logic has one home. Defaults live in `profiles.py`; `probe.py`/`launcher.py`
import from there (removes the current `launcher → probe` constant import).

### `profiles.py`

```python
DEFAULT_RESIDENT_FACTOR = 1.05      # weights→resident memory; loose prior (effective
                                    # factor measured 0.88–1.10 across models). HELD FIXED.
DEFAULT_FIXED_OVERHEAD_GB = 1.0     # framework/Metal/python base; loose M4-Pro prior
                                    # ("calibrated loosely" per probe.py), and the floor
                                    # below which a calibrated overhead is never stored.

def machine_key() -> tuple[str, int, int]:
    """(device_name, total_ram_bytes, macos_major) for the current machine."""
    # device_name + memory_size from mx.device_info(); macos_major from
    # system.macos_major(). macOS detection failure -> macos_major = 0 (stable key).

def cold_start_constants(con) -> tuple[float, float, str]:
    """Return (resident_factor, fixed_overhead_gb, source).
    source == "profile" if a row matches machine_key(), else "default"
    (DEFAULT_RESIDENT_FACTOR, DEFAULT_FIXED_OVERHEAD_GB)."""
```

`cold_start_constants` takes the open `con` so callers control DB lifetime and tests
can inject an in-memory DB. It builds the key itself via `machine_key()` (which reads
`mx.device_info()` + `system.macos_major()` — total RAM is needed in *bytes*, which
`SystemLimits.total_gb` doesn't carry), so it takes no `limits`.

### `system.py`

```python
def macos_major() -> int:
    """Major macOS version (e.g. 15), or 0 if undetectable."""
    # platform.mac_ver()[0].split(".")[0]; guard empty/parse errors -> 0.
```

`device_limits()` already exposes `memory_size` (total RAM) and `device_name`.

### `db.py` — new table + accessors

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

- `upsert_profile(con, key, *, resident_factor, fixed_overhead_gb, model_id, n_points, mlx_version)`
  — `INSERT ... ON CONFLICT(primary key) DO UPDATE`, sets `calibrated_at` to now.
- `get_profile(con, key) -> dict | None`.

Additive `CREATE TABLE IF NOT EXISTS`, consistent with existing schema; no migration
needed. `resident_factor` is stored even though fixed, for transparency and future
change.

## Measurement flow — `probe.calibrate(...)`

`calibrate(model: str | None = None, *, margin_gb=None, repeats=DEFAULT_REPEATS,
worker_python=None, verbose=True) -> dict`

1. **Pick model.** If `model` given, use it (must be causal — else error). Else
   `models.scan_cache()`, `describe()` each, keep causal ones, pick smallest
   `weights_gb`. None cached → `SystemExit` instructing the user to download a small
   `mlx-community` model (e.g. a 0.5–1.5B) or pass `--model`.
2. **Resolve** `machine_key()`, `read_limits()`, `threshold`, `kv_bits` (per cache
   type, as `characterize` does).
3. **Two small safe rungs** — `CALIBRATE_RAMP = [512, 2048]` tokens — each through
   the existing `_measure_rung()` (isolated subprocess, N-repeat median over a
   settled baseline). Same pre-flight gate as `characterize` guards every launch;
   for a tiny model at ≤2048 tokens this is deep in the safe zone. If a rung's
   pre-flight refuses or the rung errors, abort with that reason (do not force).
4. **Solve overhead at c→0.** `_linfit()` the two `(ctx_k, delta)` points → intercept
   `base_delta0`. Then
   `fixed_overhead = max(DEFAULT_FIXED_OVERHEAD_GB, base_delta0 − DEFAULT_RESIDENT_FACTOR * weights_gb)`.
   Using the fitted intercept (not a raw rung) removes the `slope·c` KV contribution,
   so KV growth isn't folded into overhead. **Floored at the default (1.0 GB):**
   calibration may only make the cold-start estimate *more* conservative than today's
   M4 baseline, never less — for a purely-estimated run, too-low overhead is the unsafe
   direction. (See "Why the fixed factor is acceptable" below for why the residual is
   measured cleanly despite the factor being held constant.)

**Why the fixed factor is acceptable.** Committed `fits` data shows the effective
resident factor varies ~0.88–1.10 across measured models, so a single fixed 1.05 is an
approximation. Choosing the *smallest* model neutralizes this for the overhead reading:
with tiny weights, `1.05·weights` is small, so `base_delta0 ≈ true overhead` almost
regardless of the factor. The factor remains an approximation for *large* uncharacterized
models (≈ ±0.05·weights of slop), but that is inherent to a conservative cold-start
heuristic — `characterize` measures the true `model_base`/`slope` and overrides it
entirely, and the run-time "characterize now?" prompt nudges users there.
5. **Persist** via `upsert_profile(...)` (`resident_factor = DEFAULT_RESIDENT_FACTOR`,
   `fixed_overhead_gb = fixed_overhead`, model id, `n_points = 2`, mlx version).
6. **Return / print** a summary dict: machine key, measured overhead vs. the
   `DEFAULT_FIXED_OVERHEAD_GB`, model used, n_points.

`calibrate` reuses `_measure_rung`, `_linfit`, `models.describe`, `read_limits`,
`sample_settled_baseline` — no new measurement primitive.

## Reading the profile in the estimators

`probe.estimate_base_gb(info, limits, con)`:
```python
factor, overhead, _ = profiles.cold_start_constants(con)
os_baseline = max(limits.wired_now_gb, 2.5)
return os_baseline + info.weights_gb * factor + overhead
```

`launcher.plan(hf_id, *, margin_gb=None)` (estimated branch only):
```python
factor, overhead, cold_source = profiles.cold_start_constants(con)
model_base = info.weights_gb * factor + overhead
slope = _estimated_slope_gb_per_k(info)   # unchanged: fp16 KV * PREFILL_SPIKE_MULT
source = "estimated"
p["cold_start_profile"] = cold_source     # "profile" | "default"
```

The measured-fit branch (`source == "measured"`) is unchanged — characterized models
never touch the cold-start constants, so `cold_start_profile` is only meaningful when
`source == "estimated"`.

Adding `con` to `estimate_base_gb` touches all three of its callers: `characterize`
(`probe.py:154`) and two test sites (`tests/test_probe_math.py:71,76`) which currently
pass only `(info, limits)`. Pass `con` (or accept `con=None` → open one) consistently.

## No-profile warning wiring

- **`run` (`cli._run`)**: after planning, when `p["source"] == "estimated"` and
  `p.get("cold_start_profile") == "default"`, print to stderr:
  `[run] WARNING: using default cold-start constants tuned for Apple M4 Pro; run
  'wmx-suite calibrate' to tune them for this machine.`
  Not shown for characterized (measured) runs or when a profile exists.
- **`health` (`cli.cmd_health`)**: if `get_profile(con, machine_key())` is None, print
  one banner line near the top:
  `No calibration profile for <device>/<RAM>/<macOS>; cold-start estimates use
  M4-Pro defaults. Run 'wmx-suite calibrate'.`
  Per-model verdicts are unchanged.
- **`system` (`cli.cmd_system`)**: print one line stating whether a profile is active
  for this machine and, if so, its measured overhead and `calibrated_at` (read-only,
  informational). Note: `cmd_system` opens no DB today, so this adds a `db.connect()`
  there (read-only).

## CLI

```
wmx-suite calibrate [--model <hf_id>] [--margin GB]
```
`cmd_calibrate(args)`: resolves margin via `_configured_margin`, calls
`probe.calibrate(...)`, prints the summary, exits non-zero on the no-model / aborted
cases. Registered in `_main_argparse` alongside the other subcommands. README/AGENTS
command lists updated to mention `calibrate` and to state that `characterize` remains
the per-model per-machine adaptation.

## Error handling & edge cases

- **No causal model cached** → clear `SystemExit` with remediation (download a small
  model or `--model`).
- **`--model` is non-causal / not in cache** → error (reuse `describe` + `is_causal`).
- **macOS version undetectable** → `macos_major = 0`; key stays stable and consistent.
- **Pre-flight refuses even the smallest rung** (machine under heavy load) → abort with
  the refusal reason; do not force.
- **Negative / low residual** (tiny model where `factor·weights` meets or exceeds the
  measured base, or just a leaner machine) → overhead floored at `DEFAULT_FIXED_OVERHEAD_GB`
  (1.0), per the safety policy that calibration only tightens, never loosens, the estimate.
- **Re-running `calibrate`** → upsert replaces the row for this machine key.
- **Stale profile after macOS/RAM change** → key no longer matches; falls back to
  defaults and the warning reappears, prompting re-calibration. (We do not delete old
  rows; they're harmless and keyed distinctly.)

## Testing (all hardware-free)

- **`profiles`**: `machine_key()` parsing incl. macOS-undetectable → 0;
  `cold_start_constants()` hit (in-memory DB row) vs miss (returns defaults + source).
- **`db`**: `upsert_profile`/`get_profile` round-trip; upsert replaces.
- **`probe.calibrate`**: monkeypatch `_measure_rung` to return synthetic deltas and
  assert the intercept solve + overhead formula (incl. clamp ≥ 0); monkeypatch
  `scan_cache`/`describe` for auto-pick-smallest and the no-model error; assert it
  persists a profile row. No model load, no MLX allocation, no production DB.
- **`launcher.plan`**: with an injected profile row → uses stored overhead and sets
  `cold_start_profile="profile"`; with none → defaults and `"default"`. Measured-fit
  branch still ignores cold-start.
- **`probe.estimate_base_gb`**: reads profile overhead when present.
- **`cli`**: `run` warning emitted only on estimated+default (not measured, not with a
  profile); `health` banner when no profile; `cmd_calibrate` happy path + no-model
  error (monkeypatch `probe.calibrate`).

Run `uv run pytest -q` and `uv run python -m compileall -q wmx_suite tests` before
declaring complete.

## Files touched

New: `wmx_suite/profiles.py`, tests `tests/test_profiles.py`.
Modified: `wmx_suite/db.py`, `system.py`, `probe.py`, `launcher.py`, `cli.py`;
tests `test_db.py`, `test_launcher.py`, `test_models.py`/`test_probe_math.py` (as
relevant), `test_cli.py`; `README.md`, `AGENTS.md`.
