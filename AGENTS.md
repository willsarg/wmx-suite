# Agent guide — Will's MLX Suite

Orientation for any AI agent working in this repo. Read this before running anything.

## What this project is

A custom memory/stress bench for local MLX inference on **Apple M-series Macs**. It finds
each model's **safe context ceiling** so we never run a model+context that crashes the
machine. It does this by **extrapolating from safe measurements**, never by probing into
the danger zone.

Every machine-specific number (the crash wall, ambient baseline, per-model fits) is
**measured live on whatever machine the suite runs on** — nothing is hardcoded to one
host. The **primary testbed is Will's M4 Pro MacBook (24 GB)**: it generates the reference
data and the example numbers below, but the suite is meant to be trustworthy across
M-series SKUs.

## RULE #1 — NEVER CRASH THE LAPTOP

This overrides convenience, speed, and completeness. Concretely:

- **Never launch a model run whose predicted OS-wired peak exceeds the safe threshold**
  (`wall − margin`, default margin 2 GB; on the testbed = 15.18 GB). Use
  `probe.characterize()` / the pre-flight gate; do not call `mlx_lm` directly at high
  context to "just see."
- **The crash wall is `max_recommended_working_set_size`**, not total RAM — read live per
  machine and used as an exact, measured value, never rounded toward (on the testbed it is
  17.18 GB). MLX wires (non-swappable) Metal buffers; swap is ~1 GB, so crossing the wall
  can **hard-lock or kernel-panic the machine**, not fail gracefully.
- **Do not `sudo sysctl iogpu.wired_limit_mb=...` to raise the ceiling** without explicit
  user approval — it eats the OS floor and makes a hard lock *more* likely.
- When in doubt, probe **smaller** first and let the extrapolation tell you the ceiling.

## Core facts the design encodes (don't re-derive, don't contradict)

- **Danger metric = OS-wired high-water** (`vm_stat` "Pages wired down"). MLX's
  `get_peak_memory()` undercounts true footprint by ~40% (excludes the buffer cache,
  which the OS still wires). `mlx_true = active + cache` is closer but still misses the
  prefill transient.
- **The prefill spike** (transient buffers while chewing the prompt) drives the crash and
  grows several× faster than steady-state KV. It's invisible in MLX's reported peak.
- **Two memory classes:**
  - **RotatingKVCache** (sliding-window: Gemma, GPT-OSS) → **cannot quantize KV**.
    `--kv-bits 4` raises `NotImplementedError` at `quantized_kv_start` (5000 tokens), so
    it crashes any longer prompt. These models must run **fp16 KV**. Their sliding window
    caps most layers, so growth is already gentle.
  - **standard KVCache** (linear+full: Qwen3.5 9B/27B) → quantizes fine; 4-bit lowers
    high-context memory but has a transient spike at the 5k quantization threshold.
  - `models.describe()` classifies this automatically — trust `can_quantize_kv`.
- **Only `full_attention` layers** grow KV with context (sliding = window-capped,
  linear = fixed recurrent state).
- **Measure in isolated subprocesses** — one fresh process per context. Reusing a process
  leaves wired residue that inflates the next reading (`probe_worker.py` exists for this).

## Commands

```bash
uv sync                                   # install deps into .venv
uv run wmx-suite system                   # machine wall, swap, baseline
uv run wmx-suite health                   # live pressure + per-model go/no-go (read-only)
uv run wmx-suite scan                     # register mlx-community models from HF cache
uv run wmx-suite show <hf_id>             # architecture + memory class
uv run wmx-suite characterize <hf_id>     # SAFE probe -> fitted ceiling (use this; --speed quick|standard|full)
uv run wmx-suite calibrate                # seed this machine's cold-start overhead profile
uv run wmx-suite list                     # ceilings from the DB
uv run wmx-suite run --model <hf_id> ...  # SAFE launch of mlx_lm.generate (use this, not mlx_lm directly)
uv run wmx-suite web                      # launch the Flask web UI dashboard (default port 5001)
```

**`run` is the only sanctioned way to launch a model** (all logic lives in `launcher.py`).
It picks `--kv-bits` by cache type (omits it for RotatingKVCache models that can't
quantize), caps `--max-kv-size` from the measured ceiling against the live baseline, and
refuses if the model would breach the wall on load. Use `--dry-run` to inspect the plan;
`--force` overrides a refusal at the user's own risk. Never call `mlx_lm.generate` directly
at non-trivial context.

## Conventions (Will's global prefs — enforce them)

- **Python packages: use `uv` only. NEVER `--break-system-packages`.**
- **HuggingFace CLI: use `hf`, not the deprecated `huggingface-cli`.**
- Stick to **`mlx-community`** models.
- SQLite is the datastore (`data/suite.db`, gitignored). Flask web UI is under the `web` extra (run `uv run wmx-suite web` to launch).
- The global safety margin defaults to 2 GB and can be set with
  `WMX_SUITE_MARGIN_GB`; an explicit `--margin` takes precedence. Reject negative,
  NaN, or infinite values.
- Match **production inference settings** when measuring: `run` uses `--kv-bits 4` with
  `kv_group_size=64`, `quantized_kv_start=5000` — but only for quantizable models.

## Testing requirements

Tests are part of every behavior change, not follow-up work:

- **Bug fixes require a regression test** that fails for the original bug and passes
  with the fix.
- **New features and changed behavior require tests** for the normal path, relevant
  boundaries, and refusal/error behavior.
- **Safety-critical changes** (`probe.py`, `launcher.py`, `probe_worker.py`, `system.py`,
  model cache classification, or launch argument handling) require hardware-free unit
  tests for the safety boundary. Never use a risky model run as the only proof.
- Update existing tests when an intentional behavior change invalidates their
  assumptions. Do not weaken or delete a test merely to make the suite pass.
- If an automated test is genuinely impractical, explain why in the PR and provide a
  deterministic manual verification procedure. This is an exception, not the default.

Before declaring work complete, run:

```bash
uv run pytest -q
uv run python -m compileall -q wmx_suite tests
```

Run the narrowest relevant test while developing, then run the full suite before
commit/push. Tests must not load models, invoke MLX runtime allocation, probe live
memory, or access the production database unless the test is explicitly designed and
reviewed as an integration test.

## Architecture

```
wmx_suite/
  config.py         # validated runtime defaults (for example WMX_SUITE_MARGIN_GB)
  system.py         # device wall, swap, current wired memory
  models.py         # HF-cache config reader + memory-class classifier
  db.py             # SQLite schema: models, probe_runs, measurements, fits, generation_log
  probe_worker.py   # ONE isolated (model, context) measurement -> JSON line
  probe.py          # safe ramp orchestrator + linear fit + ceiling solve
  cli.py            # entry point
```

Data flow: `scan/describe` → register model → `characterize` ramps context safely,
storing each measurement and a fitted line (`os_wired = intercept + slope·context`) →
solve for safe ceiling and hard wall.

`characterize --speed {quick,standard,full}` (default `standard`) trades fit granularity
for fewer cold model loads — the dominant cost is loads (= rungs × repeats), not algorithm
complexity. Presets live in `probe.SPEED_PRESETS`; `resolve_speed()` maps a preset to
`(ramp, repeats)` and an explicit `--repeats` overrides the preset's repeats (ramp always
comes from the preset). `quick` gets its ~3× speedup from `repeats=1`, **not** from fewer
rungs: real memory grows super-linearly (KV storage is linear, but attention-prefill
scratch + OS wired overhead bend the curve up), so a sparse low-context ramp would
extrapolate the ceiling *optimistically* — the unsafe direction. `quick` therefore keeps a
mid-dense ramp that spans the bend; the pre-flight gate prunes high rungs anyway. The
preset never touches the safety gate: every rung is still measured and the ramp still stops
before launching any rung predicted to breach the threshold.

## Architecture (launch path)

```
launcher.py   # plan() + predict(): cache-aware kv-bits, --max-kv-size cap, refuse gate
cli.py run    # parses passthrough args, prints the plan, execs mlx_lm.generate (PTY-tee
              # by default to log tok/s; --no-log falls back to bare execvp)
```

`run` is intercepted in `cli.main()` before argparse so it can forward arbitrary flags to
`mlx_lm.generate` (argparse.REMAINDER mishandles leading optionals).

`list` and `run` compare the newest artifact mtime in the model's cached HF snapshots
against the latest characterization run. Unused blobs, mutable `refs/`, and `.no_exist/`
metadata are ignored. A newer cache produces a warning only; it never auto-runs
`characterize`.

Before launch, `run` loads only the tokenizer and counts the effective prompt after the
chat template. It warns above 80% of the selected cap and refuses above the cap. Stdin
and prompt-cache inputs require `--force` because their complete effective prompt cannot
be verified. Qwen3.5 is also refused unless forced while its custom MLX cache ignores
`--max-kv-size`; never describe that planned value as a runtime-enforced cap.

## Known open work

- Calibrate the pre-flight base estimate in `probe.py` (`RESIDENT_FACTOR`,
  `FIXED_OVERHEAD_GB`) and `launcher.PREFILL_SPIKE_MULT` as more models are characterized.
  `calibrate` tunes only the cold-start estimate (the `FIXED_OVERHEAD_GB` term) per machine
  and is floored at the default so it can only tighten the estimate; `characterize` remains
  the per-model per-machine adaptation mechanism.
- `run`'s uncharacterized-model fallback uses a conservative analytic estimate; prefer
  running `characterize` first for any model you'll use seriously.

## Trademarks

MLX, Apple Silicon, Metal, Mac, and macOS are trademarks of Apple Inc. This project is an
independent community tool, not affiliated with or endorsed by Apple. Name references are
descriptive only. See the README "Trademarks" section.
