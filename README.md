# Will's MLX Suite

A custom memory/stress bench for running local MLX models on this M4 Pro (24 GB).

**First rule: never crash the laptop.** The suite finds each model's safe context
ceiling by *extrapolating from measurements taken well below the hardware wall* — it
never probes into the danger zone.

## Why this exists

MLX allocates **wired** (non-swappable) Metal buffers. The real crash ceiling isn't
total RAM — it's the GPU working-set limit (`max_recommended_working_set_size`,
≈ **17.18 GB** on this machine, 67% of 25.77 GB). With almost no swap free, crossing it
can hard-lock the system rather than fail gracefully. So we predict the wall and stay
under it.

### Key facts the design is built on

- **Danger metric is OS-wired high-water** (`vm_stat` "Pages wired down"), *not* MLX's
  `get_peak_memory()`, which undercounts true footprint by ~40% (it excludes the buffer
  cache, which the OS still wires).
- **The prefill transient spike** drives the crash — it grows several× faster than the
  steady-state KV cache and is invisible in MLX's reported peak.
- **Two memory classes:**
  - *Sliding-window* (Gemma, GPT-OSS) → `RotatingKVCache`, which **cannot be quantized**
    (`--kv-bits 4` crashes them past 5000 tokens). They run fp16; their window caps most
    layers so growth is gentle.
  - *Linear+full* (Qwen3.5 9B/27B) → standard cache, quantizes fine; 4-bit lowers
    high-context memory (with a transient spike at the 5k quantization threshold).
- **Only `full_attention` layers** grow KV with context; sliding/linear layers don't.

## Usage

```bash
uv sync                                   # install deps into .venv
uv run wmx-suite system                   # show the machine's wall, swap, baseline
uv run wmx-suite scan                     # register mlx-community models from the HF cache
uv run wmx-suite show <hf_id>             # architecture + memory class
uv run wmx-suite characterize <hf_id>     # safe probe -> fitted context ceiling
uv run wmx-suite list                     # ceilings for everything characterized
uv run wmx-suite run --model <hf_id> ...  # safely launch mlx_lm.generate (replaces mlx_safe)
```

`characterize` refuses to launch any probe whose pre-flight base estimate already
exceeds the safe threshold (this is how oversized models like the 27B are handled —
predicted, never run into the wall).

### `run` — the safe launcher (replaces the old `~/bin/mlx_safe`)

`run` plans a launch and then execs `mlx_lm.generate`. It:

- picks `--kv-bits` by **cache type** — `4` for standard caches, **omitted** for
  RotatingKVCache models (Gemma, GPT-OSS) which can't quantize and would otherwise crash;
- samples the **live settled baseline** and caps `--max-kv-size` at the context where
  `live_base + model_base + slope·c` hits the safe threshold, using the model's **measured**
  curve from `suite.db` (or a conservative estimate, with a warning, if uncharacterized);
- **refuses** to launch if the model would breach the wall just to load (e.g. the 27B);
  `--force` overrides at your own risk, `--dry-run` prints the plan without launching.

```bash
uv run wmx-suite run --model mlx-community/Qwen3.5-9B-OptiQ-4bit --prompt "..." --max-tokens 200
uv run wmx-suite run --dry-run --model <hf_id> --prompt "..."   # inspect the plan only
```

## Layout

```
wmx_suite/
  system.py         # device wall, swap, current wired memory
  models.py         # HF-cache config reader + memory-class classifier
  db.py             # SQLite schema: models, probe_runs, measurements, fits
  probe_worker.py   # ONE isolated (model, context) measurement -> JSON
  probe.py          # safe ramp orchestrator + linear fit + ceiling solve
  cli.py            # command-line entry point
data/suite.db       # results (gitignored)
```

## Status

v0 scaffold. Validated methodology: predicted Gemma's ceiling to within 0.5% from safe
probes. Calibration of the pre-flight base estimate refines as more models are run.
Flask UI (`web` extra) is optional and not yet built.

## Trademarks

MLX, Apple Silicon, Metal, Mac, and macOS are trademarks of Apple Inc. This project is
an independent, community tool — it is **not affiliated with, endorsed by, or sponsored
by Apple Inc.** References to these names are descriptive only, to indicate the
technologies the suite works with. All other trademarks are the property of their
respective owners.
