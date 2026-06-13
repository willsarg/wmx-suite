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
uv run mlx-suite system                   # show the machine's wall, swap, baseline
uv run mlx-suite scan                     # register mlx-community models from the HF cache
uv run mlx-suite show <hf_id>             # architecture + memory class
uv run mlx-suite characterize <hf_id>     # safe probe -> fitted context ceiling
uv run mlx-suite list                     # ceilings for everything characterized
```

`characterize` refuses to launch any probe whose pre-flight base estimate already
exceeds the safe threshold (this is how oversized models like the 27B are handled —
predicted, never run into the wall).

## Layout

```
wills_mlx_suite/
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
