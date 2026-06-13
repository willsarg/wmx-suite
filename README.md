<div align="center">

# 🧪 Will's MLX Suite

### `wmx-suite` — find each model's safe context ceiling on Apple Silicon, *without crashing the machine*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](./LICENSE)
[![Python](https://img.shields.io/badge/Python-3.12+-3776AB.svg?style=for-the-badge&logo=python&logoColor=white)](./pyproject.toml)
[![Apple Silicon](https://img.shields.io/badge/Apple_Silicon-M1_to_M4-000000.svg?style=for-the-badge&logo=apple&logoColor=white)](#-why-this-exists)
[![Built with MLX](https://img.shields.io/badge/Built_with-MLX-FF6F61.svg?style=for-the-badge)](https://github.com/ml-explore/mlx)
[![packaged with uv](https://img.shields.io/badge/packaged_with-uv-DE5FE9.svg?style=for-the-badge)](https://github.com/astral-sh/uv)
[![PRs welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=for-the-badge)](./CONTRIBUTING.md)
[![Buy me a coffee](https://img.shields.io/badge/Buy_me_a_coffee-FFDD00.svg?style=for-the-badge&logo=buymeacoffee&logoColor=black)](https://www.buymeacoffee.com/willsarg)

</div>

> [!WARNING]
> **First rule: never crash the laptop.** The suite finds each model's safe context
> ceiling by *extrapolating from measurements taken well below the hardware wall* — it
> never probes into the danger zone.

---

## 🧠 Why this exists

MLX allocates **wired** (non-swappable) Metal buffers. The real crash ceiling isn't
total RAM — it's the GPU working-set limit (`max_recommended_working_set_size`,
≈ **17.18 GB** on the reference M4 Pro, 67% of 25.77 GB). With almost no swap free,
crossing it can hard-lock the system rather than fail gracefully. So we **predict the
wall and stay under it.**

### 🔬 Key facts the design is built on

- 📏 **Danger metric is OS-wired high-water** (`vm_stat` "Pages wired down"), *not* MLX's
  `get_peak_memory()`, which undercounts true footprint by ~40% (it excludes the buffer
  cache, which the OS still wires).
- ⚡ **The prefill transient spike** drives the crash — it grows several× faster than the
  steady-state KV cache and is invisible in MLX's reported peak.
- 🧬 **Only `full_attention` layers** grow KV with context; sliding/linear layers don't.

**Two memory classes** the suite classifies and handles differently:

| Class | Models | Cache | KV quantization |
|---|---|---|---|
| 🪟 **Sliding-window** | Gemma, GPT-OSS | `RotatingKVCache` | ❌ can't — `--kv-bits 4` crashes them past 5 k tokens, so they run **fp16** (window caps most layers, growth is gentle) |
| 📈 **Linear + full** | Qwen3.5 9B / 27B | standard | ✅ 4-bit quantizes fine and lowers high-context memory (brief spike at the 5 k threshold) |

---

## 🚀 Quick start

```bash
uv sync                          # install deps into .venv
uv run wmx-suite system          # show the machine's wall, swap, baseline
```

| Command | What it does |
|---|---|
| `uv run wmx-suite system` | Show the machine's wall, swap, baseline |
| `uv run wmx-suite scan` | Register `mlx-community` models from the HF cache |
| `uv run wmx-suite show <hf_id>` | Architecture + memory class for one model |
| `uv run wmx-suite characterize <hf_id>` | Safe probe → fitted context ceiling |
| `uv run wmx-suite list` | Ceilings for everything characterized |
| `uv run wmx-suite run --model <hf_id> …` | Safely launch `mlx_lm.generate` |

`characterize` refuses to launch any probe whose pre-flight base estimate already
exceeds the safe threshold — this is how oversized models (like the 27B) are handled:
**predicted, never run into the wall.**

### 🛡️ `run` — the safe launcher

`run` plans a launch and then execs `mlx_lm.generate`. It:

- picks `--kv-bits` by **cache type** — `4` for standard caches, **omitted** for
  RotatingKVCache models (Gemma, GPT-OSS) which can't quantize and would otherwise crash;
- samples the **live settled baseline** and caps `--max-kv-size` at the context where
  `live_base + model_base + slope·c` hits the safe threshold, using the model's
  **measured** curve from `suite.db` (or a conservative estimate, with a warning, if
  uncharacterized);
- **refuses** to launch if the model would breach the wall just to load (e.g. the 27B).

```bash
# launch safely
uv run wmx-suite run --model mlx-community/Qwen3.5-9B-OptiQ-4bit --prompt "..." --max-tokens 200

# inspect the plan only — no launch
uv run wmx-suite run --dry-run --model <hf_id> --prompt "..."
```

> `--force` overrides a refusal at your own risk; `--dry-run` prints the plan without launching.

---

## 🗂️ Layout

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

---

## 📊 Status

**v0 scaffold.** Validated methodology: predicted Gemma's ceiling to within **0.5%** from
safe probes. Calibration of the pre-flight base estimate refines as more models are run.
Flask UI (`web` extra) is optional and not yet built.

## 🤝 Contributing

Contributions are welcome — especially **memory-benchmark results from other Apple
Silicon SKUs**, which is how the suite becomes trustworthy beyond the reference M4 Pro.
See [CONTRIBUTING.md](./CONTRIBUTING.md). The prime directive applies to every change:
*never ship something that can crash a machine.*

## ☕ Support

If wmx-suite saved you a kernel panic (or you just like the idea), you can buy me a coffee:

<div align="center">

<a href="https://www.buymeacoffee.com/willsarg" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me a Coffee" width="217" height="60"></a>

</div>

## ⚖️ Trademarks

MLX, Apple Silicon, Metal, Mac, and macOS are trademarks of Apple Inc. This project is
an independent, community tool — it is **not affiliated with, endorsed by, or sponsored
by Apple Inc.** References to these names are descriptive only, to indicate the
technologies the suite works with. All other trademarks are the property of their
respective owners.
