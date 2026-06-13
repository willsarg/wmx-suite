# Contributing to wmx-suite

Thanks for your interest! wmx-suite started as a personal bench for running local MLX
models on one M4 Pro without crashing it, but the goal is for it to work safely across
the **entire Apple Silicon lineup** — so contributions, especially from other M-series
machines, are genuinely wanted.

Please read [AGENTS.md](./AGENTS.md) first. It's the source of truth for the project's
purpose, the safety methodology, and the key facts. This file covers the *human
workflow*: setup, conventions, and how to land a change.

## The one rule that overrides everything

**Never ship a change that can crash a machine.** The crash wall is the GPU working-set
limit (~17.18 GB on the reference M4 Pro), *not* total RAM. With little swap free,
crossing it can hard-lock the system rather than fail gracefully. The whole suite is
built to **predict the wall and stay under it** — it never probes into the danger zone.

If your change touches `probe.py`, `launcher.py`, `probe_worker.py`, or `system.py`,
treat it as **safety-critical**:

- Don't add code paths that launch a model at a context the pre-flight gate hasn't
  cleared.
- Budget against measured **OS-wired high-water** (`vm_stat` "Pages wired down"), never
  MLX's self-reported peak — it undercounts true wired memory by ~40%.
- Include `--dry-run` output (the plan) in your PR so reviewers can see the predicted
  ceiling without anyone having to run into the wall.

## Especially wanted: benchmark results from other Macs

The biggest gap is that the pre-flight estimate constants are calibrated on one M4 Pro.
To make wmx-suite trustworthy on other SKUs (M1/M2/M3/M4 × base/Pro/Max/Ultra × RAM
tiers), we need memory-benchmark numbers from real hardware. See issues
[#9](https://github.com/willsarg/wmx-suite/issues/9) (per-machine calibration) and
[#10](https://github.com/willsarg/wmx-suite/issues/10) (community profile registry).

If you have an Apple Silicon Mac, the most useful thing you can do is:

1. Run `uv run wmx-suite system` and `uv run wmx-suite characterize <model>` on a couple
   of `mlx-community` models.
2. Open an issue with your **chip + RAM tier**, the reported memory wall, and the fitted
   `model_base` / `slope` / ceiling for each model.

These are just hardware capability numbers — memory limits and per-model curves. Nothing
personal: no filenames, prompts, or anything identifying. That's all an opt-in profile
registry would ever collect, so predictions are earned empirically instead of hardcoded.

## Development setup

Requires **Python 3.12** and [`uv`](https://github.com/astral-sh/uv).

```bash
git clone https://github.com/willsarg/wmx-suite.git
cd wmx-suite
uv sync                      # creates .venv, installs in editable mode
uv run wmx-suite system      # smoke test — prints your machine's wall + baseline
```

## Project conventions

These are enforced — please follow them:

- **Packages: `uv` only.** Never use `pip install --break-system-packages`.
- **HuggingFace CLI: `hf`**, not the deprecated `huggingface-cli`.
- Stick to **`mlx-community`** models for measurements and examples.
- SQLite is the datastore (`data/suite.db`, gitignored). Flask is an optional extra
  (`web`) — only add UI code if the change calls for it.
- When measuring, **match production inference settings**: `--kv-bits 4` with
  `kv_group_size=64`, `quantized_kv_start=5000`, but **only for quantizable caches**.
  RotatingKVCache models (Gemma, GPT-OSS) must run fp16 — quantizing them crashes past
  5000 tokens.

## Pull requests

- Branch off `main`; keep PRs focused on a single change.
- Reference the issue you're addressing (`Fixes #N`).
- For anything non-trivial, **open an issue first** so we can agree on direction before
  you invest the work.
- Run the actual commands and paste the output — especially memory numbers. Claims like
  "this is safe" need evidence (a `--dry-run` plan or a real `characterize` run).
- Match the surrounding code style: small, focused modules; comments that explain *why*,
  not *what*.

## Reporting bugs

Open an issue with your chip + RAM, the command you ran, the full output, and what you
expected. If it's a crash or near-crash, include the `system` output and any plan
(`--dry-run`) from before the launch.

## Trademarks & license

By contributing, you agree your contributions are licensed under the project's
[MIT License](./LICENSE). MLX, Apple Silicon, Metal, Mac, and macOS are trademarks of
Apple Inc.; this is an independent project, not affiliated with Apple.
