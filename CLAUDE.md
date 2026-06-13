# CLAUDE.md

This project's agent guidance lives in **[AGENTS.md](./AGENTS.md)** — read it first.
It is the single source of truth for purpose, safety rules, facts, commands, and
conventions. This file only adds Claude Code specifics.

## The one rule that matters most

**NEVER CRASH THE LAPTOP.** The crash wall is ~17.18 GB of wired memory (not total RAM),
swap is ~1 GB, and crossing it can hard-lock the machine. Never launch a model run whose
predicted OS-wired peak exceeds the safe threshold (~15.18 GB). Use the `characterize`
pre-flight gate; do not call `mlx_lm` directly at high context to test. See AGENTS.md
for the full reasoning.

## Running things

- Use `uv run wmx-suite <cmd>` (see AGENTS.md for the command list).
- **Launch models only via `uv run wmx-suite run --model <id> ...`** — it is the only
  crash-safe launch path. Never call `mlx_lm.generate` directly at non-trivial context.
  `--dry-run` shows the plan.
- Long probes / model loads: run in the background and report back, rather than blocking.
- Package installs: **`uv` only, never `--break-system-packages`.**

## Before claiming something works

Add or update tests for every bug fix, feature, or behavior change. A bug fix needs a
regression test that demonstrates the original failure; safety-critical changes need
hardware-free boundary/refusal tests. Do not remove or weaken tests just to get green.

Run the targeted tests while developing, then run:

```bash
uv run pytest -q
uv run python -m compileall -q wmx_suite tests
```

Show the actual output before claiming success. Memory numbers especially: trust
measured `os_wired` from `vm_stat`, not MLX's self-reported peak (it undercounts ~40%).
