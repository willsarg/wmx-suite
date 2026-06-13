<!-- Thanks for contributing! Keep PRs focused on a single change. -->

## Summary

<!-- What does this change and why? -->

Related issue: <!-- e.g. Fixes #N -->

## Type of change

- [ ] Bug fix
- [ ] New feature
- [ ] Docs only
- [ ] Refactor / cleanup (no behavior change)
- [ ] Benchmark / calibration data

## Safety (the prime directive: never crash a machine)

- [ ] This change does **not** touch a safety-critical path (`probe.py`, `launcher.py`,
      `probe_worker.py`, `system.py`) — *or* the boxes below are checked.
- [ ] No new code path launches a model at a context the pre-flight gate hasn't cleared.
- [ ] Memory budgeting uses measured **OS-wired** high-water, not MLX's self-reported peak.
- [ ] I've attached evidence (a `--dry-run` plan and/or a `characterize` run) — see Testing.

## Conventions

- [ ] `uv` only (no `--break-system-packages`); `hf` not `huggingface-cli`.
- [ ] Examples/measurements use `mlx-community` models.
- [ ] `--kv-bits 4` only for quantizable caches (never for RotatingKVCache models).

## Testing

<!-- Paste the commands you ran and their output. Memory claims need evidence. -->

- [ ] I added or updated tests for each bug fix, feature, or behavior change.
- [ ] Bug fixes include a regression test that covers the original failure.
- [ ] Safety-critical changes include hardware-free boundary and refusal-path tests.
- [ ] I ran the targeted tests for the changed behavior.
- [ ] I ran `uv run pytest -q`.
- [ ] I ran `uv run python -m compileall -q wmx_suite tests`.
- [ ] If automated tests are impractical, I explained why and provided deterministic
      manual verification steps below.

```
```
