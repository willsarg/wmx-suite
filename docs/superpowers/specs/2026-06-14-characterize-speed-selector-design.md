# Characterize `--speed` selector — design

**Date:** 2026-06-14
**Status:** approved (brainstorming), pending spec review

## Problem

`characterize` is slow. The cost is not algorithmic — it is **O(n)** in the number of
context rungs, with no quadratic blowup. The wall-clock is dominated by a single fixed
cost repeated many times: each measurement spawns a fresh subprocess that **loads the
full model from disk**.

With today's defaults (`DEFAULT_RAMP` = 8 rungs, `DEFAULT_REPEATS` = 3) a full run pays
up to `8 × 3 = 24` cold model loads. On the M4 Pro testbed (~15s/load for a mid-size
model) that is ~6 minutes, with the model load dwarfing generation, baseline settling,
regression refits, and DB writes.

The fresh-subprocess-per-measurement design is deliberate and load-bearing for safety
(clean wired-memory baseline per measurement + crash isolation), so we do **not** remove
it. Instead we let the user dial down *how much work* is done.

## Goal

Add a speed selector that trades fit granularity for fewer model loads.

**Design note (revised after real-world testing).** The original plan was to make `quick`
run *fewer rungs*. Testing on `Qwen3-0.6B-4bit` showed this biases the ceiling
**optimistically**: real memory grows super-linearly (KV storage is linear, but attention
prefill scratch + OS wired overhead bend the curve upward), so a straight-line fit through
only low contexts extrapolates *too high* (48.1k vs standard's 46.2k). Optimism is the
unsafe direction for a crash-avoidance tool.

The speed lever is therefore **repeats, not rungs**. Cutting repeats (3→1) adds *unbiased
noise* instead of optimistic bias, and because the pre-flight gate already prunes high
rungs, keeping a mid-dense ramp costs almost nothing in wall-clock. Verified: repeats-based
`quick` lands at 46.5k (within 0.5% of standard) in ~70s vs standard's ~210s — ~3× faster
and conservative.

Non-goals: single-load multi-context worker (a separate, larger structural change);
changing the pre-flight safety gate; removing the subprocess isolation.

## Design

### Surface

Add `--speed {quick,standard,full}` to the `characterize` CLI command.
Default = `standard` = today's exact behavior (fully backward-compatible).

### Mechanism

A pure-data preset table in `probe.py` maps each preset to a `(ramp, repeats)` pair:

| Preset     | Ramp (rungs)                              | Repeats | vs standard |
|------------|-------------------------------------------|---------|-------------|
| `quick`    | `[2048, 8192, 16384, 32768, 65536, 131072]` (6, mid-dense) | 1 | ~3× faster, conservative |
| `standard` | current `DEFAULT_RAMP` (8)                 | 3       | baseline    |
| `full`     | `[2048, 4096, 8192, 16384, 24576, 32768, 49152, 65536, 98304, 131072]` (10) | 3 | finer, slower |

`quick`'s ramp stays mid-dense so the fit spans the memory bend; the speedup comes from
`repeats=1` plus the gate pruning high rungs.

### Override rule

The preset sets *defaults*. An explicit `--repeats N` on the CLI still wins, so
`--speed quick --repeats 1` is honored. The ramp comes from the preset (no CLI override
for ramp in this change).

### Plumbing

`characterize()` already accepts `ramp=` and `repeats=` parameters (`probe.py:144`).
The CLI resolves the preset name → `(ramp, repeats)` and passes them through. `probe`
stays decision-only; the selector is just argument resolution.

## Safety analysis (crash wall)

The preset changes only **which** contexts are probed and **how many** times — it does
**not** touch the pre-flight gate. For every rung the loop still:

1. measures actual `os_wired` for real, and
2. predicts the next rung against the linear fit and **breaks before launching** anything
   predicted to breach the safe threshold (`probe.py:263-270`).

KV-cache memory grows near-linearly in context, so a coarser ramp yields a coarser fit
but **not a less conservative one** — the gate still refuses to launch an unsafe rung.

Honest tradeoff: with `quick`, each rung is measured once (`repeats=1`), so individual
points are noisier than standard's 3-repeat median. This noise is *unbiased* (it does not
systematically over-report) and shows up in a lower R², unlike the optimistic *bias* that
cutting rungs would introduce. The mid-dense ramp ensures the fit still spans the curve.

## Testing (hardware-free)

- Preset name → expected `(ramp, repeats)` for `quick` / `standard` / `full`.
- Explicit `--repeats` overrides the preset's repeats; ramp still comes from preset.
- Regression guard: `standard` resolves to byte-identical `(DEFAULT_RAMP, DEFAULT_REPEATS)`
  so existing behavior is unchanged.
- CLI arg parsing: invalid `--speed` value is rejected; default is `standard`.

These are all pure argument-resolution tests — no model load, no hardware.
