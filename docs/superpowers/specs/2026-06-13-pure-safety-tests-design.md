# Pure Safety Tests Design

## Goal

Add a first pytest suite that protects wmx-suite's crash-prevention logic without
loading models, allocating MLX memory, starting subprocesses, reading live system
memory, or accessing the production SQLite database.

The suite should make the documented planning invariants executable and expose unsafe
argument combinations before `mlx_lm.generate` can be launched. It must not claim
end-to-end runtime safety from unit tests alone.

## Pre-Implementation Safety Finding

Self-review found that the launcher's central runtime-cap assumption is not valid for
all supported models under the installed `mlx-lm 0.31.3`:

- `mlx_lm.models.cache.make_prompt_cache(model, max_kv_size)` calls
  `model.make_cache()` without forwarding `max_kv_size` whenever the model defines that
  method.
- Qwen3.5 defines `make_cache()` and returns its own `ArraysCache`/`KVCache` list.
- Speculative generation explicitly removes `max_kv_size`.

Therefore, injecting `--max-kv-size` proves that the planner requested a cap, but does
not prove that MLX enforces it for Qwen3.5. This is a pre-existing safety gap and must be
handled as separate safety-critical work before describing `run` as enforcing a runtime
context ceiling for every supported model.

The first pure test suite may still protect planner arithmetic, refusal behavior, model
classification, and argument policy. Its names and comments must distinguish a
**planned cap** from a verified runtime cap.

## Scope

The initial suite covers pure behavior in:

- `wmx_suite.launcher`: prediction math and launch argument construction.
- `wmx_suite.models`: cache classification, growing-layer classification, and analytic
  fp16 KV growth.
- `wmx_suite.probe`: linear fitting and context-ceiling calculation.
- `wmx_suite.system`: safe-threshold arithmetic.

It does not cover model loading, probe orchestration, PTY execution, CLI output,
database persistence, `vm_stat`, `sysctl`, or MLX APIs. Those require separate
integration-test designs and stronger isolation.

## Test Infrastructure

Add pytest as a development dependency using uv. Tests live under `tests/`, grouped by
module:

- `tests/test_launcher.py`
- `tests/test_models.py`
- `tests/test_probe_math.py`
- `tests/test_system.py`

Tests use synthetic `ModelInfo` values, monkeypatched config/weight readers, and
temporary config files. Importing the target modules is safe: MLX is imported only
inside runtime functions that these tests do not call.

## Launcher Prediction Tests

Test `launcher.predict()` using hand-calculated inputs:

1. Absolute base equals `live_base + model_base`.
2. Safe context equals
   `(threshold - live_base - model_base) / slope_gb_per_k * 1000`.
3. A negative headroom produces zero safe context.
4. A calculated context above `model_max` is capped at `model_max`.
5. Base equal to the hard wall sets `breaches_wall`; the implementation intentionally
   uses `>=`, matching the rule that the wall must never be reached.
6. Base below the wall does not set `breaches_wall`, even if it has crossed the lower
   safe threshold. The safe context must still be zero.
7. A zero or negative slope with positive headroom uses `model_max`, because the fitted
   curve predicts no context-dependent growth.
8. A zero or negative slope with exhausted headroom returns zero safe context rather
   than granting the model maximum context.

Item 8 is a deliberate conservative correction to current behavior.

Test the uncharacterized-model estimate separately:

- `_estimated_slope_gb_per_k()` converts bytes/token to GB/1k tokens and applies
  `PREFILL_SPIKE_MULT`.
- Missing KV metadata produces a zero analytic slope. This test records current
  behavior but does not claim that a zero estimate is sufficient evidence to launch;
  `predict()` must still refuse exhausted base headroom.

## Launcher Plan Tests

Test `launcher.plan()` with all external collaborators monkeypatched: `models.describe`,
`read_limits`, `sample_settled_baseline`, `db.connect`, and `db.latest_fit`. No test may
open the production database or read live machine state.

Cover:

- A missing cached model returns an error before reading system limits.
- A measured fit is preferred over the analytic fallback.
- An uncharacterized model uses the documented resident-factor, fixed-overhead, and
  prefill-slope estimates.
- Cache quantization is selected only when `can_quantize_kv` is true.
- Absolute base equal to the hard wall is refused with a zero planned KV size.
- A planned context below `MIN_USEFUL_CTX` is refused.
- A planned context equal to `MIN_USEFUL_CTX` is accepted.
- The returned planned context never exceeds the model's declared maximum.

Test `probe.estimate_base_gb()` directly with synthetic `SystemLimits`, including its
documented 2.5 GB minimum baseline. This checks the pre-flight estimate independently
from launcher planning.

## Launch Argument Policy

Argument handling must recognize both `--flag value` and `--flag=value`.
Duplicate occurrences of a safety-controlled option are rejected rather than relying
on argparse's last-value-wins behavior.

For a quantizable standard cache:

- Inject `--kv-bits 4` when the user did not provide `--kv-bits`.
- Preserve an explicit KV setting.

For a non-quantizable `RotatingKVCache`:

- Omit KV quantization by default.
- Reject every explicit `--kv-bits` value, including under `--force`. This is a known
  unsupported configuration that raises `NotImplementedError`, not a memory-risk
  estimate that the user can knowingly override.

For context limits:

- Inject the planned `--max-kv-size` when absent.
- Preserve an explicit value at or below the planned cap.
- Reject an explicit value above the planned cap without `--force`.
- Preserve an explicit value above the planned cap with `--force`, consistent with the
  documented ability to override memory-based safety refusal at the user's risk.
- Reject missing, non-integer, negative, or duplicate explicit values before launch.

The following passthrough options alter memory behavior outside the measured plan:

- `--draft-model`: the installed `mlx_lm.generate` implementation explicitly removes
  `max_kv_size` for speculative generation and loads a second model.
- `--prompt-cache-file`: imports cache state whose size and quantization were not
  established by this launch plan.
- `--adapter-path`: adds uncharacterized model-resident parameters.

Reject these options by default. Permit them only under `--force`, because they are
unmeasured memory risks rather than cache implementations known to be unsupported.
Tests cover both option syntaxes where argparse accepts them.

This validation should remain pure and live in `launcher.py`. The CLI passes the parsed
`force` value into it and reports validation errors before execution. Validation occurs
after planning but before printing or executing the final command.

These tests verify only the command policy and requested arguments. They do not assert
that MLX honors the planned cap at runtime; the pre-implementation finding above
explicitly prevents that inference.

## Model Tests

For `models.describe()` tests, monkeypatch `_read_config()` and `weights_gb()` so it
uses synthetic data only. Test `_read_config()` separately with a temporary cache tree
and a monkeypatched `HUB`.

Test these documented classifications:

1. A config with a positive `sliding_window` is `RotatingKVCache` and cannot quantize
   KV.
2. A config with a `sliding_attention` layer is also `RotatingKVCache`.
3. A config without sliding behavior uses the standard cache and can quantize KV.
4. When `layer_types` exists, only `full_attention` layers count as growing.
5. Without `layer_types`, all declared hidden layers count as growing.
6. A nested `text_config` is selected by `_read_config()` using a temporary config file.
7. A missing config returns `None`.
8. `sliding_window` combined with `use_sliding_window: false` remains a standard cache.
   This is a deliberate correction to current key-presence classification, verified
   against the cached Qwen2.5-VL config and its installed MLX implementation.

Test `ModelInfo.fp16_kv_bytes_per_token()` from the documented formula:

`growing_layers * kv_heads * head_dim * 2 (K,V) * 2 (fp16 bytes)`

Missing KV-head or head-dimension metadata returns zero because no defensible analytic
estimate can be calculated.

## System Threshold Test

Test `SystemLimits.safe_threshold_gb()` directly:

- The default threshold is `wall_gb - 2.0`.
- A caller-provided margin is subtracted exactly.

These tests instantiate `SystemLimits` directly and do not call `device_limits()`,
`wired_gb()`, `swap_free_gb()`, or `read_limits()`.

## Probe Math Tests

Test `_linfit()` with exact datasets whose expected results are independently derived:

- A two-point line with known intercept and slope.
- A multi-point exact line with `R² = 1`.
- Constant observations, confirming the current zero-slope and `R² = 1` behavior.

Test `_solve_ctx()` from:

`target = ref_baseline + model_base + slope_per_k * context_in_thousands`

Cases cover positive headroom, zero headroom, negative headroom, zero slope, negative
slope, and integer truncation. These tests intentionally lock the conservative
zero-result behavior for non-positive slopes.

Do not add malformed-input tests for `_linfit()` in this first suite. Production calls
it only after collecting at least two paired measurements; validating arbitrary public
inputs is outside this private helper's contract.

## Assumption Discipline

Every non-obvious test includes a short comment naming its basis:

- A documented invariant in `AGENTS.md`.
- A boundary explicitly used by production code, such as `>= wall`.
- A hand-derived equation shown in the test.
- An approved policy decision from this design.

Tests must not infer desired behavior solely from the current implementation. Where
current behavior conflicts with this design, the test should describe the approved
safety behavior and the production code should be changed to satisfy it.

## Verification

The implementation is complete when:

- `uv run pytest` passes.
- `uv run python -m compileall -q wmx_suite tests` passes.
- No test invokes MLX runtime APIs, model loading, subprocess execution, live system
  memory commands, or the production database.
- `git diff` contains only the intended dependency metadata, tests, and narrowly scoped
  launcher/model-classifier safety changes.
- Tests demonstrate that normal argument construction cannot silently replace the
  planned `max_kv_size` through duplicate options or opt into known unmeasured launch
  modes without `--force`.
- Test names and documentation do not state or imply that `--max-kv-size` is verified
  to enforce the runtime ceiling for Qwen3.5.
