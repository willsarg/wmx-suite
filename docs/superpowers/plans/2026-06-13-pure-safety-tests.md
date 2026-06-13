# Pure Safety Tests Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add hardware-free pytest coverage for planning safety, fix confirmed conservative-boundary and cache-classification defects, and reject launch arguments that bypass the measured plan.

**Architecture:** Keep tests at pure boundaries. `launcher.py` owns prediction and passthrough policy, `models.py` owns config classification, and existing math helpers remain independently testable. All collaborators that touch MLX, live memory, subprocesses, or SQLite are replaced with synthetic values or monkeypatches.

**Tech Stack:** Python 3.12, pytest, uv, dataclasses, pytest `monkeypatch` and `tmp_path`.

---

## File Map

- Modify `pyproject.toml` and `uv.lock`: add pytest to the uv development dependency group.
- Modify `wmx_suite/launcher.py`: conservative non-positive-slope handling and pure launch-argument validation.
- Modify `wmx_suite/cli.py`: pass `force` into argument construction and report validation failures.
- Modify `wmx_suite/models.py`: respect an explicit disabled sliding-window configuration.
- Create `tests/test_launcher.py`: prediction, planning, estimation, and argument-policy tests.
- Create `tests/test_models.py`: config classification and KV arithmetic tests.
- Create `tests/test_probe_math.py`: fit, ceiling, and pre-flight estimate tests.
- Create `tests/test_system.py`: threshold arithmetic tests.

The separate Qwen3.5 runtime-cap gap is not fixed in this plan. Tests must call the
result a `planned` cap because `mlx-lm 0.31.3` does not forward `max_kv_size` to
Qwen3.5's custom `make_cache()`. Track the runtime fix in GitHub issue
[#11](https://github.com/willsarg/wmx-suite/issues/11).

### Task 1: Add pytest infrastructure

**Files:**
- Modify: `pyproject.toml`
- Modify: `uv.lock`

- [ ] **Step 1: Add pytest through uv**

Run:

```bash
uv add --dev pytest
```

Expected: `pyproject.toml` gains a development dependency group containing pytest and
`uv.lock` is updated.

- [ ] **Step 2: Verify pytest starts with no tests**

Run:

```bash
uv run pytest --collect-only
```

Expected: pytest starts successfully and reports no tests collected.

- [ ] **Step 3: Commit dependency setup**

```bash
git add pyproject.toml uv.lock
git commit -m "test: add pytest development dependency"
```

### Task 2: Test system and probe math

**Files:**
- Create: `tests/test_system.py`
- Create: `tests/test_probe_math.py`

- [ ] **Step 1: Add threshold and exact-fit tests**

Create `tests/test_system.py`:

```python
import pytest

from wmx_suite.system import SystemLimits


def limits(wall_gb: float = 17.18) -> SystemLimits:
    return SystemLimits(
        device="test",
        total_gb=24.0,
        wall_gb=wall_gb,
        max_buffer_gb=8.0,
        swap_free_gb=1.0,
        wired_now_gb=3.0,
    )


def test_safe_threshold_uses_default_two_gb_margin():
    # AGENTS.md defines the default threshold as wall minus 2 GB.
    assert limits().safe_threshold_gb() == pytest.approx(15.18)


def test_safe_threshold_uses_requested_margin():
    assert limits(20.0).safe_threshold_gb(3.5) == pytest.approx(16.5)
```

Create `tests/test_probe_math.py`:

```python
import pytest

from wmx_suite import models, probe
from wmx_suite.system import SystemLimits


def model_info(weights_gb: float = 8.0) -> models.ModelInfo:
    return models.ModelInfo(
        hf_id="mlx-community/test",
        weights_gb=weights_gb,
        n_layers=4,
        growing_layers=2,
        kv_heads=8,
        head_dim=128,
        hidden_size=1024,
        max_context=32768,
        cache_type="standard",
        can_quantize_kv=True,
        layer_types={"full_attention": 2, "linear_attention": 2},
    )


def limits(wired_now_gb: float) -> SystemLimits:
    return SystemLimits(
        device="test",
        total_gb=24.0,
        wall_gb=17.18,
        max_buffer_gb=8.0,
        swap_free_gb=1.0,
        wired_now_gb=wired_now_gb,
    )


def test_linfit_recovers_two_point_line():
    # y = 2 + 0.5x, where x is thousands of tokens.
    assert probe._linfit([2.0, 6.0], [3.0, 5.0]) == pytest.approx((2.0, 0.5, 1.0))


def test_linfit_recovers_exact_multi_point_line():
    assert probe._linfit([0.0, 2.0, 4.0], [1.0, 2.0, 3.0]) == pytest.approx(
        (1.0, 0.5, 1.0)
    )


def test_linfit_constant_observations_have_zero_slope():
    assert probe._linfit([1.0, 2.0, 3.0], [4.0, 4.0, 4.0]) == pytest.approx(
        (4.0, 0.0, 1.0)
    )


@pytest.mark.parametrize(
    ("model_base", "slope", "baseline", "target", "expected"),
    [
        (8.0, 0.1, 3.0, 15.0, 40000),
        (12.0, 0.1, 3.0, 15.0, 0),
        (13.0, 0.1, 3.0, 15.0, 0),
        (8.0, 0.0, 3.0, 15.0, 0),
        (8.0, -0.1, 3.0, 15.0, 0),
        (8.0, 0.3, 3.0, 12.0, 3333),
    ],
)
def test_solve_ctx_uses_documented_equation(
    model_base, slope, baseline, target, expected
):
    # target = baseline + model_base + slope * context_in_thousands.
    assert probe._solve_ctx(model_base, slope, baseline, target) == expected


def test_estimate_base_uses_live_baseline_above_floor():
    expected = 4.0 + 8.0 * probe.RESIDENT_FACTOR + probe.FIXED_OVERHEAD_GB
    assert probe.estimate_base_gb(model_info(), limits(4.0)) == pytest.approx(expected)


def test_estimate_base_uses_two_point_five_gb_baseline_floor():
    expected = 2.5 + 8.0 * probe.RESIDENT_FACTOR + probe.FIXED_OVERHEAD_GB
    assert probe.estimate_base_gb(model_info(), limits(1.0)) == pytest.approx(expected)
```

- [ ] **Step 2: Run the new tests**

Run:

```bash
uv run pytest tests/test_system.py tests/test_probe_math.py -v
```

Expected: all tests pass without importing MLX runtime APIs.

- [ ] **Step 3: Commit math tests**

```bash
git add tests/test_system.py tests/test_probe_math.py
git commit -m "test: cover threshold and probe math"
```

### Task 3: Correct and test model classification

**Files:**
- Modify: `wmx_suite/models.py:79-100`
- Create: `tests/test_models.py`

- [ ] **Step 1: Write model tests, including the confirmed disabled-sliding regression**

Create `tests/test_models.py` with:

```python
import json

import pytest

from wmx_suite import models


def describe(monkeypatch, config):
    monkeypatch.setattr(models, "_read_config", lambda _hf_id: config)
    monkeypatch.setattr(models, "weights_gb", lambda _hf_id: 4.25)
    return models.describe("mlx-community/test")


def test_positive_sliding_window_is_rotating(monkeypatch):
    info = describe(
        monkeypatch,
        {"num_hidden_layers": 4, "sliding_window": 512},
    )
    assert info.cache_type == "RotatingKVCache"
    assert info.can_quantize_kv is False


def test_sliding_attention_layer_is_rotating(monkeypatch):
    info = describe(
        monkeypatch,
        {
            "num_hidden_layers": 3,
            "layer_types": [
                "full_attention",
                "sliding_attention",
                "linear_attention",
            ],
        },
    )
    assert info.cache_type == "RotatingKVCache"
    assert info.growing_layers == 1


def test_standard_config_is_quantizable(monkeypatch):
    info = describe(monkeypatch, {"num_hidden_layers": 4})
    assert info.cache_type == "standard"
    assert info.can_quantize_kv is True
    assert info.growing_layers == 4


def test_explicitly_disabled_sliding_window_is_standard(monkeypatch):
    # Qwen2.5-VL has sliding_window metadata while use_sliding_window is false.
    info = describe(
        monkeypatch,
        {
            "num_hidden_layers": 4,
            "sliding_window": 32768,
            "use_sliding_window": False,
        },
    )
    assert info.cache_type == "standard"
    assert info.can_quantize_kv is True


def test_read_config_selects_nested_text_config(monkeypatch, tmp_path):
    hub = tmp_path / "hub"
    snapshot = hub / "models--mlx-community--test" / "snapshots" / "abc"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text(
        json.dumps({"text_config": {"num_hidden_layers": 7}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(models, "HUB", str(hub))
    assert models._read_config("mlx-community/test") == {"num_hidden_layers": 7}


def test_read_config_returns_none_when_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(models, "HUB", str(tmp_path))
    assert models._read_config("mlx-community/missing") is None


def test_fp16_kv_bytes_per_token_uses_only_growing_layers():
    info = models.ModelInfo(
        hf_id="mlx-community/test",
        weights_gb=4.0,
        n_layers=4,
        growing_layers=2,
        kv_heads=8,
        head_dim=128,
        hidden_size=1024,
        max_context=32768,
        cache_type="standard",
        can_quantize_kv=True,
        layer_types={"full_attention": 2, "linear_attention": 2},
    )
    assert info.fp16_kv_bytes_per_token() == 2 * 8 * 128 * 2 * 2


@pytest.mark.parametrize(("kv_heads", "head_dim"), [(None, 128), (8, None)])
def test_fp16_kv_bytes_per_token_requires_metadata(kv_heads, head_dim):
    info = models.ModelInfo(
        hf_id="mlx-community/test",
        weights_gb=4.0,
        n_layers=4,
        growing_layers=2,
        kv_heads=kv_heads,
        head_dim=head_dim,
        hidden_size=1024,
        max_context=32768,
        cache_type="standard",
        can_quantize_kv=True,
        layer_types={},
    )
    assert info.fp16_kv_bytes_per_token() == 0.0
```

- [ ] **Step 2: Run the regression test and confirm it fails**

Run:

```bash
uv run pytest tests/test_models.py::test_explicitly_disabled_sliding_window_is_standard -v
```

Expected: FAIL because current code treats key presence as sliding behavior.

- [ ] **Step 3: Implement the verified classifier condition**

In `models.describe()`, replace the `has_sliding` assignment with:

```python
    sliding_enabled = t.get("use_sliding_window", True)
    has_sliding = (
        (sliding_enabled is not False and bool(t.get("sliding_window")))
        or lt.get("sliding_attention", 0) > 0
    )
```

This keeps Gemma/GPT-OSS rotating behavior while honoring explicit disabled metadata.

- [ ] **Step 4: Run all model tests**

Run:

```bash
uv run pytest tests/test_models.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit classification coverage and fix**

```bash
git add wmx_suite/models.py tests/test_models.py
git commit -m "fix: respect disabled sliding-window metadata"
```

### Task 4: Correct and test prediction and planning

**Files:**
- Modify: `wmx_suite/launcher.py:39-56`
- Create: `tests/test_launcher.py`

- [ ] **Step 1: Add prediction and estimation tests**

Create `tests/test_launcher.py` with shared constructors and these tests:

```python
import pytest

from wmx_suite import launcher, models
from wmx_suite.system import SystemLimits


def model_info(*, quantizable=True, max_context=131072, weights_gb=8.0):
    return models.ModelInfo(
        hf_id="mlx-community/test",
        weights_gb=weights_gb,
        n_layers=4,
        growing_layers=2,
        kv_heads=8,
        head_dim=128,
        hidden_size=1024,
        max_context=max_context,
        cache_type="standard" if quantizable else "RotatingKVCache",
        can_quantize_kv=quantizable,
        layer_types={"full_attention": 2, "linear_attention": 2},
    )


def test_predict_calculates_base_headroom_and_context():
    result = launcher.predict(
        model_base_gb=8.0,
        slope_gb_per_k=0.1,
        live_base_gb=3.0,
        threshold_gb=15.0,
        wall_gb=17.0,
        model_max=100000,
    )
    assert result.base_abs_gb == pytest.approx(11.0)
    assert result.headroom_gb == pytest.approx(4.0)
    assert result.safe_ctx == 40000
    assert result.breaches_wall is False


def test_predict_caps_context_at_model_max():
    result = launcher.predict(
        model_base_gb=8.0,
        slope_gb_per_k=0.01,
        live_base_gb=3.0,
        threshold_gb=15.0,
        wall_gb=17.0,
        model_max=32768,
    )
    assert result.safe_ctx == 32768


def test_predict_marks_wall_equality_as_breach():
    # RULE #1 uses >= because reaching the wall is not safe.
    result = launcher.predict(
        model_base_gb=14.0,
        slope_gb_per_k=0.1,
        live_base_gb=3.0,
        threshold_gb=15.0,
        wall_gb=17.0,
        model_max=32768,
    )
    assert result.breaches_wall is True
    assert result.safe_ctx == 0


@pytest.mark.parametrize("slope", [0.0, -0.1])
def test_predict_nonpositive_slope_with_exhausted_headroom_is_zero(slope):
    result = launcher.predict(
        model_base_gb=12.5,
        slope_gb_per_k=slope,
        live_base_gb=3.0,
        threshold_gb=15.0,
        wall_gb=17.0,
        model_max=32768,
    )
    assert result.breaches_wall is False
    assert result.safe_ctx == 0


@pytest.mark.parametrize("slope", [0.0, -0.1])
def test_predict_nonpositive_slope_with_headroom_uses_model_max(slope):
    result = launcher.predict(
        model_base_gb=8.0,
        slope_gb_per_k=slope,
        live_base_gb=3.0,
        threshold_gb=15.0,
        wall_gb=17.0,
        model_max=32768,
    )
    assert result.safe_ctx == 32768


def test_estimated_slope_converts_units_and_applies_prefill_multiplier():
    info = model_info()
    expected = (
        info.fp16_kv_bytes_per_token()
        * 1000
        / 1e9
        * launcher.PREFILL_SPIKE_MULT
    )
    assert launcher._estimated_slope_gb_per_k(info) == pytest.approx(expected)


def test_estimated_slope_is_zero_without_kv_metadata():
    info = model_info()
    info.kv_heads = None
    assert launcher._estimated_slope_gb_per_k(info) == 0.0
```

- [ ] **Step 2: Run the conservative boundary test and confirm it fails**

Run:

```bash
uv run pytest tests/test_launcher.py::test_predict_nonpositive_slope_with_exhausted_headroom_is_zero -v
```

Expected: FAIL because current code grants `model_max`.

- [ ] **Step 3: Implement conservative non-positive-slope handling**

Change the cap calculation in `launcher.predict()` to:

```python
    if headroom <= 0:
        cap = 0
    elif slope_gb_per_k > 0:
        cap = int(headroom / slope_gb_per_k * 1000)
    else:
        cap = model_max or 0
```

- [ ] **Step 4: Add fully mocked `plan()` tests**

Append tests that monkeypatch every external collaborator:

```python
def install_plan_fakes(monkeypatch, *, info, fit, live_base=3.0):
    limits = SystemLimits(
        device="test",
        total_gb=24.0,
        wall_gb=17.0,
        max_buffer_gb=8.0,
        swap_free_gb=1.0,
        wired_now_gb=live_base,
    )
    monkeypatch.setattr(launcher.models, "describe", lambda _hf_id: info)
    monkeypatch.setattr(launcher, "read_limits", lambda: limits)
    monkeypatch.setattr(launcher, "sample_settled_baseline", lambda: live_base)
    monkeypatch.setattr(launcher.db, "connect", lambda: object())
    monkeypatch.setattr(launcher.db, "latest_fit", lambda _con, _hf_id: fit)


def test_plan_returns_missing_model_before_reading_limits(monkeypatch):
    monkeypatch.setattr(launcher.models, "describe", lambda _hf_id: None)
    monkeypatch.setattr(
        launcher,
        "read_limits",
        lambda: pytest.fail("limits must not be read for a missing model"),
    )
    assert launcher.plan("mlx-community/missing") == {
        "error": "model not found in HF cache: mlx-community/missing"
    }


def test_plan_prefers_measured_fit(monkeypatch):
    install_plan_fakes(
        monkeypatch,
        info=model_info(),
        fit={"model_base_gb": 8.0, "slope_gb_per_k": 0.1},
    )
    result = launcher.plan("mlx-community/test")
    assert result["source"] == "measured"
    assert result["max_kv_size"] == 40000
    assert result["kv_bits"] == 4


def test_plan_refuses_wall_equality(monkeypatch):
    install_plan_fakes(
        monkeypatch,
        info=model_info(),
        fit={"model_base_gb": 14.0, "slope_gb_per_k": 0.1},
    )
    result = launcher.plan("mlx-community/test")
    assert result["refuse"] is True
    assert result["max_kv_size"] == 0


@pytest.mark.parametrize(
    ("safe_ctx", "refuse"),
    [(launcher.MIN_USEFUL_CTX - 1, True), (launcher.MIN_USEFUL_CTX, False)],
)
def test_plan_minimum_useful_context_boundary(monkeypatch, safe_ctx, refuse):
    slope = 1.0
    threshold = 15.0
    live_base = 3.0
    model_base = threshold - live_base - safe_ctx / 1000
    install_plan_fakes(
        monkeypatch,
        info=model_info(),
        fit={"model_base_gb": model_base, "slope_gb_per_k": slope},
        live_base=live_base,
    )
    assert launcher.plan("mlx-community/test")["refuse"] is refuse


def test_plan_uses_estimate_for_uncharacterized_model(monkeypatch):
    info = model_info(weights_gb=4.0)
    install_plan_fakes(monkeypatch, info=info, fit=None)
    result = launcher.plan("mlx-community/test")
    assert result["source"] == "estimated"
    assert result["model_base_gb"] == round(
        info.weights_gb * launcher.RESIDENT_FACTOR + launcher.FIXED_OVERHEAD_GB,
        2,
    )


def test_plan_omits_kv_quantization_for_rotating_cache(monkeypatch):
    install_plan_fakes(
        monkeypatch,
        info=model_info(quantizable=False),
        fit={"model_base_gb": 8.0, "slope_gb_per_k": 0.1},
    )
    assert launcher.plan("mlx-community/test")["kv_bits"] is None
```

- [ ] **Step 5: Run launcher prediction and plan tests**

Run:

```bash
uv run pytest tests/test_launcher.py -v
```

Expected: all tests added so far pass.

- [ ] **Step 6: Commit prediction and planning coverage**

```bash
git add wmx_suite/launcher.py tests/test_launcher.py
git commit -m "fix: handle exhausted prediction headroom conservatively"
```

### Task 5: Validate safety-controlled passthrough arguments

**Files:**
- Modify: `wmx_suite/launcher.py:111-118`
- Modify: `wmx_suite/cli.py:257-295`
- Modify: `tests/test_launcher.py`

- [ ] **Step 1: Add failing argument-policy tests**

Append to `tests/test_launcher.py`:

```python
def plan_dict(*, kv_bits=4, max_kv_size=4096):
    return {"kv_bits": kv_bits, "max_kv_size": max_kv_size}


@pytest.mark.parametrize(
    "args",
    [
        ["--model", "x"],
        ["--model=x"],
    ],
)
def test_build_argv_injects_planned_values(args):
    result = launcher.build_argv(args, plan_dict(), force=False)
    assert result[:4] == ["--max-kv-size", "4096", "--kv-bits", "4"]


@pytest.mark.parametrize(
    "args",
    [
        ["--model", "x", "--max-kv-size", "2048"],
        ["--model=x", "--max-kv-size=2048"],
    ],
)
def test_build_argv_preserves_context_at_or_below_plan(args):
    result = launcher.build_argv(args, plan_dict(), force=False)
    assert "--max-kv-size=2048" in result or result[-2:] == ["--max-kv-size", "2048"]


@pytest.mark.parametrize(
    "args",
    [
        ["--model", "x", "--kv-bits", "8"],
        ["--model=x", "--kv-bits=8"],
    ],
)
def test_build_argv_preserves_explicit_kv_bits_for_quantizable_cache(args):
    result = launcher.build_argv(args, plan_dict(), force=False)
    assert "--kv-bits=8" in result or result[-2:] == ["--kv-bits", "8"]


@pytest.mark.parametrize(
    "args",
    [
        ["--model", "x", "--max-kv-size", "8192"],
        ["--model=x", "--max-kv-size=8192"],
    ],
)
def test_build_argv_rejects_context_above_plan_without_force(args):
    with pytest.raises(launcher.LaunchArgumentError, match="planned cap"):
        launcher.build_argv(args, plan_dict(), force=False)


def test_build_argv_preserves_context_above_plan_with_force():
    args = ["--model", "x", "--max-kv-size", "8192"]
    assert launcher.build_argv(args, plan_dict(), force=True) == [
        "--kv-bits",
        "4",
        *args,
    ]


@pytest.mark.parametrize(
    "args",
    [
        ["--model", "x", "--max-kv-size"],
        ["--model", "x", "--max-kv-size=bad"],
        ["--model", "x", "--max-kv-size=-1"],
        ["--model", "x", "--max-kv-size", "1024", "--max-kv-size=2048"],
    ],
)
def test_build_argv_rejects_invalid_or_duplicate_context(args):
    with pytest.raises(launcher.LaunchArgumentError):
        launcher.build_argv(args, plan_dict(), force=False)


@pytest.mark.parametrize(
    "args",
    [
        ["--model", "x", "--kv-bits"],
        ["--model", "x", "--kv-bits=bad"],
        ["--model", "x", "--kv-bits=-1"],
        ["--model", "x", "--kv-bits", "4", "--kv-bits=8"],
    ],
)
def test_build_argv_rejects_invalid_or_duplicate_kv_bits(args):
    with pytest.raises(launcher.LaunchArgumentError):
        launcher.build_argv(args, plan_dict(), force=False)


@pytest.mark.parametrize(
    "args",
    [
        ["--model", "x", "--kv-bits", "4"],
        ["--model=x", "--kv-bits=4"],
    ],
)
def test_build_argv_rejects_kv_quantization_for_rotating_cache_even_with_force(args):
    with pytest.raises(launcher.LaunchArgumentError, match="not quantizable"):
        launcher.build_argv(args, plan_dict(kv_bits=None), force=True)


@pytest.mark.parametrize(
    "args",
    [
        ["--model", "x", "--draft-model", "draft"],
        ["--model=x", "--draft-model=draft"],
        ["--model", "x", "--prompt-cache-file", "cache.safetensors"],
        ["--model=x", "--prompt-cache-file=cache.safetensors"],
        ["--model", "x", "--adapter-path", "adapter"],
        ["--model=x", "--adapter-path=adapter"],
    ],
)
def test_build_argv_rejects_unmeasured_modes_without_force(args):
    with pytest.raises(launcher.LaunchArgumentError, match="--force"):
        launcher.build_argv(args, plan_dict(), force=False)


def test_build_argv_allows_unmeasured_modes_with_force():
    args = ["--model", "x", "--adapter-path", "adapter"]
    result = launcher.build_argv(args, plan_dict(), force=True)
    assert result[-4:] == args
```

- [ ] **Step 2: Run the tests and confirm missing validation**

Run:

```bash
uv run pytest tests/test_launcher.py -k build_argv -v
```

Expected: FAIL because `LaunchArgumentError` and the `force` parameter do not exist.

- [ ] **Step 3: Implement a pure option reader and validator**

Add to `wmx_suite/launcher.py`:

```python
class LaunchArgumentError(ValueError):
    pass


def _option_values(argv: list[str], option: str) -> list[str | None]:
    values: list[str | None] = []
    prefix = option + "="
    for index, arg in enumerate(argv):
        if arg == option:
            values.append(argv[index + 1] if index + 1 < len(argv) else None)
        elif arg.startswith(prefix):
            values.append(arg[len(prefix):])
    return values


def _single_int_option(argv: list[str], option: str) -> int | None:
    values = _option_values(argv, option)
    if len(values) > 1:
        raise LaunchArgumentError(f"{option} may be provided only once")
    if not values:
        return None
    value = values[0]
    try:
        parsed = int(value) if value is not None else None
    except ValueError as exc:
        raise LaunchArgumentError(f"{option} requires an integer") from exc
    if parsed is None or parsed < 0:
        raise LaunchArgumentError(f"{option} requires a non-negative integer")
    return parsed
```

Replace `build_argv()` with:

```python
def build_argv(rest: list[str], p: dict, *, force: bool = False) -> list[str]:
    argv = list(rest)
    user_kv_bits = _single_int_option(argv, "--kv-bits")
    user_max_kv = _single_int_option(argv, "--max-kv-size")

    if p["kv_bits"] is None and user_kv_bits is not None:
        raise LaunchArgumentError(
            "--kv-bits is not supported for this model's non-quantizable cache"
        )
    if user_max_kv is not None and user_max_kv > p["max_kv_size"] and not force:
        raise LaunchArgumentError(
            f"--max-kv-size {user_max_kv:,} exceeds planned cap "
            f"{p['max_kv_size']:,}; pass --force to override"
        )

    for option in ("--draft-model", "--prompt-cache-file", "--adapter-path"):
        if _option_values(argv, option) and not force:
            raise LaunchArgumentError(
                f"{option} changes unmeasured memory behavior; pass --force to override"
            )

    if p["kv_bits"] is not None and user_kv_bits is None:
        argv = ["--kv-bits", str(p["kv_bits"])] + argv
    if user_max_kv is None:
        argv = ["--max-kv-size", str(p["max_kv_size"])] + argv
    return argv
```

- [ ] **Step 4: Pass force through the CLI and report validation errors**

In `wmx_suite/cli.py`, replace the direct call with:

```python
    try:
        argv = launcher.build_argv(rest, p, force=force)
    except launcher.LaunchArgumentError as exc:
        raise SystemExit(f"[run] REFUSED: {exc}") from exc
```

This remains before `dry_run` and before either execution path.

- [ ] **Step 5: Run argument-policy tests**

Run:

```bash
uv run pytest tests/test_launcher.py -k build_argv -v
```

Expected: all argument-policy tests pass.

- [ ] **Step 6: Run the full suite**

Run:

```bash
uv run pytest -v
```

Expected: all tests pass.

- [ ] **Step 7: Commit passthrough validation**

```bash
git add wmx_suite/launcher.py wmx_suite/cli.py tests/test_launcher.py
git commit -m "fix: validate safety-controlled launch arguments"
```

### Task 6: Verify isolation and document the runtime-cap gap

**Files:**
- Modify: `README.md`
- Modify: `AGENTS.md`

- [ ] **Step 1: Correct runtime-cap claims**

Update the `run` documentation in both files to distinguish a planned cap from an
enforced MLX cap. State that under `mlx-lm 0.31.3`, models with custom `make_cache()`
such as Qwen3.5 do not receive `--max-kv-size`; until that path is fixed, the launcher
must not claim runtime enforcement for those models.

Use this wording in the relevant safety description:

```markdown
The launcher computes and passes a planned `--max-kv-size`. With `mlx-lm 0.31.3`,
models that implement their own `make_cache()` (including Qwen3.5) do not receive that
value, so the cap is not currently verified as enforced for those models.
```

- [ ] **Step 2: Verify no test touches prohibited runtime APIs**

Run:

```bash
rg -n "mlx|subprocess|vm_stat|sysctl|db\\.connect|suite\\.db|probe_worker" tests
```

Expected: no prohibited runtime calls. References in comments or monkeypatch targets
must be manually inspected and must not execute those APIs.

- [ ] **Step 3: Run complete verification**

Run:

```bash
uv run pytest -v
uv run python -m compileall -q wmx_suite tests
git diff --check
git status --short
```

Expected: all tests pass, compilation exits zero, no whitespace errors, and status
shows only the intended documentation changes before commit.

- [ ] **Step 4: Commit documentation correction**

```bash
git add README.md AGENTS.md
git commit -m "docs: clarify planned context cap limitation"
```

- [ ] **Step 5: Inspect final history and worktree**

Run:

```bash
git log -6 --oneline
git status --short --branch
```

Expected: the task commits are present and the worktree is clean.
