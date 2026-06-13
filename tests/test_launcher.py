import pytest

from wmx_suite import launcher, models
from wmx_suite.system import SystemLimits


def _model_info(*, quantizable=True, max_context=131072, weights_gb=8.0):
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


def _install_plan_fakes(monkeypatch, *, info, fit, live_base=3.0):
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
    monkeypatch.setattr(launcher.models, "fit_is_stale", lambda _hf_id, _created: False)


def _plan_dict(*, kv_bits=4, max_kv_size=4096):
    return {
        "kv_bits": kv_bits,
        "kv_group_size": launcher.KV_GROUP_SIZE,
        "quantized_kv_start": launcher.QUANTIZED_KV_START,
        "max_kv_size": max_kv_size,
    }


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
def test_predict_nonpositive_slope_with_headroom_is_zero(slope):
    # A non-growing curve is not defensible evidence for granting maximum context.
    result = launcher.predict(
        model_base_gb=8.0,
        slope_gb_per_k=slope,
        live_base_gb=3.0,
        threshold_gb=15.0,
        wall_gb=17.0,
        model_max=32768,
    )
    assert result.safe_ctx == 0


def test_estimated_slope_converts_units_and_applies_prefill_multiplier():
    info = _model_info()
    expected = (
        info.fp16_kv_bytes_per_token()
        * 1000
        / 1e9
        * launcher.PREFILL_SPIKE_MULT
    )
    assert launcher._estimated_slope_gb_per_k(info) == pytest.approx(expected)


def test_estimated_slope_is_zero_without_kv_metadata():
    info = _model_info()
    info.kv_heads = None
    assert launcher._estimated_slope_gb_per_k(info) == 0.0


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
    _install_plan_fakes(
        monkeypatch,
        info=_model_info(),
        fit={"model_base_gb": 8.0, "slope_gb_per_k": 0.1},
    )
    result = launcher.plan("mlx-community/test")
    assert result["source"] == "measured"
    assert result["max_kv_size"] == 40000
    assert result["kv_bits"] == 4


def test_plan_marks_measured_fit_stale(monkeypatch):
    fit = {
        "model_base_gb": 8.0,
        "slope_gb_per_k": 0.1,
        "characterized_at": "2026-06-13T00:00:00+00:00",
    }
    _install_plan_fakes(monkeypatch, info=_model_info(), fit=fit)
    monkeypatch.setattr(launcher.models, "fit_is_stale", lambda _hf_id, _created: True)

    assert launcher.plan("mlx-community/test")["fit_stale"] is True


def test_plan_estimate_is_not_labeled_stale(monkeypatch):
    _install_plan_fakes(monkeypatch, info=_model_info(), fit=None)
    assert launcher.plan("mlx-community/test")["fit_stale"] is False


def test_plan_refuses_wall_equality(monkeypatch):
    _install_plan_fakes(
        monkeypatch,
        info=_model_info(),
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
    live_base = 3.0
    model_base = 15.0 - live_base - safe_ctx / 1000
    _install_plan_fakes(
        monkeypatch,
        info=_model_info(),
        fit={"model_base_gb": model_base, "slope_gb_per_k": slope},
        live_base=live_base,
    )
    assert launcher.plan("mlx-community/test")["refuse"] is refuse


def test_plan_uses_estimate_for_uncharacterized_model(monkeypatch):
    info = _model_info(weights_gb=4.0)
    _install_plan_fakes(monkeypatch, info=info, fit=None)
    result = launcher.plan("mlx-community/test")
    assert result["source"] == "estimated"
    assert result["model_base_gb"] == round(
        info.weights_gb * launcher.RESIDENT_FACTOR + launcher.FIXED_OVERHEAD_GB,
        2,
    )


def test_plan_refuses_when_estimated_slope_is_unavailable(monkeypatch):
    info = _model_info()
    info.kv_heads = None
    _install_plan_fakes(monkeypatch, info=info, fit=None)
    result = launcher.plan("mlx-community/test")
    assert result["source"] == "estimated"
    assert result["refuse"] is True
    assert result["max_kv_size"] == 0


def test_plan_omits_kv_quantization_for_rotating_cache(monkeypatch):
    _install_plan_fakes(
        monkeypatch,
        info=_model_info(quantizable=False),
        fit={"model_base_gb": 8.0, "slope_gb_per_k": 0.1},
    )
    assert launcher.plan("mlx-community/test")["kv_bits"] is None


def test_plan_uses_environment_margin_by_default(monkeypatch):
    monkeypatch.setenv("WMX_SUITE_MARGIN_GB", "3")
    _install_plan_fakes(
        monkeypatch,
        info=_model_info(),
        fit={"model_base_gb": 8.0, "slope_gb_per_k": 0.1},
    )
    assert launcher.plan("mlx-community/test")["threshold_gb"] == 14.0


def test_plan_explicit_margin_overrides_environment(monkeypatch):
    monkeypatch.setenv("WMX_SUITE_MARGIN_GB", "3")
    _install_plan_fakes(
        monkeypatch,
        info=_model_info(),
        fit={"model_base_gb": 8.0, "slope_gb_per_k": 0.1},
    )
    assert launcher.plan("mlx-community/test", margin_gb=1.0)["threshold_gb"] == 16.0


@pytest.mark.parametrize("args", [["--model", "x"], ["--model=x"]])
def test_build_argv_injects_planned_values(args):
    result = launcher.build_argv(args, _plan_dict(), force=False)
    assert result[:8] == [
        "--max-kv-size",
        "4096",
        "--quantized-kv-start",
        "5000",
        "--kv-group-size",
        "64",
        "--kv-bits",
        "4",
    ]


@pytest.mark.parametrize(
    "args",
    [
        ["--model", "x", "--max-kv-size", "2048"],
        ["--model=x", "--max-kv-size=2048"],
    ],
)
def test_build_argv_preserves_context_at_or_below_plan(args):
    result = launcher.build_argv(args, _plan_dict(), force=False)
    assert "--max-kv-size=2048" in result or result[-2:] == ["--max-kv-size", "2048"]


@pytest.mark.parametrize(
    "args",
    [
        ["--model", "x", "--kv-bits", "8"],
        ["--model=x", "--kv-bits=8"],
    ],
)
def test_build_argv_rejects_uncharacterized_kv_bits_without_force(args):
    with pytest.raises(launcher.LaunchArgumentError, match="characterized setting"):
        launcher.build_argv(args, _plan_dict(), force=False)


@pytest.mark.parametrize(
    "args",
    [
        ["--model", "x", "--kv-bits", "8"],
        ["--model=x", "--kv-bits=8"],
    ],
)
def test_build_argv_preserves_uncharacterized_kv_bits_with_force(args):
    result = launcher.build_argv(args, _plan_dict(), force=True)
    assert "--kv-bits=8" in result or result[-2:] == ["--kv-bits", "8"]


@pytest.mark.parametrize(
    "args",
    [
        ["--model", "x", "--kv-group-size", "32"],
        ["--model=x", "--kv-group-size=32"],
        ["--model", "x", "--quantized-kv-start", "0"],
        ["--model=x", "--quantized-kv-start=0"],
    ],
)
def test_build_argv_rejects_uncharacterized_kv_settings_without_force(args):
    with pytest.raises(launcher.LaunchArgumentError, match="characterized setting"):
        launcher.build_argv(args, _plan_dict(), force=False)


def test_build_argv_preserves_matching_explicit_kv_settings():
    args = [
        "--model",
        "x",
        "--kv-bits",
        "4",
        "--kv-group-size",
        "64",
        "--quantized-kv-start",
        "5000",
    ]
    result = launcher.build_argv(args, _plan_dict(), force=False)
    assert result == ["--max-kv-size", "4096", *args]


@pytest.mark.parametrize(
    "args",
    [
        ["--model", "x", "--max-kv-size", "8192"],
        ["--model=x", "--max-kv-size=8192"],
    ],
)
def test_build_argv_rejects_context_above_plan_without_force(args):
    with pytest.raises(launcher.LaunchArgumentError, match="planned cap"):
        launcher.build_argv(args, _plan_dict(), force=False)


def test_build_argv_preserves_context_above_plan_with_force():
    args = ["--model", "x", "--max-kv-size", "8192"]
    assert launcher.build_argv(args, _plan_dict(), force=True) == [
        "--quantized-kv-start",
        "5000",
        "--kv-group-size",
        "64",
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
        launcher.build_argv(args, _plan_dict(), force=False)


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
        launcher.build_argv(args, _plan_dict(), force=False)


@pytest.mark.parametrize(
    "args",
    [
        ["--model", "x", "--kv-group-size"],
        ["--model", "x", "--kv-group-size=bad"],
        ["--model", "x", "--kv-group-size", "64", "--kv-group-size=32"],
        ["--model", "x", "--quantized-kv-start"],
        ["--model", "x", "--quantized-kv-start=bad"],
        [
            "--model",
            "x",
            "--quantized-kv-start",
            "5000",
            "--quantized-kv-start=0",
        ],
    ],
)
def test_build_argv_rejects_invalid_or_duplicate_kv_settings(args):
    with pytest.raises(launcher.LaunchArgumentError):
        launcher.build_argv(args, _plan_dict(), force=False)


@pytest.mark.parametrize(
    "args",
    [
        ["--model", "x", "--kv-bits", "4"],
        ["--model=x", "--kv-bits=4"],
        ["--model", "x", "--kv-group-size", "64"],
        ["--model=x", "--quantized-kv-start=5000"],
    ],
)
def test_build_argv_rejects_kv_quantization_for_rotating_cache_even_with_force(args):
    with pytest.raises(launcher.LaunchArgumentError, match="not quantizable"):
        launcher.build_argv(args, _plan_dict(kv_bits=None), force=True)


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
        launcher.build_argv(args, _plan_dict(), force=False)


def test_build_argv_allows_unmeasured_modes_with_force():
    args = ["--model", "x", "--adapter-path", "adapter"]
    result = launcher.build_argv(args, _plan_dict(), force=True)
    assert result[-4:] == args
