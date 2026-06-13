import pytest

from wmx_suite import db, models, probe
from wmx_suite.system import SystemLimits


def _model_info(
    weights_gb: float = 8.0,
    is_causal: bool = True,
    can_quantize_kv: bool = True,
    max_context: int = 32768,
) -> models.ModelInfo:
    return models.ModelInfo(
        hf_id="mlx-community/test",
        weights_gb=weights_gb,
        n_layers=4,
        growing_layers=2,
        kv_heads=8,
        head_dim=128,
        hidden_size=1024,
        max_context=max_context,
        cache_type="standard",
        can_quantize_kv=can_quantize_kv,
        is_causal=is_causal,
        layer_types={"full_attention": 2, "linear_attention": 2},
    )


def _limits(wired_now_gb: float = 3.0, wall_gb: float = 17.18) -> SystemLimits:
    return SystemLimits(
        device="test",
        total_gb=24.0,
        wall_gb=wall_gb,
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


def test_estimate_base_uses_live_baseline_above_floor(monkeypatch, tmp_path):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "suite.db")
    con = db.connect()
    expected = 4.0 + 8.0 * probe.RESIDENT_FACTOR + probe.FIXED_OVERHEAD_GB
    assert probe.estimate_base_gb(_model_info(), _limits(4.0), con) == pytest.approx(expected)


def test_estimate_base_uses_two_point_five_gb_baseline_floor(monkeypatch, tmp_path):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "suite.db")
    con = db.connect()
    expected = 2.5 + 8.0 * probe.RESIDENT_FACTOR + probe.FIXED_OVERHEAD_GB
    assert probe.estimate_base_gb(_model_info(), _limits(1.0), con) == pytest.approx(expected)


def test_estimate_base_gb_uses_profile_overhead(monkeypatch, tmp_path):
    from wmx_suite import profiles
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "suite.db")
    key = ("Apple M4 Pro", 1, 15)
    monkeypatch.setattr(profiles, "machine_key", lambda: key)
    con = db.connect()
    info = _model_info(weights_gb=4.0)
    limits = _limits(wired_now_gb=3.0)
    base_default = probe.estimate_base_gb(info, limits, con)   # overhead 1.0
    db.upsert_profile(con, key, resident_factor=1.05, fixed_overhead_gb=2.5,
                      model_id="m", n_points=2, mlx_version="9.9")
    base_profile = probe.estimate_base_gb(info, limits, con)
    assert round(base_profile - base_default, 3) == 1.5   # overhead 2.5 - 1.0


def test_pick_calibration_model_smallest_causal(monkeypatch):
    from wmx_suite import models, probe
    monkeypatch.setattr(models, "scan_cache", lambda: ["org/big", "org/small", "org/notcausal"])

    def fake_describe(hf_id):
        table = {
            "org/big": _model_info(weights_gb=8.0, is_causal=True),
            "org/small": _model_info(weights_gb=0.5, is_causal=True),
            "org/notcausal": _model_info(weights_gb=0.1, is_causal=False),
        }
        return table[hf_id]
    monkeypatch.setattr(models, "describe", fake_describe)
    assert probe._pick_calibration_model() == "org/small"


def test_pick_calibration_model_errors_when_none(monkeypatch):
    import pytest
    from wmx_suite import models, probe
    monkeypatch.setattr(models, "scan_cache", lambda: [])
    with pytest.raises(SystemExit, match="no causal"):
        probe._pick_calibration_model()


def test_calibrate_solves_and_floors_overhead(monkeypatch, tmp_path):
    from wmx_suite import db, models, probe, profiles
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "suite.db")
    key = ("Apple M4 Pro", 1, 15)
    monkeypatch.setattr(profiles, "machine_key", lambda: key)
    info = _model_info(weights_gb=0.5, is_causal=True, can_quantize_kv=True, max_context=32768)
    monkeypatch.setattr(models, "describe", lambda hf_id: info)
    monkeypatch.setattr(probe, "read_limits", lambda: _limits(wall_gb=17.0, wired_now_gb=3.0))
    monkeypatch.setattr(probe, "sample_settled_baseline", lambda: 3.0)

    deltas = {512: 2.05, 2048: 2.20}  # intercept ~2.0 at c->0
    def fake_measure(py, hf_id, ctx, kv_bits, repeats, *, verbose, log):
        return {"status": "ok", "context": ctx, "repeats": 3, "delta": deltas[ctx],
                "os_wired_gb": 5.0, "mlx_peak_gb": 4.0, "spread_gb": 0.1}
    monkeypatch.setattr(probe, "_measure_rung", fake_measure)

    result = probe.calibrate("org/tiny", verbose=False)
    assert abs(result["intercept_gb"] - 2.0) < 0.05
    assert abs(result["measured_overhead_gb"] - 1.475) < 0.05   # 2.0 - 1.05*0.5
    assert result["fixed_overhead_gb"] >= profiles.DEFAULT_FIXED_OVERHEAD_GB
    stored = db.get_profile(db.connect(), key)
    assert stored["fixed_overhead_gb"] == result["fixed_overhead_gb"]


def test_calibrate_floor_applies_when_residual_low(monkeypatch, tmp_path):
    from wmx_suite import db, models, probe, profiles
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "suite.db")
    monkeypatch.setattr(profiles, "machine_key", lambda: ("Apple M4 Pro", 1, 15))
    info = _model_info(weights_gb=0.5, is_causal=True, can_quantize_kv=True, max_context=32768)
    monkeypatch.setattr(models, "describe", lambda hf_id: info)
    monkeypatch.setattr(probe, "read_limits", lambda: _limits(wall_gb=17.0, wired_now_gb=3.0))
    monkeypatch.setattr(probe, "sample_settled_baseline", lambda: 3.0)
    deltas = {512: 0.30, 2048: 0.30}  # residual goes negative -> floor applies
    monkeypatch.setattr(probe, "_measure_rung",
                        lambda py, hf, ctx, kv, r, *, verbose, log:
                        {"status": "ok", "repeats": 3, "delta": deltas[ctx], "mlx_peak_gb": 1.0,
                         "os_wired_gb": 3.3, "spread_gb": 0.0})
    result = probe.calibrate("org/tiny", verbose=False)
    assert result["fixed_overhead_gb"] == profiles.DEFAULT_FIXED_OVERHEAD_GB


def test_calibrate_aborts_when_live_pressure_high(monkeypatch, tmp_path):
    import pytest
    from wmx_suite import db, models, probe, profiles
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "suite.db")
    monkeypatch.setattr(profiles, "machine_key", lambda: ("Apple M4 Pro", 1, 15))
    info = _model_info(weights_gb=0.5, is_causal=True, can_quantize_kv=True, max_context=32768)
    monkeypatch.setattr(models, "describe", lambda hf_id: info)
    # pre-flight passes (wired_now low), but the live settled baseline has spiked:
    monkeypatch.setattr(probe, "read_limits", lambda: _limits(wall_gb=17.0, wired_now_gb=3.0))
    monkeypatch.setattr(probe, "sample_settled_baseline", lambda: 16.0)  # above threshold
    called = {"n": 0}
    def fake_measure(*a, **k):
        called["n"] += 1
        return {"status": "ok", "repeats": 3, "delta": 2.0, "mlx_peak_gb": 1.0,
                "os_wired_gb": 3.3, "spread_gb": 0.0}
    monkeypatch.setattr(probe, "_measure_rung", fake_measure)
    with pytest.raises(SystemExit, match="aborting"):
        probe.calibrate("org/tiny", verbose=False)
    assert called["n"] == 0  # aborted BEFORE launching any rung
