import pytest

from wmx_suite import db, models, probe
from wmx_suite.system import SystemLimits


def _model_info(weights_gb: float = 8.0) -> models.ModelInfo:
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


def _limits(wired_now_gb: float) -> SystemLimits:
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
