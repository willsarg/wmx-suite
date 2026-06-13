import pytest

from wmx_suite.system import SystemLimits


def _limits(wall_gb: float = 17.18) -> SystemLimits:
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
    assert _limits().safe_threshold_gb() == pytest.approx(15.18)


def test_safe_threshold_uses_requested_margin():
    assert _limits(20.0).safe_threshold_gb(3.5) == pytest.approx(16.5)
