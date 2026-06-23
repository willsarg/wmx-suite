# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
from types import SimpleNamespace

from wmx_suite import kokoro_safety as ks


def _limits(threshold):
    return SimpleNamespace(safe_threshold_gb=lambda m=2.0: threshold,
                           wall_gb=17.18, wired_now_gb=3.0)


def test_preflight_safe_when_baseline_plus_headroom_below_threshold(monkeypatch):
    monkeypatch.setattr(ks.system, "read_limits", lambda: _limits(15.18))
    monkeypatch.setattr(ks.system, "sample_settled_baseline", lambda: 3.0)
    threshold, baseline, safe = ks.preflight(2.0)
    assert (threshold, baseline) == (15.18, 3.0)
    assert safe is True


def test_preflight_unsafe_reserves_model_load_headroom(monkeypatch):
    # baseline alone (15.0) is under threshold 15.18, but + MODEL_WEIGHT_EST_GB (0.5) is not.
    monkeypatch.setattr(ks.system, "read_limits", lambda: _limits(15.18))
    monkeypatch.setattr(ks.system, "sample_settled_baseline", lambda: 15.0)
    _threshold, _baseline, safe = ks.preflight(2.0)
    assert safe is False  # would have been (wrongly) "safe" without the headroom term


def test_over_threshold_fresh_read(monkeypatch):
    monkeypatch.setattr(ks.system, "wired_gb", lambda: 10.0)
    assert ks.over_threshold(15.18) is False
    monkeypatch.setattr(ks.system, "wired_gb", lambda: 15.18)
    assert ks.over_threshold(15.18) is True


def test_over_threshold_treats_read_failure_as_unsafe(monkeypatch):
    def boom():
        raise OSError("vm_stat failed")
    monkeypatch.setattr(ks.system, "wired_gb", boom)
    assert ks.over_threshold(15.18) is True  # fail-unsafe, not stale-fallback


def test_predicted_concurrent_peak_scales_with_batch():
    # current 4.0, per-call transient 0.5 GB, batch 16 -> 4.0 + 8.0 = 12.0
    assert ks.predicted_concurrent_peak(4.0, 0.5, 16) == 12.0
    assert ks.predicted_concurrent_peak(4.0, 0.5, 0) == 4.0
