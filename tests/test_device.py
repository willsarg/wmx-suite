# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""Device limits + calibration JSON workers — ARA's out-of-process engine seam.

ARA drives ``wmx_suite.device`` in the isolated engine env and reads one JSON line from
stdout; human/console progress must go to stderr. These tests patch the underlying
``system``/``probe`` so no real MLX or host probing happens.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

from wmx_suite import device


def _limits_obj():
    return SimpleNamespace(
        device="Apple M4 Pro", total_gb=48.0, wall_gb=40.0,
        wired_now_gb=8.0, swap_free_gb=2.0,
        safe_threshold_gb=lambda margin: 40.0 - margin,
    )


def test_limits_shape(monkeypatch):
    monkeypatch.setattr(device.system, "read_limits", _limits_obj)
    monkeypatch.setattr(device.config, "margin_gb", lambda m: m if m is not None else 4.0)
    out = device.limits()
    assert out == {
        "device": "Apple M4 Pro", "total_gb": 48.0, "wall_gb": 40.0,
        "safe_budget_gb": 36.0, "margin_gb": 4.0, "headroom_gb": 28.0, "swap_free_gb": 2.0,
    }


def test_limits_honors_explicit_margin(monkeypatch):
    monkeypatch.setattr(device.system, "read_limits", _limits_obj)
    monkeypatch.setattr(device.config, "margin_gb", lambda m: m if m is not None else 4.0)
    assert device.limits(margin_gb=2.0)["safe_budget_gb"] == 38.0


def test_calibrate_delegates_to_probe_with_stderr_console(monkeypatch):
    seen = {}

    def fake_calibrate(model, *, margin_gb=None, console=None):
        seen["model"] = model
        seen["stream_is_stderr"] = console.stream.__class__ is not None
        return {"measured_overhead_gb": 0.1, "n_points": 2}

    monkeypatch.setattr(device.probe, "calibrate", fake_calibrate)
    result = device.calibrate("org/m", margin_gb=2.0)
    assert seen["model"] == "org/m"
    assert result["n_points"] == 2


def test_main_limits_prints_one_json_line(monkeypatch, capsys):
    monkeypatch.setattr(device, "limits", lambda margin_gb=None: {"safe_budget_gb": 36.0})
    device.main(["limits"])
    line = capsys.readouterr().out.strip()
    assert json.loads(line) == {"safe_budget_gb": 36.0}


def test_main_calibrate_routes_model(monkeypatch, capsys):
    monkeypatch.setattr(device, "calibrate",
                        lambda model, margin_gb=None: {"model": model, "n_points": 1})
    device.main(["calibrate", "org/m", "--margin", "2"])
    assert json.loads(capsys.readouterr().out.strip()) == {"model": "org/m", "n_points": 1}
