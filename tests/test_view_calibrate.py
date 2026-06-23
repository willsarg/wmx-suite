# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""Tests for the calibrate view."""
import io

from wmx_suite.ui import Console
from wmx_suite.views import calibrate as view


def _c(color=False):
    buf = io.StringIO()
    return Console(color=color, verbose=False, stream=buf), buf


def test_header():
    c, buf = _c()
    view.render_header(c, {"model": "mlx-community/tiny", "weights_gb": 0.5,
                           "threshold_gb": 15.18, "kv_mode": "4-bit"})
    out = buf.getvalue()
    assert "Calibrating this machine's cold-start estimate" in out
    assert "mlx-community/tiny" in out
    assert "\033" not in out


def test_rung():
    c, buf = _c()
    view.render_rung(c, {"ctx": 2048, "delta_gb": 0.30, "repeats": 3})
    out = buf.getvalue()
    assert "✓" in out and "2,048 tok" in out and "+0.30 GB over base" in out


def test_summary_changed():
    c, buf = _c()
    view.render_summary(c, {"machine": "Apple M4 Pro / 26GB / macOS 15",
                            "model": "org/tiny", "n_points": 2,
                            "measured_overhead_gb": 1.48, "default_overhead_gb": 1.0,
                            "fixed_overhead_gb": 1.48, "changed": True})
    out = buf.getvalue()
    assert "Calibrated this machine" in out
    assert "Apple M4 Pro" in out and "org/tiny" in out and "1.48" in out
    assert "tightened" in out


def test_summary_kept_floor():
    c, buf = _c()
    view.render_summary(c, {"machine": "M", "model": "m", "n_points": 2,
                            "measured_overhead_gb": 0.6, "default_overhead_gb": 1.0,
                            "fixed_overhead_gb": 1.0, "changed": False})
    assert "kept the safe default" in buf.getvalue()


def test_abort_memory_kind():
    c, buf = _c()
    view.render_abort(c, {"reason": "machine too loaded.", "kind": "memory"})
    out = buf.getvalue()
    assert "Couldn't calibrate" in out
    assert "free up memory" in out


def test_abort_load_kind_suggests_other_model():
    c, buf = _c()
    view.render_abort(c, {"reason": "load failed — incompatible checkpoint.",
                          "kind": "load"})
    out = buf.getvalue()
    # a load failure is fixed by a different model, not by freeing memory
    assert "--model <other>" in out
    assert "free up memory" not in out


def test_color_has_ansi():
    c, buf = _c(color=True)
    view.render_rung(c, {"ctx": 512, "delta_gb": 0.1, "repeats": 3})
    assert "\033" in buf.getvalue()
