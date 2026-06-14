"""Tests for the characterize view (plain mode + color)."""
import io

from wmx_suite.ui import Console
from wmx_suite.views import characterize as view


def _c(verbose=False, color=False):
    buf = io.StringIO()
    return Console(color=color, verbose=verbose, stream=buf), buf


def test_header_block():
    c, buf = _c()
    view.render_header(c, {
        "model": "mlx-community/gemma-4-e4b-it-4bit", "cache_type": "RotatingKVCache",
        "kv_mode": "fp16", "wall_gb": 17.18, "safe_budget_gb": 15.18,
        "baseline_gb": 2.77, "est_gb": 9.28, "weights_gb": 5.2,
    })
    out = buf.getvalue()
    assert "Characterizing gemma-4-e4b-it-4bit" in out
    assert "RotatingKVCache" in out and "fp16 KV" in out
    assert "safe budget" in out and "15.18 GB" in out
    assert "pre-flight est" in out
    assert "\033" not in out


def test_rung_progress():
    c, buf = _c()
    view.render_rung(c, {"ctx": 2048, "os_wired_gb": 8.2, "delta_gb": 0.3,
                         "peak_gb": 5.0, "repeats": 3, "spread_gb": 0.1})
    out = buf.getvalue()
    assert "✓" in out and "2,048 tok" in out and "8.20 GB wired" in out


def test_stop():
    c, buf = _c()
    view.render_stop(c, {"ctx": 8192, "predicted_gb": 15.3, "safe_budget_gb": 15.18})
    out = buf.getvalue()
    assert "ceiling reached" in out and "8,192" in out


def test_summary_success():
    c, buf = _c()
    view.render_summary(c, {"model": "mlx-community/gemma-4-e4b-it-OptiQ-4bit",
                            "safe_ctx": 72400, "hard_wall_ctx": 103524,
                            "r2": 0.999, "n_points": 4})
    out = buf.getvalue()
    assert "safe to run" in out.lower()
    assert "72,400" in out and "103,524" in out
    assert "R²=0.999" in out
    assert "wmx-suite run --model gemma-4-e4b-it-OptiQ-4bit" in out


def test_refusal_hopeless():
    c, buf = _c()
    view.render_refusal(c, {"model": "mlx-community/huge", "kind": "hopeless",
                            "est_gb": 18.0, "threshold_gb": 15.18, "wall_gb": 17.18})
    out = buf.getvalue()
    assert "won't fit safely" in out.lower()
    assert "18.0" in out and "17.18" in out


def test_refusal_borderline_suggests_min_probe():
    c, buf = _c()
    view.render_refusal(c, {"model": "mlx-community/tight", "kind": "borderline",
                            "est_gb": 15.5, "threshold_gb": 15.18, "wall_gb": 17.18})
    out = buf.getvalue()
    assert "--min-probe" in out


def test_failure_load_error():
    c, buf = _c()
    view.render_failure(c, {"model": "mlx-community/gemma-4-e4b-it-4bit",
                            "note": "load failed — 126 weight tensors ..."})
    out = buf.getvalue()
    assert "Couldn't measure" in out
    assert "126 weight tensors" in out


def test_color_has_ansi():
    c, buf = _c(color=True)
    view.render_summary(c, {"model": "m", "safe_ctx": 1000, "hard_wall_ctx": 2000,
                            "r2": 0.9, "n_points": 2})
    assert "\033" in buf.getvalue()
