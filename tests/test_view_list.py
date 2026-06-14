"""Tests for wmx_suite/views/list.py — pure rendering, hardware-free."""
import io
import pytest
from wmx_suite.ui import Console
from wmx_suite.views import list as list_view


# ---------------------------------------------------------------------------
# Realistic fixture matching renderListProposed in the mockup
# ---------------------------------------------------------------------------
FIXTURE = {
    "models": [
        {
            "hf_id": "mlx-community/Qwen2.5-VL-7B-Instruct-4bit",
            "loads_gb": 5.0,
            "safe_ctx": 28533,
            "speed_tps": 59.8,
            "fit": "good",
            "tight": False,
            # verbose-only
            "slope_gb_per_k": 0.2678,
            "r2": 0.9925,
            "hard_wall_ctx": 36001,
            "n_runs": 2,
        },
        {
            "hf_id": "mlx-community/Qwen3.5-9B-OptiQ-4bit",
            "loads_gb": 9.0,
            "safe_ctx": 47638,
            "speed_tps": None,
            "fit": "good",
            "tight": False,
            # verbose-only
            "slope_gb_per_k": 0.0820,
            "r2": 0.9975,
            "hard_wall_ctx": 72035,
            "n_runs": 0,
        },
        {
            "hf_id": "mlx-community/gemma-4-e4b-it-OptiQ-4bit",
            "loads_gb": 7.7,
            "safe_ctx": 74851,
            "speed_tps": 66.3,
            "fit": "good",
            "tight": False,
            # verbose-only
            "slope_gb_per_k": 0.0698,
            "r2": 0.999,
            "hard_wall_ctx": 103524,
            "n_runs": 1,
        },
        {
            "hf_id": "mlx-community/gpt-oss-20b-MXFP4-Q8",
            "loads_gb": 12.4,
            "safe_ctx": 1420,
            "speed_tps": None,
            "fit": "good",
            "tight": True,
            # verbose-only
            "slope_gb_per_k": 0.3600,
            "r2": 1.0,
            "hard_wall_ctx": 6976,
            "n_runs": 0,
        },
    ],
}


def _make_console(verbose=False, color=False) -> tuple[Console, io.StringIO]:
    stream = io.StringIO()
    console = Console(color=color, verbose=verbose, stream=stream)
    return console, stream


# ---------------------------------------------------------------------------
# Normal mode golden
# ---------------------------------------------------------------------------

def test_normal_plain_golden():
    console, stream = _make_console()
    list_view.render(console, FIXTURE)
    out = stream.getvalue()

    # Section header
    assert "Models measured on this machine" in out

    # Table headers
    assert "model" in out
    assert "loads at" in out
    assert "safe context" in out
    assert "speed" in out
    assert "fit" in out

    # Each short name (prefix stripped)
    assert "Qwen2.5-VL-7B-Instruct-4bit" in out
    assert "Qwen3.5-9B-OptiQ-4bit" in out
    assert "gemma-4-e4b-it-OptiQ-4bit" in out
    assert "gpt-oss-20b-MXFP4-Q8" in out

    # mlx-community/ prefix must be stripped
    assert "mlx-community/" not in out

    # Human loads values
    assert "5.0 GB" in out
    assert "9.0 GB" in out
    assert "7.7 GB" in out
    assert "12.4 GB" in out

    # Human-rounded safe context (nearest 100)
    # 28533 → ~28,500 tok
    assert "~28,500 tok" in out
    # 74851 → ~74,900 tok
    assert "~74,900 tok" in out

    # Speed: shown where measured, dash where not
    assert "~60 tok/s" in out  # 59.8 rounds to 60
    assert "~66 tok/s" in out  # 66.3 rounds to 66
    assert "—" in out

    # Fit labels
    assert "good" in out

    # Tight flag shown
    assert "⚠ tight" in out

    # Crash context HIDDEN in normal mode
    assert "36,001" not in out
    assert "6,976" not in out
    assert "raw fit" not in out
    assert "GB/1k" not in out

    # Legend gloss present
    assert "loads at" in out
    assert "safe context" in out
    assert "speed" in out
    assert "— = not run yet" in out

    # next_block
    assert "wmx-suite run --model <model>" in out


# ---------------------------------------------------------------------------
# Verbose mode — appends raw fit table
# ---------------------------------------------------------------------------

def test_verbose_appends_raw_fit():
    console, stream = _make_console(verbose=True)
    list_view.render(console, FIXTURE)
    out = stream.getvalue()

    # Normal content still present
    assert "Models measured on this machine" in out
    assert "~28,500 tok" in out

    # Raw fit section
    assert "raw fit (--verbose)" in out
    assert "GB/1k" in out
    assert "R²" in out
    assert "crash ctx" in out

    # Raw slope values
    assert "0.2678" in out
    assert "0.0820" in out
    assert "0.3600" in out

    # Crash contexts revealed in verbose
    assert "36,001" in out
    assert "6,976" in out

    # R² values
    assert "0.9925" in out
    assert "1.0" in out

    # Run counts
    assert "2" in out
    assert "0" in out


def test_verbose_raw_absent_in_normal():
    console, stream = _make_console(verbose=False)
    list_view.render(console, FIXTURE)
    out = stream.getvalue()
    assert "raw fit" not in out
    assert "0.2678" not in out


# ---------------------------------------------------------------------------
# Color smoke tests
# ---------------------------------------------------------------------------

def test_color_emits_ansi():
    console, stream = _make_console(color=True)
    list_view.render(console, FIXTURE)
    out = stream.getvalue()
    assert "\033[" in out


def test_plain_no_ansi():
    console, stream = _make_console(color=False)
    list_view.render(console, FIXTURE)
    out = stream.getvalue()
    assert "\033" not in out


# ---------------------------------------------------------------------------
# Tight flag only shown for tight models
# ---------------------------------------------------------------------------

def test_tight_flag_only_on_tight_model():
    console, stream = _make_console()
    list_view.render(console, FIXTURE)
    out = stream.getvalue()
    lines = out.splitlines()
    tight_lines = [l for l in lines if "⚠ tight" in l]
    assert len(tight_lines) == 1  # only gpt-oss-20b is tight
    assert "gpt-oss-20b-MXFP4-Q8" in tight_lines[0]


# ---------------------------------------------------------------------------
# Speed dash for unrun models
# ---------------------------------------------------------------------------

def test_dash_for_no_speed():
    console, stream = _make_console()
    list_view.render(console, FIXTURE)
    out = stream.getvalue()
    # Qwen3.5 and gpt-oss both have no speed; at least one dash per no-speed model
    lines = out.splitlines()
    dash_lines = [l for l in lines if "—" in l]
    # at least 2 dash appearances (one per model without speed)
    assert len(dash_lines) >= 2


# ---------------------------------------------------------------------------
# next_block at end
# ---------------------------------------------------------------------------

def test_next_block_present():
    console, stream = _make_console()
    list_view.render(console, FIXTURE)
    out = stream.getvalue()
    assert "next" in out
    assert "wmx-suite run --model <model>" in out
    assert "launch one safely" in out
