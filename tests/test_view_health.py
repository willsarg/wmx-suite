"""Tests for wmx_suite/views/health.py.

Hardware-free. Builds a Console with a StringIO stream, calls render, and
checks the output byte-for-byte against a golden string captured from the M4 Pro
testbed numbers used in the approved mockup.
"""
from __future__ import annotations

import io

import pytest

from wmx_suite.ui import Console
from wmx_suite.views.health import render

# ---------------------------------------------------------------------------
# Shared fixture data (M4 Pro testbed numbers from mockup)
# ---------------------------------------------------------------------------
FIXTURE = {
    "wall_gb": 17.18,
    "safe_budget_gb": 15.18,
    "free_now_gb": 12.04,
    "swap_free_gb": 1.13,
    "swap_warn": True,
    "margin_gb": 2.0,
    "baseline_gb": 3.14,
    "baseline_sample": "median of 3",
    "models": [
        {
            "name": "Qwen2.5-VL-7B-Instruct-4bit",
            "loads_gb": 8.10,
            "spare_gb": 7.08,
            "safe_ctx": 26439,
            "safe_ctx_str": "~26,400 tokens",
            "ok": True,
            "base_gb": 8.10,
            "slope_gb_per_k": 0.2678,
            "safe_cap_tok": 26439,
        },
        {
            "name": "Qwen3.5-9B-OptiQ-4bit",
            "loads_gb": 12.19,
            "spare_gb": 2.99,
            "safe_ctx": 36495,
            "safe_ctx_str": "~36,500 tokens",
            "ok": True,
            "base_gb": 12.19,
            "slope_gb_per_k": 0.0820,
            "safe_cap_tok": 36495,
        },
        {
            "name": "gemma-4-e4b-it-OptiQ-4bit",
            "loads_gb": 10.85,
            "spare_gb": 4.33,
            "safe_ctx": 62048,
            "safe_ctx_str": "~62,000 tokens",
            "ok": True,
            "base_gb": 10.85,
            "slope_gb_per_k": 0.0698,
            "safe_cap_tok": 62048,
        },
        {
            "name": "gpt-oss-20b-MXFP4-Q8",
            "loads_gb": 15.58,
            "spare_gb": -0.40,
            "safe_ctx": 0,
            "safe_ctx_str": "over budget — won't load",
            "ok": False,
            "base_gb": 15.58,
            "slope_gb_per_k": 0.3600,
            "safe_cap_tok": 0,
        },
    ],
}

GOLDEN_NORMAL = (
    "Can each model load and run safely right now?\n"
    "\n"
    "crash wall      : 17.18 GB   cross it and the Mac can hard-lock\n"
    "safe budget     : 15.18 GB   crash wall − 2 GB margin (the planning ceiling)\n"
    "free now        : 12.04 GB   safe budget − memory already wired\n"
    "swap free       : 1.13 GB ⚠ tiny   almost no fallback if you cross the wall\n"
    "\n"
    "      model                          loads at   spare room   safe context\n"
    "   ✓  Qwen2.5-VL-7B-Instruct-4bit     8.10 GB     +7.08 GB   ~26,400 tokens\n"
    "   ✓  Qwen3.5-9B-OptiQ-4bit          12.19 GB     +2.99 GB   ~36,500 tokens\n"
    "   ✓  gemma-4-e4b-it-OptiQ-4bit      10.85 GB     +4.33 GB   ~62,000 tokens\n"
    "   ✗  gpt-oss-20b-MXFP4-Q8           15.58 GB     −0.40 GB   over budget — won't load\n"
    "\n"
    "  loads at     memory the model needs just to load (no context yet)\n"
    "  spare room   safe budget left after it loads — bigger is safer\n"
    "  safe context largest prompt+reply we can guarantee won't cross the wall\n"
    "\n"
    "next\n"
    "  wmx-suite show <model>                architecture + KV-cache memory class for one model\n"
    "  wmx-suite run --model <m> --dry-run   preview the safe launch plan without running it\n"
    "  wmx-suite characterize <model>        measure the safe ceiling of a model not listed yet\n"
)

GOLDEN_VERBOSE = (
    "Can each model load and run safely right now?\n"
    "\n"
    "crash wall      : 17.18 GB   cross it and the Mac can hard-lock\n"
    "safe budget     : 15.18 GB   crash wall − 2 GB margin (the planning ceiling)\n"
    "free now        : 12.04 GB   safe budget − memory already wired\n"
    "swap free       : 1.13 GB ⚠ tiny   almost no fallback if you cross the wall\n"
    "\n"
    "      model                          loads at   spare room   safe context\n"
    "   ✓  Qwen2.5-VL-7B-Instruct-4bit     8.10 GB     +7.08 GB   ~26,400 tokens\n"
    "   ✓  Qwen3.5-9B-OptiQ-4bit          12.19 GB     +2.99 GB   ~36,500 tokens\n"
    "   ✓  gemma-4-e4b-it-OptiQ-4bit      10.85 GB     +4.33 GB   ~62,000 tokens\n"
    "   ✗  gpt-oss-20b-MXFP4-Q8           15.58 GB     −0.40 GB   over budget — won't load\n"
    "\n"
    "raw (--verbose)   margin 2.00 GB · baseline 3.14 GB (median of 3)\n"
    "  model                          base GB     GB/1k   safe cap tok\n"
    "  Qwen2.5-VL-7B-Instruct-4bit       8.10    0.2678          26439\n"
    "  Qwen3.5-9B-OptiQ-4bit            12.19    0.0820          36495\n"
    "  gemma-4-e4b-it-OptiQ-4bit        10.85    0.0698          62048\n"
    "  gpt-oss-20b-MXFP4-Q8             15.58    0.3600              0\n"
    "\n"
    "  loads at     memory the model needs just to load (no context yet)\n"
    "  spare room   safe budget left after it loads — bigger is safer\n"
    "  safe context largest prompt+reply we can guarantee won't cross the wall\n"
    "\n"
    "next\n"
    "  wmx-suite show <model>                architecture + KV-cache memory class for one model\n"
    "  wmx-suite run --model <m> --dry-run   preview the safe launch plan without running it\n"
    "  wmx-suite characterize <model>        measure the safe ceiling of a model not listed yet\n"
)


def _run(verbose: bool, color: bool = False) -> str:
    stream = io.StringIO()
    con = Console(color=color, verbose=verbose, stream=stream)
    render(con, FIXTURE)
    return stream.getvalue()


# ---------------------------------------------------------------------------
# Golden tests (plain, no color)
# ---------------------------------------------------------------------------

def test_health_normal_golden():
    assert _run(verbose=False) == GOLDEN_NORMAL


def test_health_verbose_golden():
    assert _run(verbose=True) == GOLDEN_VERBOSE


def test_health_verbose_contains_raw_appendix():
    out = _run(verbose=True)
    assert "raw (--verbose)" in out
    assert "base GB" in out
    assert "GB/1k" in out


def test_health_normal_no_raw_appendix():
    out = _run(verbose=False)
    assert "raw (--verbose)" not in out


def test_health_bad_row_shows_cross():
    out = _run(verbose=False)
    assert "✗" in out
    assert "over budget" in out


def test_health_good_rows_show_check():
    out = _run(verbose=False)
    assert "✓" in out
    assert "~26,400 tokens" in out


# ---------------------------------------------------------------------------
# Color smoke tests
# ---------------------------------------------------------------------------

def test_health_color_mode_contains_ansi():
    out = _run(verbose=False, color=True)
    assert "\033[" in out


def test_health_plain_mode_no_ansi():
    out = _run(verbose=False, color=False)
    assert "\033" not in out


def test_health_color_verbose_contains_ansi():
    out = _run(verbose=True, color=True)
    assert "\033[" in out
