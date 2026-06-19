"""Tests for wmx_suite/views/landing.py.

Hardware-free. Builds a Console with a StringIO stream, calls render, and
checks the output byte-for-byte against a golden string captured from the M4 Pro
testbed numbers used in the approved mockup.
"""
from __future__ import annotations

import io

import pytest

from wmx_suite.ui import Console
from wmx_suite.views.landing import render

# ---------------------------------------------------------------------------
# Shared fixture data (M4 Pro testbed numbers from mockup)
# ---------------------------------------------------------------------------
FIXTURE = {
    "device": "Apple M4 Pro",
    "free_gb": 12.04,
    "safe_budget_gb": 15.18,
    "models_ready": 3,
    "calibrated": True,
}

GOLDEN_NORMAL = (
    "  wmx-suite  —  run local MLX models on Apple Silicon without crashing your Mac\n"
    "\n"
    "  this machine: Apple M4 Pro · 12.04 GB free of a 15.18 GB safe budget · 3 models ready · calibrated\n"
    "\n"
    "  NEW HERE?  first time on this machine, run these in order:\n"
    "     1) wmx-suite characterize <m>  measure a model's safe context ceiling\n"
    "     2) wmx-suite run --model <m>   launch it safely (auto KV-bits + context cap)\n"
    "\n"
    "  YOUR MACHINE\n"
    "  system                specs + memory budget (what your Mac can safely give a model)\n"
    "  health                live go/no-go: which models can run right now, and how big\n"
    "\n"
    "  MODELS\n"
    "  characterize <model>  safely measure a model's max safe context\n"
    "  list                  show measured ceilings stored for this machine\n"
    "  calibrate             tune this machine's cold-start memory estimate\n"
    "\n"
    "  RUN\n"
    "  run --model <model>   safe launch — refuses anything that would crash the Mac\n"
    "  web                   browser dashboard for all of the above\n"
    "\n"
    "  benchmarks (TTS / embeddings, for power users): wmx-suite benchmark-* --help\n"
    "\n"
    "  details on any command:  wmx-suite <command> --help"
    "   ·   everything at once:  wmx-suite --verbose\n"
)

GOLDEN_VERBOSE = (
    "  wmx-suite  —  run local MLX models on Apple Silicon without crashing your Mac\n"
    "\n"
    "  this machine: Apple M4 Pro · 12.04 GB free of a 15.18 GB safe budget · 3 models ready · calibrated\n"
    "\n"
    "  NEW HERE?  first time on this machine, run these in order:\n"
    "     1) wmx-suite characterize <m>  measure a model's safe context ceiling\n"
    "     2) wmx-suite run --model <m>   launch it safely (auto KV-bits + context cap)\n"
    "\n"
    "  YOUR MACHINE\n"
    "  system                specs + memory budget (what your Mac can safely give a model)\n"
    "  health                live go/no-go: which models can run right now, and how big\n"
    "\n"
    "  MODELS\n"
    "  characterize <model>  safely measure a model's max safe context\n"
    "  list                  show measured ceilings stored for this machine\n"
    "  calibrate             tune this machine's cold-start memory estimate\n"
    "\n"
    "  RUN\n"
    "  run --model <model>   safe launch — refuses anything that would crash the Mac\n"
    "  web                   browser dashboard for all of the above\n"
    "\n"
    "  BENCHMARKS  (power users)\n"
    "  benchmark-kokoro      Kokoro TTS throughput (RTF / chars-per-sec) vs length\n"
    "  benchmark-kokoro-ttfa streaming time-to-first-audio latency\n"
    "  benchmark-kokoro-batchconcurrency vs throughput & peak memory\n"
    "  benchmark-kokoro-voicevoice-switch cost (warm vs cold load)\n"
    "  benchmark-kokoro-cachevoice-cache memory overhead\n"
    "  benchmark-kokoro-baselineidle synthesis memory floor\n"
    "  benchmark-embeddings  ModernBERT memory surface (batch x sequence)\n"
    "\n"
    "  details on any command:  wmx-suite <command> --help\n"
)


def _run(verbose: bool, color: bool = False) -> str:
    stream = io.StringIO()
    con = Console(color=color, verbose=verbose, stream=stream)
    render(con, FIXTURE)
    return stream.getvalue()


# ---------------------------------------------------------------------------
# Golden tests (plain, no color)
# ---------------------------------------------------------------------------

def test_landing_normal_golden():
    assert _run(verbose=False) == GOLDEN_NORMAL


def test_landing_verbose_golden():
    assert _run(verbose=True) == GOLDEN_VERBOSE


def test_landing_verbose_contains_benchmarks_section():
    out = _run(verbose=True)
    assert "BENCHMARKS" in out
    assert "benchmark-kokoro-ttfa" in out
    assert "benchmark-embeddings" in out


def test_landing_normal_benchmarks_oneliner():
    out = _run(verbose=False)
    assert "benchmark-* --help" in out
    assert "benchmark-kokoro-ttfa" not in out


def test_landing_normal_has_newhere_section():
    out = _run(verbose=False)
    assert "NEW HERE?" in out
    assert "1) wmx-suite characterize" in out


def test_landing_verbose_no_verbose_hint_in_footer():
    out = _run(verbose=True)
    assert "wmx-suite --verbose" not in out


def test_landing_normal_footer_has_verbose_hint():
    out = _run(verbose=False)
    assert "wmx-suite --verbose" in out


# ---------------------------------------------------------------------------
# Color smoke tests
# ---------------------------------------------------------------------------

def test_landing_color_mode_contains_ansi():
    out = _run(verbose=False, color=True)
    assert "\033[" in out


def test_landing_plain_mode_no_ansi():
    out = _run(verbose=False, color=False)
    assert "\033" not in out


def test_landing_color_verbose_contains_ansi():
    out = _run(verbose=True, color=True)
    assert "\033[" in out
