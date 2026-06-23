# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""Tests for wmx_suite/views/system.py.

Hardware-free. Builds a Console with a StringIO stream, calls render, and
checks the output byte-for-byte against a golden string captured from the M4 Pro
testbed numbers used in the approved mockup.
"""
from __future__ import annotations

import io

import pytest

from wmx_suite.ui import Console
from wmx_suite.views.system import render

# ---------------------------------------------------------------------------
# Shared fixture data (M4 Pro testbed numbers from mockup)
# ---------------------------------------------------------------------------
FIXTURE = {
    "device": "Apple M4 Pro",
    "total_gb": 25.77,
    "wall_gb": 17.18,
    "safe_budget_gb": 15.18,
    "wired_gb": 3.14,
    "free_headroom_gb": 12.04,
    "max_buffer_gb": 12.88,
    "swap_free_gb": 1.13,
    "swap_warn": True,
    "margin_gb": 2.0,
    "margin_source": "default (set WMX_SUITE_MARGIN_GB or --margin)",
    "wall_bytes": 18446744073,
    "calibrated": True,
    "cal_model_short": "Qwen2.5-VL-7B-Instruct-4bit",
    "cal_overhead_gb": 1.0,
    "cal_date": "2026-06-14",
    "wired_sample": "median of 3 @ 0.2s",
}

GOLDEN_NORMAL = (
    "device          : Apple M4 Pro\n"
    "\n"
    "memory budget\n"
    "total RAM       : 25.77 GB   physical memory installed\n"
    "crash wall      : 17.18 GB   most the GPU can safely lock — past this the Mac can hard-lock\n"
    "  safe budget   : 15.18 GB   crash wall − 2 GB margin — the suite never plans above this\n"
    "wired now       : 3.14 GB   memory locked right now (your starting point)\n"
    "free headroom   : 12.04 GB   safe budget − wired now → what a model may use\n"
    "\n"
    "details\n"
    "max GPU buffer  : 12.88 GB   largest single allocation Metal allows\n"
    "swap free       : 1.13 GB  ⚠ tiny   almost no fallback if you cross the wall\n"
    "calibration     : tuned   this machine's cold-start overhead is measured"
    " (Qwen2.5-VL-7B-Instruct-4bit, 2026-06-14)\n"
    "\n"
    "next\n"
    "  wmx-suite health      which of your models can run right now, and how big a context\n"
    "  wmx-suite list        safe context ceilings already measured on this machine\n"
    "  wmx-suite calibrate   re-tune this machine's cold-start estimate\n"
)

GOLDEN_VERBOSE = (
    "device          : Apple M4 Pro\n"
    "\n"
    "memory budget\n"
    "total RAM       : 25.77 GB   physical memory installed\n"
    "crash wall      : 17.18 GB   most the GPU can safely lock — past this the Mac can hard-lock\n"
    "  safe budget   : 15.18 GB   crash wall − 2 GB margin — the suite never plans above this\n"
    "wired now       : 3.14 GB   memory locked right now (your starting point)\n"
    "free headroom   : 12.04 GB   safe budget − wired now → what a model may use\n"
    "\n"
    "details\n"
    "max GPU buffer  : 12.88 GB   largest single allocation Metal allows\n"
    "swap free       : 1.13 GB  ⚠ tiny   almost no fallback if you cross the wall\n"
    "calibration     : tuned   this machine's cold-start overhead is measured"
    " (Qwen2.5-VL-7B-Instruct-4bit, 2026-06-14)\n"
    "\n"
    "raw (--verbose)\n"
    "  wall source   : max_recommended_working_set_size\n"
    "  wall bytes    : 18,446,744,073 B\n"
    "  margin        : 2.00 GB   default (set WMX_SUITE_MARGIN_GB or --margin)\n"
    "  cold-start    : overhead 1.00 GB   Qwen2.5-VL-7B-Instruct-4bit @ 2026-06-14T00:00:00Z\n"
    "  wired sample  : median of 3 @ 0.2s   vm_stat 'Pages wired down'\n"
    "\n"
    "next\n"
    "  wmx-suite health      which of your models can run right now, and how big a context\n"
    "  wmx-suite list        safe context ceilings already measured on this machine\n"
    "  wmx-suite calibrate   re-tune this machine's cold-start estimate\n"
)


def _run(verbose: bool, color: bool = False) -> str:
    stream = io.StringIO()
    con = Console(color=color, verbose=verbose, stream=stream)
    render(con, FIXTURE)
    return stream.getvalue()


# ---------------------------------------------------------------------------
# Golden tests (plain, no color)
# ---------------------------------------------------------------------------

def test_system_normal_golden():
    assert _run(verbose=False) == GOLDEN_NORMAL


def test_system_verbose_golden():
    assert _run(verbose=True) == GOLDEN_VERBOSE


def test_system_verbose_contains_raw_appendix():
    out = _run(verbose=True)
    assert "raw (--verbose)" in out
    assert "wall source" in out
    assert "cold-start" in out


def test_system_normal_no_raw_appendix():
    out = _run(verbose=False)
    assert "raw (--verbose)" not in out


# ---------------------------------------------------------------------------
# Color smoke tests
# ---------------------------------------------------------------------------

def test_system_color_mode_contains_ansi():
    out = _run(verbose=False, color=True)
    assert "\033[" in out


def test_system_plain_mode_no_ansi():
    out = _run(verbose=False, color=False)
    assert "\033" not in out


def test_system_color_verbose_contains_ansi():
    out = _run(verbose=True, color=True)
    assert "\033[" in out
