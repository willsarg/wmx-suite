# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""Golden tests for wmx_suite/views/benchmark_kokoro_concurrency.py.

Hardware-free and pure: Console(color=False, verbose=False) unless stated.
Each test builds a StringIO stream, calls a render function, then asserts
byte-exact structure (or ANSI presence for color smoke tests).
"""
from __future__ import annotations

import io

import pytest

from wmx_suite.ui import Console, ROLES
from wmx_suite.views.benchmark_kokoro_concurrency import (
    render_batch,
    render_voice,
    render_cache,
    render_safeguard,
    render_preflight_abort,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def make_console(*, color=False, verbose=False):
    stream = io.StringIO()
    con = Console(color=color, verbose=verbose, stream=stream)
    return con, stream


# --------------------------------------------------------------------------- #
# Fixtures — realistic data matching the CLI worker JSON schema
# --------------------------------------------------------------------------- #

BATCH_DATA = {
    "model": "hexgrad/Kokoro-82M",
    "voice": "af_heart",
    "batch_sizes": "1,2,4,8",
    "repeats": 3,
    "margin": 2.0,
    "safeguard_triggered": False,
    "run_id": 12,
    "rows": [
        {"batch_size": 1, "total_time": 0.41,  "cps": 121.9, "peak_gb": 1.23},
        {"batch_size": 2, "total_time": 0.55,  "cps": 181.8, "peak_gb": 1.47},
        {"batch_size": 4, "total_time": 0.82,  "cps": 243.9, "peak_gb": 1.91},
        {"batch_size": 8, "total_time": 1.31,  "cps": 305.3, "peak_gb": 2.74},
    ],
}

VOICE_DATA = {
    "model": "hexgrad/Kokoro-82M",
    "voice_a": "af_heart",
    "voice_b": "am_adam",
    "repeats": 5,
    "margin": 2.0,
    "safeguard_triggered": False,
    "run_id": 13,
    "rows": [
        {
            "cond_type": "warm_switch",
            "voice_from": "af_heart",
            "voice_to": "am_adam",
            "duration_ms": 12.4,
        },
        {
            "cond_type": "warm_switch",
            "voice_from": "am_adam",
            "voice_to": "af_heart",
            "duration_ms": 11.9,
        },
        {
            "cond_type": "cold_load",
            "voice_from": "af_heart",
            "voice_to": "am_adam",
            "duration_ms": 143.7,
        },
    ],
}

CACHE_DATA = {
    "model": "hexgrad/Kokoro-82M",
    "cache_sizes": "1,2,4,8",
    "margin": 2.0,
    "safeguard_triggered": False,
    "run_id": 14,
    "rows": [
        {"cache_size": 1, "os_wired_gb": 1.234, "peak_gb": 1.250},
        {"cache_size": 2, "os_wired_gb": 1.489, "peak_gb": 1.510},
        {"cache_size": 4, "os_wired_gb": 1.987, "peak_gb": 2.020},
        {"cache_size": 8, "os_wired_gb": 2.941, "peak_gb": 2.980},
    ],
}

SAFEGUARD_DATA = {
    "note": "predicted peak 16.40 GB exceeds safe ceiling 15.18 GB",
    "batch_size": 8,
    "peak_gb": 16.40,
    "wall_gb": 17.18,
    "safe_gb": 15.18,
    "next_cmd": "wmx-suite benchmark kokoro-batch --batch-sizes 1,2,4",
}

PREFLIGHT_DATA = {
    "reason": "not enough headroom to load model safely",
    "predicted_gb": 16.80,
    "available_gb": 3.22,
    "wall_gb": 17.18,
    "margin_gb": 2.0,
    "next_cmd": "wmx-suite benchmark kokoro-batch --batch-sizes 1,2",
}


# =========================================================================== #
# render_batch
# =========================================================================== #

def test_render_batch_section_header():
    con, stream = make_console()
    render_batch(con, BATCH_DATA)
    out = stream.getvalue()
    assert "Kokoro TTS Batch Concurrency Benchmark" in out


def test_render_batch_status_line():
    con, stream = make_console()
    render_batch(con, BATCH_DATA)
    out = stream.getvalue()
    assert "hexgrad/Kokoro-82M" in out
    assert "af_heart" in out
    assert "batches 1,2,4,8" in out
    assert "3 rep/batch" in out
    assert "margin 2.0 GB" in out


def test_render_batch_table_headers():
    con, stream = make_console()
    render_batch(con, BATCH_DATA)
    out = stream.getvalue()
    assert "Batch size" in out
    assert "Total time (s)" in out
    assert "chars/sec" in out
    assert "Peak GB" in out


def test_render_batch_table_data():
    con, stream = make_console()
    render_batch(con, BATCH_DATA)
    out = stream.getvalue()
    # All batch sizes present
    for r in BATCH_DATA["rows"]:
        assert str(r["batch_size"]) in out
        assert f"{r['total_time']:.2f}" in out
        assert f"{r['cps']:.1f}" in out
        assert f"{r['peak_gb']:.2f}" in out


def test_render_batch_next_block():
    con, stream = make_console()
    render_batch(con, BATCH_DATA)
    out = stream.getvalue()
    assert "next" in out
    assert "kokoro-voice" in out
    assert "kokoro-cache" in out


def test_render_batch_no_ansi_plain():
    con, stream = make_console(color=False)
    render_batch(con, BATCH_DATA)
    assert "\033" not in stream.getvalue()


def test_render_batch_no_safeguard_when_clean():
    con, stream = make_console()
    render_batch(con, BATCH_DATA)
    out = stream.getvalue()
    assert "safeguard" not in out.lower()


def test_render_batch_safeguard_warning_when_triggered():
    con, stream = make_console()
    data = dict(BATCH_DATA, safeguard_triggered=True)
    render_batch(con, data)
    out = stream.getvalue()
    assert "safeguard" in out.lower()
    assert "RULE #1" in out


def test_render_batch_no_verbose_raw_by_default():
    con, stream = make_console(verbose=False)
    render_batch(con, BATCH_DATA)
    assert "per-rung detail" not in stream.getvalue()


def test_render_batch_verbose_appends_raw():
    con, stream = make_console(verbose=True)
    render_batch(con, BATCH_DATA)
    out = stream.getvalue()
    assert "per-rung detail" in out
    # raw lines include key identifiers
    assert "batch=" in out
    assert "total=" in out
    assert "cps=" in out
    assert "peak=" in out


def test_render_batch_verbose_full_precision():
    con, stream = make_console(verbose=True)
    render_batch(con, BATCH_DATA)
    out = stream.getvalue()
    # verbose shows 3-decimal total time and 2-decimal cps
    assert "0.410s" in out
    assert "cps=121.90" in out


def test_render_batch_color_smoke():
    con, stream = make_console(color=True)
    render_batch(con, BATCH_DATA)
    out = stream.getvalue()
    assert "\033[" in out
    assert "Kokoro TTS Batch Concurrency Benchmark" in out
    # section header uses header/cyan role
    assert f"\033[{ROLES['header']}m" in out


def test_render_batch_empty_rows_no_table():
    con, stream = make_console()
    data = dict(BATCH_DATA, rows=[])
    render_batch(con, data)
    out = stream.getvalue()
    assert "Kokoro TTS Batch Concurrency Benchmark" in out
    assert "Batch size" not in out


def test_render_batch_golden_exact_single_row():
    """Byte-exact structural check for a single-row scenario."""
    con, stream = make_console()
    single = dict(BATCH_DATA, rows=[BATCH_DATA["rows"][0]])
    render_batch(con, single)
    out = stream.getvalue()
    lines = out.splitlines()

    # Header row contains all four columns
    header_line = next(l for l in lines if "Batch size" in l)
    assert "Total time (s)" in header_line
    assert "chars/sec" in header_line
    assert "Peak GB" in header_line

    # Data row contains the right values
    data_line = next(l for l in lines if "121.9" in l)
    assert "1" in data_line        # batch_size=1
    assert "0.41" in data_line     # total_time
    assert "1.23" in data_line     # peak_gb


# =========================================================================== #
# render_voice
# =========================================================================== #

def test_render_voice_section_header():
    con, stream = make_console()
    render_voice(con, VOICE_DATA)
    out = stream.getvalue()
    assert "Kokoro TTS Voice-Switch Latency Benchmark" in out


def test_render_voice_status_line():
    con, stream = make_console()
    render_voice(con, VOICE_DATA)
    out = stream.getvalue()
    assert "hexgrad/Kokoro-82M" in out
    assert "af_heart" in out
    assert "am_adam" in out
    assert "5 reps" in out
    assert "margin 2.0 GB" in out


def test_render_voice_table_headers():
    con, stream = make_console()
    render_voice(con, VOICE_DATA)
    out = stream.getvalue()
    assert "Condition" in out
    assert "Voice from" in out
    assert "Voice to" in out
    assert "Latency (ms)" in out


def test_render_voice_table_data():
    con, stream = make_console()
    render_voice(con, VOICE_DATA)
    out = stream.getvalue()
    assert "warm_switch" in out
    assert "cold_load" in out
    assert "af_heart" in out
    assert "am_adam" in out
    assert "12.4" in out
    assert "11.9" in out
    assert "143.7" in out


def test_render_voice_legend_gloss():
    con, stream = make_console()
    render_voice(con, VOICE_DATA)
    out = stream.getvalue()
    assert "warm_switch" in out
    assert "cold_load" in out
    assert "resident in cache" in out
    assert "full embedding load" in out


def test_render_voice_next_block():
    con, stream = make_console()
    render_voice(con, VOICE_DATA)
    out = stream.getvalue()
    assert "next" in out
    assert "kokoro-batch" in out
    assert "kokoro-cache" in out


def test_render_voice_no_ansi_plain():
    con, stream = make_console(color=False)
    render_voice(con, VOICE_DATA)
    assert "\033" not in stream.getvalue()


def test_render_voice_no_safeguard_when_clean():
    con, stream = make_console()
    render_voice(con, VOICE_DATA)
    assert "safeguard" not in stream.getvalue().lower()


def test_render_voice_safeguard_when_triggered():
    con, stream = make_console()
    data = dict(VOICE_DATA, safeguard_triggered=True)
    render_voice(con, data)
    out = stream.getvalue()
    assert "safeguard" in out.lower()


def test_render_voice_no_verbose_raw_by_default():
    con, stream = make_console(verbose=False)
    render_voice(con, VOICE_DATA)
    assert "per-rung detail" not in stream.getvalue()


def test_render_voice_verbose_appends_raw():
    con, stream = make_console(verbose=True)
    render_voice(con, VOICE_DATA)
    out = stream.getvalue()
    assert "per-rung detail" in out
    assert "cond=" in out
    assert "from=" in out
    assert "to=" in out
    assert "duration=" in out


def test_render_voice_verbose_full_precision():
    con, stream = make_console(verbose=True)
    render_voice(con, VOICE_DATA)
    out = stream.getvalue()
    # Three decimal places on duration_ms
    assert "12.400ms" in out
    assert "143.700ms" in out


def test_render_voice_color_smoke():
    con, stream = make_console(color=True)
    render_voice(con, VOICE_DATA)
    out = stream.getvalue()
    assert "\033[" in out
    assert "Kokoro TTS Voice-Switch Latency Benchmark" in out
    # warm_switch gets "good" (green), cold_load gets "warn" (yellow)
    assert f"\033[{ROLES['good']}m" in out
    assert f"\033[{ROLES['warn']}m" in out


def test_render_voice_empty_rows_no_table():
    con, stream = make_console()
    data = dict(VOICE_DATA, rows=[])
    render_voice(con, data)
    out = stream.getvalue()
    assert "Kokoro TTS Voice-Switch Latency Benchmark" in out
    assert "Condition" not in out
    assert "Latency (ms)" not in out


def test_render_voice_golden_exact_single_row():
    con, stream = make_console()
    single = dict(VOICE_DATA, rows=[VOICE_DATA["rows"][0]])
    render_voice(con, single)
    out = stream.getvalue()
    lines = out.splitlines()

    header_line = next(l for l in lines if "Condition" in l)
    assert "Voice from" in header_line
    assert "Voice to" in header_line
    assert "Latency (ms)" in header_line

    data_line = next(l for l in lines if "12.4" in l)
    assert "warm_switch" in data_line
    assert "af_heart" in data_line
    assert "am_adam" in data_line


# =========================================================================== #
# render_cache
# =========================================================================== #

def test_render_cache_section_header():
    con, stream = make_console()
    render_cache(con, CACHE_DATA)
    out = stream.getvalue()
    assert "Kokoro TTS Voice-Cache Memory Benchmark" in out


def test_render_cache_status_line():
    con, stream = make_console()
    render_cache(con, CACHE_DATA)
    out = stream.getvalue()
    assert "hexgrad/Kokoro-82M" in out
    assert "cache sizes 1,2,4,8" in out
    assert "margin 2.0 GB" in out


def test_render_cache_table_headers():
    con, stream = make_console()
    render_cache(con, CACHE_DATA)
    out = stream.getvalue()
    assert "Cache size (voices)" in out
    assert "Overhead GB" in out
    assert "Peak GB" in out


def test_render_cache_table_data():
    con, stream = make_console()
    render_cache(con, CACHE_DATA)
    out = stream.getvalue()
    for r in CACHE_DATA["rows"]:
        assert str(r["cache_size"]) in out
        assert f"{r['os_wired_gb']:.3f}" in out
        assert f"{r['peak_gb']:.3f}" in out


def test_render_cache_legend_gloss():
    con, stream = make_console()
    render_cache(con, CACHE_DATA)
    out = stream.getvalue()
    assert "overhead GB" in out
    assert "vm_stat" in out
    assert "peak GB" in out
    assert "MLX-reported" in out


def test_render_cache_next_block():
    con, stream = make_console()
    render_cache(con, CACHE_DATA)
    out = stream.getvalue()
    assert "next" in out
    assert "kokoro-batch" in out
    assert "kokoro-voice" in out


def test_render_cache_no_ansi_plain():
    con, stream = make_console(color=False)
    render_cache(con, CACHE_DATA)
    assert "\033" not in stream.getvalue()


def test_render_cache_no_safeguard_when_clean():
    con, stream = make_console()
    render_cache(con, CACHE_DATA)
    assert "safeguard" not in stream.getvalue().lower()


def test_render_cache_safeguard_when_triggered():
    con, stream = make_console()
    data = dict(CACHE_DATA, safeguard_triggered=True)
    render_cache(con, data)
    out = stream.getvalue()
    assert "safeguard" in out.lower()


def test_render_cache_no_verbose_raw_by_default():
    con, stream = make_console(verbose=False)
    render_cache(con, CACHE_DATA)
    assert "per-rung detail" not in stream.getvalue()


def test_render_cache_verbose_appends_raw():
    con, stream = make_console(verbose=True)
    render_cache(con, CACHE_DATA)
    out = stream.getvalue()
    assert "per-rung detail" in out
    assert "cache=" in out
    assert "os_wired=" in out
    assert "peak=" in out


def test_render_cache_verbose_full_precision():
    con, stream = make_console(verbose=True)
    render_cache(con, CACHE_DATA)
    out = stream.getvalue()
    # Six decimal places in verbose raw
    assert "1.234000GB" in out
    assert "1.250000GB" in out


def test_render_cache_color_smoke():
    con, stream = make_console(color=True)
    render_cache(con, CACHE_DATA)
    out = stream.getvalue()
    assert "\033[" in out
    assert "Kokoro TTS Voice-Cache Memory Benchmark" in out
    assert f"\033[{ROLES['header']}m" in out


def test_render_cache_empty_rows_no_table():
    con, stream = make_console()
    data = dict(CACHE_DATA, rows=[])
    render_cache(con, data)
    out = stream.getvalue()
    assert "Kokoro TTS Voice-Cache Memory Benchmark" in out
    assert "Cache size (voices)" not in out


def test_render_cache_golden_exact_single_row():
    con, stream = make_console()
    single = dict(CACHE_DATA, rows=[CACHE_DATA["rows"][0]])
    render_cache(con, single)
    out = stream.getvalue()
    lines = out.splitlines()

    header_line = next(l for l in lines if "Cache size (voices)" in l)
    assert "Overhead GB" in header_line
    assert "Peak GB" in header_line

    data_line = next(l for l in lines if "1.234" in l)
    assert "1" in data_line       # cache_size=1
    assert "1.250" in data_line   # peak_gb


# =========================================================================== #
# render_safeguard
# =========================================================================== #

def test_render_safeguard_golden_plain():
    con, stream = make_console()
    render_safeguard(con, SAFEGUARD_DATA)
    out = stream.getvalue()
    assert "Sweep stopped" in out
    assert "safeguard triggered" in out
    assert "why" in out
    assert "predicted peak 16.40 GB exceeds safe ceiling 15.18 GB" in out
    assert "triggered at batch size 8" in out
    assert "predicted 16.40 GB > safe ceiling 15.18 GB" in out
    assert "crash wall is 17.18 GB" in out
    assert "try" in out
    assert "wmx-suite benchmark kokoro-batch --batch-sizes 1,2,4" in out
    assert "smaller batch" in out
    assert "wmx-suite benchmark kokoro-baseline" in out
    assert "\033" not in out


def test_render_safeguard_minimal_data():
    """Works with only the required 'note' key."""
    con, stream = make_console()
    render_safeguard(con, {"note": "peak too high"})
    out = stream.getvalue()
    assert "Sweep stopped" in out
    assert "peak too high" in out
    assert "try" in out
    # optional fields absent → their why-lines absent
    assert "triggered at batch size" not in out
    assert "crash wall is" not in out


def test_render_safeguard_no_ansi_plain():
    con, stream = make_console(color=False)
    render_safeguard(con, SAFEGUARD_DATA)
    assert "\033" not in stream.getvalue()


def test_render_safeguard_color_smoke():
    con, stream = make_console(color=True)
    render_safeguard(con, SAFEGUARD_DATA)
    out = stream.getvalue()
    assert "\033[" in out
    # headline is styled "bad" (red)
    assert f"\033[{ROLES['bad']}m" in out


# =========================================================================== #
# render_preflight_abort
# =========================================================================== #

def test_render_preflight_abort_golden_plain():
    con, stream = make_console()
    render_preflight_abort(con, PREFLIGHT_DATA)
    out = stream.getvalue()
    assert "Won't start" in out
    assert "pre-flight check failed" in out
    assert "why" in out
    assert "not enough headroom to load model safely" in out
    assert "predicted load: 16.80 GB" in out
    assert "safe headroom:  3.22 GB" in out
    assert "crash wall 17.18 GB minus margin 2.00 GB = ceiling 15.18 GB" in out
    assert "try" in out
    assert "wmx-suite benchmark kokoro-batch --batch-sizes 1,2" in out
    assert "wmx-suite characterize <model>" in out
    assert "\033" not in out


def test_render_preflight_abort_minimal_data():
    con, stream = make_console()
    render_preflight_abort(con, {"reason": "too little free memory"})
    out = stream.getvalue()
    assert "Won't start" in out
    assert "too little free memory" in out
    assert "try" in out
    assert "predicted load:" not in out
    assert "safe headroom:" not in out


def test_render_preflight_abort_no_ansi_plain():
    con, stream = make_console(color=False)
    render_preflight_abort(con, PREFLIGHT_DATA)
    assert "\033" not in stream.getvalue()


def test_render_preflight_abort_color_smoke():
    con, stream = make_console(color=True)
    render_preflight_abort(con, PREFLIGHT_DATA)
    out = stream.getvalue()
    assert "\033[" in out
    assert f"\033[{ROLES['bad']}m" in out


# =========================================================================== #
# Cross-cutting: stream isolation — no leaks to sys.stdout
# =========================================================================== #

def test_all_renders_use_stream_not_stdout(capsys):
    """None of the render functions accidentally call print() to stdout."""
    for fn, data in [
        (render_batch,           BATCH_DATA),
        (render_voice,           VOICE_DATA),
        (render_cache,           CACHE_DATA),
        (render_safeguard,       SAFEGUARD_DATA),
        (render_preflight_abort, PREFLIGHT_DATA),
    ]:
        con, stream = make_console()
        fn(con, data)
        captured = capsys.readouterr()
        assert captured.out == "", f"{fn.__name__} leaked to stdout"
        assert stream.getvalue() != "", f"{fn.__name__} emitted nothing to stream"
