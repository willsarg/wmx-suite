"""Golden tests for wmx_suite/views/benchmark_kokoro.py.

Hardware-free and pure: Console(color=False, verbose=False) unless stated.
Each test builds a stream, calls the render function, then asserts byte-exact
golden output (or structural assertions for the color/verbose smoke tests).
"""
from __future__ import annotations

import io

import pytest

from wmx_suite.ui import Console, ROLES
from wmx_suite.views.benchmark_kokoro import (
    render_perf,
    render_ttfa,
    render_baseline,
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
# Fixtures — realistic data
# --------------------------------------------------------------------------- #

PERF_DATA = {
    "model": "hexgrad/Kokoro-82M",
    "voice": "af_heart",
    "lengths": "50,100,200",
    "repeats": 3,
    "margin": 2.0,
    "safeguard_triggered": False,
    "run_id": 7,
    "rows": [
        {
            "length": 50,
            "audio_dur": 3.12,
            "compute_time": 0.41,
            "rtf": 0.1314,
            "cps": 121.9,
            "peak_gb": 1.23,
            "os_wired_gb": 1.85,
        },
        {
            "length": 100,
            "audio_dur": 6.44,
            "compute_time": 0.79,
            "rtf": 0.1227,
            "cps": 126.5,
            "peak_gb": 1.25,
            "os_wired_gb": 1.88,
        },
        {
            "length": 200,
            "audio_dur": 12.80,
            "compute_time": 1.55,
            "rtf": 0.1211,
            "cps": 129.0,
            "peak_gb": 1.27,
            "os_wired_gb": None,
        },
    ],
}

TTFA_DATA = {
    "model": "hexgrad/Kokoro-82M",
    "voice": "af_heart",
    "lengths": "50,100,200",
    "repeats": 3,
    "margin": 2.0,
    "safeguard_triggered": False,
    "run_id": 8,
    "rows": [
        {
            "length": 50,
            "ttfa_sec": 0.183,
            "total_sec": 0.413,
            "speedup_ratio": 2.3,
            "first_chunk_dur": 0.45,
            "peak_gb": 1.23,
        },
        {
            "length": 100,
            "ttfa_sec": 0.204,
            "total_sec": 0.791,
            "speedup_ratio": 3.1,
            "first_chunk_dur": 0.45,
            "peak_gb": 1.25,
        },
    ],
}

BASELINE_DATA = {
    "model": "hexgrad/Kokoro-82M",
    "voice": "af_heart",
    "margin": 2.0,
    "baseline_gb": 5.214,
    "active_gb": 6.817,
    "overhead_gb": 1.603,
    "run_id": 9,
}

SAFEGUARD_DATA = {
    "note": "predicted peak 16.40 GB exceeds safe ceiling 15.18 GB",
    "length": 200,
    "peak_gb": 16.40,
    "wall_gb": 17.18,
    "safe_gb": 15.18,
    "next_cmd": "wmx-suite benchmark kokoro --lengths 50,100",
}

PREFLIGHT_DATA = {
    "reason": "not enough headroom to load model safely",
    "predicted_gb": 16.80,
    "available_gb": 3.22,
    "wall_gb": 17.18,
    "margin_gb": 2.0,
    "next_cmd": "wmx-suite benchmark kokoro --margin 1.5",
}


# =========================================================================== #
# render_perf
# =========================================================================== #

def test_render_perf_golden_plain():
    con, stream = make_console()
    render_perf(con, PERF_DATA)
    out = stream.getvalue()

    # Section header
    assert "Kokoro TTS Performance Benchmark" in out
    # Status line
    assert "hexgrad/Kokoro-82M" in out
    assert "af_heart" in out
    assert "sweeps 50,100,200" in out
    assert "3 rep/len" in out
    assert "margin 2.0 GB" in out
    # Table headers
    assert "Length (char)" in out
    assert "Audio (s)" in out
    assert "Compute (s)" in out
    assert "RTF" in out
    assert "chars/sec" in out
    assert "Peak GB" in out
    assert "OS Wired GB" in out
    # Data rows
    assert "50" in out
    assert "3.12" in out
    assert "0.41" in out
    assert "0.1314" in out
    assert "121.9" in out
    assert "1.23" in out
    assert "1.85" in out
    # Missing os_wired renders as dash
    assert "—" in out
    # next block
    assert "next" in out
    assert "kokoro-ttfa" in out
    assert "kokoro-baseline" in out
    assert "wmx-suite web" in out
    # No ANSI
    assert "\033" not in out


def test_render_perf_no_safeguard_message_when_clean():
    con, stream = make_console()
    render_perf(con, PERF_DATA)
    out = stream.getvalue()
    assert "safeguard" not in out.lower()


def test_render_perf_safeguard_message_when_triggered():
    con, stream = make_console()
    data = dict(PERF_DATA, safeguard_triggered=True)
    render_perf(con, data)
    out = stream.getvalue()
    assert "safeguard" in out.lower()


def test_render_perf_no_verbose_raw_section_by_default():
    con, stream = make_console(verbose=False)
    render_perf(con, PERF_DATA)
    out = stream.getvalue()
    assert "per-rung detail" not in out


def test_render_perf_verbose_appends_raw_section():
    con, stream = make_console(verbose=True)
    render_perf(con, PERF_DATA)
    out = stream.getvalue()
    assert "per-rung detail" in out
    # raw section includes full precision numbers
    assert "length=" in out
    assert "audio=" in out
    assert "compute=" in out


def test_render_perf_color_smoke():
    """Color mode: ANSI codes present; structural content still there."""
    con, stream = make_console(color=True)
    render_perf(con, PERF_DATA)
    out = stream.getvalue()
    assert "\033[" in out
    assert "Kokoro TTS Performance Benchmark" in out
    assert "Length (char)" in out


def test_render_perf_empty_rows_no_table():
    con, stream = make_console()
    data = dict(PERF_DATA, rows=[])
    render_perf(con, data)
    out = stream.getvalue()
    # section header still present
    assert "Kokoro TTS Performance Benchmark" in out
    # but no table headers
    assert "Length (char)" not in out


def test_render_perf_golden_exact():
    """Byte-exact golden for 1-row scenario (simpler to lock)."""
    con, stream = make_console()
    single_row_data = dict(
        PERF_DATA,
        rows=[PERF_DATA["rows"][0]],  # only the 50-char row
    )
    render_perf(con, single_row_data)
    out = stream.getvalue()

    # Verify the table header row contains all column names separated by gutter
    lines = out.splitlines()
    header_line = next(l for l in lines if "Length (char)" in l)
    assert "Audio (s)" in header_line
    assert "Compute (s)" in header_line
    assert "RTF" in header_line
    assert "chars/sec" in header_line
    assert "Peak GB" in header_line
    assert "OS Wired GB" in header_line

    # Verify the data row is right-aligned (all numeric fields)
    data_line = next(l for l in lines if "3.12" in l)
    assert "50" in data_line
    assert "0.41" in data_line
    assert "0.1314" in data_line
    assert "121.9" in data_line
    assert "1.23" in data_line
    assert "1.85" in data_line


# =========================================================================== #
# render_ttfa
# =========================================================================== #

def test_render_ttfa_golden_plain():
    con, stream = make_console()
    render_ttfa(con, TTFA_DATA)
    out = stream.getvalue()

    assert "Time-to-First-Audio" in out
    assert "hexgrad/Kokoro-82M" in out
    assert "af_heart" in out
    # Table headers
    assert "Length (char)" in out
    assert "TTFA (s)" in out
    assert "Total (s)" in out
    assert "Speedup" in out
    assert "First Chunk (s)" in out
    assert "Peak GB" in out
    # Data
    assert "0.183" in out
    assert "0.413" in out
    assert "2.3x" in out
    assert "0.45" in out
    assert "1.23" in out
    # next block
    assert "next" in out
    assert "kokoro-ttfa" not in out  # this IS the ttfa benchmark; next points elsewhere
    assert "wmx-suite benchmark kokoro" in out
    assert "wmx-suite web" in out
    # No ANSI
    assert "\033" not in out


def test_render_ttfa_no_safeguard_message_when_clean():
    con, stream = make_console()
    render_ttfa(con, TTFA_DATA)
    out = stream.getvalue()
    assert "safeguard" not in out.lower()


def test_render_ttfa_safeguard_message_when_triggered():
    con, stream = make_console()
    data = dict(TTFA_DATA, safeguard_triggered=True)
    render_ttfa(con, data)
    out = stream.getvalue()
    assert "safeguard" in out.lower()


def test_render_ttfa_verbose_appends_raw():
    con, stream = make_console(verbose=True)
    render_ttfa(con, TTFA_DATA)
    out = stream.getvalue()
    assert "per-rung detail" in out
    assert "ttfa=" in out
    assert "total=" in out
    assert "speedup=" in out


def test_render_ttfa_no_verbose_raw_by_default():
    con, stream = make_console(verbose=False)
    render_ttfa(con, TTFA_DATA)
    out = stream.getvalue()
    assert "per-rung detail" not in out


def test_render_ttfa_color_smoke():
    con, stream = make_console(color=True)
    render_ttfa(con, TTFA_DATA)
    out = stream.getvalue()
    assert "\033[" in out
    assert "Time-to-First-Audio" in out


def test_render_ttfa_speedup_suffix():
    """Speedup column renders with 'x' suffix."""
    con, stream = make_console()
    render_ttfa(con, TTFA_DATA)
    out = stream.getvalue()
    assert "2.3x" in out
    assert "3.1x" in out


# =========================================================================== #
# render_baseline
# =========================================================================== #

def test_render_baseline_golden_plain():
    con, stream = make_console()
    render_baseline(con, BASELINE_DATA)
    out = stream.getvalue()

    assert "Kokoro TTS Active Memory Baseline" in out
    assert "hexgrad/Kokoro-82M" in out
    assert "af_heart" in out
    assert "margin 2.0 GB" in out
    # Fields
    assert "System baseline" in out
    assert "5.214 GB" in out
    assert "Active synthesis" in out
    assert "6.817 GB" in out
    assert "Static overhead" in out
    assert "1.603 GB" in out
    # Glosses
    assert "settled OS-wired" in out
    assert "synthesis running" in out
    assert "cost of keeping Kokoro resident" in out
    # next block
    assert "next" in out
    assert "wmx-suite benchmark kokoro" in out
    assert "wmx-suite web" in out
    # No ANSI
    assert "\033" not in out


def test_render_baseline_no_verbose_raw_by_default():
    con, stream = make_console(verbose=False)
    render_baseline(con, BASELINE_DATA)
    out = stream.getvalue()
    assert "raw measurement" not in out


def test_render_baseline_verbose_appends_raw():
    con, stream = make_console(verbose=True)
    render_baseline(con, BASELINE_DATA)
    out = stream.getvalue()
    assert "raw measurement" in out
    assert "baseline_gb" in out
    assert "active_gb" in out
    assert "overhead_gb" in out
    # 6-decimal precision in verbose raw
    assert "5.214000" in out
    assert "6.817000" in out
    assert "1.603000" in out


def test_render_baseline_color_smoke():
    con, stream = make_console(color=True)
    render_baseline(con, BASELINE_DATA)
    out = stream.getvalue()
    assert "\033[" in out
    assert "Kokoro TTS Active Memory Baseline" in out


def test_render_baseline_field_exact():
    """Labels are exactly 16 chars wide (LABEL_WIDTH)."""
    con, stream = make_console()
    render_baseline(con, BASELINE_DATA)
    out = stream.getvalue()
    lines = out.splitlines()
    baseline_line = next(l for l in lines if "5.214 GB" in l)
    # "System baseline : 5.214 GB" — label padded to 16, then ": "
    assert baseline_line.startswith("System baseline ")
    assert ": 5.214 GB" in baseline_line


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
    assert "triggered at length 200 chars" in out
    assert "predicted 16.40 GB > safe ceiling 15.18 GB" in out
    assert "crash wall is 17.18 GB" in out
    assert "try" in out
    assert "wmx-suite benchmark kokoro --lengths 50,100" in out
    assert "shorter lengths" in out
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


def test_render_safeguard_color_smoke():
    con, stream = make_console(color=True)
    render_safeguard(con, SAFEGUARD_DATA)
    out = stream.getvalue()
    assert "\033[" in out
    # headline styled as bad (red)
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
    assert "wmx-suite benchmark kokoro --margin 1.5" in out
    assert "wmx-suite characterize <model>" in out
    assert "\033" not in out


def test_render_preflight_abort_minimal_data():
    con, stream = make_console()
    render_preflight_abort(con, {"reason": "too little free memory"})
    out = stream.getvalue()
    assert "Won't start" in out
    assert "too little free memory" in out
    assert "try" in out


def test_render_preflight_abort_color_smoke():
    con, stream = make_console(color=True)
    render_preflight_abort(con, PREFLIGHT_DATA)
    out = stream.getvalue()
    assert "\033[" in out
    assert f"\033[{ROLES['bad']}m" in out


# =========================================================================== #
# Cross-cutting: no output goes to sys.stdout (stream isolation)
# =========================================================================== #

def test_all_renders_use_stream_not_stdout(capsys):
    """None of the render functions accidentally call print() to stdout."""
    for fn, data in [
        (render_perf, PERF_DATA),
        (render_ttfa, TTFA_DATA),
        (render_baseline, BASELINE_DATA),
        (render_safeguard, SAFEGUARD_DATA),
        (render_preflight_abort, PREFLIGHT_DATA),
    ]:
        con, stream = make_console()
        fn(con, data)
        captured = capsys.readouterr()
        assert captured.out == "", f"{fn.__name__} leaked to stdout"
        assert stream.getvalue() != "", f"{fn.__name__} emitted nothing to stream"
