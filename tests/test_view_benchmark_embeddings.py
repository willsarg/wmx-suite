"""Tests for wmx_suite/views/benchmark_embeddings.py.

Three pure render functions:
    render_surface(console, data)     — batch×seq surface table + profile note + next_block
    render_profile_note(console, data) — calibration-profile status line
    render_safeguard(console, data)   — pre-flight abort / surface-wide refusal guidance

All tests are hardware-free: no model loads, no DB, no MLX imports.
Color tests assert ANSI codes are present; plain tests assert stable plain text.
"""
from __future__ import annotations

import io

import pytest

from wmx_suite.ui import Console
from wmx_suite.views.benchmark_embeddings import (
    render_profile_note,
    render_safeguard,
    render_surface,
)


# ---------------------------------------------------------------------------
# Fixtures — realistic data
# ---------------------------------------------------------------------------

def _make_console(*, verbose: bool = False, color: bool = False) -> tuple[Console, io.StringIO]:
    stream = io.StringIO()
    con = Console(color=color, verbose=verbose, stream=stream)
    return con, stream


# Batch sizes 1, 2, 4, 8; seq lengths 128, 512, 2048, 8192.
# Some cells skipped (predicted over budget), one cell measured at high seq.
SURFACE_DATA = {
    "model": "mlx-community/ModernBERT-base-4bit",
    "margin_gb": 2.0,
    "profile_source": "loaded",  # "loaded" | "cold" | "ignored"
    "mlx_version": "0.22.1",
    "batches": [1, 2, 4, 8],
    "seqs": [128, 512, 2048, 8192],
    # cells: keyed (batch, seq)
    "cells": {
        (1, 128):  {"status": "measured", "throughput_tps": 4210.5, "latency_ms": 30.4, "peak_gb": 0.41, "os_wired_gb": 3.82, "predicted_gb": 0.38},
        (1, 512):  {"status": "measured", "throughput_tps": 3890.2, "latency_ms": 131.5, "peak_gb": 0.55, "os_wired_gb": 3.94, "predicted_gb": 0.52},
        (1, 2048): {"status": "measured", "throughput_tps": 2100.0, "latency_ms": 975.2, "peak_gb": 1.10, "os_wired_gb": 4.48, "predicted_gb": 1.05},
        (1, 8192): {"status": "skipped",  "throughput_tps": None, "latency_ms": None, "peak_gb": None, "os_wired_gb": None, "predicted_gb": 3.80},
        (2, 128):  {"status": "measured", "throughput_tps": 7950.1, "latency_ms": 32.2, "peak_gb": 0.62, "os_wired_gb": 4.01, "predicted_gb": 0.59},
        (2, 512):  {"status": "measured", "throughput_tps": 6810.8, "latency_ms": 150.3, "peak_gb": 0.90, "os_wired_gb": 4.29, "predicted_gb": 0.87},
        (2, 2048): {"status": "measured", "throughput_tps": 3200.0, "latency_ms": 1280.0, "peak_gb": 1.80, "os_wired_gb": 5.18, "predicted_gb": 1.75},
        (2, 8192): {"status": "skipped",  "throughput_tps": None, "latency_ms": None, "peak_gb": None, "os_wired_gb": None, "predicted_gb": 6.90},
        (4, 128):  {"status": "measured", "throughput_tps": 14200.0, "latency_ms": 36.0, "peak_gb": 1.02, "os_wired_gb": 4.40, "predicted_gb": 0.99},
        (4, 512):  {"status": "measured", "throughput_tps": 11000.5, "latency_ms": 186.4, "peak_gb": 1.55, "os_wired_gb": 4.93, "predicted_gb": 1.51},
        (4, 2048): {"status": "skipped",  "throughput_tps": None, "latency_ms": None, "peak_gb": None, "os_wired_gb": None, "predicted_gb": 3.40},
        (4, 8192): {"status": "skipped",  "throughput_tps": None, "latency_ms": None, "peak_gb": None, "os_wired_gb": None, "predicted_gb": 13.20},
        (8, 128):  {"status": "measured", "throughput_tps": 22100.0, "latency_ms": 46.0, "peak_gb": 1.88, "os_wired_gb": 5.26, "predicted_gb": 1.84},
        (8, 512):  {"status": "skipped",  "throughput_tps": None, "latency_ms": None, "peak_gb": None, "os_wired_gb": None, "predicted_gb": 2.90},
        (8, 2048): {"status": "skipped",  "throughput_tps": None, "latency_ms": None, "peak_gb": None, "os_wired_gb": None, "predicted_gb": 6.10},
        (8, 8192): {"status": "skipped",  "throughput_tps": None, "latency_ms": None, "peak_gb": None, "os_wired_gb": None, "predicted_gb": 24.80},
    },
    "n_cells_measured": 8,
    "n_cells_skipped": 8,
}

PROFILE_LOADED_DATA = {
    "profile_source": "loaded",
    "model": "mlx-community/ModernBERT-base-4bit",
    "mlx_version": "0.22.1",
}

PROFILE_COLD_DATA = {
    "profile_source": "cold",
    "model": "mlx-community/ModernBERT-base-4bit",
    "mlx_version": "0.22.1",
}

PROFILE_IGNORED_DATA = {
    "profile_source": "ignored",
    "model": "mlx-community/ModernBERT-base-4bit",
    "mlx_version": "0.22.1",
}

SAFEGUARD_DATA = {
    "model": "mlx-community/ModernBERT-base-4bit",
    "reason": "predicted peak 18.40 GB exceeds safe budget 15.18 GB",
    "predicted_gb": 18.40,
    "safe_budget_gb": 15.18,
    "margin_gb": 2.0,
}


# ===========================================================================
# render_profile_note
# ===========================================================================

class TestRenderProfileNote:
    def test_loaded_contains_loaded_keyword(self):
        con, stream = _make_console()
        render_profile_note(con, PROFILE_LOADED_DATA)
        out = stream.getvalue()
        assert "loaded" in out.lower()

    def test_cold_mentions_cold_start(self):
        con, stream = _make_console()
        render_profile_note(con, PROFILE_COLD_DATA)
        out = stream.getvalue()
        assert "cold" in out.lower()

    def test_ignored_mentions_ignore_profile(self):
        con, stream = _make_console()
        render_profile_note(con, PROFILE_IGNORED_DATA)
        out = stream.getvalue()
        # Must signal the profile was actively ignored
        assert "ignore" in out.lower() or "ignored" in out.lower()

    def test_plain_loaded_golden(self):
        con, stream = _make_console()
        render_profile_note(con, PROFILE_LOADED_DATA)
        out = stream.getvalue()
        assert "\033" not in out
        # Must have a field-label-style line
        assert "calibration" in out.lower()
        assert "loaded" in out.lower()

    def test_color_emits_ansi(self):
        con, stream = _make_console(color=True)
        render_profile_note(con, PROFILE_LOADED_DATA)
        out = stream.getvalue()
        assert "\033[" in out


# ===========================================================================
# render_safeguard
# ===========================================================================

class TestRenderSafeguard:
    def test_headline_contains_abort_or_refused(self):
        con, stream = _make_console()
        render_safeguard(con, SAFEGUARD_DATA)
        out = stream.getvalue()
        lower = out.lower()
        assert "abort" in lower or "refused" in lower or "won't" in lower or "over budget" in lower

    def test_reason_present(self):
        con, stream = _make_console()
        render_safeguard(con, SAFEGUARD_DATA)
        out = stream.getvalue()
        # The reason string or the numeric values it contains must appear
        assert "18.40" in out or "18.4" in out

    def test_safe_budget_shown(self):
        con, stream = _make_console()
        render_safeguard(con, SAFEGUARD_DATA)
        out = stream.getvalue()
        assert "15.18" in out

    def test_try_suggestions_present(self):
        con, stream = _make_console()
        render_safeguard(con, SAFEGUARD_DATA)
        out = stream.getvalue()
        # guidance() always emits a "try" section with at least one command
        assert "try" in out.lower() or "wmx-suite" in out

    def test_plain_no_ansi(self):
        con, stream = _make_console()
        render_safeguard(con, SAFEGUARD_DATA)
        out = stream.getvalue()
        assert "\033" not in out

    def test_color_emits_ansi(self):
        con, stream = _make_console(color=True)
        render_safeguard(con, SAFEGUARD_DATA)
        out = stream.getvalue()
        assert "\033[" in out


# ===========================================================================
# render_surface
# ===========================================================================

class TestRenderSurface:
    # --- structural checks ---

    def test_section_header_present(self):
        con, stream = _make_console()
        render_surface(con, SURFACE_DATA)
        out = stream.getvalue()
        # Must have a human-readable section header, not a raw JSON key
        lower = out.lower()
        assert "embedding" in lower or "memory" in lower or "throughput" in lower

    def test_status_line_contains_model(self):
        con, stream = _make_console()
        render_surface(con, SURFACE_DATA)
        out = stream.getvalue()
        assert "ModernBERT" in out

    def test_status_line_contains_margin(self):
        con, stream = _make_console()
        render_surface(con, SURFACE_DATA)
        out = stream.getvalue()
        assert "2.0" in out or "2.00" in out

    def test_status_line_contains_profile_source(self):
        con, stream = _make_console()
        render_surface(con, SURFACE_DATA)
        out = stream.getvalue()
        # "loaded" from SURFACE_DATA["profile_source"]
        assert "loaded" in out.lower()

    def test_table_has_seq_headers(self):
        """Column headers should show seq lengths."""
        con, stream = _make_console()
        render_surface(con, SURFACE_DATA)
        out = stream.getvalue()
        # All four seq lengths should appear
        assert "128" in out
        assert "512" in out
        assert "2048" in out
        assert "8192" in out

    def test_table_has_batch_rows(self):
        """Batch sizes should appear as row headers."""
        con, stream = _make_console()
        render_surface(con, SURFACE_DATA)
        out = stream.getvalue()
        assert "1" in out
        assert "2" in out
        assert "4" in out
        assert "8" in out

    def test_measured_cells_show_primary_metric(self):
        """Measured cells should show throughput (the default primary metric)."""
        con, stream = _make_console()
        render_surface(con, SURFACE_DATA)
        out = stream.getvalue()
        # batch=1, seq=128 → throughput_tps=4210.5 → rendered as "4,210 tok/s" (comma-sep)
        assert "4,210" in out or "4,211" in out

    def test_skipped_cells_show_skip_marker(self):
        """Skipped cells should use glyph("bad") or "skip" / "—" marker."""
        con, stream = _make_console()
        render_surface(con, SURFACE_DATA)
        out = stream.getvalue()
        # Must convey skip visually; "✗" or "skip" or "—" are all valid
        has_marker = "✗" in out or "skip" in out.lower() or "—" in out or "--" in out
        assert has_marker

    def test_skipped_cells_show_predicted_gb(self):
        """Skipped cells should show the predicted GB so the user knows why."""
        con, stream = _make_console()
        render_surface(con, SURFACE_DATA)
        out = stream.getvalue()
        # (1,8192) predicted_gb=3.80 → "3.80" or "3.8" must appear
        assert "3.80" in out or "3.8" in out

    def test_summary_counts_present(self):
        """Measured / skipped cell counts should be reported."""
        con, stream = _make_console()
        render_surface(con, SURFACE_DATA)
        out = stream.getvalue()
        # n_cells_measured=8, n_cells_skipped=8
        assert "8" in out

    def test_next_block_present(self):
        con, stream = _make_console()
        render_surface(con, SURFACE_DATA)
        out = stream.getvalue()
        assert "next" in out

    def test_profile_note_embedded(self):
        """The profile line must appear inside render_surface output."""
        con, stream = _make_console()
        render_surface(con, SURFACE_DATA)
        out = stream.getvalue()
        assert "calibration" in out.lower()

    def test_plain_no_ansi(self):
        con, stream = _make_console()
        render_surface(con, SURFACE_DATA)
        out = stream.getvalue()
        assert "\033" not in out

    # --- verbose ---

    def test_verbose_raw_appendix_appears(self):
        """verbose=True appends a raw appendix with per-cell numbers."""
        con, stream = _make_console(verbose=True)
        render_surface(con, SURFACE_DATA)
        out = stream.getvalue()
        # raw appendix must show latency and peak_gb for measured cells
        # batch=1, seq=128 → latency_ms=30.4, peak_gb=0.41
        assert "30.4" in out or "30" in out
        assert "0.41" in out

    def test_verbose_raw_appendix_absent_when_not_verbose(self):
        """Non-verbose must NOT show raw per-cell numbers (latency/peak_gb)."""
        con, stream = _make_console(verbose=False)
        render_surface(con, SURFACE_DATA)
        out = stream.getvalue()
        # latency_ms=30.4 for (1,128) should NOT appear in normal output
        # (throughput is shown, not latency in default mode)
        # We confirm no raw appendix section header like "raw" appears alone
        # Actually check latency for cell (2,2048)=1280.0 is not in plain out
        assert "1280.0" not in out and "1280" not in out

    def test_verbose_secondary_metrics_present(self):
        """Verbose raw appendix should include both latency and MLX peak per cell."""
        con, stream = _make_console(verbose=True)
        render_surface(con, SURFACE_DATA)
        out = stream.getvalue()
        # (2,2048): latency_ms=1280.0, peak_gb=1.80
        assert "1280" in out
        assert "1.80" in out or "1.8" in out

    def test_verbose_color_smoke(self):
        """Verbose+color: no crash, ANSI codes present."""
        con, stream = _make_console(verbose=True, color=True)
        render_surface(con, SURFACE_DATA)
        out = stream.getvalue()
        assert "\033[" in out

    # --- color smoke ---

    def test_color_smoke(self):
        con, stream = _make_console(color=True)
        render_surface(con, SURFACE_DATA)
        out = stream.getvalue()
        assert "\033[" in out

    # --- golden for non-verbose plain output structure ---

    def test_plain_golden_structure(self):
        """
        Lock the high-level structure of normal (non-verbose) plain output.

        We assert the exact section/status/table/next line order rather than
        byte-exact content, so that small wording tweaks don't break this test
        while the shape is still locked.
        """
        con, stream = _make_console()
        render_surface(con, SURFACE_DATA)
        out = stream.getvalue()

        # 1. A section title line appears before the table
        lines = out.splitlines()
        non_empty = [l for l in lines if l.strip()]
        assert len(non_empty) > 5, "expected multiple non-empty lines"

        # 2. "next" footer appears AFTER at least 5 non-empty lines
        next_idx = next((i for i, l in enumerate(non_empty) if "next" in l.lower()), None)
        assert next_idx is not None, "'next' block missing"
        assert next_idx > 3, "'next' block should come after the table content"

        # 3. The table header row (with all seq lengths) comes before the data rows
        seq_header_idx = next(
            (i for i, l in enumerate(non_empty) if "128" in l and "8192" in l), None
        )
        assert seq_header_idx is not None, "no single line contains all seq headers"
        assert seq_header_idx < next_idx, "seq header must appear before next block"
