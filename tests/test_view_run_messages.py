"""Tests for wmx_suite/views/run_messages.py — PURE render functions.

Hardware-free: all fixtures are static dicts. Uses Console(color=False, verbose=False)
so output is plain text (no ANSI) and stable for golden assertions.
"""
from __future__ import annotations

import io

import pytest

from wmx_suite.ui import Console
from wmx_suite.views.run_messages import (
    render_refusal,
    render_plan,
    render_not_found,
    render_no_models,
)


# ---------------------------------------------------------------------------
# Shared fixtures (real numbers from mockup / M4 Pro testbed)
# ---------------------------------------------------------------------------

REFUSAL_DATA = {
    "model": "gpt-oss-20b-MXFP4-Q8",          # short name (mlx-community/ stripped)
    "needs_gb": 15.56,                          # live_base + model_base
    "budget_gb": 15.18,                         # threshold (wall − margin)
    "wall_gb": 17.18,
    "live_base_gb": 3.12,
    "model_base_gb": 12.44,
    "slope_gb_per_k": 0.36003,
    "safe_cap_tok": 0,
    "source": "measured",
    "cache_type": "RotatingKVCache",
    "kv_mode": "fp16 (not quantizable)",
}

PLAN_DATA = {
    "model": "Qwen2.5-VL-7B-Instruct-4bit",
    "source": "measured",
    "cache_type": "KVCache",
    "kv_mode": "4-bit",
    "live_base_gb": 3.14,
    "model_base_gb": 8.10,
    "budget_gb": 15.18,
    "wall_gb": 17.18,
    "slope_gb_per_k": 0.2678,
    "max_kv_size": 26439,
    "model_max": 131072,
    "max_kv_size_enforced": True,
}

NOT_FOUND_DATA = {
    "model": "mlx-community/Llama-3-70B",
    "cache_path": "~/.cache/huggingface/hub",
    "hf_home_set": False,
}

NO_MODELS_DATA: dict = {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _console(*, verbose: bool = False, color: bool = False) -> tuple[Console, io.StringIO]:
    buf = io.StringIO()
    con = Console(color=color, verbose=verbose, stream=buf)
    return con, buf


# ===========================================================================
# render_refusal
# ===========================================================================

class TestRenderRefusal:
    def test_normal_golden(self):
        """Normal output matches the mockup's renderRefusalProposed (no verbose appendix)."""
        con, buf = _console()
        render_refusal(con, REFUSAL_DATA)
        out = buf.getvalue()

        # Headline
        assert "Won't run gpt-oss-20b-MXFP4-Q8" in out
        assert "can't fit safely on this Mac" in out

        # why line — human-rounded numbers
        assert "15.56 GB" in out
        assert "15.18 GB" in out

        # crash-wall context
        assert "17.18 GB" in out
        assert "crash wall" in out or "hard-lock" in out

        # try options — all three present
        assert "wmx-suite health" in out
        assert "free up memory" in out
        assert "run --force" in out
        assert "may crash" in out

        # NO raw appendix in normal mode
        assert "source=" not in out
        assert "slope" not in out

    def test_verbose_appends_raw(self):
        """Verbose mode appends a raw section with the underlying numbers."""
        con, buf = _console(verbose=True)
        render_refusal(con, REFUSAL_DATA)
        out = buf.getvalue()

        # Normal content still present
        assert "Won't run gpt-oss-20b-MXFP4-Q8" in out

        # Raw appendix
        assert "source=measured" in out
        assert "RotatingKVCache" in out
        assert "3.12" in out    # live_base
        assert "12.44" in out   # model_base
        assert "15.56" in out   # total (may also appear in normal; that's fine)
        assert "0.36003" in out  # slope
        assert "17.18" in out
        assert "15.18" in out
        assert "safe cap" in out or "512" in out  # MIN_USEFUL_CTX

    def test_color_smoke(self):
        """Runs without error in color mode; headline uses bad-role text."""
        con, buf = _console(color=True)
        render_refusal(con, REFUSAL_DATA)
        out = buf.getvalue()
        # ANSI red escape for 'bad' role (\033[31m)
        assert "\033[31m" in out
        assert "Won't run" in out

    def test_refusal_try_order(self):
        """health is listed before free-up-memory, which comes before run --force."""
        con, buf = _console()
        render_refusal(con, REFUSAL_DATA)
        out = buf.getvalue()
        health_pos = out.index("wmx-suite health")
        free_pos = out.index("free up memory")
        force_pos = out.index("run --force")
        assert health_pos < free_pos < force_pos

    def test_strips_mlx_community_prefix(self):
        """If model name has mlx-community/ prefix it is stripped in the headline."""
        con, buf = _console()
        data = {**REFUSAL_DATA, "model": "mlx-community/gpt-oss-20b-MXFP4-Q8"}
        render_refusal(con, data)
        out = buf.getvalue()
        assert "mlx-community/" not in out
        assert "gpt-oss-20b-MXFP4-Q8" in out


# ===========================================================================
# render_plan
# ===========================================================================

class TestRenderPlan:
    def test_normal_shows_key_fields(self):
        """Normal plan block shows model, source, cache type, kv mode, budget, max-kv-size."""
        con, buf = _console()
        render_plan(con, PLAN_DATA)
        out = buf.getvalue()

        assert "Qwen2.5-VL-7B-Instruct-4bit" in out
        assert "measured" in out
        assert "KVCache" in out or "4-bit" in out
        assert "15.18 GB" in out or "budget" in out
        assert "26,439" in out or "26439" in out  # max-kv-size cap

    def test_normal_no_raw_numbers(self):
        """Normal mode does not dump raw slope/live_base."""
        con, buf = _console()
        render_plan(con, PLAN_DATA)
        out = buf.getvalue()
        # slope should not appear in non-verbose output
        assert "0.2678" not in out

    def test_verbose_may_add_raw(self):
        """Verbose mode may include the slope or budget math."""
        con, buf = _console(verbose=True)
        render_plan(con, PLAN_DATA)
        out = buf.getvalue()
        # Must still show the plan fundamentals
        assert "Qwen2.5-VL-7B-Instruct-4bit" in out

    def test_color_smoke(self):
        """Runs without error in color mode."""
        con, buf = _console(color=True)
        render_plan(con, PLAN_DATA)
        out = buf.getvalue()
        assert "Qwen2.5-VL-7B-Instruct-4bit" in out

    def test_no_refuse_or_safety_logic(self):
        """render_plan must not emit any refusal text — it's the non-refusal path."""
        con, buf = _console()
        render_plan(con, PLAN_DATA)
        out = buf.getvalue()
        assert "Won't run" not in out
        assert "REFUSED" not in out
        assert "can't fit" not in out


# ===========================================================================
# render_not_found
# ===========================================================================

class TestRenderNotFound:
    def test_normal_golden(self):
        """Normal output matches renderNotFoundProposed: headline + context + 3 tries."""
        con, buf = _console()
        render_not_found(con, NOT_FOUND_DATA)
        out = buf.getvalue()

        # Headline
        assert "mlx-community/Llama-3-70B" in out
        assert "Hugging Face cache" in out

        # Context gloss
        assert "already downloaded" in out or "already download" in out or "suite only sees" in out

        # Two try options (download, then list)
        assert "hf download mlx-community/Llama-3-70B" in out
        assert "wmx-suite list" in out

        # NO cache path in normal mode
        assert "~/.cache/huggingface" not in out

    def test_verbose_appends_cache_path(self):
        """Verbose mode appends the searched cache path."""
        con, buf = _console(verbose=True)
        render_not_found(con, NOT_FOUND_DATA)
        out = buf.getvalue()

        assert "~/.cache/huggingface/hub" in out

    def test_verbose_hf_home_unset_note(self):
        """When hf_home_set is False, verbose notes HF_HOME is unset."""
        con, buf = _console(verbose=True)
        render_not_found(con, {**NOT_FOUND_DATA, "hf_home_set": False})
        out = buf.getvalue()
        assert "HF_HOME" in out or "unset" in out

    def test_color_smoke(self):
        """Runs without error in color mode; headline uses bad-role text."""
        con, buf = _console(color=True)
        render_not_found(con, NOT_FOUND_DATA)
        out = buf.getvalue()
        assert "\033[31m" in out
        assert "Llama-3-70B" in out

    def test_try_order(self):
        """hf download first, then list."""
        con, buf = _console()
        render_not_found(con, NOT_FOUND_DATA)
        out = buf.getvalue()
        dl_pos = out.index("hf download")
        list_pos = out.index("wmx-suite list")
        assert dl_pos < list_pos


# ===========================================================================
# render_no_models
# ===========================================================================

class TestRenderNoModels:
    def test_normal_guidance(self):
        """No-models state points at downloading + characterize."""
        con, buf = _console()
        render_no_models(con, NO_MODELS_DATA)
        out = buf.getvalue()

        # Some kind of "no models" headline
        assert "no" in out.lower() or "empty" in out.lower() or "haven't" in out.lower()

        # Must point at how to get a model and how to measure it
        assert "hf download" in out
        assert "characterize" in out

    def test_color_smoke(self):
        """Runs without error in color mode."""
        con, buf = _console(color=True)
        render_no_models(con, NO_MODELS_DATA)
        out = buf.getvalue()
        assert len(out) > 0

    def test_verbose_no_crash(self):
        """Verbose mode doesn't crash on empty data."""
        con, buf = _console(verbose=True)
        render_no_models(con, NO_MODELS_DATA)
        out = buf.getvalue()
        assert len(out) > 0
