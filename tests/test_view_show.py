"""Tests for wmx_suite/views/show.py — pure rendering, hardware-free."""
import io
import pytest
from wmx_suite.ui import Console
from wmx_suite.views import show


# ---------------------------------------------------------------------------
# Realistic fixture: gpt-oss-20b-MXFP4-Q8 (non-quantizable, fp16-only)
# Matches renderShowProposed in docs/mockups/cli-output-mockup.html
# ---------------------------------------------------------------------------
FIXTURE_FP16 = {
    "hf_id": "mlx-community/gpt-oss-20b-MXFP4-Q8",
    "weights_gb": 12.1,
    "n_layers": 24,
    "growing_layers": 12,
    "max_context": 131072,
    "kv_label": "sliding-window (RotatingKVCache)",
    "can_quantize_kv": False,
    "growth_gb_per_1k": 0.025,
    # verbose-only raw fields
    "cache_type": "RotatingKVCache",
    "kv_heads": 8,
    "head_dim": 64,
    "hidden_size": 2880,
    "layer_types": '{"sliding_attention": 12, "full_attention": 12}',
    "max_kv_size_enforced": True,
    "is_causal": True,
    "fp16_kv_bytes_per_token": 24576,
}

# A quantizable model fixture for the alternative branch
FIXTURE_QUANTIZABLE = {
    "hf_id": "mlx-community/Qwen2.5-VL-7B-Instruct-4bit",
    "weights_gb": 5.0,
    "n_layers": 28,
    "growing_layers": 28,
    "max_context": 32768,
    "kv_label": "standard (KVCache)",
    "can_quantize_kv": True,
    "growth_gb_per_1k": 0.015,
    # verbose-only
    "cache_type": "KVCache",
    "kv_heads": 8,
    "head_dim": 128,
    "hidden_size": 3584,
    "layer_types": '{"full_attention": 28}',
    "max_kv_size_enforced": True,
    "is_causal": True,
    "fp16_kv_bytes_per_token": 14336,
}


def _make_console(verbose=False, color=False) -> tuple[Console, io.StringIO]:
    stream = io.StringIO()
    console = Console(color=color, verbose=verbose, stream=stream)
    return console, stream


# ---------------------------------------------------------------------------
# Normal mode — fp16-only model
# ---------------------------------------------------------------------------

def test_normal_plain_fp16():
    console, stream = _make_console()
    show.render(console, FIXTURE_FP16)
    out = stream.getvalue()

    # Model name header (prefix kept — it IS the accent header)
    assert "mlx-community/gpt-oss-20b-MXFP4-Q8" in out

    # Section headers
    assert "what it is" in out
    assert "how its memory behaves" in out

    # Field: weights
    assert "12.1 GB" in out
    assert "size it occupies just to load" in out

    # Field: layers
    assert "24" in out
    assert "12 of them grow with context" in out

    # Field: max context — comma-formatted
    assert "131,072 tok" in out
    assert "NOT your safe limit" in out

    # KV cache field — fp16 model
    assert "sliding-window" in out
    assert "can't be 4-bit compressed" in out

    # Growth field
    assert "0.025 GB / 1k tok" in out

    # next_block
    assert "wmx-suite characterize gpt-oss-20b-MXFP4-Q8" in out
    assert "wmx-suite run --model gpt-oss-20b-MXFP4-Q8 --dry-run" in out

    # Verbose raw section header and raw fields should NOT appear in normal mode
    assert "raw architecture" not in out
    assert "24576" not in out
    # n_layers as a raw dump key should not appear (but the value "24" can)
    assert "n_layers" not in out


def test_normal_plain_quantizable():
    console, stream = _make_console()
    show.render(console, FIXTURE_QUANTIZABLE)
    out = stream.getvalue()

    assert "Qwen2.5-VL-7B-Instruct-4bit" in out
    assert "standard" in out
    assert "can be compressed to 4-bit" in out
    assert "5.0 GB" in out


# ---------------------------------------------------------------------------
# Verbose mode — appends raw architecture dump
# ---------------------------------------------------------------------------

def test_verbose_appends_raw():
    console, stream = _make_console(verbose=True)
    show.render(console, FIXTURE_FP16)
    out = stream.getvalue()

    # Normal content still present
    assert "what it is" in out
    assert "12.1 GB" in out

    # Verbose appendix
    assert "raw architecture (--verbose)" in out
    assert "RotatingKVCache" in out
    assert "24576" in out
    assert "n_layers" in out
    assert "growing_layers" in out
    assert "kv_heads" in out
    assert "head_dim" in out
    assert "hidden_size" in out
    assert "can_quantize_kv" in out
    assert "is_causal" in out
    assert "max_kv_size_enforced" in out
    assert "layer_types" in out


def test_verbose_raw_absent_in_normal():
    console, stream = _make_console(verbose=False)
    show.render(console, FIXTURE_FP16)
    out = stream.getvalue()
    assert "raw architecture" not in out
    assert "24576" not in out


# ---------------------------------------------------------------------------
# Color smoke tests
# ---------------------------------------------------------------------------

def test_color_emits_ansi():
    console, stream = _make_console(color=True)
    show.render(console, FIXTURE_FP16)
    out = stream.getvalue()
    assert "\033[" in out


def test_plain_no_ansi():
    console, stream = _make_console(color=False)
    show.render(console, FIXTURE_FP16)
    out = stream.getvalue()
    assert "\033" not in out


# ---------------------------------------------------------------------------
# Prefix stripping in next_block
# ---------------------------------------------------------------------------

def test_short_name_in_next_block():
    """next_block commands use the short name without mlx-community/."""
    console, stream = _make_console()
    show.render(console, FIXTURE_FP16)
    out = stream.getvalue()
    # Commands in next_block should not repeat the full prefix
    # (the full hf_id appears once as the accent header — that's fine)
    next_idx = out.index("next")
    tail = out[next_idx:]
    assert "gpt-oss-20b-MXFP4-Q8" in tail
    # But full prefix should NOT appear in the next_block tail
    assert "mlx-community/" not in tail


# ---------------------------------------------------------------------------
# Structural: two section headers in normal mode (not three)
# ---------------------------------------------------------------------------

def test_section_count_normal():
    console, stream = _make_console(verbose=False)
    show.render(console, FIXTURE_FP16)
    out = stream.getvalue()
    assert out.count("what it is") == 1
    assert out.count("how its memory behaves") == 1
    # raw section absent
    assert out.count("raw architecture") == 0


def test_section_count_verbose():
    console, stream = _make_console(verbose=True)
    show.render(console, FIXTURE_FP16)
    out = stream.getvalue()
    assert out.count("raw architecture") == 1
