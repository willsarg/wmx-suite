"""Tests for wmx_suite/views/scan.py — pure rendering, hardware-free."""
import io
import pytest
from wmx_suite.ui import Console
from wmx_suite.views import scan


# ---------------------------------------------------------------------------
# Realistic fixture matching the mockup (renderScanProposed)
# ---------------------------------------------------------------------------
FIXTURE = {
    "models": [
        {
            "hf_id": "mlx-community/Qwen2.5-VL-7B-Instruct-4bit",
            "weights_gb": 5.0,
            "kv_label": "quantizable KV",
            "cache_type": "KVCache",
            "weights_gb_exact": 4.98,
        },
        {
            "hf_id": "mlx-community/Qwen3.5-9B-OptiQ-4bit",
            "weights_gb": 9.0,
            "kv_label": "quantizable KV",
            "cache_type": "KVCache",
            "weights_gb_exact": 9.01,
        },
        {
            "hf_id": "mlx-community/gemma-4-e4b-it-OptiQ-4bit",
            "weights_gb": 7.7,
            "kv_label": "quantizable KV",
            "cache_type": "KVCache",
            "weights_gb_exact": 7.68,
        },
        {
            "hf_id": "mlx-community/gpt-oss-20b-MXFP4-Q8",
            "weights_gb": 12.1,
            "kv_label": "fp16-only KV (sliding-window cache)",
            "cache_type": "RotatingKVCache",
            "weights_gb_exact": 12.10,
        },
    ],
    "registered": 4,
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
    scan.render(console, FIXTURE)
    out = stream.getvalue()

    # Section header
    assert "Found these MLX models in your Hugging Face cache:" in out

    # Each model appears with its short name (prefix stripped)
    assert "Qwen2.5-VL-7B-Instruct-4bit" in out
    assert "Qwen3.5-9B-OptiQ-4bit" in out
    assert "gemma-4-e4b-it-OptiQ-4bit" in out
    assert "gpt-oss-20b-MXFP4-Q8" in out

    # mlx-community/ prefix must be stripped
    assert "mlx-community/" not in out

    # ✓ glyph
    assert "✓" in out

    # Human GB values (right-aligned column)
    assert "5.0 GB" in out
    assert "9.0 GB" in out
    assert "7.7 GB" in out
    assert "12.1 GB" in out

    # KV labels
    assert "quantizable KV" in out
    assert "fp16-only KV (sliding-window cache)" in out

    # Registered count
    assert "registered 4 models" in out

    # Inline gloss
    assert "can compress its context cache to 4-bit" in out
    assert "cache must stay full precision" in out

    # next_block: characterize suggestion
    assert "wmx-suite characterize <model>" in out

    # Verbose-only raw detail should NOT be present
    assert "KVCache" not in out
    assert "RotatingKVCache" not in out
    assert "4.98" not in out


# ---------------------------------------------------------------------------
# Verbose mode — must append cache_type and exact GB
# ---------------------------------------------------------------------------

def test_verbose_appends_raw_detail():
    console, stream = _make_console(verbose=True)
    scan.render(console, FIXTURE)
    out = stream.getvalue()

    # All normal content still present
    assert "Found these MLX models in your Hugging Face cache:" in out
    assert "registered 4 models" in out

    # Verbose appendix
    assert "KVCache" in out
    assert "RotatingKVCache" in out
    assert "4.98" in out
    assert "9.01" in out
    assert "7.68" in out
    assert "12.10" in out


# ---------------------------------------------------------------------------
# Color smoke test
# ---------------------------------------------------------------------------

def test_color_emits_ansi():
    console, stream = _make_console(color=True)
    scan.render(console, FIXTURE)
    out = stream.getvalue()
    assert "\033[" in out, "Expected ANSI escape codes when color=True"


def test_plain_no_ansi():
    console, stream = _make_console(color=False)
    scan.render(console, FIXTURE)
    out = stream.getvalue()
    assert "\033" not in out, "No ANSI escape codes expected when color=False"


# ---------------------------------------------------------------------------
# Structure: each model line has exactly one ✓
# ---------------------------------------------------------------------------

def test_each_model_gets_checkmark():
    console, stream = _make_console()
    scan.render(console, FIXTURE)
    out = stream.getvalue()
    assert out.count("✓") == len(FIXTURE["models"])


# ---------------------------------------------------------------------------
# next_block appears at the end
# ---------------------------------------------------------------------------

def test_next_block_last():
    console, stream = _make_console()
    scan.render(console, FIXTURE)
    out = stream.getvalue()
    nxt_idx = out.rfind("next")
    char_idx = out.rfind("wmx-suite characterize <model>")
    assert nxt_idx != -1
    assert char_idx > nxt_idx, "characterize command should follow 'next' header"
