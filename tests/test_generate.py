# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""Governed one-shot generate worker (2026-06-23-generate-verb).

The same L4 safety-first discipline as measure_one: the model is described and the
existing safety gate runs *before* any load. An unknown/non-causal model or a vetoing
gate refuses with no load; only a safe (model, ctx) reaches mlx_lm. Emits one JSON line:

    success: {"context": <int>, "completion": "<text>"}
    refused: {"context": <int>, "refused": true, "reason": "<why>"}

These tests never touch real mlx — mlx_lm.load/generate are monkeypatched and the
refusal paths never reach a load.
"""
from __future__ import annotations

import io
import json
from types import SimpleNamespace

from wmx_suite import generate


def _info(weights_gb=0.1, slope=0.01, is_causal=True, can_quant=True, max_ctx=None):
    return SimpleNamespace(
        weights_gb=weights_gb,
        is_causal=is_causal,
        can_quantize_kv=can_quant,
        max_context=max_ctx,
        estimated_slope_gb_per_k=lambda kv_bits=None: slope,
    )


def _limits(wall_gb=40.0, wired_now_gb=8.0):
    return SimpleNamespace(
        wall_gb=wall_gb,
        wired_now_gb=wired_now_gb,
        safe_threshold_gb=lambda margin: wall_gb - margin,
    )


def _patch_safe_env(monkeypatch):
    monkeypatch.setattr(generate.system, "read_limits", lambda: _limits())
    monkeypatch.setattr(generate.system, "sample_settled_baseline", lambda: 8.0)


def _fake_mlx(monkeypatch, completion="hello world"):
    """Make the lazy `from mlx_lm import generate, load` resolve to fakes."""
    import sys

    captured = {}

    def fake_load(hf_id):
        captured["loaded"] = hf_id
        return ("MODEL", "TOK")

    def fake_generate(model, tok, prompt=None, max_tokens=None, **kw):
        captured["model"] = model
        captured["tok"] = tok
        captured["prompt"] = prompt
        captured["max_tokens"] = max_tokens
        captured["kw"] = kw
        return completion

    monkeypatch.setitem(sys.modules, "mlx_lm",
                        SimpleNamespace(load=fake_load, generate=fake_generate))
    return captured


def _fake_tokenizer(monkeypatch, n_tokens):
    """Make the lazy `from transformers import AutoTokenizer` count to *n_tokens*.

    Encodes the prompt to a list of length *n_tokens* without loading any weights —
    the test asserts no model load happens before the gate.
    """
    import sys

    class FakeTok:
        def encode(self, text):
            return list(range(n_tokens))

    fake_auto = SimpleNamespace(from_pretrained=lambda hf_id: FakeTok())
    monkeypatch.setitem(sys.modules, "transformers",
                        SimpleNamespace(AutoTokenizer=fake_auto))


# --------------------------------------------------------------------------- #
# generate() — describe + gate + (safe) load+generate
# --------------------------------------------------------------------------- #
def test_generate_happy_path_returns_completion(monkeypatch):
    monkeypatch.setattr(generate.models, "describe", lambda hf: _info())
    _patch_safe_env(monkeypatch)
    monkeypatch.setattr(generate.measure_one, "safety_gate", lambda *a, **k: None)
    _fake_tokenizer(monkeypatch, n_tokens=5)
    captured = _fake_mlx(monkeypatch, completion="the answer is 42")

    out = generate.generate("small/model", 4000, prompt="what is the answer?",
                            margin_gb=4.0, overhead_gb=1.0, max_tokens=16)
    assert out == {"context": 4000, "completion": "the answer is 42"}
    assert captured["loaded"] == "small/model"
    assert captured["prompt"] == "what is the answer?"
    assert captured["max_tokens"] == 16


def test_generate_defaults_to_fp16_no_kv_args(monkeypatch):
    monkeypatch.setattr(generate.models, "describe", lambda hf: _info())
    _patch_safe_env(monkeypatch)
    monkeypatch.setattr(generate.measure_one, "safety_gate", lambda *a, **k: None)
    _fake_tokenizer(monkeypatch, n_tokens=5)
    captured = _fake_mlx(monkeypatch)
    generate.generate("small/model", 4000, prompt="hi", margin_gb=4.0, overhead_gb=1.0,
                      max_tokens=16)
    assert "kv_bits" not in captured["kw"]  # fp16: no quant args passed to mlx_lm


def test_generate_passes_kv_bits_when_quantizable(monkeypatch):
    monkeypatch.setattr(generate.models, "describe", lambda hf: _info(can_quant=True))
    _patch_safe_env(monkeypatch)
    monkeypatch.setattr(generate.measure_one, "safety_gate", lambda *a, **k: None)
    _fake_tokenizer(monkeypatch, n_tokens=5)
    captured = _fake_mlx(monkeypatch)
    generate.generate("small/model", 4000, prompt="hi", margin_gb=4.0, overhead_gb=1.0,
                      max_tokens=16, kv_bits=4)
    assert captured["kw"]["kv_bits"] == 4
    assert captured["kw"]["kv_group_size"] == 64
    assert captured["kw"]["quantized_kv_start"] == 5000  # match production knobs


def test_generate_forces_fp16_for_nonquantizable_model(monkeypatch):
    monkeypatch.setattr(generate.models, "describe", lambda hf: _info(can_quant=False))
    _patch_safe_env(monkeypatch)
    monkeypatch.setattr(generate.measure_one, "safety_gate", lambda *a, **k: None)
    _fake_tokenizer(monkeypatch, n_tokens=5)
    captured = _fake_mlx(monkeypatch)
    generate.generate("sliding/model", 4000, prompt="hi", margin_gb=4.0, overhead_gb=1.0,
                      max_tokens=16, kv_bits=4)
    assert "kv_bits" not in captured["kw"]  # RotatingKVCache must run fp16 (Rule #1)


def test_generate_refuses_unknown_or_noncausal_model(monkeypatch):
    monkeypatch.setattr(generate.models, "describe", lambda hf: None)
    out = generate.generate("mystery/model", 4000, prompt="hi",
                            margin_gb=4.0, overhead_gb=1.0, max_tokens=8)
    assert out == {"context": 4000, "refused": True, "reason": out["reason"]}
    assert "causal" in out["reason"] or "not found" in out["reason"]


def test_generate_refuses_noncausal(monkeypatch):
    monkeypatch.setattr(generate.models, "describe", lambda hf: _info(is_causal=False))
    out = generate.generate("vision/model", 4000, prompt="hi",
                            margin_gb=4.0, overhead_gb=1.0, max_tokens=8)
    assert out["refused"] is True and "causal" in out["reason"]


def test_generate_refuses_when_gate_vetoes_without_loading(monkeypatch):
    monkeypatch.setattr(generate.models, "describe", lambda hf: _info())
    _patch_safe_env(monkeypatch)
    monkeypatch.setattr(generate.measure_one, "safety_gate",
                        lambda *a, **k: "predicted 99GB at 4000 tok >= safe budget 36GB")
    _fake_tokenizer(monkeypatch, n_tokens=2)
    # if this loads a model, the test fails loudly
    import sys
    monkeypatch.setitem(sys.modules, "mlx_lm", SimpleNamespace(
        load=lambda *a, **k: (_ for _ in ()).throw(AssertionError("loaded!")),
        generate=lambda *a, **k: (_ for _ in ()).throw(AssertionError("generated!")),
    ))
    out = generate.generate("small/model", 4000, prompt="hi",
                            margin_gb=4.0, overhead_gb=1.0, max_tokens=8)
    assert out["refused"] is True and out["context"] == 4000
    assert "safe budget" in out["reason"]


def test_generate_passes_gate_args_like_measure_one(monkeypatch):
    captured = {}
    monkeypatch.setattr(generate.models, "describe", lambda hf: _info())
    monkeypatch.setattr(generate.system, "read_limits", lambda: _limits())
    monkeypatch.setattr(generate.system, "sample_settled_baseline", lambda: 7.5)

    def fake_gate(info, limits, ctx, *, margin_gb, overhead_gb, live_base, kv_bits=None):
        captured.update(ctx=ctx, margin_gb=margin_gb, overhead_gb=overhead_gb,
                        live_base=live_base)
        return None

    monkeypatch.setattr(generate.measure_one, "safety_gate", fake_gate)
    # 3 prompt tokens + 8 max_tokens = 11, well under the 4000 ceiling, so the gate
    # sees the effective context, not the raw ceiling.
    _fake_tokenizer(monkeypatch, n_tokens=3)
    _fake_mlx(monkeypatch)
    generate.generate("small/model", 4000, prompt="hi",
                      margin_gb=4.0, overhead_gb=1.0, max_tokens=8)
    assert captured == {"ctx": 11, "margin_gb": 4.0, "overhead_gb": 1.0,
                        "live_base": 7.5}


def test_generate_gates_on_effective_context_not_ceiling(monkeypatch):
    """The gate must see prompt_tokens + max_tokens, capped at the ceiling — not ctx."""
    seen = {}
    monkeypatch.setattr(generate.models, "describe", lambda hf: _info())
    _patch_safe_env(monkeypatch)

    def fake_gate(info, limits, ctx, *, margin_gb, overhead_gb, live_base, kv_bits=None):
        seen["ctx"] = ctx
        return None

    monkeypatch.setattr(generate.measure_one, "safety_gate", fake_gate)
    _fake_tokenizer(monkeypatch, n_tokens=6)   # short prompt
    _fake_mlx(monkeypatch, completion="done")

    out = generate.generate("small/model", 40960, prompt="a short prompt",
                            margin_gb=4.0, overhead_gb=1.0, max_tokens=256)
    assert seen["ctx"] == 6 + 256            # effective context, NOT the 40960 ceiling
    assert out == {"context": 40960, "completion": "done"}  # report the ceiling


def test_generate_caps_effective_context_at_ceiling(monkeypatch):
    """A prompt long enough that prompt_tokens + max_tokens > ctx caps at the ceiling."""
    seen = {}
    monkeypatch.setattr(generate.models, "describe", lambda hf: _info())
    _patch_safe_env(monkeypatch)

    def fake_gate(info, limits, ctx, *, margin_gb, overhead_gb, live_base, kv_bits=None):
        seen["ctx"] = ctx
        return None

    monkeypatch.setattr(generate.measure_one, "safety_gate", fake_gate)
    _fake_tokenizer(monkeypatch, n_tokens=4000)   # long prompt
    _fake_mlx(monkeypatch, completion="done")

    out = generate.generate("small/model", 4096, prompt="x" * 99,
                            margin_gb=4.0, overhead_gb=1.0, max_tokens=512)
    assert seen["ctx"] == 4096               # 4000 + 512 > 4096 -> capped at ceiling
    assert out == {"context": 4096, "completion": "done"}


def test_generate_empty_prompt_counts_zero_tokens(monkeypatch):
    """Empty prompt -> prompt_tokens = 0 without touching the tokenizer or weights."""
    seen = {}
    monkeypatch.setattr(generate.models, "describe", lambda hf: _info())
    _patch_safe_env(monkeypatch)

    def fake_gate(info, limits, ctx, *, margin_gb, overhead_gb, live_base, kv_bits=None):
        seen["ctx"] = ctx
        return None

    monkeypatch.setattr(generate.measure_one, "safety_gate", fake_gate)
    # No transformers in sys.modules: an empty prompt must short-circuit before import.
    import sys
    monkeypatch.setitem(sys.modules, "transformers", None)
    _fake_mlx(monkeypatch, completion="done")

    out = generate.generate("small/model", 4096, prompt="",
                            margin_gb=4.0, overhead_gb=1.0, max_tokens=32)
    assert seen["ctx"] == 32                  # 0 + 32
    assert out == {"context": 4096, "completion": "done"}


# --------------------------------------------------------------------------- #
# main() — reads prompt from stdin, prints one JSON line
# --------------------------------------------------------------------------- #
def test_main_reads_prompt_from_stdin_prints_one_json_line(monkeypatch, capsys):
    seen = {}

    def fake_generate(hf_id, ctx, *, prompt, margin_gb, overhead_gb, max_tokens,
                      kv_bits=None):
        seen.update(hf_id=hf_id, ctx=ctx, prompt=prompt, margin_gb=margin_gb,
                    overhead_gb=overhead_gb, max_tokens=max_tokens, kv_bits=kv_bits)
        return {"context": ctx, "completion": "ok"}

    monkeypatch.setattr(generate, "generate", fake_generate)
    monkeypatch.setattr("sys.stdin", io.StringIO("explain quicksort"))
    generate.main(["small/model", "4000", "--margin", "4", "--overhead", "1",
                   "--max-tokens", "32"])
    line = capsys.readouterr().out.strip()
    assert json.loads(line) == {"context": 4000, "completion": "ok"}
    assert seen == {"hf_id": "small/model", "ctx": 4000, "prompt": "explain quicksort",
                    "margin_gb": 4.0, "overhead_gb": 1.0, "max_tokens": 32, "kv_bits": None}


def test_main_threads_kv_bits_to_generate(monkeypatch):
    seen = {}
    monkeypatch.setattr(generate, "generate",
                        lambda *a, **k: seen.update(k) or {"context": 1, "completion": "x"})
    monkeypatch.setattr("sys.stdin", io.StringIO("hi"))
    generate.main(["small/model", "4000", "--margin", "4", "--overhead", "1",
                   "--max-tokens", "8", "--kv-bits", "8"])
    assert seen["kv_bits"] == 8


def test_main_prints_refusal_json_line(monkeypatch, capsys):
    monkeypatch.setattr(generate, "generate",
                        lambda *a, **k: {"context": 4000, "refused": True,
                                         "reason": "won't load"})
    monkeypatch.setattr("sys.stdin", io.StringIO("anything"))
    generate.main(["big/model", "4000", "--margin", "4", "--overhead", "1",
                   "--max-tokens", "8"])
    line = capsys.readouterr().out.strip()
    assert json.loads(line) == {"context": 4000, "refused": True, "reason": "won't load"}
