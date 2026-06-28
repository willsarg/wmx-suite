# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""Governed multi-prompt MLX benchmark worker (2026-06-28-benchmark-verb).

Gate runs before any weight load (Rule #1): a vetoed gate prints a refused JSON line to
stdout and exits 1 without loading the model. Per-prompt governance (governed_max_tokens)
rejects individual prompts that would overflow the ceiling. Successful prompts get
completions in order. All tests mock mlx_lm — no weights are loaded.
"""
from __future__ import annotations

import io
import json
import sys
from types import SimpleNamespace

import pytest

from wmx_suite import benchmark


# --------------------------------------------------------------------------- #
# Shared stubs
# --------------------------------------------------------------------------- #

def _word_tok():
    """Fake tokenizer: encode counts space-delimited words."""
    class _FakeTok:
        def encode(self, text: str) -> list[int]:
            return list(range(len(text.split())))
    return _FakeTok()


def _fake_mlx(monkeypatch, completion: str = "ok"):
    """Patch sys.modules['mlx_lm'] with fake load + generate; return captured-call dict."""
    captured: dict = {}

    class _FakeTok:
        def encode(self, text: str) -> list[int]:
            return list(range(len(text.split())))

    def fake_load(hf_id: str):
        captured["loaded"] = hf_id
        return ("MODEL", _FakeTok())

    def fake_generate(model, tok, prompt=None, max_tokens=None, **kw):
        captured.setdefault("calls", []).append(
            {"prompt": prompt, "max_tokens": max_tokens, "kw": kw}
        )
        return completion

    monkeypatch.setitem(
        sys.modules, "mlx_lm",
        SimpleNamespace(load=fake_load, generate=fake_generate),
    )
    return captured


# --------------------------------------------------------------------------- #
# _run_prompts — pure inner loop, unit-testable without weights
# --------------------------------------------------------------------------- #

def test_run_prompts_returns_completions_in_order():
    fake_gen = lambda model, tok, prompt=None, max_tokens=None, **kw: f"reply: {prompt}"
    results = benchmark._run_prompts(
        ["hello world", "hi there"],
        _word_tok(),
        max_tokens=100,
        ceiling=512,
        effective_kv=None,
        mlx_generate=fake_gen,
        model="M",
    )
    assert results == [
        {"prompt_index": 0, "completion": "reply: hello world"},
        {"prompt_index": 1, "completion": "reply: hi there"},
    ]


def test_run_prompts_refuses_prompt_that_fills_ceiling():
    """A prompt whose token count >= ceiling gets a per-item refused, generate is NOT called."""

    def boom(*a, **kw):
        raise AssertionError("generate must not be called for an overlong prompt")

    class _LongTok:
        def encode(self, text: str) -> list[int]:
            return list(range(600))  # 600 > ceiling 512

    results = benchmark._run_prompts(
        ["very long prompt"],
        _LongTok(),
        max_tokens=100,
        ceiling=512,
        effective_kv=None,
        mlx_generate=boom,
        model="M",
    )
    assert len(results) == 1
    assert results[0]["prompt_index"] == 0
    assert results[0]["refused"] is True
    assert "512" in results[0]["reason"]   # ceiling in reason


def test_run_prompts_passes_kv_kwargs_when_quantized():
    kw_seen: dict = {}

    def fake_gen(model, tok, prompt=None, max_tokens=None, **kw):
        kw_seen.update(kw)
        return "done"

    benchmark._run_prompts(
        ["hello"],
        _word_tok(),
        max_tokens=100,
        ceiling=512,
        effective_kv=4,
        mlx_generate=fake_gen,
        model="M",
    )
    assert kw_seen["kv_bits"] == 4
    assert kw_seen["kv_group_size"] == 64
    assert kw_seen["quantized_kv_start"] == 5000


def test_run_prompts_no_kv_kwargs_for_fp16():
    kw_seen: dict = {}

    def fake_gen(model, tok, prompt=None, max_tokens=None, **kw):
        kw_seen.update(kw)
        return "done"

    benchmark._run_prompts(
        ["hello"],
        _word_tok(),
        max_tokens=100,
        ceiling=512,
        effective_kv=None,
        mlx_generate=fake_gen,
        model="M",
    )
    assert "kv_bits" not in kw_seen


def test_run_prompts_mixed_short_and_long_prompts():
    """Short prompt succeeds; long prompt (> ceiling) gets per-item refused — in index order."""

    class _VariableTok:
        def encode(self, text: str) -> list[int]:
            return list(range(600)) if "long" in text else list(range(5))

    calls = []

    def fake_gen(model, tok, prompt=None, max_tokens=None, **kw):
        calls.append(prompt)
        return "completion"

    results = benchmark._run_prompts(
        ["short", "this is long"],
        _VariableTok(),
        max_tokens=100,
        ceiling=512,
        effective_kv=None,
        mlx_generate=fake_gen,
        model="M",
    )
    assert results[0] == {"prompt_index": 0, "completion": "completion"}
    assert results[1]["prompt_index"] == 1
    assert results[1]["refused"] is True
    assert calls == ["short"]   # long prompt must not reach generate


# --------------------------------------------------------------------------- #
# benchmark() — gate + load once + loop
# --------------------------------------------------------------------------- #

def test_benchmark_gate_refusal_returns_refused_dict_without_loading(monkeypatch):
    """Gate veto must prevent any weight load (Rule #1)."""
    monkeypatch.setattr(
        benchmark, "_pre_load_gate",
        lambda *a, **k: ({"refused": True, "reason": "too big"}, None),
    )
    monkeypatch.setitem(sys.modules, "mlx_lm", SimpleNamespace(
        load=lambda *a: (_ for _ in ()).throw(AssertionError("weights loaded — gate failed!")),
        generate=lambda *a, **k: (_ for _ in ()).throw(AssertionError("generated!")),
    ))
    out = benchmark.benchmark("big/model", 4096, prompts=["hi"],
                              margin_gb=4.0, overhead_gb=1.0)
    assert out == {"context": 4096, "refused": True, "reason": "too big"}


def test_benchmark_normal_path_loads_once_and_returns_all_results(monkeypatch):
    monkeypatch.setattr(benchmark, "_pre_load_gate", lambda *a, **k: (None, None))
    captured = _fake_mlx(monkeypatch, completion="the answer")
    out = benchmark.benchmark("org/m", 512, prompts=["hello world", "hi there"],
                              margin_gb=4.0, overhead_gb=1.0)
    assert out["context"] == 512
    assert len(out["results"]) == 2
    assert out["results"][0] == {"prompt_index": 0, "completion": "the answer"}
    assert out["results"][1] == {"prompt_index": 1, "completion": "the answer"}
    assert captured["loaded"] == "org/m"      # model loaded exactly once


def test_benchmark_per_prompt_refusal_when_prompt_too_long(monkeypatch):
    """A prompt that fills the ceiling gets a per-item refused dict, not a crash."""
    monkeypatch.setattr(benchmark, "_pre_load_gate", lambda *a, **k: (None, None))

    class _LongTok:
        def encode(self, text: str) -> list[int]:
            return list(range(600))   # 600 > ceiling 512

    def fake_load(hf_id: str):
        return ("MODEL", _LongTok())

    monkeypatch.setitem(sys.modules, "mlx_lm",
                        SimpleNamespace(load=fake_load, generate=lambda *a, **k: "ok"))
    out = benchmark.benchmark("org/m", 512, prompts=["long prompt"],
                              margin_gb=4.0, overhead_gb=1.0)
    assert out["results"][0]["refused"] is True
    assert "512" in out["results"][0]["reason"]


# --------------------------------------------------------------------------- #
# main() — CLI wiring
# --------------------------------------------------------------------------- #

def test_main_reads_prompts_from_stdin_and_prints_json(monkeypatch, capsys):
    seen: dict = {}

    def fake_benchmark(hf_id, ceiling, *, prompts, margin_gb, overhead_gb,
                       max_tokens, kv_bits=None):
        seen.update(hf_id=hf_id, ceiling=ceiling, prompts=prompts,
                    margin_gb=margin_gb, overhead_gb=overhead_gb,
                    max_tokens=max_tokens, kv_bits=kv_bits)
        return {"context": ceiling, "results": []}

    monkeypatch.setattr(benchmark, "benchmark", fake_benchmark)
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(["hello", "world"])))
    benchmark.main(["org/m", "4096", "--margin", "4", "--overhead", "1"])
    line = capsys.readouterr().out.strip()
    assert json.loads(line) == {"context": 4096, "results": []}
    assert seen == {
        "hf_id": "org/m", "ceiling": 4096, "prompts": ["hello", "world"],
        "margin_gb": 4.0, "overhead_gb": 1.0, "max_tokens": 256, "kv_bits": None,
    }


def test_main_exits_1_and_prints_refusal_on_gate_veto(monkeypatch, capsys):
    monkeypatch.setattr(
        benchmark, "benchmark",
        lambda *a, **k: {"context": 4096, "refused": True, "reason": "too big"},
    )
    monkeypatch.setattr("sys.stdin", io.StringIO("[]"))
    with pytest.raises(SystemExit) as exc:
        benchmark.main(["big/model", "4096", "--margin", "4", "--overhead", "1"])
    assert exc.value.code == 1
    out = capsys.readouterr().out.strip()
    assert json.loads(out) == {"context": 4096, "refused": True, "reason": "too big"}


def test_main_threads_kv_bits_to_benchmark(monkeypatch):
    seen: dict = {}
    monkeypatch.setattr(benchmark, "benchmark",
                        lambda *a, **k: seen.update(k) or {"context": 1, "results": []})
    monkeypatch.setattr("sys.stdin", io.StringIO("[]"))
    benchmark.main(["org/m", "4096", "--margin", "4", "--overhead", "1", "--kv-bits", "8"])
    assert seen["kv_bits"] == 8


def test_main_custom_max_tokens(monkeypatch):
    seen: dict = {}
    monkeypatch.setattr(benchmark, "benchmark",
                        lambda *a, **k: seen.update(k) or {"context": 1, "results": []})
    monkeypatch.setattr("sys.stdin", io.StringIO("[]"))
    benchmark.main(["org/m", "4096", "--margin", "4", "--overhead", "1", "--max-tokens", "512"])
    assert seen["max_tokens"] == 512


def test_main_default_max_tokens_is_256(monkeypatch):
    seen: dict = {}
    monkeypatch.setattr(benchmark, "benchmark",
                        lambda *a, **k: seen.update(k) or {"context": 1, "results": []})
    monkeypatch.setattr("sys.stdin", io.StringIO("[]"))
    benchmark.main(["org/m", "4096", "--margin", "4", "--overhead", "1"])
    assert seen["max_tokens"] == 256
