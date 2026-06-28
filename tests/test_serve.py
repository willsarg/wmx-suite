# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""Governed MLX serve worker (2026-06-28-serve-worker).

Tests the governance logic, gate discipline, HTTP decision function, SSE streaming,
and OpenAI tool-calling — without loading real models or binding sockets. mlx_lm is
monkeypatched throughout; no weights downloaded.

    refused: {"refused": true, "reason": "<why>"}        — pre-load gate veto
    ready:   {"ready": true, "url": "http://127.0.0.1:<PORT>", "context": <ceiling>}
"""
from __future__ import annotations

import io
import json
import sys
from types import SimpleNamespace

import pytest

from wmx_suite import serve


# --------------------------------------------------------------------------- #
# Shared stubs
# --------------------------------------------------------------------------- #

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
    monkeypatch.setattr(serve.system, "read_limits", lambda: _limits())
    monkeypatch.setattr(serve.system, "sample_settled_baseline", lambda: 8.0)


class _FakeTok:
    """Word-splitting tokenizer stub — no weights, no network.

    apply_chat_template joins all message contents with a space; encode counts words.
    This makes prompt_tokens predictable from the raw content strings in tests.
    """

    def apply_chat_template(self, messages, *, tokenize=False, add_generation_prompt=True):
        return " ".join(m.get("content", "") for m in messages)

    def encode(self, text: str) -> list[int]:
        return list(range(max(0, len(text.split()))))


class _ToolTok(_FakeTok):
    """Tokenizer stub that advertises tool-calling capability."""

    has_tool_calling = True
    tool_call_start = "<tool_call>"
    tool_call_end = "</tool_call>"

    def apply_chat_template(self, messages, *, tokenize=False, add_generation_prompt=True,
                             tools=None):
        return " ".join(m.get("content", "") for m in messages)

    def tool_parser(self, text, tools):  # noqa: ARG002
        return {"name": "f", "arguments": {"x": 1}}


def _fake_mlx_generate(monkeypatch, completion="reply"):
    """Patch sys.modules['mlx_lm'] with a generate stub that records call args."""
    captured: dict = {}

    def fake_generate(model, tok, prompt=None, max_tokens=None, **kw):
        captured.update(prompt=prompt, max_tokens=max_tokens, kw=kw)
        return completion

    monkeypatch.setitem(
        sys.modules, "mlx_lm",
        SimpleNamespace(generate=fake_generate, load=None),
    )
    return captured


# --------------------------------------------------------------------------- #
# Streaming stubs
# --------------------------------------------------------------------------- #

class _FakeStreamResponse:
    """Minimal stand-in for mlx_lm.GenerationResponse."""

    def __init__(self, text: str, finish_reason=None):
        self.text = text
        self.finish_reason = finish_reason


class _FakeHandler:
    """Fake BaseHTTPRequestHandler that captures SSE writes without a real socket."""

    def __init__(self):
        self.wfile = io.BytesIO()
        self._response_code = None
        self._headers: dict[str, str] = {}
        self._end_headers_called = False

    def send_response(self, code: int) -> None:
        self._response_code = code

    def send_header(self, key: str, value: str) -> None:
        self._headers[key] = value

    def end_headers(self) -> None:
        self._end_headers_called = True


def _fake_mlx_stream_generate(monkeypatch, responses):
    """Patch sys.modules['mlx_lm'] with a stream_generate stub."""

    def fake_stream_generate(model, tok, prompt, max_tokens=None, **kw):
        yield from responses

    monkeypatch.setitem(
        sys.modules, "mlx_lm",
        SimpleNamespace(stream_generate=fake_stream_generate, generate=None, load=None),
    )


# --------------------------------------------------------------------------- #
# governed_max_tokens — pure helper, no deps
# --------------------------------------------------------------------------- #

def test_governed_max_tokens_rejects_when_need_exceeds_ceiling():
    # 100 prompt + 500 requested = 600 > 512 → None
    assert serve.governed_max_tokens(100, 500, 512) is None


def test_governed_max_tokens_accepts_when_within_ceiling():
    # 100 + 200 = 300 <= 512 → 200
    assert serve.governed_max_tokens(100, 200, 512) == 200


def test_governed_max_tokens_accepts_exactly_at_boundary():
    # 100 + 412 = 512 == 512 — need > ceiling is the veto condition, equality passes
    assert serve.governed_max_tokens(100, 412, 512) == 412


def test_governed_max_tokens_rejects_when_prompt_alone_at_ceiling():
    # prompt_tokens == ceiling → no room for any output → None
    assert serve.governed_max_tokens(512, 1, 512) is None


def test_governed_max_tokens_rejects_when_prompt_exceeds_ceiling():
    assert serve.governed_max_tokens(600, 1, 512) is None


def test_governed_max_tokens_zero_requested_within_ceiling():
    # requested=0, 5+0=5 <= 100 → accepted (clamp to 0)
    assert serve.governed_max_tokens(5, 0, 100) == 0


# --------------------------------------------------------------------------- #
# _pre_load_gate — mirrors generate.py's gate discipline
# --------------------------------------------------------------------------- #

def test_gate_vetoes_unknown_model(monkeypatch):
    monkeypatch.setattr(serve.models, "describe", lambda hf: None)
    refusal, kv = serve._pre_load_gate("unknown/model", 4096, margin_gb=4.0, overhead_gb=1.0)
    assert refusal is not None and refusal["refused"] is True
    assert "not found" in refusal["reason"]
    assert kv is None


def test_gate_vetoes_noncausal_model(monkeypatch):
    monkeypatch.setattr(serve.models, "describe", lambda hf: _info(is_causal=False))
    refusal, _ = serve._pre_load_gate("vision/model", 4096, margin_gb=4.0, overhead_gb=1.0)
    assert refusal["refused"] is True and "causal" in refusal["reason"]


def test_gate_vetoes_when_memory_unsafe(monkeypatch):
    monkeypatch.setattr(serve.models, "describe", lambda hf: _info())
    _patch_safe_env(monkeypatch)
    monkeypatch.setattr(serve.measure_one, "safety_gate",
                        lambda *a, **k: "predicted 99GB >= budget")
    refusal, _ = serve._pre_load_gate("big/model", 4096, margin_gb=4.0, overhead_gb=1.0)
    assert refusal["refused"] is True and "predicted 99GB" in refusal["reason"]


def test_gate_passes_when_safe_returns_none_refusal(monkeypatch):
    monkeypatch.setattr(serve.models, "describe", lambda hf: _info())
    _patch_safe_env(monkeypatch)
    monkeypatch.setattr(serve.measure_one, "safety_gate", lambda *a, **k: None)
    refusal, effective_kv = serve._pre_load_gate(
        "safe/model", 4096, margin_gb=4.0, overhead_gb=1.0
    )
    assert refusal is None
    assert effective_kv is None  # no kv_bits requested → fp16 (None)


def test_gate_resolves_effective_kv_bits_for_quantizable_model(monkeypatch):
    monkeypatch.setattr(serve.models, "describe", lambda hf: _info(can_quant=True))
    _patch_safe_env(monkeypatch)
    monkeypatch.setattr(serve.measure_one, "safety_gate", lambda *a, **k: None)
    _, effective_kv = serve._pre_load_gate(
        "quant/model", 4096, margin_gb=4.0, overhead_gb=1.0, kv_bits=4
    )
    assert effective_kv == 4  # quantizable model → kv_bits honoured


def test_gate_forces_fp16_for_nonquantizable_model(monkeypatch):
    monkeypatch.setattr(serve.models, "describe", lambda hf: _info(can_quant=False))
    _patch_safe_env(monkeypatch)
    monkeypatch.setattr(serve.measure_one, "safety_gate", lambda *a, **k: None)
    _, effective_kv = serve._pre_load_gate(
        "sliding/model", 4096, margin_gb=4.0, overhead_gb=1.0, kv_bits=4
    )
    assert effective_kv is None  # RotatingKVCache must run fp16 (Rule #1)


# --------------------------------------------------------------------------- #
# _handle_completions — pure HTTP decision function, no socket
# --------------------------------------------------------------------------- #

def test_handle_completions_returns_400_when_need_exceeds_ceiling(monkeypatch):
    _fake_mlx_generate(monkeypatch)  # must NOT be called
    # "hello world" → 2 tokens; requested=500 → need=502 > ceiling=400 → 400
    status, body = serve._handle_completions(
        {"messages": [{"role": "user", "content": "hello world"}], "max_tokens": 500},
        _FakeTok(), "MODEL", ceiling=400, kv_bits=None, hf_id="some/model",
    )
    assert status == 400
    assert body["error"]["type"] == "context_length_exceeded"
    assert "400" in body["error"]["message"]   # ceiling in message
    assert "502" in body["error"]["message"]   # need (2+500) in message


def test_handle_completions_returns_200_within_ceiling(monkeypatch):
    captured = _fake_mlx_generate(monkeypatch, completion="great answer")
    # "hi" → 1 token; requested=10 → need=11 <= 100 ceiling → accept
    status, body = serve._handle_completions(
        {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 10},
        _FakeTok(), "MODEL", ceiling=100, kv_bits=None, hf_id="some/model",
    )
    assert status == 200
    assert body["choices"][0]["message"]["content"] == "great answer"
    assert body["model"] == "some/model"
    assert captured["max_tokens"] == 10


def test_handle_completions_defaults_max_tokens_when_absent(monkeypatch):
    captured = _fake_mlx_generate(monkeypatch)
    # No max_tokens → defaults to 512; 1 prompt token + 512 <= 8192 → accepted
    serve._handle_completions(
        {"messages": [{"role": "user", "content": "hello"}]},
        _FakeTok(), "MODEL", ceiling=8192, kv_bits=None, hf_id="some/model",
    )
    assert captured["max_tokens"] == 512


def test_handle_completions_passes_exact_allowed_max_tokens_to_generate(monkeypatch):
    """mlx_lm must receive governed_max_tokens, not the raw requested value."""
    captured = _fake_mlx_generate(monkeypatch)
    # "test" → 1 token; requested=99; 1+99=100 == ceiling → accepted; allowed=99
    serve._handle_completions(
        {"messages": [{"role": "user", "content": "test"}], "max_tokens": 99},
        _FakeTok(), "MODEL", ceiling=100, kv_bits=None, hf_id="some/model",
    )
    assert captured["max_tokens"] == 99


def test_handle_completions_passes_kv_kwargs_when_quantized(monkeypatch):
    captured = _fake_mlx_generate(monkeypatch)
    serve._handle_completions(
        {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 10},
        _FakeTok(), "MODEL", ceiling=8192, kv_bits=4, hf_id="some/model",
    )
    assert captured["kw"]["kv_bits"] == 4
    assert captured["kw"]["kv_group_size"] == 64
    assert captured["kw"]["quantized_kv_start"] == 5000


def test_handle_completions_no_kv_kwargs_for_fp16(monkeypatch):
    captured = _fake_mlx_generate(monkeypatch)
    serve._handle_completions(
        {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 10},
        _FakeTok(), "MODEL", ceiling=8192, kv_bits=None, hf_id="some/model",
    )
    assert "kv_bits" not in captured["kw"]


def test_handle_completions_rejects_when_prompt_alone_fills_ceiling(monkeypatch):
    _fake_mlx_generate(monkeypatch)
    # "a b c" → 3 tokens; ceiling=3 → prompt_tokens >= ceiling → 400
    status, body = serve._handle_completions(
        {"messages": [{"role": "user", "content": "a b c"}], "max_tokens": 1},
        _FakeTok(), "MODEL", ceiling=3, kv_bits=None, hf_id="some/model",
    )
    assert status == 400
    assert body["error"]["type"] == "context_length_exceeded"


# --------------------------------------------------------------------------- #
# _handle_completions — tool-calling
# --------------------------------------------------------------------------- #

def test_handle_completions_returns_tool_calls_when_marker_present(monkeypatch):
    """When the model emits tool markers, the response uses the tool_calls shape."""
    tool_text = "<tool_call>some args</tool_call>"
    _fake_mlx_generate(monkeypatch, completion=tool_text)

    tools = [{"type": "function", "function": {"name": "f"}}]
    status, body = serve._handle_completions(
        {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 10, "tools": tools},
        _ToolTok(), "MODEL", ceiling=100, kv_bits=None, hf_id="some/model",
    )
    assert status == 200
    assert body["object"] == "chat.completion"
    choices = body["choices"]
    assert len(choices) == 1
    assert choices[0]["finish_reason"] == "tool_calls"
    msg = choices[0]["message"]
    assert msg["role"] == "assistant"
    assert msg["content"] is None
    assert len(msg["tool_calls"]) == 1
    tc = msg["tool_calls"][0]
    assert tc["type"] == "function"
    assert tc["id"].startswith("call_")
    assert tc["function"]["name"] == "f"
    assert json.loads(tc["function"]["arguments"]) == {"x": 1}


def test_handle_completions_plain_text_when_no_tool_marker(monkeypatch):
    """Without a tool marker in the output, response is a normal text completion."""
    _fake_mlx_generate(monkeypatch, completion="plain answer")

    tools = [{"type": "function", "function": {"name": "f"}}]
    status, body = serve._handle_completions(
        {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 10, "tools": tools},
        _ToolTok(), "MODEL", ceiling=100, kv_bits=None, hf_id="some/model",
    )
    assert status == 200
    assert body["choices"][0]["message"]["content"] == "plain answer"
    assert body["choices"][0]["finish_reason"] == "stop"


def test_handle_completions_plain_text_when_no_tools_provided(monkeypatch):
    """Without tools in the request, tool-calling path is never entered."""
    _fake_mlx_generate(monkeypatch, completion="answer")

    status, body = serve._handle_completions(
        {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 10},
        _ToolTok(), "MODEL", ceiling=100, kv_bits=None, hf_id="some/model",
    )
    assert status == 200
    assert body["choices"][0]["message"]["content"] == "answer"
    assert body["choices"][0]["finish_reason"] == "stop"


def test_handle_completions_plain_text_when_tokenizer_lacks_tool_calling(monkeypatch):
    """_FakeTok has no has_tool_calling — tool markers in output are treated as plain text."""
    tool_text = "<tool_call>args</tool_call>"
    _fake_mlx_generate(monkeypatch, completion=tool_text)

    tools = [{"type": "function", "function": {"name": "f"}}]
    status, body = serve._handle_completions(
        {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 10, "tools": tools},
        _FakeTok(), "MODEL", ceiling=100, kv_bits=None, hf_id="some/model",
    )
    assert status == 200
    assert body["choices"][0]["message"]["content"] == tool_text
    assert body["choices"][0]["finish_reason"] == "stop"


# --------------------------------------------------------------------------- #
# _stream_completions — SSE streaming
# --------------------------------------------------------------------------- #

def test_stream_completions_yields_delta_chunks_and_done(monkeypatch):
    """Happy path: each non-final response emits a data: chunk; [DONE] closes the stream."""
    responses = [
        _FakeStreamResponse("Hello", None),
        _FakeStreamResponse(" world", None),
        _FakeStreamResponse("", "stop"),
    ]
    _fake_mlx_stream_generate(monkeypatch, responses)

    handler = _FakeHandler()
    result = serve._stream_completions(
        {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 10},
        _FakeTok(), "MODEL", ceiling=100, kv_bits=None, hf_id="some/model",
        handler=handler,
    )

    assert result is None  # streaming started — no error tuple returned
    assert handler._response_code == 200
    assert handler._headers["Content-Type"] == "text/event-stream"
    assert handler._headers["Cache-Control"] == "no-cache"
    assert handler._end_headers_called

    output = handler.wfile.getvalue().decode()
    data_lines = [ln for ln in output.split("\n") if ln.startswith("data:")]

    # Content chunks
    content_lines = [ln for ln in data_lines if ln != "data: [DONE]"]
    contents = []
    for ln in content_lines:
        chunk = json.loads(ln[len("data: "):])
        delta = chunk["choices"][0]["delta"]
        if "content" in delta:
            contents.append(delta["content"])

    assert "Hello" in contents
    assert " world" in contents

    # Final chunk: delta={}, finish_reason set
    final_chunk = json.loads(content_lines[-1][len("data: "):])
    assert final_chunk["choices"][0]["delta"] == {}
    assert final_chunk["choices"][0]["finish_reason"] == "stop"

    # Sentinel
    assert "data: [DONE]" in data_lines


def test_stream_completions_finish_reason_propagated(monkeypatch):
    """finish_reason from the last GenerationResponse reaches the final SSE chunk."""
    responses = [
        _FakeStreamResponse("tok", None),
        _FakeStreamResponse("", "length"),
    ]
    _fake_mlx_stream_generate(monkeypatch, responses)

    handler = _FakeHandler()
    serve._stream_completions(
        {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 5},
        _FakeTok(), "MODEL", ceiling=100, kv_bits=None, hf_id="some/model",
        handler=handler,
    )

    output = handler.wfile.getvalue().decode()
    data_lines = [ln for ln in output.split("\n")
                  if ln.startswith("data:") and ln != "data: [DONE]"]
    final_chunk = json.loads(data_lines[-1][len("data: "):])
    assert final_chunk["choices"][0]["finish_reason"] == "length"


def test_stream_completions_governance_reject_returns_error_tuple(monkeypatch):
    """When governance fails, return (400, error) and write NO SSE at all."""
    _fake_mlx_stream_generate(monkeypatch, [])  # must not be called

    handler = _FakeHandler()
    # "hello world" → 2 tokens; requested=500 → need=502 > ceiling=400
    result = serve._stream_completions(
        {"messages": [{"role": "user", "content": "hello world"}], "max_tokens": 500},
        _FakeTok(), "MODEL", ceiling=400, kv_bits=None, hf_id="some/model",
        handler=handler,
    )

    assert result is not None
    status, err = result
    assert status == 400
    assert err["error"]["type"] == "context_length_exceeded"
    # No SSE headers or body were written
    assert handler._response_code is None
    assert handler.wfile.getvalue() == b""


def test_stream_completions_passes_kv_kwargs(monkeypatch):
    """KV-quantization kwargs are forwarded to stream_generate."""
    received: dict = {}

    def fake_stream_generate(model, tok, prompt, max_tokens=None, **kw):
        received.update(kw)
        yield _FakeStreamResponse("ok", "stop")

    monkeypatch.setitem(
        sys.modules, "mlx_lm",
        SimpleNamespace(stream_generate=fake_stream_generate, generate=None, load=None),
    )

    handler = _FakeHandler()
    serve._stream_completions(
        {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 10},
        _FakeTok(), "MODEL", ceiling=100, kv_bits=4, hf_id="some/model",
        handler=handler,
    )

    assert received["kv_bits"] == 4
    assert received["kv_group_size"] == 64
    assert received["quantized_kv_start"] == 5000


def test_stream_completions_model_id_in_chunks(monkeypatch):
    """Every SSE chunk carries the hf_id as the model field."""
    _fake_mlx_stream_generate(monkeypatch, [
        _FakeStreamResponse("x", None),
        _FakeStreamResponse("", "stop"),
    ])

    handler = _FakeHandler()
    serve._stream_completions(
        {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 10},
        _FakeTok(), "MODEL", ceiling=100, kv_bits=None, hf_id="org/my-model",
        handler=handler,
    )

    output = handler.wfile.getvalue().decode()
    for ln in output.split("\n"):
        if ln.startswith("data:") and ln != "data: [DONE]":
            chunk = json.loads(ln[len("data: "):])
            assert chunk["model"] == "org/my-model"


# --------------------------------------------------------------------------- #
# serve() — integration: gate veto + ready handshake (no real sockets/weights)
# --------------------------------------------------------------------------- #

def test_serve_prints_refusal_json_and_exits_1_when_gate_vetoes(monkeypatch, capsys):
    monkeypatch.setattr(
        serve, "_pre_load_gate",
        lambda *a, **k: ({"refused": True, "reason": "too big for budget"}, None),
    )
    # If load is reached the test explodes — proving Rule #1 holds
    monkeypatch.setitem(sys.modules, "mlx_lm", SimpleNamespace(
        load=lambda *a: (_ for _ in ()).throw(AssertionError("weights loaded — gate failed!")),
    ))
    with pytest.raises(SystemExit) as exc:
        serve.serve("big/model", 4096, margin_gb=4.0, overhead_gb=1.0, port=8080)
    assert exc.value.code == 1
    out = capsys.readouterr().out.strip()
    data = json.loads(out)
    assert data == {"refused": True, "reason": "too big for budget"}


def test_serve_prints_ready_json_after_successful_gate_and_load(monkeypatch, capsys):
    monkeypatch.setattr(serve, "_pre_load_gate", lambda *a, **k: (None, None))
    monkeypatch.setitem(
        sys.modules, "mlx_lm",
        SimpleNamespace(load=lambda hf: ("MODEL", _FakeTok())),
    )

    class _FakeServer:
        def __init__(self, addr, handler_class): pass
        def serve_forever(self): pass  # don't block

    monkeypatch.setattr(serve, "HTTPServer", _FakeServer)

    serve.serve("safe/model", 4096, margin_gb=4.0, overhead_gb=1.0, port=9001)
    out = capsys.readouterr().out.strip()
    data = json.loads(out)
    assert data == {"ready": True, "url": "http://127.0.0.1:9001", "context": 4096}


def test_serve_ready_json_reflects_port_and_context(monkeypatch, capsys):
    monkeypatch.setattr(serve, "_pre_load_gate", lambda *a, **k: (None, None))
    monkeypatch.setitem(
        sys.modules, "mlx_lm",
        SimpleNamespace(load=lambda hf: ("MODEL", _FakeTok())),
    )

    class _FakeServer:
        def __init__(self, addr, handler_class): pass
        def serve_forever(self): pass

    monkeypatch.setattr(serve, "HTTPServer", _FakeServer)

    serve.serve("safe/model", 8192, margin_gb=4.0, overhead_gb=1.0, port=11434)
    out = capsys.readouterr().out.strip()
    data = json.loads(out)
    assert data["url"] == "http://127.0.0.1:11434"
    assert data["context"] == 8192


def test_serve_calls_serve_forever(monkeypatch):
    """serve() must reach serve_forever() — not return early after printing ready."""
    monkeypatch.setattr(serve, "_pre_load_gate", lambda *a, **k: (None, None))
    monkeypatch.setitem(
        sys.modules, "mlx_lm",
        SimpleNamespace(load=lambda hf: ("MODEL", _FakeTok())),
    )

    served = []

    class _FakeServer:
        def __init__(self, addr, handler_class): pass
        def serve_forever(self): served.append(True)

    monkeypatch.setattr(serve, "HTTPServer", _FakeServer)

    serve.serve("safe/model", 4096, margin_gb=4.0, overhead_gb=1.0, port=9000)
    assert served, "serve_forever was never called"


# --------------------------------------------------------------------------- #
# main() — CLI argument parsing wires through to serve()
# --------------------------------------------------------------------------- #

def test_main_threads_args_to_serve(monkeypatch):
    seen: dict = {}
    monkeypatch.setattr(
        serve, "serve",
        lambda hf, ctx, *, margin_gb, overhead_gb, port, kv_bits=None:
        seen.update(hf=hf, ctx=ctx, margin_gb=margin_gb,
                    overhead_gb=overhead_gb, port=port, kv_bits=kv_bits),
    )
    serve.main(["my/model", "4096", "--margin", "4", "--overhead", "1", "--port", "8080"])
    assert seen == {
        "hf": "my/model", "ctx": 4096, "margin_gb": 4.0,
        "overhead_gb": 1.0, "port": 8080, "kv_bits": None,
    }


def test_main_threads_kv_bits_to_serve(monkeypatch):
    seen: dict = {}
    monkeypatch.setattr(serve, "serve", lambda *a, **k: seen.update(k))
    serve.main(["my/model", "4096", "--margin", "4", "--overhead", "1",
                "--port", "8080", "--kv-bits", "8"])
    assert seen["kv_bits"] == 8


# --------------------------------------------------------------------------- #
# do_POST helper — invoke do_POST without a real socket
# --------------------------------------------------------------------------- #

def _run_do_post(monkeypatch, body_bytes, *, path="/v1/chat/completions",
                 content_length=None, model="MODEL", tokenizer=None,
                 ceiling=100, kv_bits=None, hf_id="test/model"):
    """Create a handler instance via __new__, wire fake I/O, call do_POST, return (status, raw)."""
    if tokenizer is None:
        tokenizer = _FakeTok()
    handler_cls = serve._make_handler(model, tokenizer, ceiling, kv_bits, hf_id)
    inst = handler_cls.__new__(handler_cls)

    cl = content_length if content_length is not None else len(body_bytes)
    inst.path = path
    inst.headers = {"Content-Length": str(cl)}
    inst.rfile = io.BytesIO(body_bytes)

    wfile = io.BytesIO()
    inst.wfile = wfile
    status_holder = [None]
    resp_headers: dict = {}

    inst.send_response = lambda code: status_holder.__setitem__(0, code)
    inst.send_header = lambda k, v: resp_headers.__setitem__(k, v)
    inst.end_headers = lambda: None

    inst.do_POST()
    return status_holder[0], wfile.getvalue()


# --------------------------------------------------------------------------- #
# Fix 1 — streaming truncation: final yield with text + finish_reason is emitted
# --------------------------------------------------------------------------- #

def test_stream_completions_final_text_with_finish_reason_is_emitted(monkeypatch):
    """Last stream yield (text='!', finish_reason='stop') must appear in SSE content."""
    responses = [
        _FakeStreamResponse("Hello", None),
        _FakeStreamResponse("!", "stop"),  # text AND finish_reason — must NOT be dropped
    ]
    _fake_mlx_stream_generate(monkeypatch, responses)

    handler = _FakeHandler()
    serve._stream_completions(
        {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 10},
        _FakeTok(), "MODEL", ceiling=100, kv_bits=None, hf_id="test/model",
        handler=handler,
    )

    output = handler.wfile.getvalue().decode()
    all_content = []
    for ln in output.split("\n"):
        if ln.startswith("data:") and ln != "data: [DONE]":
            chunk = json.loads(ln[len("data: "):])
            delta = chunk["choices"][0]["delta"]
            if "content" in delta:
                all_content.append(delta["content"])

    assert "!" in all_content, "Final text from last stream chunk was dropped"


# --------------------------------------------------------------------------- #
# Fix 2 — [DONE] sent even when stream_generate raises mid-stream
# --------------------------------------------------------------------------- #

def test_stream_completions_sends_done_on_mid_stream_exception(monkeypatch):
    """A mid-stream exception must still write data: [DONE] before propagating."""

    def bad_stream(model, tok, prompt, max_tokens=None, **kw):
        yield _FakeStreamResponse("partial", None)
        raise RuntimeError("GPU OOM")

    monkeypatch.setitem(
        sys.modules, "mlx_lm",
        SimpleNamespace(stream_generate=bad_stream, generate=None, load=None),
    )

    handler = _FakeHandler()
    with pytest.raises(RuntimeError, match="GPU OOM"):
        serve._stream_completions(
            {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 10},
            _FakeTok(), "MODEL", ceiling=100, kv_bits=None, hf_id="test/model",
            handler=handler,
        )

    output = handler.wfile.getvalue().decode()
    assert "data: [DONE]" in output


# --------------------------------------------------------------------------- #
# Fix 3 — first SSE chunk carries delta.role == "assistant"
# --------------------------------------------------------------------------- #

def test_stream_completions_first_chunk_has_role_assistant(monkeypatch):
    """First SSE chunk must carry delta.role == 'assistant' (OpenAI shape)."""
    _fake_mlx_stream_generate(monkeypatch, [
        _FakeStreamResponse("hello", None),
        _FakeStreamResponse("", "stop"),
    ])

    handler = _FakeHandler()
    serve._stream_completions(
        {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 10},
        _FakeTok(), "MODEL", ceiling=100, kv_bits=None, hf_id="test/model",
        handler=handler,
    )

    output = handler.wfile.getvalue().decode()
    data_lines = [ln for ln in output.split("\n")
                  if ln.startswith("data:") and ln != "data: [DONE]"]
    first_chunk = json.loads(data_lines[0][len("data: "):])
    assert first_chunk["choices"][0]["delta"].get("role") == "assistant"


# --------------------------------------------------------------------------- #
# Fix 4 — negative / zero max_tokens rejected before generation
# --------------------------------------------------------------------------- #

def test_handle_completions_rejects_negative_max_tokens(monkeypatch):
    _fake_mlx_generate(monkeypatch)  # must NOT be called
    status, body = serve._handle_completions(
        {"messages": [{"role": "user", "content": "hi"}], "max_tokens": -1},
        _FakeTok(), "MODEL", ceiling=100, kv_bits=None, hf_id="test/model",
    )
    assert status == 400
    assert body["error"]["type"] == "invalid_request_error"
    assert "max_tokens" in body["error"]["message"]


def test_handle_completions_rejects_zero_max_tokens(monkeypatch):
    _fake_mlx_generate(monkeypatch)  # must NOT be called
    status, body = serve._handle_completions(
        {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 0},
        _FakeTok(), "MODEL", ceiling=100, kv_bits=None, hf_id="test/model",
    )
    assert status == 400
    assert body["error"]["type"] == "invalid_request_error"


def test_stream_completions_rejects_negative_max_tokens(monkeypatch):
    """Streaming: max_tokens <= 0 returns (400, error) before any SSE output."""
    _fake_mlx_stream_generate(monkeypatch, [])  # must NOT be called
    handler = _FakeHandler()
    result = serve._stream_completions(
        {"messages": [{"role": "user", "content": "hi"}], "max_tokens": -1},
        _FakeTok(), "MODEL", ceiling=100, kv_bits=None, hf_id="test/model",
        handler=handler,
    )
    assert result is not None
    status, body = result
    assert status == 400
    assert body["error"]["type"] == "invalid_request_error"
    assert handler._response_code is None  # no SSE headers written


# --------------------------------------------------------------------------- #
# Fix 5 — Content-Length > 1MB → 413 before reading body
# --------------------------------------------------------------------------- #

def test_do_post_returns_413_when_content_length_exceeds_limit(monkeypatch):
    _fake_mlx_generate(monkeypatch)
    status, raw = _run_do_post(
        monkeypatch, b"ignored",
        content_length=1_048_577,
    )
    assert status == 413


# --------------------------------------------------------------------------- #
# Fix 6 — non-dict JSON body → 400
# --------------------------------------------------------------------------- #

def test_do_post_returns_400_for_non_dict_body(monkeypatch):
    """JSON body that is a list (not a dict) → 400 invalid_request_error."""
    _fake_mlx_generate(monkeypatch)
    body_bytes = json.dumps([]).encode()
    status, raw = _run_do_post(monkeypatch, body_bytes)
    assert status == 400
    resp = json.loads(raw)
    assert resp["error"]["type"] == "invalid_request_error"


# --------------------------------------------------------------------------- #
# Fix 7 — multiple tool_call blocks → multiple tool_calls; unterminated → text
# --------------------------------------------------------------------------- #

class _MultiToolTok(_FakeTok):
    """Tokenizer stub that parses multiple tool blocks; name encodes call order."""

    has_tool_calling = True
    tool_call_start = "<tool_call>"
    tool_call_end = "</tool_call>"
    _counter = 0

    def apply_chat_template(self, messages, *, tokenize=False, add_generation_prompt=True,
                             tools=None):
        return " ".join(m.get("content", "") for m in messages)

    def tool_parser(self, text, tools):  # noqa: ARG002
        self._counter += 1
        return {"name": f"fn{self._counter}", "arguments": {"n": self._counter}}


def test_handle_completions_parses_multiple_tool_call_blocks(monkeypatch):
    """Two sequential <tool_call> blocks → two entries in tool_calls."""
    text = "<tool_call>args1</tool_call><tool_call>args2</tool_call>"
    _fake_mlx_generate(monkeypatch, completion=text)

    tok = _MultiToolTok()
    tools = [{"type": "function", "function": {"name": "fn1"}},
             {"type": "function", "function": {"name": "fn2"}}]
    status, body = serve._handle_completions(
        {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 10, "tools": tools},
        tok, "MODEL", ceiling=100, kv_bits=None, hf_id="test/model",
    )
    assert status == 200
    assert body["choices"][0]["finish_reason"] == "tool_calls"
    assert len(body["choices"][0]["message"]["tool_calls"]) == 2


def test_handle_completions_unterminated_tool_marker_falls_through_to_text(monkeypatch):
    """tool_call_start present but tool_call_end absent → plain-text fallback, no crash."""
    text = "<tool_call>no closing tag"
    _fake_mlx_generate(monkeypatch, completion=text)

    tools = [{"type": "function", "function": {"name": "f"}}]
    status, body = serve._handle_completions(
        {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 10, "tools": tools},
        _ToolTok(), "MODEL", ceiling=100, kv_bits=None, hf_id="test/model",
    )
    assert status == 200
    assert body["choices"][0]["message"]["content"] == text
    assert body["choices"][0]["finish_reason"] == "stop"
