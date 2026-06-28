# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""Governed one-shot text generation — the engine-side generate verb for ARA's `run`.

Same Safety-first (Rule #1) discipline as ``measure_one``: describe the model and run the
existing L4 safety gate *before* loading any weights. An unknown/non-causal model, or a
gate that vetoes, refuses with no load. The gate runs at the *effective* context the
one-shot will actually reach — ``min(ctx, prompt_tokens + max_tokens)`` — because MLX
grows its KV cache dynamically; the ceiling ``ctx`` stays the hard cap. Prompt tokens are
counted with the tokenizer only (no weights). Only a safe (model, ctx) reaches mlx_lm. The
prompt is read from **stdin**, never argv. Emits one JSON line:

    success: {"context": <int>, "completion": "<generated text>"}
    refused: {"context": <int>, "refused": true, "reason": "<why>"}

Usage:
    python -m wmx_suite.generate <hf_id> <ctx> --margin G --overhead G --max-tokens N
    (prompt on stdin)
"""
from __future__ import annotations

import argparse
import json
import sys

from . import measure_one, models, system

# reuse measure_one's canonical refusal shape for consistency across workers
_refused = measure_one._refused


def _count_prompt_tokens(hf_id: str, prompt: str) -> int:
    """Tokenize *prompt* to a count WITHOUT loading model weights.

    ``transformers.AutoTokenizer.from_pretrained`` reads only the tokenizer artifacts
    from the HF cache — it never touches the weight tensors — so this preserves the
    refuse-before-load property (Rule #1). Lazy-imported like the ``mlx_lm`` import below
    so the module imports without transformers installed and tests can monkeypatch it.
    """
    if not prompt:
        return 0
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(hf_id)
    return len(tok.encode(prompt))


def generate(hf_id: str, ctx: int, *, prompt: str, margin_gb: float,
             overhead_gb: float, max_tokens: int, kv_bits: int | None = None) -> dict:
    """Gate then (if safe) load + generate; return the canonical result dict.

    Refuses before loading if the model is unknown/non-causal or if the shared safety
    gate predicts the footprint at the *effective* context would reach the safe budget.

    ``ctx`` is the characterized ceiling — the hard cap we never gate or generate beyond.
    But MLX grows its KV cache dynamically, so a one-shot from a short prompt only reaches
    ``prompt_tokens + max_tokens`` of context, not the full ceiling. We therefore gate on
    the *effective* context the run will actually reach, capped at the ceiling::

        effective_ctx = min(ctx, prompt_tokens + max_tokens)

    Gating the raw ceiling here would over-predict memory and refuse runs that
    ``characterize`` already certified safe. The reported context stays the ceiling.
    """
    info = models.describe(hf_id)
    if info is None:
        return _refused(ctx, f"model not found in HF cache: {hf_id}")
    if not info.is_causal:
        return _refused(ctx, f"{hf_id} is not a supported causal language model")

    # Count prompt tokens without loading weights (tokenizer only), then gate on the
    # effective context this one-shot will actually reach — capped at the ceiling.
    prompt_tokens = _count_prompt_tokens(hf_id, prompt)
    effective_ctx = min(ctx, prompt_tokens + max_tokens)

    # fp16 (None) unless the cache type can quantize — keeps run consistent with characterize
    # and never quantizes a RotatingKVCache model (which would crash past the quant threshold).
    kv_bits = measure_one._effective_kv_bits(info, kv_bits)

    limits = system.read_limits()
    live_base = system.sample_settled_baseline()
    reason = measure_one.safety_gate(info, limits, effective_ctx, margin_gb=margin_gb,
                                     overhead_gb=overhead_gb, live_base=live_base,
                                     kv_bits=kv_bits)
    if reason is not None:
        return _refused(ctx, reason)

    # Lazy import (like probe_worker) so the module imports without mlx installed and
    # tests can monkeypatch sys.modules["mlx_lm"].
    from mlx_lm import generate as mlx_generate, load

    # Match production quant knobs when quantizing; pass nothing for fp16 (mlx_lm default).
    kv_kwargs = ({} if kv_bits is None
                 else {"kv_bits": kv_bits, "kv_group_size": 64, "quantized_kv_start": 5000})
    try:
        model, tok = load(hf_id)
    except Exception as exc:
        first = str(exc).splitlines()[0][:200] if str(exc) else ""
        return _refused(ctx, f"failed to load {hf_id}: {type(exc).__name__}: {first}")
    text = mlx_generate(model, tok, prompt=prompt, max_tokens=max_tokens, **kv_kwargs)
    return {"context": ctx, "completion": text}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Governed one-shot text generation.")
    ap.add_argument("hf_id")
    ap.add_argument("ctx", type=int)
    ap.add_argument("--margin", type=float, required=True)
    ap.add_argument("--overhead", type=float, required=True)
    ap.add_argument("--max-tokens", type=int, required=True)
    ap.add_argument("--kv-bits", type=int, default=None,
                    help="KV-cache quantization bits (8 or 4); omit for fp16. Ignored for "
                         "non-quantizable (sliding-window) models.")
    args = ap.parse_args(argv)
    prompt = sys.stdin.read()
    result = generate(args.hf_id, args.ctx, prompt=prompt, margin_gb=args.margin,
                      overhead_gb=args.overhead, max_tokens=args.max_tokens,
                      kv_bits=args.kv_bits)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
