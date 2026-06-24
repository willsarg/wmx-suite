# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""Governed one-shot text generation — the engine-side generate verb for ARA's `run`.

Same Safety-first (Rule #1) discipline as ``measure_one``: describe the model and run the
existing L4 safety gate *before* loading anything. An unknown/non-causal model, or a gate
that vetoes at the requested context, refuses with no load. Only a safe (model, ctx)
reaches mlx_lm. The prompt is read from **stdin**, never argv. Emits one JSON line:

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


def generate(hf_id: str, ctx: int, *, prompt: str, margin_gb: float,
             overhead_gb: float, max_tokens: int) -> dict:
    """Gate then (if safe) load + generate; return the canonical result dict.

    Refuses before loading if the model is unknown/non-causal or if the shared safety
    gate predicts the footprint at *ctx* would reach the safe budget.
    """
    info = models.describe(hf_id)
    if info is None:
        return _refused(ctx, f"model not found in HF cache: {hf_id}")
    if not info.is_causal:
        return _refused(ctx, f"{hf_id} is not a supported causal language model")

    limits = system.read_limits()
    live_base = system.sample_settled_baseline()
    reason = measure_one.safety_gate(info, limits, ctx, margin_gb=margin_gb,
                                     overhead_gb=overhead_gb, live_base=live_base)
    if reason is not None:
        return _refused(ctx, reason)

    # Lazy import (like probe_worker) so the module imports without mlx installed and
    # tests can monkeypatch sys.modules["mlx_lm"].
    from mlx_lm import generate as mlx_generate, load

    model, tok = load(hf_id)
    text = mlx_generate(model, tok, prompt=prompt, max_tokens=max_tokens)
    return {"context": ctx, "completion": text}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Governed one-shot text generation.")
    ap.add_argument("hf_id")
    ap.add_argument("ctx", type=int)
    ap.add_argument("--margin", type=float, required=True)
    ap.add_argument("--overhead", type=float, required=True)
    ap.add_argument("--max-tokens", type=int, required=True)
    args = ap.parse_args(argv)
    prompt = sys.stdin.read()
    result = generate(args.hf_id, args.ctx, prompt=prompt, margin_gb=args.margin,
                      overhead_gb=args.overhead, max_tokens=args.max_tokens)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
