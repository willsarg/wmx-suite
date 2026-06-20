"""Self-vetoing single-measurement primitive — the engine-side hard veto (L4).

ARA's thin path drives the ramp, but crash-prevention is checked at every layer. This is
the engine's: given a model and one context, it reads the *live* wall, estimates the
footprint conservatively, and **refuses before loading anything** if the base alone — or
the prediction at that context — would reach the safe budget. Only when safe does it spawn
the isolated probe worker (one fresh process, so wired residue never contaminates the
reading). It emits a single canonical JSON object:

    safe:    {"context": <int>, "mem_gb": <os-wired high-water>}
    refused: {"context": <int>, "refused": true, "reason": "<why>"}

Usage: ``python -m wmx_suite.measure_one <hf_id> <ctx> --margin G --overhead G``
"""
from __future__ import annotations

import argparse
import json
import sys

from . import models, probe, profiles, system


def safety_gate(info, limits, ctx: int, *, margin_gb: float, overhead_gb: float,
                live_base: float) -> str | None:
    """Return a refusal reason if probing (model, ctx) is unsafe, else None.

    Two independent refusals, both conservative (``>=`` — never round toward the wall):
    the base estimate must fit on its own (the model can load), and the predicted
    footprint at *ctx* (live baseline + model base + a-priori slope) must stay under budget.
    """
    threshold = limits.safe_threshold_gb(margin_gb)
    est_base = probe.estimate_base_gb(info, limits, overhead_gb)
    if est_base >= threshold:
        return (f"base estimate {est_base:.2f}GB >= safe budget {threshold:.2f}GB — "
                f"won't load")
    model_base = info.weights_gb * profiles.DEFAULT_RESIDENT_FACTOR + overhead_gb
    predicted = live_base + model_base + info.estimated_slope_gb_per_k() * (ctx / 1000)
    if predicted >= threshold:
        return (f"predicted {predicted:.2f}GB at {ctx} tok >= safe budget "
                f"{threshold:.2f}GB")
    return None


def preflight(hf_id: str, *, margin_gb: float, overhead_gb: float) -> dict:
    """No-load estimate for ARA's scheduler: the context→0 base, a-priori slope, budget.

    ARA owns the ramp methodology but not model knowledge, so the engine supplies the
    facts (from the model config + the live wall); ARA fits and solves on top.
    """
    info = models.describe(hf_id)
    if info is None:
        return {"error": f"model not found in HF cache: {hf_id}"}
    if not info.is_causal:
        return {"error": f"{hf_id} is not a supported causal language model"}
    limits = system.read_limits()
    live_base = system.sample_settled_baseline()
    model_base = info.weights_gb * profiles.DEFAULT_RESIDENT_FACTOR + overhead_gb
    return {
        "base_gb": round(live_base + model_base, 4),
        "slope_gb_per_k": info.estimated_slope_gb_per_k(),
        "budget_gb": limits.safe_threshold_gb(margin_gb),
        "max_context": info.max_context,
    }


def _spawn_worker(hf_id: str, ctx: int, kv_bits, abort_wired_gb) -> dict:
    """Run the isolated probe worker in this interpreter; return its raw result dict.

    The worker is handed the hard wired limit (the safe budget) so its own watchdog (L5)
    can abort if the live footprint reaches it despite the pre-flight gate.
    """
    return probe._run_worker(sys.executable, hf_id, ctx, kv_bits,
                             abort_wired_gb=abort_wired_gb)


def _refused(ctx: int, reason: str) -> dict:
    return {"context": ctx, "refused": True, "reason": reason}


def run(hf_id: str, ctx: int, *, margin_gb: float, overhead_gb: float) -> dict:
    """Gate then (if safe) measure; return the canonical result dict."""
    info = models.describe(hf_id)
    if info is None:
        return _refused(ctx, f"model not found in HF cache: {hf_id}")
    if not info.is_causal:
        return _refused(ctx, f"{hf_id} is not a supported causal language model")

    limits = system.read_limits()
    live_base = system.sample_settled_baseline()
    reason = safety_gate(info, limits, ctx, margin_gb=margin_gb,
                         overhead_gb=overhead_gb, live_base=live_base)
    if reason is not None:
        return _refused(ctx, reason)

    kv_bits = 4 if info.can_quantize_kv else None
    threshold = limits.safe_threshold_gb(margin_gb)
    raw = _spawn_worker(hf_id, ctx, kv_bits, abort_wired_gb=threshold)
    if raw.get("status") != "ok":
        return _refused(ctx, f"probe failed: {raw.get('note', 'no output')}")
    return {"context": ctx, "mem_gb": raw["os_wired_gb"]}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Safe single-context memory measurement.")
    ap.add_argument("hf_id")
    ap.add_argument("ctx", type=int)
    ap.add_argument("--margin", type=float, required=True)
    ap.add_argument("--overhead", type=float, required=True)
    ap.add_argument("--preflight", action="store_true",
                    help="print the no-load estimate (base/slope/budget) and exit")
    args = ap.parse_args(argv)
    if args.preflight:
        result = preflight(args.hf_id, margin_gb=args.margin, overhead_gb=args.overhead)
    else:
        result = run(args.hf_id, args.ctx, margin_gb=args.margin, overhead_gb=args.overhead)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
