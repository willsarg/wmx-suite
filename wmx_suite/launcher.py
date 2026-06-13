"""Safe launch planning for mlx_lm.generate.

Two failure modes a naive launcher hits, and how we avoid them:
  1. Forcing `--kv-bits 4` on every model crashes RotatingKVCache models (Gemma,
     GPT-OSS) past 5000 tokens. We quantize ONLY models whose cache supports it.
  2. Budgeting against TOTAL RAM ignores the real wired wall (17.18 GB) and the prefill
     spike. We budget against the measured per-model curve and the live system baseline,
     refusing to launch if the model can't even load without breaching the wall.
"""
from __future__ import annotations

from dataclasses import dataclass

from . import db, models
from .probe import FIXED_OVERHEAD_GB, RESIDENT_FACTOR
from .system import read_limits, sample_settled_baseline

# Measured: the OS-wired slope is dominated by the prefill transient, ~5x the analytic
# fp16 KV-cache slope (Gemma: analytic 0.0143 vs measured 0.070 GB/1k). For UNCHARACTERIZED
# models we apply this multiplier so the estimated cap stays conservative.
PREFILL_SPIKE_MULT = 5.0
MIN_USEFUL_CTX = 512


def _estimated_slope_gb_per_k(info: models.ModelInfo) -> float:
    """Conservative os-wired slope estimate for an uncharacterized model (GB per 1k tokens)."""
    kv_slope = info.fp16_kv_bytes_per_token() * 1000 / 1e9  # fp16 steady-state KV
    return kv_slope * PREFILL_SPIKE_MULT


@dataclass(frozen=True)
class Prediction:
    base_abs_gb: float     # live baseline + model's own footprint at context -> 0
    headroom_gb: float     # threshold - base_abs (negative => won't even load safely)
    breaches_wall: bool    # base_abs alone crosses the hard wall — can't load at all
    safe_ctx: int          # max context before predicted peak hits the threshold


def predict(*, model_base_gb: float, slope_gb_per_k: float, live_base_gb: float,
            threshold_gb: float, wall_gb: float, model_max: int | None) -> Prediction:
    """Single source of truth for the crash-prediction math, shared by `plan` and `health`.

    Given a model's curve (base footprint + slope) and the live system baseline, work out
    the absolute base load and the safe context ceiling under the threshold. Keeping this
    in one place means `health`'s verdict can never drift from what `run` actually does.
    """
    base_abs = live_base_gb + model_base_gb
    headroom = threshold_gb - base_abs
    if slope_gb_per_k > 0:
        cap = int(max(0.0, headroom / slope_gb_per_k) * 1000)
    else:
        cap = model_max or 0
    if model_max:
        cap = min(cap, model_max)
    return Prediction(base_abs_gb=base_abs, headroom_gb=headroom,
                      breaches_wall=base_abs >= wall_gb, safe_ctx=cap)


def plan(hf_id: str, *, margin_gb: float = 2.0) -> dict:
    """Decide kv_bits and a safe --max-kv-size for a launch, or refuse."""
    info = models.describe(hf_id)
    if info is None:
        return {"error": f"model not found in HF cache: {hf_id}"}

    limits = read_limits()
    threshold = limits.safe_threshold_gb(margin_gb)
    wall = limits.wall_gb
    live_base = sample_settled_baseline()
    kv_bits = 4 if info.can_quantize_kv else None  # fix #1: quantize only quantizable caches

    con = db.connect()
    fit = db.latest_fit(con, hf_id)
    if fit and fit.get("slope_gb_per_k"):
        model_base = float(fit["model_base_gb"])
        slope = float(fit["slope_gb_per_k"])
        source = "measured"
    else:
        model_base = info.weights_gb * RESIDENT_FACTOR + FIXED_OVERHEAD_GB
        slope = _estimated_slope_gb_per_k(info)
        source = "estimated"

    pred = predict(model_base_gb=model_base, slope_gb_per_k=slope, live_base_gb=live_base,
                   threshold_gb=threshold, wall_gb=wall, model_max=info.max_context)
    p = {
        "hf_id": hf_id, "kv_bits": kv_bits, "source": source,
        "cache_type": info.cache_type, "model_max": info.max_context,
        "live_base_gb": round(live_base, 2), "model_base_gb": round(model_base, 2),
        "base_abs_gb": round(pred.base_abs_gb, 2), "slope_gb_per_k": round(slope, 5),
        "threshold_gb": round(threshold, 2), "wall_gb": round(wall, 2),
    }

    # RULE #1: would it breach the wall just to load?
    if pred.breaches_wall:
        p["refuse"] = True
        p["reason"] = (f"weights+baseline ≈ {pred.base_abs_gb:.2f}GB ≥ wall {wall:.2f}GB — "
                       f"would breach the wall on load. Cannot run safely on this machine.")
        p["max_kv_size"] = 0
        return p

    cap = pred.safe_ctx
    p["max_kv_size"] = cap
    if cap < MIN_USEFUL_CTX:
        p["refuse"] = True
        p["reason"] = (f"safe cap {cap:,} tok < {MIN_USEFUL_CTX} — base leaves no useful "
                       f"context headroom.")
    else:
        p["refuse"] = False
    return p


def build_argv(rest: list[str], p: dict) -> list[str]:
    """Inject --kv-bits (only if quantizable) and --max-kv-size, respecting user overrides."""
    argv = list(rest)
    if p["kv_bits"] is not None and "--kv-bits" not in argv:
        argv = ["--kv-bits", str(p["kv_bits"])] + argv
    if "--max-kv-size" not in argv:
        argv = ["--max-kv-size", str(p["max_kv_size"])] + argv
    return argv
