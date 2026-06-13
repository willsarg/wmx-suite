"""Safe characterizer: find a model's context ceiling by extrapolation, never by crashing.

Strategy:
  1. Pre-flight: estimate base footprint from weights. If it already exceeds the safe
     threshold, refuse to probe at all (this is how the 27B is handled — never launched).
  2. Ramp context upward through safe rungs, one isolated subprocess each.
  3. Before every rung, predict its OS-wired peak from the fit so far. If the prediction
     crosses the safe threshold, STOP — do not launch it.
  4. Fit os_wired = intercept + slope * context and solve for the safe ceiling and wall.
"""
from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass

from . import db, models
from .system import SystemLimits, read_limits

DEFAULT_RAMP = [2048, 8192, 16384, 32768, 49152, 65536, 98304, 131072]
# rough base-footprint estimate (GB) ~= OS baseline wired + weights resident + fixed overhead.
# Calibrated loosely on Gemma/Qwen; refined as more models are characterized.
RESIDENT_FACTOR = 1.05
FIXED_OVERHEAD_GB = 1.0


@dataclass
class Fit:
    intercept_gb: float
    slope_gb_per_k: float
    r2: float
    threshold_gb: float
    safe_ceiling_ctx: int
    hard_wall_ctx: int
    n_points: int


def _linfit(xs_k: list[float], ys: list[float]) -> tuple[float, float, float]:
    """Least-squares y = a + b*x. Returns (a, b, r2). x in thousands of tokens."""
    n = len(xs_k)
    mx_ = sum(xs_k) / n
    my = sum(ys) / n
    sxx = sum((x - mx_) ** 2 for x in xs_k)
    sxy = sum((x - mx_) * (y - my) for x, y in zip(xs_k, ys))
    b = sxy / sxx if sxx else 0.0
    a = my - b * mx_
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((y - (a + b * x)) ** 2 for x, y in zip(xs_k, ys))
    r2 = 1 - ss_res / ss_tot if ss_tot else 1.0
    return a, b, r2


def _solve_ctx(intercept: float, slope_per_k: float, target_gb: float) -> int:
    if slope_per_k <= 0:
        return 0
    return int(max(0.0, (target_gb - intercept) / slope_per_k) * 1000)


def estimate_base_gb(info: models.ModelInfo, limits: SystemLimits) -> float:
    """Pre-flight guess of base (context->0) OS-wired footprint, before any probe."""
    os_baseline = max(limits.wired_now_gb, 2.5)
    return os_baseline + info.weights_gb * RESIDENT_FACTOR + FIXED_OVERHEAD_GB


def characterize(hf_id: str, *, margin_gb: float = 2.0, ramp=None,
                 worker_python: str | None = None, verbose=True) -> dict:
    ramp = ramp or DEFAULT_RAMP
    limits = read_limits()
    threshold = limits.safe_threshold_gb(margin_gb)
    info = models.describe(hf_id)
    if info is None:
        raise SystemExit(f"model not found in HF cache: {hf_id}")

    # production KV policy: quantize only if the cache type supports it
    kv_bits = 4 if info.can_quantize_kv else None
    py = worker_python or sys.executable

    import mlx.core as mx
    con = db.connect()
    db.upsert_model(con, info.as_dict())
    run_id = db.start_run(
        con, hf_id, kv_bits=kv_bits, kv_group_size=64, quantized_kv_start=5000,
        mlx_version=mx.__version__, wall_gb=limits.wall_gb, safe_threshold_gb=threshold,
    )

    def log(*a):
        if verbose:
            print(*a, flush=True)

    log(f"# {hf_id}")
    log(f"# wall={limits.wall_gb:.2f}GB  safe_threshold={threshold:.2f}GB  "
        f"kv_bits={kv_bits}  cache={info.cache_type}")

    # ---- pre-flight gate -------------------------------------------------
    est_base = estimate_base_gb(info, limits)
    log(f"# pre-flight base estimate: {est_base:.2f}GB (weights {info.weights_gb}GB)")
    if est_base >= threshold:
        log(f"# REFUSED: estimated base {est_base:.2f}GB >= safe threshold "
            f"{threshold:.2f}GB. Not launching any probe.")
        db.add_measurement(con, run_id, 0, status="skipped",
                           note=f"pre-flight base {est_base:.2f}GB exceeds threshold")
        return {"hf_id": hf_id, "refused": True, "est_base_gb": est_base,
                "threshold_gb": threshold, "run_id": run_id}

    # ---- safe ramp -------------------------------------------------------
    xs_k: list[float] = []
    ys: list[float] = []
    fit: Fit | None = None
    for ctx in ramp:
        if ctx > (info.max_context or ctx):
            break
        # predict this rung before launching it
        if fit is not None:
            pred = fit.intercept_gb + fit.slope_gb_per_k * (ctx / 1000)
            if pred >= threshold:
                log(f"# STOP before {ctx}: predicted {pred:.2f}GB >= threshold "
                    f"{threshold:.2f}GB. Ceiling reached safely.")
                break

        cmd = [py, "-m", "wills_mlx_suite.probe_worker", hf_id, str(ctx)]
        if kv_bits is not None:
            cmd += ["--kv-bits", str(kv_bits)]
        out = subprocess.run(cmd, capture_output=True, text=True)
        line = next((l for l in out.stdout.splitlines() if l.startswith("{")), None)
        if not line:
            log(f"# {ctx}: no result (stderr: {out.stderr.strip()[-200:]})")
            db.add_measurement(con, run_id, ctx, status="error", note="no json output")
            break
        m = json.loads(line)
        if m.get("status") != "ok":
            log(f"# {ctx}: {m.get('status')} — {m.get('note')}")
            db.add_measurement(con, run_id, ctx, status=m.get("status"), note=m.get("note"))
            # a crash/NYI at this rung means production can't go here either; stop
            break

        db.add_measurement(con, run_id, ctx, mlx_peak_gb=m["mlx_peak_gb"],
                           mlx_true_gb=m["mlx_true_gb"], os_wired_gb=m["os_wired_gb"])
        xs_k.append(ctx / 1000)
        ys.append(m["os_wired_gb"])
        log(f"{ctx:>8}  os_wired={m['os_wired_gb']:.2f}GB  "
            f"mlx_peak={m['mlx_peak_gb']:.2f}GB  mlx_true={m['mlx_true_gb']:.2f}GB")

        if len(xs_k) >= 2:
            a, b, r2 = _linfit(xs_k, ys)
            fit = Fit(
                intercept_gb=round(a, 3), slope_gb_per_k=round(b, 5), r2=round(r2, 4),
                threshold_gb=threshold,
                safe_ceiling_ctx=_solve_ctx(a, b, threshold),
                hard_wall_ctx=_solve_ctx(a, b, limits.wall_gb),
                n_points=len(xs_k),
            )

    result = {"hf_id": hf_id, "refused": False, "run_id": run_id,
              "threshold_gb": threshold, "wall_gb": limits.wall_gb}
    if fit is not None:
        db.save_fit(con, run_id, fit.__dict__)
        result["fit"] = fit.__dict__
        log(f"# FIT: os_wired = {fit.intercept_gb} + {fit.slope_gb_per_k}*c[k]  "
            f"(R²={fit.r2}, n={fit.n_points})")
        log(f"# safe ceiling ≈ {fit.safe_ceiling_ctx:,} tokens   "
            f"hard wall ≈ {fit.hard_wall_ctx:,} tokens")
    else:
        log("# not enough points to fit (need >=2 safe rungs)")
    return result
