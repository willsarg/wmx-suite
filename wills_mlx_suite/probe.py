"""Safe characterizer: find a model's context ceiling by extrapolation, never by crashing.

Strategy:
  1. Pre-flight: estimate base footprint from weights.
       * estimate >= hard wall  -> HARD REFUSE (hopeless; e.g. the 27B). Never probe.
       * estimate >= threshold  -> borderline. Refuse UNLESS allow_min_probe, in which
         case run ONE supervised 512-token probe (deep in the safe zone) to replace the
         blind guess with ground truth, then decide.
       * estimate <  threshold  -> proceed with the ramp.
  2. Ramp context upward through safe rungs, one isolated subprocess each.
  3. Fit the model's DELTA over its own launch baseline: delta = os_wired - baseline.
     This isolates the model's footprint from background noise (browsers, IDEs, etc.).
  4. Before every rung, predict its ABSOLUTE peak = live_baseline + (model_base + slope*c)
     and STOP if it would cross the safe threshold.
  5. Solve for the safe ceiling and hard wall against a reference baseline.
"""
from __future__ import annotations

import json
import statistics
import subprocess
import sys
from dataclasses import dataclass

from . import db, models
from .system import SystemLimits, read_limits, sample_settled_baseline

DEFAULT_RAMP = [2048, 8192, 16384, 32768, 49152, 65536, 98304, 131072]
MIN_PROBE_CTX = 512  # supervised calibration probe — deep in the safe zone
DEFAULT_REPEATS = 3  # N-repeat median per rung, to smooth prefill-transient sampling jitter
# rough base-footprint estimate (GB): weights resident + fixed overhead, on top of the
# live system baseline. Calibrated loosely on Gemma/Qwen; refined as more models run.
RESIDENT_FACTOR = 1.05
FIXED_OVERHEAD_GB = 1.0


@dataclass
class Fit:
    model_base_gb: float      # delta intercept — model's own footprint at c->0 (invariant)
    slope_gb_per_k: float
    r2: float
    ref_baseline_gb: float
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


def _solve_ctx(model_base: float, slope_per_k: float, ref_baseline: float,
               target_gb: float) -> int:
    """Context (tokens) where ref_baseline + model_base + slope*c == target."""
    if slope_per_k <= 0:
        return 0
    headroom = target_gb - ref_baseline - model_base
    return int(max(0.0, headroom / slope_per_k) * 1000)


def estimate_base_gb(info: models.ModelInfo, limits: SystemLimits) -> float:
    """Pre-flight guess of ABSOLUTE base (context->0) OS-wired footprint, before any probe."""
    os_baseline = max(limits.wired_now_gb, 2.5)
    return os_baseline + info.weights_gb * RESIDENT_FACTOR + FIXED_OVERHEAD_GB


def _run_worker(py: str, hf_id: str, ctx: int, kv_bits, *, verbose, log) -> dict | None:
    cmd = [py, "-m", "wills_mlx_suite.probe_worker", hf_id, str(ctx)]
    if kv_bits is not None:
        cmd += ["--kv-bits", str(kv_bits)]
    out = subprocess.run(cmd, capture_output=True, text=True)
    line = next((l for l in out.stdout.splitlines() if l.startswith("{")), None)
    if not line:
        if verbose:
            log(f"# {ctx}: no result (stderr: {out.stderr.strip()[-200:]})")
        return None
    return json.loads(line)


def _measure_rung(py: str, hf_id: str, ctx: int, kv_bits, repeats: int, *, verbose, log):
    """Run a rung `repeats` times in fresh processes; return the MEDIAN high-water.

    Each prefill can land its peak between sampler windows (±~1GB jitter), so a single run
    makes ceilings look erratic. The median of N isolated runs is the textbook smoother.
    Returns a dict with median os_wired/delta/mlx_peak, or the first failure dict / None.
    """
    abss, deltas, peaks = [], [], []
    for _ in range(max(1, repeats)):
        m = _run_worker(py, hf_id, ctx, kv_bits, verbose=verbose, log=log)
        if m is None or m.get("status") != "ok":
            return m  # propagate error/None — stop the ramp
        abss.append(m["os_wired_gb"])
        deltas.append(m["os_wired_gb"] - m["baseline_wired_gb"])
        peaks.append(m["mlx_peak_gb"])
    return {
        "status": "ok", "context": ctx, "repeats": len(abss),
        "os_wired_gb": round(statistics.median(abss), 3),
        "delta": round(statistics.median(deltas), 3),
        "mlx_peak_gb": round(statistics.median(peaks), 3),
        "spread_gb": round(max(abss) - min(abss), 3),
    }


def characterize(hf_id: str, *, margin_gb: float = 2.0, ramp=None,
                 allow_min_probe: bool = False, repeats: int = DEFAULT_REPEATS,
                 worker_python: str | None = None, verbose=True) -> dict:
    ramp = ramp or DEFAULT_RAMP
    limits = read_limits()
    threshold = limits.safe_threshold_gb(margin_gb)
    wall = limits.wall_gb
    sweep_baseline = limits.wired_now_gb  # reference baseline for stored ceiling numbers
    info = models.describe(hf_id)
    if info is None:
        raise SystemExit(f"model not found in HF cache: {hf_id}")

    kv_bits = 4 if info.can_quantize_kv else None  # quantize only quantizable cache types
    py = worker_python or sys.executable

    import mlx.core as mx
    con = db.connect()
    db.upsert_model(con, info.as_dict())
    run_id = db.start_run(
        con, hf_id, kv_bits=kv_bits, kv_group_size=64, quantized_kv_start=5000,
        mlx_version=mx.__version__, wall_gb=wall, safe_threshold_gb=threshold,
    )

    def log(*a):
        if verbose:
            print(*a, flush=True)

    log(f"# {hf_id}")
    log(f"# wall={wall:.2f}GB  safe_threshold={threshold:.2f}GB  baseline={sweep_baseline:.2f}GB  "
        f"kv_bits={kv_bits}  cache={info.cache_type}  repeats={repeats}")

    xs_k: list[float] = []   # context in thousands of tokens
    ys: list[float] = []     # DELTA over launch baseline (model's own footprint)

    # ---- pre-flight gate -------------------------------------------------
    est = estimate_base_gb(info, limits)
    log(f"# pre-flight base estimate: {est:.2f}GB (weights {info.weights_gb}GB)")

    def _refuse(reason: str):
        log(f"# REFUSED: {reason}")
        db.add_measurement(con, run_id, 0, status="skipped", note=reason)
        return {"hf_id": hf_id, "refused": True, "reason": reason,
                "est_base_gb": est, "threshold_gb": threshold, "wall_gb": wall,
                "run_id": run_id}

    if est >= wall:
        return _refuse(f"estimated base {est:.2f}GB >= hard wall {wall:.2f}GB — "
                       f"cannot load without breaching the wall. Never probe.")
    if est >= threshold:
        if not allow_min_probe:
            return _refuse(f"estimated base {est:.2f}GB >= safe threshold {threshold:.2f}GB "
                           f"(but < wall {wall:.2f}GB). Borderline — re-run with --min-probe "
                           f"to measure the true base with a supervised 512-token probe.")
        # supervised minimal probe: deep in the safe zone, replaces the blind guess
        log(f"# borderline — running supervised {MIN_PROBE_CTX}-token calibration probe...")
        m = _run_worker(py, hf_id, MIN_PROBE_CTX, kv_bits, verbose=verbose, log=log)
        if not m or m.get("status") != "ok":
            note = (m or {}).get("note", "no output")
            return _refuse(f"min-probe failed: {note}")
        true_abs = m["os_wired_gb"]
        true_delta = round(true_abs - m["baseline_wired_gb"], 3)
        db.add_measurement(con, run_id, MIN_PROBE_CTX, mlx_peak_gb=m["mlx_peak_gb"],
                           mlx_true_gb=m["mlx_true_gb"], os_wired_gb=true_abs,
                           note=f"min-probe; delta={true_delta}")
        log(f"# measured @ {MIN_PROBE_CTX}: os_wired={true_abs:.2f}GB  "
            f"delta={true_delta:.2f}GB (est was {est:.2f}GB)")
        if true_abs >= threshold:
            return _refuse(f"measured base {true_abs:.2f}GB >= threshold {threshold:.2f}GB "
                           f"— genuinely too tight, not just a pessimistic estimate.")
        log(f"# cleared by ground truth ({true_abs:.2f}GB < {threshold:.2f}GB). Proceeding.")
        xs_k.append(MIN_PROBE_CTX / 1000)
        ys.append(true_delta)

    # ---- safe ramp -------------------------------------------------------
    fit: Fit | None = None

    def _refit():
        nonlocal fit
        if len(xs_k) >= 2:
            a, b, r2 = _linfit(xs_k, ys)
            fit = Fit(
                model_base_gb=round(a, 3), slope_gb_per_k=round(b, 5), r2=round(r2, 4),
                ref_baseline_gb=round(sweep_baseline, 3), threshold_gb=threshold,
                safe_ceiling_ctx=_solve_ctx(a, b, sweep_baseline, threshold),
                hard_wall_ctx=_solve_ctx(a, b, sweep_baseline, wall),
                n_points=len(xs_k),
            )

    _refit()  # in case the min-probe seeded a point (still need >=2 to fit)
    for ctx in ramp:
        if ctx > (info.max_context or ctx):
            break
        if xs_k and ctx <= xs_k[-1] * 1000:
            continue  # already covered by the min-probe seed
        # predict this rung against a SETTLED live baseline before launching it
        # (settle neutralizes un-reclaimed wired pages from the just-exited worker)
        if fit is not None:
            live_base = sample_settled_baseline()
            pred = live_base + fit.model_base_gb + fit.slope_gb_per_k * (ctx / 1000)
            if pred >= threshold:
                log(f"# STOP before {ctx}: predicted {pred:.2f}GB "
                    f"(live_base {live_base:.2f} + model {fit.model_base_gb:.2f} + "
                    f"{fit.slope_gb_per_k:.4f}*{ctx/1000:.1f}) >= {threshold:.2f}GB. "
                    f"Ceiling reached safely.")
                break

        m = _measure_rung(py, hf_id, ctx, kv_bits, repeats, verbose=verbose, log=log)
        if m is None:
            db.add_measurement(con, run_id, ctx, status="error", note="no json output")
            break
        if m.get("status") != "ok":
            log(f"# {ctx}: {m.get('status')} — {m.get('note')}")
            db.add_measurement(con, run_id, ctx, status=m.get("status"), note=m.get("note"))
            break  # a crash/NYI here means production can't go here either

        delta = m["delta"]
        db.add_measurement(con, run_id, ctx, mlx_peak_gb=m["mlx_peak_gb"],
                           os_wired_gb=m["os_wired_gb"],
                           note=f"median of {m['repeats']}; delta={delta}; spread={m['spread_gb']}")
        xs_k.append(ctx / 1000)
        ys.append(delta)
        log(f"{ctx:>8}  os_wired={m['os_wired_gb']:.2f}GB  delta={delta:.2f}GB  "
            f"mlx_peak={m['mlx_peak_gb']:.2f}GB  (median of {m['repeats']}, "
            f"spread {m['spread_gb']:.2f}GB)")
        _refit()

    result = {"hf_id": hf_id, "refused": False, "run_id": run_id,
              "threshold_gb": threshold, "wall_gb": wall}
    if fit is not None:
        db.save_fit(con, run_id, fit.__dict__)
        result["fit"] = fit.__dict__
        log(f"# FIT (delta): model_base={fit.model_base_gb}GB  slope={fit.slope_gb_per_k}GB/1k  "
            f"(R²={fit.r2}, n={fit.n_points})")
        log(f"# @ baseline {fit.ref_baseline_gb}GB:  safe ceiling ≈ {fit.safe_ceiling_ctx:,} tok"
            f"   hard wall ≈ {fit.hard_wall_ctx:,} tok")
    else:
        log("# not enough points to fit (need >=2 safe rungs)")
    return result
