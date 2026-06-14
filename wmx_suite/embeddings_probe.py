"""Safe 2D (batch x seq_len) OS-wired memory-surface benchmark for encoder embeddings.

Mirrors probe.py's "extrapolate, never crash" approach for a NON-causal encoder:
  * one isolated subprocess per grid cell (probe_worker_embeddings)
  * the predictive gate runs in THIS (parent) process BEFORE spawning each cell
  * per-row ramp; skip the rest of a row once a cell is predicted to breach the wall
  * the gate trusts a fit of REAL measured high-water peaks; a conservative analytic prior
    is used only for cold start (before MIN_FIT_POINTS cells exist)
"""
from __future__ import annotations

import json
import subprocess
import sys

from . import config, db
from .system import read_limits, sample_settled_baseline

DEFAULT_BATCHES = [1, 2, 4, 8, 16, 32]
DEFAULT_SEQS = [128, 256, 512, 1024, 2048, 4096, 8192]

MIN_FIT_POINTS = 3
PRED_SAFETY = 1.25
MODEL_BASE_SEED_GB = 1.0  # weight-residency seed so cold-start/pre-flight aren't zero

# ModernBERT-base architecture constants (verified from config.json).
NUM_LAYERS = 22
HIDDEN_SIZE = 768
NUM_HEADS = 12

# Cold-start OVER-estimate (sum over all layers as if global): a safe upper bound that only
# ever gates the first tiny cells, since the sweep ramps seq from the smallest value.
A_COLD = NUM_LAYERS * HIDDEN_SIZE * 2 / 1e9   # GB per (batch*seq)
B_COLD = NUM_LAYERS * NUM_HEADS * 2 / 1e9     # GB per (batch*seq^2)

# Physical one-layer FLOOR (lower bound: >=1 layer's attention + residual resident at peak).
A_FLOOR = HIDDEN_SIZE * 2 / 1e9
B_FLOOR = NUM_HEADS * 2 / 1e9


def _default_event(_event: dict) -> None:
    pass


def _fit_ab(points: list[tuple[float, float, float]]) -> tuple[float, float]:
    """Least-squares delta = a*x1 + b*x2 through the origin (no intercept).

    points: (x1=batch*seq, x2=batch*seq^2, delta). Returns (a, b); (0,0) if singular.
    """
    s11 = s12 = s22 = sd1 = sd2 = 0.0
    for x1, x2, d in points:
        s11 += x1 * x1
        s12 += x1 * x2
        s22 += x2 * x2
        sd1 += x1 * d
        sd2 += x2 * d
    det = s11 * s22 - s12 * s12
    if abs(det) < 1e-30:
        return 0.0, 0.0
    a = (sd1 * s22 - sd2 * s12) / det
    b = (s11 * sd2 - s12 * sd1) / det
    return a, b


def _coeffs(points: list[tuple[float, float, float]]) -> tuple[float, float]:
    """Gate coefficients (a, b): cold over-estimate before MIN_FIT_POINTS, else fit
    clamped to the one-layer physical floor."""
    if len(points) < MIN_FIT_POINTS:
        return A_COLD, B_COLD
    a_fit, b_fit = _fit_ab(points)
    return max(A_FLOOR, max(0.0, a_fit)), max(B_FLOOR, max(0.0, b_fit))


def _run_cell(py: str, model: str, batch: int, seq: int, repeats: int, margin: float) -> dict:
    cmd = [py, "-m", "wmx_suite.probe_worker_embeddings",
           "--model", model, "--batch", str(batch), "--seq", str(seq),
           "--repeats", str(repeats), "--margin", str(margin)]
    out = subprocess.run(cmd, capture_output=True, text=True)
    line = next((l for l in out.stdout.splitlines() if l.startswith("{")), None)
    if not line:
        return {"status": "error",
                "note": f"no result (stderr: {out.stderr.strip()[-200:]})"}
    return json.loads(line)


def sweep(con, run_id: int, model: str, batches=None, seqs=None, repeats: int = 3,
          margin_gb: float | None = None, *, on_event=None, persist: bool = True) -> dict:
    batches = batches or DEFAULT_BATCHES
    seqs = sorted(seqs or DEFAULT_SEQS)
    margin = config.margin_gb(margin_gb)
    on_event = on_event or _default_event
    py = sys.executable

    limits = read_limits()
    threshold = limits.safe_threshold_gb(margin)

    points: list[tuple[float, float, float]] = []  # (x1, x2, delta)
    model_base = MODEL_BASE_SEED_GB
    smallest_unsafe_seq: float = float("inf")  # for monotonic pruning across batches
    n_measured = 0
    n_skipped = 0

    # Pre-flight: if even a tiny cell can't fit given current pressure, abort the sweep.
    live = sample_settled_baseline()
    if live + model_base >= threshold:
        on_event({"event": "preflight_abort",
                  "note": (f"host pressure {live:.2f} GB + model seed {model_base} GB "
                           f">= threshold {threshold:.2f} GB")})
        return {"model": model, "run_id": run_id, "n_cells_measured": 0,
                "n_cells_skipped": 0, "aborted": True}

    for batch in batches:
        for seq in seqs:
            # Monotonic pruning: memory grows with batch at fixed seq.
            if seq >= smallest_unsafe_seq:
                on_event({"event": "row_skipped", "batch": batch, "seq": seq,
                          "predicted_gb": None})
                n_skipped += 1
                continue

            live_base = sample_settled_baseline()
            a, b = _coeffs(points)
            x1, x2 = batch * seq, batch * seq * seq
            predicted = live_base + model_base + PRED_SAFETY * (a * x1 + b * x2)
            if predicted >= threshold:
                on_event({"event": "row_skipped", "batch": batch, "seq": seq,
                          "predicted_gb": round(predicted, 3)})
                n_skipped += 1
                smallest_unsafe_seq = min(smallest_unsafe_seq, seq)
                break  # skip the rest of this (ascending-seq) row

            result = _run_cell(py, model, batch, seq, repeats, margin)
            if result.get("status") != "rung_done":
                on_event({"event": "error", "batch": batch, "seq": seq,
                          "note": result.get("note", "worker error")})
                return {"model": model, "run_id": run_id,
                        "n_cells_measured": n_measured, "n_cells_skipped": n_skipped,
                        "error": result.get("note")}

            delta = max(0.0, result["os_wired_gb"] - live_base)
            points.append((x1, x2, delta))
            if n_measured == 0:
                model_base = max(MODEL_BASE_SEED_GB,
                                 result["os_wired_gb"] - live_base - (a * x1 + b * x2))
            n_measured += 1

            if persist and con is not None:
                db.add_embeddings_measurement(
                    con, run_id, batch_size=batch, seq_len=seq,
                    os_wired_gb=result["os_wired_gb"], peak_gb=result["peak_gb"],
                    throughput_tps=result["throughput_tps"], latency_ms=result["latency_ms"],
                )
            on_event({"event": "cell_done", "batch": batch, "seq": seq,
                      "os_wired_gb": result["os_wired_gb"], "peak_gb": result["peak_gb"],
                      "throughput_tps": result["throughput_tps"],
                      "latency_ms": result["latency_ms"]})

    return {"model": model, "run_id": run_id, "n_cells_measured": n_measured,
            "n_cells_skipped": n_skipped}
