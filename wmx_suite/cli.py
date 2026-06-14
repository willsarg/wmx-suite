"""Command-line entry point for the suite.

    uv run wmx-suite system                 # show the machine's memory wall + swap
    uv run wmx-suite health                  # live snapshot: pressure + per-model go/no-go
    uv run wmx-suite scan                    # register all mlx-community models in the cache
    uv run wmx-suite show <hf_id>            # architecture + memory class for one model
    uv run wmx-suite characterize <hf_id>   # safe probe -> fitted context ceiling
    uv run wmx-suite list                    # characterized models + ceilings from the DB
"""
from __future__ import annotations

import argparse
import fcntl
import os
import pty
import re
import signal
import struct
import subprocess
import sys
import termios
import threading
from statistics import median

from mlx_lm.utils import load_tokenizer

from . import config, db, launcher, models, probe, profiles
from .system import read_limits, sample_settled_baseline
from .ui import Console
from .views import health as view_health
from .views import landing as view_landing
from .views import list as view_list
from .views import scan as view_scan
from .views import show as view_show
from .views import system as view_system
from .views import run_messages as view_run
from .views import benchmark_kokoro as view_bench_kokoro
from .views import benchmark_kokoro_concurrency as view_bench_kokoro_conc
from .views import benchmark_embeddings as view_bench_embed

# Default Console for the `run` fast path; main() replaces it per-invocation.
CONSOLE = Console.from_args()


def _configured_margin(value=None) -> float:
    try:
        margin = config.margin_gb(value)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if margin == 0:
        print("[warning] margin is 0 GB — the safe threshold equals the crash wall, with "
              "no cushion. Crossing it can hard-lock the machine.", file=sys.stderr)
    return margin


def _stream_worker(
    cmd: list[str],
    on_line,
    *,
    capture_stderr: bool = True,
) -> tuple[int, str]:
    """Spawn *cmd* and stream its stdout line-by-line via *on_line(line)*.

    Stderr is drained concurrently on a background thread so a large stderr
    payload (e.g. MLX/Metal warnings or a traceback) can never fill the OS
    pipe buffer and deadlock the parent.

    If *on_line* returns a truthy value for a given line, *_stream_worker*
    immediately terminates the child process (SIGTERM), stops consuming stdout,
    and returns.  This is used by handlers to abort on a genuine ``error``
    event so the worker cannot continue processing further rungs.

    Returns ``(returncode, stderr_text)``.  If *capture_stderr* is False the
    subprocess inherits the terminal's stderr (no deadlock risk, empty string
    returned for stderr_text).

    The caller is responsible for handling KeyboardInterrupt around this call
    if it wants to terminate the child cleanly.
    """
    stderr_pipe = subprocess.PIPE if capture_stderr else None
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=stderr_pipe, text=True)

    # Drain stderr on a background thread so it never fills (~64 KB limit).
    stderr_buf: list[str] = []
    if capture_stderr:
        def _drain_stderr():
            assert proc.stderr is not None
            for chunk in iter(lambda: proc.stderr.read(4096), ""):
                stderr_buf.append(chunk)
            proc.stderr.close()

        stderr_thread = threading.Thread(target=_drain_stderr, daemon=True)
        stderr_thread.start()

    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            if on_line(line):
                # on_line signalled an abort: terminate the child so no further
                # rungs are processed or persisted, then drain remaining stdout
                # to avoid an OS pipe buffer deadlock before we close it.
                proc.terminate()
                for _ in proc.stdout:
                    pass  # drain without processing
                break
    finally:
        proc.stdout.close()

    if capture_stderr:
        stderr_thread.join()

    proc.wait()
    stderr_text = "".join(stderr_buf).strip() if capture_stderr else ""
    return proc.returncode, stderr_text


def _strip_prefix(model_id: str) -> str:
    return model_id.split("/", 1)[-1] if "/" in model_id else model_id


def cmd_system(args):
    s = read_limits()
    margin = _configured_margin()
    safe = s.safe_threshold_gb(margin)
    con = db.connect()
    prof = db.get_profile(con, profiles.machine_key())
    swap = s.swap_free_gb
    data = {
        "device": s.device,
        "total_gb": s.total_gb,
        "wall_gb": s.wall_gb,
        "safe_budget_gb": safe,
        "wired_gb": s.wired_now_gb,
        "free_headroom_gb": safe - s.wired_now_gb,
        "max_buffer_gb": s.max_buffer_gb,
        "swap_free_gb": swap,
        "swap_warn": swap is not None and swap < 2.0,
        "margin_gb": margin,
        "margin_source": ("WMX_SUITE_MARGIN_GB"
                          if os.environ.get("WMX_SUITE_MARGIN_GB") else "default"),
        "wall_bytes": int(round(s.wall_gb * 1e9)),
        "wired_sample": "median of 3 @ 0.2s",
        "calibrated": prof is not None,
        "cal_model_short": _strip_prefix(prof["model_id"]) if prof else "",
        "cal_overhead_gb": prof["fixed_overhead_gb"] if prof else 0.0,
        "cal_date": (prof["calibrated_at"][:10] if prof else ""),
    }
    view_system.render(args.console, data)


def _safe_ctx_str(pred) -> str:
    """Human label for a model's safe-context verdict (caller-side policy)."""
    if pred.breaches_wall:
        return "over budget — won't load"
    if pred.safe_ctx < launcher.MIN_USEFUL_CTX:
        return f"only ~{pred.safe_ctx:,} tok (too small)"
    return f"~{int(round(pred.safe_ctx, -2)):,} tokens"


def cmd_health(args):
    """Live 'can I run things safely right now?' snapshot: system pressure + per-model go/no-go."""
    margin = _configured_margin(args.margin)
    s = read_limits()
    threshold = s.safe_threshold_gb(margin)
    live_base = sample_settled_baseline()  # same baseline `run` uses, sampled once

    con = db.connect()
    key = profiles.machine_key()
    calibrated = db.get_profile(con, key) is not None
    rows = con.execute(
        "SELECT DISTINCT m.hf_id, m.max_context FROM models m "
        "JOIN probe_runs r ON r.hf_id = m.hf_id JOIN fits f ON f.run_id = r.id "
        "ORDER BY m.hf_id"
    ).fetchall()

    model_rows = []
    for r in rows:
        fit = db.latest_fit(con, r["hf_id"])
        if not fit or not fit.get("slope_gb_per_k"):
            continue
        pred = launcher.predict(
            model_base_gb=float(fit["model_base_gb"]),
            slope_gb_per_k=float(fit["slope_gb_per_k"]),
            live_base_gb=live_base, threshold_gb=threshold,
            wall_gb=s.wall_gb, model_max=r["max_context"],
        )
        go = (not pred.breaches_wall) and pred.safe_ctx >= launcher.MIN_USEFUL_CTX
        model_rows.append({
            "name": _strip_prefix(r["hf_id"]),
            "loads_gb": pred.base_abs_gb,
            "spare_gb": pred.headroom_gb,
            "safe_ctx": pred.safe_ctx,
            "safe_ctx_str": _safe_ctx_str(pred),
            "ok": go,
            "base_gb": float(fit["model_base_gb"]),
            "slope_gb_per_k": float(fit["slope_gb_per_k"]),
            "safe_cap_tok": pred.safe_ctx,
        })

    swap = s.swap_free_gb
    data = {
        "wall_gb": s.wall_gb,
        "safe_budget_gb": threshold,
        "free_now_gb": threshold - live_base,
        "swap_free_gb": swap,
        "swap_warn": swap is not None and swap < 2.0,
        "margin_gb": margin,
        "baseline_gb": live_base,
        "baseline_sample": "median of 3",
        "models": model_rows,
    }
    view_health.render(args.console, data)
    if not calibrated:
        dev, ram, osv = key
        args.console.emit()
        args.console.emit(args.console.style(
            "warn",
            f"No calibration profile for {dev} / {ram / 1e9:.0f}GB / macOS {osv}; "
            "cold-start estimates use fallback priors — run 'wmx-suite calibrate'."))


def cmd_scan(args):
    con = db.connect()
    found = models.scan_cache()
    model_rows = []
    for hf_id in found:
        info = models.describe(hf_id)
        if info is None or not info.is_causal:
            continue
        db.upsert_model(con, info.as_dict())
        model_rows.append({
            "hf_id": hf_id,
            "weights_gb": info.weights_gb,
            "weights_gb_exact": info.weights_gb,
            "kv_label": ("quantizable KV" if info.can_quantize_kv
                         else "fp16-only KV (sliding-window cache)"),
            "cache_type": info.as_dict().get("cache_type", ""),
        })
    view_scan.render(args.console, {"models": model_rows, "registered": len(model_rows)})


def cmd_show(args):
    info = models.describe(args.hf_id)
    if info is None:
        view_run.render_not_found(args.console, {
            "model": args.hf_id,
            "cache_path": os.environ.get("HF_HOME", "~/.cache/huggingface/hub"),
            "hf_home_set": "HF_HOME" in os.environ,
        })
        raise SystemExit(1)
    if not info.is_causal:
        print(f"WARNING: Model {args.hf_id} is in HF cache but is not a supported causal language model.")
    d = info.as_dict()
    bpt = info.fp16_kv_bytes_per_token()
    cache_type = d["cache_type"]
    data = {
        "hf_id": args.hf_id,
        "weights_gb": info.weights_gb,
        "n_layers": d["n_layers"],
        "growing_layers": d["growing_layers"],
        "max_context": d["max_context"],
        "kv_label": (f"sliding-window ({cache_type})" if not info.can_quantize_kv
                     else f"standard ({cache_type})"),
        "can_quantize_kv": info.can_quantize_kv,
        "growth_gb_per_1k": bpt * 1000 / 1e9,
        "cache_type": cache_type,
        "kv_heads": d["kv_heads"],
        "head_dim": d["head_dim"],
        "hidden_size": d["hidden_size"],
        "layer_types": str(d["layer_types"]),
        "max_kv_size_enforced": d["max_kv_size_enforced"],
        "is_causal": d["is_causal"],
        "fp16_kv_bytes_per_token": int(bpt),
    }
    view_show.render(args.console, data)


def cmd_characterize(args):
    margin = _configured_margin(args.margin)
    probe.characterize(args.hf_id, margin_gb=margin, allow_min_probe=args.min_probe,
                       repeats=args.repeats)


def cmd_calibrate(args):
    """Measure this machine's cold-start overhead and store a per-machine profile."""
    margin = _configured_margin(args.margin)
    result = probe.calibrate(args.model, margin_gb=margin)
    dev, ram, osv = result["machine_key"]
    print("=" * 60)
    print("  Calibrated cold-start overhead for this machine")
    print("=" * 60)
    print(f"  Machine    : {dev} / {ram / 1e9:.0f}GB / macOS {osv}")
    print(f"  Model used : {result['hf_id']} ({result['n_points']} rungs)")
    print(f"  Measured   : {result['measured_overhead_gb']:.2f} GB overhead "
          f"(default {result['default_overhead_gb']:.2f} GB)")
    print(f"  Stored     : {result['fixed_overhead_gb']:.2f} GB  (floored at default)")
    print("=" * 60)


def _fit_quality(r2: float) -> str:
    if r2 >= 0.99:
        return "good"
    if r2 >= 0.95:
        return "ok"
    return "poor"


def cmd_list(args):
    con = db.connect()
    rows = db.latest_fits(con)
    if not rows:
        view_run.render_no_models(args.console, {})
        return
    speeds = db.gen_speeds(con)
    model_rows = []
    for r in rows:
        s = speeds.get(r["hf_id"])
        model_rows.append({
            "hf_id": r["hf_id"],
            "loads_gb": r["model_base_gb"],
            "safe_ctx": r["safe_ceiling_ctx"],
            "speed_tps": median(s) if s else None,
            "fit": _fit_quality(r["r2"]),
            "tight": r["safe_ceiling_ctx"] < 8192,
            "stale": models.fit_is_stale(r["hf_id"], r["characterized_at"]),
            "slope_gb_per_k": r["slope_gb_per_k"],
            "r2": r["r2"],
            "hard_wall_ctx": r["hard_wall_ctx"],
            "n_runs": len(s) if s else 0,
        })
    view_list.render(args.console, {"models": model_rows})
    stale = [m["hf_id"] for m in model_rows if m["stale"]]
    if stale:
        args.console.emit()
        for hf_id in stale:
            args.console.emit(args.console.style(
                "warn", f"⚠ {_strip_prefix(hf_id)}: fit may be stale — "
                        "re-run 'wmx-suite characterize'."))


def cmd_web(args):
    """Launch the Flask dashboard web server."""
    try:
        from wmx_suite.web.app import create_app
    except ImportError as exc:
        raise SystemExit(
            "Flask is not installed or web modules are missing. "
            "Please run: pip install wills-wmx-suite[web] or uv sync --extra web"
        ) from exc

    app = create_app()
    if app is None:
        raise SystemExit("Failed to initialize Flask application.")

    if not args.debug:
        import logging
        logging.getLogger("werkzeug").setLevel(logging.ERROR)

    print(f"Launching wmx-suite dashboard on http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)


def cmd_benchmark_embeddings(args):
    """Run the ModernBERT embeddings 2D (batch x seq) memory-surface sweep and log it."""
    from . import embeddings_probe

    margin_val = _configured_margin(args.margin)
    batches = [int(x) for x in str(args.batches).split(",") if x.strip()]
    seqs = [int(x) for x in str(args.seqs).split(",") if x.strip()]

    import mlx.core as mx
    mlx_version = mx.__version__

    con = db.connect()
    run_id = db.start_embeddings_run(con, args.model, mlx_version)

    ignore_profile = getattr(args, "ignore_profile", False)
    if not ignore_profile and profiles.embedding_coeffs(con, args.model, mlx_version):
        profile_source = "loaded"
    elif ignore_profile:
        profile_source = "ignored"
    else:
        profile_source = "cold"

    console = getattr(args, "console", None) or CONSOLE

    # Accumulate cells for the final render.  Aborted flags come from events.
    cells: dict = {}
    aborted = False
    safeguard_data: dict | None = None

    def on_event(event):
        nonlocal aborted, safeguard_data
        ev = event.get("event")
        if ev == "preflight_abort":
            aborted = True
            # Render the safeguard guidance immediately so the user sees it.
            view_bench_embed.render_safeguard(console, {
                "model": args.model,
                "reason": event.get("note", "pre-flight check refused this run"),
                "predicted_gb": event.get("predicted_gb") or 0.0,
                "safe_budget_gb": event.get("safe_budget_gb") or 0.0,
                "margin_gb": margin_val,
            })
        elif ev == "error":
            aborted = True
            console.emit(console.style(
                "warn",
                f"ERROR at batch {event.get('batch')} seq {event.get('seq')}: "
                f"{event.get('note')}"
            ))
        elif ev == "row_skipped":
            b, s = event["batch"], event["seq"]
            cells[(b, s)] = {
                "status": "skipped",
                "throughput_tps": None,
                "latency_ms": None,
                "peak_gb": None,
                "os_wired_gb": None,
                "predicted_gb": event.get("predicted_gb"),
            }
        elif ev == "cell_done":
            b, s = event["batch"], event["seq"]
            cells[(b, s)] = {
                "status": "measured",
                "throughput_tps": event.get("throughput_tps"),
                "latency_ms": event.get("latency_ms"),
                "peak_gb": event.get("peak_gb"),
                "os_wired_gb": event.get("os_wired_gb"),
                "predicted_gb": event.get("predicted_gb"),
            }

    summary = embeddings_probe.sweep(
        con, run_id, args.model, batches=batches, seqs=seqs,
        repeats=args.repeats, margin_gb=margin_val, mlx_version=mlx_version,
        ignore_profile=ignore_profile, on_event=on_event,
    )

    if aborted:
        sys.exit(1)

    view_bench_embed.render_surface(console, {
        "model": args.model,
        "margin_gb": margin_val,
        "profile_source": profile_source,
        "mlx_version": mlx_version,
        "batches": batches,
        "seqs": seqs,
        "cells": cells,
        "n_cells_measured": summary.get("n_cells_measured", 0),
        "n_cells_skipped": summary.get("n_cells_skipped", 0),
    })


def cmd_benchmark_kokoro(args):
    """Run Kokoro TTS performance sweep and log to database."""
    import json

    margin_val = _configured_margin(args.margin)

    py = sys.executable
    cmd = [
        py, "-m", "wmx_suite.probe_worker_kokoro",
        "--model", args.model,
        "--voice", args.voice,
        "--lengths", args.lengths,
        "--repeats", str(args.repeats),
        "--margin", str(margin_val)
    ]

    import mlx.core as mx
    mlx_version = mx.__version__

    con = db.connect()
    run_id = db.start_kokoro_run(con, args.model, args.voice, mlx_version)

    console = getattr(args, "console", None) or CONSOLE

    rows: list[dict] = []
    safeguard_triggered = False
    safeguard_note: str = ""
    error_triggered = False

    def on_line(raw_line):
        nonlocal safeguard_triggered, safeguard_note, error_triggered
        line = raw_line.strip()
        if not line.startswith("{"):
            return None
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            return None

        status = data.get("status")
        if status == "warmup_done":
            pass  # no output — the view renders the full table at the end
        elif status == "safeguard_triggered":
            safeguard_triggered = True
            safeguard_note = data.get("note", "")
        elif status == "rung_done":
            length = data["length"]
            audio_dur = data["audio_duration"]
            comp_time = data["compute_time"]
            rtf = data["rtf"]
            cps = data["cps"]
            peak_gb = data["peak_gb"]
            os_wired = data.get("os_wired_gb")
            rows.append({
                "length": length,
                "audio_dur": audio_dur,
                "compute_time": comp_time,
                "rtf": rtf,
                "cps": cps,
                "peak_gb": peak_gb,
                "os_wired_gb": os_wired,
            })
            db.add_kokoro_measurement(
                con, run_id,
                text_length=length,
                audio_duration=audio_dur,
                compute_time=comp_time,
                rtf=rtf,
                cps=cps,
                peak_gb=peak_gb,
                os_wired_gb=os_wired
            )
        elif status == "error":
            console.emit(console.style("warn", f"ERROR: {data.get('note')}"))
            error_triggered = True
            return True  # signal _stream_worker to terminate the child immediately
        return None

    try:
        returncode, stderr_text = _stream_worker(cmd, on_line)
    except KeyboardInterrupt:
        console.emit(console.style("warn", "Benchmark interrupted by user."))
        sys.exit(1)

    if error_triggered:
        sys.exit(1)

    if returncode != 0:
        console.emit(console.style("warn", f"Worker exited with code {returncode}."))
        if stderr_text:
            console.emit(console.style("gloss", f"Stderr: {stderr_text}"))
        sys.exit(returncode)

    if safeguard_triggered:
        view_bench_kokoro.render_safeguard(console, {
            "note": safeguard_note,
            "next_cmd": "wmx-suite benchmark-kokoro --lengths <shorter>",
        })

    view_bench_kokoro.render_perf(console, {
        "model": args.model,
        "voice": args.voice,
        "lengths": args.lengths,
        "repeats": args.repeats,
        "margin": margin_val,
        "rows": rows,
        "safeguard_triggered": safeguard_triggered,
        "run_id": run_id,
    })


def cmd_benchmark_kokoro_ttfa(args):
    """Run Kokoro TTS TTFA latency benchmark sweep and log to database."""
    import json

    margin_val = _configured_margin(args.margin)

    py = sys.executable
    cmd = [
        py, "-m", "wmx_suite.probe_worker_kokoro_ttfa",
        "--model", args.model,
        "--voice", args.voice,
        "--lengths", args.lengths,
        "--repeats", str(args.repeats),
        "--margin", str(margin_val)
    ]

    import mlx.core as mx
    mlx_version = mx.__version__

    con = db.connect()
    run_id = db.start_kokoro_ttfa_run(con, args.model, args.voice, mlx_version)

    console = getattr(args, "console", None) or CONSOLE

    rows: list[dict] = []
    safeguard_triggered = False
    safeguard_note: str = ""
    error_triggered = False

    def on_line(raw_line):
        nonlocal safeguard_triggered, safeguard_note, error_triggered
        line = raw_line.strip()
        if not line.startswith("{"):
            return None
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            return None

        status = data.get("status")
        if status == "warmup_done":
            pass
        elif status == "safeguard_triggered":
            safeguard_triggered = True
            safeguard_note = data.get("note", "")
        elif status == "rung_done":
            length = data["length"]
            ttfa_sec = data["ttfa_sec"]
            total_sec = data["total_sec"]
            speedup = data["speedup_ratio"]
            chunk_dur = data["first_chunk_duration"]
            peak_gb = data["peak_gb"]
            rows.append({
                "length": length,
                "ttfa_sec": ttfa_sec,
                "total_sec": total_sec,
                "speedup_ratio": speedup,
                "first_chunk_dur": chunk_dur,
                "peak_gb": peak_gb,
            })
            db.add_kokoro_ttfa_measurement(
                con, run_id,
                text_length=length,
                ttfa_sec=ttfa_sec,
                total_sec=total_sec,
                speedup_ratio=speedup,
                first_chunk_duration=chunk_dur,
                peak_gb=peak_gb
            )
        elif status == "error":
            console.emit(console.style("warn", f"ERROR: {data.get('note')}"))
            error_triggered = True
            return True  # signal _stream_worker to terminate the child immediately
        return None

    try:
        returncode, stderr_text = _stream_worker(cmd, on_line)
    except KeyboardInterrupt:
        console.emit(console.style("warn", "Benchmark interrupted by user."))
        sys.exit(1)

    if error_triggered:
        sys.exit(1)

    if returncode != 0:
        console.emit(console.style("warn", f"Worker exited with code {returncode}."))
        if stderr_text:
            console.emit(console.style("gloss", f"Stderr: {stderr_text}"))
        sys.exit(returncode)

    if safeguard_triggered:
        view_bench_kokoro.render_safeguard(console, {
            "note": safeguard_note,
            "next_cmd": "wmx-suite benchmark-kokoro-ttfa --lengths <shorter>",
        })

    view_bench_kokoro.render_ttfa(console, {
        "model": args.model,
        "voice": args.voice,
        "lengths": args.lengths,
        "repeats": args.repeats,
        "margin": margin_val,
        "rows": rows,
        "safeguard_triggered": safeguard_triggered,
        "run_id": run_id,
    })


def cmd_benchmark_kokoro_batch(args):
    """Run Kokoro TTS batch performance sweep and log to database."""
    import json

    margin_val = _configured_margin(args.margin)

    py = sys.executable
    cmd = [
        py, "-m", "wmx_suite.probe_worker_kokoro_batch",
        "--model", args.model,
        "--voice", args.voice,
        "--batch-sizes", args.batch_sizes,
        "--repeats", str(args.repeats),
        "--margin", str(margin_val)
    ]

    import mlx.core as mx
    mlx_version = mx.__version__

    con = db.connect()
    run_id = db.start_kokoro_batch_run(con, args.model, args.voice, mlx_version)

    console = getattr(args, "console", None) or CONSOLE

    rows: list[dict] = []
    safeguard_triggered = False
    safeguard_note: str = ""
    error_triggered = False

    def on_line(raw_line):
        nonlocal safeguard_triggered, safeguard_note, error_triggered
        line = raw_line.strip()
        if not line.startswith("{"):
            return None
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            return None

        status = data.get("status")
        if status == "warmup_done":
            pass
        elif status == "safeguard_triggered":
            safeguard_triggered = True
            safeguard_note = data.get("note", "")
        elif status == "rung_done":
            batch_size = data["batch_size"]
            total_time = data["total_time"]
            cps = data["cps"]
            peak_gb = data["peak_gb"]
            rows.append({
                "batch_size": batch_size,
                "total_time": total_time,
                "cps": cps,
                "peak_gb": peak_gb,
            })
            db.add_kokoro_batch_measurement(
                con, run_id,
                batch_size=batch_size,
                total_time=total_time,
                cps=cps,
                peak_gb=peak_gb
            )
        elif status == "error":
            console.emit(console.style("warn", f"ERROR: {data.get('note')}"))
            error_triggered = True
            return True  # signal _stream_worker to terminate the child immediately
        return None

    try:
        returncode, stderr_text = _stream_worker(cmd, on_line)
    except KeyboardInterrupt:
        console.emit(console.style("warn", "Benchmark interrupted by user."))
        sys.exit(1)

    if error_triggered:
        sys.exit(1)

    if returncode != 0:
        console.emit(console.style("warn", f"Worker exited with code {returncode}."))
        if stderr_text:
            console.emit(console.style("gloss", f"Stderr: {stderr_text}"))
        sys.exit(returncode)

    if safeguard_triggered:
        view_bench_kokoro_conc.render_safeguard(console, {
            "note": safeguard_note,
            "next_cmd": "wmx-suite benchmark-kokoro-batch --batch-sizes <smaller>",
        })

    view_bench_kokoro_conc.render_batch(console, {
        "model": args.model,
        "voice": args.voice,
        "batch_sizes": args.batch_sizes,
        "repeats": args.repeats,
        "margin": margin_val,
        "rows": rows,
        "safeguard_triggered": safeguard_triggered,
        "run_id": run_id,
    })


def cmd_benchmark_kokoro_voice(args):
    """Run Kokoro TTS voice switching performance sweep and log to database."""
    import json

    margin_val = _configured_margin(args.margin)

    py = sys.executable
    cmd = [
        py, "-m", "wmx_suite.probe_worker_kokoro_voice",
        "--model", args.model,
        "--voice-a", args.voice_a,
        "--voice-b", args.voice_b,
        "--repeats", str(args.repeats),
        "--margin", str(margin_val)
    ]

    import mlx.core as mx
    mlx_version = mx.__version__

    con = db.connect()
    run_id = db.start_kokoro_voice_run(con, args.model, mlx_version)

    console = getattr(args, "console", None) or CONSOLE

    rows: list[dict] = []
    safeguard_triggered = False
    safeguard_note: str = ""
    error_triggered = False

    def on_line(raw_line):
        nonlocal safeguard_triggered, safeguard_note, error_triggered
        line = raw_line.strip()
        if not line.startswith("{"):
            return None
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            return None

        status = data.get("status")
        if status == "warmup_done":
            pass
        elif status == "safeguard_triggered":
            safeguard_triggered = True
            safeguard_note = data.get("note", "")
        elif status == "rung_done":
            cond_type = data["cond_type"]
            voice_from = data["voice_from"]
            voice_to = data["voice_to"]
            duration_ms = data["duration_ms"]
            rows.append({
                "cond_type": cond_type,
                "voice_from": voice_from,
                "voice_to": voice_to,
                "duration_ms": duration_ms,
            })
            db.add_kokoro_voice_measurement(
                con, run_id,
                cond_type=cond_type,
                voice_from=voice_from,
                voice_to=voice_to,
                duration_ms=duration_ms
            )
        elif status == "error":
            console.emit(console.style("warn", f"ERROR: {data.get('note')}"))
            error_triggered = True
            return True  # signal _stream_worker to terminate the child immediately
        return None

    try:
        returncode, stderr_text = _stream_worker(cmd, on_line)
    except KeyboardInterrupt:
        console.emit(console.style("warn", "Benchmark interrupted by user."))
        sys.exit(1)

    if error_triggered:
        sys.exit(1)

    if returncode != 0:
        console.emit(console.style("warn", f"Worker exited with code {returncode}."))
        if stderr_text:
            console.emit(console.style("gloss", f"Stderr: {stderr_text}"))
        sys.exit(returncode)

    if safeguard_triggered:
        view_bench_kokoro_conc.render_safeguard(console, {
            "note": safeguard_note,
            "next_cmd": "wmx-suite benchmark-kokoro-voice --repeats <fewer>",
        })

    view_bench_kokoro_conc.render_voice(console, {
        "model": args.model,
        "voice_a": args.voice_a,
        "voice_b": args.voice_b,
        "repeats": args.repeats,
        "margin": margin_val,
        "rows": rows,
        "safeguard_triggered": safeguard_triggered,
        "run_id": run_id,
    })


def cmd_benchmark_kokoro_cache(args):
    """Run Kokoro TTS cache scaling performance sweep and log to database."""
    import json

    margin_val = _configured_margin(args.margin)

    py = sys.executable
    cmd = [
        py, "-m", "wmx_suite.probe_worker_kokoro_cache",
        "--model", args.model,
        "--cache-sizes", args.cache_sizes,
        "--margin", str(margin_val)
    ]

    import mlx.core as mx
    mlx_version = mx.__version__

    con = db.connect()
    run_id = db.start_kokoro_cache_run(con, args.model, mlx_version)

    console = getattr(args, "console", None) or CONSOLE

    rows: list[dict] = []
    safeguard_triggered = False
    safeguard_note: str = ""
    error_triggered = False

    def on_line(raw_line):
        nonlocal safeguard_triggered, safeguard_note, error_triggered
        line = raw_line.strip()
        if not line.startswith("{"):
            return None
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            return None

        status = data.get("status")
        if status == "warmup_done":
            pass
        elif status == "safeguard_triggered":
            safeguard_triggered = True
            safeguard_note = data.get("note", "")
        elif status == "rung_done":
            cache_size = data["cache_size"]
            os_wired_gb = data["os_wired_gb"]
            peak_gb = data["peak_gb"]
            rows.append({
                "cache_size": cache_size,
                "os_wired_gb": os_wired_gb,
                "peak_gb": peak_gb,
            })
            db.add_kokoro_cache_measurement(
                con, run_id,
                cache_size=cache_size,
                os_wired_gb=os_wired_gb,
                peak_gb=peak_gb
            )
        elif status == "error":
            console.emit(console.style("warn", f"ERROR: {data.get('note')}"))
            error_triggered = True
            return True  # signal _stream_worker to terminate the child immediately
        return None

    try:
        returncode, stderr_text = _stream_worker(cmd, on_line)
    except KeyboardInterrupt:
        console.emit(console.style("warn", "Benchmark interrupted by user."))
        sys.exit(1)

    if error_triggered:
        sys.exit(1)

    if returncode != 0:
        console.emit(console.style("warn", f"Worker exited with code {returncode}."))
        if stderr_text:
            console.emit(console.style("gloss", f"Stderr: {stderr_text}"))
        sys.exit(returncode)

    if safeguard_triggered:
        view_bench_kokoro_conc.render_safeguard(console, {
            "note": safeguard_note,
            "next_cmd": "wmx-suite benchmark-kokoro-cache --cache-sizes <smaller>",
        })

    view_bench_kokoro_conc.render_cache(console, {
        "model": args.model,
        "cache_sizes": args.cache_sizes,
        "margin": margin_val,
        "rows": rows,
        "safeguard_triggered": safeguard_triggered,
        "run_id": run_id,
    })


def cmd_benchmark_kokoro_baseline(args):
    """Run Kokoro TTS active baseline memory benchmark and log to database."""
    import json

    margin_val = _configured_margin(args.margin)

    py = sys.executable
    cmd = [
        py, "-m", "wmx_suite.probe_worker_kokoro_baseline",
        "--model", args.model,
        "--voice", args.voice,
        "--margin", str(margin_val)
    ]

    import mlx.core as mx
    mlx_version = mx.__version__

    con = db.connect()
    run_id = db.start_kokoro_baseline_run(con, args.model, mlx_version)

    console = getattr(args, "console", None) or CONSOLE

    baseline_result: dict = {}
    error_triggered = False

    def on_line(raw_line):
        nonlocal error_triggered
        line = raw_line.strip()
        if not line.startswith("{"):
            return None
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            return None

        status = data.get("status")
        if status == "rung_done":
            baseline_gb = data["baseline_gb"]
            active_gb = data["active_gb"]
            overhead_gb = data["overhead_gb"]
            baseline_result["baseline_gb"] = baseline_gb
            baseline_result["active_gb"] = active_gb
            baseline_result["overhead_gb"] = overhead_gb
            db.add_kokoro_baseline_measurement(
                con, run_id,
                baseline_gb=baseline_gb,
                active_gb=active_gb,
                overhead_gb=overhead_gb
            )
        elif status == "error":
            console.emit(console.style("warn", f"ERROR: {data.get('note')}"))
            error_triggered = True
            return True  # signal _stream_worker to terminate the child immediately
        return None

    try:
        # capture_stderr=False: inherit terminal stderr, matching original behaviour.
        returncode, stderr_text = _stream_worker(cmd, on_line, capture_stderr=False)
    except KeyboardInterrupt:
        console.emit(console.style("warn", "Benchmark interrupted by user."))
        sys.exit(1)

    if error_triggered:
        sys.exit(1)

    if returncode != 0:
        console.emit(console.style("warn", f"Worker exited with code {returncode}."))
        sys.exit(returncode)

    view_bench_kokoro.render_baseline(console, {
        "model": args.model,
        "voice": args.voice,
        "margin": margin_val,
        "baseline_gb": baseline_result.get("baseline_gb", 0.0),
        "active_gb": baseline_result.get("active_gb", 0.0),
        "overhead_gb": baseline_result.get("overhead_gb", 0.0),
        "run_id": run_id,
    })







RUN_HELP = """usage: wmx-suite run [--margin GB] [--force] [--dry-run] [--co-run-kokoro] -- <mlx_lm.generate args>

Safely launch mlx_lm.generate. Picks kv-bits by cache type, caps --max-kv-size from the
measured ceiling, and refuses if the run would breach the wall.
The passthrough args must include --model <hf_id>.

  --margin GB         safety cushion under the wall (default: WMX_SUITE_MARGIN_GB or 2.0)
  --force             launch even if the planner refuses (may crash the machine)
  --dry-run           print the plan, do not launch
  --no-log            do not record generation speed (bare exec passthrough)
  --co-run-kokoro     subtract Kokoro static active overhead from the safe ceiling
"""

_PROMPT_RE = re.compile(r"Prompt:\s*(\d+)\s*tokens,\s*([\d.]+)\s*tokens-per-sec")
_GEN_RE = re.compile(r"Generation:\s*(\d+)\s*tokens,\s*([\d.]+)\s*tokens-per-sec")
_PEAK_RE = re.compile(r"Peak memory:\s*([\d.]+)\s*GB")


def _record_generation(text: str, model_id: str, max_kv_size: int) -> None:
    """Parse mlx_lm's tok/s lines from captured output and store them. Best-effort —
    a cancelled or errored run has no generation stats, so nothing is logged."""
    gm = _GEN_RE.search(text)
    if not gm:
        return
    pm, km = _PROMPT_RE.search(text), _PEAK_RE.search(text)
    try:
        con = db.connect()
        db.log_generation(
            con, model_id,
            prompt_tokens=int(pm.group(1)) if pm else None,
            prompt_tps=float(pm.group(2)) if pm else None,
            gen_tokens=int(gm.group(1)),
            gen_tps=float(gm.group(2)),
            peak_gb=float(km.group(1)) if km else None,
            max_kv_size=max_kv_size,
        )
        print(f"[run] logged {gm.group(1)} tok @ {float(gm.group(2)):.1f} tok/s", file=sys.stderr)
    except Exception as e:  # logging must never break a run
        print(f"[run] (speed log failed: {e})", file=sys.stderr)


def _exec_logged(argv: list[str], model_id: str, max_kv_size: int) -> None:
    """Run mlx_lm.generate under a PTY so output streams live to the terminal unchanged,
    while we capture a copy to parse its tok/s stats. A PTY (not a plain pipe) is required:
    Python block-buffers stdout when it isn't a tty, which would batch the token stream."""
    master, slave = pty.openpty()
    if sys.stdout.isatty():  # match the child's window size so tqdm renders correctly
        try:
            cols, rows = os.get_terminal_size(sys.stdout.fileno())
            fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))
        except Exception:
            pass
    proc = subprocess.Popen(["mlx_lm.generate"] + argv,
                            stdin=slave, stdout=slave, stderr=slave, close_fds=True)
    os.close(slave)
    captured = bytearray()
    try:
        while True:
            try:
                data = os.read(master, 4096)
            except OSError:  # EIO once the child exits and closes its end
                break
            if not data:
                break
            os.write(1, data)  # tee live to our stdout, byte-for-byte
            captured.extend(data)
    except KeyboardInterrupt:
        try:
            proc.send_signal(signal.SIGINT)
        except Exception:
            pass
    finally:
        os.close(master)
    rc = proc.wait()
    _record_generation(captured.decode(errors="replace"), model_id, max_kv_size)
    sys.exit(rc)


def cmd_run_raw(run_args: list[str]):
    """Parse leading suite flags, then treat the remainder as mlx_lm.generate passthrough.

    Done manually (not argparse) because argparse.REMAINDER mishandles optionals that
    precede the positional, and we want `run --model X ...` to work as a drop-in.
    """
    margin, force, dry, log = None, False, False, True
    co_run_kokoro = False
    i = 0
    while i < len(run_args):
        a = run_args[i]
        if a in ("-h", "--help"):
            print(RUN_HELP); return
        if a == "--margin":
            if i + 1 >= len(run_args):
                raise SystemExit("[run] --margin requires a value")
            margin = run_args[i + 1]; i += 2; continue
        if a.startswith("--margin="):
            margin = a.split("=", 1)[1]; i += 1; continue
        if a == "--force":
            force = True; i += 1; continue
        if a == "--dry-run":
            dry = True; i += 1; continue
        if a == "--no-log":
            log = False; i += 1; continue
        if a == "--co-run-kokoro":
            co_run_kokoro = True; i += 1; continue
        break  # first non-suite token: the rest is passthrough
    rest = run_args[i:]
    if rest and rest[0] == "--":
        rest = rest[1:]
    _run(rest, margin=margin, force=force, dry_run=dry, log=log, co_run_kokoro=co_run_kokoro)


def _run(rest: list[str], *, margin: float | str | None, force: bool,
         dry_run: bool, log: bool = True, co_run_kokoro: bool = False):
    """Crash-safe launch: plan a launch, then exec mlx_lm.generate."""
    model_id = None
    for i, a in enumerate(rest):
        if a == "--model" and i + 1 < len(rest):
            model_id = rest[i + 1]
            break
        if a.startswith("--model="):
            model_id = a.split("=", 1)[1]
            break
    if model_id is None:
        raise SystemExit("[run] --model is required")

    margin_gb = _configured_margin(margin)
    if co_run_kokoro:
        kokoro_overhead = 0.85  # default fallback
        try:
            con = db.connect()
            latest_base = db.get_latest_kokoro_baseline(con)
            if latest_base and latest_base.get("overhead_gb") is not None:
                kokoro_overhead = float(latest_base["overhead_gb"])
        except Exception:
            pass
        margin_gb += kokoro_overhead

    p = launcher.plan(model_id, margin_gb=margin_gb)
    if p.get("error"):
        raise SystemExit(f"[run] {p['error']}")

    if p["source"] == "estimated" and not force:
        if sys.stdin.isatty():
            print(f"[run] Model {model_id} has not been characterized yet.", file=sys.stderr)
            try:
                ans = input("[run] Characterize it now to find its safe context ceiling? [y/N]: ").strip().lower()
            except KeyboardInterrupt:
                print(file=sys.stderr)
                raise SystemExit("[run] Aborted.")
            if ans in ("y", "yes"):
                print(f"[run] Running characterization for {model_id}...", file=sys.stderr)
                try:
                    probe.characterize(model_id, margin_gb=margin_gb, allow_min_probe=True)
                    # Re-plan with the newly saved fit
                    p = launcher.plan(model_id, margin_gb=margin_gb)
                    if p.get("error"):
                        raise SystemExit(f"[run] Re-planning failed: {p['error']}")
                except Exception as e:
                    raise SystemExit(f"[run] Characterization failed: {e}")
            else:
                raise SystemExit("[run] Aborted. Run characterize first or pass --force to run with estimated limits.")
        else:
            raise SystemExit("[run] REFUSED: Model is uncharacterized and shell is non-interactive. Run characterize first or pass --force.")

    # Run diagnostics render to stderr (keeps the model's stdout clean), with the
    # same color/verbose policy as the rest of the CLI. Built at call time so
    # captured streams (tests) and the real stderr both work.
    console = Console(color=CONSOLE.color, verbose=CONSOLE.verbose, stream=sys.stderr)
    kv_mode = ("fp16 (not quantizable)" if p["kv_bits"] is None
               else f"{p['kv_bits']}-bit")
    if p.get("fit_stale"):
        console.emit(console.style(
            "warn", "fit may be stale — consider re-running 'wmx-suite characterize'."))
    if p["source"] == "estimated" and p.get("cold_start_profile") == "default":
        console.emit(console.style(
            "warn", "using fallback cold-start constants (Apple M4 Pro testbed); "
                    "run 'wmx-suite calibrate' to tune them for this machine."))

    if p.get("refuse"):
        view_run.render_refusal(console, {
            "model": model_id,
            "needs_gb": p["base_abs_gb"],
            "budget_gb": p["threshold_gb"],
            "wall_gb": p["wall_gb"],
            "live_base_gb": p["live_base_gb"],
            "model_base_gb": p["model_base_gb"],
            "slope_gb_per_k": p["slope_gb_per_k"],
            "safe_cap_tok": p.get("max_kv_size", 0) or 0,
            "source": p["source"],
            "cache_type": p["cache_type"],
            "kv_mode": kv_mode,
        })
        if not force:
            sys.exit(2)
        console.emit(console.style(
            "warn", "--force given; proceeding against safety advice."))

    try:
        argv = launcher.build_argv(rest, p, force=force)
    except launcher.LaunchArgumentError as exc:
        raise SystemExit(f"[run] REFUSED: {exc}") from exc
    effective_cap = launcher.effective_max_kv_size(rest, p)
    prompt_check = None
    try:
        tokenizer = load_tokenizer(
            model_id,
            tokenizer_config_extra=launcher.tokenizer_config(rest),
        )
        prompt_check = launcher.check_prompt(rest, p, tokenizer)
    except launcher.LaunchArgumentError as exc:
        if not force:
            raise SystemExit(f"[run] REFUSED: {exc}") from exc
        print(f"[run] WARNING: prompt preflight bypassed: {exc}", file=sys.stderr)
    except Exception as exc:
        if not force:
            raise SystemExit(
                f"[run] REFUSED: prompt tokenization failed: {exc}; "
                "pass --force to bypass prompt preflight"
            ) from exc
        print(f"[run] WARNING: prompt tokenization failed and was bypassed: {exc}",
              file=sys.stderr)
    if not p.get("max_kv_size_enforced", True):
        print("[run] WARNING: this model's custom MLX cache does not enforce "
              "--max-kv-size", file=sys.stderr)
    if prompt_check is not None:
        print(f"[run] prompt {prompt_check.tokens:,} tokens "
              f"({prompt_check.tokens / prompt_check.cap:.0%} of cap)",
              file=sys.stderr)
        if prompt_check.tokens > prompt_check.cap:
            message = (f"prompt is {prompt_check.tokens:,} tokens, above "
                       f"the {prompt_check.cap:,}-token cap")
            if not force:
                raise SystemExit(f"[run] REFUSED: {message}")
            print(f"[run] WARNING: {message}; --force is overriding this refusal",
                  file=sys.stderr)
        if prompt_check.warn:
            print(f"[run] WARNING: prompt exceeds "
                  f"{launcher.PROMPT_WARNING_FRACTION:.0%} of the context cap",
                  file=sys.stderr)
    view_run.render_plan(console, {
        "model": model_id,
        "source": p["source"],
        "cache_type": p["cache_type"],
        "kv_mode": kv_mode,
        "live_base_gb": p["live_base_gb"],
        "model_base_gb": p["model_base_gb"],
        "budget_gb": p["threshold_gb"],
        "wall_gb": p["wall_gb"],
        "slope_gb_per_k": p["slope_gb_per_k"],
        "max_kv_size": effective_cap,
        "model_max": p["model_max"],
        "max_kv_size_enforced": p.get("max_kv_size_enforced", True),
    })
    print(f"[run] exec: mlx_lm.generate {' '.join(argv)}\n", file=sys.stderr)
    if dry_run:
        print("[run] --dry-run: not launching.", file=sys.stderr)
        return
    if log:
        _exec_logged(argv, model_id, effective_cap)
    else:
        os.execvp("mlx_lm.generate", ["mlx_lm.generate"] + argv)


def _build_global_parser() -> argparse.ArgumentParser:
    """Parent parser providing the global --verbose/--no-color flags.

    Passed as ``parents=[...]`` to every subcommand so the flags are accepted
    after any command (e.g. ``wmx-suite system --verbose``).
    """
    g = argparse.ArgumentParser(add_help=False)
    g.add_argument("--verbose", "-v", action="store_true",
                   help="show the power-user appendix (raw numbers) on each command")
    g.add_argument("--no-color", action="store_true",
                   help="never emit ANSI color, even on a TTY")
    return g


def _build_parser() -> argparse.ArgumentParser:
    gp = _build_global_parser()
    ap = argparse.ArgumentParser(prog="wmx-suite")
    sub = ap.add_subparsers(dest="cmd", required=False, parser_class=lambda **kw:
                            argparse.ArgumentParser(parents=[gp], **kw))
    sub.add_parser("system").set_defaults(func=cmd_system)
    p = sub.add_parser("health")
    p.add_argument("--margin", default=None,
                   help="safety cushion in GB (overrides WMX_SUITE_MARGIN_GB)")
    p.set_defaults(func=cmd_health)
    sub.add_parser("scan").set_defaults(func=cmd_scan)
    p = sub.add_parser("show"); p.add_argument("hf_id"); p.set_defaults(func=cmd_show)
    p = sub.add_parser("characterize"); p.add_argument("hf_id")
    p.add_argument("--margin", default=None,
                   help="safety cushion in GB (overrides WMX_SUITE_MARGIN_GB)")
    p.add_argument("--min-probe", action="store_true",
                   help="for borderline models, run a supervised 512-token probe to measure "
                        "the true base instead of refusing on the pessimistic estimate")
    p.add_argument("--repeats", type=int, default=probe.DEFAULT_REPEATS,
                   help="isolated runs per context rung; the median high-water is used "
                        "(smooths prefill-transient jitter)")
    p.set_defaults(func=cmd_characterize)
    sub.add_parser("list").set_defaults(func=cmd_list)
    p = sub.add_parser("calibrate", help="measure this machine's cold-start overhead constant")
    p.add_argument("--model", default=None,
                   help="model to calibrate with (default: smallest cached causal model)")
    p.add_argument("--margin", default=None,
                   help="safety cushion in GB (overrides WMX_SUITE_MARGIN_GB)")
    p.set_defaults(func=cmd_calibrate)
    p = sub.add_parser("web", help="Launch the Flask web UI dashboard")
    p.add_argument("--host", default="127.0.0.1", help="Host interface to bind to")
    p.add_argument("--port", type=int, default=5001, help="Port to listen on")
    p.add_argument("--debug", action="store_true", help="Run in Flask debug mode")
    p.set_defaults(func=cmd_web)
    # `run` is intercepted before argparse (see below) so it can pass arbitrary flags
    # through to mlx_lm.generate; this stub only makes it show up in `--help`.
    sub.add_parser("run", help="safely launch mlx_lm.generate: picks kv-bits by cache "
                               "type, caps --max-kv-size from the measured ceiling, "
                               "refuses if it would breach the wall")
    
    p = sub.add_parser("benchmark-kokoro", help="Benchmark Kokoro TTS performance")
    p.add_argument("--model", default="mlx-community/Kokoro-82M-bf16",
                   help="HuggingFace model ID or path")
    p.add_argument("--voice", default="af_heart",
                   help="Voice name to use")
    p.add_argument("--lengths", default="10,50,100,200,500,1000,2000,3000",
                   help="Comma-separated character lengths to sweep")
    p.add_argument("--repeats", type=int, default=3,
                   help="Number of trials per length")
    p.add_argument("--margin", default=None,
                   help="safety cushion in GB (overrides WMX_SUITE_MARGIN_GB)")
    p.set_defaults(func=cmd_benchmark_kokoro)

    p = sub.add_parser("benchmark-kokoro-ttfa", help="Benchmark Kokoro TTS TTFA latency")
    p.add_argument("--model", default="mlx-community/Kokoro-82M-bf16",
                   help="HuggingFace model ID or path")
    p.add_argument("--voice", default="af_heart",
                   help="Voice name to use")
    p.add_argument("--lengths", default="10,50,100,200,500,1000,2000,3000",
                   help="Comma-separated character lengths to sweep")
    p.add_argument("--repeats", type=int, default=3,
                   help="Number of trials per length")
    p.add_argument("--margin", default=None,
                   help="safety cushion in GB (overrides WMX_SUITE_MARGIN_GB)")
    p.set_defaults(func=cmd_benchmark_kokoro_ttfa)

    p = sub.add_parser("benchmark-kokoro-batch", help="Benchmark Kokoro TTS batch performance (concurrency vs throughput)")
    p.add_argument("--model", default="mlx-community/Kokoro-82M-bf16",
                   help="HuggingFace model ID or path")
    p.add_argument("--voice", default="af_heart",
                   help="Voice name to use")
    p.add_argument("--batch-sizes", default="1,2,4,8,16",
                   help="Comma-separated batch sizes to sweep")
    p.add_argument("--repeats", type=int, default=3,
                   help="Number of trials per batch size")
    p.add_argument("--margin", default=None,
                   help="safety cushion in GB (overrides WMX_SUITE_MARGIN_GB)")
    p.set_defaults(func=cmd_benchmark_kokoro_batch)

    p = sub.add_parser("benchmark-kokoro-voice", help="Benchmark Kokoro TTS voice switching performance")
    p.add_argument("--model", default="mlx-community/Kokoro-82M-bf16",
                   help="HuggingFace model ID or path")
    p.add_argument("--voice-a", default="af_heart",
                   help="Voice name A")
    p.add_argument("--voice-b", default="am_adam",
                   help="Voice name B")
    p.add_argument("--repeats", type=int, default=5,
                   help="Number of trials")
    p.add_argument("--margin", default=None,
                   help="safety cushion in GB (overrides WMX_SUITE_MARGIN_GB)")
    p.set_defaults(func=cmd_benchmark_kokoro_voice)

    p = sub.add_parser("benchmark-kokoro-cache", help="Benchmark Kokoro TTS voice cache memory overhead")
    p.add_argument("--model", default="mlx-community/Kokoro-82M-bf16",
                   help="HuggingFace model ID or path")
    p.add_argument("--cache-sizes", default="0,1,2,4,8,16,24,32",
                   help="Comma-separated cache sizes to sweep")
    p.add_argument("--margin", default=None,
                   help="safety cushion in GB (overrides WMX_SUITE_MARGIN_GB)")
    p.set_defaults(func=cmd_benchmark_kokoro_cache)

    p = sub.add_parser("benchmark-kokoro-baseline", help="Benchmark Kokoro TTS static active RAM overhead")
    p.add_argument("--model", default="mlx-community/Kokoro-82M-bf16",
                   help="HuggingFace model ID or path")
    p.add_argument("--voice", default="af_heart",
                   help="Voice name")
    p.add_argument("--margin", default=None,
                   help="safety cushion in GB (overrides WMX_SUITE_MARGIN_GB)")
    p.set_defaults(func=cmd_benchmark_kokoro_baseline)

    p = sub.add_parser("benchmark-embeddings",
                       help="Benchmark ModernBERT embeddings batch x seq memory scaling")
    p.add_argument("--model", default="mlx-community/nomicai-modernbert-embed-base-bf16",
                   help="HuggingFace model ID or path")
    p.add_argument("--batches", default="1,2,4,8,16,32",
                   help="Comma-separated batch sizes to sweep")
    p.add_argument("--seqs", default="128,256,512,1024,2048,4096,8192",
                   help="Comma-separated sequence lengths to sweep")
    p.add_argument("--repeats", type=int, default=3,
                   help="Forward passes per cell (median timing, max memory)")
    p.add_argument("--margin", default=None,
                   help="safety cushion in GB (overrides WMX_SUITE_MARGIN_GB)")
    p.add_argument("--ignore-profile", action="store_true",
                   help="ignore any stored calibration profile (cold start); still re-fits")
    p.set_defaults(func=cmd_benchmark_embeddings)
    return ap


def _main_argparse(argv=None):
    ap = _build_parser()
    args = ap.parse_args(argv)
    args.console = Console.from_args(
        no_color=getattr(args, "no_color", False),
        verbose=getattr(args, "verbose", False),
    )
    args.func(args)


def _strip_global_flags(argv: list[str]) -> tuple[list[str], bool, bool]:
    """Pull --verbose/-v/--no-color out of *argv* (for the `run` fast path).

    `run` is intercepted before argparse so it can forward arbitrary flags to
    mlx_lm.generate; the global UX flags must not leak into that passthrough.
    Returns ``(remaining_argv, verbose, no_color)``.
    """
    remaining: list[str] = []
    verbose = no_color = False
    for a in argv:
        if a in ("--verbose", "-v"):
            verbose = True
        elif a == "--no-color":
            no_color = True
        else:
            remaining.append(a)
    return remaining, verbose, no_color


def cmd_landing(console: Console) -> None:
    """Front door: shown for `wmx-suite` with no subcommand.

    Gathers a live machine status line and renders the grouped command
    overview. ``console`` carries the --verbose/--no-color policy. Degrades
    gracefully if the machine can't be read (shows the screen with zeros).
    """
    try:
        s = read_limits()
        margin = _configured_margin()
        safe = s.safe_threshold_gb(margin)
        con = db.connect()
        n_ready = con.execute(
            "SELECT COUNT(DISTINCT m.hf_id) FROM models m "
            "JOIN probe_runs r ON r.hf_id = m.hf_id JOIN fits f ON f.run_id = r.id"
        ).fetchone()[0]
        calibrated = db.get_profile(con, profiles.machine_key()) is not None
        data = {
            "device": s.device,
            "free_gb": safe - s.wired_now_gb,
            "safe_budget_gb": safe,
            "models_ready": n_ready,
            "calibrated": calibrated,
        }
    except Exception:
        data = {"device": "unknown", "free_gb": 0.0, "safe_budget_gb": 0.0,
                "models_ready": 0, "calibrated": False}
    view_landing.render(console, data)


def main():
    argv = sys.argv[1:]
    if argv and argv[0] == "run":
        run_args, verbose, no_color = _strip_global_flags(argv[1:])
        # Run diagnostics go to stderr (model output owns stdout); color policy
        # follows stderr's TTY. _run reads CONSOLE's color/verbose policy.
        global CONSOLE
        CONSOLE = Console.from_args(stream=sys.stderr, no_color=no_color, verbose=verbose)
        return cmd_run_raw(run_args)
    # Front door: no subcommand present (empty, or only global/help flags).
    rest, verbose, no_color = _strip_global_flags(argv)
    if not rest or rest[0] in ("-h", "--help"):
        console = Console.from_args(no_color=no_color, verbose=verbose)
        return cmd_landing(console)
    return _main_argparse(argv)


if __name__ == "__main__":
    main()
