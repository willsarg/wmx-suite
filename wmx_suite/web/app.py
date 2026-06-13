"""Flask application for wmx-suite dashboard.

Surfaces SQLite characterization data and live system memory limits in a browser.
"""
from __future__ import annotations

import os
from statistics import median

try:
    from flask import Flask, render_template, jsonify, g, abort
except ImportError:
    # Fallback if imported outside the web CLI context
    Flask = None

from wmx_suite import db, system, config, models


def create_app():
    if Flask is None:
        raise ImportError("Flask is required to run the web UI.")

    app = Flask(__name__)

    # Custom jinja filters for formatting
    @app.template_filter("format_number")
    def format_number(val):
        if val is None:
            return ""
        try:
            return f"{val:,}"
        except (ValueError, TypeError):
            return str(val)

    def get_db():
        if "db" not in g:
            g.db = db.connect()
        return g.db

    @app.teardown_appcontext
    def close_db(e):
        db_conn = g.pop("db", None)
        if db_conn is not None:
            try:
                db_conn.close()
            except Exception:
                pass

    @app.route("/")
    def index():
        con = get_db()
        fits = db.latest_fits(con)
        speeds = db.gen_speeds(con)
        limits = system.read_limits()
        margin = config.margin_gb()
        threshold = limits.safe_threshold_gb(margin)

        # Decorate fits with median speed, stale status, and progress towards wall
        for f in fits:
            hf_id = f["hf_id"]
            if hf_id in speeds:
                f["median_speed"] = round(median(speeds[hf_id]), 1)
                f["speed_runs"] = len(speeds[hf_id])
            else:
                f["median_speed"] = None
                f["speed_runs"] = 0
            f["fit_stale"] = models.fit_is_stale(hf_id, f["characterized_at"])

        return render_template(
            "dashboard.html",
            fits=fits,
            limits=limits,
            margin=margin,
            threshold=threshold,
        )

    @app.route("/model/<path:hf_id>")
    def model_detail(hf_id):
        con = get_db()
        model = db.get_model(con, hf_id)
        if not model:
            abort(404, description=f"Model not found: {hf_id}")

        runs = db.get_model_runs_and_fits(con, hf_id)
        latest_fit = db.latest_fit(con, hf_id)
        speeds = db.gen_speeds(con).get(hf_id, [])
        med_speed = round(median(speeds), 1) if speeds else None

        # Fetch measurements for the latest run if it exists
        latest_measurements = []
        if latest_fit:
            latest_measurements = db.get_measurements(con, latest_fit["run_id"])

        # Determine if fit is stale
        fit_stale = False
        if latest_fit:
            fit_stale = models.fit_is_stale(hf_id, latest_fit["characterized_at"])

        limits = system.read_limits()
        margin = config.margin_gb()
        threshold = limits.safe_threshold_gb(margin)

        return render_template(
            "model.html",
            model=model,
            runs=runs,
            latest_fit=latest_fit,
            measurements=latest_measurements,
            median_speed=med_speed,
            speed_runs=len(speeds),
            fit_stale=fit_stale,
            limits=limits,
            margin=margin,
            threshold=threshold,
        )

    @app.route("/compare")
    def compare():
        con = get_db()
        fits = db.latest_fits(con)
        limits = system.read_limits()
        margin = config.margin_gb()
        threshold = limits.safe_threshold_gb(margin)

        return render_template(
            "compare.html",
            fits=fits,
            limits=limits,
            threshold=threshold,
        )

    @app.route("/kokoro")
    def kokoro_dashboard():
        con = get_db()
        runs = db.get_all_kokoro_runs(con)
        limits = system.read_limits()

        decorated_runs = []
        for r in runs:
            m = db.get_kokoro_measurements(con, r["id"])
            if m:
                avg_rtf = round(sum(x["rtf"] for x in m) / len(m), 4)
                avg_cps = round(sum(x["cps"] for x in m) / len(m), 1)
                # Find max peak_gb safely
                peaks = [x["peak_gb"] for x in m if x["peak_gb"] is not None]
                peak_gb = round(max(peaks), 2) if peaks else None
                decorated_runs.append({
                    "id": r["id"],
                    "model_id": r["model_id"],
                    "voice": r["voice"],
                    "mlx_version": r["mlx_version"],
                    "created_at": r["created_at"],
                    "n_sweeps": len(m),
                    "avg_rtf": avg_rtf,
                    "avg_cps": avg_cps,
                    "peak_gb": peak_gb
                })
            else:
                decorated_runs.append({
                    "id": r["id"],
                    "model_id": r["model_id"],
                    "voice": r["voice"],
                    "mlx_version": r["mlx_version"],
                    "created_at": r["created_at"],
                    "n_sweeps": 0,
                    "avg_rtf": None,
                    "avg_cps": None,
                    "peak_gb": None
                })

        return render_template(
            "kokoro_dashboard.html",
            runs=decorated_runs,
            limits=limits
        )

    @app.route("/kokoro/run/<int:run_id>")
    def kokoro_run_detail(run_id):
        con = get_db()
        runs = db.get_all_kokoro_runs(con)
        run = next((r for r in runs if r["id"] == run_id), None)
        if not run:
            abort(404, description=f"Kokoro Run not found: {run_id}")

        measurements = db.get_kokoro_measurements(con, run_id)
        limits = system.read_limits()

        return render_template(
            "kokoro_run.html",
            run=run,
            measurements=measurements,
            limits=limits
        )

    @app.route("/kokoro-ttfa")
    def kokoro_ttfa_dashboard():
        con = get_db()
        runs = db.get_all_kokoro_ttfa_runs(con)
        limits = system.read_limits()

        decorated_runs = []
        for r in runs:
            m = db.get_kokoro_ttfa_measurements(con, r["id"])
            if m:
                avg_ttfa = round(sum(x["ttfa_sec"] for x in m) / len(m), 3)
                avg_total = round(sum(x["total_sec"] for x in m) / len(m), 3)
                speedups = [x["speedup_ratio"] for x in m if x["speedup_ratio"] is not None]
                max_speedup = round(max(speedups), 1) if speedups else None
                peaks = [x["peak_gb"] for x in m if x["peak_gb"] is not None]
                peak_gb = round(max(peaks), 2) if peaks else None
                decorated_runs.append({
                    "id": r["id"],
                    "model_id": r["model_id"],
                    "voice": r["voice"],
                    "mlx_version": r["mlx_version"],
                    "created_at": r["created_at"],
                    "n_sweeps": len(m),
                    "avg_ttfa": avg_ttfa,
                    "avg_total": avg_total,
                    "max_speedup": max_speedup,
                    "peak_gb": peak_gb
                })
            else:
                decorated_runs.append({
                    "id": r["id"],
                    "model_id": r["model_id"],
                    "voice": r["voice"],
                    "mlx_version": r["mlx_version"],
                    "created_at": r["created_at"],
                    "n_sweeps": 0,
                    "avg_ttfa": None,
                    "avg_total": None,
                    "max_speedup": None,
                    "peak_gb": None
                })

        return render_template(
            "kokoro_ttfa_dashboard.html",
            runs=decorated_runs,
            limits=limits
        )

    @app.route("/kokoro-ttfa/run/<int:run_id>")
    def kokoro_ttfa_run_detail(run_id):
        con = get_db()
        runs = db.get_all_kokoro_ttfa_runs(con)
        run = next((r for r in runs if r["id"] == run_id), None)
        if not run:
            abort(404, description=f"Kokoro TTFA Run not found: {run_id}")

        measurements = db.get_kokoro_ttfa_measurements(con, run_id)
        limits = system.read_limits()

        return render_template(
            "kokoro_ttfa_run.html",
            run=run,
            measurements=measurements,
            limits=limits
        )

    @app.route("/kokoro-batch")
    def kokoro_batch_dashboard():
        con = get_db()
        runs = db.get_all_kokoro_batch_runs(con)
        limits = system.read_limits()

        decorated_runs = []
        for r in runs:
            m = db.get_kokoro_batch_measurements(con, r["id"])
            if m:
                max_cps = round(max(x["cps"] for x in m), 1)
                best_m = max(m, key=lambda x: x["cps"])
                optimal_batch = best_m["batch_size"]
                peaks = [x["peak_gb"] for x in m if x["peak_gb"] is not None]
                peak_gb = round(max(peaks), 2) if peaks else None
                decorated_runs.append({
                    "id": r["id"],
                    "model_id": r["model_id"],
                    "voice": r["voice"],
                    "mlx_version": r["mlx_version"],
                    "created_at": r["created_at"],
                    "n_sweeps": len(m),
                    "max_throughput": max_cps,
                    "optimal_batch_size": optimal_batch,
                    "peak_gb": peak_gb
                })
            else:
                decorated_runs.append({
                    "id": r["id"],
                    "model_id": r["model_id"],
                    "voice": r["voice"],
                    "mlx_version": r["mlx_version"],
                    "created_at": r["created_at"],
                    "n_sweeps": 0,
                    "max_throughput": None,
                    "optimal_batch_size": None,
                    "peak_gb": None
                })

        return render_template(
            "kokoro_batch_dashboard.html",
            runs=decorated_runs,
            limits=limits
        )

    @app.route("/kokoro-batch/run/<int:run_id>")
    def kokoro_batch_run_detail(run_id):
        con = get_db()
        runs = db.get_all_kokoro_batch_runs(con)
        run = next((r for r in runs if r["id"] == run_id), None)
        if not run:
            abort(404, description=f"Kokoro Batch Run not found: {run_id}")

        measurements = db.get_kokoro_batch_measurements(con, run_id)
        limits = system.read_limits()

        return render_template(
            "kokoro_batch_run.html",
            run=run,
            measurements=measurements,
            limits=limits
        )

    @app.route("/kokoro-voice")
    def kokoro_voice_dashboard():
        con = get_db()
        runs = db.get_all_kokoro_voice_runs(con)
        limits = system.read_limits()

        decorated_runs = []
        for r in runs:
            m = db.get_kokoro_voice_measurements(con, r["id"])
            if m:
                colds = [x["duration_ms"] for x in m if x["cond_type"] == "cold_load"]
                warms = [x["duration_ms"] for x in m if x["cond_type"] == "warm_switch"]
                statics = [x["duration_ms"] for x in m if x["cond_type"] == "static_baseline"]

                avg_cold = round(sum(colds) / len(colds), 1) if colds else None
                avg_warm = round(sum(warms) / len(warms), 1) if warms else None
                avg_static = round(sum(statics) / len(statics), 1) if statics else None

                decorated_runs.append({
                    "id": r["id"],
                    "model_id": r["model_id"],
                    "mlx_version": r["mlx_version"],
                    "created_at": r["created_at"],
                    "n_sweeps": len(m),
                    "avg_cold_ms": avg_cold,
                    "avg_warm_ms": avg_warm,
                    "avg_static_ms": avg_static
                })
            else:
                decorated_runs.append({
                    "id": r["id"],
                    "model_id": r["model_id"],
                    "mlx_version": r["mlx_version"],
                    "created_at": r["created_at"],
                    "n_sweeps": 0,
                    "avg_cold_ms": None,
                    "avg_warm_ms": None,
                    "avg_static_ms": None
                })

        return render_template(
            "kokoro_voice_dashboard.html",
            runs=decorated_runs,
            limits=limits
        )

    @app.route("/kokoro-voice/run/<int:run_id>")
    def kokoro_voice_run_detail(run_id):
        con = get_db()
        runs = db.get_all_kokoro_voice_runs(con)
        run = next((r for r in runs if r["id"] == run_id), None)
        if not run:
            abort(404, description=f"Kokoro Voice Run not found: {run_id}")

        measurements = db.get_kokoro_voice_measurements(con, run_id)
        limits = system.read_limits()

        return render_template(
            "kokoro_voice_run.html",
            run=run,
            measurements=measurements,
            limits=limits
        )

    @app.route("/kokoro-cache")
    def kokoro_cache_dashboard():
        con = get_db()
        runs = db.get_all_kokoro_cache_runs(con)
        limits = system.read_limits()

        decorated_runs = []
        for r in runs:
            m = db.get_kokoro_cache_measurements(con, r["id"])
            if m:
                peaks = [x["peak_gb"] for x in m]
                max_peak = round(max(peaks), 2) if peaks else None
                max_wired = round(max(x["os_wired_gb"] for x in m), 2)

                base_m = next((x for x in m if x["cache_size"] == 0), None)
                base_wired = base_m["os_wired_gb"] if base_m else None
                max_m = max(m, key=lambda x: x["cache_size"])
                max_wired_size = max_m["cache_size"]
                max_m_wired = max_m["os_wired_gb"]

                overhead = None
                if base_wired is not None:
                    overhead = round(max_m_wired - base_wired, 3)

                decorated_runs.append({
                    "id": r["id"],
                    "model_id": r["model_id"],
                    "mlx_version": r["mlx_version"],
                    "created_at": r["created_at"],
                    "n_sweeps": len(m),
                    "max_peak_gb": max_peak,
                    "max_wired_gb": max_wired,
                    "voices_cached": max_wired_size,
                    "cache_overhead_gb": overhead
                })
            else:
                decorated_runs.append({
                    "id": r["id"],
                    "model_id": r["model_id"],
                    "mlx_version": r["mlx_version"],
                    "created_at": r["created_at"],
                    "n_sweeps": 0,
                    "max_peak_gb": None,
                    "max_wired_gb": None,
                    "voices_cached": None,
                    "cache_overhead_gb": None
                })

        return render_template(
            "kokoro_cache_dashboard.html",
            runs=decorated_runs,
            limits=limits
        )

    @app.route("/kokoro-cache/run/<int:run_id>")
    def kokoro_cache_run_detail(run_id):
        con = get_db()
        runs = db.get_all_kokoro_cache_runs(con)
        run = next((r for r in runs if r["id"] == run_id), None)
        if not run:
            abort(404, description=f"Kokoro Cache Run not found: {run_id}")

        measurements = db.get_kokoro_cache_measurements(con, run_id)
        limits = system.read_limits()

        return render_template(
            "kokoro_cache_run.html",
            run=run,
            measurements=measurements,
            limits=limits
        )

    @app.route("/kokoro-baseline")
    def kokoro_baseline_dashboard():
        con = get_db()
        runs = db.get_all_kokoro_baseline_runs(con)
        limits = system.read_limits()
        latest_base = db.get_latest_kokoro_baseline(con)

        decorated_runs = []
        for r in runs:
            m = db.get_kokoro_baseline_measurements(con, r["id"])
            if m:
                m0 = m[0]
                decorated_runs.append({
                    "id": r["id"],
                    "model_id": r["model_id"],
                    "mlx_version": r["mlx_version"],
                    "created_at": r["created_at"],
                    "baseline_gb": round(m0["baseline_gb"], 3),
                    "active_gb": round(m0["active_gb"], 3),
                    "overhead_gb": round(m0["overhead_gb"], 3)
                })
            else:
                decorated_runs.append({
                    "id": r["id"],
                    "model_id": r["model_id"],
                    "mlx_version": r["mlx_version"],
                    "created_at": r["created_at"],
                    "baseline_gb": None,
                    "active_gb": None,
                    "overhead_gb": None
                })

        return render_template(
            "kokoro_baseline.html",
            runs=decorated_runs,
            latest_baseline=latest_base,
            limits=limits
        )

    @app.route("/api/system")
    def api_system():
        limits = system.read_limits()
        margin = config.margin_gb()
        threshold = limits.safe_threshold_gb(margin)
        return jsonify({
            "device": limits.device,
            "total_gb": round(limits.total_gb, 2),
            "wall_gb": round(limits.wall_gb, 2),
            "max_buffer_gb": round(limits.max_buffer_gb, 2),
            "swap_free_gb": round(limits.swap_free_gb, 2) if limits.swap_free_gb is not None else None,
            "wired_now_gb": round(limits.wired_now_gb, 2),
            "safe_threshold_gb": round(threshold, 2),
        })

    @app.route("/api/model/<path:hf_id>/measurements/<int:run_id>")
    def api_measurements(hf_id, run_id):
        con = get_db()
        measurements = db.get_measurements(con, run_id)
        return jsonify(measurements)

    return app
