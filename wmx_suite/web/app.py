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

app = None
if Flask is not None:
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


def create_app():
    if Flask is None:
        raise ImportError("Flask is required to run the web UI.")
    return app
