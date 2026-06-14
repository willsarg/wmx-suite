import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from wmx_suite import db


def test_embeddings_db_lifecycle(monkeypatch, tmp_path):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "suite.db")
    con = db.connect()

    run_id = db.start_embeddings_run(con, "mlx-community/test-modernbert", "0.31.2")
    assert isinstance(run_id, int)

    db.add_embeddings_measurement(
        con, run_id, batch_size=2, seq_len=512,
        os_wired_gb=4.5, peak_gb=3.1, throughput_tps=12345.0, latency_ms=82.9,
    )
    db.add_embeddings_measurement(
        con, run_id, batch_size=4, seq_len=128,
        os_wired_gb=4.8, peak_gb=3.3, throughput_tps=22222.0, latency_ms=44.1,
    )

    runs = db.get_all_embeddings_runs(con)
    assert len(runs) == 1
    assert runs[0]["model_id"] == "mlx-community/test-modernbert"
    assert runs[0]["mlx_version"] == "0.31.2"
    assert runs[0]["created_at"]  # populated by _now()

    rows = db.get_embeddings_measurements(con, run_id)
    assert len(rows) == 2
    assert (rows[0]["batch_size"], rows[0]["seq_len"]) == (2, 512)
    assert (rows[1]["batch_size"], rows[1]["seq_len"]) == (4, 128)
    assert rows[0]["throughput_tps"] == 12345.0

    latest = db.get_latest_embeddings_run(con)
    assert latest["id"] == run_id

    # FK cascade: rows exist now, and deleting the parent run wipes them.
    assert len(db.get_embeddings_measurements(con, run_id)) == 2
    con.execute("DELETE FROM embeddings_runs WHERE id = ?", (run_id,))
    con.commit()
    assert db.get_embeddings_measurements(con, run_id) == []


class _FakeArray:
    """Stand-in for an mx.array; only needs a .dtype attribute for the dtype lookup."""
    def __init__(self, dtype="bf16"):
        self.dtype = dtype


class _FakeModel:
    """Callable stand-in: exposes model.model.embeddings.tok_embeddings.weight.dtype and
    returns an output object with .last_hidden_state from __call__."""
    def __init__(self):
        self.model = SimpleNamespace(
            embeddings=SimpleNamespace(
                tok_embeddings=SimpleNamespace(weight=_FakeArray(dtype="bf16"))
            )
        )

    def __call__(self, *args, **kwargs):
        return SimpleNamespace(last_hidden_state=_FakeArray())


def _fake_model():
    return _FakeModel()


def _install_fake_worker_env(monkeypatch, *, wired_now, threshold, wired_series):
    """Patch mlx_embeddings/mx/system on the worker's module objects.

    wired_series: floats the sampler/reads return in order, last value repeats.
    """
    from wmx_suite import probe_worker_embeddings as w

    limits = SimpleNamespace(
        wired_now_gb=wired_now,
        safe_threshold_gb=lambda margin=2.0: threshold,
    )
    monkeypatch.setattr(w.system, "read_limits", lambda: limits)
    series = list(wired_series)
    def fake_wired_gb():
        return series[0] if len(series) == 1 else series.pop(0)
    monkeypatch.setattr(w.system, "wired_gb", fake_wired_gb)

    monkeypatch.setattr(w.mx, "clear_cache", lambda: None)
    monkeypatch.setattr(w.mx, "reset_peak_memory", lambda: None)
    monkeypatch.setattr(w.mx, "get_peak_memory", lambda: int(3.0 * 1e9))
    monkeypatch.setattr(w.mx, "eval", lambda *a, **k: None)
    monkeypatch.setattr(w.mx, "zeros", lambda shape, dtype=None: _FakeArray())
    monkeypatch.setattr(w.mx, "ones", lambda shape, dtype=None: _FakeArray())
    return w


def test_worker_happy_path(monkeypatch, capsys):
    w = _install_fake_worker_env(
        monkeypatch, wired_now=3.0, threshold=15.18,
        wired_series=[3.0, 4.0, 5.5, 5.0],
    )
    load_calls = []
    def fake_load(model_id):
        load_calls.append(model_id)
        return _fake_model(), object()
    monkeypatch.setattr(w.mlx_embeddings, "load", fake_load)

    monkeypatch.setattr(sys, "argv", [
        "probe_worker_embeddings", "--model", "m", "--batch", "2", "--seq", "128",
        "--repeats", "2", "--margin", "2.0",
    ])
    w.main()

    out = capsys.readouterr().out
    line = next(l for l in out.splitlines() if l.startswith("{"))
    data = json.loads(line)
    assert data["status"] == "rung_done"
    assert data["batch"] == 2 and data["seq"] == 128
    assert load_calls == ["m"]
    assert data["throughput_tps"] > 0 and data["latency_ms"] >= 0
    assert "os_wired_gb" in data and "peak_gb" in data


def test_worker_preflight_refusal_never_loads(monkeypatch, capsys):
    w = _install_fake_worker_env(
        monkeypatch, wired_now=15.0, threshold=15.18,
        wired_series=[15.0],
    )
    load_calls = []
    monkeypatch.setattr(w.mlx_embeddings, "load",
                        lambda mid: load_calls.append(mid) or (_fake_model(), object()))

    monkeypatch.setattr(sys, "argv", [
        "probe_worker_embeddings", "--model", "m", "--batch", "1", "--seq", "128",
        "--repeats", "1", "--margin", "2.0",
    ])
    with pytest.raises(SystemExit) as ei:
        w.main()
    assert ei.value.code == 0
    out = capsys.readouterr().out
    data = json.loads(next(l for l in out.splitlines() if l.startswith("{")))
    assert data["status"] == "error"
    assert load_calls == []  # model NEVER loaded — RULE #1 guard



def test_cold_start_gate_uses_nonzero_model_base(monkeypatch):
    from wmx_suite import embeddings_probe as ep
    spawned = []
    monkeypatch.setattr(ep, "sample_settled_baseline",
                        lambda: 15.18 - ep.MODEL_BASE_SEED_GB + 0.01)
    monkeypatch.setattr(ep, "read_limits",
                        lambda: SimpleNamespace(safe_threshold_gb=lambda m=2.0: 15.18,
                                                wall_gb=17.18, wired_now_gb=3.0))
    monkeypatch.setattr(ep, "_run_cell",
                        lambda *a, **k: spawned.append(a) or {"status": "rung_done"})
    events = []
    summary = ep.sweep(con=None, run_id=1, model="m",
                       batches=[1], seqs=[128], repeats=1, margin_gb=2.0,
                       on_event=events.append, persist=False)
    assert spawned == []
    assert any(e["event"] == "preflight_abort" for e in events)


def test_predictive_skip_does_not_spawn_unsafe_cell(monkeypatch):
    from wmx_suite import embeddings_probe as ep
    spawned = []

    def fake_run_cell(py, model, batch, seq, repeats, margin):
        spawned.append((batch, seq))
        x2 = batch * seq * seq
        delta = 5.0e-7 * x2
        return {"status": "rung_done", "batch": batch, "seq": seq,
                "os_wired_gb": 3.0 + delta, "peak_gb": 1.0,
                "throughput_tps": 1.0, "latency_ms": 1.0}

    monkeypatch.setattr(ep, "_run_cell", fake_run_cell)
    monkeypatch.setattr(ep, "sample_settled_baseline", lambda: 3.0)
    monkeypatch.setattr(ep, "read_limits",
                        lambda: SimpleNamespace(safe_threshold_gb=lambda m=2.0: 15.18,
                                                wall_gb=17.18, wired_now_gb=3.0))
    events = []
    ep.sweep(con=None, run_id=1, model="m",
             batches=[1], seqs=[512, 1024, 2048, 4096, 8192], repeats=1, margin_gb=2.0,
             on_event=events.append, persist=False)
    skipped = [e for e in events if e["event"] == "row_skipped"]
    assert skipped, "expected at least one predictive skip"
    skipped_seqs = {e["seq"] for e in skipped}
    assert not (skipped_seqs & {s for (_, s) in spawned})


def test_monotonic_pruning_skips_larger_batch_same_seq(monkeypatch):
    from wmx_suite import embeddings_probe as ep
    spawned = []

    def fake_run_cell(py, model, batch, seq, repeats, margin):
        spawned.append((batch, seq))
        x2 = batch * seq * seq
        return {"status": "rung_done", "batch": batch, "seq": seq,
                "os_wired_gb": 3.0 + 5.0e-7 * x2, "peak_gb": 1.0,
                "throughput_tps": 1.0, "latency_ms": 1.0}

    monkeypatch.setattr(ep, "_run_cell", fake_run_cell)
    monkeypatch.setattr(ep, "sample_settled_baseline", lambda: 3.0)
    monkeypatch.setattr(ep, "read_limits",
                        lambda: SimpleNamespace(safe_threshold_gb=lambda m=2.0: 15.18,
                                                wall_gb=17.18, wired_now_gb=3.0))
    events = []
    ep.sweep(con=None, run_id=1, model="m",
             batches=[1, 32], seqs=[2048, 8192], repeats=1, margin_gb=2.0,
             on_event=events.append, persist=False)
    # The worst cell (largest batch AND largest seq) must never be spawned given the
    # strong quadratic signal; and (1, 8192) being unsafe implies (32, 8192) is too.
    assert (32, 8192) not in spawned
    assert (1, 8192) not in spawned


def test_cmd_benchmark_embeddings_persists_and_renders(monkeypatch, tmp_path, capsys):
    from wmx_suite import cli, db, embeddings_probe
    from wmx_suite.ui import Console

    monkeypatch.setattr(db, "DB_PATH", tmp_path / "suite.db")

    def fake_sweep(con, run_id, model, batches, seqs, repeats, margin_gb=None,
                   *, mlx_version=None, ignore_profile=False, on_event=None, persist=True):
        for (bsz, seq) in [(1, 128), (2, 256)]:
            db.add_embeddings_measurement(con, run_id, batch_size=bsz, seq_len=seq,
                                          os_wired_gb=4.0, peak_gb=2.0,
                                          throughput_tps=100.0, latency_ms=5.0)
            on_event({"event": "cell_done", "batch": bsz, "seq": seq,
                      "os_wired_gb": 4.0, "peak_gb": 2.0,
                      "throughput_tps": 100.0, "latency_ms": 5.0})
        on_event({"event": "row_skipped", "batch": 2, "seq": 8192, "predicted_gb": 99.0})
        return {"model": model, "run_id": run_id, "n_cells_measured": 2, "n_cells_skipped": 1}

    monkeypatch.setattr(embeddings_probe, "sweep", fake_sweep)

    args = SimpleNamespace(model="mlx-community/test", batches="1,2", seqs="128,256,8192",
                           repeats=1, margin=None, console=Console(color=False, verbose=False))
    cli.cmd_benchmark_embeddings(args)

    out = capsys.readouterr().out
    # The view renders skipped cells with '✗' marker and 'skipped' in the summary line
    assert "skipped" in out.lower()

    con = db.connect()
    latest = db.get_latest_embeddings_run(con)
    assert latest is not None
    rows = db.get_embeddings_measurements(con, latest["id"])
    assert len(rows) == 2


def test_cmd_benchmark_embeddings_preflight_abort_exits(monkeypatch, tmp_path, capsys):
    from wmx_suite import cli, db, embeddings_probe
    from wmx_suite.ui import Console

    monkeypatch.setattr(db, "DB_PATH", tmp_path / "suite.db")

    def fake_sweep(con, run_id, model, batches, seqs, repeats, margin_gb=None,
                   *, mlx_version=None, ignore_profile=False, on_event=None, persist=True):
        on_event({"event": "preflight_abort", "note": "host too hot"})
        return {"model": model, "run_id": run_id, "n_cells_measured": 0,
                "n_cells_skipped": 0, "aborted": True}

    monkeypatch.setattr(embeddings_probe, "sweep", fake_sweep)

    args = SimpleNamespace(model="mlx-community/test", batches="1", seqs="128",
                           repeats=1, margin=None, console=Console(color=False, verbose=False))
    with pytest.raises(SystemExit) as ei:
        cli.cmd_benchmark_embeddings(args)
    assert ei.value.code == 1
    # The view renders a 'Won't run' guidance block for preflight aborts.
    assert "Won't run" in capsys.readouterr().out


def test_cmd_benchmark_embeddings_worker_error_exits(monkeypatch, tmp_path, capsys):
    from wmx_suite import cli, db, embeddings_probe
    from wmx_suite.ui import Console

    monkeypatch.setattr(db, "DB_PATH", tmp_path / "suite.db")

    def fake_sweep(con, run_id, model, batches, seqs, repeats, margin_gb=None,
                   *, mlx_version=None, ignore_profile=False, on_event=None, persist=True):
        on_event({"event": "error", "batch": 1, "seq": 128, "note": "worker blew up"})
        return {"model": model, "run_id": run_id, "n_cells_measured": 0,
                "n_cells_skipped": 0, "error": "worker blew up"}

    monkeypatch.setattr(embeddings_probe, "sweep", fake_sweep)

    args = SimpleNamespace(model="mlx-community/test", batches="1", seqs="128",
                           repeats=1, margin=None, console=Console(color=False, verbose=False))
    with pytest.raises(SystemExit) as ei:
        cli.cmd_benchmark_embeddings(args)
    assert ei.value.code == 1
    assert "ERROR at batch 1 seq 128" in capsys.readouterr().out


def test_embedding_profile_roundtrip_and_key_mismatch(monkeypatch, tmp_path):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "suite.db")
    con = db.connect()
    key = ("Apple M4 Pro", 25769803776, 15, "0.31.2", "mlx-community/test-embed")
    assert db.get_embedding_profile(con, key) is None

    db.upsert_embedding_profile(con, key, coef_intercept_gb=1.07,
                                coef_linear=2.1e-5, coef_quad=6.6e-9, n_points=20)
    row = db.get_embedding_profile(con, key)
    assert row["coef_intercept_gb"] == 1.07
    assert row["coef_linear"] == 2.1e-5
    assert row["coef_quad"] == 6.6e-9
    assert row["n_points"] == 20
    assert row["created_at"]

    db.upsert_embedding_profile(con, key, coef_intercept_gb=2.0,
                                coef_linear=3.0e-5, coef_quad=7.0e-9, n_points=30)
    assert db.get_embedding_profile(con, key)["coef_intercept_gb"] == 2.0
    assert con.execute("SELECT COUNT(*) FROM embedding_profiles").fetchone()[0] == 1

    stale = ("Apple M4 Pro", 25769803776, 15, "0.32.0", "mlx-community/test-embed")
    assert db.get_embedding_profile(con, stale) is None


def test_fit_cab_recovers_known_coeffs_and_handles_singular():
    from wmx_suite import embeddings_probe as ep
    # delta = 1.0 + 1e-6*x1 + 2e-8*x2 exactly, over varied (batch, seq)
    pts = []
    for bsz, s in [(1, 128), (1, 512), (2, 256), (4, 512), (8, 1024)]:
        x1, x2 = bsz * s, bsz * s * s
        pts.append((x1, x2, 1.0 + 1e-6 * x1 + 2e-8 * x2))
    c, a, b = ep._fit_cab(pts)
    assert c == pytest.approx(1.0, abs=1e-6)
    assert a == pytest.approx(1e-6, rel=1e-3)
    assert b == pytest.approx(2e-8, rel=1e-3)

    # < 3 points -> None
    assert ep._fit_cab([(128.0, 16384.0, 1.0), (256.0, 65536.0, 1.1)]) is None
    # all-identical (degenerate) feature rows -> singular -> None
    same = [(128.0, 16384.0, 1.0)] * 5
    assert ep._fit_cab(same) is None
    # degenerate at LARGE scale (perfectly collinear x1/x2 columns) still -> None; this is
    # the magnitude regime where a fixed tiny epsilon is meaningless and the relative
    # threshold matters.
    bigdegen = [(1e9, 1e18, 1.0), (2e9, 2e18, 2.0), (3e9, 3e18, 3.0)]
    assert ep._fit_cab(bigdegen) is None


def test_coeffs_falls_back_to_cold_overestimate():
    from wmx_suite import embeddings_probe as ep
    # Below MIN_FIT_POINTS -> cold over-estimate with zero intercept.
    assert ep._coeffs([(128.0, 16384.0, 0.1)], None) == (0.0, ep.A_COLD, ep.B_COLD)
    # Degenerate (singular) even with enough points -> cold over-estimate.
    degen = [(128.0, 16384.0, 1.0)] * ep.MIN_FIT_POINTS
    assert ep._coeffs(degen, None) == (0.0, ep.A_COLD, ep.B_COLD)
    # A stored profile is used when there's no usable in-run fit yet.
    assert ep._coeffs([(1.0, 1.0, 0.1)], (1.5, 2e-6, 3e-9)) == (1.5, 2e-6, 3e-9)


def test_profiles_embedding_coeffs_roundtrip(monkeypatch, tmp_path):
    from wmx_suite import profiles
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "suite.db")
    monkeypatch.setattr(profiles, "machine_key",
                        lambda: ("Apple M4 Pro", 25769803776, 15))
    con = db.connect()

    assert profiles.embedding_coeffs(con, "org/m", "0.31.2") is None

    profiles.upsert_embedding_coeffs(con, "org/m", "0.31.2",
                                     coef_intercept_gb=1.1, coef_linear=2e-5,
                                     coef_quad=6e-9, n_points=12)
    assert profiles.embedding_coeffs(con, "org/m", "0.31.2") == (1.1, 2e-5, 6e-9)
    # different model or mlx version -> miss
    assert profiles.embedding_coeffs(con, "org/other", "0.31.2") is None
    assert profiles.embedding_coeffs(con, "org/m", "9.9.9") is None


# Real Run-2 os_wired values (GB), keyed (batch, seq). Surface is flat (~2.7 baseline).
_RUN2_OSWIRED = {
    (1, 128): 2.72, (1, 256): 3.50, (1, 512): 3.35, (1, 1024): 3.63, (1, 2048): 4.07, (1, 4096): 4.26,
    (2, 128): 3.29, (2, 256): 3.48, (2, 512): 3.63, (2, 1024): 4.23, (2, 2048): 4.10, (2, 4096): 4.47,
    (4, 128): 3.50, (4, 256): 3.62, (4, 512): 4.11, (4, 1024): 4.18, (4, 2048): 4.33, (4, 4096): 4.49,
    (8, 128): 3.61, (8, 256): 4.10, (8, 512): 4.18, (8, 1024): 4.29, (8, 2048): 4.20,
    (16, 128): 4.11, (16, 256): 4.16, (16, 512): 4.26, (16, 1024): 4.04, (16, 2048): 4.79,
    (32, 128): 4.17, (32, 256): 4.25, (32, 512): 3.99, (32, 1024): 4.50,
}


def test_intercept_gate_measures_previously_skipped_safe_cells(monkeypatch):
    from wmx_suite import embeddings_probe as ep

    def fake_run_cell(py, model, batch, seq, repeats, margin):
        ow = _RUN2_OSWIRED.get((batch, seq), 4.6)
        return {"status": "rung_done", "batch": batch, "seq": seq,
                "os_wired_gb": ow, "peak_gb": 1.0, "throughput_tps": 1.0, "latency_ms": 1.0}

    monkeypatch.setattr(ep, "_run_cell", fake_run_cell)
    monkeypatch.setattr(ep, "sample_settled_baseline", lambda: 2.71)
    monkeypatch.setattr(ep, "read_limits",
                        lambda: SimpleNamespace(safe_threshold_gb=lambda m=2.0: 15.18,
                                                wall_gb=17.18, wired_now_gb=2.71))
    events = []
    ep.sweep(con=None, run_id=1, model="m", repeats=1, margin_gb=2.0,
             on_event=events.append, persist=False)
    measured = {(e["batch"], e["seq"]) for e in events if e["event"] == "cell_done"}
    skipped = {(e["batch"], e["seq"]) for e in events if e["event"] == "row_skipped"}
    # Cells the OLD through-origin gate wrongly skipped are now measured:
    assert (8, 4096) in measured       # old gate predicted 18.27 GB
    assert (32, 2048) in measured      # old gate predicted 15.33 GB
    assert (1, 8192) in measured       # old gate predicted 16.39 GB
    # The danger boundary still fires: batch=32 breaks at seq=4096 (predicted over wall),
    # so (32, 4096) is skipped and the larger (32, 8192) is never even reached/measured.
    assert (32, 4096) in skipped
    assert (32, 8192) not in measured


def test_model_base_clamped_to_seed_on_negative_intercept(monkeypatch):
    from wmx_suite import embeddings_probe as ep
    spawned = []

    def fake_run_cell(py, model, batch, seq, repeats, margin):
        spawned.append((batch, seq))
        return {"status": "rung_done", "batch": batch, "seq": seq,
                "os_wired_gb": 2.8, "peak_gb": 1.0, "throughput_tps": 1.0, "latency_ms": 1.0}

    monkeypatch.setattr(ep, "_run_cell", fake_run_cell)
    monkeypatch.setattr(ep, "sample_settled_baseline", lambda: 2.71)
    monkeypatch.setattr(ep, "read_limits",
                        lambda: SimpleNamespace(safe_threshold_gb=lambda m=2.0: 15.18,
                                                wall_gb=17.18, wired_now_gb=2.71))
    # stored coeffs with a NEGATIVE intercept; floors clamp slopes, seed clamps model_base
    monkeypatch.setattr(ep, "_coeffs", lambda points, stored: (-5.0, 1e-9, 1e-12))
    ep.sweep(con=None, run_id=1, model="m", batches=[1], seqs=[128], repeats=1,
             margin_gb=2.0, on_event=lambda e: None, persist=False)
    assert spawned == [(1, 128)]  # didn't crash on negative c, didn't falsely abort


def test_sweep_autoupserts_profile_after_run(monkeypatch, tmp_path):
    from wmx_suite import embeddings_probe as ep, profiles
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "suite.db")
    monkeypatch.setattr(profiles, "machine_key",
                        lambda: ("Apple M4 Pro", 25769803776, 15))
    con = db.connect()

    def fake_run_cell(py, model, batch, seq, repeats, margin):
        return {"status": "rung_done", "batch": batch, "seq": seq,
                "os_wired_gb": 2.71 + 0.001 * batch * seq, "peak_gb": 1.0,
                "throughput_tps": 1.0, "latency_ms": 1.0}

    monkeypatch.setattr(ep, "_run_cell", fake_run_cell)
    monkeypatch.setattr(ep, "sample_settled_baseline", lambda: 2.71)
    monkeypatch.setattr(ep, "read_limits",
                        lambda: SimpleNamespace(safe_threshold_gb=lambda m=2.0: 15.18,
                                                wall_gb=17.18, wired_now_gb=2.71))
    run_id = db.start_embeddings_run(con, "org/m", "0.31.2")
    ep.sweep(con, run_id, "org/m", batches=[1], seqs=[128, 256, 512, 1024, 2048],
             repeats=1, margin_gb=2.0, mlx_version="0.31.2", on_event=lambda e: None)
    assert profiles.embedding_coeffs(con, "org/m", "0.31.2") is not None


def test_sweep_loads_stored_profile(monkeypatch, tmp_path):
    from wmx_suite import embeddings_probe as ep, profiles
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "suite.db")
    monkeypatch.setattr(profiles, "machine_key",
                        lambda: ("Apple M4 Pro", 25769803776, 15))
    con = db.connect()
    profiles.upsert_embedding_coeffs(con, "org/m", "0.31.2",
                                     coef_intercept_gb=1.1, coef_linear=2e-5,
                                     coef_quad=6e-9, n_points=20)

    seen = {}
    real_coeffs = ep._coeffs
    def spy_coeffs(points, stored):
        seen["stored"] = stored
        return real_coeffs(points, stored)
    monkeypatch.setattr(ep, "_coeffs", spy_coeffs)

    def fake_run_cell(py, model, batch, seq, repeats, margin):
        return {"status": "rung_done", "batch": batch, "seq": seq, "os_wired_gb": 2.8,
                "peak_gb": 1.0, "throughput_tps": 1.0, "latency_ms": 1.0}
    monkeypatch.setattr(ep, "_run_cell", fake_run_cell)
    monkeypatch.setattr(ep, "sample_settled_baseline", lambda: 2.71)
    monkeypatch.setattr(ep, "read_limits",
                        lambda: SimpleNamespace(safe_threshold_gb=lambda m=2.0: 15.18,
                                                wall_gb=17.18, wired_now_gb=2.71))
    run_id = db.start_embeddings_run(con, "org/m", "0.31.2")
    ep.sweep(con, run_id, "org/m", batches=[1], seqs=[128], repeats=1, margin_gb=2.0,
             mlx_version="0.31.2", on_event=lambda e: None)
    assert seen["stored"] == (1.1, 2e-5, 6e-9)


def test_sweep_ignore_profile_skips_load_but_still_upserts(monkeypatch, tmp_path):
    from wmx_suite import embeddings_probe as ep, profiles
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "suite.db")
    monkeypatch.setattr(profiles, "machine_key",
                        lambda: ("Apple M4 Pro", 25769803776, 15))
    con = db.connect()
    profiles.upsert_embedding_coeffs(con, "org/m", "0.31.2",
                                     coef_intercept_gb=9.9, coef_linear=1.0,
                                     coef_quad=1.0, n_points=1)
    seen = {}
    real_coeffs = ep._coeffs
    monkeypatch.setattr(ep, "_coeffs",
                        lambda points, stored: seen.setdefault("stored", stored) or real_coeffs(points, stored))

    def fake_run_cell(py, model, batch, seq, repeats, margin):
        return {"status": "rung_done", "batch": batch, "seq": seq,
                "os_wired_gb": 2.71 + 0.001 * batch * seq, "peak_gb": 1.0,
                "throughput_tps": 1.0, "latency_ms": 1.0}
    monkeypatch.setattr(ep, "_run_cell", fake_run_cell)
    monkeypatch.setattr(ep, "sample_settled_baseline", lambda: 2.71)
    monkeypatch.setattr(ep, "read_limits",
                        lambda: SimpleNamespace(safe_threshold_gb=lambda m=2.0: 15.18,
                                                wall_gb=17.18, wired_now_gb=2.71))
    run_id = db.start_embeddings_run(con, "org/m", "0.31.2")
    ep.sweep(con, run_id, "org/m", batches=[1], seqs=[128, 256, 512, 1024, 2048],
             repeats=1, margin_gb=2.0, mlx_version="0.31.2", ignore_profile=True,
             on_event=lambda e: None)
    assert seen["stored"] is None
    assert db.get_embedding_profile(
        con, profiles.embedding_machine_key("org/m", "0.31.2"))["n_points"] >= 4


def test_cmd_benchmark_embeddings_passes_mlx_version_and_ignore_flag(monkeypatch, tmp_path, capsys):
    from wmx_suite import cli, db, embeddings_probe
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "suite.db")
    captured = {}

    def fake_sweep(con, run_id, model, batches, seqs, repeats, margin_gb=None, *,
                   mlx_version=None, ignore_profile=False, on_event=None, persist=True):
        captured["mlx_version"] = mlx_version
        captured["ignore_profile"] = ignore_profile
        return {"model": model, "run_id": run_id, "n_cells_measured": 0, "n_cells_skipped": 0}

    monkeypatch.setattr(embeddings_probe, "sweep", fake_sweep)
    args = SimpleNamespace(model="org/m", batches="1", seqs="128", repeats=1,
                           margin=None, ignore_profile=True)
    cli.cmd_benchmark_embeddings(args)
    assert captured["ignore_profile"] is True
    assert captured["mlx_version"]  # a real mlx version string was passed through
