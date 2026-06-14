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
