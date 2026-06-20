"""Self-vetoing single-measurement primitive (L4 of ARA's defense-in-depth).

The gate runs *before* any model load: it reads the live wall, estimates the model's
base + conservative slope, and refuses (no load) if either the base alone or the
prediction at the requested context would reach the safe budget. Only when safe does it
spawn the isolated probe worker. Emits one canonical JSON object for ARA to consume.

These tests never touch real mlx — the refusal paths never reach a load, and the safe
path patches the isolated worker.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

from wmx_suite import measure_one


def _info(weights_gb=0.1, slope=0.01, is_causal=True, can_quant=True, max_ctx=None):
    return SimpleNamespace(
        weights_gb=weights_gb,
        is_causal=is_causal,
        can_quantize_kv=can_quant,
        max_context=max_ctx,
        estimated_slope_gb_per_k=lambda: slope,
    )


def _limits(wall_gb=40.0, wired_now_gb=8.0):
    return SimpleNamespace(
        wall_gb=wall_gb,
        wired_now_gb=wired_now_gb,
        safe_threshold_gb=lambda margin: wall_gb - margin,
    )


# --------------------------------------------------------------------------- #
# safety_gate (pure)
# --------------------------------------------------------------------------- #
def test_gate_allows_a_safe_probe():
    reason = measure_one.safety_gate(
        _info(), _limits(), ctx=4000, margin_gb=4.0, overhead_gb=1.0, live_base=8.0)
    assert reason is None


def test_gate_refuses_when_base_alone_exceeds_budget():
    # weights 30 GB → base ~ 8 + 30*1.05 + 1 = 40.5 >= threshold 36 → won't even load
    reason = measure_one.safety_gate(
        _info(weights_gb=30.0), _limits(), ctx=2000, margin_gb=4.0,
        overhead_gb=1.0, live_base=8.0)
    assert reason is not None and "load" in reason.lower()


def test_gate_refuses_when_prediction_at_ctx_breaches():
    # small base but steep slope at high ctx → predicted reaches the wall
    reason = measure_one.safety_gate(
        _info(slope=2.0), _limits(), ctx=20000, margin_gb=4.0,
        overhead_gb=1.0, live_base=8.0)
    assert reason is not None and "20000" in reason


# --------------------------------------------------------------------------- #
# run() — describe + gate + (safe) spawn, returns canonical dict
# --------------------------------------------------------------------------- #
def test_run_refuses_unknown_or_noncausal_model(monkeypatch):
    monkeypatch.setattr(measure_one.models, "describe", lambda hf: None)
    out = measure_one.run("mystery/model", 4000, margin_gb=4.0, overhead_gb=1.0)
    assert out == {"context": 4000, "refused": True, "reason": out["reason"]}
    assert "causal" in out["reason"] or "not found" in out["reason"]


def test_run_refuses_when_gate_vetoes_without_loading(monkeypatch):
    monkeypatch.setattr(measure_one.models, "describe", lambda hf: _info(weights_gb=30.0))
    monkeypatch.setattr(measure_one.system, "read_limits", lambda: _limits())
    monkeypatch.setattr(measure_one.system, "sample_settled_baseline", lambda: 8.0)
    # if this spawns a worker, the test fails loudly
    monkeypatch.setattr(measure_one, "_spawn_worker",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("loaded!")))
    out = measure_one.run("big/model", 2000, margin_gb=4.0, overhead_gb=1.0)
    assert out["refused"] is True and out["context"] == 2000


def test_run_returns_median_delta_on_safe_measurement(monkeypatch):
    monkeypatch.setattr(measure_one.models, "describe", lambda hf: _info())
    monkeypatch.setattr(measure_one.system, "read_limits", lambda: _limits())
    monkeypatch.setattr(measure_one.system, "sample_settled_baseline", lambda: 8.0)
    # three fresh runs with jittery absolute wired but a stable delta over baseline
    runs = iter([
        {"status": "ok", "os_wired_gb": 9.3, "baseline_wired_gb": 4.0},   # delta 5.3
        {"status": "ok", "os_wired_gb": 10.1, "baseline_wired_gb": 4.9},  # delta 5.2
        {"status": "ok", "os_wired_gb": 8.9, "baseline_wired_gb": 3.6},   # delta 5.3
    ])
    monkeypatch.setattr(measure_one, "_spawn_worker",
                        lambda hf, ctx, kv_bits, abort_wired_gb: next(runs))
    out = measure_one.run("small/model", 4000, margin_gb=4.0, overhead_gb=1.0, repeats=3)
    assert out == {"context": 4000, "mem_gb": 5.3}     # median delta, ambient drift removed


def test_run_refuses_when_worker_load_errors(monkeypatch):
    monkeypatch.setattr(measure_one.models, "describe", lambda hf: _info())
    monkeypatch.setattr(measure_one.system, "read_limits", lambda: _limits())
    monkeypatch.setattr(measure_one.system, "sample_settled_baseline", lambda: 8.0)
    monkeypatch.setattr(measure_one, "_spawn_worker",
                        lambda hf, ctx, kv_bits, abort_wired_gb: {"status": "error", "note": "OOM loading"})
    out = measure_one.run("small/model", 4000, margin_gb=4.0, overhead_gb=1.0)
    assert out["refused"] is True and "OOM loading" in out["reason"]


def test_run_passes_safe_budget_as_watchdog_limit(monkeypatch):
    # L5 plumbing: the worker is told the hard wired limit (= safe budget) to abort at
    captured = {}
    monkeypatch.setattr(measure_one.models, "describe", lambda hf: _info())
    monkeypatch.setattr(measure_one.system, "read_limits", lambda: _limits())  # wall 40, margin 4 → 36
    monkeypatch.setattr(measure_one.system, "sample_settled_baseline", lambda: 8.0)

    def fake_spawn(hf, ctx, kv_bits, abort_wired_gb):
        captured["abort"] = abort_wired_gb
        return {"status": "ok", "os_wired_gb": 9.0, "baseline_wired_gb": 4.0}

    monkeypatch.setattr(measure_one, "_spawn_worker", fake_spawn)
    measure_one.run("small/model", 4000, margin_gb=4.0, overhead_gb=1.0, repeats=1)
    assert captured["abort"] == 36.0


def test_preflight_returns_estimate_without_loading(monkeypatch):
    monkeypatch.setattr(measure_one.models, "describe", lambda hf: _info(weights_gb=0.1, slope=0.02, max_ctx=8192))
    monkeypatch.setattr(measure_one.system, "read_limits", lambda: _limits())  # wall40, margin4 → 36
    monkeypatch.setattr(measure_one.system, "sample_settled_baseline", lambda: 8.0)
    est = measure_one.preflight("small/model", margin_gb=4.0, overhead_gb=1.0)
    # base = live 8 + (0.1*1.05 + 1) = 9.105 ; budget = 36 ; slope/max passthrough
    assert est["base_gb"] == round(8.0 + 0.1 * 1.05 + 1.0, 4)
    assert est["ref_baseline_gb"] == 8.0     # live OS baseline, added at solve time
    assert est["slope_gb_per_k"] == 0.02
    assert est["budget_gb"] == 36.0
    assert est["max_context"] == 8192


def test_preflight_errors_for_unknown_model(monkeypatch):
    monkeypatch.setattr(measure_one.models, "describe", lambda hf: None)
    est = measure_one.preflight("mystery", margin_gb=4.0, overhead_gb=1.0)
    assert "error" in est


def test_main_preflight_flag_prints_estimate(monkeypatch, capsys):
    monkeypatch.setattr(measure_one, "preflight", lambda *a, **k: {"base_gb": 9.1})
    measure_one.main(["small/model", "0", "--margin", "4", "--overhead", "1", "--preflight"])
    assert json.loads(capsys.readouterr().out.strip()) == {"base_gb": 9.1}


def test_main_prints_one_canonical_json_line(monkeypatch, capsys):
    monkeypatch.setattr(measure_one, "run",
                        lambda *a, **k: {"context": 4000, "mem_gb": 9.3})
    measure_one.main(["small/model", "4000", "--margin", "4", "--overhead", "1"])
    line = capsys.readouterr().out.strip()
    assert json.loads(line) == {"context": 4000, "mem_gb": 9.3}
