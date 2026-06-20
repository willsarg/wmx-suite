"""The runtime watchdog (L5) — the last line of defense.

If the pre-flight prediction was wrong and live wired memory crosses a hard limit mid-probe,
the worker aborts immediately (freeing memory by dying) rather than riding into the wall.
The trip decision is a pure helper so it's testable without threads or real memory.
"""
from __future__ import annotations

from wmx_suite import probe_worker


def test_should_abort_trips_at_or_over_limit():
    assert probe_worker._should_abort(36.0, 36.0) is True
    assert probe_worker._should_abort(37.5, 36.0) is True


def test_should_abort_false_below_limit():
    assert probe_worker._should_abort(35.9, 36.0) is False


def test_should_abort_disabled_when_no_limit():
    # no limit passed → watchdog off, preserving the original unguarded behavior
    assert probe_worker._should_abort(999.0, None) is False


# --- the limit is plumbed from the parent through _run_worker to the subprocess --- #
from wmx_suite import probe  # noqa: E402


def test_run_worker_forwards_abort_limit_flag(monkeypatch):
    captured = {}

    class P:
        stdout = '{"status":"ok","os_wired_gb":9.0}'
        stderr = ""

    def fake_run(cmd, capture_output, text):
        captured["cmd"] = cmd
        return P()

    monkeypatch.setattr(probe.subprocess, "run", fake_run)
    probe._run_worker("python", "m/x", 4000, 4, abort_wired_gb=36.0)
    assert "--abort-wired-gb" in captured["cmd"]
    assert "36.0" in captured["cmd"]


def test_run_worker_omits_abort_flag_when_none(monkeypatch):
    captured = {}

    class P:
        stdout = '{"status":"ok"}'
        stderr = ""

    monkeypatch.setattr(probe.subprocess, "run",
                        lambda cmd, capture_output, text: captured.update(cmd=cmd) or P())
    probe._run_worker("python", "m/x", 4000, None)
    assert "--abort-wired-gb" not in captured["cmd"]
