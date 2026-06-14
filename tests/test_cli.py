import json
import sys
import threading
from types import SimpleNamespace

import pytest

from wmx_suite import cli, config
from wmx_suite.system import SystemLimits
from wmx_suite.ui import Console


def _ns(**kw):
    """Args namespace with a no-color Console (so command output is plain text)."""
    kw.setdefault("console", Console(color=False, verbose=False))
    return SimpleNamespace(**kw)


class _Tokenizer:
    has_chat_template = False

    def encode(self, prompt, **_kwargs):
        return prompt.split()


@pytest.fixture(autouse=True)
def _stub_tokenizer_loader(monkeypatch):
    monkeypatch.setattr(
        cli,
        "load_tokenizer",
        lambda _model, tokenizer_config_extra=None: _Tokenizer(),
    )


def _plan():
    return {
        "hf_id": "mlx-community/test",
        "kv_bits": 4,
        "kv_group_size": 64,
        "quantized_kv_start": 5000,
        "source": "measured",
        "cache_type": "standard",
        "model_max": 32768,
        "live_base_gb": 3.0,
        "model_base_gb": 8.0,
        "base_abs_gb": 11.0,
        "slope_gb_per_k": 0.1,
        "threshold_gb": 15.0,
        "wall_gb": 17.0,
        "max_kv_size": 4096,
        "fit_stale": False,
        "max_kv_size_enforced": True,
        "refuse": False,
    }


def test_run_reports_launch_argument_refusal(monkeypatch):
    monkeypatch.setattr(cli.launcher, "plan", lambda _model, margin_gb: _plan())
    monkeypatch.setattr(
        cli.launcher,
        "build_argv",
        lambda _rest, _plan, force: (_ for _ in ()).throw(
            cli.launcher.LaunchArgumentError("unsafe argument")
        ),
    )

    with pytest.raises(SystemExit, match=r"\[run\] REFUSED: unsafe argument"):
        cli._run(
            ["--model", "mlx-community/test"],
            margin=2.0,
            force=False,
            dry_run=True,
        )


def test_run_refuses_prompt_above_cap(monkeypatch):
    monkeypatch.setattr(cli.launcher, "plan", lambda _model, margin_gb: _plan())
    monkeypatch.setattr(cli.launcher, "build_argv", lambda rest, plan, *, force: rest)
    monkeypatch.setattr(
        cli.launcher,
        "check_prompt",
        lambda rest, plan, tokenizer: cli.launcher.PromptCheck(
            tokens=4097, cap=4096, warn=True
        ),
    )
    monkeypatch.setattr(
        cli,
        "load_tokenizer",
        lambda _model, tokenizer_config_extra=None: object(),
    )

    with pytest.raises(SystemExit, match="above the 4,096-token cap"):
        cli._run(
            ["--model", "mlx-community/test", "--prompt", "large"],
            margin=2.0,
            force=False,
            dry_run=True,
        )


def test_run_warns_when_prompt_is_near_cap(monkeypatch, capsys):
    monkeypatch.setattr(cli.launcher, "plan", lambda _model, margin_gb: _plan())
    monkeypatch.setattr(cli.launcher, "build_argv", lambda rest, plan, *, force: rest)
    monkeypatch.setattr(
        cli.launcher,
        "check_prompt",
        lambda rest, plan, tokenizer: cli.launcher.PromptCheck(
            tokens=3500, cap=4096, warn=True
        ),
    )
    monkeypatch.setattr(
        cli,
        "load_tokenizer",
        lambda _model, tokenizer_config_extra=None: object(),
    )

    cli._run(
        ["--model", "mlx-community/test", "--prompt", "large"],
        margin=2.0,
        force=False,
        dry_run=True,
    )

    assert "prompt exceeds 80% of the context cap" in capsys.readouterr().err


def test_run_force_bypasses_unverifiable_prompt_with_warning(monkeypatch, capsys):
    monkeypatch.setattr(cli.launcher, "plan", lambda _model, margin_gb: _plan())
    monkeypatch.setattr(cli.launcher, "build_argv", lambda rest, plan, *, force: rest)

    cli._run(
        ["--model", "mlx-community/test", "--prompt", "-"],
        margin=2.0,
        force=True,
        dry_run=True,
    )

    assert "prompt preflight bypassed" in capsys.readouterr().err


def test_run_reports_effective_user_cap(monkeypatch, capsys):
    monkeypatch.setattr(cli.launcher, "plan", lambda _model, margin_gb: _plan())

    cli._run(
        [
            "--model", "mlx-community/test",
            "--prompt", "small",
            "--max-kv-size", "2048",
        ],
        margin=2.0,
        force=False,
        dry_run=True,
    )

    assert "max-kv-size 2,048 tokens" in capsys.readouterr().err


def test_run_passes_force_to_argument_validation(monkeypatch):
    seen = {}
    monkeypatch.setattr(cli.launcher, "plan", lambda _model, margin_gb: _plan())

    def build_argv(rest, plan, *, force):
        seen["force"] = force
        return rest

    monkeypatch.setattr(cli.launcher, "build_argv", build_argv)
    cli._run(
        ["--model", "mlx-community/test"],
        margin=2.0,
        force=True,
        dry_run=True,
    )

    assert seen["force"] is True


def test_run_accepts_equals_form_model_argument(monkeypatch):
    seen = {}

    def plan(model_id, *, margin_gb):
        seen["model_id"] = model_id
        return _plan()

    monkeypatch.setattr(cli.launcher, "plan", plan)
    monkeypatch.setattr(
        cli.launcher,
        "build_argv",
        lambda rest, plan, *, force: rest,
    )
    cli._run(
        ["--model=mlx-community/test"],
        margin=2.0,
        force=False,
        dry_run=True,
    )

    assert seen["model_id"] == "mlx-community/test"


def test_run_uses_margin_from_environment(monkeypatch):
    seen = {}
    monkeypatch.setenv(config.MARGIN_ENV, "3.25")

    def plan(model_id, *, margin_gb):
        seen["margin"] = margin_gb
        return _plan()

    monkeypatch.setattr(cli.launcher, "plan", plan)
    monkeypatch.setattr(
        cli.launcher,
        "build_argv",
        lambda rest, plan, *, force: rest,
    )
    cli._run(
        ["--model", "mlx-community/test"],
        margin=None,
        force=False,
        dry_run=True,
    )

    assert seen["margin"] == 3.25


def test_run_explicit_margin_overrides_environment(monkeypatch):
    seen = {}
    monkeypatch.setenv(config.MARGIN_ENV, "3.25")

    def plan(model_id, *, margin_gb):
        seen["margin"] = margin_gb
        return _plan()

    monkeypatch.setattr(cli.launcher, "plan", plan)
    monkeypatch.setattr(
        cli.launcher,
        "build_argv",
        lambda rest, plan, *, force: rest,
    )
    cli._run(
        ["--model", "mlx-community/test"],
        margin="1.5",
        force=False,
        dry_run=True,
    )

    assert seen["margin"] == 1.5


@pytest.mark.parametrize(
    ("run_args", "expected"),
    [
        (["--margin", "1.5", "--dry-run", "--model=x"], "1.5"),
        (["--margin=1.75", "--dry-run", "--model=x"], "1.75"),
    ],
)
def test_run_raw_parses_margin_syntaxes(monkeypatch, run_args, expected):
    seen = {}

    def run(rest, *, margin, force, dry_run, log, **kwargs):
        seen.update(rest=rest, margin=margin, dry_run=dry_run)

    monkeypatch.setattr(cli, "_run", run)
    cli.cmd_run_raw(run_args)

    assert seen == {
        "rest": ["--model=x"],
        "margin": expected,
        "dry_run": True,
    }


def test_run_rejects_invalid_environment_margin(monkeypatch):
    monkeypatch.setenv(config.MARGIN_ENV, "nan")
    with pytest.raises(SystemExit, match="finite and non-negative"):
        cli._run(
            ["--model", "mlx-community/test"],
            margin=None,
            force=False,
            dry_run=True,
        )


def test_system_displays_configured_margin(monkeypatch, tmp_path, capsys):
    from wmx_suite import db
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "suite.db")
    monkeypatch.setenv(config.MARGIN_ENV, "3")
    monkeypatch.setattr(
        cli,
        "read_limits",
        lambda: SystemLimits(
            device="test",
            total_gb=24.0,
            wall_gb=17.0,
            max_buffer_gb=8.0,
            swap_free_gb=1.0,
            wired_now_gb=3.0,
        ),
    )

    cli.cmd_system(_ns())

    out = capsys.readouterr().out
    assert "safe budget" in out
    assert "14.00 GB" in out
    assert "− 3 GB margin" in out


def test_health_uses_environment_margin(monkeypatch, capsys):
    monkeypatch.setenv(config.MARGIN_ENV, "3")
    limits = SystemLimits(
        device="test",
        total_gb=24.0,
        wall_gb=17.0,
        max_buffer_gb=8.0,
        swap_free_gb=3.0,
        wired_now_gb=3.0,
    )

    class EmptyConnection:
        def execute(self, _sql, _params=()):
            return self

        def fetchone(self):
            return None

        def fetchall(self):
            return []

    monkeypatch.setattr(cli, "read_limits", lambda: limits)
    monkeypatch.setattr(cli, "sample_settled_baseline", lambda: 3.0)
    monkeypatch.setattr(cli.db, "connect", EmptyConnection)

    cli.cmd_health(_ns(margin=None))

    out = capsys.readouterr().out
    assert "safe budget" in out
    assert "14.00 GB" in out
    assert "3 GB margin" in out


def test_characterize_uses_environment_margin(monkeypatch):
    seen = {}
    monkeypatch.setenv(config.MARGIN_ENV, "3")

    def characterize(hf_id, *, margin_gb, allow_min_probe, repeats):
        seen.update(
            hf_id=hf_id,
            margin_gb=margin_gb,
            allow_min_probe=allow_min_probe,
            repeats=repeats,
        )

    monkeypatch.setattr(cli.probe, "characterize", characterize)
    cli.cmd_characterize(
        SimpleNamespace(
            hf_id="mlx-community/test",
            margin=None,
            min_probe=False,
            repeats=3,
        )
    )

    assert seen["margin_gb"] == 3.0


def test_run_warns_when_fit_is_stale(monkeypatch, capsys):
    plan = {**_plan(), "fit_stale": True}
    monkeypatch.setattr(cli.launcher, "plan", lambda _model, margin_gb: plan)
    monkeypatch.setattr(
        cli.launcher,
        "build_argv",
        lambda rest, plan, *, force: rest,
    )

    cli._run(
        ["--model", "mlx-community/test"],
        margin=2.0,
        force=False,
        dry_run=True,
    )

    assert "fit may be stale" in capsys.readouterr().err


def test_list_warns_when_fit_is_stale(monkeypatch, capsys):
    rows = [
        {
            "hf_id": "mlx-community/test",
            "cache_type": "standard",
            "model_base_gb": 8.0,
            "slope_gb_per_k": 0.1,
            "ref_baseline_gb": 3.0,
            "safe_ceiling_ctx": 40000,
            "hard_wall_ctx": 60000,
            "r2": 1.0,
            "characterized_at": "2026-06-13T00:00:00+00:00",
        }
    ]

    connection = object()
    monkeypatch.setattr(cli.db, "connect", lambda: connection)
    monkeypatch.setattr(cli.db, "latest_fits", lambda _con: rows)
    monkeypatch.setattr(cli.db, "gen_speeds", lambda _con: {})
    monkeypatch.setattr(cli.models, "fit_is_stale", lambda _hf_id, _created: True)

    cli.cmd_list(_ns())

    assert "fit may be stale" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Regression test: _stream_worker must not deadlock when stderr > 64 KB
# ---------------------------------------------------------------------------
# The old code captured stderr with PIPE but only read it *after* proc.wait().
# If the worker writes more than the OS pipe buffer (~64 KB) to stderr before
# stdout closes, the worker blocks on the full pipe and the parent blocks
# waiting for stdout → deadlock / hang.
#
# The fix drains stderr concurrently on a background thread so the pipe can
# never fill.  This test spawns a synthetic worker that writes >64 KB to
# stderr then emits one valid JSON status line to stdout and exits 0.  It
# runs _stream_worker inside a thread with a tight timeout to detect a hang.
# ---------------------------------------------------------------------------

_LARGE_STDERR_WORKER = (
    'import sys, json\n'
    'sys.stderr.write("X" * 200_000)\n'
    'sys.stderr.flush()\n'
    'print(json.dumps({"status": "rung_done", "value": 42}), flush=True)\n'
    'sys.exit(0)\n'
)


def test_stream_worker_no_deadlock_with_large_stderr():
    """_stream_worker completes without hanging even if the worker writes >64 KB
    to stderr before stdout closes.  A 10-second timeout detects a hang."""

    received: list[dict] = []

    def on_line(raw_line):
        line = raw_line.strip()
        if line.startswith("{"):
            try:
                received.append(json.loads(line))
            except json.JSONDecodeError:
                pass

    cmd = [sys.executable, "-c", _LARGE_STDERR_WORKER]
    result: list = []
    exc_holder: list = []

    def run():
        try:
            rc, stderr_text = cli._stream_worker(cmd, on_line)
            result.append((rc, stderr_text))
        except Exception as exc:
            exc_holder.append(exc)

    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(timeout=10)

    assert not t.is_alive(), (
        "_stream_worker hung (pipe-deadlock regression): the thread is still "
        "running after the 10-second timeout"
    )
    assert not exc_holder, f"_stream_worker raised unexpectedly: {exc_holder[0]}"
    assert result, "_stream_worker did not return a result"
    returncode, stderr_text = result[0]
    assert returncode == 0, f"Worker exited with unexpected code {returncode}"
    assert len(received) == 1 and received[0].get("status") == "rung_done", (
        f"Expected one rung_done JSON line on stdout; got: {received}"
    )
    # The stderr payload should have been captured (non-empty).
    assert len(stderr_text) > 100_000, (
        f"Expected >100 KB of captured stderr; got {len(stderr_text)} bytes"
    )


# ---------------------------------------------------------------------------
# Regression tests: safeguard_triggered vs error signal distinction
# ---------------------------------------------------------------------------
# These tests cover the two early-stop paths that the refactor collapsed into
# one `stop_requested` flag, causing incorrect exit codes and banner printing.
#
# Test 1 — safeguard_triggered is a SUCCESSFUL early stop:
#   Worker emits warmup_done, two rung_done rungs, then safeguard_triggered,
#   then exits 0.  The handler must exit 0 AND print the completion banner.
#
# Test 2 — error aborts the run and terminates the worker:
#   Worker emits one rung_done, then an error, then MORE rung_done lines that
#   simulate subsequent rungs the worker would have processed.  The handler
#   must exit 1, print the ERROR line, NOT print the completion banner, and
#   NOT persist the post-error rungs to the DB.
# ---------------------------------------------------------------------------

# Synthetic worker: warmup_done, two rungs, safeguard_triggered, exit 0.
# Uses the "main" benchmark fields so we can plug into cmd_benchmark_kokoro.
_SAFEGUARD_WORKER = """\
import sys, json, time

def emit(obj):
    print(json.dumps(obj), flush=True)

emit({"status": "warmup_done"})
emit({"status": "rung_done", "length": 10, "audio_duration": 1.0,
      "compute_time": 0.5, "rtf": 0.5, "cps": 20.0, "peak_gb": 0.3})
emit({"status": "rung_done", "length": 50, "audio_duration": 3.0,
      "compute_time": 1.2, "rtf": 0.4, "cps": 40.0, "peak_gb": 0.4})
emit({"status": "safeguard_triggered", "note": "wired memory threshold reached"})
sys.exit(0)
"""

# Synthetic worker: one rung_done, then error, then MORE rung_done lines that
# MUST NOT be persisted.  Exits 0 (the parent should terminate it before that,
# but we test that even if it didn't, the handler would exit 1 without banner).
_ERROR_WORKER = """\
import sys, json

def emit(obj):
    print(json.dumps(obj), flush=True)

emit({"status": "warmup_done"})
emit({"status": "rung_done", "length": 10, "audio_duration": 1.0,
      "compute_time": 0.5, "rtf": 0.5, "cps": 20.0, "peak_gb": 0.3})
emit({"status": "error", "note": "synthesis failed for length 50"})
# These lines simulate the worker continuing after an error (the bug).
# The parent should terminate the child before these are processed.
emit({"status": "rung_done", "length": 50, "audio_duration": 3.0,
      "compute_time": 1.2, "rtf": 0.4, "cps": 40.0, "peak_gb": 0.4})
emit({"status": "rung_done", "length": 100, "audio_duration": 6.0,
      "compute_time": 2.0, "rtf": 0.33, "cps": 50.0, "peak_gb": 0.5})
sys.exit(0)
"""


def _make_kokoro_args(model="mlx-community/Kokoro-82M-bf16", voice="af_heart"):
    return SimpleNamespace(
        model=model,
        voice=voice,
        lengths="10,50,100",
        repeats=1,
        margin=None,
    )


def _patch_kokoro_env(monkeypatch, add_measurement_tracker=None):
    """Stub out the DB and mlx imports used by cmd_benchmark_kokoro."""
    import types

    # Stub mlx.core so the handler can do `import mlx.core as mx`
    fake_mlx_core = types.ModuleType("mlx.core")
    fake_mlx_core.__version__ = "0.0.0-test"
    fake_mlx = types.ModuleType("mlx")
    fake_mlx.core = fake_mlx_core
    monkeypatch.setitem(sys.modules, "mlx", fake_mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", fake_mlx_core)

    # Stub db.connect + db.start_kokoro_run + db.add_kokoro_measurement
    fake_con = object()
    monkeypatch.setattr(cli.db, "connect", lambda: fake_con)
    monkeypatch.setattr(
        cli.db, "start_kokoro_run",
        lambda con, model, voice, mlx_ver: 42
    )

    calls: list[dict] = []
    if add_measurement_tracker is not None:
        add_measurement_tracker.extend([])  # initialise in place

    def fake_add(con, run_id, **kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(cli.db, "add_kokoro_measurement", fake_add)
    return calls


def test_safeguard_triggered_exits_zero_with_banner(monkeypatch, capsys):
    """safeguard_triggered is a successful early stop: exit 0 + completion banner."""
    calls = _patch_kokoro_env(monkeypatch)

    # Replace the worker command so we run our synthetic script
    original_stream_worker = cli._stream_worker

    def fake_stream_worker(cmd, on_line, *, capture_stderr=True):
        # Run the synthetic safeguard worker instead of the real Kokoro worker
        real_cmd = [sys.executable, "-c", _SAFEGUARD_WORKER]
        return original_stream_worker(real_cmd, on_line, capture_stderr=capture_stderr)

    monkeypatch.setattr(cli, "_stream_worker", fake_stream_worker)

    args = _make_kokoro_args()
    # Handler must exit 0: either returns normally or raises SystemExit(0/None)
    try:
        cli.cmd_benchmark_kokoro(args)
    except SystemExit as exc:
        assert exc.code in (0, None), (
            f"safeguard_triggered must exit 0 (success), got SystemExit({exc.code!r})"
        )

    out = capsys.readouterr().out
    assert "Benchmark complete" in out, (
        "safeguard_triggered must print the completion banner (exit-0 path)"
    )
    assert "safeguard" in out.lower(), (
        "safeguard_triggered must print the safeguard warning"
    )
    # Two pre-safeguard rungs must have been persisted
    assert len(calls) == 2, (
        f"Expected 2 measurements persisted before safeguard; got {len(calls)}"
    )


def test_error_exits_one_no_banner_terminates_worker(monkeypatch, capsys):
    """error aborts the run: exit 1, ERROR printed, no banner, post-error rungs not persisted."""
    calls = _patch_kokoro_env(monkeypatch)

    original_stream_worker = cli._stream_worker

    def fake_stream_worker(cmd, on_line, *, capture_stderr=True):
        real_cmd = [sys.executable, "-c", _ERROR_WORKER]
        return original_stream_worker(real_cmd, on_line, capture_stderr=capture_stderr)

    monkeypatch.setattr(cli, "_stream_worker", fake_stream_worker)

    args = _make_kokoro_args()
    with pytest.raises(SystemExit) as exc_info:
        cli.cmd_benchmark_kokoro(args)

    assert exc_info.value.code == 1, (
        f"error must cause exit 1; got {exc_info.value.code!r}"
    )

    out = capsys.readouterr().out
    assert "ERROR" in out, "error status must print the ERROR line"
    assert "Benchmark complete" not in out, (
        "error must NOT print the completion banner"
    )
    # Only the pre-error rung (length=10) should have been persisted.
    # The post-error rungs (length=50, 100) must NOT be persisted because
    # _stream_worker terminates the child on error.
    assert len(calls) == 1, (
        f"Only 1 measurement (pre-error rung) must be persisted; got {len(calls)}: {calls}"
    )
    assert calls[0]["text_length"] == 10, (
        f"Persisted rung should be length=10 (the pre-error one); got {calls[0]}"
    )


def test_cmd_calibrate_prints_summary(monkeypatch, capsys):
    from types import SimpleNamespace
    from wmx_suite import cli, probe
    monkeypatch.setattr(probe, "calibrate", lambda model, margin_gb=None: {
        "hf_id": "org/tiny", "machine_key": ("Apple M4 Pro", 25769803776, 15),
        "intercept_gb": 2.0, "measured_overhead_gb": 1.48, "fixed_overhead_gb": 1.48,
        "default_overhead_gb": 1.0, "n_points": 2,
    })
    cli.cmd_calibrate(SimpleNamespace(model="org/tiny", margin=None))
    out = capsys.readouterr().out
    assert "org/tiny" in out
    assert "Apple M4 Pro" in out
    assert "1.48" in out


def test_cmd_calibrate_propagates_no_model_error(monkeypatch):
    import pytest
    from types import SimpleNamespace
    from wmx_suite import cli, probe
    def boom(model, margin_gb=None):
        raise SystemExit("[calibrate] no causal mlx-community model found in the HF cache.")
    monkeypatch.setattr(probe, "calibrate", boom)
    with pytest.raises(SystemExit, match="no causal"):
        cli.cmd_calibrate(SimpleNamespace(model=None, margin=None))


def _limits(*, wall_gb=17.0, wired_now_gb=3.0, **kwargs):
    return SystemLimits(
        device="Apple M4 Pro",
        total_gb=24.0,
        wall_gb=wall_gb,
        max_buffer_gb=8.0,
        swap_free_gb=3.0,
        wired_now_gb=wired_now_gb,
    )


def test_health_warns_when_no_profile(monkeypatch, tmp_path, capsys):
    from types import SimpleNamespace
    from wmx_suite import cli, db, profiles
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "suite.db")
    monkeypatch.setattr(profiles, "machine_key", lambda: ("Apple M4 Pro", 25769803776, 15))
    monkeypatch.setattr(cli, "read_limits", lambda: _limits(wall_gb=17.0, wired_now_gb=3.0))
    monkeypatch.setattr(cli, "sample_settled_baseline", lambda: 3.0)
    cli.cmd_health(_ns(margin=None))
    out = capsys.readouterr().out
    assert "No calibration profile" in out
    assert "calibrate" in out


def test_system_reports_no_profile(monkeypatch, tmp_path, capsys):
    from types import SimpleNamespace
    from wmx_suite import cli, db, profiles
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "suite.db")
    monkeypatch.setattr(profiles, "machine_key", lambda: ("Apple M4 Pro", 25769803776, 15))
    monkeypatch.setattr(cli, "read_limits", lambda: _limits(wall_gb=17.0, wired_now_gb=3.0))
    cli.cmd_system(_ns())
    out = capsys.readouterr().out
    assert "calibration" in out.lower()
    assert "uncalibrated" in out.lower()
    assert "calibrate" in out.lower()


def test_health_no_profile_warning_generalizes_to_non_testbed_machine(
    monkeypatch, tmp_path, capsys
):
    """The uncalibrated warning must fire on any M-series machine, not just the M4 Pro
    testbed — guards against special-casing the testbed's machine_key."""
    from types import SimpleNamespace
    from wmx_suite import cli, db, profiles
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "suite.db")
    # A different, never-calibrated SKU (not the M4 Pro testbed).
    monkeypatch.setattr(profiles, "machine_key", lambda: ("Apple M2", 17179869184, 14))
    monkeypatch.setattr(
        cli, "read_limits",
        lambda: SystemLimits(
            device="Apple M2", total_gb=16.0, wall_gb=11.0, max_buffer_gb=6.0,
            swap_free_gb=2.0, wired_now_gb=2.0,
        ),
    )
    monkeypatch.setattr(cli, "sample_settled_baseline", lambda: 2.0)
    cli.cmd_health(_ns(margin=None))
    out = capsys.readouterr().out
    assert "No calibration profile" in out
    assert "Apple M2" in out          # reports the actual machine, not a hardcoded testbed
    assert "calibrate" in out
