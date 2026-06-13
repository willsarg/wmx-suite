import pytest

from wmx_suite import cli


def _plan():
    return {
        "hf_id": "mlx-community/test",
        "kv_bits": 4,
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
