from types import SimpleNamespace

import pytest

from wmx_suite import cli, config
from wmx_suite.system import SystemLimits


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

    def run(rest, *, margin, force, dry_run, log):
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


def test_system_displays_configured_margin(monkeypatch, capsys):
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

    cli.cmd_system(None)

    assert "safe threshold       : 14.00 GB  (wall − 3GB margin)" in capsys.readouterr().out


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
        def execute(self, _sql):
            return self

        def fetchall(self):
            return []

    monkeypatch.setattr(cli, "read_limits", lambda: limits)
    monkeypatch.setattr(cli, "sample_settled_baseline", lambda: 3.0)
    monkeypatch.setattr(cli.db, "connect", EmptyConnection)

    cli.cmd_health(SimpleNamespace(margin=None))

    assert "safe threshold      : 14.00 GB  (wall − 3GB margin)" in capsys.readouterr().out


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

    cli.cmd_list(None)

    assert "fit may be stale" in capsys.readouterr().out
