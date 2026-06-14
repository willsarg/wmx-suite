"""Task 1.3: global --verbose/--no-color plumbing in cli.main().

Behavior-preserving: these only assert that the flags parse on every subcommand,
that a Console is attached to args, and that the `run` fast path strips the flags
out of the argv forwarded to mlx_lm.generate.
"""
from __future__ import annotations

import pytest

from wmx_suite import cli
from wmx_suite.ui import Console


def test_subcommands_accept_global_flags_and_attach_console(monkeypatch):
    captured = {}

    def fake_func(args):
        captured["args"] = args

    # Route `system` to our spy so we can inspect the parsed namespace.
    monkeypatch.setattr(cli, "cmd_system", fake_func)
    cli._main_argparse(["system", "--verbose", "--no-color"])

    args = captured["args"]
    assert args.verbose is True
    assert args.no_color is True
    assert isinstance(args.console, Console)
    # --no-color forces color off regardless of TTY.
    assert args.console.color is False
    assert args.console.verbose is True


def test_global_flags_default_off(monkeypatch):
    captured = {}
    monkeypatch.setattr(cli, "cmd_system", lambda a: captured.update(args=a))
    cli._main_argparse(["system"])
    args = captured["args"]
    assert args.verbose is False
    assert args.no_color is False
    assert isinstance(args.console, Console)


@pytest.mark.parametrize("cmd", ["system", "scan", "list", "health"])
def test_global_flags_on_multiple_subcommands(monkeypatch, cmd):
    captured = {}
    # Spy on whichever command we route to so no real work runs.
    monkeypatch.setattr(cli, f"cmd_{cmd}", lambda a: captured.update(args=a))
    cli._main_argparse([cmd, "--verbose"])
    assert captured["args"].verbose is True
    assert isinstance(captured["args"].console, Console)


def test_run_path_strips_global_flags(monkeypatch):
    forwarded = {}

    def fake_run_raw(run_args):
        forwarded["run_args"] = run_args

    monkeypatch.setattr(cli, "cmd_run_raw", fake_run_raw)
    monkeypatch.setattr(cli.sys, "argv",
                        ["wmx-suite", "run", "--model", "X", "--verbose",
                         "--max-tokens", "10", "--no-color"])
    cli.main()

    # The UX flags must NOT leak into the mlx_lm.generate passthrough.
    assert forwarded["run_args"] == ["--model", "X", "--max-tokens", "10"]
    # A module-level Console was built for the run path.
    assert isinstance(cli.CONSOLE, Console)
    assert cli.CONSOLE.verbose is True


def test_strip_global_flags_helper():
    rem, verbose, no_color = cli._strip_global_flags(
        ["--model", "X", "-v", "--prompt", "hi"])
    assert rem == ["--model", "X", "--prompt", "hi"]
    assert verbose is True
    assert no_color is False


# --------------------------------------------------------------------------- #
# Front door (no subcommand) — Phase 1.5
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("argv,want_verbose", [
    ([], False),
    (["--verbose"], True),
    (["-v"], True),
    (["--help"], False),
    (["-h"], False),
    (["--no-color"], False),
])
def test_bare_invocation_reaches_landing(monkeypatch, argv, want_verbose):
    captured = {}
    monkeypatch.setattr(cli, "cmd_landing",
                        lambda console: captured.update(console=console))
    monkeypatch.setattr(cli.sys, "argv", ["wmx-suite", *argv])
    cli.main()  # must NOT raise (no argparse "required: cmd" error)
    console = captured["console"]
    assert isinstance(console, Console)
    assert console.verbose is want_verbose


def test_subcommand_does_not_trigger_landing(monkeypatch):
    captured = {}
    monkeypatch.setattr(cli, "cmd_landing",
                        lambda console: captured.update(landed=True))
    monkeypatch.setattr(cli, "cmd_system", lambda a: captured.update(ran=True))
    monkeypatch.setattr(cli.sys, "argv", ["wmx-suite", "system"])
    cli.main()
    assert captured.get("ran") is True
    assert "landed" not in captured
