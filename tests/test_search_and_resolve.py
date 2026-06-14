"""Short-name resolution, run-flag-anywhere parsing, and `search` command."""
import io
import json
import subprocess
from types import SimpleNamespace

import pytest

from wmx_suite import cli, models
from wmx_suite.ui import Console


def _console(buf):
    return Console(color=False, verbose=False, stream=buf)


# --------------------------------------------------------------------------- #
# models.resolve_hf_id
# --------------------------------------------------------------------------- #
def test_resolve_full_id_unchanged():
    assert models.resolve_hf_id("mlx-community/foo") == "mlx-community/foo"
    assert models.resolve_hf_id("other-org/bar") == "other-org/bar"


def test_resolve_short_defaults_to_mlx_community(monkeypatch):
    monkeypatch.setattr(models, "scan_cache", lambda: [])
    assert models.resolve_hf_id("gemma-4-e4b-it-4bit") == "mlx-community/gemma-4-e4b-it-4bit"


def test_resolve_short_unique_cache_match(monkeypatch):
    monkeypatch.setattr(models, "scan_cache",
                        lambda: ["mlx-community/gemma-4-e4b-it-4bit", "mlx-community/qwen"])
    assert models.resolve_hf_id("gemma-4-e4b-it-4bit") == "mlx-community/gemma-4-e4b-it-4bit"


# --------------------------------------------------------------------------- #
# cmd_run_raw — suite flags recognized ANYWHERE (regression: --dry-run at end)
# --------------------------------------------------------------------------- #
def test_run_flags_after_passthrough(monkeypatch):
    seen = {}
    monkeypatch.setattr(cli, "_run", lambda rest, **kw: seen.update(rest=rest, **kw))
    cli.cmd_run_raw(["--model", "X", "--prompt", "hi", "--max-tokens", "10",
                     "--dry-run", "--force"])
    assert seen["dry_run"] is True
    assert seen["force"] is True
    assert seen["rest"] == ["--model", "X", "--prompt", "hi", "--max-tokens", "10"]


def test_run_double_dash_is_verbatim_passthrough(monkeypatch):
    seen = {}
    monkeypatch.setattr(cli, "_run", lambda rest, **kw: seen.update(rest=rest, **kw))
    cli.cmd_run_raw(["--model", "X", "--", "--dry-run"])
    assert seen["dry_run"] is False                 # not consumed — it's after --
    assert seen["rest"] == ["--model", "X", "--dry-run"]


def test_run_margin_anywhere(monkeypatch):
    seen = {}
    monkeypatch.setattr(cli, "_run", lambda rest, **kw: seen.update(rest=rest, **kw))
    cli.cmd_run_raw(["--model", "X", "--margin", "1.5", "--prompt", "hi"])
    assert seen["margin"] == "1.5"
    assert seen["rest"] == ["--model", "X", "--prompt", "hi"]


# --------------------------------------------------------------------------- #
# cmd_search — shells out to `hf models list`
# --------------------------------------------------------------------------- #
def _fake_proc(stdout="", returncode=0, stderr=""):
    return subprocess.CompletedProcess(args=[], returncode=returncode,
                                       stdout=stdout, stderr=stderr)


def test_search_renders_results(monkeypatch):
    payload = json.dumps([
        {"id": "mlx-community/gemma-4-e4b-it-4bit", "downloads": 43648, "likes": 18,
         "tags": ["mlx", "4-bit"], "library_name": "mlx"},
        {"id": "mlx-community/gemma-4-12B-it-8bit", "downloads": 55348, "likes": 33,
         "tags": ["mlx", "8-bit"], "library_name": "mlx"},
    ])
    monkeypatch.setattr(cli.subprocess, "run", lambda *a, **k: _fake_proc(stdout=payload))
    buf = io.StringIO()
    cli.cmd_search(SimpleNamespace(query="gemma-4", limit=5, all_authors=False,
                                   console=_console(buf)))
    out = buf.getvalue()
    assert "gemma-4-e4b-it-4bit" in out
    assert "4-bit" in out and "8-bit" in out
    assert "hf download mlx-community/gemma-4-e4b-it-4bit" in out  # top result


def test_search_empty_results_guidance(monkeypatch):
    monkeypatch.setattr(cli.subprocess, "run", lambda *a, **k: _fake_proc(stdout="[]"))
    buf = io.StringIO()
    cli.cmd_search(SimpleNamespace(query="zzzzz", limit=5, all_authors=False,
                                   console=_console(buf)))
    assert "No models found" in buf.getvalue()


def test_search_hf_cli_missing(monkeypatch):
    def boom(*a, **k):
        raise FileNotFoundError()
    monkeypatch.setattr(cli.subprocess, "run", boom)
    buf = io.StringIO()
    with pytest.raises(SystemExit):
        cli.cmd_search(SimpleNamespace(query="x", limit=5, all_authors=False,
                                       console=_console(buf)))
    assert "isn't installed" in buf.getvalue()
