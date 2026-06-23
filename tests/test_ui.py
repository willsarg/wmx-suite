# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
"""Tests for the shared CLI rendering schema (wmx_suite/ui.py).

Hardware-free and pure: no model loads, no live memory probing, no DB. Color is
tested by asserting on raw ANSI codes; plain mode is asserted byte-exact so the
schema's text contract is locked.
"""
from __future__ import annotations

import io

import pytest

from wmx_suite.ui import ROLES, Console


class FakeStream:
    """Minimal stream with a controllable isatty()."""

    def __init__(self, tty: bool):
        self._tty = tty
        self.buf = io.StringIO()

    def isatty(self) -> bool:
        return self._tty

    def write(self, s: str) -> int:
        return self.buf.write(s)

    def getvalue(self) -> str:
        return self.buf.getvalue()


# --------------------------------------------------------------------------- #
# style
# --------------------------------------------------------------------------- #
def test_style_color_wraps_with_code_and_reset():
    con = Console(color=True, verbose=False)
    out = con.style("good", "ok")
    assert out == f"\033[{ROLES['good']}mok\033[0m"
    assert "\033[0m" in out


def test_style_no_color_is_byte_identical():
    con = Console(color=False, verbose=False)
    out = con.style("good", "ok")
    assert out == "ok"
    assert "\033" not in out


def test_style_unknown_role_returns_text_unchanged_even_with_color():
    con = Console(color=True, verbose=False)
    assert con.style("nope", "x") == "x"


# --------------------------------------------------------------------------- #
# from_args / color policy
# --------------------------------------------------------------------------- #
def test_from_args_color_true_only_for_tty_no_nocolor_env_unset(monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    con = Console.from_args(stream=FakeStream(tty=True), no_color=False, verbose=False)
    assert con.color is True


def test_from_args_color_false_when_not_tty(monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    con = Console.from_args(stream=FakeStream(tty=False), no_color=False)
    assert con.color is False


def test_from_args_color_false_when_no_color_flag(monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    con = Console.from_args(stream=FakeStream(tty=True), no_color=True)
    assert con.color is False


def test_from_args_color_false_when_no_color_env(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    con = Console.from_args(stream=FakeStream(tty=True), no_color=False)
    assert con.color is False


def test_from_args_color_false_when_no_color_env_empty(monkeypatch):
    # no-color.org: presence disables color regardless of value (NO_COLOR=)
    monkeypatch.setenv("NO_COLOR", "")
    con = Console.from_args(stream=FakeStream(tty=True), no_color=False)
    assert con.color is False


def test_from_args_verbose_passthrough(monkeypatch):
    monkeypatch.delenv("NO_COLOR", raising=False)
    con = Console.from_args(stream=FakeStream(tty=True), verbose=True)
    assert con.verbose is True


# --------------------------------------------------------------------------- #
# field
# --------------------------------------------------------------------------- #
def test_field_shape_plain():
    con = Console(color=False, verbose=False)
    out = con.field("device", "Apple M4 Pro")
    # 16-wide label, then ": " (colon at a fixed column), then value
    assert out == "device          : Apple M4 Pro"


def test_field_gloss_appended():
    con = Console(color=False, verbose=False)
    out = con.field("total RAM", "25.77 GB", "physical memory installed")
    assert out.startswith("total RAM")
    assert "25.77 GB" in out
    assert out.endswith("physical memory installed")
    assert "25.77 GB" in out and "physical memory installed" in out


def test_field_indent():
    con = Console(color=False, verbose=False)
    out = con.field("safe budget", "15.18 GB", indent=2)
    assert out.startswith("  ")


# --------------------------------------------------------------------------- #
# table
# --------------------------------------------------------------------------- #
def test_table_alignment_autowidth_header_plain():
    con = Console(color=False, verbose=False)
    cols = [("model", "l", "value"), ("loads at", "r", "value")]
    rows = [("short", "5.0 GB"), ("a-much-longer-name", "12.4 GB")]
    out = con.table(cols, rows)
    lines = out.splitlines()
    # header present
    assert "model" in lines[0]
    assert "loads at" in lines[0]
    # every line same visual length (consistent columns)
    assert len({len(l) for l in lines}) == 1
    # right-aligned numeric column lines up: the "GB" ends at the same offset
    idx = [l.rindex("GB") for l in lines[1:]]
    assert len(set(idx)) == 1


def test_table_per_cell_role_only_when_color():
    cols = [("model", "l", "value"), ("fit", "l", "value")]
    rows = [("m1", ("good", "good"))]
    plain = Console(color=False, verbose=False).table(cols, rows)
    assert "\033" not in plain
    colored = Console(color=True, verbose=False).table(cols, rows)
    assert f"\033[{ROLES['good']}m" in colored


def test_table_left_align():
    con = Console(color=False, verbose=False)
    cols = [("a", "l", "value")]
    rows = [("x",), ("longer",)]
    out = con.table(cols, rows)
    lines = out.splitlines()
    assert lines[1].startswith("x")
    assert lines[2].startswith("longer")


# --------------------------------------------------------------------------- #
# glyph
# --------------------------------------------------------------------------- #
def test_glyph_symbols():
    con = Console(color=False, verbose=False)
    assert con.glyph("ok") == "✓"
    assert con.glyph("bad") == "✗"
    assert con.glyph("warn") == "⚠"


def test_glyph_colored():
    con = Console(color=True, verbose=False)
    assert con.glyph("ok") == f"\033[{ROLES['good']}m✓\033[0m"
    assert con.glyph("bad") == f"\033[{ROLES['bad']}m✗\033[0m"
    assert con.glyph("warn") == f"\033[{ROLES['warn']}m⚠\033[0m"


# --------------------------------------------------------------------------- #
# next_block / guidance / status_line
# --------------------------------------------------------------------------- #
def test_next_block_structure():
    con = Console(color=False, verbose=False)
    out = con.next_block([
        ("wmx-suite health", "which models can run now"),
        ("wmx-suite list", "ceilings measured"),
    ])
    assert "next" in out
    assert "wmx-suite health" in out
    assert "which models can run now" in out
    assert "wmx-suite list" in out
    assert "ceilings measured" in out


def test_guidance_structure():
    con = Console(color=False, verbose=False)
    out = con.guidance(
        "Won't run gpt-oss",
        ["loading it needs 15.56 GB", "no room for context"],
        [("wmx-suite health", "see which models fit"),
         ("run --force", "override at your own risk")],
    )
    assert "Won't run gpt-oss" in out
    assert "why" in out
    assert "loading it needs 15.56 GB" in out
    assert "no room for context" in out
    assert "try" in out
    assert "wmx-suite health" in out
    assert "see which models fit" in out
    assert "run --force" in out


def test_status_line_join():
    con = Console(color=False, verbose=False)
    out = con.status_line([("Apple M4 Pro", "metric"), ("12.04 GB free", "good")])
    assert out == "Apple M4 Pro · 12.04 GB free"


# --------------------------------------------------------------------------- #
# raw (verbose appendix)
# --------------------------------------------------------------------------- #
def test_raw_empty_when_not_verbose():
    con = Console(color=False, verbose=False)
    assert con.raw("raw (--verbose)", ["wall bytes 18446744073"]) == ""


def test_raw_present_when_verbose():
    con = Console(color=False, verbose=True)
    out = con.raw("raw (--verbose)", ["wall bytes 18446744073"])
    assert out != ""
    assert "raw (--verbose)" in out
    assert "wall bytes 18446744073" in out


# --------------------------------------------------------------------------- #
# section / emit
# --------------------------------------------------------------------------- #
def test_section_contains_title():
    con = Console(color=False, verbose=False)
    assert "memory budget" in con.section("memory budget")


def test_emit_writes_to_stream():
    s = FakeStream(tty=False)
    con = Console(color=False, verbose=False, stream=s)
    con.emit("hello")
    assert s.getvalue() == "hello\n"


# --------------------------------------------------------------------------- #
# Plain-mode golden — locks the text contract
# --------------------------------------------------------------------------- #
def test_plain_mode_golden():
    con = Console(color=False, verbose=False)
    parts = [
        con.section("memory budget"),
        con.field("total RAM", "25.77 GB", "physical memory installed"),
        con.field("safe budget", "15.18 GB", indent=2),
        con.next_block([("wmx-suite health", "which models can run now")]),
    ]
    out = "\n".join(parts)
    expected = (
        "memory budget\n"
        "total RAM       : 25.77 GB   physical memory installed\n"
        "  safe budget   : 15.18 GB\n"
        "\nnext\n"
        "  wmx-suite health   which models can run now"
    )
    assert out == expected
    assert "\033" not in out
