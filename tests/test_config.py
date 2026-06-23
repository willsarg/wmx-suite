# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
import pytest

from wmx_suite import config


def test_margin_defaults_to_two_gb(monkeypatch):
    monkeypatch.delenv(config.MARGIN_ENV, raising=False)
    assert config.margin_gb() == 2.0


def test_margin_reads_environment(monkeypatch):
    monkeypatch.setenv(config.MARGIN_ENV, "3.5")
    assert config.margin_gb() == 3.5


def test_explicit_margin_overrides_environment(monkeypatch):
    monkeypatch.setenv(config.MARGIN_ENV, "3.5")
    assert config.margin_gb(1.25) == 1.25


@pytest.mark.parametrize("value", ["bad", "-1", "nan", "inf", "-inf"])
def test_margin_rejects_unsafe_values(value):
    with pytest.raises(ValueError):
        config.margin_gb(value)


def test_zero_margin_is_an_explicit_supported_value():
    assert config.margin_gb(0) == 0.0
