# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Will Sarg
import sys

import pytest

from wmx_suite import system
from wmx_suite.system import SystemLimits


class _FakeLib:
    """Stand-in for libSystem so the ctypes seams are exercised without a real syscall."""

    def __init__(self, *, host_stats_kr=0, page_size_kr=0, swap_rc=0, host=1):
        self._host_stats_kr = host_stats_kr
        self._page_size_kr = page_size_kr
        self._swap_rc = swap_rc
        self._host = host
        self.host_self_calls = 0

    def mach_host_self(self):
        self.host_self_calls += 1
        return self._host

    def host_statistics64(self, host, flavor, stats_ref, count_ref):
        return self._host_stats_kr

    def host_page_size(self, host, ps_ref):
        return self._page_size_kr

    def sysctlbyname(self, name, buf_ref, len_ref, newp, newlen):
        return self._swap_rc


def _limits(wall_gb: float = 17.18) -> SystemLimits:
    return SystemLimits(
        device="test",
        total_gb=24.0,
        wall_gb=wall_gb,
        max_buffer_gb=8.0,
        swap_free_gb=1.0,
        wired_now_gb=3.0,
    )


def test_safe_threshold_uses_default_two_gb_margin():
    # AGENTS.md defines the default threshold as wall minus 2 GB.
    assert _limits().safe_threshold_gb() == pytest.approx(15.18)


def test_safe_threshold_uses_requested_margin():
    assert _limits(20.0).safe_threshold_gb(3.5) == pytest.approx(16.5)


def test_macos_major_parses_version(monkeypatch):
    monkeypatch.setattr("platform.mac_ver", lambda: ("15.7.4", ("", "", ""), "arm64"))
    assert system.macos_major() == 15


def test_macos_major_returns_zero_when_undetectable(monkeypatch):
    monkeypatch.setattr("platform.mac_ver", lambda: ("", ("", "", ""), ""))
    assert system.macos_major() == 0


# --- native Mach memory reads (host_statistics64 / sysctlbyname) ---------------------

def test_libsystem_loads_once_and_caches(monkeypatch):
    calls = []

    class FakeCDLL:
        def __init__(self, path, use_errno=False):
            calls.append(path)

    monkeypatch.setattr(system, "_LIB", None)
    monkeypatch.setattr(system.ctypes, "CDLL", FakeCDLL)
    first = system._libsystem()
    second = system._libsystem()
    assert first is second
    assert calls == [system._LIBSYSTEM]


def test_mach_host_acquires_once_and_caches(monkeypatch):
    fake = _FakeLib(host=42)
    monkeypatch.setattr(system, "_libsystem", lambda: fake)
    monkeypatch.setattr(system, "_HOST", None)
    assert system._mach_host() == 42
    assert system._mach_host() == 42
    assert fake.host_self_calls == 1  # acquired once, not per call (no port-ref leak)


def test_mach_wired_pages_returns_struct_field_on_success(monkeypatch):
    monkeypatch.setattr(system, "_libsystem", lambda: _FakeLib(host_stats_kr=0))
    monkeypatch.setattr(system, "_mach_host", lambda: 1)
    assert system._mach_wired_pages() == 0  # zeroed fake struct


def test_mach_wired_pages_raises_on_kernel_error(monkeypatch):
    monkeypatch.setattr(system, "_libsystem", lambda: _FakeLib(host_stats_kr=5))
    monkeypatch.setattr(system, "_mach_host", lambda: 1)
    with pytest.raises(OSError):
        system._mach_wired_pages()


def test_mach_page_size_returns_value_on_success(monkeypatch):
    monkeypatch.setattr(system, "_libsystem", lambda: _FakeLib(page_size_kr=0))
    monkeypatch.setattr(system, "_mach_host", lambda: 1)
    assert system._mach_page_size() == 0  # zeroed fake out-param


def test_mach_page_size_raises_on_kernel_error(monkeypatch):
    monkeypatch.setattr(system, "_libsystem", lambda: _FakeLib(page_size_kr=5))
    monkeypatch.setattr(system, "_mach_host", lambda: 1)
    with pytest.raises(OSError):
        system._mach_page_size()


def test_native_wired_gb_converts_pages_times_pagesize(monkeypatch):
    monkeypatch.setattr(system, "_mach_wired_pages", lambda: 160126)
    monkeypatch.setattr(system, "_mach_page_size", lambda: 16384)
    assert system._native_wired_gb() == pytest.approx(160126 * 16384 / 1e9)


def test_plausible_gb_bounds():
    assert system._plausible_gb(2.6)
    assert not system._plausible_gb(0.0)
    assert not system._plausible_gb(-1.0)
    assert not system._plausible_gb(99999.0)


def test_wired_gb_uses_native_when_available(monkeypatch):
    monkeypatch.setattr(system, "_native_wired_gb", lambda: 2.624)
    monkeypatch.setattr(system, "_wired_gb_vmstat",
                        lambda: pytest.fail("must not fall back when native works"))
    assert system.wired_gb() == pytest.approx(2.624)


def test_wired_gb_falls_back_to_vmstat_on_native_error(monkeypatch):
    def boom():
        raise OSError("kernel said no")
    monkeypatch.setattr(system, "_native_wired_gb", boom)
    monkeypatch.setattr(system, "_wired_gb_vmstat", lambda: 3.0)
    assert system.wired_gb() == 3.0


def test_wired_gb_falls_back_when_native_value_implausible(monkeypatch):
    monkeypatch.setattr(system, "_native_wired_gb", lambda: 0.0)  # implausible
    monkeypatch.setattr(system, "_wired_gb_vmstat", lambda: 3.0)
    assert system.wired_gb() == 3.0


def test_wired_gb_vmstat_fallback_parses_vm_stat(monkeypatch):
    out = ("Mach Virtual Memory Statistics: (page size of 16384 bytes)\n"
           "Pages free:                  1.\n"
           "Pages wired down:       160138.\n")
    monkeypatch.setattr(system.subprocess, "check_output", lambda *a, **k: out.encode())
    assert system._wired_gb_vmstat() == pytest.approx(160138 * 16384 / 1e9)


def test_native_swap_free_gb_returns_avail_on_success(monkeypatch):
    monkeypatch.setattr(system, "_libsystem", lambda: _FakeLib(swap_rc=0))
    monkeypatch.setattr(system, "_mach_host", lambda: 1)
    assert system._native_swap_free_gb() == 0.0  # zeroed fake xsw_usage


def test_native_swap_free_gb_raises_on_error(monkeypatch):
    monkeypatch.setattr(system, "_libsystem", lambda: _FakeLib(swap_rc=-1))
    monkeypatch.setattr(system, "_mach_host", lambda: 1)
    with pytest.raises(OSError):
        system._native_swap_free_gb()


def test_swap_free_gb_uses_native_when_available(monkeypatch):
    monkeypatch.setattr(system, "_native_swap_free_gb", lambda: 1.25)
    monkeypatch.setattr(system, "_swap_free_gb_sysctl",
                        lambda: pytest.fail("must not fall back when native works"))
    assert system.swap_free_gb() == 1.25


def test_swap_free_gb_falls_back_on_native_error(monkeypatch):
    def boom():
        raise OSError()
    monkeypatch.setattr(system, "_native_swap_free_gb", boom)
    monkeypatch.setattr(system, "_swap_free_gb_sysctl", lambda: 0.5)
    assert system.swap_free_gb() == 0.5


@pytest.mark.parametrize("text,expected", [
    ("vm.swapusage: total = 2048.00M  used = 793.56M  free = 1254.44M  (encrypted)",
     1254.44 / 1024),
    ("vm.swapusage: total = 4.00G  used = 1.00G  free = 3.00G", 3.0),
])
def test_swap_free_gb_sysctl_fallback_parses(monkeypatch, text, expected):
    monkeypatch.setattr(system.subprocess, "check_output", lambda *a, **k: text.encode())
    assert system._swap_free_gb_sysctl() == pytest.approx(expected)


def test_swap_free_gb_sysctl_none_when_no_match(monkeypatch):
    monkeypatch.setattr(system.subprocess, "check_output", lambda *a, **k: b"nothing here")
    assert system._swap_free_gb_sysctl() is None


def test_swap_free_gb_sysctl_none_on_subprocess_error(monkeypatch):
    def boom(*a, **k):
        raise OSError()
    monkeypatch.setattr(system.subprocess, "check_output", boom)
    assert system._swap_free_gb_sysctl() is None


@pytest.mark.skipif(sys.platform != "darwin",
                    reason="on-device integration: reads the live macOS Mach kernel")
def test_native_wired_matches_vmstat_on_device():
    """Integration cross-check (read-only; no model load, no allocation): the native Mach
    read agrees with the vm_stat text parse it replaces."""
    assert system._native_wired_gb() == pytest.approx(system._wired_gb_vmstat(), abs=0.3)


def test_sample_settled_baseline_takes_min_of_samples(monkeypatch):
    readings = iter([5.5, 4.0, 4.2])  # transient-high first, then the reclaimed floor
    monkeypatch.setattr(system, "wired_gb", lambda: next(readings))
    monkeypatch.setattr(system.time, "sleep", lambda *_: None)
    assert system.sample_settled_baseline(n=3) == 4.0


def test_read_limits_assembles_systemlimits(monkeypatch):
    monkeypatch.setattr(system, "device_limits", lambda: {
        "device": "M-test", "total_gb": 24.0, "wall_gb": 17.18, "max_buffer_gb": 8.0})
    monkeypatch.setattr(system, "swap_free_gb", lambda: 1.0)
    monkeypatch.setattr(system, "wired_gb", lambda: 3.0)
    limits = system.read_limits()
    assert (limits.device, limits.wall_gb, limits.swap_free_gb, limits.wired_now_gb) == \
        ("M-test", 17.18, 1.0, 3.0)
