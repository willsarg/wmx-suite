"""Per-machine cold-start constants for the pre-flight base estimate.

The crash wall, the ambient baseline, and per-model fits already adapt to the host
at runtime. The only M4-Pro-tuned part is the cold-start base estimate for models that
haven't been characterized yet. `wmx-suite calibrate` measures this machine's real
FIXED_OVERHEAD_GB once and stores it (keyed by chip + RAM + macOS major) so the estimate
is trustworthy on any Apple Silicon SKU. `characterize` remains the per-model mechanism.
"""
from __future__ import annotations

import sqlite3

from . import db, system

# Loose priors, measured loosely on the M4 Pro (see probe.py). The resident factor is
# HELD FIXED (effective factor measured 0.88-1.10 across models); only the overhead is
# calibrated per machine, and never stored below this default (calibration only tightens).
DEFAULT_RESIDENT_FACTOR = 1.05
DEFAULT_FIXED_OVERHEAD_GB = 1.0


def machine_key() -> tuple[str, int, int]:
    """(device_name, total_ram_bytes, macos_major) identifying the current machine.

    total RAM is read in BYTES from mx.device_info()['memory_size'] (SystemLimits.total_gb
    is already divided by 1e9, so it can't supply a stable integer key).
    """
    import mlx.core as mx
    d = mx.device_info()
    return (str(d.get("device_name", "")), int(d.get("memory_size", 0)), system.macos_major())


def cold_start_constants(con: sqlite3.Connection) -> tuple[float, float, str]:
    """Return (resident_factor, fixed_overhead_gb, source).

    source == "profile" when a stored profile matches this machine, else "default".
    The factor is always the default (held fixed); only the overhead is profile-specific.
    """
    profile = db.get_profile(con, machine_key())
    if profile is not None:
        return DEFAULT_RESIDENT_FACTOR, float(profile["fixed_overhead_gb"]), "profile"
    return DEFAULT_RESIDENT_FACTOR, DEFAULT_FIXED_OVERHEAD_GB, "default"
