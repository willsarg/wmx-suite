"""System memory facts for this machine.

The crash wall is NOT total RAM — it's the GPU/Metal recommended working-set size,
because MLX allocates wired (non-swappable) Metal buffers. On the M4 Pro (24GB) this
is ~17.18 GB (67% of total). Crossing it, with almost no swap, can hard-lock the system.
"""
from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass


@dataclass(frozen=True)
class SystemLimits:
    device: str
    total_gb: float
    wall_gb: float          # max_recommended_working_set_size — the crash wall
    max_buffer_gb: float    # largest single allocation Metal allows
    swap_free_gb: float | None
    wired_now_gb: float      # OS-wired memory right now (baseline pressure)

    def safe_threshold_gb(self, margin_gb: float = 2.0) -> float:
        """The line we never let predicted peak cross. Default keeps a 2 GB cushion."""
        return self.wall_gb - margin_gb


def device_limits() -> dict:
    import mlx.core as mx

    d = mx.device_info()
    return {
        "device": str(d.get("device_name", "")),
        "total_gb": d.get("memory_size", 0) / 1e9,
        "wall_gb": d.get("max_recommended_working_set_size", 0) / 1e9,
        "max_buffer_gb": d.get("max_buffer_length", 0) / 1e9,
    }


def swap_free_gb() -> float | None:
    try:
        out = subprocess.check_output(["sysctl", "vm.swapusage"]).decode()
    except Exception:
        return None
    m = re.search(r"free = ([\d.]+)([MG])", out)
    if not m:
        return None
    v = float(m.group(1))
    return v / 1024 if m.group(2) == "M" else v


def wired_gb() -> float:
    """OS-wired memory in GB, from vm_stat 'Pages wired down'. This is the metric that
    actually predicts a crash — MLX's own get_peak_memory() undercounts it by ~40%
    because it excludes the buffer cache, which the OS still wires."""
    out = subprocess.check_output(["vm_stat"]).decode()
    page_size = 4096
    wired_pages = 0
    for line in out.splitlines():
        if "page size of" in line:
            page_size = int(line.split()[-2])
        if "Pages wired down" in line:
            wired_pages = int(line.split()[-1].strip("."))
    return wired_pages * page_size / 1e9


def read_limits() -> SystemLimits:
    d = device_limits()
    return SystemLimits(
        device=d["device"],
        total_gb=d["total_gb"],
        wall_gb=d["wall_gb"],
        max_buffer_gb=d["max_buffer_gb"],
        swap_free_gb=swap_free_gb(),
        wired_now_gb=wired_gb(),
    )
