"""GPU sampler via nvidia-smi — package-independent (subprocess only).

Periodically samples GPU utilization + VRAM usage of the target GPU so that the
load demo can deliver a hardware-anchored statement:
"on <GPU> at <VRAM peak> / <util> we sustain <concurrency> at <latency>".
"""

from __future__ import annotations

import shutil
import subprocess
import threading
import time
from typing import Any


def gpu_available() -> bool:
    return shutil.which("nvidia-smi") is not None


def gpu_info() -> dict[str, Any]:
    """Static GPU specs (name, total VRAM, driver)."""
    if not gpu_available():
        return {}
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader,nounits"],
            text=True, timeout=5,
        ).strip().splitlines()
        gpus = []
        for line in out:
            name, mem_total, drv = (x.strip() for x in line.split(","))
            gpus.append({"name": name, "mem_total_mib": float(mem_total), "driver": drv})
        return {"gpus": gpus}
    except Exception:  # noqa: BLE001
        return {}


class GpuSampler:
    """Background sampler: GPU util % + VRAM MiB at the given interval."""

    def __init__(self, interval: float = 1.0, gpu_index: int = 0) -> None:
        self.interval = interval
        self.gpu_index = gpu_index
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.samples: list[dict[str, float]] = []

    def _run(self, t0: float) -> None:
        while not self._stop.is_set():
            self._stop.wait(self.interval)
            try:
                out = subprocess.check_output(
                    ["nvidia-smi", f"--id={self.gpu_index}",
                     "--query-gpu=utilization.gpu,memory.used,power.draw",
                     "--format=csv,noheader,nounits"],
                    text=True, timeout=5,
                ).strip()
                util, mem_used, power = (x.strip() for x in out.split(","))
                self.samples.append({
                    "t": round(time.monotonic() - t0, 2),
                    "util_pct": float(util),
                    "mem_used_mib": float(mem_used),
                    "power_w": float(power) if power not in ("", "[N/A]") else 0.0,
                })
            except Exception:  # noqa: BLE001
                pass

    def start(self) -> None:
        if not gpu_available():
            return
        self._thread = threading.Thread(target=self._run, args=(time.monotonic(),), daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        if not self.samples:
            return {}
        util = [s["util_pct"] for s in self.samples]
        mem = [s["mem_used_mib"] for s in self.samples]
        power = [s["power_w"] for s in self.samples]
        return {
            "samples": self.samples,
            "util_pct": {"mean": round(sum(util) / len(util), 1), "peak": max(util)},
            "mem_used_mib": {"mean": round(sum(mem) / len(mem), 1), "peak": max(mem)},
            "power_w": {"mean": round(sum(power) / len(power), 1), "peak": max(power)},
        }
