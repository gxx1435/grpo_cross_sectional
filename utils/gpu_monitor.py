"""Live nvidia-smi / torch memory sampling. Never fabricates utilization."""

from __future__ import annotations

import csv
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

try:
    import psutil
except ImportError:  # pragma: no cover - recorded as unavailable at runtime
    psutil = None


class GpuMonitor:
    def __init__(self, path: Path, interval: float = 5.0) -> None:
        self.path = Path(path)
        self.interval = float(interval)
        self.rows: List[Dict[str, Any]] = []
        self._stop = threading.Event()
        self._th: Optional[threading.Thread] = None
        self.peak_alloc = 0.0
        self.peak_reserved = 0.0
        self.oom_events = 0

    def start(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._th = threading.Thread(target=self._loop, daemon=True)
        self._th.start()

    def _smi(self) -> Dict[str, Any]:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,utilization.gpu,memory.used,memory.total,driver_version",
                "--format=csv,noheader,nounits",
                "-i",
                "0",
            ],
            text=True,
            timeout=3,
        )
        name, util, mem_u, mem_t, drv = [x.strip() for x in out.strip().split(",")[:5]]
        return {
            "gpu_name": name,
            "util_gpu": float(util),
            "mem_used_mb": float(mem_u),
            "mem_total_mb": float(mem_t),
            "driver": drv,
        }

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            row: Dict[str, Any] = {"t": time.time()}
            try:
                row.update(self._smi())
            except Exception as e:
                row["smi_error"] = str(e)
            if torch.cuda.is_available():
                alloc = torch.cuda.max_memory_allocated() / 1e6
                reserved = torch.cuda.max_memory_reserved() / 1e6
                self.peak_alloc = max(self.peak_alloc, alloc)
                self.peak_reserved = max(self.peak_reserved, reserved)
                row["torch_alloc_mb"] = alloc
                row["torch_reserved_mb"] = reserved
            if psutil is not None:
                row["process_rss_mb"] = psutil.Process().memory_info().rss / 1e6
                row["system_ram_used_mb"] = psutil.virtual_memory().used / 1e6
            self.rows.append(row)

    def record_oom(self) -> None:
        self.oom_events += 1

    def stop(self) -> Dict[str, Any]:
        self._stop.set()
        if self._th:
            self._th.join(timeout=2)
        utils = [r["util_gpu"] for r in self.rows if isinstance(r.get("util_gpu"), float)]
        mems = [r["mem_used_mb"] for r in self.rows if isinstance(r.get("mem_used_mb"), float)]
        rss = [r["process_rss_mb"] for r in self.rows if isinstance(r.get("process_rss_mb"), float)]
        system_ram = [r["system_ram_used_mb"] for r in self.rows if isinstance(r.get("system_ram_used_mb"), float)]
        stats = {
            "n_samples": len(self.rows),
            "interval_s": self.interval,
            "util_gpu_mean": float(sum(utils) / len(utils)) if utils else None,
            "util_gpu_max": float(max(utils)) if utils else None,
            "mem_used_mb_mean": float(sum(mems) / len(mems)) if mems else None,
            "mem_used_mb_max": float(max(mems)) if mems else None,
            "process_rss_mb_mean": float(sum(rss) / len(rss)) if rss else None,
            "process_rss_mb_max": float(max(rss)) if rss else None,
            "system_ram_used_mb_max": float(max(system_ram)) if system_ram else None,
            "torch_peak_alloc_mb": self.peak_alloc,
            "torch_peak_reserved_mb": self.peak_reserved,
            "gpu_name": self.rows[0].get("gpu_name") if self.rows else None,
            "oom_events": self.oom_events,
            "source": "nvidia-smi+torch" if utils else "unavailable",
        }
        if self.rows:
            keys = sorted({k for r in self.rows for k in r})
            with self.path.open("w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader()
                w.writerows(self.rows)
        return stats
