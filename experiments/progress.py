"""Nested experiment progress bars (store / A–F / month / model / step)."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from typing import Optional

_GPU_POLL_SEC = 30.0
_GPU_CACHE = {"t": 0.0, "s": ""}


def set_gpu_poll_interval(sec: float) -> None:
    global _GPU_POLL_SEC
    _GPU_POLL_SEC = max(float(sec), 5.0)


def _gpu_tag() -> str:
    now = time.time()
    interval = float(os.environ.get("SSFM_GPU_POLL_SEC", _GPU_POLL_SEC))
    if now - float(_GPU_CACHE["t"]) < interval and _GPU_CACHE["s"]:
        return str(_GPU_CACHE["s"])
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=2,
        )
        util, mem, total = [x.strip() for x in out.strip().split(",")[:3]]
        tag = f"GPU {util}% {mem}/{total}MB"
    except Exception:
        tag = ""
    _GPU_CACHE["t"] = now
    _GPU_CACHE["s"] = tag
    return tag


def _bar(frac: float, width: int = 28) -> str:
    frac = min(max(float(frac), 0.0), 1.0)
    n = int(round(width * frac))
    return "#" * n + "-" * (width - n)


class Progress:
    """
    Live bar on stderr (\\r) plus a newline log every `log_every` steps
    so run.log also has a durable trace.
    """

    def __init__(self, total: int, title: str, log_every: int = 1, width: int = 28, always_nl: bool = False) -> None:
        self.total = max(int(total), 1)
        self.title = title
        self.log_every = max(int(log_every), 1)
        self.width = int(width)
        self.always_nl = bool(always_nl)
        self.n = 0
        self.t0 = time.time()
        self._last_msg = ""
        self.render(force_nl=True)

    def _line(self, msg: str, gpu: str) -> str:
        elapsed = time.time() - self.t0
        rate = elapsed / max(self.n, 1)
        remain = max(self.total - self.n, 0)
        eta = rate * remain
        frac = self.n / self.total
        return (
            f"{self.title} [{_bar(frac, self.width)}] {self.n}/{self.total} "
            f"{100.0 * frac:5.1f}% {msg} "
            f"{rate:.2f}s/it elapsed={elapsed/60:.1f}m eta={eta/60:.1f}m {gpu}"
        ).rstrip()

    def render(self, msg: str = "", force_nl: bool = False) -> None:
        gpu = _gpu_tag()
        line = self._line(msg, gpu)
        self._last_msg = line
        if self.always_nl:
            print(line, flush=True)
            return
        cols = shutil.get_terminal_size(fallback=(160, 20)).columns
        sys.stderr.write("\r" + line[: max(cols - 1, 40)].ljust(min(cols - 1, len(line) + 4)))
        sys.stderr.flush()
        if force_nl or self.n == 0 or self.n == self.total or self.n % self.log_every == 0:
            sys.stderr.write("\n")
            sys.stderr.flush()
            print(line, flush=True)

    def update(self, n: Optional[int] = None, msg: str = "") -> None:
        if n is None:
            self.n += 1
        else:
            self.n = min(int(n), self.total)
        self.render(msg)

    def close(self, msg: str = "done") -> None:
        self.n = self.total
        self.render(msg, force_nl=True)


def count_month_units(cfg: dict, n_pred: int, skip_gen: bool, skip_rl: bool, g_grid) -> int:
    """One unit ≈ one visible experiment block inside a month."""
    pools = list((cfg.get("matrix") or {}).get("allocation_pools") or (cfg.get("portfolio") or {}).get("allocation_pools") or ["top30"])
    n_pools = max(len(pools), 1)
    n = n_pred  # A models
    n += n_pools  # B baseline blocks
    if not skip_gen:
        n += 5 * n_pools
    if not skip_rl:
        n += len(cfg["rl"]["algorithms"]) * n_pools
    if g_grid:
        n += len(list(g_grid)) * n_pools
    n += 2  # test inference + audit
    return n
