"""Git / environment snapshot."""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from typing import Any, Dict

import torch


def git_commit(root: str) -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
            timeout=5,
            stderr=subprocess.DEVNULL,
        )
        return out.strip()
    except Exception:
        return "unknown"


def environment_info(root: str) -> Dict[str, Any]:
    gpu_name = None
    gpu_mem = None
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = int(torch.cuda.get_device_properties(0).total_memory)
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "pytorch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu_name": gpu_name,
        "gpu_total_memory": gpu_mem,
        "git_commit": git_commit(root),
        "cwd": os.getcwd(),
    }
