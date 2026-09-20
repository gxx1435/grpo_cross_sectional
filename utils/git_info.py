"""Git / environment snapshot."""

from __future__ import annotations

import os
import platform
import subprocess
import sys
import hashlib
from pathlib import Path
from importlib import metadata
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
    try:
        driver = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader", "-i", "0"],
            text=True,
            timeout=5,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        driver = None
    packages = {
        str(dist.metadata.get("Name") or "unknown"): str(dist.version)
        for dist in metadata.distributions()
    }
    try:
        raw_status = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=all"], cwd=root, text=True, timeout=10)
        ignored_artifact_prefixes = ("results/", "data/processed/")
        status_lines = []
        for line in raw_status.splitlines():
            path_part = line[3:] if len(line) > 3 else ""
            if path_part.startswith(ignored_artifact_prefixes):
                continue
            status_lines.append(line)
        status = "\n".join(status_lines) + ("\n" if status_lines else "")
        diff = subprocess.check_output(["git", "diff", "--binary", "HEAD"], cwd=root, timeout=30)
        untracked = subprocess.check_output(
            ["git", "ls-files", "--others", "--exclude-standard", "-z"], cwd=root, timeout=10
        ).split(b"\0")
        dirty_hash = hashlib.sha256()
        dirty_hash.update(status.encode("utf-8"))
        dirty_hash.update(diff)
        for rel_bytes in sorted(
            value
            for value in untracked
            if value and not value.decode("utf-8", errors="surrogateescape").startswith(ignored_artifact_prefixes)
        ):
            dirty_hash.update(rel_bytes)
            path = Path(root) / rel_bytes.decode("utf-8", errors="surrogateescape")
            if path.is_file():
                dirty_hash.update(path.read_bytes())
        git_dirty = bool(status)
        git_diff_sha256 = dirty_hash.hexdigest() if git_dirty else None
    except Exception:
        status = "unavailable"
        git_dirty = None
        git_diff_sha256 = None
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "pytorch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu_name": gpu_name,
        "gpu_total_memory": gpu_mem,
        "gpu_driver": driver,
        "cuda_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cuda_deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "packages": dict(sorted(packages.items(), key=lambda item: item[0].lower())),
        "git_commit": git_commit(root),
        "git_dirty": git_dirty,
        "git_status_porcelain": status.splitlines(),
        "git_diff_sha256": git_diff_sha256,
        "cwd": os.getcwd(),
    }
