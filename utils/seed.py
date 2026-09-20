"""Unified seeding and determinism."""

from __future__ import annotations

import os
import random
from typing import Any, Dict, List

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True, warn_only: bool = True) -> Dict[str, Any]:
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        # Must be set before the first CUDA BLAS handle is created.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    notes: List[str] = []
    if deterministic:
        try:
            torch.use_deterministic_algorithms(True, warn_only=bool(warn_only))
        except Exception as e:
            notes.append(f"use_deterministic_algorithms failed: {e}")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        notes.append("deterministic=True lowers GPU occupancy; configure_gpu() may override for high_utilization")
    return {
        "seed": seed,
        "deterministic": bool(deterministic),
        "warn_only": bool(warn_only),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "warnings": notes,
    }
