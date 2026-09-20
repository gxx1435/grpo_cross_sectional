from __future__ import annotations

from typing import Callable

import torch

from flow_matching.simplex import simplex_violation


def sample_and_audit(fn: Callable[[], torch.Tensor]) -> tuple[torch.Tensor, dict]:
    w = fn()
    viol = simplex_violation(w)
    if not viol["ok"]:
        raise FloatingPointError(f"sampled portfolio violates simplex: {viol}")
    viol["repaired"] = False
    return w, viol
