from __future__ import annotations

from typing import Callable

import torch

from flow_matching.simplex import simplex_violation


def sample_and_audit(fn: Callable[[], torch.Tensor]) -> tuple[torch.Tensor, dict]:
    w = fn()
    viol = simplex_violation(w)
    if not viol["ok"]:
        w = torch.softmax(torch.log(w.clamp_min(1e-8)), dim=-1)
        viol = simplex_violation(w)
        viol["repaired"] = True
    else:
        viol["repaired"] = False
    return w, viol
