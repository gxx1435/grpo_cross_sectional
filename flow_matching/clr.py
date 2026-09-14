from __future__ import annotations

import torch
import torch.nn.functional as F


def clr(w: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    logw = torch.log(w.clamp_min(eps))
    return logw - logw.mean(dim=-1, keepdim=True)


def inv_clr(z: torch.Tensor) -> torch.Tensor:
    return F.softmax(z, dim=-1)
