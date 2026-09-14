from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPPolicy(nn.Module):
    def __init__(self, n_assets: int, cond_dim: int, hidden: int) -> None:
        super().__init__()
        self.n_assets = int(n_assets)
        self.net = nn.Sequential(nn.Linear(cond_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, n_assets))

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        if cond.dim() == 1:
            cond = cond.unsqueeze(0)
        return F.softmax(self.net(cond), dim=-1)
