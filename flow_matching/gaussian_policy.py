from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianPolicy(nn.Module):
    def __init__(self, n_assets: int, cond_dim: int, hidden: int) -> None:
        super().__init__()
        self.n_assets = int(n_assets)
        self.backbone = nn.Sequential(nn.Linear(cond_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU())
        self.mu = nn.Linear(hidden, n_assets)
        self.log_std = nn.Parameter(torch.zeros(n_assets))

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        if cond.dim() == 1:
            cond = cond.unsqueeze(0)
        return F.softmax(self.mu(self.backbone(cond)), dim=-1)

    def sample(self, cond: torch.Tensor, n_samples: int) -> torch.Tensor:
        if cond.dim() == 1:
            cond = cond.unsqueeze(0)
        mean = self.mu(self.backbone(cond)).expand(n_samples, -1)
        std = self.log_std.exp().clamp(0.05, 2.0)
        return F.softmax(mean + std * torch.randn_like(mean), dim=-1)
