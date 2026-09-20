from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianPolicy(nn.Module):
    def __init__(self, n_assets: int, cond_dim: int, hidden: int, min_std: float, max_std: float) -> None:
        super().__init__()
        self.n_assets = int(n_assets)
        self.backbone = nn.Sequential(nn.Linear(cond_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU())
        self.mu = nn.Linear(hidden, n_assets)
        self.log_std = nn.Parameter(torch.zeros(n_assets))
        self.min_std = float(min_std)
        self.max_std = float(max_std)

    def mean_logits(self, cond: torch.Tensor) -> torch.Tensor:
        if cond.dim() == 1:
            cond = cond.unsqueeze(0)
        return self.mu(self.backbone(cond))

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.mean_logits(cond), dim=-1)

    def nll(self, cond: torch.Tensor, target_weights: torch.Tensor) -> torch.Tensor:
        target = torch.log(target_weights.clamp_min(1e-8))
        target = target - target.mean(dim=-1, keepdim=True)
        mean = self.mean_logits(cond)
        log_std = self.log_std.clamp(math.log(self.min_std), math.log(self.max_std))
        std = log_std.exp()
        return (0.5 * ((target - mean) / std).square() + log_std).mean()

    def sample(self, cond: torch.Tensor, n_samples: int) -> torch.Tensor:
        if cond.dim() == 1:
            cond = cond.unsqueeze(0)
        mean = self.mean_logits(cond).expand(n_samples, -1)
        std = self.log_std.exp().clamp(self.min_std, self.max_std)
        return F.softmax(mean + std * torch.randn_like(mean), dim=-1)
