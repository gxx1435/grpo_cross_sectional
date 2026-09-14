from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class StandardFlowMatching(nn.Module):
    def __init__(self, n_assets: int, cond_dim: int, hidden: int) -> None:
        super().__init__()
        self.n_assets = int(n_assets)
        self.net = nn.Sequential(
            nn.Linear(n_assets + 1 + cond_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, n_assets),
        )

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        return self.net(torch.cat([x_t, t, cond], dim=-1))


def std_fm_loss(model: StandardFlowMatching, w: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
    w_unc = torch.log(w.clamp_min(1e-8))
    z = torch.randn_like(w_unc)
    tau = torch.rand(w.size(0), 1, device=w.device, dtype=w.dtype)
    x = (1.0 - tau) * z + tau * w_unc
    return F.mse_loss(model(x, tau, cond), w_unc - z)


@torch.no_grad()
def sample_std_fm(model: StandardFlowMatching, cond: torch.Tensor, n_samples: int, n_steps: int) -> torch.Tensor:
    if cond.dim() == 1:
        cond = cond.unsqueeze(0)
    g, k = int(n_samples), model.n_assets
    x = torch.randn(g, k, device=cond.device)
    cond_g = cond.expand(g, -1)
    dt = 1.0 / max(n_steps, 1)
    for s in range(n_steps):
        t = torch.full((g, 1), s * dt, device=cond.device, dtype=x.dtype)
        x = x + dt * model(x, t, cond_g)
    return F.softmax(x, dim=-1)
