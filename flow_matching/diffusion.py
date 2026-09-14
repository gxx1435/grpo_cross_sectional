from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightDiffusion(nn.Module):
    def __init__(self, n_assets: int, cond_dim: int, hidden: int, n_steps: int) -> None:
        super().__init__()
        self.n_assets = int(n_assets)
        self.n_steps = int(n_steps)
        self.net = nn.Sequential(
            nn.Linear(n_assets + 1 + cond_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, n_assets),
        )
        betas = torch.linspace(1e-4, 0.02, self.n_steps)
        self.register_buffer("alphas_cumprod", torch.cumprod(1.0 - betas, dim=0))

    def forward(self, x_t: torch.Tensor, t_idx: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        t = (t_idx.float() / max(self.n_steps - 1, 1)).unsqueeze(-1)
        return self.net(torch.cat([x_t, t, cond], dim=-1))


def diffusion_loss(model: WeightDiffusion, w: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
    x0 = torch.log(w.clamp_min(1e-8))
    t = torch.randint(0, model.n_steps, (w.size(0),), device=w.device)
    noise = torch.randn_like(x0)
    a = model.alphas_cumprod[t].unsqueeze(-1)
    x_t = torch.sqrt(a) * x0 + torch.sqrt(1.0 - a) * noise
    return F.mse_loss(model(x_t, t, cond), noise)


@torch.no_grad()
def sample_diffusion(model: WeightDiffusion, cond: torch.Tensor, n_samples: int) -> torch.Tensor:
    if cond.dim() == 1:
        cond = cond.unsqueeze(0)
    g, k = int(n_samples), model.n_assets
    x = torch.randn(g, k, device=cond.device)
    cond_g = cond.expand(g, -1)
    for i in reversed(range(model.n_steps)):
        t = torch.full((g,), i, device=cond.device, dtype=torch.long)
        eps = model(x, t, cond_g)
        a = model.alphas_cumprod[i]
        a_prev = model.alphas_cumprod[i - 1] if i > 0 else torch.tensor(1.0, device=cond.device)
        x0 = (x - torch.sqrt(1 - a) * eps) / torch.sqrt(a).clamp_min(1e-8)
        x = x0 if i == 0 else torch.sqrt(a_prev) * x0 + torch.sqrt(1.0 - a_prev) * torch.randn_like(x)
    return F.softmax(x, dim=-1)
