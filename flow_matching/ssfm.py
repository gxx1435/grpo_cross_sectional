from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from flow_matching.clr import clr, inv_clr
from flow_matching.simplex import tangent_project


class SimplexSpaceFlow(nn.Module):
    def __init__(self, n_assets: int, cond_dim: int, hidden: int, n_teachers: int, teacher_emb: int) -> None:
        super().__init__()
        self.n_assets = int(n_assets)
        self.n_teachers = int(n_teachers)
        self.teacher_emb = nn.Embedding(n_teachers, teacher_emb)
        self.net = nn.Sequential(
            nn.Linear(n_assets + 1 + cond_dim + teacher_emb, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, n_assets),
        )

    def forward(self, z_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor, teacher_id: torch.Tensor) -> torch.Tensor:
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        if teacher_id.dim() == 0:
            teacher_id = teacher_id.expand(z_t.size(0))
        raw = self.net(torch.cat([z_t, t, cond, self.teacher_emb(teacher_id.long())], dim=-1))
        return tangent_project(raw)


def ss_fm_loss(model: SimplexSpaceFlow, teacher_w: torch.Tensor, cond: torch.Tensor, teacher_id: torch.Tensor, eps: float) -> torch.Tensor:
    z1 = clr(teacher_w, eps)
    z0 = tangent_project(torch.randn_like(z1))
    tau = torch.rand(teacher_w.size(0), 1, device=teacher_w.device, dtype=teacher_w.dtype)
    z_t = (1.0 - tau) * z0 + tau * z1
    u = tangent_project(z1 - z0)
    return F.mse_loss(model(z_t, tau, cond, teacher_id), u)


@torch.no_grad()
def sample_ss_fm(
    model: SimplexSpaceFlow,
    cond: torch.Tensor,
    teacher_id: int,
    n_samples: int,
    n_steps: int,
    seed_noise: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if cond.dim() == 1:
        cond = cond.unsqueeze(0)
    g, k = int(n_samples), model.n_assets
    cond_g = cond.expand(g, -1)
    tid = torch.full((g,), int(teacher_id), device=cond.device, dtype=torch.long)
    z = tangent_project(torch.randn(g, k, device=cond.device) if seed_noise is None else seed_noise.to(cond.device))
    dt = 1.0 / max(n_steps, 1)
    for s in range(n_steps):
        t = torch.full((g, 1), s * dt, device=cond.device, dtype=z.dtype)
        z = tangent_project(z + dt * model(z, t, cond_g, tid))
    return inv_clr(z)


def sample_ss_fm_mixed(model: SimplexSpaceFlow, cond: torch.Tensor, n_samples: int, n_steps: int) -> torch.Tensor:
    chunks = []
    rem = n_samples
    per = max(n_samples // model.n_teachers, 1)
    for tid in range(model.n_teachers):
        g = per if tid < model.n_teachers - 1 else rem
        if g <= 0:
            break
        chunks.append(sample_ss_fm(model, cond, tid, g, n_steps))
        rem -= g
    return torch.cat(chunks, dim=0)[:n_samples]
