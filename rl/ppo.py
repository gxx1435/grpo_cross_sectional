from __future__ import annotations

from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ValueHead(nn.Module):
    def __init__(self, cond_dim: int, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(cond_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        if cond.dim() == 1:
            cond = cond.unsqueeze(0)
        return self.net(cond).squeeze(-1)


class WeightPolicy(nn.Module):
    def __init__(self, cond_dim: int, n_assets: int, hidden: int) -> None:
        super().__init__()
        self.n_assets = int(n_assets)
        self.net = nn.Sequential(nn.Linear(cond_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, n_assets))

    def forward_logits(self, cond: torch.Tensor) -> torch.Tensor:
        if cond.dim() == 1:
            cond = cond.unsqueeze(0)
        return self.net(cond)

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.forward_logits(cond), dim=-1)


def gaussian_log_prob(mean: torch.Tensor, sample: torch.Tensor, std: float) -> torch.Tensor:
    if mean.dim() == 1:
        mean = mean.unsqueeze(0).expand_as(sample)
    var = float(std) ** 2
    return -0.5 * (((sample - mean) ** 2) / var + np.log(2 * np.pi * var)).sum(dim=-1)


@torch.no_grad()
def sample_portfolios(policy: WeightPolicy, cond: torch.Tensor, n_samples: int, noise_std: float):
    logits = policy.forward_logits(cond).squeeze(0)
    eps = torch.randn(int(n_samples), logits.numel(), device=logits.device) * float(noise_std)
    logits_s = logits.unsqueeze(0) + eps
    return F.softmax(logits_s, dim=-1), logits_s, gaussian_log_prob(logits, logits_s, noise_std)


def ppo_update(policy, critic, cond, sample_logits, rewards, optimizer, cfg: dict) -> Dict[str, float]:
    p = cfg["rl"]["ppo"]
    noise = float(cfg["rl"]["noise_std"])
    with torch.no_grad():
        old_mean = policy.forward_logits(cond).squeeze(0)
        old_logp = gaussian_log_prob(old_mean, sample_logits, noise)
        v = critic(cond)
        adv = rewards - v.mean()
        adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-6)
    last = {}
    for _ in range(int(p["inner_epochs"])):
        mean = policy.forward_logits(cond).squeeze(0)
        logp = gaussian_log_prob(mean, sample_logits, noise)
        ratio = torch.exp(logp - old_logp)
        clip = float(p["cliprange"])
        surr = torch.min(ratio * adv, torch.clamp(ratio, 1 - clip, 1 + clip) * adv)
        value = critic(cond)
        vf = F.mse_loss(value.expand_as(rewards), rewards.detach())
        ent = -logp.mean()
        loss = -surr.mean() + float(p["vf_coef"]) * vf - float(p["entropy_coef"]) * ent
        optimizer.zero_grad()
        loss.backward()
        gn = nn.utils.clip_grad_norm_(list(policy.parameters()) + list(critic.parameters()), float(cfg["rl"]["grad_clip"]))
        optimizer.step()
        approx_kl = float((old_logp - logp).mean().detach())
        last = {
            "loss": float(loss.detach()),
            "approx_kl": approx_kl,
            "clip_frac": float(((ratio - 1.0).abs() > clip).float().mean().detach()),
            "grad_norm": float(gn),
            "adv_mean": float(adv.mean().detach()),
        }
        if approx_kl > float(p["kl_stop"]):
            last["kl_stop"] = True
            break
    return last
