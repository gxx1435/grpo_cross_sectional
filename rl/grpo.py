from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from rl.diagnostics import group_stats
from rl.ppo import gaussian_log_prob


def grpo_update(policy, cond, sample_logits, rewards, optimizer, cfg: dict) -> Dict[str, float]:
    noise = float(cfg["rl"]["noise_std"])
    adv = (rewards - rewards.mean()) / (rewards.std(unbiased=False) + 1e-6)
    mean = policy.forward_logits(cond).squeeze(0)
    logp = gaussian_log_prob(mean, sample_logits, noise)
    loss = -(adv.detach() * logp).mean()
    optimizer.zero_grad()
    loss.backward()
    gn = nn.utils.clip_grad_norm_(policy.parameters(), float(cfg["rl"]["grad_clip"]))
    optimizer.step()
    stats = group_stats(rewards.detach().cpu().numpy(), torch.softmax(sample_logits, dim=-1).detach().cpu().numpy(), cfg)
    out = {"loss": float(loss.detach()), "grad_norm": float(gn), **{k: stats[k] for k in stats if k != "warning"}}
    if stats.get("warning"):
        out["warning"] = stats["warning"]
    return out
