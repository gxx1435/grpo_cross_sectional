"""PPO policy that conditions on frozen SS-FM candidates, then re-outputs weights."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from rl.ppo import gaussian_log_prob


class CandidateWeightPolicy(nn.Module):
    """Encode SS-FM candidates + market cond → fresh simplex logits.

    Candidates are features (per-candidate MLP + mean/max pool), not a discrete
    pick among G. Output is a new weight vector via softmax(logits).
    """

    def __init__(self, cond_dim: int, n_assets: int, hidden: int) -> None:
        super().__init__()
        self.n_assets = int(n_assets)
        self.hidden = int(hidden)
        self.cond_enc = nn.Sequential(nn.Linear(int(cond_dim), hidden), nn.SiLU())
        self.cand_enc = nn.Sequential(nn.Linear(int(n_assets), hidden), nn.SiLU())
        # cond + cand_mean + cand_max
        self.state_dim = 3 * hidden
        self.head = nn.Sequential(
            nn.Linear(self.state_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, int(n_assets)),
        )

    def encode(self, cond: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        if cond.dim() == 1:
            cond = cond.unsqueeze(0)
        if candidates.dim() == 1:
            candidates = candidates.unsqueeze(0)
        # candidates: (G, K)
        ce = self.cand_enc(candidates)
        c_mean = ce.mean(dim=0, keepdim=True)
        c_max = ce.max(dim=0, keepdim=True).values
        h = torch.cat([self.cond_enc(cond), c_mean, c_max], dim=-1)
        return h.squeeze(0)

    def forward_logits(self, state: torch.Tensor) -> torch.Tensor:
        if state.dim() == 1:
            state = state.unsqueeze(0)
        return self.head(state)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.forward_logits(state), dim=-1)


@torch.no_grad()
def sample_portfolios_from_state(policy: CandidateWeightPolicy, state: torch.Tensor, n_samples: int, noise_std: float):
    logits = policy.forward_logits(state).squeeze(0)
    eps = torch.randn(int(n_samples), logits.numel(), device=logits.device) * float(noise_std)
    logits_s = logits.unsqueeze(0) + eps
    return F.softmax(logits_s, dim=-1), logits_s, gaussian_log_prob(logits, logits_s, noise_std)


def ppo_update_candidate(policy, critic, cond, candidates, sample_logits, rewards, optimizer, cfg: dict):
    """Clipped PPO with re-encode each step so candidate encoders also train."""
    p = cfg["rl"]["ppo"]
    noise = float(cfg["rl"]["noise_std"])
    with torch.no_grad():
        old_state = policy.encode(cond, candidates)
        old_mean = policy.forward_logits(old_state).squeeze(0)
        old_logp = gaussian_log_prob(old_mean, sample_logits, noise)
        v = critic(old_state)
        adv = rewards - v.mean()
        adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-6)
    last = {}
    for _ in range(int(p["inner_epochs"])):
        state = policy.encode(cond, candidates)
        mean = policy.forward_logits(state).squeeze(0)
        logp = gaussian_log_prob(mean, sample_logits, noise)
        ratio = torch.exp(logp - old_logp)
        clip = float(p["cliprange"])
        surr = torch.min(ratio * adv, torch.clamp(ratio, 1 - clip, 1 + clip) * adv)
        value = critic(state)
        vf = F.mse_loss(value.expand_as(rewards), rewards.detach())
        ent = -logp.mean()
        loss = -surr.mean() + float(p["vf_coef"]) * vf - float(p["entropy_coef"]) * ent
        optimizer.zero_grad()
        loss.backward()
        gn = nn.utils.clip_grad_norm_(list(policy.parameters()) + list(critic.parameters()), 1.0)
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
