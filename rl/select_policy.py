"""PPO that selects among frozen candidates (no new weight generation)."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CandidateSelectPolicy(nn.Module):
    """Score each candidate given market cond; pick via categorical / argmax.

    Candidates are discrete actions (SS-FM samples ± teachers). Output is an
    index into the candidate set — never a newly generated weight vector.
    """

    def __init__(self, cond_dim: int, n_assets: int, hidden: int) -> None:
        super().__init__()
        self.n_assets = int(n_assets)
        self.hidden = int(hidden)
        self.cond_enc = nn.Sequential(nn.Linear(int(cond_dim), hidden), nn.SiLU())
        self.cand_enc = nn.Sequential(nn.Linear(int(n_assets), hidden), nn.SiLU())
        self.score = nn.Sequential(
            nn.Linear(2 * hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        self.state_dim = hidden

    def encode_cond(self, cond: torch.Tensor) -> torch.Tensor:
        if cond.dim() == 1:
            cond = cond.unsqueeze(0)
        return self.cond_enc(cond).squeeze(0)

    def forward_scores(self, cond: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        """Return logits over candidates. candidates: (N, K)."""
        if candidates.dim() == 1:
            candidates = candidates.unsqueeze(0)
        h_c = self.encode_cond(cond)
        h_w = self.cand_enc(candidates)
        h = torch.cat([h_c.unsqueeze(0).expand(h_w.size(0), -1), h_w], dim=-1)
        return self.score(h).squeeze(-1)

    def forward_probs(self, cond: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.forward_scores(cond, candidates), dim=-1)

    @torch.no_grad()
    def select(self, cond: torch.Tensor, candidates: torch.Tensor) -> Tuple[int, torch.Tensor]:
        scores = self.forward_scores(cond, candidates)
        j = int(torch.argmax(scores).item())
        return j, candidates[j]


@torch.no_grad()
def sample_indices(policy: CandidateSelectPolicy, cond: torch.Tensor, candidates: torch.Tensor, n_samples: int):
    """Sample n discrete indices from Cat(scores). Returns idxs (n,), logp (n,)."""
    scores = policy.forward_scores(cond, candidates)
    dist = torch.distributions.Categorical(logits=scores)
    idxs = dist.sample((int(n_samples),))
    logp = dist.log_prob(idxs)
    return idxs, logp, scores


def ppo_update_select(
    policy: CandidateSelectPolicy,
    critic: nn.Module,
    cond: torch.Tensor,
    candidates: torch.Tensor,
    sample_idxs: torch.Tensor,
    rewards: torch.Tensor,
    optimizer,
    cfg: dict,
) -> Dict[str, float]:
    """Clipped PPO on categorical selection over a fixed candidate set."""
    p = cfg["rl"]["ppo"]
    with torch.no_grad():
        old_scores = policy.forward_scores(cond, candidates)
        old_dist = torch.distributions.Categorical(logits=old_scores)
        old_logp = old_dist.log_prob(sample_idxs)
        h = policy.encode_cond(cond)
        v = critic(h)
        adv = rewards - v.mean()
        adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-6)
    last: Dict[str, float] = {}
    for _ in range(int(p["inner_epochs"])):
        scores = policy.forward_scores(cond, candidates)
        dist = torch.distributions.Categorical(logits=scores)
        logp = dist.log_prob(sample_idxs)
        ratio = torch.exp(logp - old_logp)
        clip = float(p["cliprange"])
        surr = torch.min(ratio * adv, torch.clamp(ratio, 1 - clip, 1 + clip) * adv)
        h = policy.encode_cond(cond)
        value = critic(h)
        vf = F.mse_loss(value.expand_as(rewards), rewards.detach())
        ent = dist.entropy().mean()
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
            "entropy": float(ent.detach()),
        }
        if approx_kl > float(p["kl_stop"]):
            last["kl_stop"] = True  # type: ignore[assignment]
            break
    return last
