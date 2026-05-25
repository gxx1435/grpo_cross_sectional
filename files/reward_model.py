"""
Multi-objective episode reward for GRPO with auto-learned component weights.

Components (all per rollout, episode-level scalar unless noted)
---------------------------------------------------------------
* ic          : mean per-bar Pearson corr(weights, returns)
* sharpe      : Sharpe of portfolio returns over M bars
* turnover    : mean L1 turnover between consecutive weight vectors (penalty)
* downside    : mean squared negative portfolio returns (penalty)
* max_dd      : max drawdown of cumulative portfolio returns (penalty)
* size_exp    : |corr(weights, size_z)|  — size tilt (penalty)
* beta_exp    : |corr(weights, beta_z)| — market-beta tilt (penalty)
* hhi         : Herfindahl of |weights| — concentration (penalty)

Total reward = sum_k  lambda_k * component_k   (turnover/downside/dd/exp/hhi are
already signed so their lambdas are typically positive magnitudes applied to
negative quantities).

Auto-learning
-------------
``search_reward_weights`` grid-searches small candidate sets on a validation
episode batch and picks the combo with highest mean composite score
(IC + Sharpe - penalties).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


@dataclass
class RewardConfig:
    top_k: int = 5
    bottom_k: int = 5
    # component weights (positive magnitudes; penalties subtracted in reward)
    ic_weight: float = 1.0
    sharpe_weight: float = 0.8
    turnover_weight: float = 0.3
    downside_weight: float = 0.4
    max_dd_weight: float = 0.5
    size_exp_weight: float = 0.2
    beta_exp_weight: float = 0.2
    hhi_weight: float = 0.1

    # grid candidates for auto-learn
    ic_candidates: Tuple[float, ...] = (0.5, 1.0, 1.5)
    sharpe_candidates: Tuple[float, ...] = (0.5, 0.8, 1.2)
    penalty_candidates: Tuple[float, ...] = (0.2, 0.4, 0.6)


def empirical_reward_config(top_k: int = 5, bottom_k: int = 5) -> RewardConfig:
    """Fixed empirical weights (no validation grid-search)."""
    return RewardConfig(
        top_k=top_k,
        bottom_k=bottom_k,
        ic_weight=1.0,
        sharpe_weight=0.8,
        turnover_weight=0.3,
        downside_weight=0.4,
        max_dd_weight=0.5,
        size_exp_weight=0.2,
        beta_exp_weight=0.2,
        hhi_weight=0.1,
    )


def _pearson_rows(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Pearson along last dim; a,b shape (..., N) -> (...)"""
    ac = a - a.mean(dim=-1, keepdim=True)
    bc = b - b.mean(dim=-1, keepdim=True)
    num = (ac * bc).sum(dim=-1)
    den = ac.norm(dim=-1) * bc.norm(dim=-1) + 1e-8
    return num / den


def episode_reward_v2(
    weights: torch.Tensor,
    realized_returns: torch.Tensor,
    size_z: Optional[torch.Tensor] = None,
    beta_z: Optional[torch.Tensor] = None,
    cfg: RewardConfig = RewardConfig(),
) -> Dict[str, torch.Tensor]:
    """
    Args:
        weights:          (G, M, N)  long-short weights (need not sum to 1)
        realized_returns: (M, N)
        size_z:           (M, N) optional cross-sectional size z-score
        beta_z:           (M, N) optional rolling beta z-score
    """
    G, M, N = weights.shape
    r = realized_returns.unsqueeze(0)  # (1, M, N)

    ic_per = _pearson_rows(weights, r)          # (G, M)
    ic = ic_per.mean(dim=1)                     # (G,)

    port_pr = (weights * r).sum(dim=-1)         # (G, M)
    sharpe = port_pr.mean(dim=1) / (port_pr.std(dim=1) + 1e-8)

    # turnover: L1 change in weights bar-to-bar
    if M > 1:
        turnover = (weights[:, 1:] - weights[:, :-1]).abs().sum(dim=-1).mean(dim=1)
    else:
        turnover = torch.zeros(G, device=weights.device, dtype=weights.dtype)

    downside = torch.clamp(port_pr, max=0.0).pow(2).mean(dim=1)

    cum = port_pr.cumsum(dim=1)
    running_max = cum.cummax(dim=1).values
    max_dd = (running_max - cum).max(dim=1).values

    # exposure penalties (mean abs corr across bars)
    if size_z is not None:
        sz = size_z.unsqueeze(0)
        size_exp = _pearson_rows(weights, sz).abs().mean(dim=1)
    else:
        size_exp = torch.zeros(G, device=weights.device, dtype=weights.dtype)

    if beta_z is not None:
        bz = beta_z.unsqueeze(0)
        beta_exp = _pearson_rows(weights, bz).abs().mean(dim=1)
    else:
        beta_exp = torch.zeros(G, device=weights.device, dtype=weights.dtype)

    abs_w = weights.abs()
    hhi = (abs_w / (abs_w.sum(dim=-1, keepdim=True) + 1e-8)).pow(2).sum(dim=-1).mean(dim=1)

    reward = (
        cfg.ic_weight * ic
        + cfg.sharpe_weight * sharpe
        - cfg.turnover_weight * turnover
        - cfg.downside_weight * downside
        - cfg.max_dd_weight * max_dd
        - cfg.size_exp_weight * size_exp
        - cfg.beta_exp_weight * beta_exp
        - cfg.hhi_weight * hhi
    )

    return {
        "reward": reward,
        "ic": ic,
        "sharpe": sharpe,
        "turnover": turnover,
        "downside": downside,
        "max_dd": max_dd,
        "size_exp": size_exp,
        "beta_exp": beta_exp,
        "hhi": hhi,
        "portfolio_pr": port_pr,
    }


def _val_score(metrics: Dict[str, torch.Tensor], cfg: RewardConfig) -> float:
    """Scalar validation objective (same formula as reward)."""
    return float(
        cfg.ic_weight * metrics["ic"].mean()
        + cfg.sharpe_weight * metrics["sharpe"].mean()
        - cfg.turnover_weight * metrics["turnover"].mean()
        - cfg.downside_weight * metrics["downside"].mean()
        - cfg.max_dd_weight * metrics["max_dd"].mean()
        - cfg.size_exp_weight * metrics["size_exp"].mean()
        - cfg.beta_exp_weight * metrics["beta_exp"].mean()
        - cfg.hhi_weight * metrics["hhi"].mean()
    )


def search_reward_weights(
    val_weights: torch.Tensor,
    val_returns: torch.Tensor,
    val_size_z: Optional[torch.Tensor] = None,
    val_beta_z: Optional[torch.Tensor] = None,
    base_cfg: RewardConfig = RewardConfig(),
    max_trials: int = 40,
) -> RewardConfig:
    """
    Grid-search reward component weights on a validation batch.

    ``val_weights`` : (E, M, N)  E validation episodes
    ``val_returns`` : (E, M, N)  matching forward returns
    """
    E = val_weights.shape[0]
    candidates = list(itertools.product(
        base_cfg.ic_candidates,
        base_cfg.sharpe_candidates,
        base_cfg.penalty_candidates,
    ))
    if len(candidates) > max_trials:
        step = max(1, len(candidates) // max_trials)
        candidates = candidates[::step][:max_trials]

    best_cfg = base_cfg
    best_score = -1e18

    for ic_w, sh_w, pen_w in candidates:
        trial = RewardConfig(
            top_k=base_cfg.top_k,
            bottom_k=base_cfg.bottom_k,
            ic_weight=ic_w,
            sharpe_weight=sh_w,
            turnover_weight=pen_w,
            downside_weight=pen_w,
            max_dd_weight=pen_w,
            size_exp_weight=pen_w * 0.5,
            beta_exp_weight=pen_w * 0.5,
            hhi_weight=pen_w * 0.25,
        )
        scores = []
        for e in range(E):
            sz_e = val_size_z[e] if val_size_z is not None else None
            bz_e = val_beta_z[e] if val_beta_z is not None else None
            m = episode_reward_v2(
                val_weights[e: e + 1],
                val_returns[e],
                sz_e,
                bz_e,
                trial,
            )
            scores.append(_val_score(m, trial))
        score = float(np.mean(scores)) if scores else -1e18
        if score > best_score:
            best_score = score
            best_cfg = trial

    return best_cfg
