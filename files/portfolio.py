"""
Long-short portfolio construction from model scores.

Replaces the old Dirichlet-mean mapping that saturated at equal-ish
extreme weights.  Policy scores are ranked; top-K receive equal long
weight, bottom-K equal short weight, middle names get zero.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import torch


@dataclass
class LongShortConfig:
    top_k: int = 5
    bottom_k: int = 5
    long_gross: float = 0.5      # total long leg (sum of positive weights)
    short_gross: float = 0.5     # total short leg (sum of |negative weights|)
    score_noise_std: float = 0.05  # Gaussian noise on scores during GRPO rollouts


def scores_to_long_short_weights(
    scores: torch.Tensor,
    cfg: LongShortConfig = LongShortConfig(),
) -> torch.Tensor:
    """
    Map scores ``(..., N)`` to dollar-neutral long-short weights ``(..., N)``.

    Top ``top_k`` stocks:  ``+long_gross / top_k``
    Bottom ``bottom_k``: ``-short_gross / bottom_k``
    Others: 0
    """
    s = scores.clone()
    *batch, n = s.shape
    k_long = min(cfg.top_k, n)
    k_short = min(cfg.bottom_k, max(n - k_long, 0))

    order = s.argsort(dim=-1, descending=True)
    w = torch.zeros_like(s)

    long_w = cfg.long_gross / max(k_long, 1)
    short_w = cfg.short_gross / max(k_short, 1)

    # scatter long weights
    top_idx = order[..., :k_long]
    w.scatter_(-1, top_idx, long_w)

    if k_short > 0:
        bot_idx = order[..., -k_short:]
        w.scatter_(-1, bot_idx, -short_w)

    return w


def scores_to_long_only_weights(
    scores: torch.Tensor,
    cfg: LongShortConfig | None = None,
    *,
    top_k: int = 5,
    long_gross: float = 1.0,
) -> torch.Tensor:
    """
    Long-only Top-K: top ``top_k`` scores get ``+long_gross / top_k``, others 0.
    Default ``long_gross=1.0`` → 100% invested in Top-K.
    """
    ls = cfg or LongShortConfig(
        top_k=top_k, bottom_k=0, long_gross=long_gross, short_gross=0.0,
    )
    ls = LongShortConfig(
        top_k=ls.top_k,
        bottom_k=0,
        long_gross=ls.long_gross,
        short_gross=0.0,
        score_noise_std=ls.score_noise_std,
    )
    return scores_to_long_short_weights(scores, ls)


def scores_to_portfolio_weights(
    scores: torch.Tensor,
    cfg: LongShortConfig,
    portfolio_mode: str = "long_short",
) -> torch.Tensor:
    """Dispatch long-short vs long-only portfolio construction."""
    if portfolio_mode == "long_only":
        return scores_to_long_only_weights(scores, cfg)
    return scores_to_long_short_weights(scores, cfg)


def long_short_equal_weight_vector(
    n_stocks: int,
    cfg: LongShortConfig = LongShortConfig(),
) -> np.ndarray:
    """
    Fixed dollar-neutral LS equal-weight baseline (no model signal).

    Long the first ``top_k`` names (+long_gross/top_k each), short the last
    ``bottom_k`` names (-short_gross/bottom_k each).  Matches GRPO gross
    structure but with deterministic index ordering in the universe list.
    """
    n = int(n_stocks)
    w = np.zeros(n, dtype=np.float32)
    k_long = min(cfg.top_k, n)
    k_short = min(cfg.bottom_k, max(n - k_long, 0))
    if k_long > 0:
        w[:k_long] = cfg.long_gross / k_long
    if k_short > 0:
        w[-k_short:] = -cfg.short_gross / k_short
    return w


def sample_noisy_scores(
    scores: torch.Tensor,
    cfg: LongShortConfig,
    num_generations: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample ``G`` perturbed score tensors for GRPO rollouts.

    Returns:
        noisy_scores : (G, *batch, N)
        log_prob     : (G, *batch)  factorised Gaussian log-density
    """
    *batch, n = scores.shape
    std = max(cfg.score_noise_std, 1e-4)
    eps = torch.randn((num_generations, *batch, n), device=scores.device, dtype=scores.dtype)
    base = scores.unsqueeze(0).expand(num_generations, *([-1] * (1 + len(batch))))
    noisy = base + std * eps
    log_prob = (-0.5 * (eps ** 2) - torch.log(torch.tensor(std)) - 0.5 * torch.log(torch.tensor(2 * 3.14159265))).sum(dim=-1)
    return noisy, log_prob


def weights_to_long_short(
    weights: torch.Tensor,
    cfg: LongShortConfig = LongShortConfig(),
) -> torch.Tensor:
    """Re-rank arbitrary weights into long-short form (for legacy Dirichlet paths)."""
    return scores_to_long_short_weights(weights, cfg)
