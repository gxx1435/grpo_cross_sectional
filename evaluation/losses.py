"""SFT losses. Rank term is pairwise and must stay within one trading day."""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn.functional as F

_LN2 = math.log(2.0)


def pairwise_rank_loss(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Softplus pairwise ranking: prefer higher ŷ when y_i > y_j. Shapes [N]."""
    s = scores.reshape(-1)
    y = labels.reshape(-1)
    if s.numel() < 3:
        return s.new_tensor(0.0)
    score_diff = s.unsqueeze(-1) - s.unsqueeze(-2)
    label_diff = y.unsqueeze(-1) - y.unsqueeze(-2)
    sign = label_diff.sign()
    valid = sign != 0
    if not bool(valid.any()):
        return s.new_tensor(0.0)
    return F.softplus(-score_diff[valid] * sign[valid]).mean()


def cs_zscore(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Same-day cross-sectional z-score. Constant slice → zeros."""
    v = x.reshape(-1).float()
    if v.numel() < 2:
        return v
    sd = v.std(unbiased=False).clamp_min(eps)
    return (v - v.mean()) / sd


def sft_asof_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    asof_sizes: Sequence[int],
    rank_weight: float,
    cs_zscore_flag: bool = False,
) -> torch.Tensor:
    """Per-asof CS z-score, then MSE + λ × (rank / ln2). Both terms O(1)."""
    w = float(rank_weight)
    mse_parts = []
    rank_parts = []
    off = 0
    for n in asof_sizes:
        n = int(n)
        if n <= 0:
            continue
        p = pred[off : off + n]
        t = target[off : off + n]
        if cs_zscore_flag:
            p = cs_zscore(p)
            t = cs_zscore(t)
        mse_parts.append(F.mse_loss(p, t))
        if w > 0.0:
            rank_parts.append(pairwise_rank_loss(p, t))
        off += n
    if not mse_parts:
        return pred.reshape(-1).new_tensor(0.0)
    mse = torch.stack(mse_parts).mean()
    if w <= 0.0 or not rank_parts:
        return mse
    rank = torch.stack(rank_parts).mean() / mse.new_tensor(_LN2)
    return mse + w * rank
