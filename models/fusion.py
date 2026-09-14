"""Fusion ablation: minute-only, concat, unidirectional cross-attn, gated residual."""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MinuteOnlyFusion(nn.Module):
    def forward(self, h_hist: torch.Tensor, h_auc: torch.Tensor) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        return h_hist, {}


class ConcatFusion(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.GELU(), nn.Linear(d_model, d_model))

    def forward(self, h_hist: torch.Tensor, h_auc: torch.Tensor) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        return self.mlp(torch.cat([h_hist, h_auc], dim=-1)), {}


class CrossAttentionFusion(nn.Module):
    """Unidirectional: Q=H_hist, K=V=H_auction."""

    def __init__(self, d_model: int, n_heads: int = 4) -> None:
        super().__init__()
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.n_heads = int(n_heads)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, h_hist: torch.Tensor, h_auc: torch.Tensor) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        q, k, v = self.q(h_hist), self.k(h_auc), self.v(h_auc)
        heads = self.n_heads
        d = q.size(-1) // heads
        lead = q.shape[:-1]
        qs = q.view(*lead, heads, d)
        ks = k.view(*lead, heads, d)
        vs = v.view(*lead, heads, d)
        attn = torch.softmax((qs * ks).sum(-1, keepdim=True) / (d ** 0.5), dim=-1)
        ctx = (attn * vs).reshape_as(h_hist)
        return h_hist + self.out(ctx), {"cross_attn": attn.detach()}


class GatedResidualFusion(nn.Module):
    """H_final = H_hist + g ⊙ H_auction. Auction is incremental pre-market info."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, h_hist: torch.Tensor, h_auc: torch.Tensor) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        g = torch.sigmoid(self.gate(torch.cat([h_hist, h_auc], dim=-1)))
        h_auc_p = self.proj(h_auc)
        return h_hist + g * h_auc_p, {"gate": g.detach()}


def build_fusion(name: str, d_model: int, n_heads: int) -> nn.Module:
    name = name.lower()
    if name in ("minute_only", "minute-only"):
        return MinuteOnlyFusion()
    if name == "concat":
        return ConcatFusion(d_model)
    if name in ("cross_attention", "cross-attention"):
        return CrossAttentionFusion(d_model, n_heads)
    if name in ("gated_residual", "ours"):
        return GatedResidualFusion(d_model)
    raise ValueError(name)
