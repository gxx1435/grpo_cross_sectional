"""Patch-based temporal Transformer with learned causal-history pooling."""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbed(nn.Module):
    def __init__(self, feat_dim: int, d_model: int, patch: int) -> None:
        super().__init__()
        self.patch_size = int(patch)
        self.proj = nn.Linear(feat_dim * self.patch_size, d_model)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.dim() != 3:
            raise RuntimeError("MinuteEncoder expects [stocks, minutes, features]")
        n, length, feat = x.shape
        patch = self.patch_size
        padded = int(math.ceil(length / patch) * patch)
        if padded != length:
            x = F.pad(x, (0, 0, 0, padded - length))
            if mask is not None:
                mask = F.pad(mask, (0, padded - length))
        if mask is None:
            mask = torch.ones((n, padded), device=x.device, dtype=x.dtype)
        h = self.proj(x.reshape(n, padded // patch, patch * feat))
        patch_valid = mask.reshape(n, padded // patch, patch).sum(dim=-1) > 0
        return h, patch_valid


class MinuteEncoder(nn.Module):
    def __init__(self, feat_dim: int, d_model: int, patch: int, n_heads: int, n_layers: int, dropout: float, max_patches: int) -> None:
        super().__init__()
        self.patch = PatchEmbed(feat_dim, d_model, patch)
        self.position = nn.Parameter(torch.zeros(1, int(max_patches), d_model))
        nn.init.normal_(self.position, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model,
            n_heads,
            d_model * 2,
            dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=int(n_layers))
        self.pool_query = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.pool_query, std=0.02)
        self.pool = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.last_attention_summary: Dict[str, torch.Tensor] = {}

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        h, patch_valid = self.patch(x, mask)
        patches = h.size(1)
        if patches > self.position.size(1):
            raise RuntimeError(f"{patches} patches exceed positional capacity {self.position.size(1)}")
        h = h + self.position[:, :patches]
        pad = ~patch_valid
        all_pad = pad.all(dim=1)
        if bool(all_pad.any()):
            pad = pad.clone()
            pad[all_pad, 0] = False
        h = self.encoder(h, src_key_padding_mask=pad)
        query = self.pool_query.expand(h.size(0), -1, -1)
        pooled, weight = self.pool(query, h, h, key_padding_mask=pad, need_weights=True, average_attn_weights=False)
        # [N, heads, 1, patches] -> one auditable distribution over history.
        mean_weight = weight.mean(dim=(0, 1, 2))
        prob = mean_weight / mean_weight.sum().clamp_min(1e-12)
        entropy = -(prob.clamp_min(1e-12) * prob.clamp_min(1e-12).log()).sum()
        self.last_attention_summary = {
            "temporal_patch_attention": prob.detach(),
            "temporal_attention_entropy": entropy.detach(),
            "temporal_attention_concentration": prob.max().detach(),
            "temporal_mask_rate": pad.float().mean().detach(),
        }
        return self.norm(pooled.squeeze(1))
