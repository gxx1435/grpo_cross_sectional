"""Temporal / patch Transformer over 2400 minute tokens."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    def __init__(self, feat_dim: int, d_model: int, patch: int) -> None:
        super().__init__()
        self.patch = int(patch)
        self.proj = nn.Linear(feat_dim * int(patch), d_model)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        squeeze = False
        if x.dim() == 3:
            x = x.unsqueeze(0)
            squeeze = True
            if mask is not None:
                mask = mask.unsqueeze(0)
        b, n, l, f = x.shape
        p = self.patch
        n_p = l // p
        x = x[:, :, : n_p * p].reshape(b, n, n_p, p * f)
        h = self.proj(x)
        pm = None
        if mask is not None:
            m = mask[:, :, : n_p * p].reshape(b, n, n_p, p).mean(dim=-1)
            pm = m < 0.25
        if squeeze:
            return h.squeeze(0), None if pm is None else pm.squeeze(0)
        return h, pm


class MinuteEncoder(nn.Module):
    def __init__(self, feat_dim: int, d_model: int, patch: int, n_heads: int, n_layers: int, dropout: float, max_patches: int = 160) -> None:
        super().__init__()
        self.patch = PatchEmbed(feat_dim, d_model, patch)
        self.pe = nn.Parameter(torch.zeros(1, max_patches, d_model))
        nn.init.normal_(self.pe, std=0.02)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, d_model * 2, dropout, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.last_attn: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        h, key_pad = self.patch(x, mask)
        p = h.size(-2)
        h = h + self.pe[:, :p]
        if h.dim() == 3:
            h = self.enc(h, src_key_padding_mask=key_pad)
            return h[:, -1]
        b, n, p, d = h.shape
        h = h.reshape(b * n, p, d)
        kp = None if key_pad is None else key_pad.reshape(b * n, p)
        h = self.enc(h, src_key_padding_mask=kp)
        return h[:, -1].reshape(b, n, d)
