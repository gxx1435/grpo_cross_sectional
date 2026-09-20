"""Masked cross-stock self-attention with auditable attention statistics."""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn


class CrossStockBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model * 2, d_model))

    def forward(self, x: torch.Tensor, pad: Optional[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        q = self.norm1(x)
        a, weights = self.attn(q, q, q, key_padding_mask=pad, need_weights=True, average_attn_weights=False)
        x = x + a
        x = x + self.ff(self.norm2(x))
        return x, weights


class CrossStockAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_layers: int, dropout: float) -> None:
        super().__init__()
        self.layers = nn.ModuleList([CrossStockBlock(d_model, n_heads, dropout) for _ in range(int(n_layers))])
        self.last_attention_summary: Dict[str, torch.Tensor] = {}

    def forward(self, h: torch.Tensor, valid: Optional[torch.Tensor] = None) -> torch.Tensor:
        squeeze = h.dim() == 2
        x = h.unsqueeze(0) if squeeze else h
        pad = None
        if valid is not None:
            value = valid.bool()
            pad = ~value.unsqueeze(0) if value.dim() == 1 else ~value
        last = None
        for layer in self.layers:
            x, last = layer(x, pad)
        if last is not None:
            # Average query stocks and heads; distribution remains over source stocks.
            prob = last.mean(dim=(0, 1, 2))
            prob = prob / prob.sum().clamp_min(1e-12)
            entropy = -(prob.clamp_min(1e-12) * prob.clamp_min(1e-12).log()).sum()
            self.last_attention_summary = {
                "cross_stock_attention_entropy": entropy.detach(),
                "cross_stock_attention_concentration": prob.max().detach(),
                "cross_stock_mask_rate": (pad.float().mean() if pad is not None else torch.zeros((), device=x.device)).detach(),
            }
        return x.squeeze(0) if squeeze else x
