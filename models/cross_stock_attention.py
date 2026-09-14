"""Cross-stock attention on the same-day valid universe only."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class CrossStockAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_layers: int, dropout: float) -> None:
        super().__init__()
        layer = nn.TransformerEncoderLayer(d_model, n_heads, d_model * 2, dropout, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(self, h: torch.Tensor, valid: Optional[torch.Tensor] = None) -> torch.Tensor:
        if h.dim() == 2:
            h = h.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False
        pad = None
        if valid is not None:
            pad = ~valid.bool()
            if pad.dim() == 1:
                pad = pad.unsqueeze(0)
        out = self.enc(h, src_key_padding_mask=pad)
        return out.squeeze(0) if squeeze else out
