from __future__ import annotations

import torch
import torch.nn as nn

from models.minute_encoder import PatchEmbed


class StandardTransformerEncoder(nn.Module):
    def __init__(self, feat_dim: int, d_model: int, patch: int, n_heads: int, n_layers: int, dropout: float) -> None:
        super().__init__()
        self.patch = PatchEmbed(feat_dim, d_model, patch)
        self.pe = nn.Parameter(torch.zeros(1, 160, d_model))
        enc = nn.TransformerEncoderLayer(d_model, n_heads, d_model * 2, dropout, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(enc, num_layers=n_layers)

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        h, pad = self.patch(x, mask)
        h = h + self.pe[:, : h.size(1)]
        h = self.enc(h, src_key_padding_mask=pad)
        return h[:, -1]
