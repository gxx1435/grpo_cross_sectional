from __future__ import annotations

import torch
import torch.nn as nn

from models.minute_encoder import PatchEmbed


class _Block(nn.Module):
    def __init__(self, ch: int, dil: int, drop: float) -> None:
        super().__init__()
        k = 3
        self.conv = nn.Conv1d(ch, ch, k, padding=(k - 1) * dil, dilation=dil)
        self.drop = nn.Dropout(drop)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)[..., : x.size(-1)]
        return self.act(self.drop(y)) + x


class TCNEncoder(nn.Module):
    def __init__(self, feat_dim: int, d_model: int, patch: int) -> None:
        super().__init__()
        self.patch = PatchEmbed(feat_dim, d_model, patch)
        self.tcn = nn.Sequential(_Block(d_model, 1, 0.1), _Block(d_model, 2, 0.1), _Block(d_model, 4, 0.1))

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        h, _ = self.patch(x, mask)
        h = self.tcn(h.transpose(1, 2)).transpose(1, 2)
        return h[:, -1]
