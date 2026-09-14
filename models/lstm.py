from __future__ import annotations

import torch
import torch.nn as nn

from models.minute_encoder import PatchEmbed


class LSTMEncoder(nn.Module):
    def __init__(self, feat_dim: int, d_model: int, patch: int) -> None:
        super().__init__()
        self.patch = PatchEmbed(feat_dim, d_model, patch)
        self.lstm = nn.LSTM(d_model, d_model, num_layers=1, batch_first=True)

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        h, _ = self.patch(x, mask)
        out, _ = self.lstm(h)
        return out[:, -1]
