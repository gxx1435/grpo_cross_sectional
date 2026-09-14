"""Independent 9-d auction encoder. Never expands onto minute tokens."""

from __future__ import annotations

import torch
import torch.nn as nn


class AuctionEncoder(nn.Module):
    def __init__(self, auction_dim: int, d_model: int, dropout: float = 0.1) -> None:
        super().__init__()
        if int(auction_dim) != 9:
            raise RuntimeError(f"AuctionEncoder expects 9 fields, got {auction_dim}")
        self.net = nn.Sequential(
            nn.LayerNorm(auction_dim),
            nn.Linear(auction_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        return self.net(a)
