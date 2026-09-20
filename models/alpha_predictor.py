"""The only enabled alpha model: minute Transformer plus cross-stock attention."""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn

from models.cross_stock_attention import CrossStockAttention
from models.minute_encoder import MinuteEncoder

STRATEGY_NAME = "minute_transformer_cross_stock_attention"


class AlphaPredictor(nn.Module):
    def __init__(self, cfg: dict, feat_dim: int) -> None:
        super().__init__()
        p = cfg["prediction"]
        if str(p["strategy_name"]) != STRATEGY_NAME:
            raise RuntimeError(f"unsupported alpha strategy {p['strategy_name']}")
        d_model = int(p["d_model"])
        patch = int(p["patch"])
        max_tokens = int(cfg["calendar"]["lookback_days"]) * int(cfg["calendar"]["bars_per_day"])
        max_patches = int(math.ceil(max_tokens / patch)) + 1
        self.strategy_name = STRATEGY_NAME
        self.stock_chunk = max(1, int(p["stock_chunk"]))
        self.use_ckpt = bool(cfg["gpu"]["gradient_checkpointing"])
        self.minute = MinuteEncoder(
            feat_dim,
            d_model,
            patch,
            int(p["n_heads"]),
            int(p["n_layers_temporal"]),
            float(p["dropout"]),
            max_patches,
        )
        self.cross = CrossStockAttention(d_model, int(p["n_heads"]), int(p["n_layers_cross"]), float(p["dropout"]))
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, 1))
        self.last_aux: Dict[str, torch.Tensor] = {}

    def _encode_minute(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        return self.minute(x, mask)

    def forward(self, x: torch.Tensor, valid: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if x.dim() != 3:
            raise RuntimeError("expected X [stocks, historical_minutes, features]")
        encoded = []
        temporal: Dict[str, list[torch.Tensor]] = {}
        for start in range(0, x.size(0), self.stock_chunk):
            xb = x[start : start + self.stock_chunk]
            mb = None if mask is None else mask[start : start + self.stock_chunk]
            if self.use_ckpt and self.training:
                from torch.utils.checkpoint import checkpoint

                token = checkpoint(self._encode_minute, xb, mb, use_reentrant=False)
            else:
                token = self._encode_minute(xb, mb)
            encoded.append(token)
            for key, value in self.minute.last_attention_summary.items():
                temporal.setdefault(key, []).append(value)
        history = torch.cat(encoded, dim=0)
        cross = self.cross(history, valid)
        aux: Dict[str, torch.Tensor] = {}
        for key, values in temporal.items():
            aux[key] = torch.stack(values).mean(dim=0)
        aux.update(self.cross.last_attention_summary)
        self.last_aux = aux
        return self.head(cross).squeeze(-1)


def build_predictor(name: str, cfg: dict, feat_dim: int) -> AlphaPredictor:
    if str(name) != STRATEGY_NAME:
        raise RuntimeError(f"only {STRATEGY_NAME} is enabled; requested {name}")
    return AlphaPredictor(cfg, feat_dim)
