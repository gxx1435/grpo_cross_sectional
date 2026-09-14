"""Full alpha stack and baseline predictors. Output is ŷ = intraday log return only."""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from models.auction_encoder import AuctionEncoder
from models.cross_stock_attention import CrossStockAttention
from models.fusion import build_fusion
from models.lstm import LSTMEncoder
from models.minute_encoder import MinuteEncoder
from models.patchtst import PatchTSTEncoder
from models.tcn import TCNEncoder
from models.transformer import StandardTransformerEncoder


class AlphaPredictor(nn.Module):
    def __init__(self, cfg: dict, feat_dim: int, auction_dim: int = 9, encoder: str = "minute") -> None:
        super().__init__()
        p = cfg["prediction"]
        d = int(p["d_model"])
        sc = int(p["stock_chunk"])
        self.stock_chunk = sc if sc > 0 else 10**9
        self.use_ckpt = bool(cfg["gpu"]["gradient_checkpointing"])
        self.fusion_name = encoder
        if encoder == "lstm":
            self.minute = LSTMEncoder(feat_dim, d, int(p["patch"]))
            self.need_auction = False
            self.fusion = build_fusion("minute_only", d, int(p["n_heads"]))
        elif encoder == "tcn":
            self.minute = TCNEncoder(feat_dim, d, int(p["patch"]))
            self.need_auction = False
            self.fusion = build_fusion("minute_only", d, int(p["n_heads"]))
        elif encoder in ("standard_transformer", "minute_only"):
            self.minute = StandardTransformerEncoder(feat_dim, d, int(p["patch"]), int(p["n_heads"]), int(p["n_layers_temporal"]), float(p["dropout"]))
            self.need_auction = False
            self.fusion = build_fusion("minute_only", d, int(p["n_heads"]))
        elif encoder == "patchtst":
            self.minute = PatchTSTEncoder(feat_dim, d, int(p["patch"]), int(p["n_heads"]), int(p["n_layers_temporal"]), float(p["dropout"]))
            self.need_auction = False
            self.fusion = build_fusion("minute_only", d, int(p["n_heads"]))
            self.stock_chunk = 16 if sc <= 0 else min(int(sc), 16)
            self.use_ckpt = True
        else:
            self.minute = MinuteEncoder(feat_dim, d, int(p["patch"]), int(p["n_heads"]), int(p["n_layers_temporal"]), float(p["dropout"]))
            self.need_auction = encoder != "minute_only"
            self.fusion = build_fusion(encoder if encoder != "minute_only" else "minute_only", d, int(p["n_heads"]))
        self.cross = CrossStockAttention(d, int(p["n_heads"]), int(p["n_layers_cross"]), float(p["dropout"]))
        self.auction = AuctionEncoder(auction_dim, d, float(p["dropout"]))
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d // 2), nn.GELU(), nn.Linear(d // 2, 1))
        self.last_aux: Dict[str, torch.Tensor] = {}

    def _encode_minute(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        return self.minute(x, mask) if "mask" in self.minute.forward.__code__.co_varnames else self.minute(x)

    def forward(
        self,
        x: torch.Tensor,
        auction: Optional[torch.Tensor] = None,
        valid: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if x.dim() != 3:
            raise RuntimeError("expected X [N, 2400, F]")
        n = x.size(0)
        toks = []
        for i0 in range(0, n, self.stock_chunk):
            xc = x[i0 : i0 + self.stock_chunk]
            mc = None if mask is None else mask[i0 : i0 + self.stock_chunk]
            if self.use_ckpt and self.training:
                from torch.utils.checkpoint import checkpoint
                tok = checkpoint(self._encode_minute, xc, mc, use_reentrant=False)
            else:
                tok = self._encode_minute(xc, mc)
            toks.append(tok)
        h_hist = torch.cat(toks, dim=0)
        h_hist = self.cross(h_hist, valid)
        if self.need_auction:
            if auction is None:
                raise RuntimeError("auction tensor required")
            h_auc = self.auction(auction)
        else:
            h_auc = torch.zeros_like(h_hist)
        h, aux = self.fusion(h_hist, h_auc)
        self.last_aux = aux
        return self.head(h).squeeze(-1)


def build_predictor(name: str, cfg: dict, feat_dim: int) -> AlphaPredictor:
    return AlphaPredictor(cfg, feat_dim, encoder=name)
