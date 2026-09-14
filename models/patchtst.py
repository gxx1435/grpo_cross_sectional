"""PatchTST-style temporal encoder (Nie et al., 2023).

Channel-independent patches + shared Transformer. Output is one token per
stock so it can sit in the same AlphaPredictor stack as LSTM / TCN /
standard_transformer (CrossStockAttention + minute_only fusion).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class PatchTSTEncoder(nn.Module):
    def __init__(
        self,
        feat_dim: int,
        d_model: int,
        patch: int,
        n_heads: int,
        n_layers: int,
        dropout: float,
        max_patches: int = 160,
    ) -> None:
        super().__init__()
        self.patch = int(patch)
        self.feat_dim = int(feat_dim)
        self.proj = nn.Linear(self.patch, d_model)
        self.pe = nn.Parameter(torch.zeros(1, max_patches, d_model))
        nn.init.normal_(self.pe, std=0.02)
        enc = nn.TransformerEncoderLayer(
            d_model, n_heads, d_model * 2, dropout, batch_first=True, norm_first=True
        )
        self.enc = nn.TransformerEncoder(enc, num_layers=n_layers)
        self.channel_mix = nn.Linear(d_model, d_model)
        # Channel-independence expands N → N*C sequences; keep Transformer calls small.
        self.stock_microbatch = 8

    def _encode_chunk(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        n, t, c = x.shape
        p = self.patch
        n_p = t // p
        x = x[:, : n_p * p]
        mu = x.mean(dim=1, keepdim=True)
        sd = x.std(dim=1, keepdim=True).clamp_min(1e-5)
        x = (x - mu) / sd
        x = x.reshape(n, n_p, p, c).permute(0, 3, 1, 2).reshape(n * c, n_p, p)
        h = self.proj(x) + self.pe[:, :n_p]
        key_pad = None
        if mask is not None:
            m = mask[:, : n_p * p].reshape(n, n_p, p).mean(dim=-1)
            key_pad = (m < 0.25).unsqueeze(1).expand(n, c, n_p).reshape(n * c, n_p)
        h = self.enc(h, src_key_padding_mask=key_pad)
        tok = h[:, -1].reshape(n, c, -1).mean(dim=1)
        return self.channel_mix(tok)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if x.dim() != 3:
            raise RuntimeError("PatchTSTEncoder expected X [N, T, C]")
        n, t, _ = x.shape
        p = self.patch
        if t // p <= 0:
            raise RuntimeError(f"sequence {t} shorter than patch {p}")
        step = max(1, int(self.stock_microbatch))
        if n <= step:
            return self._encode_chunk(x, mask)
        toks = []
        for i0 in range(0, n, step):
            sl = slice(i0, i0 + step)
            mc = None if mask is None else mask[sl]
            toks.append(self._encode_chunk(x[sl], mc))
        return torch.cat(toks, dim=0)
