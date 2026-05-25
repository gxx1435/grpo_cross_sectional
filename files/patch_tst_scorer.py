"""
PatchTST-style cross-sectional stock scorer.

Input: (N, L, F) or (B, N, L, F)
Output: (N,) or (B, N) scores in [0, 1]

Pipeline per stock:
  1. Non-overlapping (or strided) patches along L
  2. Patch embedding + learnable positional encoding
  3. Transformer encoder over patch tokens
  4. Mean-pool patch tokens -> stock vector

Then one cross-sectional Transformer layer over N stocks (same as
``StockTemporalTransformer`` stage 2) and a sigmoid head.

Default hyper-params follow common PatchTST practice scaled to L=30:
  patch_len=6, stride=6 -> 5 patches; d_model=64; e_layers=2; heads=4.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class _LearnablePositionalEncoding(nn.Module):
    def __init__(self, max_len: int, d_model: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.pe = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)
        nn.init.normal_(self.pe.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pos = torch.arange(x.size(1), device=x.device)
        return self.dropout(x + self.pe(pos))


def _encoder_layer(d_model: int, num_heads: int, ffn_dim: int, dropout: float) -> nn.TransformerEncoderLayer:
    return nn.TransformerEncoderLayer(
        d_model=d_model,
        nhead=num_heads,
        dim_feedforward=ffn_dim,
        dropout=dropout,
        activation="gelu",
        batch_first=True,
        norm_first=True,
    )


class PatchTSTScorer(nn.Module):
    """PatchTST temporal encoder + cross-sectional attention."""

    def __init__(
        self,
        feat_dim: int,
        seq_len: int = 30,
        patch_len: int = 6,
        patch_stride: int = 6,
        d_model: int = 64,
        num_heads: int = 4,
        e_layers: int = 2,
        ffn_dim: int = 128,
        dropout: float = 0.1,
        cs_layers: int = 1,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")
        if patch_len > seq_len:
            raise ValueError("patch_len must be <= seq_len.")

        self.seq_len = seq_len
        self.patch_len = patch_len
        self.patch_stride = patch_stride
        self.n_patches = max(1, (seq_len - patch_len) // patch_stride + 1)

        patch_dim = patch_len * feat_dim
        self.patch_embed = nn.Sequential(
            nn.Linear(patch_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.patch_pe = _LearnablePositionalEncoding(self.n_patches, d_model, dropout)
        self.temporal_encoder = nn.TransformerEncoder(
            _encoder_layer(d_model, num_heads, ffn_dim, dropout),
            num_layers=e_layers,
        )
        self.temporal_norm = nn.LayerNorm(d_model)

        self.cs_encoder = nn.TransformerEncoder(
            _encoder_layer(d_model, num_heads, ffn_dim, dropout),
            num_layers=cs_layers,
        )

        self.output_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid(),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B*N, L, F) -> patches (B*N, P, patch_len*F)"""
        bn, L, F = x.shape
        if L < self.patch_len:
            pad = self.patch_len - L
            x = torch.nn.functional.pad(x, (0, 0, pad, 0))
            L = self.patch_len
        patches = x.unfold(dimension=1, size=self.patch_len, step=self.patch_stride)
        return patches.reshape(bn, patches.size(1), -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(0)
            single = True
        elif x.dim() == 4:
            single = False
        else:
            raise ValueError("x must have shape (N, L, F) or (B, N, L, F).")

        B, N, L, F = x.shape
        h = x.reshape(B * N, L, F)
        patches = self._patchify(h)
        h = self.patch_embed(patches)
        h = self.patch_pe(h)
        h = self.temporal_encoder(h)
        h = self.temporal_norm(h.mean(dim=1))
        h = h.reshape(B, N, -1)
        h = self.cs_encoder(h)
        scores = self.output_head(h).squeeze(-1)
        if single:
            scores = scores.squeeze(0)
        return scores


if __name__ == "__main__":
    N, L, F = 30, 30, 17
    model = PatchTSTScorer(feat_dim=F, seq_len=L)
    x = torch.randn(N, L, F)
    s = model(x)
    print(f"PatchTST: {tuple(x.shape)} -> {tuple(s.shape)}, params={sum(p.numel() for p in model.parameters()):,}")
