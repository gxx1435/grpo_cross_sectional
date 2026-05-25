"""
Two-stage single-layer Transformer for cross-sectional stock scoring.

Input shape:
    (N, L, F)   single cross section, N stocks, L lookback bars, F features.
    (B, N, L, F) batched.

Output:
    (N,)  or  (B, N)  scores in [0, 1].

Stage 1 (temporal, 1 layer): per-stock self-attention along L, with a
    learnable positional encoding. Pool to one vector per stock by taking
    the last token's representation (the "now" bar).

Stage 2 (cross section, 1 layer): the N stocks attend to each other.

This is the model used by both SFT and GRPO post-training.
"""

import torch
import torch.nn as nn


class _LearnablePositionalEncoding(nn.Module):
    def __init__(self, max_len: int, d_model: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.pe = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)
        nn.init.normal_(self.pe.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        L = x.size(1)
        pos = torch.arange(L, device=x.device)
        return self.dropout(x + self.pe(pos))


def _make_encoder_layer(d_model: int, num_heads: int, ffn_dim: int, dropout: float) -> nn.TransformerEncoderLayer:
    return nn.TransformerEncoderLayer(
        d_model=d_model,
        nhead=num_heads,
        dim_feedforward=ffn_dim,
        dropout=dropout,
        activation="gelu",
        batch_first=True,
        norm_first=True,
    )


class StockTemporalTransformer(nn.Module):
    """One temporal layer + one cross-sectional layer."""

    def __init__(
        self,
        feat_dim: int,
        max_lookback: int = 64,
        d_model: int = 64,
        num_heads: int = 4,
        ffn_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")

        self.input_proj = nn.Sequential(
            nn.Linear(feat_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.temporal_pe = _LearnablePositionalEncoding(max_lookback, d_model, dropout)
        self.temporal_encoder = nn.TransformerEncoder(
            _make_encoder_layer(d_model, num_heads, ffn_dim, dropout),
            num_layers=1,
        )
        self.temporal_norm = nn.LayerNorm(d_model)

        self.cs_encoder = nn.TransformerEncoder(
            _make_encoder_layer(d_model, num_heads, ffn_dim, dropout),
            num_layers=1,
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
        h = self.input_proj(h)
        h = self.temporal_pe(h)
        h = self.temporal_encoder(h)
        h = self.temporal_norm(h[:, -1, :])
        h = h.reshape(B, N, -1)

        h = self.cs_encoder(h)
        scores = self.output_head(h).squeeze(-1)

        if single:
            scores = scores.squeeze(0)
        return scores


if __name__ == "__main__":
    N, L, F = 32, 30, 11
    model = StockTemporalTransformer(feat_dim=F, max_lookback=L)

    x_single = torch.randn(N, L, F)
    s_single = model(x_single)
    print(f"single  : input {tuple(x_single.shape)} -> scores {tuple(s_single.shape)}, range [{s_single.min():.4f}, {s_single.max():.4f}]")

    x_batch = torch.randn(4, N, L, F)
    s_batch = model(x_batch)
    print(f"batched : input {tuple(x_batch.shape)} -> scores {tuple(s_batch.shape)}")

    print(f"params  : {sum(p.numel() for p in model.parameters()):,}")
