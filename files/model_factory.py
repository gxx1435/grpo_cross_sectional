"""Factory for cross-sectional scorer backbones (Transformer / PatchTST)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Union

import torch.nn as nn

from patch_tst_scorer import PatchTSTScorer
from stock_transformer import StockTemporalTransformer

ModelType = Literal["transformer", "patchtst"]


@dataclass
class ModelBuildConfig:
    model_type: ModelType = "transformer"
    feat_dim: int = 11
    lookback: int = 30
    d_model: int = 64
    num_heads: int = 4
    ffn_dim: int = 128
    dropout: float = 0.1
    # PatchTST
    patch_len: int = 6
    patch_stride: int = 6
    patch_e_layers: int = 2
    patch_cs_layers: int = 1


def build_scorer(cfg: ModelBuildConfig) -> nn.Module:
    if cfg.model_type == "patchtst":
        return PatchTSTScorer(
            feat_dim=cfg.feat_dim,
            seq_len=cfg.lookback,
            patch_len=cfg.patch_len,
            patch_stride=cfg.patch_stride,
            d_model=cfg.d_model,
            num_heads=cfg.num_heads,
            e_layers=cfg.patch_e_layers,
            ffn_dim=cfg.ffn_dim,
            dropout=cfg.dropout,
            cs_layers=cfg.patch_cs_layers,
        )
    return StockTemporalTransformer(
        feat_dim=cfg.feat_dim,
        max_lookback=cfg.lookback,
        d_model=cfg.d_model,
        num_heads=cfg.num_heads,
        ffn_dim=cfg.ffn_dim,
        dropout=cfg.dropout,
    )


def model_build_config_from_sft(cfg) -> ModelBuildConfig:
    return ModelBuildConfig(
        model_type=getattr(cfg, "model_type", "transformer"),
        feat_dim=getattr(cfg, "_feat_dim", 11),
        lookback=cfg.lookback,
        d_model=cfg.d_model,
        num_heads=cfg.num_heads,
        ffn_dim=cfg.ffn_dim,
        dropout=cfg.dropout,
        patch_len=getattr(cfg, "patch_len", 6),
        patch_stride=getattr(cfg, "patch_stride", 6),
        patch_e_layers=getattr(cfg, "patch_e_layers", 2),
        patch_cs_layers=getattr(cfg, "patch_cs_layers", 1),
    )


def model_build_config_from_grpo(cfg, feat_dim: int) -> ModelBuildConfig:
    return ModelBuildConfig(
        model_type=getattr(cfg, "model_type", "transformer"),
        feat_dim=feat_dim,
        lookback=cfg.lookback,
        d_model=cfg.d_model,
        num_heads=cfg.num_heads,
        ffn_dim=cfg.ffn_dim,
        dropout=cfg.dropout,
        patch_len=getattr(cfg, "patch_len", 6),
        patch_stride=getattr(cfg, "patch_stride", 6),
        patch_e_layers=getattr(cfg, "patch_e_layers", 2),
        patch_cs_layers=getattr(cfg, "patch_cs_layers", 1),
    )
