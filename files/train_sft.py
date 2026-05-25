"""
SFT training for ``StockTemporalTransformer`` on HF cross-sectional data.

Each sample
-----------
    x : (N, L, F)   N stocks, L lookback bars, F features
    y : (N,)        rank-normalized linear-factor label in [0, 1].

The student (Transformer) is trained to mimic the teacher (linear factor
composite produced by ``hf_data.py``). After SFT the same model is fine-
tuned by GRPO against IC + Sharpe + drawdown.
"""

from __future__ import annotations

import csv
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from hf_data import (
    DEFAULT_POOL_DIR,
    HFConfig,
    HFSFTDataset,
    bar_indices_for_dates,
    build_hf_dataset,
    split_train_val_test,
)
from stock_transformer import StockTemporalTransformer
from model_factory import build_scorer, model_build_config_from_sft


HERE = Path(__file__).resolve().parent


@dataclass
class SFTConfig:
    pool_dir: Path = DEFAULT_POOL_DIR
    start_day: int = 0
    num_days: int = 3
    lookback: int = 30
    horizon: int = 5
    n_stocks_max: Optional[int] = None
    stock_ids: Optional[List[str]] = None
    cache_tag: Optional[str] = None
    extra_pool_dirs: Optional[List[Path]] = None
    use_regime_features: bool = False
    regime_lookback_days: int = 30
    train_dates: Optional[List[str]] = None   # SFT split only these calendar days

    d_model: int = 64
    num_heads: int = 4
    ffn_dim: int = 128
    dropout: float = 0.1

    epochs: int = 8
    batch_size: int = 16
    lr: float = 5e-4
    weight_decay: float = 1e-2
    warmup_ratio: float = 0.05
    grad_clip: float = 1.0

    val_ratio: float = 0.2
    test_ratio: float = 0.2

    mse_weight: float = 0.5
    rank_weight: float = 0.2

    model_type: str = "transformer"   # "transformer" | "patchtst"
    patch_len: int = 6
    patch_stride: int = 6
    patch_e_layers: int = 2
    patch_cs_layers: int = 1

    seed: int = 42
    ckpt_dir: Path = HERE / "checkpoints"
    log_dir: Path = HERE / "logs"


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Loss + metrics
# ---------------------------------------------------------------------------
def pearson_corr(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    s = scores - scores.mean(dim=-1, keepdim=True)
    l = labels - labels.mean(dim=-1, keepdim=True)
    num = (s * l).sum(dim=-1)
    den = s.norm(dim=-1) * l.norm(dim=-1) + 1e-8
    return num / den


def rank_tensor(x: torch.Tensor) -> torch.Tensor:
    return x.argsort(dim=-1).argsort(dim=-1).float()


def rank_ic(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return pearson_corr(rank_tensor(scores), rank_tensor(labels))


def pairwise_rank_loss(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    score_diff = scores.unsqueeze(-1) - scores.unsqueeze(-2)
    label_diff = labels.unsqueeze(-1) - labels.unsqueeze(-2)
    sign = label_diff.sign()
    valid = sign != 0
    if not valid.any():
        return scores.new_tensor(0.0)
    return nn.functional.softplus(-score_diff[valid] * sign[valid]).mean()


def sft_loss(scores: torch.Tensor, labels: torch.Tensor, cfg: SFTConfig) -> torch.Tensor:
    s = scores.clamp(1e-6, 1.0 - 1e-6)
    y = labels.clamp(0.0, 1.0)
    bce = nn.functional.binary_cross_entropy(s, y)
    mse = nn.functional.mse_loss(s, y)
    rnk = pairwise_rank_loss(scores, labels)
    return bce + cfg.mse_weight * mse + cfg.rank_weight * rnk


# ---------------------------------------------------------------------------
# Schedule
# ---------------------------------------------------------------------------
def get_scheduler(
    optimizer: optim.Optimizer, total_steps: int, warmup_steps: int
) -> optim.lr_scheduler.LambdaLR:
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Epoch
# ---------------------------------------------------------------------------
def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Optional[optim.Optimizer],
    scheduler: Optional[optim.lr_scheduler.LambdaLR],
    cfg: SFTConfig,
    device: torch.device,
) -> Tuple[float, float, float]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = total_ic = total_ric = 0.0
    n_batches = 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            scores = model(x)               # (B, N)
            loss = sft_loss(scores, y, cfg)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

            with torch.no_grad():
                total_loss += loss.item()
                total_ic += pearson_corr(scores, y).mean().item()
                total_ric += rank_ic(scores, y).mean().item()
                n_batches += 1

    return total_loss / n_batches, total_ic / n_batches, total_ric / n_batches


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
def train_sft(cfg: SFTConfig = SFTConfig()) -> StockTemporalTransformer:
    set_seed(cfg.seed)
    cfg.ckpt_dir.mkdir(parents=True, exist_ok=True)
    cfg.log_dir.mkdir(parents=True, exist_ok=True)

    hf_cfg = HFConfig(
        pool_dir=cfg.pool_dir,
        extra_pool_dirs=cfg.extra_pool_dirs,
        start_day=cfg.start_day,
        num_days=cfg.num_days,
        lookback=cfg.lookback,
        horizon=cfg.horizon,
        n_stocks_max=cfg.n_stocks_max,
        stock_ids=cfg.stock_ids,
        cache_tag=cfg.cache_tag,
        use_regime_features=cfg.use_regime_features,
        regime_lookback_days=cfg.regime_lookback_days,
    )
    bundle = build_hf_dataset(hf_cfg)
    cand = (
        bar_indices_for_dates(bundle, cfg.train_dates)
        if cfg.train_dates
        else None
    )
    splits = split_train_val_test(
        bundle, val_ratio=cfg.val_ratio, test_ratio=cfg.test_ratio, embargo=cfg.horizon,
        candidate_indices=cand,
    )

    train_ds = HFSFTDataset(bundle, cfg.lookback, splits["train"])
    val_ds = HFSFTDataset(bundle, cfg.lookback, splits["val"])
    test_ds = HFSFTDataset(bundle, cfg.lookback, splits["test"])

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)

    feat_dim = bundle["x_seq"].shape[-1]
    n_stocks = bundle["x_seq"].shape[1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mb = model_build_config_from_sft(cfg)
    mb.feat_dim = feat_dim
    model = build_scorer(mb).to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        betas=(0.9, 0.999),
    )
    total_steps = max(cfg.epochs * math.ceil(len(train_ds) / cfg.batch_size), 1)
    warmup_steps = max(int(total_steps * cfg.warmup_ratio), 1)
    scheduler = get_scheduler(optimizer, total_steps, warmup_steps)

    log_path = cfg.log_dir / "sft_log.csv"
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow([
            "epoch", "train_loss", "train_ic", "train_rank_ic",
            "val_loss", "val_ic", "val_rank_ic", "lr",
        ])

    print(f"device      : {device}")
    print(f"data shape  : N={n_stocks}, F={feat_dim}, L={cfg.lookback}, H={cfg.horizon}")
    print(f"samples     : train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")
    print(f"params      : {sum(p.numel() for p in model.parameters()):,}")
    print(f"loss        : BCE + {cfg.mse_weight}*MSE + {cfg.rank_weight}*pairwise_rank")
    print("-" * 80)

    best_val_ric = -float("inf")
    for epoch in range(1, cfg.epochs + 1):
        tr_loss, tr_ic, tr_ric = run_epoch(model, train_loader, optimizer, scheduler, cfg, device)
        va_loss, va_ic, va_ric = run_epoch(model, val_loader, None, None, cfg, device)
        lr_now = scheduler.get_last_lr()[0]

        print(
            f"epoch {epoch:03d}/{cfg.epochs} "
            f"loss={tr_loss:.4f} ic={tr_ic:.4f} ric={tr_ric:.4f} | "
            f"val_loss={va_loss:.4f} val_ic={va_ic:.4f} val_ric={va_ric:.4f} "
            f"lr={lr_now:.2e}"
        )
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, tr_loss, tr_ic, tr_ric, va_loss, va_ic, va_ric, lr_now])

        if va_ric > best_val_ric:
            best_val_ric = va_ric
            torch.save(model.state_dict(), cfg.ckpt_dir / "best_sft_model.pt")

    # Test eval (best model)
    model.load_state_dict(torch.load(cfg.ckpt_dir / "best_sft_model.pt", map_location=device))
    te_loss, te_ic, te_ric = run_epoch(model, test_loader, None, None, cfg, device)
    print("-" * 80)
    print(f"TEST       : loss={te_loss:.4f} ic={te_ic:.4f} rank_ic={te_ric:.4f}")
    print(f"best val   : rank_ic={best_val_ric:.4f}")

    torch.save(model.state_dict(), cfg.ckpt_dir / "last_sft_model.pt")
    return model


if __name__ == "__main__":
    train_sft()
