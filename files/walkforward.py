"""
Walk-forward GRPO experiment (Apr 2026).

Each fold:
  1. SFT on 3 training days
  2. Auto-learn reward weights on val split
  3. GRPO with long-short topK/bottomK portfolio + multi-objective reward
  4. OOS backtest on the next 3 trading days
  5. **Save every intermediate GRPO model** under a dedicated experiment tree
     (never touches ``files/checkpoints/`` or ``files/backtest_out/``).

Experiment layout
-----------------
files/experiments/walkforward_202604/
  universe/top30_ytd2026.csv
  folds/
    fold_000_20260401_20260403/
      meta.json
      best_sft_model.pt
      grpo_model.pt          <-- kept per fold
      reward_weights.json
      sft_log.csv
      grpo_log.csv
    fold_001_...
  backtest/
    stitched_bar_returns.csv
    walkforward_curves.png
    fold_summary.csv
"""

from __future__ import annotations

import copy
import csv
import json
import math
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from backtest import BARS_PER_YEAR, annualized_stats, simulate
from hf_data import (
    HFConfig,
    HFEpisodeDataset,
    HFSFTDataset,
    bar_indices_for_dates,
    build_hf_dataset,
    list_trading_days,
    split_train_val_test,
    trading_day_index,
)
from portfolio import (
    LongShortConfig,
    long_short_equal_weight_vector,
    sample_noisy_scores,
    scores_to_portfolio_weights,
)
from reward_model import (
    RewardConfig,
    empirical_reward_config,
    episode_reward_v2,
    search_reward_weights,
)
from stock_select import select_top_n
from stock_transformer import StockTemporalTransformer
from train_grpo_trl import (
    GRPOTrainConfig,
    _build_model,
    _load_sft_weights,
    compute_advantages,
    get_scheduler,
    grpo_loss,
    set_seed,
)
from train_sft import SFTConfig, train_sft


HERE = Path(__file__).resolve().parent

# ── isolated experiment root (do NOT reuse old artifact dirs) ─────────────
EXPERIMENT_ROOT = HERE / "experiments" / "walkforward_202604"
FOLDS_DIR = EXPERIMENT_ROOT / "folds"
BACKTEST_DIR = EXPERIMENT_ROOT / "backtest"
UNIVERSE_DIR = EXPERIMENT_ROOT / "universe"

POOL_DIR = HERE / "csi500" / "2026_1min"
POOL_DIR_2025 = HERE / "csi500" / "2025"
LOOKBACK = 30
HORIZON = 5
N_STOCKS = 30
TOP_K = 5
BOTTOM_K = 5

# calendar anchors (data ends 2026-05-08; 05-11/05-14 not available)
TRAIN_CAL_START = "2026-04-01"
TRAIN_CAL_END = "2026-05-08"      # closest to user's 05-11
OOS_CAL_START = "2026-04-07"      # first trading day after 04-04 holiday gap
OOS_CAL_END = "2026-05-08"        # closest to user's 05-14

TRAIN_DAYS_PER_FOLD = 3
OOS_DAYS_PER_FOLD = 3


@dataclass
class WalkForwardConfig:
    experiment_tag: str = "walkforward_202604"
    experiment_root: Optional[Path] = None
    pool_dir: Path = POOL_DIR
    stock_ids: Optional[List[str]] = None
    train_days: int = TRAIN_DAYS_PER_FOLD
    oos_days: int = OOS_DAYS_PER_FOLD
    lookback: int = LOOKBACK
    horizon: int = HORIZON
    top_k: int = TOP_K
    bottom_k: int = BOTTOM_K
    reward_mode: str = "auto"          # "auto" | "empirical"
    reward_config: Optional[RewardConfig] = None
    warm_chain: bool = False           # SFT only fold0; later folds init from prev grpo_model.pt
    schedule_mode: str = "block"       # "block" | "rolling_1d" | "monthly_semiannual"
    train_cal_start: Optional[str] = None
    train_cal_end: Optional[str] = None
    oos_cal_start: Optional[str] = None
    oos_cal_end: Optional[str] = None
    extra_pool_dirs: Optional[List[Path]] = None
    oos_stitch_start: Optional[str] = None   # e.g. "2026-01" — filter plotted OOS
    oos_stitch_end: Optional[str] = None     # e.g. "2026-04"
    sft_epochs: int = 4
    sft_batch_size: int = 16
    grpo_epochs: int = 2
    grpo_generations: int = 4
    episode_len: int = 12
    seed: int = 42
    baseline_mode: str = "long_only"   # "long_only" | "ls_equal"
    portfolio_mode: str = "long_short" # "long_short" | "long_only"
    model_type: str = "transformer"    # "transformer" | "patchtst" | "linear"
    use_regime_features: bool = False
    regime_lookback_days: int = 30
    patch_len: int = 6
    patch_stride: int = 6
    patch_e_layers: int = 2
    patch_cs_layers: int = 1
    eval_extended: bool = False        # excess return + corr metrics + extended plot

    def __post_init__(self) -> None:
        if self.experiment_root is None:
            self.experiment_root = HERE / "experiments" / self.experiment_tag


def _td_idx(cfg: WalkForwardConfig, date: str) -> int:
    return trading_day_index(cfg.pool_dir, date, cfg.extra_pool_dirs)


def _hf_data_span(
    cfg: WalkForwardConfig,
    fold_info: Dict,
    *,
    for_oos: bool = False,
) -> Tuple[int, int]:
    """
    Calendar span to **load** into HF cache.

    When regime is enabled, prepend ``regime_lookback_days`` of history so each
    bar can compute regime from the prior ~30d market — without expanding the
    actual train/OOS sample window (see ``bar_indices_for_dates``).

    When ``lookback >= bars_per_day`` (1 trading day), prepend enough prior
    trading days so the first bar of train/OOS can fill a full L-bar window
    (cross-day minute history).
    """
    bars_per_day = 240
    if for_oos:
        start = fold_info["oos_start_day"]
        num = fold_info["oos_num_days"]
    else:
        start = fold_info["train_start_day"]
        num = fold_info["train_num_days"]
    hist_start = start
    if cfg.use_regime_features and cfg.regime_lookback_days > 0:
        hist_start = min(hist_start, max(0, start - cfg.regime_lookback_days))
    if cfg.lookback >= bars_per_day:
        lb_days = math.ceil((cfg.lookback - 1) / bars_per_day)
        hist_start = min(hist_start, max(0, start - lb_days))
    if hist_start < start:
        return hist_start, start + num - hist_start
    return start, num


def _hf_config(
    cfg: WalkForwardConfig,
    start_day: int,
    num_days: int,
    stock_ids: List[str],
    cache_tag: str,
) -> HFConfig:
    return HFConfig(
        pool_dir=cfg.pool_dir,
        extra_pool_dirs=cfg.extra_pool_dirs,
        start_day=start_day,
        num_days=num_days,
        lookback=cfg.lookback,
        horizon=cfg.horizon,
        stock_ids=stock_ids,
        cache_tag=cache_tag,
        use_regime_features=cfg.use_regime_features,
        regime_lookback_days=cfg.regime_lookback_days,
    )


def _portfolio_config(cfg: WalkForwardConfig) -> LongShortConfig:
    if cfg.portfolio_mode == "long_only":
        return LongShortConfig(
            top_k=cfg.top_k,
            bottom_k=0,
            long_gross=1.0,
            short_gross=0.0,
        )
    return LongShortConfig(
        top_k=cfg.top_k,
        bottom_k=cfg.bottom_k,
        long_gross=0.5,
        short_gross=0.5,
    )


def _cache_tag(cfg: WalkForwardConfig, fold_idx: int, split: str) -> str:
    """Unique HF cache key per experiment + fold + split."""
    tag = f"{cfg.experiment_tag}_f{fold_idx:03d}_{split}"
    if cfg.use_regime_features:
        tag += "_regime"
    if cfg.model_type == "patchtst":
        tag += "_patchtst"
    if cfg.model_type == "linear":
        tag += "_linear"
    return tag


def _fold_dir(root: Path, fold_idx: int, train_dates: List[str]) -> Path:
    tag = f"fold_{fold_idx:03d}_{train_dates[0].replace('-','')}_{train_dates[-1].replace('-','')}"
    return root / "folds" / tag


def _ts_from_int64(arr: np.ndarray) -> pd.DatetimeIndex:
    a = np.asarray(arr, dtype="int64")
    m = int(np.abs(a).max()) if a.size else 0
    unit = "ns" if m >= 10**17 else "us" if m >= 10**14 else "ms"
    return pd.to_datetime(a, unit=unit)


def _calendar_bounds(cfg: WalkForwardConfig) -> Tuple[str, str, str, str]:
    return (
        cfg.train_cal_start or TRAIN_CAL_START,
        cfg.train_cal_end or TRAIN_CAL_END,
        cfg.oos_cal_start or OOS_CAL_START,
        cfg.oos_cal_end or OOS_CAL_END,
    )


def _append_fold(
    folds: List[Dict],
    fold_idx: int,
    train_dates: List[str],
    oos_dates: List[str],
    cfg: WalkForwardConfig,
) -> None:
    folds.append({
        "fold_idx": fold_idx,
        "train_dates": train_dates,
        "oos_dates": oos_dates,
        "train_start_day": _td_idx(cfg, train_dates[0]),
        "train_num_days": len(train_dates),
        "oos_start_day": _td_idx(cfg, oos_dates[0]),
        "oos_num_days": len(oos_dates),
    })


# (train_start_month, train_end_month, oos_month) — month strings YYYY-MM
MONTHLY_SEMIANNUAL_SPECS: List[Tuple[str, str, str]] = [
    ("2025-06", "2025-12", "2026-01"),
    ("2025-07", "2026-01", "2026-02"),
    ("2025-08", "2026-02", "2026-03"),
    ("2025-09", "2026-03", "2026-04"),
]


def _build_monthly_semiannual_schedule(cfg: WalkForwardConfig) -> List[Dict]:
    """~6mo train (Jun–Dec style) with 1mo OOS; slide train/OOS by one month."""
    specs = MONTHLY_SEMIANNUAL_SPECS
    if cfg.oos_stitch_start and cfg.oos_stitch_end:
        specs = [
            s for s in specs
            if cfg.oos_stitch_start <= s[2] <= cfg.oos_stitch_end
        ]
    all_days = list_trading_days(
        cfg.pool_dir, "2025-01-01", "2026-06-30", cfg.extra_pool_dirs,
    )
    day_list = [d.strftime("%Y-%m-%d") for d in all_days]

    folds: List[Dict] = []
    for fold_idx, (train_lo, train_hi, oos_m) in enumerate(specs):
        train_dates = [d for d in day_list if train_lo <= d[:7] <= train_hi]
        oos_dates = [d for d in day_list if d.startswith(oos_m)]
        if len(train_dates) < 20 or len(oos_dates) < 5:
            continue
        _append_fold(folds, fold_idx, train_dates, oos_dates, cfg)
    return folds


def _build_rolling_1d_schedule(cfg: WalkForwardConfig) -> List[Dict]:
    """Rolling 3d-train → 1d-OOS: each OOS day uses the prior ``train_days`` window."""
    train_cal_start, train_cal_end, oos_cal_start, oos_cal_end = _calendar_bounds(cfg)
    # Load extended calendar so the window ending before oos_cal_start is available.
    cal_start = pd.Timestamp(train_cal_start) - pd.Timedelta(days=45)
    all_days = list_trading_days(
        cfg.pool_dir, cal_start.strftime("%Y-%m-%d"), oos_cal_end,
    )
    day_list = [d.strftime("%Y-%m-%d") for d in all_days]
    available = set(day_list)

    folds: List[Dict] = []
    fold_idx = 0
    for oos_date in day_list:
        if oos_date < oos_cal_start or oos_date > oos_cal_end:
            continue
        if oos_date not in available:
            continue
        oos_i = day_list.index(oos_date)
        train_end_i = oos_i - 1
        if train_end_i < 0:
            continue
        train_start_i = train_end_i - cfg.train_days + 1
        if train_start_i < 0:
            continue
        train_dates = day_list[train_start_i: train_end_i + 1]
        if len(train_dates) != cfg.train_days:
            continue
        if train_dates[0] < train_cal_start or train_dates[-1] > train_cal_end:
            continue
        oos_dates = [oos_date]
        _append_fold(folds, fold_idx, train_dates, oos_dates, cfg)
        fold_idx += 1
    return folds


def build_fold_schedule(cfg: WalkForwardConfig) -> List[Dict]:
    if cfg.schedule_mode == "monthly_semiannual":
        return _build_monthly_semiannual_schedule(cfg)
    if cfg.schedule_mode == "rolling_1d":
        return _build_rolling_1d_schedule(cfg)

    train_cal_start, train_cal_end, oos_cal_start, oos_cal_end = _calendar_bounds(cfg)
    all_days = list_trading_days(cfg.pool_dir, train_cal_start, oos_cal_end)
    day_list = [d.strftime("%Y-%m-%d") for d in all_days]

    oos_start_idx = day_list.index(pd.Timestamp(oos_cal_start).strftime("%Y-%m-%d"))

    folds: List[Dict] = []
    fold_idx = 0
    oos_i = oos_start_idx
    while oos_i + cfg.oos_days <= len(day_list):
        oos_dates = day_list[oos_i: oos_i + cfg.oos_days]
        train_end_i = oos_i - 1
        train_start_i = train_end_i - cfg.train_days + 1
        if train_start_i < 0:
            break
        train_dates = day_list[train_start_i: train_end_i + 1]
        if train_dates[0] < train_cal_start:
            break
        if train_dates[-1] > train_cal_end:
            break
        _append_fold(folds, fold_idx, train_dates, oos_dates, cfg)
        fold_idx += 1
        oos_i += cfg.oos_days
    return folds


def _collate_episodes(items):
    if len(items[0]) == 4:
        xs = torch.stack([it[0] for it in items], dim=0)
        rs = torch.stack([it[1] for it in items], dim=0)
        sz = torch.stack([it[2] for it in items], dim=0)
        bz = torch.stack([it[3] for it in items], dim=0)
        return xs, rs, sz, bz
    xs = torch.stack([it[0] for it in items], dim=0)
    rs = torch.stack([it[1] for it in items], dim=0)
    return xs, rs


def _gaussian_log_prob(scores: torch.Tensor, noisy: torch.Tensor, std: float) -> torch.Tensor:
    """scores (M,N), noisy (G,M,N) -> log_prob (G,M) summed over N in caller."""
    eps = (noisy - scores.unsqueeze(0)) / std
    lp = -0.5 * eps.pow(2) - math.log(std) - 0.5 * math.log(2 * math.pi)
    return lp.sum(dim=-1)


def grpo_step_longshort(
    policy: nn.Module,
    ref_policy: nn.Module,
    x_episodes: torch.Tensor,
    r_episodes: torch.Tensor,
    sz_episodes: Optional[torch.Tensor],
    bz_episodes: Optional[torch.Tensor],
    optimizer: optim.Optimizer,
    scheduler: Optional[optim.lr_scheduler.LambdaLR],
    grpo_cfg: GRPOTrainConfig,
    reward_cfg: RewardConfig,
    ls_cfg: LongShortConfig,
    portfolio_mode: str = "long_short",
) -> Dict[str, float]:
    B = x_episodes.shape[0]
    device = x_episodes.device
    metrics_acc: Dict[str, float] = {}
    optimizer.zero_grad()

    for b in range(B):
        x_ep = x_episodes[b]
        r_ep = r_episodes[b]
        sz_ep = sz_episodes[b] if sz_episodes is not None else None
        bz_ep = bz_episodes[b] if bz_episodes is not None else None

        scores = policy(x_ep)
        noisy, _ = sample_noisy_scores(scores, ls_cfg, grpo_cfg.num_generations)
        std = max(ls_cfg.score_noise_std, 1e-4)
        log_prob_new = _gaussian_log_prob(scores, noisy, std)
        log_prob_old = log_prob_new.detach()

        weights_gm = torch.stack([
            scores_to_portfolio_weights(noisy[g], ls_cfg, portfolio_mode)
            for g in range(grpo_cfg.num_generations)
        ], dim=0)

        with torch.no_grad():
            ref_scores = ref_policy(x_ep)
            log_prob_ref = _gaussian_log_prob(ref_scores, noisy, std)
            rew_dict = episode_reward_v2(
                weights_gm, r_ep,
                sz_ep, bz_ep, reward_cfg,
            )
            rewards = rew_dict["reward"]
            advantages = compute_advantages(rewards, grpo_cfg.scale_rewards)

        sum_lp_new = log_prob_new.sum(dim=1)
        sum_lp_old = log_prob_old.sum(dim=1)
        sum_lp_ref = log_prob_ref.sum(dim=1)

        loss, m = grpo_loss(sum_lp_new, sum_lp_old, sum_lp_ref, advantages, grpo_cfg)
        (loss / B).backward()

        with torch.no_grad():
            for k, v in m.items():
                metrics_acc[k] = metrics_acc.get(k, 0.0) + v
            metrics_acc["reward_mean"] = metrics_acc.get("reward_mean", 0.0) + rewards.mean().item()
            metrics_acc["ic_mean"] = metrics_acc.get("ic_mean", 0.0) + rew_dict["ic"].mean().item()
            metrics_acc["sharpe_mean"] = metrics_acc.get("sharpe_mean", 0.0) + rew_dict["sharpe"].mean().item()
            metrics_acc["max_dd_mean"] = metrics_acc.get("max_dd_mean", 0.0) + rew_dict["max_dd"].mean().item()

    nn.utils.clip_grad_norm_(policy.parameters(), grpo_cfg.grad_clip)
    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    return {k: v / B for k, v in metrics_acc.items()}


def calibrate_reward(
    policy: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    ls_cfg: LongShortConfig,
    base_reward: RewardConfig,
    portfolio_mode: str = "long_short",
) -> RewardConfig:
    """Grid-search reward weights on validation episodes."""
    all_w: List[torch.Tensor] = []
    all_r: List[torch.Tensor] = []
    all_sz: List[torch.Tensor] = []
    all_bz: List[torch.Tensor] = []
    has_exp = False
    policy.eval()
    with torch.no_grad():
        for batch in val_loader:
            if len(batch) == 4:
                x, r, sz, bz = batch
                has_exp = True
            else:
                x, r = batch
                sz = bz = None
            x = x.to(device)
            if sz is not None:
                sz = sz.to(device)
                bz = bz.to(device)
            for b in range(x.shape[0]):
                scores = policy(x[b])
                w = scores_to_portfolio_weights(scores, ls_cfg, portfolio_mode).unsqueeze(0)
                all_w.append(w.cpu())
                all_r.append(r[b : b + 1].cpu())
                if has_exp and sz is not None:
                    all_sz.append(sz[b : b + 1].cpu())
                    all_bz.append(bz[b : b + 1].cpu())
    if not all_w:
        return base_reward
    W = torch.cat(all_w, dim=0)          # (E, 1, M, N) -> need squeeze?
    # each w is (1, M, N); cat -> (E, M, N)
    W = W.squeeze(1) if W.dim() == 4 else W
    R = torch.cat(all_r, dim=0)          # (E, M, N)
    SZ = torch.cat(all_sz, dim=0) if all_sz else None
    BZ = torch.cat(all_bz, dim=0) if all_bz else None
    return search_reward_weights(W, R, SZ, BZ, base_reward)


def train_fold_grpo(
    fold_info: Dict,
    fold_dir: Path,
    stock_ids: List[str],
    init_ckpt: Path,
    cfg: WalkForwardConfig,
    reward_cfg: RewardConfig,
) -> Path:
    """Train GRPO for one fold; save ``grpo_model.pt`` inside ``fold_dir``."""
    fold_dir.mkdir(parents=True, exist_ok=True)
    if not init_ckpt.is_file():
        raise FileNotFoundError(f"Init checkpoint not found: {init_ckpt}")
    cache_tag = _cache_tag(cfg, fold_info["fold_idx"], "train")

    data_start, data_days = _hf_data_span(cfg, fold_info, for_oos=False)
    hf_cfg = _hf_config(cfg, data_start, data_days, stock_ids, cache_tag)
    bundle = build_hf_dataset(hf_cfg)
    train_bars = bar_indices_for_dates(bundle, fold_info["train_dates"])
    splits = split_train_val_test(
        bundle, val_ratio=0.25, test_ratio=0.1, embargo=cfg.horizon,
        candidate_indices=train_bars,
    )
    train_ds = HFEpisodeDataset(bundle, cfg.lookback, cfg.episode_len, splits["train"])
    if len(train_ds) == 0:
        raise RuntimeError(f"Fold {fold_info['fold_idx']}: empty train episodes.")

    train_loader = DataLoader(
        train_ds, batch_size=2, shuffle=True, collate_fn=_collate_episodes,
    )

    grpo_cfg = GRPOTrainConfig(
        pool_dir=cfg.pool_dir,
        start_day=fold_info["train_start_day"],
        num_days=fold_info["train_num_days"],
        lookback=cfg.lookback,
        horizon=cfg.horizon,
        episode_len=cfg.episode_len,
        epochs=cfg.grpo_epochs,
        batch_size=2,
        num_generations=cfg.grpo_generations,
        sft_ckpt=init_ckpt,
        ckpt_dir=fold_dir,
        log_dir=fold_dir,
        model_type=cfg.model_type,
        patch_len=cfg.patch_len,
        patch_stride=cfg.patch_stride,
        patch_e_layers=cfg.patch_e_layers,
        patch_cs_layers=cfg.patch_cs_layers,
    )
    ls_cfg = _portfolio_config(cfg)

    feat_dim = bundle["x_seq"].shape[-1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = _build_model(feat_dim, grpo_cfg).to(device)
    policy.load_state_dict(torch.load(init_ckpt, map_location=device))
    print(f"  loaded init checkpoint: {init_ckpt.name}")
    ref_policy = copy.deepcopy(policy).to(device)
    for p in ref_policy.parameters():
        p.requires_grad_(False)
    ref_policy.eval()

    optimizer = optim.AdamW(policy.parameters(), lr=grpo_cfg.lr, weight_decay=grpo_cfg.weight_decay)
    total_steps = max(grpo_cfg.epochs * math.ceil(len(train_ds) / 2), 1)
    warmup = max(int(total_steps * grpo_cfg.warmup_ratio), 1)
    scheduler = get_scheduler(optimizer, total_steps, warmup)

    log_path = fold_dir / "grpo_log.csv"
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow([
            "epoch", "step", "loss", "reward_mean", "ic_mean", "sharpe_mean", "max_dd_mean", "lr",
        ])

    global_step = 0
    for epoch in range(1, grpo_cfg.epochs + 1):
        policy.train()
        ep_m: Dict[str, float] = {}
        n = 0
        for batch in train_loader:
            if len(batch) == 4:
                x, r, sz, bz = batch
                sz, bz = sz.to(device), bz.to(device)
            else:
                x, r = batch
                sz = bz = None
            x, r = x.to(device), r.to(device)
            m = grpo_step_longshort(
                policy, ref_policy, x, r, sz, bz,
                optimizer, scheduler, grpo_cfg, reward_cfg, ls_cfg,
                portfolio_mode=cfg.portfolio_mode,
            )
            global_step += 1
            n += 1
            for k, v in m.items():
                ep_m[k] = ep_m.get(k, 0.0) + v
            with open(log_path, "a", newline="") as f:
                csv.writer(f).writerow([
                    epoch, global_step, m["loss"], m.get("reward_mean", 0),
                    m.get("ic_mean", 0), m.get("sharpe_mean", 0),
                    m.get("max_dd_mean", 0), scheduler.get_last_lr()[0],
                ])
        avg = {k: v / n for k, v in ep_m.items()}
        print(
            f"  GRPO ep {epoch}/{grpo_cfg.epochs} "
            f"loss={avg['loss']:.4f} reward={avg.get('reward_mean',0):.4f} "
            f"ic={avg.get('ic_mean',0):.4f}"
        )

    grpo_path = fold_dir / "grpo_model.pt"
    torch.save(policy.state_dict(), grpo_path)
    print(f"  saved GRPO checkpoint -> {grpo_path}")
    return grpo_path


def backtest_fold_oos(
    fold_info: Dict,
    fold_dir: Path,
    grpo_ckpt: Path,
    stock_ids: List[str],
    cfg: WalkForwardConfig,
) -> pd.DataFrame:
    """Run OOS backtest for one fold using its dedicated ``grpo_model.pt``."""
    cache_tag = _cache_tag(cfg, fold_info["fold_idx"], "oos")
    data_start, data_days = _hf_data_span(cfg, fold_info, for_oos=True)
    hf_cfg = _hf_config(cfg, data_start, data_days, stock_ids, cache_tag)
    bundle = build_hf_dataset(hf_cfg)
    x_seq = bundle["x_seq"]
    fwd = bundle["fwd_return"]
    oos_bars = bar_indices_for_dates(bundle, fold_info["oos_dates"])
    L = cfg.lookback
    T_bt = len(oos_bars)
    n_stocks = x_seq.shape[1]
    feat_dim = x_seq.shape[-1]

    grpo_cfg = GRPOTrainConfig(
        lookback=cfg.lookback,
        model_type=cfg.model_type,
        patch_len=cfg.patch_len,
        patch_stride=cfg.patch_stride,
        patch_e_layers=cfg.patch_e_layers,
        patch_cs_layers=cfg.patch_cs_layers,
    )
    ls_cfg = _portfolio_config(cfg)
    model = _build_model(feat_dim, grpo_cfg)
    model.load_state_dict(torch.load(grpo_ckpt, map_location="cpu"))
    model.eval()

    weights = np.zeros((T_bt, n_stocks), dtype=np.float32)
    returns = np.zeros((T_bt, n_stocks), dtype=np.float32)
    scores_arr = np.zeros((T_bt, n_stocks), dtype=np.float32)
    ts_idx = np.zeros(T_bt, dtype=np.int64)

    with torch.no_grad():
        for i, t in enumerate(oos_bars):
            t = int(t)
            window = x_seq[t - L + 1: t + 1]
            x = np.transpose(window, (1, 0, 2)).copy()
            x_t = torch.from_numpy(x).float()
            scores = model(x_t)
            w = scores_to_portfolio_weights(scores, ls_cfg, cfg.portfolio_mode).numpy()
            weights[i] = w
            returns[i] = fwd[t]
            scores_arr[i] = scores.numpy()
            ts_idx[i] = t

    from backtest import BacktestConfig
    bt_cfg = BacktestConfig(
        lookback=cfg.lookback, horizon=cfg.horizon,
        fee_bps=5.0, rebalance_every=5,
    )
    sim = simulate(weights, returns, bt_cfg)
    row_w = _baseline_weights(n_stocks, cfg)
    eq_weights = np.tile(row_w, (T_bt, 1))
    sim_eq = simulate(eq_weights, returns, bt_cfg)
    ts = _ts_from_int64(bundle["timestamps_ns"][ts_idx])

    df = pd.DataFrame({
        "fold": fold_info["fold_idx"],
        "ts": ts,
        "net_pr": sim["net_pr"],
        "ew_net_pr": sim_eq["net_pr"],
        "fold_dir": fold_dir.name,
    })
    if cfg.eval_extended:
        from eval_metrics import bar_level_correlations
        corrs = bar_level_correlations(scores_arr, weights, returns)
        df["score_return_corr"] = corrs["score_return_corr"]
        df["weight_return_corr"] = corrs["weight_return_corr"]
    df["nav"] = (1.0 + df["net_pr"]).groupby(df["fold"]).cumprod()
    df["ew_nav"] = (1.0 + df["ew_net_pr"]).groupby(df["fold"]).cumprod()
    return df


def backtest_fold_oos_linear(
    fold_info: Dict,
    fold_dir: Path,
    stock_ids: List[str],
    cfg: WalkForwardConfig,
) -> pd.DataFrame:
    """OOS backtest using precomputed linear-factor scores (SFT teacher labels)."""
    cache_tag = _cache_tag(cfg, fold_info["fold_idx"], "oos")
    data_start, data_days = _hf_data_span(cfg, fold_info, for_oos=True)
    hf_cfg = _hf_config(cfg, data_start, data_days, stock_ids, cache_tag)
    bundle = build_hf_dataset(hf_cfg)
    sft_label = bundle["sft_label"]
    fwd = bundle["fwd_return"]
    oos_bars = bar_indices_for_dates(bundle, fold_info["oos_dates"])
    T_bt = len(oos_bars)
    n_stocks = sft_label.shape[1]
    ls_cfg = _portfolio_config(cfg)

    weights = np.zeros((T_bt, n_stocks), dtype=np.float32)
    returns = np.zeros((T_bt, n_stocks), dtype=np.float32)
    scores_arr = np.zeros((T_bt, n_stocks), dtype=np.float32)
    ts_idx = np.zeros(T_bt, dtype=np.int64)

    for i, t in enumerate(oos_bars):
        t = int(t)
        scores = torch.from_numpy(sft_label[t].astype(np.float32))
        w = scores_to_portfolio_weights(scores, ls_cfg, cfg.portfolio_mode).numpy()
        weights[i] = w
        returns[i] = fwd[t]
        scores_arr[i] = sft_label[t]
        ts_idx[i] = t

    from backtest import BacktestConfig
    bt_cfg = BacktestConfig(
        lookback=cfg.lookback, horizon=cfg.horizon,
        fee_bps=5.0, rebalance_every=5,
    )
    sim = simulate(weights, returns, bt_cfg)
    row_w = _baseline_weights(n_stocks, cfg)
    eq_weights = np.tile(row_w, (T_bt, 1))
    sim_eq = simulate(eq_weights, returns, bt_cfg)
    ts = _ts_from_int64(bundle["timestamps_ns"][ts_idx])

    df = pd.DataFrame({
        "fold": fold_info["fold_idx"],
        "ts": ts,
        "net_pr": sim["net_pr"],
        "ew_net_pr": sim_eq["net_pr"],
        "fold_dir": fold_dir.name,
    })
    if cfg.eval_extended:
        from eval_metrics import bar_level_correlations
        corrs = bar_level_correlations(scores_arr, weights, returns)
        df["score_return_corr"] = corrs["score_return_corr"]
        df["weight_return_corr"] = corrs["weight_return_corr"]
    df["nav"] = (1.0 + df["net_pr"]).groupby(df["fold"]).cumprod()
    df["ew_nav"] = (1.0 + df["ew_net_pr"]).groupby(df["fold"]).cumprod()
    return df


def _baseline_weights(n_stocks: int, cfg: WalkForwardConfig) -> np.ndarray:
    if cfg.baseline_mode == "ls_equal":
        ls_cfg = LongShortConfig(top_k=cfg.top_k, bottom_k=cfg.bottom_k)
        return long_short_equal_weight_vector(n_stocks, ls_cfg)
    return np.full(n_stocks, 1.0 / n_stocks, dtype=np.float32)


def _baseline_label(cfg: WalkForwardConfig) -> str:
    if cfg.baseline_mode == "ls_equal":
        return f"LS Equal-weight (Top{cfg.top_k}/Bot{cfg.bottom_k})"
    return "Equal-weight (long-only)"


def _backtest_fold_equal_weight(
    fold_info: Dict,
    stock_ids: List[str],
    cfg: WalkForwardConfig,
) -> pd.DataFrame:
    """OOS baseline for one fold: long-only EW or fixed LS equal-weight."""
    cache_tag = _cache_tag(cfg, fold_info["fold_idx"], "oos")
    data_start, data_days = _hf_data_span(cfg, fold_info, for_oos=True)
    hf_cfg = _hf_config(cfg, data_start, data_days, stock_ids, cache_tag)
    bundle = build_hf_dataset(hf_cfg)
    fwd = bundle["fwd_return"]
    oos_bars = bar_indices_for_dates(bundle, fold_info["oos_dates"])
    T_bt = len(oos_bars)
    n_stocks = fwd.shape[1]
    returns = np.stack([fwd[int(t)] for t in oos_bars], axis=0)
    row_w = _baseline_weights(n_stocks, cfg)
    eq_weights = np.tile(row_w, (T_bt, 1))

    from backtest import BacktestConfig
    bt_cfg = BacktestConfig(
        lookback=cfg.lookback, horizon=cfg.horizon,
        fee_bps=5.0, rebalance_every=5,
    )
    sim_eq = simulate(eq_weights, returns, bt_cfg)
    ts = _ts_from_int64(bundle["timestamps_ns"][oos_bars])
    return pd.DataFrame({"fold": fold_info["fold_idx"], "ts": ts, "ew_net_pr": sim_eq["net_pr"]})


def plot_stitched(
    stitched: pd.DataFrame,
    out_path: Path,
    stats: Dict,
    stats_eq: Optional[Dict] = None,
    baseline_label: str = "Equal-weight",
    grpo_label: str = "Walk-forward GRPO",
) -> None:
    bar_idx = np.arange(len(stitched))
    nav = (1.0 + stitched["net_pr"]).cumprod()
    has_ew = "ew_net_pr" in stitched.columns
    ew_nav = (1.0 + stitched["ew_net_pr"]).cumprod() if has_ew else None

    win = min(120, max(20, len(stitched) // 5))
    rs = pd.Series(stitched["net_pr"])
    sh = (np.sqrt(BARS_PER_YEAR) * rs.rolling(win).mean() / (rs.rolling(win).std() + 1e-12)).to_numpy()
    if has_ew:
        rs_eq = pd.Series(stitched["ew_net_pr"])
        sh_eq = (
            np.sqrt(BARS_PER_YEAR) * rs_eq.rolling(win).mean() / (rs_eq.rolling(win).std() + 1e-12)
        ).to_numpy()

    dates = stitched["ts"].dt.normalize()
    day_change = np.r_[True, dates.values[1:] != dates.values[:-1]]
    day_starts = bar_idx[day_change]
    day_labels = stitched["ts"].iloc[day_starts].dt.strftime("%m-%d").tolist()

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]})
    ax = axes[0]
    ax.plot(bar_idx, nav, color="#d6336c", lw=1.6, label=grpo_label)
    if ew_nav is not None:
        ax.plot(bar_idx, ew_nav, color="#1864ab", lw=1.4, label=baseline_label)
    ax.axhline(1.0, color="#aaa", ls="--")
    for d in day_starts[1:]:
        ax.axvline(d, color="#ccc", ls=":", lw=0.7)
    ax.set_ylabel("Cumulative NAV")
    title = (
        f"Walk-forward OOS (~1 month)  "
        f"GRPO total={stats.get('total_return', 0) * 100:+.2f}%  "
        f"Sharpe={stats.get('annualized_sharpe', 0):+.2f}  "
        f"MDD={stats.get('max_drawdown', 0) * 100:.2f}%"
    )
    if stats_eq:
        title += (
            f"  |  EW total={stats_eq.get('total_return', 0) * 100:+.2f}%  "
            f"Sharpe={stats_eq.get('annualized_sharpe', 0):+.2f}"
        )
    ax.set_title(title)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(bar_idx, sh, color="#d6336c", lw=1.2, label=grpo_label.split()[0])
    if has_ew:
        ax.plot(bar_idx, sh_eq, color="#1864ab", lw=1.0, label=baseline_label)
    ax.axhline(0, color="#aaa", ls="--")
    for d in day_starts[1:]:
        ax.axvline(d, color="#ccc", ls=":", lw=0.7)
    ax.set_ylabel(f"Rolling Sharpe (win={win})")
    ax.set_xlabel("Trading bar index (gaps removed)")
    ax.legend(loc="upper left")
    ax.set_xticks(day_starts)
    ax.set_xticklabels(day_labels, rotation=30, ha="right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_longonly_extended(
    stitched: pd.DataFrame,
    out_path: Path,
    stats: Dict,
    stats_eq: Dict,
    eval_summary: Optional[Dict] = None,
    grpo_label: str = "GRPO Long-only Top5",
    baseline_label: str = "Equal-weight (long-only)",
) -> None:
    """NAV + rolling Sharpe + cumulative excess return; long-only GRPO vs EW."""
    bar_idx = np.arange(len(stitched))
    nav = (1.0 + stitched["net_pr"]).cumprod()
    ew_nav = (1.0 + stitched["ew_net_pr"]).cumprod()
    excess_nav = nav / ew_nav

    win = min(120, max(20, len(stitched) // 5))
    rs = pd.Series(stitched["net_pr"])
    rs_eq = pd.Series(stitched["ew_net_pr"])
    sh = (np.sqrt(BARS_PER_YEAR) * rs.rolling(win).mean() / (rs.rolling(win).std() + 1e-12)).to_numpy()
    sh_eq = (
        np.sqrt(BARS_PER_YEAR) * rs_eq.rolling(win).mean() / (rs_eq.rolling(win).std() + 1e-12)
    ).to_numpy()

    dates = stitched["ts"].dt.normalize()
    day_change = np.r_[True, dates.values[1:] != dates.values[:-1]]
    day_starts = bar_idx[day_change]
    day_labels = stitched["ts"].iloc[day_starts].dt.strftime("%m-%d").tolist()

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True,
                             gridspec_kw={"height_ratios": [2.2, 1, 1]})

    ax = axes[0]
    ax.plot(bar_idx, nav, color="#d6336c", lw=1.6, label=grpo_label)
    ax.plot(bar_idx, ew_nav, color="#1864ab", lw=1.4, label=baseline_label)
    ax.axhline(1.0, color="#aaa", ls="--")
    for d in day_starts[1:]:
        ax.axvline(d, color="#ccc", ls=":", lw=0.7)
    ax.set_ylabel("Cumulative NAV")
    title = (
        f"Long-only OOS  GRPO={stats.get('total_return', 0) * 100:+.2f}%  "
        f"EW={stats_eq.get('total_return', 0) * 100:+.2f}%  "
        f"Excess={(stats.get('total_return', 0) - stats_eq.get('total_return', 0)) * 100:+.2f}%"
    )
    if eval_summary:
        title += (
            f"  |  score-corr={eval_summary.get('score_return_corr', 0):+.3f}  "
            f"w-corr={eval_summary.get('weight_return_corr', 0):+.3f}"
        )
    ax.set_title(title)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(bar_idx, sh, color="#d6336c", lw=1.2, label="GRPO")
    ax.plot(bar_idx, sh_eq, color="#1864ab", lw=1.0, label="EW")
    ax.axhline(0, color="#aaa", ls="--")
    ax.set_ylabel(f"Rolling Sharpe (win={win})")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(bar_idx, excess_nav, color="#2b8a3e", lw=1.3, label="GRPO / EW (relative)")
    ax.axhline(1.0, color="#aaa", ls="--")
    ax.set_ylabel("Relative NAV (excess)")
    ax.set_xlabel("Trading bar index")
    ax.set_xticks(day_starts)
    ax.set_xticklabels(day_labels, rotation=30, ha="right")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def _grpo_label(cfg: WalkForwardConfig) -> str:
    if cfg.model_type == "linear":
        return f"Linear Factor Long-only Top{cfg.top_k}"
    prefix = "PatchTST GRPO" if cfg.model_type == "patchtst" else "GRPO"
    if cfg.portfolio_mode == "long_only":
        return f"{prefix} Long-only Top{cfg.top_k}"
    return f"{prefix} (Long-Short)"


def _filter_stitched_by_oos_month(
    stitched: pd.DataFrame,
    start_month: Optional[str],
    end_month: Optional[str],
) -> pd.DataFrame:
    if not start_month or not end_month:
        return stitched
    ts = pd.to_datetime(stitched["ts"])
    mask = (ts.dt.strftime("%Y-%m") >= start_month) & (ts.dt.strftime("%Y-%m") <= end_month)
    return stitched.loc[mask].reset_index(drop=True)


def replot_walkforward_curves(
    experiment_root: Path = EXPERIMENT_ROOT,
    cfg: WalkForwardConfig = WalkForwardConfig(),
    recompute_baseline: bool = False,
) -> None:
    """Regenerate ``walkforward_curves.png`` with baseline (no retrain)."""
    root = Path(experiment_root)
    bt_dir = root / "backtest"
    stitched_path = bt_dir / "stitched_bar_returns.csv"
    baseline_label = _baseline_label(cfg)

    meta0 = root / "folds"
    fold_dirs = sorted(meta0.glob("fold_*"))
    if not fold_dirs:
        raise FileNotFoundError(f"No folds under {root / 'folds'}")

    oos_frames = []
    for fold_dir in fold_dirs:
        meta = json.loads((fold_dir / "meta.json").read_text(encoding="utf-8"))
        stock_ids = meta["stock_ids"]
        fold_info = {
            "fold_idx": meta["fold_idx"],
            "oos_start_day": _td_idx(cfg, meta["oos_dates"][0]),
            "oos_num_days": len(meta["oos_dates"]),
        }
        pol_csv = fold_dir / "oos_bar_returns.csv"
        if not pol_csv.is_file():
            raise FileNotFoundError(f"Missing {pol_csv}")
        pol = pd.read_csv(pol_csv, parse_dates=["ts"])
        if recompute_baseline or "ew_net_pr" not in pol.columns:
            ew = _backtest_fold_equal_weight(fold_info, stock_ids, cfg)
            pol = pol.drop(columns=["ew_net_pr", "ew_nav"], errors="ignore")
            pol = pol.merge(ew[["ts", "ew_net_pr"]], on="ts", how="left")
            pol["ew_nav"] = (1.0 + pol["ew_net_pr"]).groupby(pol["fold"]).cumprod()
            pol.to_csv(pol_csv, index=False)
        oos_frames.append(pol)

    stitched = pd.concat(oos_frames, ignore_index=True)
    stitched.to_csv(stitched_path, index=False)

    overall = annualized_stats(stitched["net_pr"].to_numpy(), BARS_PER_YEAR)
    overall_eq = annualized_stats(stitched["ew_net_pr"].to_numpy(), BARS_PER_YEAR)
    plot_stitched(
        stitched, bt_dir / "walkforward_curves.png", overall, overall_eq,
        baseline_label=baseline_label,
        grpo_label=_grpo_label(cfg),
    )
    print(f"Replotted -> {bt_dir / 'walkforward_curves.png'}")
    print(f"  baseline: {baseline_label}")
    print(f"  GRPO : total={overall.get('total_return', 0) * 100:+.2f}%  "
          f"Sharpe={overall.get('annualized_sharpe', 0):+.2f}")
    print(f"  Base : total={overall_eq.get('total_return', 0) * 100:+.2f}%  "
          f"Sharpe={overall_eq.get('annualized_sharpe', 0):+.2f}")


def run_walkforward(cfg: WalkForwardConfig = WalkForwardConfig()) -> Dict:
    set_seed(cfg.seed)
    root = Path(cfg.experiment_root)
    bt_dir = root / "backtest"
    universe_dir = root / "universe"
    (root / "folds").mkdir(parents=True, exist_ok=True)
    bt_dir.mkdir(parents=True, exist_ok=True)
    universe_dir.mkdir(parents=True, exist_ok=True)

    reward_note = (
        "empirical fixed weights"
        if cfg.reward_mode == "empirical"
        else "auto-learned via val grid-search"
    )
    chain_note = (
        "linear factor baseline (no training; SFT teacher scores at inference)"
        if cfg.model_type == "linear"
        else "SFT on fold0 only; GRPO warm-starts from previous fold grpo_model.pt"
        if cfg.warm_chain
        else "independent SFT+GRPO per fold"
    )
    (root / "README.txt").write_text(
        f"Walk-forward GRPO experiment (Apr-May 2026).\n"
        f"Tag: {cfg.experiment_tag}\n"
        f"Reward: {reward_note}\n"
        f"Portfolio: {cfg.portfolio_mode}\n"
        f"Baseline: {cfg.baseline_mode}\n"
        f"Training: {chain_note}\n"
        f"Each sub-folder under folds/ keeps its own grpo_model.pt.\n"
        f"Do NOT mix with other experiment folders.\n",
        encoding="utf-8",
    )

    print("=" * 80)
    print(f"Experiment root: {root}")
    print(f"Reward mode    : {cfg.reward_mode}")
    print(f"Portfolio mode : {cfg.portfolio_mode}")
    print(f"Baseline mode  : {cfg.baseline_mode}")
    print(f"Model type     : {cfg.model_type}")
    print(f"Warm chain     : {cfg.warm_chain}")
    print("=" * 80)

    # 1) Top-30 universe by 2026 YTD (Jan-Mar)
    if cfg.stock_ids:
        stock_ids = cfg.stock_ids
    else:
        stock_ids, rank_df = select_top_n(
            N_STOCKS, cfg.pool_dir,
            start_date="2026-01-05", end_date="2026-03-31",
            save_path=universe_dir / "top30_ytd2026.csv",
        )
        print(f"Universe: top {len(stock_ids)} by 2026 YTD return")
        print(rank_df.head(10).to_string(index=False))

    folds = build_fold_schedule(cfg)
    sched_desc = (
        f"{cfg.train_days}d train -> {cfg.oos_days}d OOS (rolling daily)"
        if cfg.schedule_mode == "rolling_1d"
        else "~6mo train -> 1mo OOS (monthly slide)"
        if cfg.schedule_mode == "monthly_semiannual"
        else f"{cfg.train_days}d train -> {cfg.oos_days}d OOS"
    )
    print(f"\nScheduled {len(folds)} folds ({sched_desc})")
    for f in folds:
        print(f"  fold {f['fold_idx']:03d}: train {f['train_dates']} -> OOS {f['oos_dates']}")

    fold_summaries = []
    oos_frames = []
    prev_grpo_path: Optional[Path] = None
    shared_sft_dir = root / "shared_sft"

    for fold in folds:
        fold_dir = _fold_dir(root, fold["fold_idx"], fold["train_dates"])
        fold_dir.mkdir(parents=True, exist_ok=True)

        meta = {
            "fold_idx": fold["fold_idx"],
            "train_dates": fold["train_dates"],
            "oos_dates": fold["oos_dates"],
            "stock_ids": stock_ids,
            "experiment_root": str(root),
            "experiment_tag": cfg.experiment_tag,
            "reward_mode": cfg.reward_mode,
            "portfolio_mode": cfg.portfolio_mode,
            "baseline_mode": cfg.baseline_mode,
            "warm_chain": cfg.warm_chain,
            "model_type": cfg.model_type,
            "universe": "preset" if cfg.stock_ids else "top_ytd",
        }
        (fold_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

        print("\n" + "-" * 80)
        print(f"FOLD {fold['fold_idx']:03d}  train={fold['train_dates']}  OOS={fold['oos_dates']}")
        print(f"  artifacts -> {fold_dir}")
        print("-" * 80)

        if cfg.model_type == "linear":
            meta["init_source"] = "linear_factor_teacher"
            (fold_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
            print("  linear baseline: skip SFT/GRPO, use teacher factor scores directly")
            oos_df = backtest_fold_oos_linear(fold, fold_dir, stock_ids, cfg)
        else:
            # SFT (once on fold0 when warm_chain, else every fold)
            if cfg.warm_chain:
                if fold["fold_idx"] == 0:
                    shared_sft_dir.mkdir(parents=True, exist_ok=True)
                    data_start, data_days = _hf_data_span(cfg, fold, for_oos=False)
                    sft_cfg = SFTConfig(
                        pool_dir=cfg.pool_dir,
                        extra_pool_dirs=cfg.extra_pool_dirs,
                        start_day=data_start,
                        num_days=data_days,
                        lookback=cfg.lookback,
                        horizon=cfg.horizon,
                        stock_ids=stock_ids,
                        cache_tag=_cache_tag(cfg, fold["fold_idx"], "train"),
                        epochs=cfg.sft_epochs,
                        batch_size=cfg.sft_batch_size,
                        ckpt_dir=shared_sft_dir,
                        log_dir=shared_sft_dir,
                        model_type=cfg.model_type,
                        patch_len=cfg.patch_len,
                        patch_stride=cfg.patch_stride,
                        patch_e_layers=cfg.patch_e_layers,
                        patch_cs_layers=cfg.patch_cs_layers,
                        use_regime_features=cfg.use_regime_features,
                        regime_lookback_days=cfg.regime_lookback_days,
                        train_dates=fold["train_dates"],
                    )
                    train_sft(sft_cfg)
                    init_ckpt = shared_sft_dir / "best_sft_model.pt"
                    meta["init_source"] = "sft_fold0_shared"
                    print(f"  warm-chain: SFT done once -> {init_ckpt}")
                else:
                    if prev_grpo_path is None or not prev_grpo_path.is_file():
                        raise RuntimeError(
                            f"warm_chain fold {fold['fold_idx']}: missing previous grpo_model.pt"
                        )
                    init_ckpt = prev_grpo_path
                    meta["init_source"] = str(init_ckpt.relative_to(root))
                    print(f"  warm-chain: skip SFT, init GRPO from {init_ckpt.name}")
            else:
                data_start, data_days = _hf_data_span(cfg, fold, for_oos=False)
                sft_cfg = SFTConfig(
                    pool_dir=cfg.pool_dir,
                    extra_pool_dirs=cfg.extra_pool_dirs,
                    start_day=data_start,
                    num_days=data_days,
                    lookback=cfg.lookback,
                    horizon=cfg.horizon,
                    stock_ids=stock_ids,
                    cache_tag=_cache_tag(cfg, fold["fold_idx"], "train"),
                    epochs=cfg.sft_epochs,
                    batch_size=cfg.sft_batch_size,
                    ckpt_dir=fold_dir,
                    log_dir=fold_dir,
                    model_type=cfg.model_type,
                    patch_len=cfg.patch_len,
                    patch_stride=cfg.patch_stride,
                    patch_e_layers=cfg.patch_e_layers,
                    patch_cs_layers=cfg.patch_cs_layers,
                    use_regime_features=cfg.use_regime_features,
                    regime_lookback_days=cfg.regime_lookback_days,
                    train_dates=fold["train_dates"],
                )
                train_sft(sft_cfg)
                init_ckpt = fold_dir / "best_sft_model.pt"
                meta["init_source"] = "sft_per_fold"

            (fold_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

            # Reward weights: empirical fixed OR auto-learn on val
            if cfg.reward_mode == "empirical":
                reward_cfg = cfg.reward_config or empirical_reward_config(
                    top_k=cfg.top_k, bottom_k=cfg.bottom_k,
                )
                print(
                    f"  empirical reward: ic={reward_cfg.ic_weight:.2f} "
                    f"sharpe={reward_cfg.sharpe_weight:.2f} "
                    f"turnover={reward_cfg.turnover_weight:.2f} "
                    f"downside={reward_cfg.downside_weight:.2f} "
                    f"max_dd={reward_cfg.max_dd_weight:.2f} "
                    f"size={reward_cfg.size_exp_weight:.2f} "
                    f"beta={reward_cfg.beta_exp_weight:.2f} "
                    f"hhi={reward_cfg.hhi_weight:.2f}"
                )
            else:
                hf_cfg = _hf_config(
                    cfg, *_hf_data_span(cfg, fold, for_oos=False),
                    stock_ids, _cache_tag(cfg, fold["fold_idx"], "train"),
                )
                bundle = build_hf_dataset(hf_cfg)
                train_bars = bar_indices_for_dates(bundle, fold["train_dates"])
                splits = split_train_val_test(
                    bundle, 0.25, 0.1, cfg.horizon, candidate_indices=train_bars,
                )
                val_ds = HFEpisodeDataset(bundle, cfg.lookback, cfg.episode_len, splits["val"])
                val_loader = DataLoader(val_ds, batch_size=1, collate_fn=_collate_episodes)

                feat_dim = bundle["x_seq"].shape[-1]
                probe = _build_model(feat_dim, GRPOTrainConfig(lookback=cfg.lookback))
                probe.load_state_dict(torch.load(init_ckpt, map_location="cpu"))
                device = torch.device("cpu")
                probe.to(device)
                reward_cfg = calibrate_reward(
                    probe, val_loader, device,
                    _portfolio_config(cfg),
                    RewardConfig(top_k=cfg.top_k, bottom_k=cfg.bottom_k),
                    portfolio_mode=cfg.portfolio_mode,
                )
                print(
                    f"  learned reward weights: ic={reward_cfg.ic_weight:.2f} "
                    f"sharpe={reward_cfg.sharpe_weight:.2f} "
                    f"penalty={reward_cfg.turnover_weight:.2f}"
                )
            (fold_dir / "reward_weights.json").write_text(
                json.dumps(asdict(reward_cfg), indent=2), encoding="utf-8",
            )

            # GRPO -> save grpo_model.pt in fold_dir
            grpo_path = train_fold_grpo(fold, fold_dir, stock_ids, init_ckpt, cfg, reward_cfg)
            prev_grpo_path = grpo_path

            # OOS backtest with THIS fold's model
            oos_df = backtest_fold_oos(fold, fold_dir, grpo_path, stock_ids, cfg)

        oos_path = fold_dir / "oos_bar_returns.csv"
        oos_df.to_csv(oos_path, index=False)
        oos_frames.append(oos_df)

        st = annualized_stats(oos_df["net_pr"].to_numpy(), BARS_PER_YEAR)
        st_ew = annualized_stats(oos_df["ew_net_pr"].to_numpy(), BARS_PER_YEAR)
        fold_row = {
            "fold": fold["fold_idx"],
            "train_start": fold["train_dates"][0],
            "train_end": fold["train_dates"][-1],
            "oos_start": fold["oos_dates"][0],
            "oos_end": fold["oos_dates"][-1],
            "grpo_ckpt": "linear_factor_teacher" if cfg.model_type == "linear"
            else str(grpo_path.relative_to(root)),
            "total_return": st.get("total_return", 0),
            "sharpe": st.get("annualized_sharpe", 0),
            "max_dd": st.get("max_drawdown", 0),
        }
        if cfg.eval_extended:
            fold_row["ew_total_return"] = st_ew.get("total_return", 0)
            fold_row["excess_return"] = fold_row["total_return"] - fold_row["ew_total_return"]
            if "score_return_corr" in oos_df.columns:
                fold_row["score_return_corr"] = float(oos_df["score_return_corr"].iloc[0])
                fold_row["weight_return_corr"] = float(oos_df["weight_return_corr"].iloc[0])
        fold_summaries.append(fold_row)
        print(f"  OOS fold total={st.get('total_return',0)*100:+.3f}%  "
              f"Sharpe={st.get('annualized_sharpe',0):+.2f}")

    # Stitch all OOS segments
    stitched = pd.concat(oos_frames, ignore_index=True)
    stitched.to_csv(bt_dir / "stitched_bar_returns.csv", index=False)
    summary_df = pd.DataFrame(fold_summaries)
    summary_df.to_csv(bt_dir / "fold_summary.csv", index=False)

    plot_df = _filter_stitched_by_oos_month(
        stitched, cfg.oos_stitch_start, cfg.oos_stitch_end,
    )
    if len(plot_df) < len(stitched):
        tag = f"{cfg.oos_stitch_start.replace('-', '')}_{cfg.oos_stitch_end.replace('-', '')}"
        plot_path = bt_dir / f"walkforward_curves_{tag}.png"
        plot_df.to_csv(bt_dir / f"stitched_oos_{tag}.csv", index=False)
    else:
        plot_path = bt_dir / "walkforward_curves.png"

    overall = annualized_stats(plot_df["net_pr"].to_numpy(), BARS_PER_YEAR)
    overall_eq = annualized_stats(plot_df["ew_net_pr"].to_numpy(), BARS_PER_YEAR)
    if cfg.eval_extended:
        from eval_metrics import summarize_oos_comparison
        eval_summary = summarize_oos_comparison(plot_df)
        pd.DataFrame([eval_summary]).to_csv(bt_dir / "overall_eval.csv", index=False)
        plot_longonly_extended(
            plot_df, plot_path, overall, overall_eq, eval_summary,
            grpo_label=_grpo_label(cfg),
            baseline_label=_baseline_label(cfg),
        )
    else:
        plot_stitched(
            plot_df, plot_path, overall, overall_eq,
            baseline_label=_baseline_label(cfg),
            grpo_label=_grpo_label(cfg),
        )
    # also save full-range plot when filtered
    if len(plot_df) < len(stitched):
        overall_all = annualized_stats(stitched["net_pr"].to_numpy(), BARS_PER_YEAR)
        overall_eq_all = annualized_stats(stitched["ew_net_pr"].to_numpy(), BARS_PER_YEAR)
        plot_stitched(
            stitched, bt_dir / "walkforward_curves_full.png", overall_all, overall_eq_all,
            baseline_label=_baseline_label(cfg),
            grpo_label=_grpo_label(cfg),
        )

    print("\n" + "=" * 80)
    print("WALK-FORWARD COMPLETE")
    print(f"  experiment root : {root}")
    ckpt_note = (
        "linear baseline (no grpo_model.pt)"
        if cfg.model_type == "linear"
        else f"{len(folds)} (each with grpo_model.pt)"
    )
    print(f"  folds saved     : {ckpt_note}")
    print(f"  stitched OOS    : {len(stitched)} bars")
    print(f"  overall total   : {overall.get('total_return',0)*100:+.3f}%")
    print(f"  overall Sharpe  : {overall.get('annualized_sharpe',0):+.2f}")
    print(f"  EW total        : {overall_eq.get('total_return',0)*100:+.3f}%")
    print(f"  EW Sharpe       : {overall_eq.get('annualized_sharpe',0):+.2f}")
    if cfg.eval_extended:
        ex = overall.get('total_return', 0) - overall_eq.get('total_return', 0)
        print(f"  Excess return   : {ex*100:+.3f}%")
        if (bt_dir / "overall_eval.csv").is_file():
            ev = pd.read_csv(bt_dir / "overall_eval.csv").iloc[0]
            print(f"  Score-ret corr  : {ev.get('score_return_corr', 0):+.4f}")
            print(f"  Weight-ret corr : {ev.get('weight_return_corr', 0):+.4f}")
    print(f"  curves          : {plot_path}")
    if len(plot_df) < len(stitched):
        print(f"  full curves     : {bt_dir / 'walkforward_curves_full.png'}")
    print("=" * 80)

    return {
        "experiment_root": root,
        "fold_summaries": summary_df,
        "stitched": stitched,
        "overall_stats": overall,
    }


if __name__ == "__main__":
    run_walkforward()
