"""
GRPO post-training for ``StockTemporalTransformer`` with a TRL-style API.

What "TRL-style" means here
---------------------------
``trl.GRPOTrainer`` is hard-wired to causal-LM generation (it samples
completions through ``model.generate`` over a tokenizer and computes
per-token log-probs). Our policy is a regression-style scorer that maps
``(N, L, F) -> (N,) in [0,1]`` and takes actions on the simplex, so the
literal trainer does not fit. Instead we:

* import ``trl.GRPOConfig`` and reuse its hyperparameter names
  (``num_generations``, ``epsilon``, ``beta``, ``learning_rate``, ...);
* keep TRL's exact GRPO loss formulation (PPO-clipped policy gradient
  + KL-to-reference using the k3 estimator, ratio computed in log-space);
* swap "completions" for sampled portfolio-weight trajectories on episodes
  of M consecutive bars, with reward = IC + Sharpe - drawdown.

Episode rollout
---------------
For one episode:

    x : (M, N, L, F)   features for M consecutive cross sections
    r : (M, N)         realized log-returns over the next H bars
                       (the "ground truth" return at each step).

The policy outputs scores -> Dirichlet concentration -> samples G
alternative weight trajectories (G, M, N). Each trajectory is then
scored:

    portfolio_returns = (weights * r).sum(dim=-1)        # (G, M)
    sharpe   = mean / std along M
    max_dd   = max drawdown of cumulative returns
    ic_per_bar = corr(weights, r) at each bar
    reward   = w_ic * mean(ic_per_bar) + w_sharpe * sharpe - w_dd * max_dd

GRPO advantage normalizes rewards across G inside each episode.
"""

from __future__ import annotations

import copy
import csv
import math
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Dirichlet
from torch.utils.data import DataLoader

try:
    from trl import GRPOConfig
except ImportError:
    GRPOConfig = None  # type: ignore[misc, assignment]

from hf_data import (
    DEFAULT_POOL_DIR,
    HFConfig,
    HFEpisodeDataset,
    build_hf_dataset,
    split_train_val_test,
)
from stock_transformer import StockTemporalTransformer
from model_factory import build_scorer, model_build_config_from_grpo


HERE = Path(__file__).resolve().parent

EpisodeRewardFn = Callable[..., torch.Tensor]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class GRPOTrainConfig:
    """Wraps a ``trl.GRPOConfig`` and adds finance / data fields.

    The TRL part exposes (``num_generations``, ``epsilon``, ``beta``,
    ``learning_rate``, ``temperature``, ``loss_type``, ``scale_rewards``)
    so swapping in TRL's own GRPOTrainer in the future is a small lift.
    """

    # ── data / model shape ────────────────────────────────────────────
    pool_dir: Path = DEFAULT_POOL_DIR
    start_day: int = 0
    num_days: int = 3
    lookback: int = 30
    horizon: int = 5
    n_stocks_max: Optional[int] = None
    episode_len: int = 12

    d_model: int = 64
    num_heads: int = 4
    ffn_dim: int = 128
    dropout: float = 0.1

    sft_ckpt: Path = HERE / "checkpoints" / "best_sft_model.pt"

    # ── policy parametrization ────────────────────────────────────────
    alpha_base: float = 1.0
    alpha_scale: float = 10.0

    # ── optimization (mirrors trl.GRPOConfig keys where applicable) ───
    epochs: int = 4
    batch_size: int = 4
    lr: float = 1e-5
    weight_decay: float = 1e-2
    warmup_ratio: float = 0.05
    grad_clip: float = 1.0

    # ── trl GRPO loss knobs ───────────────────────────────────────────
    num_generations: int = 8
    epsilon: float = 0.2          # PPO clip range
    epsilon_high: Optional[float] = None
    beta: float = 0.04            # KL-to-reference coefficient
    temperature: float = 1.0      # action sampling temperature on Dirichlet
    scale_rewards: str = "group"  # group | none

    # ── episode reward weights (IC + Sharpe + drawdown) ───────────────
    ic_weight: float = 1.0
    sharpe_weight: float = 0.5
    dd_weight: float = 0.5

    # ── runtime ───────────────────────────────────────────────────────
    val_ratio: float = 0.2
    test_ratio: float = 0.2
    seed: int = 42
    ckpt_dir: Path = HERE / "checkpoints"
    log_dir: Path = HERE / "logs"

    model_type: str = "transformer"
    patch_len: int = 6
    patch_stride: int = 6
    patch_e_layers: int = 2
    patch_cs_layers: int = 1

    def to_trl_config(self) -> Optional["GRPOConfig"]:
        """Best-effort projection onto a real ``trl.GRPOConfig`` instance."""
        if GRPOConfig is None:
            return None
        try:
            return GRPOConfig(
                output_dir=str(self.ckpt_dir),
                learning_rate=self.lr,
                per_device_train_batch_size=self.batch_size,
                num_generations=self.num_generations,
                epsilon=self.epsilon,
                epsilon_high=self.epsilon_high,
                beta=self.beta,
                temperature=self.temperature,
                scale_rewards=self.scale_rewards,
                num_train_epochs=float(self.epochs),
                seed=self.seed,
            )
        except Exception as e:
            print(f"[trl] GRPOConfig instantiation skipped: {e.__class__.__name__}")
            return None


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def scores_to_alpha(scores: torch.Tensor, base: float, scale: float, temperature: float) -> torch.Tensor:
    """Map [0,1] scores to Dirichlet concentration; lower temperature -> sharper."""
    s = scores.clamp(0.0, 1.0)
    return (base + scale * s) / max(temperature, 1e-6)


def build_dirichlet(
    model: nn.Module, x: torch.Tensor, cfg: GRPOTrainConfig
) -> Dirichlet:
    """Forward the policy on x: (..., N, L, F) and return a batched Dirichlet."""
    scores = model(x)  # supports (N,L,F), (B,N,L,F), (M,N,L,F), ...
    alpha = scores_to_alpha(scores, cfg.alpha_base, cfg.alpha_scale, cfg.temperature)
    return Dirichlet(alpha)


# ---------------------------------------------------------------------------
# Reward (IC + Sharpe - drawdown), episode level
# ---------------------------------------------------------------------------
def episode_reward(
    weights: torch.Tensor,
    realized_returns: torch.Tensor,
    cfg: GRPOTrainConfig,
) -> Dict[str, torch.Tensor]:
    """
    Args:
        weights:           (G, M, N) sampled portfolio weights, simplex per row.
        realized_returns:  (M, N)    realized H-bar log returns at each step.

    Returns dict with:
        reward       : (G,) total scalar reward per rollout
        ic           : (G,) mean per-bar Pearson(weights, returns)
        sharpe       : (G,)
        max_dd       : (G,)
        portfolio_pr : (G, M)
    """
    G, M, N = weights.shape
    r = realized_returns.unsqueeze(0)                                # (1, M, N)

    # Per-bar Pearson correlation between sampled weights and realized returns.
    w_centered = weights - weights.mean(dim=-1, keepdim=True)
    r_centered = r - r.mean(dim=-1, keepdim=True)
    num = (w_centered * r_centered).sum(dim=-1)                      # (G, M)
    den = w_centered.norm(dim=-1) * r_centered.norm(dim=-1) + 1e-8
    ic_per_bar = num / den
    ic = ic_per_bar.mean(dim=1)                                      # (G,)

    portfolio_pr = (weights * r).sum(dim=-1)                         # (G, M)
    sharpe = portfolio_pr.mean(dim=1) / (portfolio_pr.std(dim=1) + 1e-8)

    cum_pr = portfolio_pr.cumsum(dim=1)                              # (G, M)
    running_max = cum_pr.cummax(dim=1).values
    drawdown = running_max - cum_pr
    max_dd = drawdown.max(dim=1).values                              # (G,)

    reward = (
        cfg.ic_weight * ic
        + cfg.sharpe_weight * sharpe
        - cfg.dd_weight * max_dd
    )
    return {
        "reward": reward,
        "ic": ic,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "portfolio_pr": portfolio_pr,
    }


# ---------------------------------------------------------------------------
# GRPO loss (TRL formulation, exactly)
# ---------------------------------------------------------------------------
def compute_advantages(rewards: torch.Tensor, scale: str) -> torch.Tensor:
    """rewards: (G,). Returns standardized advantages (G,)."""
    mean = rewards.mean()
    if scale == "none":
        return rewards - mean
    return (rewards - mean) / (rewards.std(unbiased=False) + 1e-8)


def grpo_loss(
    log_prob_new: torch.Tensor,
    log_prob_old: torch.Tensor,
    log_prob_ref: torch.Tensor,
    advantages: torch.Tensor,
    cfg: GRPOTrainConfig,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    All inputs are (G,) — episode-level log-probs (sum over M bars).

    Loss:
        pg = -mean( min(ratio * A, clip(ratio, 1-eps, 1+eps_high) * A) )
        kl = mean( exp(ref - new) - (ref - new) - 1 )      # k3 estimator
        total = pg + beta * kl
    """
    log_ratio = log_prob_new - log_prob_old
    ratio = log_ratio.exp()

    eps_low = cfg.epsilon
    eps_high = cfg.epsilon_high if cfg.epsilon_high is not None else cfg.epsilon
    unclipped = ratio * advantages
    clipped = ratio.clamp(1.0 - eps_low, 1.0 + eps_high) * advantages
    pg = -torch.min(unclipped, clipped).mean()

    log_ratio_ref = log_prob_ref - log_prob_new
    kl = (log_ratio_ref.exp() - log_ratio_ref - 1.0).mean()

    loss = pg + cfg.beta * kl
    metrics = {
        "loss": loss.item(),
        "pg_loss": pg.item(),
        "kl": kl.item(),
        "ratio_mean": ratio.mean().item(),
        "ratio_std": ratio.std(unbiased=False).item(),
    }
    return loss, metrics


# ---------------------------------------------------------------------------
# One GRPO step (one episode batch)
# ---------------------------------------------------------------------------
def grpo_step(
    policy: nn.Module,
    ref_policy: nn.Module,
    x_episodes: torch.Tensor,    # (B, M, N, L, F)
    r_episodes: torch.Tensor,    # (B, M, N)
    optimizer: optim.Optimizer,
    scheduler: Optional[optim.lr_scheduler.LambdaLR],
    cfg: GRPOTrainConfig,
) -> Dict[str, float]:
    """
    Process B episodes independently. Each episode produces G rollouts;
    advantages are normalized within the G rollouts of that episode.
    """
    B = x_episodes.shape[0]
    metrics_acc: Dict[str, float] = {}

    optimizer.zero_grad()

    for b in range(B):
        x_ep = x_episodes[b]     # (M, N, L, F)
        r_ep = r_episodes[b]     # (M, N)

        # Single forward through the policy on the M cross sections.
        dist_new = build_dirichlet(policy, x_ep, cfg)            # batched over M

        with torch.no_grad():
            weights_gM = dist_new.sample((cfg.num_generations,)) # (G, M, N)
        log_prob_new = dist_new.log_prob(weights_gM)             # (G, M)
        log_prob_old = log_prob_new.detach()                      # (G, M)

        with torch.no_grad():
            dist_ref = build_dirichlet(ref_policy, x_ep, cfg)
            log_prob_ref = dist_ref.log_prob(weights_gM)         # (G, M)
            rew_dict = episode_reward(weights_gM, r_ep, cfg)
            rewards = rew_dict["reward"]                          # (G,)
            advantages = compute_advantages(rewards, cfg.scale_rewards)

        # Episode-level log-probs (sum across M bars)
        sum_lp_new = log_prob_new.sum(dim=1)
        sum_lp_old = log_prob_old.sum(dim=1)
        sum_lp_ref = log_prob_ref.sum(dim=1)

        loss, m = grpo_loss(sum_lp_new, sum_lp_old, sum_lp_ref, advantages, cfg)
        (loss / B).backward()

        with torch.no_grad():
            for k, v in m.items():
                metrics_acc[k] = metrics_acc.get(k, 0.0) + v
            metrics_acc["reward_mean"] = metrics_acc.get("reward_mean", 0.0) + rewards.mean().item()
            metrics_acc["ic_mean"] = metrics_acc.get("ic_mean", 0.0) + rew_dict["ic"].mean().item()
            metrics_acc["sharpe_mean"] = metrics_acc.get("sharpe_mean", 0.0) + rew_dict["sharpe"].mean().item()
            metrics_acc["max_dd_mean"] = metrics_acc.get("max_dd_mean", 0.0) + rew_dict["max_dd"].mean().item()
            metrics_acc["adv_abs_mean"] = metrics_acc.get("adv_abs_mean", 0.0) + advantages.abs().mean().item()

    nn.utils.clip_grad_norm_(policy.parameters(), cfg.grad_clip)
    optimizer.step()
    if scheduler is not None:
        scheduler.step()

    return {k: v / B for k, v in metrics_acc.items()}


# ---------------------------------------------------------------------------
# Trainer entry point
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


def _build_model(feat_dim: int, cfg: GRPOTrainConfig) -> nn.Module:
    mb = model_build_config_from_grpo(cfg, feat_dim)
    return build_scorer(mb)


def _load_sft_weights(model: nn.Module, ckpt: Path) -> None:
    if ckpt.is_file():
        model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        print(f"loaded SFT checkpoint: {ckpt}")
    else:
        print(f"WARNING: SFT checkpoint not found at {ckpt}; starting from random init.")


def _collate_episodes(items):
    xs = torch.stack([it[0] for it in items], dim=0)
    rs = torch.stack([it[1] for it in items], dim=0)
    return xs, rs


def train_grpo(cfg: GRPOTrainConfig = GRPOTrainConfig()) -> nn.Module:
    set_seed(cfg.seed)
    cfg.ckpt_dir.mkdir(parents=True, exist_ok=True)
    cfg.log_dir.mkdir(parents=True, exist_ok=True)

    hf_cfg = HFConfig(
        pool_dir=cfg.pool_dir,
        start_day=cfg.start_day,
        num_days=cfg.num_days,
        lookback=cfg.lookback,
        horizon=cfg.horizon,
        n_stocks_max=cfg.n_stocks_max,
    )
    bundle = build_hf_dataset(hf_cfg)
    splits = split_train_val_test(
        bundle, val_ratio=cfg.val_ratio, test_ratio=cfg.test_ratio, embargo=cfg.horizon
    )

    train_ds = HFEpisodeDataset(bundle, cfg.lookback, cfg.episode_len, splits["train"])
    val_ds = HFEpisodeDataset(bundle, cfg.lookback, cfg.episode_len, splits["val"])

    if len(train_ds) == 0:
        raise RuntimeError("Empty train episodes. Reduce episode_len / horizon / lookback.")

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=_collate_episodes
    )

    feat_dim = bundle["x_seq"].shape[-1]
    n_stocks = bundle["x_seq"].shape[1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = _build_model(feat_dim, cfg).to(device)
    _load_sft_weights(policy, cfg.sft_ckpt)

    ref_policy = copy.deepcopy(policy).to(device)
    for p in ref_policy.parameters():
        p.requires_grad_(False)
    ref_policy.eval()

    optimizer = optim.AdamW(
        policy.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        betas=(0.9, 0.999),
    )
    total_steps = max(cfg.epochs * math.ceil(len(train_ds) / cfg.batch_size), 1)
    warmup_steps = max(int(total_steps * cfg.warmup_ratio), 1)
    scheduler = get_scheduler(optimizer, total_steps, warmup_steps)

    log_path = cfg.log_dir / "grpo_log.csv"
    log_fields = [
        "epoch", "step", "loss", "pg_loss", "kl",
        "reward_mean", "ic_mean", "sharpe_mean", "max_dd_mean",
        "adv_abs_mean", "ratio_mean", "lr",
    ]
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(log_fields)

    print(f"device       : {device}")
    print(f"data shape   : N={n_stocks}, F={feat_dim}, L={cfg.lookback}, H={cfg.horizon}")
    print(f"episode len  : M={cfg.episode_len}")
    print(f"episodes     : train={len(train_ds)} val={len(val_ds)}")
    print(f"GRPO config  : G={cfg.num_generations} eps={cfg.epsilon} beta={cfg.beta} temp={cfg.temperature}")
    print(f"reward       : {cfg.ic_weight}*IC + {cfg.sharpe_weight}*Sharpe - {cfg.dd_weight}*MaxDD")
    trl_cfg = cfg.to_trl_config()
    if trl_cfg is not None:
        print(f"trl GRPOConfig: live, G={trl_cfg.num_generations}")
    else:
        print("trl GRPOConfig: skipped (torch<2.4); using TRL-compatible local config.")
    print("-" * 80)

    global_step = 0
    for epoch in range(1, cfg.epochs + 1):
        policy.train()
        ep_metrics: Dict[str, float] = {}
        n_steps = 0
        for x_eps, r_eps in train_loader:
            x_eps = x_eps.to(device)
            r_eps = r_eps.to(device)
            m = grpo_step(policy, ref_policy, x_eps, r_eps, optimizer, scheduler, cfg)
            global_step += 1
            n_steps += 1
            for k, v in m.items():
                ep_metrics[k] = ep_metrics.get(k, 0.0) + v
            with open(log_path, "a", newline="") as f:
                csv.writer(f).writerow([
                    epoch, global_step, m["loss"], m["pg_loss"], m["kl"],
                    m.get("reward_mean", 0.0), m.get("ic_mean", 0.0),
                    m.get("sharpe_mean", 0.0), m.get("max_dd_mean", 0.0),
                    m.get("adv_abs_mean", 0.0), m.get("ratio_mean", 0.0),
                    scheduler.get_last_lr()[0],
                ])
        avg = {k: v / n_steps for k, v in ep_metrics.items()}
        print(
            f"epoch {epoch:03d}/{cfg.epochs} "
            f"loss={avg['loss']:.4f} pg={avg['pg_loss']:.4f} kl={avg['kl']:.4f} | "
            f"reward={avg['reward_mean']:.4f} ic={avg['ic_mean']:.4f} "
            f"sharpe={avg['sharpe_mean']:.4f} dd={avg['max_dd_mean']:.4f} | "
            f"lr={scheduler.get_last_lr()[0]:.2e}"
        )

    torch.save(policy.state_dict(), cfg.ckpt_dir / "last_grpo_model.pt")
    print("-" * 80)
    print("GRPO training done.")
    return policy


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def policy_weights(
    x: torch.Tensor,
    cfg: GRPOTrainConfig = GRPOTrainConfig(),
    ckpt: Path = HERE / "checkpoints" / "last_grpo_model.pt",
    stochastic: bool = False,
) -> torch.Tensor:
    """Return portfolio weights for an input x of shape (N, L, F)."""
    feat_dim = x.shape[-1]
    model = _build_model(feat_dim, cfg)
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval()
    with torch.no_grad():
        scores = model(x.float())
        alpha = scores_to_alpha(scores, cfg.alpha_base, cfg.alpha_scale, cfg.temperature)
        if stochastic:
            return Dirichlet(alpha).sample()
        return alpha / alpha.sum()


if __name__ == "__main__":
    train_grpo()
