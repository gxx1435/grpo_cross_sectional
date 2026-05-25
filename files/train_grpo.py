"""
GRPO post-training for StockTransformer (cross-sectional portfolio policy).

Why we do not call ``trl.GRPOTrainer`` directly
-----------------------------------------------
TRL's GRPOTrainer is tightly wired to causal LMs: it samples actions via
``model.generate`` over a tokenizer, scores prompts with text reward
functions, and tracks per-token log-probs. Our StockTransformer is a
regression-style scorer over a stock cross section, so plugging it into
GRPOTrainer is more painful than helpful. Instead, this file mirrors TRL's
GRPO algorithm faithfully:

    1. Sample a group of G completions per "prompt" (= cross section).
    2. Score each completion with a reward function.
    3. Compute group-wise normalized advantages.
    4. PPO-style clipped policy gradient.
    5. KL penalty against a frozen reference policy (here: the SFT model).

The only thing different from TRL is the action space:

    - Action  := portfolio weight vector w in the (N-1)-simplex.
    - Policy  := Dirichlet(alpha) where
                 alpha_i = alpha_base + alpha_scale * sigmoid_score_i.
    - Higher SFT score => higher expected weight on that stock.

If TRL is installed we still import ``GRPOConfig`` for hyperparameter
naming consistency; otherwise we fall back to a local dataclass.

Reward
------
``default_reward_fn`` is a placeholder returning

    reward = portfolio_return + entropy_bonus * normalized_entropy

where ``portfolio_return = w . labels`` and ``labels`` is the SFT label.
Replace this with your real reward when ready.
"""

import copy
import csv
import math
import os
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Dirichlet
from torch.utils.data import DataLoader

from stock_transformer import StockTransformer
from train_sft import (
    Config as SFTConfig,
    StockSFTDataset,
    load_samples,
    make_placeholder_samples,
)


RewardFn = Callable[..., torch.Tensor]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class GRPOConfig:
    """GRPO hyperparameters. Names follow trl.GRPOConfig where possible."""

    # Data shape (must match SFT).
    feat_dim: int = 64
    num_stocks: int = 500
    num_samples: int = 512

    # Model (must match SFT to load the checkpoint).
    d_model: int = 128
    num_heads: int = 4
    ffn_dim: int = 256
    dropout: float = 0.1

    # Policy parametrization: alpha = alpha_base + alpha_scale * score.
    alpha_base: float = 1.0
    alpha_scale: float = 10.0

    # GRPO sampling / loss.
    num_generations: int = 8       # G: completions per cross section.
    cliprange: float = 0.2         # PPO clip range epsilon.
    kl_coef: float = 0.04          # beta: KL penalty vs reference.
    entropy_bonus: float = 0.05    # weight of the entropy term in default reward.

    # Optimization.
    epochs: int = 20
    batch_size: int = 4
    lr: float = 1e-5               # smaller than SFT lr — RL is fragile.
    weight_decay: float = 1e-2
    warmup_ratio: float = 0.05
    grad_clip: float = 1.0

    # Runtime.
    seed: int = 42
    sft_ckpt: str = "checkpoints/best_sft_model.pt"
    ckpt_dir: str = "checkpoints"
    log_dir: str = "logs"

    # Optional: hand a fresh SFTConfig for dataset construction.
    sft_for_data: SFTConfig = field(default_factory=SFTConfig)


cfg = GRPOConfig()


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def scores_to_alpha(scores: torch.Tensor, base: float, scale: float) -> torch.Tensor:
    """Map model scores in [0, 1] to Dirichlet concentration parameters."""
    return base + scale * scores.clamp(min=0.0, max=1.0)


def build_dirichlet(
    model: nn.Module,
    x: torch.Tensor,
    base: float,
    scale: float,
) -> Dirichlet:
    """Forward the policy on x: (B, N, F) and return a batched Dirichlet."""
    scores = model(x)  # (B, N) thanks to StockTransformer's batched path.
    alpha = scores_to_alpha(scores, base, scale)
    return Dirichlet(alpha)


# ---------------------------------------------------------------------------
# Default reward
# ---------------------------------------------------------------------------
def default_reward_fn(
    weights: torch.Tensor,
    features: torch.Tensor,
    labels: torch.Tensor,
    config: GRPOConfig,
    **_: Dict,
) -> torch.Tensor:
    """
    Initial reward: long-only portfolio "return" plus a small entropy bonus
    so the policy doesn't collapse into a single-stock bet.

    Args:
        weights:  (B, G, N) sampled weights, each row a simplex vector.
        features: (B, N, F) the inputs (unused here; kept for the hook).
        labels:   (B, N)    SFT-style targets in [0, 1].

    Returns:
        rewards: (B, G)
    """
    del features  # unused in the default reward.
    portfolio_return = (weights * labels.unsqueeze(1)).sum(dim=-1)  # (B, G)

    eps = 1e-8
    entropy = -(weights * (weights + eps).log()).sum(dim=-1)        # (B, G)
    max_entropy = math.log(weights.size(-1))
    norm_entropy = entropy / max_entropy

    return portfolio_return + config.entropy_bonus * norm_entropy


# ---------------------------------------------------------------------------
# GRPO loss
# ---------------------------------------------------------------------------
def compute_advantages(rewards: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Group-wise standardized advantages (per cross section, across G)."""
    mean = rewards.mean(dim=-1, keepdim=True)
    std = rewards.std(dim=-1, keepdim=True)
    return (rewards - mean) / (std + eps)


def grpo_loss(
    log_prob_new: torch.Tensor,
    log_prob_old: torch.Tensor,
    log_prob_ref: torch.Tensor,
    advantages: torch.Tensor,
    cliprange: float,
    kl_coef: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    PPO-clipped policy gradient + KL-to-reference, exactly as in TRL's GRPO,
    but using one log-prob per sampled action (not per token).
    """
    log_ratio = log_prob_new - log_prob_old
    ratio = log_ratio.exp()

    unclipped = ratio * advantages
    clipped = ratio.clamp(1.0 - cliprange, 1.0 + cliprange) * advantages
    pg_loss = -torch.min(unclipped, clipped).mean()

    # k3 unbiased KL estimator used by TRL: exp(ref - new) - (ref - new) - 1.
    log_ratio_ref = log_prob_ref - log_prob_new
    kl = (log_ratio_ref.exp() - log_ratio_ref - 1.0).mean()

    loss = pg_loss + kl_coef * kl

    metrics = {
        "loss": loss.item(),
        "pg_loss": pg_loss.item(),
        "kl": kl.item(),
        "ratio_mean": ratio.mean().item(),
        "ratio_std": ratio.std(unbiased=False).item(),
    }
    return loss, metrics


# ---------------------------------------------------------------------------
# One GRPO step
# ---------------------------------------------------------------------------
def grpo_step(
    policy: nn.Module,
    ref_policy: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    optimizer: optim.Optimizer,
    scheduler: Optional[optim.lr_scheduler.LambdaLR],
    config: GRPOConfig,
    reward_fn: RewardFn,
) -> Dict[str, float]:
    """One GRPO update on a (B, N, F) batch with labels (B, N).

    We do *one* forward through the policy and reuse it for both sampling
    and the policy-gradient log-prob. With a single inner update step this
    matches TRL's GRPO (ratio == 1 at the start of the step) and avoids the
    dropout-mismatch issue that two separate forwards would cause.
    """
    G = config.num_generations

    # 1. Single forward through the policy (with grad).
    dist_new = build_dirichlet(policy, x, config.alpha_base, config.alpha_scale)

    # 2. Sample G actions per cross section (no grad through sampling).
    with torch.no_grad():
        weights_gbn = dist_new.sample((G,))            # (G, B, N)
    weights = weights_gbn.permute(1, 0, 2)             # (B, G, N)

    # 3. Log-probs.
    log_prob_new = dist_new.log_prob(weights_gbn).permute(1, 0)  # (B, G), grad
    log_prob_old = log_prob_new.detach()                          # (B, G)

    with torch.no_grad():
        dist_ref = build_dirichlet(
            ref_policy, x, config.alpha_base, config.alpha_scale
        )
        log_prob_ref = dist_ref.log_prob(weights_gbn).permute(1, 0)   # (B, G)

        rewards = reward_fn(weights=weights, features=x, labels=y, config=config)  # (B, G)
        advantages = compute_advantages(rewards)

    # 4. Loss + update.
    loss, metrics = grpo_loss(
        log_prob_new=log_prob_new,
        log_prob_old=log_prob_old,
        log_prob_ref=log_prob_ref,
        advantages=advantages,
        cliprange=config.cliprange,
        kl_coef=config.kl_coef,
    )

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(policy.parameters(), config.grad_clip)
    optimizer.step()
    if scheduler is not None:
        scheduler.step()

    metrics["reward_mean"] = rewards.mean().item()
    metrics["reward_std"] = rewards.std(unbiased=False).item()
    metrics["adv_abs_mean"] = advantages.abs().mean().item()
    return metrics


# ---------------------------------------------------------------------------
# Trainer entry point
# ---------------------------------------------------------------------------
def _make_dataset(config: GRPOConfig) -> StockSFTDataset:
    """Reuse the SFT dataset: GRPO consumes (x, y) the same way."""
    samples: Optional[Sequence[Tuple[torch.Tensor, torch.Tensor]]] = load_samples()
    if samples is None:
        samples = make_placeholder_samples(
            num_samples=config.num_samples,
            num_stocks=config.num_stocks,
            feat_dim=config.feat_dim,
            seed=config.seed,
        )
        print("Using placeholder GRPO samples. Replace load_samples() for real data.")
    return StockSFTDataset(samples)


def _build_model(config: GRPOConfig) -> StockTransformer:
    return StockTransformer(
        feat_dim=config.feat_dim,
        d_model=config.d_model,
        num_heads=config.num_heads,
        ffn_dim=config.ffn_dim,
        dropout=config.dropout,
    )


def _load_sft_weights(model: StockTransformer, ckpt_path: str) -> None:
    if os.path.isfile(ckpt_path):
        state = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(state)
        print(f"Loaded SFT checkpoint: {ckpt_path}")
    else:
        print(
            f"SFT checkpoint not found at {ckpt_path}. "
            "Starting GRPO from a randomly initialized model."
        )


def get_scheduler(
    optimizer: optim.Optimizer,
    total_steps: int,
    warmup_steps: int,
) -> optim.lr_scheduler.LambdaLR:
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_grpo(
    config: GRPOConfig = cfg,
    reward_fn: RewardFn = default_reward_fn,
) -> StockTransformer:
    set_seed(config.seed)
    os.makedirs(config.ckpt_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)

    dataset = _make_dataset(config)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy = _build_model(config).to(device)
    _load_sft_weights(policy, config.sft_ckpt)

    ref_policy = copy.deepcopy(policy).to(device)
    for p in ref_policy.parameters():
        p.requires_grad_(False)
    ref_policy.eval()

    optimizer = optim.AdamW(
        policy.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.999),
    )
    total_steps = config.epochs * math.ceil(len(dataset) / config.batch_size)
    warmup_steps = int(total_steps * config.warmup_ratio)
    scheduler = get_scheduler(optimizer, total_steps, warmup_steps)

    log_path = os.path.join(config.log_dir, "grpo_log.csv")
    log_fields = [
        "epoch", "step", "loss", "pg_loss", "kl",
        "reward_mean", "reward_std", "adv_abs_mean",
        "ratio_mean", "ratio_std", "lr",
    ]
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(log_fields)

    print(f"device      : {device}")
    print(f"data shape  : N={config.num_stocks}, F={config.feat_dim}")
    print(f"group size  : G={config.num_generations}")
    print(f"policy      : Dirichlet(alpha = {config.alpha_base} + {config.alpha_scale} * score)")
    print(f"reward      : default = portfolio_return + {config.entropy_bonus} * norm_entropy")
    print(f"loss        : PPO-clip(eps={config.cliprange}) + {config.kl_coef} * KL_to_ref")
    print("-" * 80)

    global_step = 0
    for epoch in range(1, config.epochs + 1):
        policy.train()
        epoch_metrics: Dict[str, float] = {}
        n_steps = 0

        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            metrics = grpo_step(
                policy=policy,
                ref_policy=ref_policy,
                x=x,
                y=y,
                optimizer=optimizer,
                scheduler=scheduler,
                config=config,
                reward_fn=reward_fn,
            )

            global_step += 1
            n_steps += 1
            for k, v in metrics.items():
                epoch_metrics[k] = epoch_metrics.get(k, 0.0) + v

            with open(log_path, "a", newline="") as f:
                csv.writer(f).writerow(
                    [epoch, global_step, metrics["loss"], metrics["pg_loss"], metrics["kl"],
                     metrics["reward_mean"], metrics["reward_std"], metrics["adv_abs_mean"],
                     metrics["ratio_mean"], metrics["ratio_std"], scheduler.get_last_lr()[0]]
                )

        avg = {k: v / n_steps for k, v in epoch_metrics.items()}
        print(
            f"epoch {epoch:03d}/{config.epochs} "
            f"loss={avg['loss']:.4f} pg={avg['pg_loss']:.4f} kl={avg['kl']:.4f} "
            f"reward={avg['reward_mean']:.4f}±{avg['reward_std']:.4f} "
            f"ratio={avg['ratio_mean']:.3f} lr={scheduler.get_last_lr()[0]:.2e}"
        )

        torch.save(
            policy.state_dict(),
            os.path.join(config.ckpt_dir, f"grpo_epoch_{epoch:03d}.pt"),
        )

    torch.save(policy.state_dict(), os.path.join(config.ckpt_dir, "last_grpo_model.pt"))
    print("-" * 80)
    print("GRPO training done.")
    return policy


# ---------------------------------------------------------------------------
# Inference: deterministic portfolio weights from a trained policy
# ---------------------------------------------------------------------------
def policy_weights(
    x: torch.Tensor,
    config: GRPOConfig = cfg,
    ckpt_path: str = "checkpoints/last_grpo_model.pt",
    stochastic: bool = False,
) -> torch.Tensor:
    """
    Compute portfolio weights for a cross section x: (N, F).

    If ``stochastic=False`` we use the Dirichlet mean (alpha / sum(alpha)),
    which is the deterministic policy. If ``stochastic=True`` we sample once
    from the Dirichlet for an exploratory weight vector.
    """
    model = _build_model(config)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model.eval()
    with torch.no_grad():
        scores = model(x.float())                              # (N,)
        alpha = scores_to_alpha(scores, config.alpha_base, config.alpha_scale)
        if stochastic:
            return Dirichlet(alpha).sample()
        return alpha / alpha.sum()


if __name__ == "__main__":
    train_grpo()
