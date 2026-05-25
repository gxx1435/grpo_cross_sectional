"""
OOS backtest for the trained Stock policy on 1-minute CSI 500 data.

Workflow
--------
1. Load a checkpoint (GRPO if available, else SFT).
2. Build features for the backtest window via ``hf_data.build_hf_dataset``
   (with ``start_day`` set to the OOS start).
3. For every bar t in the window:
       scores_t  = policy(x_t)               # x_t is (N, L, F)
       alpha_t   = base + scale * scores_t
       weights_t = alpha_t / alpha_t.sum()   # Dirichlet mean (deterministic)
       pr_t      = weights_t . forward_return_t
   Optional cost: ``- fee_bps * 1e-4 * |w_t - w_{t-rebalance}|_1`` whenever
   we rebalance (default: every H bars).
4. Aggregate:
       - Bar-level returns
       - Daily returns
       - Cumulative net value
       - Annualized return / volatility / Sharpe / MaxDD / hit rate / turnover
       - Monthly return table
5. Compare against equal-weight baseline on the same universe.

Run
---
    python3 files/backtest.py
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.distributions import Dirichlet

from hf_data import HFConfig, build_hf_dataset
from stock_transformer import StockTemporalTransformer


HERE = Path(__file__).resolve().parent

# Bars per trading day after dropping 09:30 and 15:00 auction prints.
BARS_PER_DAY = 240
BARS_PER_YEAR = BARS_PER_DAY * 244


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class BacktestConfig:
    pool_dir: Path = HERE / "csi500" / "2026_1min"
    start_day: int = 25                 # first OOS day index inside pool_dir
    num_days: int = 60                  # ~3 months of trading days
    lookback: int = 30
    horizon: int = 5                    # H bars: rebalance / label horizon
    n_stocks_max: Optional[int] = None  # None == full universe

    d_model: int = 64
    num_heads: int = 4
    ffn_dim: int = 128
    dropout: float = 0.1
    max_lookback: int = 30

    alpha_base: float = 1.0
    alpha_scale: float = 10.0

    fee_bps: float = 5.0                # one-side cost in bps applied at rebalance
    rebalance_every: int = 5            # rebalance every K bars; H bars ahead
    annualization: int = BARS_PER_YEAR  # used for Sharpe annualization

    sft_ckpt: Path = HERE / "checkpoints" / "best_sft_model.pt"
    grpo_ckpt: Path = HERE / "checkpoints" / "last_grpo_model.pt"
    output_dir: Path = HERE / "backtest_out"

    seed: int = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _build_model(feat_dim: int, cfg: BacktestConfig) -> StockTemporalTransformer:
    return StockTemporalTransformer(
        feat_dim=feat_dim,
        max_lookback=cfg.max_lookback,
        d_model=cfg.d_model,
        num_heads=cfg.num_heads,
        ffn_dim=cfg.ffn_dim,
        dropout=cfg.dropout,
    )


def _load_policy(cfg: BacktestConfig, feat_dim: int) -> Tuple[StockTemporalTransformer, str]:
    model = _build_model(feat_dim, cfg)
    ckpt_used = "random"
    if cfg.grpo_ckpt.is_file():
        model.load_state_dict(torch.load(cfg.grpo_ckpt, map_location="cpu"))
        ckpt_used = f"GRPO ({cfg.grpo_ckpt.name})"
    elif cfg.sft_ckpt.is_file():
        model.load_state_dict(torch.load(cfg.sft_ckpt, map_location="cpu"))
        ckpt_used = f"SFT ({cfg.sft_ckpt.name})"
    model.eval()
    return model, ckpt_used


def scores_to_weights(scores: torch.Tensor, base: float, scale: float) -> torch.Tensor:
    s = scores.clamp(0.0, 1.0)
    alpha = base + scale * s
    return alpha / alpha.sum(dim=-1, keepdim=True)


# ---------------------------------------------------------------------------
# Forward sweep
# ---------------------------------------------------------------------------
def run_policy_sweep(
    cfg: BacktestConfig,
    bundle: Dict[str, np.ndarray],
    model: StockTemporalTransformer,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute deterministic policy weights for every bar in the valid window.

    Returns:
        weights : (T_bt, N)
        returns : (T_bt, N)        realized H-bar log returns
        ts_idx  : (T_bt,)          row indices into the bundle
    """
    x_seq = bundle["x_seq"]                # (T, N, F)
    fwd = bundle["fwd_return"]             # (T, N)
    valid_start = int(bundle["valid_start"])
    valid_end = int(bundle["valid_end"])
    L = cfg.lookback
    T_bt = valid_end - valid_start

    weights = np.zeros((T_bt, x_seq.shape[1]), dtype=np.float32)
    returns = np.zeros((T_bt, x_seq.shape[1]), dtype=np.float32)
    ts_idx = np.zeros(T_bt, dtype=np.int64)

    device = next(model.parameters()).device
    BATCH = 64

    with torch.no_grad():
        for batch_start in range(0, T_bt, BATCH):
            batch_end = min(batch_start + BATCH, T_bt)
            batch_xs = []
            for i in range(batch_start, batch_end):
                t = valid_start + i
                window = x_seq[t - L + 1: t + 1]            # (L, N, F)
                batch_xs.append(np.transpose(window, (1, 0, 2)))
            x_batch = torch.from_numpy(np.stack(batch_xs)).to(device)  # (B, N, L, F)
            scores = model(x_batch)                                   # (B, N)
            w = scores_to_weights(scores, cfg.alpha_base, cfg.alpha_scale)
            weights[batch_start:batch_end] = w.cpu().numpy()

            for i in range(batch_start, batch_end):
                t = valid_start + i
                returns[i] = fwd[t]
                ts_idx[i] = t

    return weights, returns, ts_idx


# ---------------------------------------------------------------------------
# Portfolio simulation
# ---------------------------------------------------------------------------
def simulate(
    weights: np.ndarray,
    returns: np.ndarray,
    cfg: BacktestConfig,
) -> Dict[str, np.ndarray]:
    """
    Run a simple "rebalance every K bars, hold for H bars" simulation.

    For overlapping H>K, we still attribute the H-ahead return to the
    weight at the rebalance bar (this is the standard "stale held" view —
    fine as a first-cut backtest). A non-overlapping setup (K==H) is the
    cleanest read; the default.
    """
    T_bt, N = weights.shape
    K = max(cfg.rebalance_every, 1)

    rebalance_idx = np.arange(0, T_bt, K)

    # Weights actually held: stepwise constant between rebalances.
    held = np.zeros_like(weights)
    last_w = weights[rebalance_idx[0]]
    for t in range(T_bt):
        if t in rebalance_idx:
            last_w = weights[t]
        held[t] = last_w

    # Per-bar gross portfolio return
    gross_pr = (held * returns).sum(axis=1)            # (T_bt,)

    # Costs at rebalance bars only
    cost_bps = cfg.fee_bps * 1e-4
    cost = np.zeros(T_bt, dtype=np.float32)
    prev_w = held[0]
    for t in rebalance_idx:
        cost[t] = cost_bps * np.abs(weights[t] - prev_w).sum()
        prev_w = weights[t]

    net_pr = gross_pr - cost
    return {
        "gross_pr": gross_pr,
        "cost": cost,
        "net_pr": net_pr,
        "weights_held": held,
        "rebalance_idx": rebalance_idx,
    }


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------
def annualized_stats(net_pr: np.ndarray, ann: int) -> Dict[str, float]:
    if len(net_pr) == 0:
        return {}
    mu = net_pr.mean()
    sigma = net_pr.std(ddof=0)
    sharpe = math.sqrt(ann) * mu / (sigma + 1e-12)
    cum = (1.0 + net_pr).cumprod()
    total_ret = float(cum[-1] - 1.0)
    ann_ret = (1.0 + total_ret) ** (ann / max(len(net_pr), 1)) - 1.0
    running_max = np.maximum.accumulate(cum)
    dd = 1.0 - cum / running_max
    max_dd = float(dd.max())
    hit = float((net_pr > 0).mean())
    return {
        "bars": float(len(net_pr)),
        "mean_ret_per_bar": float(mu),
        "std_ret_per_bar": float(sigma),
        "annualized_sharpe": float(sharpe),
        "total_return": total_ret,
        "annualized_return": float(ann_ret),
        "max_drawdown": max_dd,
        "hit_rate": hit,
    }


def daily_aggregate(
    net_pr: np.ndarray, ts_ns: np.ndarray
) -> pd.DataFrame:
    ts = pd.to_datetime(ts_ns)
    df = pd.DataFrame({"ts": ts, "net_pr": net_pr})
    df["date"] = df["ts"].dt.normalize()
    daily = df.groupby("date")["net_pr"].apply(
        lambda x: (1.0 + x.values).prod() - 1.0
    ).reset_index().rename(columns={"net_pr": "daily_return"})
    return daily


def monthly_table(daily: pd.DataFrame) -> pd.DataFrame:
    daily = daily.copy()
    daily["month"] = daily["date"].dt.to_period("M").astype(str)
    monthly = daily.groupby("month")["daily_return"].apply(
        lambda x: (1.0 + x.values).prod() - 1.0
    ).reset_index().rename(columns={"daily_return": "monthly_return"})
    return monthly


# ---------------------------------------------------------------------------
# Top-level entry
# ---------------------------------------------------------------------------
def backtest(cfg: BacktestConfig = BacktestConfig()) -> Dict[str, object]:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    hf_cfg = HFConfig(
        pool_dir=cfg.pool_dir,
        start_day=cfg.start_day,
        num_days=cfg.num_days,
        lookback=cfg.lookback,
        horizon=cfg.horizon,
        n_stocks_max=cfg.n_stocks_max,
    )
    bundle = build_hf_dataset(hf_cfg)
    feat_dim = bundle["x_seq"].shape[-1]
    n_stocks = bundle["x_seq"].shape[1]

    model, ckpt_label = _load_policy(cfg, feat_dim)

    print(f"backtest config")
    print(f"  pool_dir   : {cfg.pool_dir}")
    print(f"  date range : start_day={cfg.start_day}, num_days={cfg.num_days}")
    print(f"  N x F x L  : {n_stocks} x {feat_dim} x {cfg.lookback}")
    print(f"  rebalance  : every {cfg.rebalance_every} bars (H={cfg.horizon})")
    print(f"  fee        : {cfg.fee_bps} bps per rebalance one-side")
    print(f"  policy     : {ckpt_label}")
    print("-" * 80)

    # 1) Sweep policy
    weights, returns, ts_row_idx = run_policy_sweep(cfg, bundle, model)
    sim = simulate(weights, returns, cfg)
    ts_ns = bundle["timestamps_ns"][ts_row_idx]

    # 2) Equal-weight baseline
    eq_weights = np.full_like(weights, 1.0 / n_stocks, dtype=np.float32)
    sim_eq = simulate(eq_weights, returns, cfg)

    # 3) Stats
    stats_policy = annualized_stats(sim["net_pr"], cfg.annualization)
    stats_eq = annualized_stats(sim_eq["net_pr"], cfg.annualization)

    daily_pol = daily_aggregate(sim["net_pr"], ts_ns)
    daily_eq = daily_aggregate(sim_eq["net_pr"], ts_ns)
    daily_pol["cum_nav"] = (1.0 + daily_pol["daily_return"]).cumprod()
    daily_eq["cum_nav"] = (1.0 + daily_eq["daily_return"]).cumprod()

    months = monthly_table(daily_pol)
    months_eq = monthly_table(daily_eq)
    months["benchmark_eq"] = months_eq["monthly_return"].values
    months["alpha_vs_eq"] = months["monthly_return"] - months["benchmark_eq"]

    # 4) Print summary
    def fmt(stats):
        if not stats:
            return "(empty)"
        return (
            f"bars={int(stats['bars'])} "
            f"total={stats['total_return']*100:.2f}% "
            f"ann={stats['annualized_return']*100:.2f}% "
            f"sharpe={stats['annualized_sharpe']:.2f} "
            f"MDD={stats['max_drawdown']*100:.2f}% "
            f"hit={stats['hit_rate']*100:.1f}%"
        )

    print("=== summary (3-month OOS, 1-min bars) ===")
    print(f"  policy  : {fmt(stats_policy)}")
    print(f"  EW base : {fmt(stats_eq)}")
    print()
    print("=== monthly returns (policy / equal-weight / alpha) ===")
    for _, row in months.iterrows():
        print(
            f"  {row['month']}  policy={row['monthly_return']*100:+.2f}%   "
            f"EW={row['benchmark_eq']*100:+.2f}%   "
            f"alpha={row['alpha_vs_eq']*100:+.2f}%"
        )
    print()
    print("=== daily NAV (last 10) ===")
    print(daily_pol.tail(10).to_string(index=False))

    # 5) Persist
    daily_pol.to_csv(cfg.output_dir / "daily_policy.csv", index=False)
    daily_eq.to_csv(cfg.output_dir / "daily_equal_weight.csv", index=False)
    months.to_csv(cfg.output_dir / "monthly_returns.csv", index=False)
    pd.DataFrame([stats_policy]).to_csv(cfg.output_dir / "summary_policy.csv", index=False)
    pd.DataFrame([stats_eq]).to_csv(cfg.output_dir / "summary_equal_weight.csv", index=False)
    print(f"\nresults saved to {cfg.output_dir}")

    return {
        "stats_policy": stats_policy,
        "stats_eq": stats_eq,
        "daily_policy": daily_pol,
        "daily_eq": daily_eq,
        "monthly": months,
        "weights": weights,
        "returns": returns,
        "timestamps_ns": ts_ns,
    }


if __name__ == "__main__":
    backtest()
