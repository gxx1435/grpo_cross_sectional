"""
Cross-sectional ranking accuracy for walk-forward GRPO OOS predictions.

Metrics (per bar, then aggregated):
  - Rank IC (Spearman): corr(rank(score), rank(fwd_return))
  - Pearson IC
  - LS direction hit: 1 if mean(top_k fwd) > mean(bottom_k fwd)
  - Top-K overlap: |pred_top ∩ true_top| / K
  - Pairwise accuracy: P(score_i > score_j => ret_i > ret_j)

Outputs under ``<experiment_root>/ranking_accuracy/``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from hf_data import HFConfig, build_hf_dataset, trading_day_index
from train_grpo_trl import GRPOTrainConfig, _build_model
from walkforward import WalkForwardConfig, _cache_tag, _ts_from_int64


def _rank_ic(scores: np.ndarray, rets: np.ndarray) -> float:
    m = np.isfinite(scores) & np.isfinite(rets)
    if m.sum() < 5:
        return np.nan
    rs = pd.Series(scores[m]).rank().to_numpy()
    rr = pd.Series(rets[m]).rank().to_numpy()
    return float(np.corrcoef(rs, rr)[0, 1])


def _pearson_ic(scores: np.ndarray, rets: np.ndarray) -> float:
    m = np.isfinite(scores) & np.isfinite(rets)
    if m.sum() < 5:
        return np.nan
    s, r = scores[m], rets[m]
    if s.std() < 1e-12 or r.std() < 1e-12:
        return np.nan
    return float(np.corrcoef(s, r)[0, 1])


def _topk_indices(x: np.ndarray, k: int, largest: bool = True) -> np.ndarray:
    m = np.isfinite(x)
    idx = np.where(m)[0]
    if len(idx) < k:
        return idx
    order = idx[np.argsort(x[idx])]
    return order[-k:] if largest else order[:k]


def _pairwise_accuracy(scores: np.ndarray, rets: np.ndarray, max_pairs: int = 200) -> float:
    m = np.isfinite(scores) & np.isfinite(rets)
    idx = np.where(m)[0]
    n = len(idx)
    if n < 2:
        return np.nan
    rng = np.random.default_rng(42)
    pairs = []
    limit = min(max_pairs, n * (n - 1) // 2)
    for _ in range(limit):
        i, j = rng.choice(idx, size=2, replace=False)
        if scores[i] == scores[j] or rets[i] == rets[j]:
            continue
        pairs.append((i, j))
    if not pairs:
        return np.nan
    correct = sum(
        (scores[i] > scores[j]) == (rets[i] > rets[j])
        for i, j in pairs
    )
    return correct / len(pairs)


def eval_fold_oos(
    fold_dir: Path,
    stock_ids: List[str],
    cfg: WalkForwardConfig,
    fold_info: Dict,
    top_k: int = 5,
) -> pd.DataFrame:
    cache_tag = _cache_tag(cfg, fold_info["fold_idx"], "rank_oos")
    hf_cfg = HFConfig(
        pool_dir=cfg.pool_dir,
        start_day=fold_info["oos_start_day"],
        num_days=fold_info["oos_num_days"],
        lookback=cfg.lookback,
        horizon=cfg.horizon,
        stock_ids=stock_ids,
        cache_tag=cache_tag,
    )
    bundle = build_hf_dataset(hf_cfg)
    x_seq = bundle["x_seq"]
    fwd = bundle["fwd_return"]
    vs, ve = int(bundle["valid_start"]), int(bundle["valid_end"])
    L = cfg.lookback
    T = ve - vs

    grpo_cfg = GRPOTrainConfig(lookback=cfg.lookback)
    model = _build_model(x_seq.shape[-1], grpo_cfg)
    ckpt = fold_dir / "grpo_model.pt"
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval()

    rows = []
    with torch.no_grad():
        for i in range(T):
            t = vs + i
            window = x_seq[t - L + 1: t + 1]
            x = np.transpose(window, (1, 0, 2)).copy()
            scores = model(torch.from_numpy(x).float()).numpy()
            rets = fwd[t]
            ric = _rank_ic(scores, rets)
            pic = _pearson_ic(scores, rets)
            pred_top = set(_topk_indices(scores, top_k, largest=True).tolist())
            pred_bot = set(_topk_indices(scores, top_k, largest=False).tolist())
            true_top = set(_topk_indices(rets, top_k, largest=True).tolist())
            true_bot = set(_topk_indices(rets, top_k, largest=False).tolist())
            top_overlap = len(pred_top & true_top) / top_k if top_k else np.nan
            bot_overlap = len(pred_bot & true_bot) / top_k if top_k else np.nan
            top_mean = np.nanmean(rets[list(pred_top)]) if pred_top else np.nan
            bot_mean = np.nanmean(rets[list(pred_bot)]) if pred_bot else np.nan
            ls_hit = float(top_mean > bot_mean) if np.isfinite(top_mean) and np.isfinite(bot_mean) else np.nan
            pw_acc = _pairwise_accuracy(scores, rets)
            ts = _ts_from_int64(np.array([bundle["timestamps_ns"][t]]))[0]
            rows.append({
                "fold": fold_info["fold_idx"],
                "ts": ts,
                "rank_ic": ric,
                "pearson_ic": pic,
                "ls_direction_hit": ls_hit,
                "top5_overlap": top_overlap,
                "bottom5_overlap": bot_overlap,
                "pairwise_acc": pw_acc,
                "pred_top_mean_ret": top_mean,
                "pred_bot_mean_ret": bot_mean,
                "pred_ls_spread": top_mean - bot_mean if np.isfinite(top_mean) and np.isfinite(bot_mean) else np.nan,
            })
    return pd.DataFrame(rows)


def plot_results(bar_df: pd.DataFrame, fold_df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Fig 1: 2x2 summary ---
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    ax = axes[0, 0]
    ax.hist(bar_df["rank_ic"].dropna(), bins=40, color="#d6336c", alpha=0.75, edgecolor="white")
    ax.axvline(bar_df["rank_ic"].mean(), color="#1864ab", ls="--", lw=2,
               label=f"mean={bar_df['rank_ic'].mean():+.3f}")
    ax.axvline(0, color="#aaa", ls="-", lw=1)
    ax.set_xlabel("Rank IC (Spearman)")
    ax.set_ylabel("Bar count")
    ax.set_title("OOS Rank IC distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    x = np.arange(len(fold_df))
    w = 0.35
    ax.bar(x - w / 2, fold_df["rank_ic_mean"], w, label="Rank IC", color="#d6336c")
    ax.bar(x + w / 2, fold_df["pearson_ic_mean"], w, label="Pearson IC", color="#1864ab")
    ax.axhline(0, color="#aaa", ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels([f"F{int(i)}" for i in fold_df["fold"]], rotation=0)
    ax.set_ylabel("Mean IC")
    ax.set_title("Per-fold mean IC")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[1, 0]
    metrics = ["ls_direction_hit", "top5_overlap", "pairwise_acc"]
    labels = ["LS direction\n(top>bot)", "Top5 overlap", "Pairwise acc"]
    means = [bar_df[m].mean() for m in metrics]
    colors = ["#2f9e44", "#e67700", "#7048e8"]
    bars = ax.bar(labels, means, color=colors, alpha=0.85)
    ax.axhline(0.5, color="#aaa", ls="--", label="random=0.5")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy / rate")
    ax.set_title("Overall ranking accuracy metrics")
    for b, v in zip(bars, means):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.1%}", ha="center", fontsize=10)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[1, 1]
    bar_df = bar_df.sort_values("ts").reset_index(drop=True)
    bar_df["date"] = bar_df["ts"].dt.normalize()
    daily_ic = bar_df.groupby("date")["rank_ic"].mean()
    ax.plot(daily_ic.index, daily_ic.values, color="#d6336c", lw=1.5, marker="o", ms=4)
    ax.axhline(0, color="#aaa", ls="--")
    ax.axhline(daily_ic.mean(), color="#1864ab", ls=":", label=f"daily mean={daily_ic.mean():+.3f}")
    ax.set_ylabel("Mean Rank IC")
    ax.set_title("Daily mean Rank IC (OOS)")
    ax.legend()
    fig.autofmt_xdate()
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        "GRPO cross-sectional ranking accuracy — random30 walk-forward OOS",
        fontsize=13, y=1.01,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "ranking_accuracy_summary.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # --- Fig 2: rolling Rank IC ---
    fig, ax = plt.subplots(figsize=(12, 4))
    win = min(120, max(30, len(bar_df) // 10))
    roll = bar_df["rank_ic"].rolling(win, min_periods=max(20, win // 4)).mean()
    ax.plot(np.arange(len(bar_df)), roll, color="#d6336c", lw=1.4,
            label=f"rolling mean Rank IC (win={win})")
    ax.axhline(0, color="#aaa", ls="--")
    day_change = bar_df["date"].values[1:] != bar_df["date"].values[:-1]
    day_starts = np.r_[0, np.where(day_change)[0] + 1]
    for d in day_starts[1:]:
        ax.axvline(d, color="#eee", ls=":", lw=0.6)
    ax.set_xlabel("OOS bar index")
    ax.set_ylabel("Rank IC")
    ax.set_title("Rolling Rank IC — GRPO OOS predictions")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "ranking_ic_rolling.png", dpi=130)
    plt.close(fig)

    # --- Fig 3: per-fold accuracy bars ---
    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(fold_df))
    w = 0.25
    ax.bar(x - w, fold_df["ls_direction_hit"], w, label="LS direction hit", color="#2f9e44")
    ax.bar(x, fold_df["top5_overlap"], w, label="Top5 overlap", color="#e67700")
    ax.bar(x + w, fold_df["pairwise_acc"], w, label="Pairwise acc", color="#7048e8")
    ax.axhline(0.5, color="#aaa", ls="--", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"F{int(r.fold)}\n{r.oos_start}" for r in fold_df.itertuples()],
        fontsize=8,
    )
    ax.set_ylim(0, 1)
    ax.set_ylabel("Rate")
    ax.set_title("Per-fold ranking accuracy")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_dir / "ranking_accuracy_by_fold.png", dpi=130)
    plt.close(fig)


def run_analysis(experiment_root: Path, cfg: WalkForwardConfig) -> Dict:
    root = Path(experiment_root)
    out_dir = root / "ranking_accuracy"
    fold_dirs = sorted((root / "folds").glob("fold_*"))
    if not fold_dirs:
        raise FileNotFoundError(f"No folds in {root / 'folds'}")

    bar_frames = []
    fold_rows = []
    for fold_dir in fold_dirs:
        meta = json.loads((fold_dir / "meta.json").read_text(encoding="utf-8"))
        fold_info = {
            "fold_idx": meta["fold_idx"],
            "oos_start_day": trading_day_index(cfg.pool_dir, meta["oos_dates"][0]),
            "oos_num_days": len(meta["oos_dates"]),
        }
        df = eval_fold_oos(fold_dir, meta["stock_ids"], cfg, fold_info, top_k=cfg.top_k)
        bar_frames.append(df)
        fold_rows.append({
            "fold": meta["fold_idx"],
            "oos_start": meta["oos_dates"][0],
            "oos_end": meta["oos_dates"][-1],
            "n_bars": len(df),
            "rank_ic_mean": df["rank_ic"].mean(),
            "rank_ic_std": df["rank_ic"].std(),
            "pearson_ic_mean": df["pearson_ic"].mean(),
            "ls_direction_hit": df["ls_direction_hit"].mean(),
            "top5_overlap": df["top5_overlap"].mean(),
            "bottom5_overlap": df["bottom5_overlap"].mean(),
            "pairwise_acc": df["pairwise_acc"].mean(),
            "pred_ls_spread_bps": df["pred_ls_spread"].mean() * 1e4,
        })

    bar_df = pd.concat(bar_frames, ignore_index=True)
    fold_df = pd.DataFrame(fold_rows)

    out_dir.mkdir(parents=True, exist_ok=True)
    bar_df.to_csv(out_dir / "bar_level_ranking_metrics.csv", index=False)
    fold_df.to_csv(out_dir / "fold_ranking_summary.csv", index=False)

    summary = {
        "experiment": root.name,
        "n_bars": len(bar_df),
        "n_folds": len(fold_df),
        "rank_ic_mean": float(bar_df["rank_ic"].mean()),
        "rank_ic_std": float(bar_df["rank_ic"].std()),
        "rank_ic_positive_pct": float((bar_df["rank_ic"] > 0).mean()),
        "pearson_ic_mean": float(bar_df["pearson_ic"].mean()),
        "ls_direction_hit": float(bar_df["ls_direction_hit"].mean()),
        "top5_overlap": float(bar_df["top5_overlap"].mean()),
        "bottom5_overlap": float(bar_df["bottom5_overlap"].mean()),
        "pairwise_acc": float(bar_df["pairwise_acc"].mean()),
        "random_baseline_pairwise": 0.5,
        "random_baseline_top5_overlap": cfg.top_k / 30 if cfg.top_k else 1 / 6,
    }
    pd.DataFrame([summary]).to_csv(out_dir / "overall_summary.csv", index=False)

    plot_results(bar_df, fold_df, out_dir)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="GRPO cross-sectional ranking accuracy")
    parser.add_argument(
        "--experiment",
        type=Path,
        default=Path("files/experiments/walkforward_202604_empirical_random30"),
    )
    args = parser.parse_args()

    tag = args.experiment.name
    cfg = WalkForwardConfig(
        experiment_tag=tag,
        experiment_root=args.experiment,
        top_k=5,
        bottom_k=5,
    )
    summary = run_analysis(args.experiment, cfg)
    out = args.experiment / "ranking_accuracy"
    print(f"Saved ranking accuracy analysis -> {out}")
    print(f"  Rank IC mean     : {summary['rank_ic_mean']:+.4f}  "
          f"({summary['rank_ic_positive_pct']:.1%} bars > 0)")
    print(f"  LS direction hit : {summary['ls_direction_hit']:.1%}")
    print(f"  Top5 overlap     : {summary['top5_overlap']:.1%}  "
          f"(random ≈ {summary['random_baseline_top5_overlap']:.1%})")
    print(f"  Pairwise acc     : {summary['pairwise_acc']:.1%}  (random = 50%)")
    print(f"  Charts           : ranking_accuracy_summary.png, ranking_ic_rolling.png, "
          f"ranking_accuracy_by_fold.png")


if __name__ == "__main__":
    main()
