"""
Inspect the last cross section of the 5-day OOS backtest window using the
v1 checkpoint (files/checkpoints/, trained on 30 stocks x 10 days).

What this prints / saves
------------------------
For the LAST valid timestamp inside the backtest window:
    rank, stock_id, score (model output in [0,1]),
    alpha (= alpha_base + alpha_scale * score),
    weight (= alpha / sum(alpha) -- the Dirichlet mean, deterministic).

Output:
    files/backtest_out/last_section_weights.csv
    (sorted by weight, descending; full universe of 30 stocks).

Run:
    python3 files/inspect_last_section.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from backtest import BacktestConfig, _load_policy, scores_to_weights
from hf_data import HFConfig, build_hf_dataset


HERE = Path(__file__).resolve().parent


def main() -> None:
    # ---- Match the v1 run_demo settings exactly ----------------------------
    cfg = BacktestConfig(
        pool_dir=HERE / "csi500" / "2026_1min",
        start_day=10,
        num_days=5,
        n_stocks_max=30,
        lookback=30,
        horizon=5,
        d_model=64, num_heads=4, ffn_dim=128, dropout=0.1, max_lookback=30,
        alpha_base=1.0, alpha_scale=10.0,
        sft_ckpt=HERE / "checkpoints" / "best_sft_model.pt",
        grpo_ckpt=HERE / "checkpoints" / "last_grpo_model.pt",
        output_dir=HERE / "backtest_out",
    )
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Build data + model -----------------------------------------------
    hf_cfg = HFConfig(
        pool_dir=cfg.pool_dir,
        start_day=cfg.start_day,
        num_days=cfg.num_days,
        lookback=cfg.lookback,
        horizon=cfg.horizon,
        n_stocks_max=cfg.n_stocks_max,
    )
    bundle = build_hf_dataset(hf_cfg)
    x_seq = bundle["x_seq"]                  # (T, N, F)
    stock_ids = list(bundle["stock_ids"])    # (N,)
    timestamps_ns = bundle["timestamps_ns"]  # (T,)
    valid_start = int(bundle["valid_start"])
    valid_end = int(bundle["valid_end"])
    feat_dim = x_seq.shape[-1]
    n_stocks = x_seq.shape[1]

    model, ckpt_label = _load_policy(cfg, feat_dim)

    # ---- Pick the LAST valid timestamp ------------------------------------
    t_last = valid_end - 1                                   # last index with a label
    L = cfg.lookback
    window = x_seq[t_last - L + 1: t_last + 1]               # (L, N, F)
    x = np.transpose(window, (1, 0, 2)).copy()               # (N, L, F)
    x_t = torch.from_numpy(x).float()                        # no batch dim -> (N, L, F)

    with torch.no_grad():
        scores = model(x_t)                                  # (N,)
    weights = scores_to_weights(scores, cfg.alpha_base, cfg.alpha_scale)   # (N,)
    alpha = cfg.alpha_base + cfg.alpha_scale * scores.clamp(0.0, 1.0)

    # ---- Forward realized return (next H bars) for context ----------------
    fwd_t = bundle["fwd_return"][t_last]                     # (N,)
    ts = pd.to_datetime(timestamps_ns[t_last])

    # ---- Build a tidy frame ------------------------------------------------
    df = pd.DataFrame({
        "stock_id": stock_ids,
        "score": scores.numpy(),
        "alpha": alpha.numpy(),
        "weight": weights.numpy(),
        "fwd_return_H": fwd_t.astype(np.float64),            # next-H-bar log return
    })
    df = df.sort_values("weight", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", np.arange(1, len(df) + 1))

    # ---- Report ------------------------------------------------------------
    print("=" * 80)
    print(f"  checkpoint   : {ckpt_label}")
    print(f"  window       : start_day={cfg.start_day} num_days={cfg.num_days}")
    print(f"  universe     : N={n_stocks} stocks (first {cfg.n_stocks_max} of csi500/2026_1min)")
    print(f"  last bar ts  : {ts}")
    print(f"  policy       : weight = (alpha_base + alpha_scale * score) / sum(...)")
    print(f"                 alpha_base={cfg.alpha_base}  alpha_scale={cfg.alpha_scale}")
    print("=" * 80)

    # Pretty print full table (it's just 30 stocks).
    with pd.option_context("display.max_rows", None,
                           "display.max_columns", None,
                           "display.float_format", "{:.6f}".format):
        print(df.to_string(index=False))

    print("-" * 80)
    print(f"  sum(weights) = {df['weight'].sum():.6f}  (sanity: should be 1)")
    print(f"  max weight   = {df['weight'].max():.4%}  (stock: {df.iloc[0]['stock_id']})")
    print(f"  min weight   = {df['weight'].min():.4%}  (stock: {df.iloc[-1]['stock_id']})")
    print(f"  HHI (concentration) = {(df['weight']**2).sum():.4f}  "
          f"(equal-weight HHI = {1.0/n_stocks:.4f})")

    out_csv = cfg.output_dir / "last_section_weights.csv"
    # Write with header line that captures the context.
    with open(out_csv, "w") as f:
        f.write(f"# checkpoint: {ckpt_label}\n")
        f.write(f"# window: start_day={cfg.start_day}, num_days={cfg.num_days}\n")
        f.write(f"# last_bar_ts: {ts}\n")
        f.write(f"# alpha_base={cfg.alpha_base}, alpha_scale={cfg.alpha_scale}\n")
    df.to_csv(out_csv, mode="a", index=False)
    print("-" * 80)
    print(f"  saved -> {out_csv}")


if __name__ == "__main__":
    main()
