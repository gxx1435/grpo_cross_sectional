"""
Walk-forward GRPO — long-only Top5, random 30 stocks, semi-annual rolling train.

Same schedule/universe/reward as ``walkforward_longonly_r30_semiannual``, but with
a 1-trading-day input window (L=240 minute bars, including the current bar).

Training: SFT once on fold 0 (``shared_sft/``), then warm-chain GRPO —
folds 1–3 init from the previous fold's ``grpo_model.pt``.

Input:  (N, L=240, F=11)   cross-sectional features over the past trading day
Output: Top5 long-only weights, rebalanced every H=5 bars (unchanged)

Run PatchTST variant after this completes:
  run_walkforward_longonly_r30_semiannual_L1d_patchtst.py

Artifacts:
  files/experiments/walkforward_longonly_r30_semiannual_L1d_warmchain/
"""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "files"))
warnings.filterwarnings("ignore")

from stock_select import select_random_n
from walkforward import POOL_DIR_2025, WalkForwardConfig, run_walkforward

TAG = "walkforward_longonly_r30_semiannual_L1d_warmchain"
ROOT = Path(__file__).resolve().parent / "files" / "experiments" / TAG
BASELINE_ROOT = (
    Path(__file__).resolve().parent
    / "files"
    / "experiments"
    / "walkforward_longonly_r30_semiannual"
)
POOL_2025 = POOL_DIR_2025
POOL_2026 = Path(__file__).resolve().parent / "files" / "csi500" / "2026_1min"
SEED = 42
N_STOCKS = 30
LOOKBACK = 240   # 1 trading day of 1-min bars (incl. current bar)
HORIZON = 5

if __name__ == "__main__":
    ROOT.mkdir(parents=True, exist_ok=True)
    universe_path = ROOT / "universe" / "random30_seed42.csv"
    baseline_universe = BASELINE_ROOT / "universe" / "random30_seed42.csv"
    if baseline_universe.is_file():
        import shutil
        universe_path.parent.mkdir(parents=True, exist_ok=True)
        if not universe_path.is_file():
            shutil.copy2(baseline_universe, universe_path)
        import pandas as pd
        stock_ids = pd.read_csv(universe_path)["stock_id"].tolist()
        print(f"Reusing universe from {baseline_universe}")
    else:
        stock_ids, _ = select_random_n(
            n=N_STOCKS,
            pool_dir=POOL_2025,
            seed=SEED,
            save_path=universe_path,
        )

    print(f"Long-only R30 semi-annual L1d warm-chain (transformer) -> {ROOT}")
    print(f"L={LOOKBACK}, H={HORIZON}, N={len(stock_ids)}, warm_chain=True")
    print(f"Universe: {', '.join(stock_ids[:5])} ...\n")

    run_walkforward(WalkForwardConfig(
        experiment_tag=TAG,
        experiment_root=ROOT,
        pool_dir=POOL_2025,
        extra_pool_dirs=[POOL_2026],
        stock_ids=stock_ids,
        lookback=LOOKBACK,
        horizon=HORIZON,
        portfolio_mode="long_only",
        baseline_mode="long_only",
        top_k=5,
        bottom_k=0,
        reward_mode="auto",
        schedule_mode="monthly_semiannual",
        oos_stitch_start="2026-01",
        oos_stitch_end="2026-04",
        model_type="transformer",
        warm_chain=True,
        sft_batch_size=4,
        seed=SEED,
    ))
