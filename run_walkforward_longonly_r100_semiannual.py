"""
Walk-forward GRPO — long-only Top5, random 100 stocks, semi-annual rolling train.

Same portfolio/reward settings as ``walkforward_202604_longonly``:
  - Long-only Top5 (20% each), no short leg
  - Auto-learn reward weights
  - Baseline: 100-stock equal-weight long-only

Schedule (monthly slide, ~6mo train → 1mo OOS):
  fold0: train 2025-06..2025-12 → OOS 2026-01
  fold1: train 2025-07..2026-01 → OOS 2026-02
  fold2: train 2025-08..2026-02 → OOS 2026-03
  fold3: train 2025-09..2026-03 → OOS 2026-04

Primary deliverable: stitched OOS curve for 2026-01 .. 2026-04.

Artifacts:
  files/experiments/walkforward_longonly_r100_semiannual/
"""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "files"))
warnings.filterwarnings("ignore")

from stock_select import select_random_n
from walkforward import POOL_DIR_2025, WalkForwardConfig, run_walkforward

TAG = "walkforward_longonly_r100_semiannual"
ROOT = Path(__file__).resolve().parent / "files" / "experiments" / TAG
POOL_2025 = POOL_DIR_2025
POOL_2026 = Path(__file__).resolve().parent / "files" / "csi500" / "2026_1min"
SEED = 42
N_STOCKS = 100

if __name__ == "__main__":
    ROOT.mkdir(parents=True, exist_ok=True)
    stock_ids, uni_df = select_random_n(
        n=N_STOCKS,
        pool_dir=POOL_2025,
        seed=SEED,
        save_path=ROOT / "universe" / "random100_seed42.csv",
    )
    print(f"Long-only R100 semi-annual experiment -> {ROOT}")
    print(f"Universe ({len(stock_ids)}): {', '.join(stock_ids[:5])} ...\n")

    run_walkforward(WalkForwardConfig(
        experiment_tag=TAG,
        experiment_root=ROOT,
        pool_dir=POOL_2025,
        extra_pool_dirs=[POOL_2026],
        stock_ids=stock_ids,
        portfolio_mode="long_only",
        baseline_mode="long_only",
        top_k=5,
        bottom_k=0,
        reward_mode="auto",
        schedule_mode="monthly_semiannual",
        oos_stitch_start="2026-01",
        oos_stitch_end="2026-04",
        seed=SEED,
    ))
