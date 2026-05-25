"""
Walk-forward linear-factor baseline — long-only Top5, random 30, L=240 (1 trading day).

Comparison experiment for ``walkforward_longonly_r30_semiannual_L1d`` (Transformer+GRPO).
Same schedule, universe, portfolio rules, and I/O layout; no SFT/GRPO training.

Scores at each bar come from the SFT teacher: rank-normalized composite of
(-ret_5, -ret_30, -vol_z, -rv_30) cross-sectional z-scores.

Artifacts:
  files/experiments/walkforward_longonly_r30_semiannual_L1d_linear/
"""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "files"))
warnings.filterwarnings("ignore")

from stock_select import select_random_n
from walkforward import POOL_DIR_2025, WalkForwardConfig, run_walkforward

TAG = "walkforward_longonly_r30_semiannual_L1d_linear"
ROOT = Path(__file__).resolve().parent / "files" / "experiments" / TAG
BASELINE_ROOT = (
    Path(__file__).resolve().parent
    / "files"
    / "experiments"
    / "walkforward_longonly_r30_semiannual"
)
L1D_ROOT = (
    Path(__file__).resolve().parent
    / "files"
    / "experiments"
    / "walkforward_longonly_r30_semiannual_L1d"
)
POOL_2025 = POOL_DIR_2025
POOL_2026 = Path(__file__).resolve().parent / "files" / "csi500" / "2026_1min"
SEED = 42
N_STOCKS = 30
LOOKBACK = 240
HORIZON = 5

if __name__ == "__main__":
    ROOT.mkdir(parents=True, exist_ok=True)
    universe_path = ROOT / "universe" / "random30_seed42.csv"
    for src in (L1D_ROOT / "universe" / "random30_seed42.csv",
                BASELINE_ROOT / "universe" / "random30_seed42.csv"):
        if src.is_file():
            import shutil
            universe_path.parent.mkdir(parents=True, exist_ok=True)
            if not universe_path.is_file():
                shutil.copy2(src, universe_path)
            import pandas as pd
            stock_ids = pd.read_csv(universe_path)["stock_id"].tolist()
            print(f"Reusing universe from {src}")
            break
    else:
        stock_ids, _ = select_random_n(
            n=N_STOCKS,
            pool_dir=POOL_2025,
            seed=SEED,
            save_path=universe_path,
        )

    print(f"Long-only R30 semi-annual L1d LINEAR baseline -> {ROOT}")
    print(f"L={LOOKBACK}, H={HORIZON}, N={len(stock_ids)}, model=linear (no training)")
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
        model_type="linear",
        seed=SEED,
    ))
