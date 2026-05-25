"""
Walk-forward GRPO — L1d window (L=240) with PatchTST backbone.

Run after ``run_walkforward_longonly_r30_semiannual_L1d.py`` (transformer) completes.

Artifacts:
  files/experiments/walkforward_longonly_r30_semiannual_L1d_patchtst/
"""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "files"))
warnings.filterwarnings("ignore")

import pandas as pd
from walkforward import POOL_DIR_2025, WalkForwardConfig, run_walkforward

TAG = "walkforward_longonly_r30_semiannual_L1d_patchtst_warmchain"
ROOT = Path(__file__).resolve().parent / "files" / "experiments" / TAG
TRANSFORMER_ROOT = (
    Path(__file__).resolve().parent
    / "files"
    / "experiments"
    / "walkforward_longonly_r30_semiannual_L1d_warmchain"
)
POOL_2025 = POOL_DIR_2025
POOL_2026 = Path(__file__).resolve().parent / "files" / "csi500" / "2026_1min"
LOOKBACK = 240
HORIZON = 5

if __name__ == "__main__":
    ROOT.mkdir(parents=True, exist_ok=True)
    universe_path = TRANSFORMER_ROOT / "universe" / "random30_seed42.csv"
    if not universe_path.is_file():
        raise FileNotFoundError(
            f"Run transformer L1d experiment first; missing {universe_path}"
        )
    stock_ids = pd.read_csv(universe_path)["stock_id"].tolist()

    print(f"Long-only R30 semi-annual L1d warm-chain (PatchTST) -> {ROOT}")
    print(f"L={LOOKBACK}, H={HORIZON}, N={len(stock_ids)}, warm_chain=True\n")

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
        model_type="patchtst",
        patch_len=12,
        patch_stride=12,
        patch_e_layers=2,
        patch_cs_layers=1,
        warm_chain=True,
        sft_batch_size=4,
        seed=42,
    ))
