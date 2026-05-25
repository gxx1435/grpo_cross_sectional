"""
Walk-forward model comparison: StockTemporalTransformer + market regime, random30.

Identical schedule / reward / universe as ``walkforward_patchtst_r30_regime_202604``,
only ``model_type="transformer"`` (72K two-stage Transformer vs PatchTST 110K).

Artifacts:
  files/experiments/walkforward_transformer_r30_regime_202604/
"""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "files"))
warnings.filterwarnings("ignore")

from reward_model import empirical_reward_config
from stock_select import select_random_n
from walkforward import POOL_DIR, WalkForwardConfig, run_walkforward

TAG = "walkforward_transformer_r30_regime_202604"
ROOT = Path(__file__).resolve().parent / "files" / "experiments" / TAG
SEED = 42
N_STOCKS = 30

if __name__ == "__main__":
    ROOT.mkdir(parents=True, exist_ok=True)
    stock_ids, _ = select_random_n(
        n=N_STOCKS,
        pool_dir=POOL_DIR,
        seed=SEED,
        save_path=ROOT / "universe" / "random30_seed42.csv",
    )
    print(f"Transformer + regime R30 (202604 small) -> {ROOT}")
    print(f"Universe ({len(stock_ids)}): {', '.join(stock_ids[:5])} ...\n")

    run_walkforward(WalkForwardConfig(
        experiment_tag=TAG,
        experiment_root=ROOT,
        pool_dir=POOL_DIR,
        stock_ids=stock_ids,
        train_days=3,
        oos_days=3,
        train_cal_start="2026-04-01",
        train_cal_end="2026-05-11",
        oos_cal_start="2026-04-07",
        oos_cal_end="2026-05-11",
        schedule_mode="block",
        portfolio_mode="long_only",
        baseline_mode="long_only",
        top_k=5,
        bottom_k=0,
        reward_mode="empirical",
        reward_config=empirical_reward_config(top_k=5, bottom_k=0),
        seed=SEED,
        model_type="transformer",
        use_regime_features=True,
        regime_lookback_days=30,
        eval_extended=True,
    ))
