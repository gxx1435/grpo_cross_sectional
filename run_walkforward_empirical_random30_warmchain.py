"""
Walk-forward GRPO — empirical reward + random 30 stocks + warm-chain training.

Same universe/reward/folds as ``walkforward_202604_empirical_random30``, but:
  - SFT only on the first sliding window (fold 0)
  - Folds 1+ warm-start GRPO from the previous fold's ``grpo_model.pt``

Artifacts:
  files/experiments/walkforward_202604_empirical_random30_warmchain/
"""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "files"))
warnings.filterwarnings("ignore")

from reward_model import empirical_reward_config
from stock_select import select_random_n
from walkforward import WalkForwardConfig, run_walkforward

TAG = "walkforward_202604_empirical_random30_warmchain"
ROOT = Path(__file__).resolve().parent / "files" / "experiments" / TAG
SEED = 42

if __name__ == "__main__":
    ROOT.mkdir(parents=True, exist_ok=True)
    stock_ids, uni_df = select_random_n(
        n=30,
        seed=SEED,
        save_path=ROOT / "universe" / "random30_seed42.csv",
    )
    print(f"Random-30 empirical warm-chain experiment -> {ROOT}")
    print(f"Universe ({len(stock_ids)}): {', '.join(stock_ids[:5])} ...\n")

    run_walkforward(WalkForwardConfig(
        experiment_tag=TAG,
        experiment_root=ROOT,
        stock_ids=stock_ids,
        reward_mode="empirical",
        reward_config=empirical_reward_config(top_k=5, bottom_k=5),
        warm_chain=True,
        seed=SEED,
    ))
