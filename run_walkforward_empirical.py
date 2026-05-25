"""
Walk-forward GRPO with **fixed empirical reward weights** (no val grid-search).

Artifacts go to a separate folder from the auto-learn run:
  files/experiments/walkforward_202604_empirical/
"""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "files"))
warnings.filterwarnings("ignore")

from reward_model import empirical_reward_config
from walkforward import WalkForwardConfig, run_walkforward

TAG = "walkforward_202604_empirical"
ROOT = Path(__file__).resolve().parent / "files" / "experiments" / TAG

if __name__ == "__main__":
    ROOT.mkdir(parents=True, exist_ok=True)
    print(f"Empirical-reward experiment -> {ROOT}\n")
    run_walkforward(WalkForwardConfig(
        experiment_tag=TAG,
        experiment_root=ROOT,
        reward_mode="empirical",
        reward_config=empirical_reward_config(top_k=5, bottom_k=5),
    ))
