"""
Walk-forward GRPO — Top30 YTD, **long-only Top5** (no short leg).

Same setup as ``walkforward_202604`` (auto-learn reward, 3d train → 3d OOS),
but portfolio is 100% long in Top-5 scored stocks (20% each). Baseline is
30-stock equal-weight long-only.

Artifacts:
  files/experiments/walkforward_202604_longonly/
"""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "files"))
warnings.filterwarnings("ignore")

from walkforward import WalkForwardConfig, run_walkforward

TAG = "walkforward_202604_longonly"
ROOT = Path(__file__).resolve().parent / "files" / "experiments" / TAG

if __name__ == "__main__":
    ROOT.mkdir(parents=True, exist_ok=True)
    print(f"Long-only Top5 GRPO experiment -> {ROOT}\n")

    run_walkforward(WalkForwardConfig(
        experiment_tag=TAG,
        experiment_root=ROOT,
        portfolio_mode="long_only",
        baseline_mode="long_only",
        top_k=5,
        bottom_k=0,
        reward_mode="auto",
        seed=42,
    ))
