"""
Walk-forward GRPO — random30, empirical reward, rolling 3d-train → 1d-OOS.

Universe: same 30 stocks as ``walkforward_202604_empirical_random30`` (seed=42).
Reward: fixed empirical weights (same as random30 experiment).
Schedule: each trading day in [2026-04-01, 2026-05-12] is predicted using a model
trained on the prior 3 trading days (train window constrained to [2026-04-01, 2026-05-11]).

Note: minute data currently ends on the last available trading day in the pool
(often 2026-05-08); dates beyond that are skipped automatically.

Artifacts:
  files/experiments/walkforward_202604_empirical_random30_roll1d/
"""

import sys
import warnings
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "files"))
warnings.filterwarnings("ignore")

from reward_model import empirical_reward_config
from walkforward import WalkForwardConfig, run_walkforward

TAG = "walkforward_202604_empirical_random30_roll1d"
ROOT = Path(__file__).resolve().parent / "files" / "experiments" / TAG
UNIVERSE_CSV = (
    Path(__file__).resolve().parent
    / "files"
    / "experiments"
    / "walkforward_202604_empirical_random30"
    / "universe"
    / "random30_seed42.csv"
)

if __name__ == "__main__":
    ROOT.mkdir(parents=True, exist_ok=True)
    stock_ids = pd.read_csv(UNIVERSE_CSV)["stock_id"].tolist()
    print(f"Random-30 rolling-1d experiment -> {ROOT}")
    print(f"Universe ({len(stock_ids)}): {', '.join(stock_ids[:5])} ...\n")

    run_walkforward(WalkForwardConfig(
        experiment_tag=TAG,
        experiment_root=ROOT,
        stock_ids=stock_ids,
        reward_mode="empirical",
        reward_config=empirical_reward_config(top_k=5, bottom_k=5),
        schedule_mode="rolling_1d",
        train_days=3,
        oos_days=1,
        train_cal_start="2026-04-01",
        train_cal_end="2026-05-11",
        oos_cal_start="2026-04-01",
        oos_cal_end="2026-05-12",
        seed=42,
    ))
