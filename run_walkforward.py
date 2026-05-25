"""Entry point: walk-forward GRPO experiment (isolated artifact dirs)."""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "files"))
warnings.filterwarnings("ignore")

from walkforward import run_walkforward, WalkForwardConfig, EXPERIMENT_ROOT

if __name__ == "__main__":
    EXPERIMENT_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"All artifacts will be saved under: {EXPERIMENT_ROOT}")
    print("(separate from files/checkpoints/ and files/backtest_out/)\n")
    run_walkforward(WalkForwardConfig())
