"""Train SS-FM / generative models for one walk-forward month."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

from data.store import ResearchStore
from experiments.engine import run_month
from utils.config import load_config, resolve_path
from utils.gpu_monitor import GpuMonitor
from utils.logging import Tee, log
from utils.runtime import estimate_asdict, estimate_resources
from utils.seed import set_seed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-month", default="2025-05")
    ap.add_argument("--oos", default="strict_fixed_oos", choices=["strict_fixed_oos", "sequential_oos_weekly_retrain"])
    args = ap.parse_args()
    cfg = load_config()
    set_seed(int(cfg["experiment"]["seed"]), bool(cfg["experiment"]["deterministic"]), bool(cfg["experiment"]["deterministic_warn_only"]))
    out = resolve_path(cfg, cfg["paths"]["results_dir"]) / "entry_train_ssfm"
    tee = Tee(out / "run.log")
    sys.stdout = tee  # type: ignore
    sys.stderr = tee  # type: ignore
    log(str(estimate_asdict(estimate_resources(cfg, 1))))
    gpu = GpuMonitor(out / "gpu_logs.csv", float(cfg["gpu"]["monitor_interval_sec"]))
    gpu.start()
    store = ResearchStore(cfg)
    run_month(
        cfg, store, pd.Timestamp(args.test_month), args.oos, out,
        models_filter=[cfg["prediction"]["primary_model"]], skip_rl=True, skip_gen=False,
    )
    log(str(gpu.stop()))


if __name__ == "__main__":
    main()
