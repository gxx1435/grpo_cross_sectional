"""Run data preparation and the complete strict fixed-OOS S&P 500 pipeline."""

from __future__ import annotations

import argparse
import copy
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTHONUNBUFFERED", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from experiments.run_all import run_all
from utils.config import load_config, resolve_path
from utils.gpu import configure_gpu
from utils.logging import Tee, log
from utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--months", nargs="*", default=None, help="optional Test months such as 2025-06")
    parser.add_argument("--prepare-only", action="store_true", help="validate and persist source/features, then stop")
    parser.add_argument("--force-features", action="store_true", help="regenerate version-matched feature partitions")
    parser.add_argument("--smoke", action="store_true", help="real one-month run with reduced training steps")
    args = parser.parse_args()
    cfg = load_config()
    if args.smoke:
        cfg = copy.deepcopy(cfg)
        cfg["prediction"]["epochs"] = 1
        for key in ("ssfm", "standard_fm", "diffusion", "mlp_policy", "gaussian_policy"):
            cfg[key]["steps"] = min(int(cfg[key]["steps"]), 4)
        cfg["ssfm"]["validation_interval"] = 2
        cfg["rl"]["epochs"] = 1
        cfg["rl"]["distill_steps"] = 4
        cfg["runtime"]["train_day_stride"] = 40
        cfg["runtime"]["state_day_stride"] = 40
        if not args.months:
            args.months = ["2025-06"]
        cfg["experiment"]["name"] += "_smoke"
    set_seed(int(cfg["experiment"]["seed"]), bool(cfg["experiment"]["deterministic"]), bool(cfg["experiment"]["deterministic_warn_only"]))
    configure_gpu(cfg)
    out = resolve_path(cfg, cfg["paths"]["results_dir"])
    Tee.install(out / "run.log")
    log(f"S&P500 strict_fixed_oos pipeline -> {out}")
    summary = run_all(cfg, out, months_filter=args.months, prepare_only=args.prepare_only, force_features=args.force_features)
    log(f"pipeline result: {summary}")


if __name__ == "__main__":
    main()
