"""Run a strict monthly pipeline whose first stage trains the alpha predictor."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from experiments.run_all import run_all
from utils.config import load_config, resolve_path
from utils.gpu import configure_gpu
from utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-month", default="2025-06")
    args = parser.parse_args()
    cfg = load_config()
    set_seed(int(cfg["experiment"]["seed"]), True, bool(cfg["experiment"]["deterministic_warn_only"]))
    configure_gpu(cfg)
    run_all(cfg, resolve_path(cfg, cfg["paths"]["results_dir"]), months_filter=[args.test_month])


if __name__ == "__main__":
    main()
