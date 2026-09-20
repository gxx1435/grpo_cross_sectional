"""Replay a saved strict run, or execute a strict monthly pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from backtest.metrics import summarize_nav
from evaluation.plots import plot_cum
from experiments.run_all import run_all
from utils.config import load_config, resolve_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-month", default="2025-06")
    parser.add_argument("--from-dir", default="")
    args = parser.parse_args()
    cfg = load_config()
    out = resolve_path(cfg, cfg["paths"]["results_dir"])
    if args.from_dir:
        source = Path(args.from_dir) / "backtest_daily.csv"
        if not source.is_file():
            raise FileNotFoundError(source)
        replay = out / "backtest_replay"
        replay.mkdir(parents=True, exist_ok=True)
        daily = pd.read_csv(source)
        summarize_nav(daily).to_csv(replay / "performance_summary.csv", index=False)
        plot_cum(daily, replay / "cumulative_return.png", "strict OOS replay")
        return
    run_all(cfg, out, months_filter=[args.test_month])


if __name__ == "__main__":
    main()
