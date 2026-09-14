"""Replay saved daily portfolios if present; otherwise run one-month backtest."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

from backtest.metrics import summarize_nav
from data.store import ResearchStore
from evaluation.plots import plot_cum
from experiments.engine import run_month
from utils.config import load_config, resolve_path
from utils.logging import Tee, log, write_json


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-month", default="2025-05")
    ap.add_argument("--oos", default="strict_fixed_oos")
    ap.add_argument("--from-dir", default="")
    args = ap.parse_args()
    cfg = load_config()
    out = resolve_path(cfg, cfg["paths"]["results_dir"]) / "entry_run_backtest"
    tee = Tee(out / "run.log")
    sys.stdout = tee  # type: ignore
    sys.stderr = tee  # type: ignore
    if args.from_dir:
        src = Path(args.from_dir) / "backtest_daily.csv"
        if not src.is_file():
            raise FileNotFoundError(src)
        df = pd.read_csv(src)
        summarize_nav(df).to_csv(out / "performance_summary.csv", index=False)
        plot_cum(df, out / "cumulative_return.png", "replay")
        log(f"replayed {src} -> {out}")
        return
    store = ResearchStore(cfg)
    rec = run_month(
        cfg, store, pd.Timestamp(args.test_month), args.oos, out,
        models_filter=[cfg["prediction"]["primary_model"]], skip_rl=True, skip_gen=True,
    )
    write_json(out / "replay_index.json", {"experiment_id": rec["experiment_id"]})


if __name__ == "__main__":
    main()
