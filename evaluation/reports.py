from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from evaluation.plots import plot_cum
from evaluation.portfolio_metrics import summarize_nav
from utils.logging import write_json


def write_month_reports(out: Path, daily: pd.DataFrame, pred: pd.DataFrame, extra: Dict[str, Any]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    if len(daily):
        daily.to_csv(out / "backtest_daily.csv", index=False)
        summarize_nav(daily).to_csv(out / "performance_summary.csv", index=False)
        write_json(out / "performance_summary.json", summarize_nav(daily).to_dict(orient="records"))
        plot_cum(daily, out / "cumulative_return.png", extra.get("title", "NAV"))
    if len(pred):
        pred.to_csv(out / "prediction_metrics.csv", index=False)
    write_json(out / "extra.json", extra)


def merge_and_compare(paths: List[Path], dest: Path, title: str) -> None:
    frames = [pd.read_csv(p) for p in paths if p.is_file()]
    if not frames:
        return
    df = pd.concat(frames, ignore_index=True)
    dest.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest / "backtest_daily.csv", index=False)
    summarize_nav(df).to_csv(dest / "performance_summary.csv", index=False)
    plot_cum(df, dest / "cumulative_return.png", title)
