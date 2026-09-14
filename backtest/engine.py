from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from backtest.drawdown import smooth_dd_to_date
from backtest.execution import execute_day
from backtest.metrics import causal_sharpe


def run_path(
    dates: List[pd.Timestamp],
    weights: List[np.ndarray],
    y_list: List[np.ndarray],
    cfg: dict,
    extra: Optional[List[dict]] = None,
) -> pd.DataFrame:
    prev = None
    hist: List[float] = []
    rows = []
    for i, day in enumerate(dates):
        met = execute_day(weights[i], y_list[i], prev, cfg)
        sharpe = causal_sharpe(hist, cfg["portfolio"]["sharpe_min_obs"], cfg["portfolio"]["sharpe_fallback"])
        smdd = smooth_dd_to_date(np.asarray(hist), cfg["portfolio"]["smooth_dd_temperature"])
        row = {
            "date": str(pd.Timestamp(day).date()),
            **met,
            "sharpe": sharpe,
            "drawdown": smdd,
        }
        if extra:
            row.update(extra[i])
        rows.append(row)
        hist.append(met["net_return"])
        prev = np.asarray(weights[i], dtype=np.float64)
    return pd.DataFrame(rows)
