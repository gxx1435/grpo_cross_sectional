from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from backtest.drawdown import smooth_dd_to_date
from backtest.execution import execute_day, execute_day_by_id
from backtest.metrics import causal_sharpe


def run_path(
    dates: List[pd.Timestamp],
    weights: List[np.ndarray],
    y_list: List[np.ndarray],
    cfg: dict,
    extra: Optional[List[dict]] = None,
    asset_ids: Optional[List[List[str]]] = None,
) -> pd.DataFrame:
    prev = None
    prev_ids: Optional[List[str]] = None
    hist: List[float] = []
    rows = []
    for i, day in enumerate(dates):
        current_w = np.asarray(weights[i], dtype=np.float64)
        current_y = np.asarray(y_list[i], dtype=np.float64)
        if asset_ids is not None:
            current_ids = [str(x) for x in asset_ids[i]]
            met = execute_day_by_id(current_w, current_y, current_ids, prev, prev_ids, cfg)
            prev_ids = current_ids
        else:
            met = execute_day(current_w, current_y, prev, cfg)
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
        prev = current_w
    return pd.DataFrame(rows)
