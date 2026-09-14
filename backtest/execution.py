from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from backtest.transaction_cost import cost_from_turnover, turnover_l1


def execute_day(
    w: np.ndarray,
    y_intraday: np.ndarray,
    prev_w: Optional[np.ndarray],
    cfg: dict,
) -> Dict[str, float]:
    w = np.asarray(w, dtype=np.float64).ravel()
    y = np.asarray(y_intraday, dtype=np.float64).ravel()
    if cfg["portfolio"]["return_type"] != "log":
        raise RuntimeError("config must keep return_type=log unless all modules switch together")
    gross = float(w @ y)
    turn = turnover_l1(w, prev_w, cfg["portfolio"]["initial_turnover_mode"])
    cost = cost_from_turnover(turn, cfg["portfolio"]["cost_bps"])
    return {
        "gross_return": gross,
        "turnover": turn,
        "transaction_cost": cost,
        "net_return": gross - cost,
        "simplex_sum": float(w.sum()),
        "min_weight": float(w.min()),
        "max_weight": float(w.max()),
    }
