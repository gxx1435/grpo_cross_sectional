from __future__ import annotations

from typing import Dict, Optional, Sequence

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
    turn = turnover_l1(w, prev_w, cfg["portfolio"]["initial_turnover_policy"])
    cost = cost_from_turnover(turn, cfg["portfolio"]["transaction_cost_rate"])
    return {
        "gross_return": gross,
        "turnover": turn,
        "transaction_cost": cost,
        "net_return": gross - cost,
        "simplex_sum": float(w.sum()),
        "min_weight": float(w.min()),
        "max_weight": float(w.max()),
    }


def execute_day_by_id(
    w: np.ndarray,
    y_intraday: np.ndarray,
    asset_ids: Sequence[str],
    prev_w: Optional[np.ndarray],
    prev_asset_ids: Optional[Sequence[str]],
    cfg: dict,
) -> Dict[str, float]:
    """Execute a changing Top-K portfolio with turnover aligned by security ID."""
    current_w = np.asarray(w, dtype=np.float64).ravel()
    current_y = np.asarray(y_intraday, dtype=np.float64).ravel()
    current_ids = [str(x) for x in asset_ids]
    if len(current_ids) != len(current_w) or len(current_y) != len(current_w):
        raise RuntimeError("asset_ids, weights, and realized returns must have equal length")
    if prev_w is None or prev_asset_ids is None:
        return execute_day(current_w, current_y, None, cfg)
    previous_w = np.asarray(prev_w, dtype=np.float64).ravel()
    previous_ids = [str(x) for x in prev_asset_ids]
    if len(previous_ids) != len(previous_w):
        raise RuntimeError("previous asset_ids and weights must have equal length")
    union = list(dict.fromkeys([*previous_ids, *current_ids]))
    location = {code: j for j, code in enumerate(union)}
    aligned_w = np.zeros(len(union), dtype=np.float64)
    aligned_y = np.zeros(len(union), dtype=np.float64)
    aligned_prev = np.zeros(len(union), dtype=np.float64)
    for code, value, realized in zip(current_ids, current_w, current_y):
        j = location[code]
        aligned_w[j] = value
        aligned_y[j] = realized
    for code, value in zip(previous_ids, previous_w):
        aligned_prev[location[code]] = value
    return execute_day(aligned_w, aligned_y, aligned_prev, cfg)
