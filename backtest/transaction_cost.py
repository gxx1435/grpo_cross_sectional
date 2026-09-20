from __future__ import annotations

from typing import Optional

import numpy as np


def turnover_l1(w: np.ndarray, prev: Optional[np.ndarray], initial_mode: str) -> float:
    w = np.asarray(w, dtype=np.float64).ravel()
    if prev is None:
        if initial_mode in ("ignore_first", "ignore_first_trade"):
            return 0.0
        return 0.5 * float(np.abs(w).sum())
    return 0.5 * float(np.abs(w - np.asarray(prev, dtype=np.float64).ravel()).sum())


def cost_from_turnover(turnover: float, rate: float) -> float:
    return float(turnover) * float(rate)
