from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd


def select_topk(alpha: np.ndarray, valid: np.ndarray, k: int) -> np.ndarray:
    a = np.asarray(alpha, dtype=np.float64).copy()
    a[~np.asarray(valid, dtype=bool)] = -1e18
    k = min(int(k), int(valid.sum()))
    if k <= 0:
        raise RuntimeError("empty valid universe")
    return np.argpartition(-a, k - 1)[:k]


def topk_table(codes: List[str], alpha: np.ndarray, y: np.ndarray, idx: np.ndarray, valid: np.ndarray) -> pd.DataFrame:
    rank = np.full(len(codes), -1, dtype=int)
    order = np.argsort(-np.asarray(alpha)[idx])
    for r, j in enumerate(idx[order]):
        rank[int(j)] = r + 1
    return pd.DataFrame(
        {
            "stock_code": codes,
            "predicted_alpha": alpha,
            "true_intraday_return": y,
            "rank": rank,
            "selected_flag": np.isin(np.arange(len(codes)), idx).astype(int),
            "valid_flag": valid.astype(int),
            "valid_universe_size": int(valid.sum()),
        }
    )
