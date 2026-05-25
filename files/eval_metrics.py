"""Extended OOS evaluation metrics for walk-forward experiments."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch

from backtest import BARS_PER_YEAR, annualized_stats


def _pearson_vec(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    den = np.linalg.norm(a) * np.linalg.norm(b) + 1e-12
    return float(np.dot(a, b) / den)


def bar_level_correlations(
    scores: np.ndarray,
    weights: np.ndarray,
    returns: np.ndarray,
) -> Dict[str, float]:
    """
    Mean per-bar Pearson correlation between signals and realized returns.

    scores:   (T, N)
    weights:  (T, N)
    returns:  (T, N)
    """
    T = scores.shape[0]
    score_ics, weight_ics = [], []
    for t in range(T):
        score_ics.append(_pearson_vec(scores[t], returns[t]))
        weight_ics.append(_pearson_vec(weights[t], returns[t]))
    return {
        "score_return_corr": float(np.mean(score_ics)),
        "weight_return_corr": float(np.mean(weight_ics)),
    }


def summarize_oos_comparison(stitched: pd.DataFrame) -> Dict[str, float]:
    """Absolute / excess return and return correlations for GRPO vs EW."""
    grpo = stitched["net_pr"].to_numpy(dtype=np.float64)
    ew = stitched["ew_net_pr"].to_numpy(dtype=np.float64)
    excess = grpo - ew

    st_grpo = annualized_stats(grpo, BARS_PER_YEAR)
    st_ew = annualized_stats(ew, BARS_PER_YEAR)
    st_ex = annualized_stats(excess, BARS_PER_YEAR)

    out = {
        "grpo_abs_return": st_grpo.get("total_return", 0.0),
        "ew_abs_return": st_ew.get("total_return", 0.0),
        "excess_return": st_grpo.get("total_return", 0.0) - st_ew.get("total_return", 0.0),
        "grpo_sharpe": st_grpo.get("annualized_sharpe", 0.0),
        "ew_sharpe": st_ew.get("annualized_sharpe", 0.0),
        "excess_sharpe": st_ex.get("annualized_sharpe", 0.0),
        "grpo_max_dd": st_grpo.get("max_drawdown", 0.0),
        "ew_max_dd": st_ew.get("max_drawdown", 0.0),
    }
    if "score_return_corr" in stitched.columns:
        out["score_return_corr"] = float(stitched["score_return_corr"].mean())
    if "weight_return_corr" in stitched.columns:
        out["weight_return_corr"] = float(stitched["weight_return_corr"].mean())
    return out
