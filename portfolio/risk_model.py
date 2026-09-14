from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd

from portfolio.constraints import sanitize_cov


def hist_mu_sigma(daily: pd.DataFrame, asof: pd.Timestamp, names: List[str], lookback: int) -> Tuple[np.ndarray, np.ndarray]:
    """μ, Σ from completed days strictly before `asof` (D-1 close and earlier)."""
    asof = pd.Timestamp(asof).normalize()
    hist = daily.loc[daily.index < asof, names].tail(int(lookback))
    if len(hist) < 5:
        raise RuntimeError(f"insufficient hist risk window before {asof.date()}")
    mu = hist.mean().to_numpy(np.float64)
    sigma = sanitize_cov(hist.cov().to_numpy(np.float64))
    return mu, sigma
