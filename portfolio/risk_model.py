from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd

from portfolio.constraints import sanitize_cov


def hist_mu_sigma(daily: pd.DataFrame, asof: pd.Timestamp, names: List[str], lookback: int, covariance_epsilon: float) -> Tuple[np.ndarray, np.ndarray]:
    """μ, Σ from completed days strictly before `asof` (D-1 close and earlier)."""
    asof = pd.Timestamp(asof).normalize()
    hist = daily.loc[daily.index < asof, names].tail(int(lookback))
    if len(hist) < 5:
        raise RuntimeError(f"insufficient hist risk window before {asof.date()}")
    # Only past observations are used. Missing historical returns are replaced
    # by that security's past-window mean (and zero only when no past mean
    # exists), never by a future value.
    means = hist.mean()
    clean = hist.fillna(means).fillna(0.0)
    mu = clean.mean().to_numpy(np.float64)
    sigma = sanitize_cov(clean.cov().fillna(0.0).to_numpy(np.float64), float(covariance_epsilon))
    return mu, sigma
