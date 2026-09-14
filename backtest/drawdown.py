from __future__ import annotations

import numpy as np


def smooth_peak_logsumexp(nav: np.ndarray, temperature: float) -> np.ndarray:
    x = np.asarray(nav, dtype=np.float64).ravel()
    t = float(temperature)
    out = np.empty_like(x)
    m = x[0]
    out[0] = m
    for i in range(1, len(x)):
        m = np.logaddexp(t * m, t * x[i]) / t
        out[i] = m
    return out


def smooth_dd_to_date(hist_net: np.ndarray, temperature: float) -> float:
    """Causal SmoothMDD using only already-realized net returns."""
    x = np.asarray(hist_net, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0
    v = np.cumsum(x)
    peak = smooth_peak_logsumexp(v, temperature)
    return float(np.maximum(peak - v, 0.0).max())


def hard_mdd(hist_net: np.ndarray) -> float:
    x = np.asarray(hist_net, dtype=np.float64)
    if x.size == 0:
        return 0.0
    nav = np.cumprod(1.0 + np.expm1(x)) if False else np.exp(np.cumsum(x))
    peak = np.maximum.accumulate(nav)
    return float(np.maximum(1.0 - nav / np.maximum(peak, 1e-12), 0.0).max())
