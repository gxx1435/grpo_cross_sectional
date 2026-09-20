from __future__ import annotations

import numpy as np


def project_simplex(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).ravel()
    n = v.size
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - 1))[0]
    if len(rho) == 0:
        return np.ones(n) / n
    rho = int(rho[-1])
    theta = (cssv[rho] - 1.0) / (rho + 1.0)
    w = np.maximum(v - theta, 0.0)
    s = w.sum()
    return w / s if s > 0 else np.ones(n) / n


def apply_valid_mask(w: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Zero invalid names and renormalize onto the simplex."""
    w = np.asarray(w, dtype=np.float64).copy()
    m = np.asarray(valid, dtype=bool).ravel()
    if m.shape[0] != w.shape[0]:
        raise ValueError("valid mask length must match weights")
    w[~m] = 0.0
    s = float(w.sum())
    if s <= 1e-12:
        if m.any():
            w[m] = 1.0 / float(m.sum())
        return w
    return w / s


def cap_and_simplex(w: np.ndarray, max_weight: float) -> np.ndarray:
    """Euclidean projection onto {w >= 0, sum(w)=1, w <= max_weight}."""
    values = np.asarray(w, dtype=np.float64).ravel()
    cap = float(max_weight)
    if values.size * cap < 1.0 - 1e-12:
        raise ValueError(f"infeasible max_weight={cap} for n={values.size}")
    if not np.isfinite(values).all():
        values = np.ones_like(values)
    lo = float(np.min(values - cap))
    hi = float(np.max(values))
    for _ in range(100):
        theta = 0.5 * (lo + hi)
        candidate = np.clip(values - theta, 0.0, cap)
        if candidate.sum() > 1.0:
            lo = theta
        else:
            hi = theta
    projected = np.clip(values - hi, 0.0, cap)
    residual = 1.0 - float(projected.sum())
    if abs(residual) > 1e-10:
        room = cap - projected if residual > 0 else projected
        eligible = room > 1e-14
        projected[eligible] += residual * room[eligible] / float(room[eligible].sum())
    return projected


def sanitize_cov(sigma: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    sigma = np.asarray(sigma, dtype=np.float64)
    sigma = 0.5 * (sigma + sigma.T)
    return sigma + eps * np.eye(sigma.shape[0])
