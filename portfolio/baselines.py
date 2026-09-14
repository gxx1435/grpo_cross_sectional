from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from portfolio.constraints import cap_and_simplex, project_simplex, sanitize_cov
from portfolio.teacher_portfolios import max_sharpe, mvo, risk_parity


def equal_weight(k: int) -> np.ndarray:
    return np.ones(k, dtype=np.float64) / k


def score_softmax(alpha: np.ndarray, valid: Optional[np.ndarray] = None, tau: float = 0.5) -> np.ndarray:
    """Long-only tilt over the whole pool: softmax(z(alpha)/τ), invalid names get 0."""
    a = np.asarray(alpha, dtype=np.float64).ravel()
    if valid is None:
        m = np.isfinite(a)
    else:
        m = np.asarray(valid, dtype=bool).ravel() & np.isfinite(a)
    w = np.zeros_like(a)
    if int(m.sum()) < 2:
        if m.any():
            w[m] = 1.0 / float(m.sum())
        return w
    x = a[m]
    z = (x - float(x.mean())) / (float(x.std()) + 1e-6)
    z = z / max(float(tau), 1e-6)
    z = z - float(z.max())
    e = np.exp(z)
    w[m] = e / float(e.sum())
    return w


def kelly(mu: np.ndarray, sigma: np.ndarray, max_w: float) -> np.ndarray:
    raw = np.linalg.pinv(sanitize_cov(sigma)) @ np.asarray(mu, dtype=np.float64)
    if not np.isfinite(raw).all():
        raw = np.ones_like(mu)
    return cap_and_simplex(project_simplex(raw), max_w)


def black_litterman(mu: np.ndarray, sigma: np.ndarray, views: np.ndarray, max_w: float, tau: float = 0.05) -> np.ndarray:
    """Views = predicted alpha only. No future realized returns."""
    sigma = sanitize_cov(sigma)
    k = mu.size
    pi = np.asarray(mu, dtype=np.float64)
    q = np.asarray(views, dtype=np.float64)
    omega = np.diag(np.maximum(np.diag(sigma) * 0.25, 1e-8))
    tau_s = float(tau) * sigma
    mid = np.linalg.pinv(np.linalg.pinv(tau_s) + np.linalg.pinv(omega))
    mu_bl = mid @ (np.linalg.pinv(tau_s) @ pi + np.linalg.pinv(omega) @ q)
    return mvo(mu_bl, sigma, 1.0, max_w)


def all_baselines(mu: np.ndarray, sigma: np.ndarray, alpha: Optional[np.ndarray], max_w: float) -> Dict[str, np.ndarray]:
    views = alpha if alpha is not None else mu
    return {
        "equal_weight": equal_weight(len(mu)),
        "mvo": mvo(mu, sigma, 1.0, max_w),
        "max_sharpe": max_sharpe(mu, sigma, max_w),
        "risk_parity": risk_parity(sigma, max_w),
        "black_litterman": black_litterman(mu, sigma, views, max_w),
        "kelly": kelly(mu, sigma, max_w),
    }
