from __future__ import annotations

from typing import Dict

import numpy as np

from portfolio.constraints import cap_and_simplex, project_simplex, sanitize_cov

TEACHER_NAMES = ("mvo", "max_sharpe", "risk_parity")
TEACHER_IDS = {"mvo": 0, "max_sharpe": 1, "risk_parity": 2}


def mvo(mu: np.ndarray, sigma: np.ndarray, risk_aversion: float, max_w: float) -> np.ndarray:
    sigma = sanitize_cov(sigma)
    raw = np.linalg.pinv(sigma) @ np.asarray(mu, dtype=np.float64) / max(float(risk_aversion), 1e-6)
    if not np.isfinite(raw).all() or np.allclose(raw, 0):
        raw = np.ones_like(mu)
    return cap_and_simplex(project_simplex(raw), max_w)


def max_sharpe(mu: np.ndarray, sigma: np.ndarray, max_w: float, n_grid: int = 40) -> np.ndarray:
    best_w, best = None, -1e18
    for lam in np.geomspace(0.05, 20.0, n_grid):
        w = mvo(mu, sigma, float(lam), max_w)
        sh = float(w @ mu) / np.sqrt(max(float(w @ sigma @ w), 1e-12))
        if sh > best:
            best, best_w = sh, w
    return best_w if best_w is not None else mvo(mu, sigma, 1.0, max_w)


def risk_parity(sigma: np.ndarray, max_w: float, n_iter: int = 200) -> np.ndarray:
    sigma = sanitize_cov(sigma)
    k = sigma.shape[0]
    w = np.ones(k) / k
    for _ in range(n_iter):
        sig_w = sigma @ w
        var = float(w @ sig_w)
        if var <= 1e-18:
            break
        w = w * (var / k) / np.clip(sig_w, 1e-12, None)
        w = project_simplex(w)
    return cap_and_simplex(w, max_w)


def teachers(mu: np.ndarray, sigma: np.ndarray, max_w: float, risk_aversion: float = 1.0) -> Dict[str, np.ndarray]:
    return {
        "mvo": mvo(mu, sigma, risk_aversion, max_w),
        "max_sharpe": max_sharpe(mu, sigma, max_w),
        "risk_parity": risk_parity(sigma, max_w),
    }
