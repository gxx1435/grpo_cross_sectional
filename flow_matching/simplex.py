from __future__ import annotations

import torch


def tangent_project(v: torch.Tensor) -> torch.Tensor:
    """P = I - 11^T/K → sum(v_tangent)=0."""
    return v - v.mean(dim=-1, keepdim=True)


def simplex_violation(w: torch.Tensor, eps: float = 1e-5) -> dict:
    s = w.sum(dim=-1)
    return {
        "sum_abs_err": float((s - 1.0).abs().max()),
        "min_weight": float(w.min()),
        "max_weight": float(w.max()),
        "neg_rate": float((w < -eps).float().mean()),
        "ok": bool(((s - 1.0).abs() < 1e-3).all() and (w >= -eps).all()),
    }
