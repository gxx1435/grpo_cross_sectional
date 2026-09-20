from __future__ import annotations

import torch


def tangent_project(v: torch.Tensor) -> torch.Tensor:
    """P = I - 11^T/K → sum(v_tangent)=0."""
    out = v - v.mean(dim=-1, keepdim=True)
    # Remove the final floating-point accumulation residual as well. This is
    # mathematically the same tangent projection and materially tightens the
    # simplex audit for float32 tensors.
    return out - out.sum(dim=-1, keepdim=True) / out.size(-1)


def simplex_violation(w: torch.Tensor, eps: float = 1e-5) -> dict:
    s = w.sum(dim=-1)
    return {
        "sum_abs_err": float((s - 1.0).abs().max()),
        "min_weight": float(w.min()),
        "max_weight": float(w.max()),
        "neg_rate": float((w < -eps).float().mean()),
        "ok": bool(((s - 1.0).abs() < 1e-3).all() and (w >= -eps).all()),
    }
