"""Stable SS-FM Meta: dated seeds + pred_utility majority / expectation."""

from __future__ import annotations

import hashlib
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch

from flow_matching.ssfm import sample_ss_fm
from portfolio.teacher_portfolios import TEACHER_NAMES


def date_seed(asof, salt: int = 0) -> int:
    """Deterministic seed from calendar date (+ optional salt)."""
    if hasattr(asof, "date"):
        d = asof.date().isoformat()
    else:
        d = str(asof)[:10]
    h = hashlib.md5(f"{d}:{int(salt)}".encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def _seeded_noise(g: int, k: int, seed: int, device) -> torch.Tensor:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed) & 0xFFFFFFFF)
    z = torch.randn(int(g), int(k), generator=gen)
    return z.to(device=device, dtype=torch.float32)


def sample_ss_fm_mixed_seeded(
    model,
    cond: torch.Tensor,
    n_samples: int,
    n_steps: int,
    seed: int,
) -> torch.Tensor:
    """Same split-by-teacher as sample_ss_fm_mixed, but with fixed noise seed."""
    g, k = int(n_samples), int(model.n_assets)
    noise = _seeded_noise(g, k, seed, cond.device)
    chunks: List[torch.Tensor] = []
    rem = g
    per = max(g // model.n_teachers, 1)
    offset = 0
    for tid in range(model.n_teachers):
        gi = per if tid < model.n_teachers - 1 else rem
        if gi <= 0:
            break
        chunks.append(
            sample_ss_fm(
                model,
                cond,
                tid,
                gi,
                n_steps,
                seed_noise=noise[offset : offset + gi],
            )
        )
        offset += gi
        rem -= gi
    return torch.cat(chunks, dim=0)[:g]


def pred_utility_scores(ws: np.ndarray, alpha: np.ndarray, sigma: np.ndarray, risk_aversion: float) -> np.ndarray:
    ra = float(risk_aversion)
    a = np.asarray(alpha, dtype=np.float64).reshape(-1)
    s = np.asarray(sigma, dtype=np.float64)
    out = []
    for w in np.asarray(ws, dtype=np.float64):
        out.append(float(w @ a - 0.5 * ra * w @ s @ w))
    return np.asarray(out, dtype=np.float64)


def build_meta_pool_np(
    samples: np.ndarray,
    teachers: Optional[dict],
    teacher_names: Sequence[str] = TEACHER_NAMES,
) -> np.ndarray:
    parts = [np.asarray(samples, dtype=np.float64)]
    if teachers:
        for tn in teacher_names:
            if tn not in teachers:
                continue
            w = np.asarray(teachers[tn], dtype=np.float64).reshape(-1)
            if w.size == parts[0].shape[-1]:
                parts.append(w.reshape(1, -1))
    return np.concatenate(parts, axis=0) if len(parts) > 1 else parts[0]


def _weight_key(w: np.ndarray, nd: int = 4) -> tuple:
    return tuple(np.round(np.asarray(w, dtype=np.float64), nd).tolist())


def aggregate_candidates(
    ws: np.ndarray,
    alpha: np.ndarray,
    sigma: np.ndarray,
    risk_aversion: float,
    mode: str,
    top_k: int = 3,
    temperature: float = 1.0,
) -> np.ndarray:
    """Aggregate G SS-FM samples into one weight vector.

    Modes:
      - pred_utility / argmax: pick max pred_utility
      - mean / mean_utility: equal-weight average of all candidates
      - topk: average of top-k by pred_utility
      - softmax: Σ softmax(u/τ) · w_i
    """
    arr = np.asarray(ws, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    scores = pred_utility_scores(arr, alpha, sigma, risk_aversion)
    mode = str(mode or "pred_utility").lower().strip()

    if mode in ("pred_utility", "argmax", "select", "max"):
        w = arr[int(np.argmax(scores))]
    elif mode in ("mean", "mean_utility", "average", "avg"):
        w = arr.mean(axis=0)
    elif mode in ("topk", "top_k", "top-k"):
        k = max(1, min(int(top_k), arr.shape[0]))
        idx = np.argsort(scores)[::-1][:k]
        w = arr[idx].mean(axis=0)
    elif mode in ("softmax", "softmax_utility", "soft"):
        t = max(float(temperature), 1e-8)
        z = scores / t
        z = z - float(np.max(z))
        p = np.exp(z)
        p = p / max(float(p.sum()), 1e-12)
        w = (p[:, None] * arr).sum(axis=0)
    else:
        raise ValueError(f"unknown aggregate mode: {mode}")

    w = np.clip(w, 0, None)
    return w / max(float(w.sum()), 1e-12)


def sample_and_aggregate(
    model,
    cond: torch.Tensor,
    item: dict,
    cfg: dict,
    g: int,
    mode: str,
    top_k: int = 3,
    temperature: float = 1.0,
    salt: int = 0,
) -> np.ndarray:
    """Dated-seed SS-FM sample (G) then aggregate. No teachers in the pool."""
    n_steps = int(cfg["ssfm"]["n_sample_steps"])
    seed = date_seed(item["asof"], salt)
    with torch.no_grad():
        ws = sample_ss_fm_mixed_seeded(model, cond, g, n_steps, seed).detach().cpu().numpy()
    alpha = np.asarray(item["alpha"][item["idx"]], dtype=np.float64)
    sigma = np.asarray(item["sigma"], dtype=np.float64)
    ra = float(cfg["ssfm"]["risk_aversion"])
    return aggregate_candidates(ws, alpha, sigma, ra, mode=mode, top_k=top_k, temperature=temperature)


def aggregate_picks(
    picks: List[np.ndarray],
    utilities: List[float],
    mode: str = "majority",
) -> np.ndarray:
    """majority = mode of quantized weights; expectation = mean of picks."""
    mode = str(mode or "majority").lower()
    arr = [np.asarray(w, dtype=np.float64).reshape(-1) for w in picks]
    if not arr:
        raise ValueError("empty picks")
    if mode in ("expectation", "mean", "avg", "average"):
        w = np.mean(np.stack(arr, axis=0), axis=0)
    else:
        # majority vote on quantized weights; tie-break by utility
        buckets: dict = {}
        for w, u in zip(arr, utilities):
            key = _weight_key(w)
            if key not in buckets:
                buckets[key] = {"count": 0, "util": -1e30, "w": w}
            buckets[key]["count"] += 1
            if float(u) > buckets[key]["util"]:
                buckets[key]["util"] = float(u)
                buckets[key]["w"] = w
        best = max(buckets.values(), key=lambda b: (b["count"], b["util"]))
        w = np.asarray(best["w"], dtype=np.float64)
    w = np.clip(w, 0, None)
    return w / max(float(w.sum()), 1e-12)


def stable_meta_select(
    model,
    cond: torch.Tensor,
    item: dict,
    cfg: dict,
    g: int,
    n_rounds: int = 5,
    agg: str = "majority",
    include_teachers: bool = True,
) -> Tuple[np.ndarray, dict]:
    """Reproducible Meta: dated seeds + majority/expectation over pred_utility picks."""
    n_steps = int(cfg["ssfm"]["n_sample_steps"])
    ra = float(cfg["ssfm"]["risk_aversion"])
    alpha = np.asarray(item["alpha"][item["idx"]], dtype=np.float64)
    sigma = np.asarray(item["sigma"], dtype=np.float64)
    teachers = item.get("teachers") if include_teachers else None
    picks: List[np.ndarray] = []
    utils: List[float] = []
    n_rounds = max(int(n_rounds), 1)
    with torch.no_grad():
        for r in range(n_rounds):
            seed = date_seed(item["asof"], r)
            ws = sample_ss_fm_mixed_seeded(model, cond, g, n_steps, seed).detach().cpu().numpy()
            pool = build_meta_pool_np(ws, teachers)
            scores = pred_utility_scores(pool, alpha, sigma, ra)
            j = int(np.argmax(scores))
            picks.append(pool[j])
            utils.append(float(scores[j]))
    w = aggregate_picks(picks, utils, mode=agg)
    info = {
        "n_rounds": n_rounds,
        "agg": agg,
        "util_mean": float(np.mean(utils)),
        "util_std": float(np.std(utils)),
        "seed0": date_seed(item["asof"], 0),
    }
    return w, info


def stable_meta_pool_tensor(
    model,
    cond: torch.Tensor,
    item: dict,
    cfg: dict,
    g: int,
    device,
    include_teachers: bool = True,
    salt: int = 0,
) -> torch.Tensor:
    """Single dated Meta pool (samples ∪ teachers) for RL candidate features."""
    n_steps = int(cfg["ssfm"]["n_sample_steps"])
    seed = date_seed(item["asof"], salt)
    with torch.no_grad():
        ws = sample_ss_fm_mixed_seeded(model, cond, g, n_steps, seed)
    parts = [ws]
    if include_teachers:
        teachers = item.get("teachers") or {}
        for tn in TEACHER_NAMES:
            if tn not in teachers:
                continue
            w = np.asarray(teachers[tn], dtype=np.float32).reshape(-1)
            if w.size == ws.shape[-1]:
                parts.append(torch.from_numpy(w).to(device=device, dtype=ws.dtype).unsqueeze(0))
    return torch.cat(parts, dim=0) if len(parts) > 1 else ws


def meta_prior_logits(pool: torch.Tensor, item: dict, cfg: dict, temperature: float = 1.0) -> torch.Tensor:
    """Soft prior over candidates from pred_utility (SS-FM Meta prior)."""
    alpha = np.asarray(item["alpha"][item["idx"]], dtype=np.float64)
    sigma = np.asarray(item["sigma"], dtype=np.float64)
    ra = float(cfg["ssfm"]["risk_aversion"])
    scores = pred_utility_scores(pool.detach().cpu().numpy(), alpha, sigma, ra)
    t = max(float(temperature), 1e-6)
    return torch.tensor(scores / t, device=pool.device, dtype=torch.float32)
