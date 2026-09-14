from __future__ import annotations

from typing import Dict

import numpy as np


def group_stats(rewards: np.ndarray, weights: np.ndarray, cfg: dict) -> Dict[str, object]:
    r = np.asarray(rewards, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    mu, sd = float(r.mean()), float(r.std() + 1e-12)
    adv = (r - mu) / sd
    ents = []
    for g in range(w.shape[0]):
        p = np.clip(w[g], 1e-12, None)
        p = p / p.sum()
        ents.append(float(-(p * np.log(p)).sum()))
    pair = 0.0
    n = 0
    for i in range(len(w)):
        for j in range(i + 1, len(w)):
            pair += 0.5 * float(np.abs(w[i] - w[j]).sum())
            n += 1
    weak_var = sd < float(cfg["rl"]["grpo"]["weak_var_threshold"])
    weak_adv = float(np.abs(adv).mean()) < float(cfg["rl"]["grpo"]["weak_adv_threshold"])
    out = {
        "group_reward_mean": mu,
        "group_reward_std": sd,
        "group_reward_range": float(r.max() - r.min()),
        "advantage_magnitude": float(np.abs(adv).mean()),
        "group_relative_reward_variance": float(r.var()),
        "portfolio_diversity": float(np.mean(ents)) if ents else 0.0,
        "reward_diversity": pair / max(n, 1),
        "weak_signal": bool(weak_var or weak_adv),
    }
    if out["weak_signal"]:
        out["warning"] = "GRPO group-relative signal weak"
    return out
