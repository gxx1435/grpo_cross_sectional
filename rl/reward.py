from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

from data.normalizers import RobustScaler


def robust_z_with_scaler(x: float, scaler: RobustScaler) -> Dict[str, float]:
    raw = float(x)
    z = float(scaler.transform(np.array([raw])).ravel()[0]) if scaler.median is not None else 0.0
    return {"raw": raw, "z": z, "z_clipped": z}


def fit_reward_scalers(hist: Dict[str, np.ndarray], cfg: dict, fit_range: dict) -> Dict[str, RobustScaler]:
    r = cfg["reward"]
    out = {}
    for name in ("return", "sharpe", "turnover", "smooth_mdd"):
        sc = RobustScaler(eps=float(r["eps"]), zmax=float(r["zmax"]))
        sc.fit(np.asarray(hist[name], dtype=np.float64), fit_range=fit_range)
        out[name] = sc
    return out


def reward_mode(cfg: dict) -> str:
    return str((cfg.get("reward") or {}).get("mode") or "composite").strip().lower()


def composite_reward(components: Dict[str, float], scalers: Dict[str, RobustScaler], cfg: dict) -> Dict[str, object]:
    r = cfg["reward"]
    zr = robust_z_with_scaler(components["return"], scalers["return"])
    zs = robust_z_with_scaler(components["sharpe"], scalers["sharpe"])
    zt = robust_z_with_scaler(components["turnover"], scalers["turnover"])
    zd = robust_z_with_scaler(components["smooth_mdd"], scalers["smooth_mdd"])
    mode = reward_mode(cfg)
    if mode in ("sharpe", "sharpe_only", "single_sharpe"):
        # Single-objective: maximize causal Sharpe (z-scored). Other terms logged only.
        reward = float(zs["z"])
    else:
        reward = (
            float(r["w_return"]) * zr["z"]
            + float(r["w_sharpe"]) * zs["z"]
            - float(r["w_turnover"]) * zt["z"]
            - float(r["w_smooth_mdd"]) * zd["z"]
        )
    return {
        "reward": float(reward),
        "mode": mode,
        "components": {"return": zr, "sharpe": zs, "turnover": zt, "smooth_mdd": zd},
    }
