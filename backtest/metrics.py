from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import pandas as pd

from backtest.drawdown import hard_mdd, smooth_dd_to_date


def causal_sharpe(hist_net: Sequence[float], min_obs: int, fallback: float) -> float:
    x = np.asarray(list(hist_net), dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < int(min_obs):
        return float(fallback)
    sd = float(x.std(ddof=0)) + 1e-12
    return float(np.sqrt(252.0) * x.mean() / sd)


def summarize_nav(daily: pd.DataFrame, model_col: str = "model") -> pd.DataFrame:
    rows = []
    for name, g in daily.groupby(model_col):
        g = g.sort_values("date")
        r = g["net_return"].to_numpy(dtype=np.float64)
        n = int(len(r))
        wealth = np.exp(np.cumsum(r)) if n else np.asarray([], dtype=np.float64)
        last = float(wealth[-1]) if n and np.isfinite(wealth[-1]) else float("nan")
        total = float(last - 1.0) if np.isfinite(last) else float("nan")
        # 不足一个交易月时复利年化会被 252/n 放大成假异常值，先不报。
        if n >= 15 and np.isfinite(last) and last > 0:
            ann = float(last ** (252.0 / n) - 1.0)
            if not np.isfinite(ann) or abs(ann) > 5.0:
                ann = float("nan")
        else:
            ann = float("nan")
        mu, sd = float(r.mean()) if n else float("nan"), float(r.std(ddof=0)) + 1e-12 if n else 1e-12
        rows.append(
            {
                "model": name,
                "total_net_return": total,
                "ann_return": ann,
                "sharpe": float(np.sqrt(252) * mu / sd) if n else float("nan"),
                "max_drawdown": hard_mdd(r) if n else float("nan"),
                "mean_turnover": float(g["turnover"].mean()) if n else float("nan"),
                "mean_cost": float(g["transaction_cost"].mean()) if n else float("nan"),
                "n_days": n,
            }
        )
    return pd.DataFrame(rows).sort_values("total_net_return", ascending=False)
