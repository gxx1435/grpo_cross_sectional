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
        ann = float(last ** (252.0 / n) - 1.0) if n and np.isfinite(last) and last > 0 else float("nan")
        mu, sd = float(r.mean()) if n else float("nan"), float(r.std(ddof=0)) + 1e-12 if n else 1e-12
        downside = r[r < 0]
        downside_sd = float(downside.std(ddof=0)) + 1e-12 if len(downside) else float("nan")
        mdd = hard_mdd(r) if n else float("nan")
        simplex_violation = np.maximum(np.abs(g["simplex_sum"].to_numpy(float) - 1.0), np.maximum(0.0, -g["min_weight"].to_numpy(float)))
        rows.append(
            {
                "model": name,
                "cumulative_return": total,
                "total_net_return": total,
                "ann_return": ann,
                "ann_volatility": float(sd * np.sqrt(252.0)) if n else float("nan"),
                "sharpe": float(np.sqrt(252) * mu / sd) if n else float("nan"),
                "sortino": float(np.sqrt(252) * mu / downside_sd) if n and np.isfinite(downside_sd) else float("nan"),
                "max_drawdown": mdd,
                "calmar": float(ann / abs(mdd)) if np.isfinite(ann) and np.isfinite(mdd) and abs(mdd) > 1e-12 else float("nan"),
                "win_rate": float((r > 0).mean()) if n else float("nan"),
                "average_daily_return": mu,
                "mean_turnover": float(g["turnover"].mean()) if n else float("nan"),
                "mean_cost": float(g["transaction_cost"].mean()) if n else float("nan"),
                "total_transaction_cost": float(g["transaction_cost"].sum()) if n else float("nan"),
                "candidate_feasibility": float((simplex_violation <= 1e-5).mean()) if n else float("nan"),
                "n_days": n,
            }
        )
    return pd.DataFrame(rows).sort_values("total_net_return", ascending=False)
