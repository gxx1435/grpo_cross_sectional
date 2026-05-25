"""
Select stocks by historical performance (e.g. top-N YTD return in 2026).

Used by the walk-forward pipeline to pick the 30 best CSI500 names
instead of the alphabetical first-N default in ``hf_data``.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DEFAULT_POOL = HERE / "csi500" / "2026_1min"


def _read_close(path: Path) -> pd.Series:
    df = pd.read_csv(path, parse_dates=["日期"])
    df = df.sort_values("日期").drop_duplicates("日期", keep="first")
    return df.set_index("日期")["收盘"]


def rank_stocks_by_return(
    pool_dir: Path = DEFAULT_POOL,
    start_date: str = "2026-01-05",
    end_date: str = "2026-03-31",
) -> pd.DataFrame:
    """
    Rank every ``*.csv`` in ``pool_dir`` by log-return over
    ``[start_date, end_date]``. Returns a frame sorted descending.
    """
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    rows = []
    for f in sorted(pool_dir.glob("*.csv")):
        try:
            close = _read_close(f)
        except Exception:
            continue
        close = close[(close.index >= start) & (close.index <= end)]
        if len(close) < 2:
            continue
        log_ret = float(np.log(close.iloc[-1] / close.iloc[0]))
        rows.append({"stock_id": f.stem, "filename": f.name, "log_return": log_ret})
    if not rows:
        raise RuntimeError(f"No usable CSV files under {pool_dir}")
    return pd.DataFrame(rows).sort_values("log_return", ascending=False).reset_index(drop=True)


def select_top_n(
    n: int = 30,
    pool_dir: Path = DEFAULT_POOL,
    start_date: str = "2026-01-05",
    end_date: str = "2026-03-31",
    save_path: Optional[Path] = None,
) -> Tuple[List[str], pd.DataFrame]:
    """Return (stock_id list, full ranking table)."""
    rank_df = rank_stocks_by_return(pool_dir, start_date, end_date)
    top = rank_df.head(n).copy()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        top.to_csv(save_path, index=False)
    return top["stock_id"].tolist(), rank_df


def select_random_n(
    n: int = 30,
    pool_dir: Path = DEFAULT_POOL,
    seed: int = 42,
    save_path: Optional[Path] = None,
) -> Tuple[List[str], pd.DataFrame]:
    """Return ``n`` stocks sampled uniformly at random from ``pool_dir``."""
    all_ids = sorted(f.stem for f in pool_dir.glob("*.csv"))
    if len(all_ids) < n:
        raise RuntimeError(f"Need {n} stocks but only {len(all_ids)} CSVs in {pool_dir}")
    rng = np.random.default_rng(seed)
    chosen = rng.choice(all_ids, size=n, replace=False)
    out = pd.DataFrame({"stock_id": chosen, "selection": "random"})
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(save_path, index=False)
    return chosen.tolist(), out


if __name__ == "__main__":
    ids, table = select_top_n(30, save_path=HERE / "walkforward_out" / "top30_ytd2026.csv")
    print(f"top 30 ({len(ids)} stocks):")
    print(table.head(30).to_string(index=False))
