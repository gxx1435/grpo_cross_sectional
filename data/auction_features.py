"""D-day auction tensor A_D in R^{N x 9}. Never broadcast onto minute tokens."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from data.loaders import AUCTION_FEATURE_NAMES


def auction_matrix(
    auction_df: pd.DataFrame,
    codes: List[str],
    day: pd.Timestamp,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    day = pd.Timestamp(day).normalize()
    sub = auction_df[auction_df["trading_date"] == day]
    by = {r.stock_code: r for r in sub.itertuples(index=False)}
    n = len(codes)
    a = np.zeros((n, len(AUCTION_FEATURE_NAMES)), dtype=np.float32)
    valid = np.zeros((n,), dtype=bool)
    for i, code in enumerate(codes):
        r = by.get(code)
        if r is None:
            continue
        vals = [getattr(r, k) for k in AUCTION_FEATURE_NAMES]
        if not np.all(np.isfinite(vals)):
            continue
        if float(r.auction_price) <= 0 or float(r.auction_volume) <= 0:
            continue
        a[i] = np.asarray(vals, dtype=np.float32)
        valid[i] = True
    meta = {
        "auction_date": str(day.date()),
        "prediction_date": str(day.date()),
        "join_key": "(stock_code, trading_date)",
        "n_valid": int(valid.sum()),
        "n_universe": n,
        "available_clock": "09:25:00",
        "feature_names": list(AUCTION_FEATURE_NAMES),
        "broadcast_to_minutes": False,
    }
    return a, valid, meta


def assert_auction_not_in_minutes(auction_dim: int, minute_feat_dim: int) -> None:
    if auction_dim != 9:
        raise RuntimeError(f"auction dim must be 9, got {auction_dim}")
    if minute_feat_dim == 9:
        raise RuntimeError("minute F==9 looks like auction was merged into minutes")
