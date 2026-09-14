"""Causal minute features. F is auto-counted. No centered windows, no future backfill."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

OHLCV_BASE: List[str] = ["open", "high", "low", "close", "volume"]

DERIVED: List[str] = [
    "log_return_1",
    "return_5",
    "return_10",
    "return_30",
    "return_60",
    "return_120",
    "high_low_range",
    "open_close_return",
    "body_ratio",
    "upper_shadow_ratio",
    "lower_shadow_ratio",
    "close_position",
    "vol_5",
    "vol_10",
    "vol_30",
    "vol_60",
    "vol_120",
    "downside_vol_5d",
    "downside_vol_10d",
    "downside_vol_30",
    "log_volume",
    "volume_change",
    "relative_volume_5",
    "relative_volume_20",
    "vwap_deviation",
    "amihud",
    "volume_weighted_return",
    "price_ma5_deviation",
    "price_ma10_deviation",
    "price_ma20_deviation",
    "ma5_ma20",
    "cs_rank_return_30",
    "cs_rank_return_60",
    "cs_rank_relative_volume",
    "cs_rank_volatility",
]

CALENDAR: List[str] = [
    "day_of_week_sin",
    "day_of_week_cos",
    "is_monday",
    "is_friday",
    "overnight_return",
    "overnight_volatility",
]

FEATURE_NAMES: List[str] = OHLCV_BASE + DERIVED + CALENDAR
CS_RANK_NAMES = (
    "cs_rank_return_30",
    "cs_rank_return_60",
    "cs_rank_relative_volume",
    "cs_rank_volatility",
)


def feature_dim() -> int:
    return len(FEATURE_NAMES)


def _roll_mean(x: np.ndarray, w: int) -> np.ndarray:
    xf = np.nan_to_num(np.asarray(x, dtype=np.float64), nan=0.0)
    t, n = xf.shape
    c = np.cumsum(xf, axis=0)
    out = np.zeros_like(xf)
    if t >= w:
        prev = np.concatenate([np.zeros((1, n)), c[: t - w]], axis=0)
        out[w - 1 :] = (c[w - 1 :] - prev) / float(w)
    for i in range(min(w - 1, t)):
        out[i] = c[i] / float(i + 1)
    return out


def _roll_std(x: np.ndarray, w: int) -> np.ndarray:
    mu = _roll_mean(x, w)
    mu2 = _roll_mean(np.asarray(x, dtype=np.float64) ** 2, w)
    return np.sqrt(np.maximum(mu2 - mu ** 2, 0.0))


def _cs_rank(x: np.ndarray) -> np.ndarray:
    return pd.DataFrame(x).rank(axis=1, pct=True, na_option="keep").to_numpy(dtype=np.float64)


def compute_minute_features(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    amount: np.ndarray,
    ts: pd.DatetimeIndex,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """
    Causal [T, N, F] features.

    overnight_* uses already-realized previous-day close vs current-bar open
    only on historical bars. D-day Open is never written into a D-day
    prediction window because callers pass M_{D-10:D-1} only.
    """
    o = np.asarray(open_, dtype=np.float64)
    h = np.asarray(high, dtype=np.float64)
    l = np.asarray(low, dtype=np.float64)
    c = np.asarray(close, dtype=np.float64)
    v = np.asarray(volume, dtype=np.float64)
    a = np.asarray(amount, dtype=np.float64)
    t_len, n = c.shape
    names = list(FEATURE_NAMES)
    f = len(names)
    feats = np.zeros((t_len, n, f), dtype=np.float32)

    logc = np.log(np.clip(c, 1e-6, None))
    log_return_1 = np.diff(logc, axis=0, prepend=logc[:1])

    def ret_h(hh: int) -> np.ndarray:
        out = np.zeros((t_len, n), dtype=np.float64)
        out[hh:] = c[hh:] / np.clip(c[:-hh], 1e-6, None) - 1.0
        return out

    return_5, return_10 = ret_h(5), ret_h(10)
    return_30, return_60, return_120 = ret_h(30), ret_h(60), ret_h(120)
    rng = (h - l) / np.clip(c, 1e-6, None)
    open_close_return = c / np.clip(o, 1e-6, None) - 1.0
    body = np.abs(c - o)
    full = np.maximum(h - l, 1e-6)
    body_ratio = body / full
    upper_shadow_ratio = (h - np.maximum(o, c)) / full
    lower_shadow_ratio = (np.minimum(o, c) - l) / full
    close_position = (c - l) / full

    vol_5 = _roll_std(log_return_1, 5)
    vol_10 = _roll_std(log_return_1, 10)
    vol_30 = _roll_std(log_return_1, 30)
    vol_60 = _roll_std(log_return_1, 60)
    vol_120 = _roll_std(log_return_1, 120)
    down = np.where(log_return_1 < 0, log_return_1, 0.0)
    downside_vol_5d = _roll_std(down, 1200)
    downside_vol_10d = _roll_std(down, 2400)
    downside_vol_30 = _roll_std(down, 30)

    log_volume = np.log(np.clip(v, 1.0, None))
    volume_change = np.diff(log_volume, axis=0, prepend=log_volume[:1])
    vol_ma5 = _roll_mean(v, 5)
    vol_ma20 = _roll_mean(v, 20)
    relative_volume_5 = v / np.clip(vol_ma5, 1.0, None)
    relative_volume_20 = v / np.clip(vol_ma20, 1.0, None)
    vwap = np.where(v > 0, a / np.clip(v, 1.0, None), c)
    vwap_deviation = c / np.clip(vwap, 1e-6, None) - 1.0
    amihud = np.abs(log_return_1) / np.clip(a, 1.0, None)
    volume_weighted_return = log_return_1 * relative_volume_5
    ma5 = _roll_mean(c, 5)
    ma10 = _roll_mean(c, 10)
    ma20 = _roll_mean(c, 20)
    price_ma5_deviation = c / np.clip(ma5, 1e-6, None) - 1.0
    price_ma10_deviation = c / np.clip(ma10, 1e-6, None) - 1.0
    price_ma20_deviation = c / np.clip(ma20, 1e-6, None) - 1.0
    ma5_ma20 = ma5 / np.clip(ma20, 1e-6, None) - 1.0

    cs_r30 = _cs_rank(return_30)
    cs_r60 = _cs_rank(return_60)
    cs_rv = _cs_rank(relative_volume_20)
    cs_vol = _cs_rank(vol_30)

    def cs_z(x: np.ndarray) -> np.ndarray:
        mu = np.nanmean(x, axis=1, keepdims=True)
        sd = np.nanstd(x, axis=1, keepdims=True)
        return (x - mu) / np.clip(sd, 1e-6, None)

    ts = pd.DatetimeIndex(ts)
    dow = ts.dayofweek.to_numpy(dtype=np.float64)
    sin = np.sin(2 * np.pi * dow / 5.0)
    cos = np.cos(2 * np.pi * dow / 5.0)
    is_mon = (dow == 0).astype(np.float64)
    is_fri = (dow == 4).astype(np.float64)

    dates = ts.normalize()
    overnight = np.zeros((t_len, n), dtype=np.float64)
    uniq, first_idx = [], {}
    codes = pd.Series(dates).factorize()[0]
    for i, d in enumerate(codes):
        if d not in first_idx:
            first_idx[d] = i
            uniq.append(d)
    last_idx = {}
    for i, d in enumerate(codes):
        last_idx[d] = i
    for d in uniq:
        if d == 0:
            continue
        i0 = first_idx[d]
        ip = last_idx[d - 1]
        prev_c = np.clip(c[ip], 1e-8, None)
        # Historical overnight: day-d open vs day-(d-1) close. For a D-day
        # prediction window this day is at most D-1, so D Open is unused.
        overnight[i0] = np.log(np.clip(o[i0], 1e-8, None) / prev_c)
    overnight_vol = _roll_std(overnight, 20)

    channels = [
        cs_z(np.log(np.clip(o, 1e-6, None))),
        cs_z(np.log(np.clip(h, 1e-6, None))),
        cs_z(np.log(np.clip(l, 1e-6, None))),
        cs_z(np.log(np.clip(c, 1e-6, None))),
        cs_z(log_volume),
        log_return_1,
        return_5,
        return_10,
        return_30,
        return_60,
        return_120,
        rng,
        open_close_return,
        body_ratio,
        upper_shadow_ratio,
        lower_shadow_ratio,
        close_position,
        vol_5,
        vol_10,
        vol_30,
        vol_60,
        vol_120,
        downside_vol_5d,
        downside_vol_10d,
        downside_vol_30,
        log_volume,
        volume_change,
        relative_volume_5,
        relative_volume_20,
        vwap_deviation,
        amihud,
        volume_weighted_return,
        price_ma5_deviation,
        price_ma10_deviation,
        price_ma20_deviation,
        ma5_ma20,
        cs_r30,
        cs_r60,
        cs_rv,
        cs_vol,
        np.broadcast_to(sin[:, None], (t_len, n)),
        np.broadcast_to(cos[:, None], (t_len, n)),
        np.broadcast_to(is_mon[:, None], (t_len, n)),
        np.broadcast_to(is_fri[:, None], (t_len, n)),
        overnight,
        overnight_vol,
    ]
    if len(channels) != f:
        raise RuntimeError(f"F auto-count mismatch: {len(channels)} vs {f}")
    for i, ch in enumerate(channels):
        feats[:, :, i] = np.asarray(ch, dtype=np.float32)
    np.nan_to_num(feats, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    miss = {}
    for i, name in enumerate(names):
        miss[name] = float(np.mean(~np.isfinite(channels[i])))
    meta = {
        "feature_names": names,
        "feat_dim": f,
        "rolling": "trailing_only",
        "centered_window": False,
        "cs_rank_names": list(CS_RANK_NAMES),
        "cs_rank_timing": "contemporaneous_cross_section_at_bar",
        "overnight_rule": "hist_day_open_vs_prev_close_only",
        "missing_rate": miss,
    }
    return feats, meta
