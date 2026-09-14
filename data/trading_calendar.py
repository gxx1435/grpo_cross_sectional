"""SSE/SZSE continuous-session 240-minute template. No future-bar fill."""

from __future__ import annotations

from datetime import datetime, time, timedelta
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd


def _hm(s: str) -> time:
    hh, mm = s.split(":")
    return time(int(hh), int(mm))


def session_times(cfg: dict) -> List[time]:
    cal = cfg["calendar"]
    lo_m, hi_m = cal["morning"]
    lo_a, hi_a = cal["afternoon"]
    out: List[time] = []
    t = datetime.combine(datetime.today().date(), _hm(lo_m))
    end = datetime.combine(datetime.today().date(), _hm(hi_m))
    while t <= end:
        out.append(t.time())
        t += timedelta(minutes=1)
    t = datetime.combine(datetime.today().date(), _hm(lo_a))
    end = datetime.combine(datetime.today().date(), _hm(hi_a))
    while t <= end:
        out.append(t.time())
        t += timedelta(minutes=1)
    n = int(cal["bars_per_day"])
    if len(out) != n:
        raise RuntimeError(f"minute template has {len(out)} slots, expected {n}")
    return out


def is_dropped_clock(ts: pd.Timestamp, drop: Sequence[str]) -> bool:
    clock = f"{ts.hour:02d}:{ts.minute:02d}"
    return clock in set(drop)


def align_day_to_template(
    day_index: pd.DatetimeIndex,
    values: np.ndarray,
    template: Sequence[time],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Map a day's irregular minutes onto the 240-slot template.
    Missing slots are zero + mask=0. Never copies a later minute backward.
    """
    n = len(template)
    if values.ndim == 1:
        aligned = np.zeros((n,), dtype=np.float32)
    else:
        aligned = np.zeros((n,) + values.shape[1:], dtype=np.float32)
    mask = np.zeros((n,), dtype=np.float32)
    if len(day_index) == 0:
        return aligned, mask
    clock_to_i = {(t.hour, t.minute): i for i, t in enumerate(template)}
    for j, ts in enumerate(day_index):
        ts = pd.Timestamp(ts)
        key = (int(ts.hour), int(ts.minute))
        i = clock_to_i.get(key)
        if i is None:
            continue
        aligned[i] = values[j]
        mask[i] = 1.0
    return aligned, mask


def trading_days_from_index(idx: pd.DatetimeIndex) -> List[pd.Timestamp]:
    return list(sorted({pd.Timestamp(t).normalize() for t in idx}))


def stamp(day: pd.Timestamp, clock: str) -> pd.Timestamp:
    return pd.Timestamp(f"{pd.Timestamp(day).date()} {clock}")
