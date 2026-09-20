"""US regular-session minute template with explicit missing-bar padding."""

from __future__ import annotations

from datetime import datetime, time, timedelta
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd


def _hm(value: str) -> time:
    hh, mm = str(value).split(":")[:2]
    return time(int(hh), int(mm))


def session_times(cfg: dict) -> List[time]:
    cal = cfg["calendar"]
    cur = datetime.combine(datetime.today().date(), _hm(cal["regular_session_start"]))
    end = datetime.combine(datetime.today().date(), _hm(cal["regular_session_end"]))
    out: List[time] = []
    while cur <= end:
        out.append(cur.time())
        cur += timedelta(minutes=1)
    expected = int(cal["bars_per_day"])
    if len(out) != expected:
        raise RuntimeError(f"minute template has {len(out)} slots, expected {expected}")
    return out


def align_day_to_template(local_timestamps: pd.DatetimeIndex, values: np.ndarray, template: Sequence[time]) -> Tuple[np.ndarray, np.ndarray]:
    n = len(template)
    aligned = np.full((n,) + values.shape[1:], np.nan, dtype=np.float32)
    mask = np.zeros((n,), dtype=np.float32)
    clock_to_i = {(t.hour, t.minute): i for i, t in enumerate(template)}
    for row, ts in enumerate(local_timestamps):
        t = pd.Timestamp(ts)
        slot = clock_to_i.get((t.hour, t.minute))
        if slot is None:
            continue
        aligned[slot] = values[row]
        mask[slot] = 1.0
    return aligned, mask


def stamp(day: pd.Timestamp, clock: str, timezone: str = "America/New_York") -> pd.Timestamp:
    return pd.Timestamp(f"{pd.Timestamp(day).date()} {clock}", tz=timezone)
