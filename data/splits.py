"""Monthly walk-forward and Friday weekly-retrain cutoffs."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import pandas as pd


def month_starts(lo: str, hi: str) -> List[pd.Timestamp]:
    return [p.to_timestamp() for p in pd.period_range(lo, hi, freq="M")]


def month_bounds(month_start: pd.Timestamp) -> Tuple[pd.Timestamp, pd.Timestamp]:
    p = pd.Period(month_start, freq="M")
    return pd.Timestamp(p.start_time.normalize()), pd.Timestamp(p.end_time.normalize())


def split_for_test_month(
    days: Sequence[pd.Timestamp],
    test_month: pd.Timestamp,
    train_offset_months: int = 13,
) -> Dict[str, object]:
    """
    TrainMonths(T) = [T-13m, T-2m]  (12 full months)
    ValidationMonth(T) = T-1m
    TestMonth(T) = T
    """
    test_lo, test_hi = month_bounds(test_month)
    val_month = test_month - pd.offsets.MonthBegin(1)
    val_lo, val_hi = month_bounds(val_month)
    train_hi_m = val_month - pd.offsets.MonthBegin(1)
    train_lo_m = test_month - pd.offsets.MonthBegin(int(train_offset_months))
    train_lo, train_hi = pd.Timestamp(train_lo_m), pd.Timestamp(pd.Period(train_hi_m, "M").end_time.normalize())

    days = [pd.Timestamp(d).normalize() for d in days]

    def in_range(lo: pd.Timestamp, hi: pd.Timestamp) -> List[pd.Timestamp]:
        return [d for d in days if lo <= d <= hi]

    train_days = in_range(train_lo, train_hi)
    val_days = in_range(val_lo, val_hi)
    test_days = in_range(test_lo, test_hi)
    return {
        "test_month": str(pd.Timestamp(test_month).strftime("%Y-%m")),
        "train_start_date": str(train_lo.date()),
        "train_end_date": str(train_hi.date()),
        "validation_start_date": str(val_lo.date()),
        "validation_end_date": str(val_hi.date()),
        "test_start_date": str(test_lo.date()),
        "test_end_date": str(test_hi.date()),
        "train_days": train_days,
        "val_days": val_days,
        "test_days": test_days,
    }


def fridays(days: Sequence[pd.Timestamp]) -> List[pd.Timestamp]:
    return [pd.Timestamp(d).normalize() for d in days if pd.Timestamp(d).dayofweek == 4]


def next_week_start(friday: pd.Timestamp, days: Sequence[pd.Timestamp]) -> pd.Timestamp:
    friday = pd.Timestamp(friday).normalize()
    after = [pd.Timestamp(d).normalize() for d in days if pd.Timestamp(d) > friday]
    if not after:
        raise KeyError(f"no trading day after Friday {friday.date()}")
    return after[0]


def sequential_retrain_windows(
    split: Dict[str, object],
) -> List[Dict[str, object]]:
    """
    Each Friday close may update the model using data realized through that
    Friday close. The new checkpoint is valid only from the next trading week.
    """
    train = list(split["train_days"])
    val = list(split["val_days"])
    test = list(split["test_days"])
    windows = [
        {
            "cutoff": split["validation_end_date"],
            "effective_from": split["test_start_date"],
            "includes_test_realized": False,
            "train_days": train,
            "val_days": val,
            "label": "pre_test_frozen",
        }
    ]
    for fri in fridays(test):
        realized = [d for d in test if d <= fri]
        windows.append(
            {
                "cutoff": str(fri.date()),
                "effective_from": str(next_week_start(fri, test).date()) if any(d > fri for d in test) else None,
                "includes_test_realized": True,
                "train_days": train + val + realized,
                "val_days": [fri],
                "label": "weekly_retrain_after_friday_close",
            }
        )
    return windows
