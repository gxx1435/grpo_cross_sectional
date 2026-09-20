"""Monthly 12-train + 1-validation + fixed-test walk-forward splits."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import pandas as pd


def month_starts(lo: str, hi: str) -> List[pd.Timestamp]:
    return [p.to_timestamp() for p in pd.period_range(lo, hi, freq="M")]


def infer_test_months(days: Sequence[pd.Timestamp], cfg: dict) -> List[pd.Timestamp]:
    if len(days) == 0:
        return []
    configured_start = pd.Period(str(cfg["walkforward"]["test_start"]), freq="M")
    configured_end = cfg["walkforward"].get("test_end", "auto")
    last_observed_day = max(pd.Timestamp(d).normalize() for d in days)
    observed_end = pd.Period(last_observed_day, freq="M")
    # A terminal source partition is eligible as a Test month only when it
    # reaches the final three business days of that month.  Earlier months are
    # known complete because a later partition exists.  This excludes the
    # vendor delivery's truncated 2026-05 partition (ending 2026-05-08).
    near_month_end = last_observed_day >= (observed_end.end_time.normalize() - pd.offsets.BDay(3))
    if not near_month_end:
        observed_end -= 1
    end = observed_end if str(configured_end).lower() == "auto" else min(observed_end, pd.Period(str(configured_end), freq="M"))
    return [p.to_timestamp() for p in pd.period_range(configured_start, end, freq="M")]


def month_bounds(month_start: pd.Timestamp) -> Tuple[pd.Timestamp, pd.Timestamp]:
    p = pd.Period(month_start, freq="M")
    return pd.Timestamp(p.start_time.normalize()), pd.Timestamp(p.end_time.normalize())


def split_for_test_month(days: Sequence[pd.Timestamp], test_month: pd.Timestamp, train_offset_months: int = 13) -> Dict[str, object]:
    test_lo, test_hi = month_bounds(test_month)
    val_month = pd.Period(test_month, freq="M") - 1
    train_first = pd.Period(test_month, freq="M") - int(train_offset_months)
    train_last = val_month - 1
    train_lo, _ = month_bounds(train_first.to_timestamp())
    _, train_hi = month_bounds(train_last.to_timestamp())
    val_lo, val_hi = month_bounds(val_month.to_timestamp())
    normalized = [pd.Timestamp(d).normalize() for d in days]

    def select(lo: pd.Timestamp, hi: pd.Timestamp) -> List[pd.Timestamp]:
        return [d for d in normalized if lo <= d <= hi]

    train_days = select(train_lo, train_hi)
    val_days = select(val_lo, val_hi)
    test_days = select(test_lo, test_hi)
    train_periods = sorted({str(pd.Period(d, freq="M")) for d in train_days})
    return {
        "test_month": str(pd.Period(test_month, freq="M")),
        "train_start_date": str(train_lo.date()),
        "train_end_date": str(train_hi.date()),
        "validation_start_date": str(val_lo.date()),
        "validation_end_date": str(val_hi.date()),
        "test_start_date": str(test_lo.date()),
        "test_end_date": str(test_hi.date()),
        "train_days": train_days,
        "val_days": val_days,
        "test_days": test_days,
        "train_months_observed": train_periods,
        "train_month_count": len(train_periods),
        "strict_fixed_oos": True,
    }
