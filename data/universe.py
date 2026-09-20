"""Auditable dynamic data-availability proxy universe."""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

UNIVERSE_TYPE = "data_available_equity_universe"


def decision_exclusion_reason(previous_close: float, previous_minutes: int, required_previous: int, history_minutes: int, required_history: int) -> str:
    """Reasons knowable at the D-day pre-open decision timestamp."""
    if not np.isfinite(previous_close) or previous_close <= 0:
        return "previous_close_unavailable"
    if int(previous_minutes) < int(required_previous):
        return "previous_session_minutes_insufficient"
    if int(history_minutes) < int(required_history):
        return "history_minutes_insufficient"
    return ""


def evaluation_exclusion_reason(target_open: float, target_close: float, target_minutes: int, target_required: int) -> str:
    """Post-close label/execution audit; never used to form the decision set."""
    if not np.isfinite(target_open) or target_open <= 0:
        return "target_open_unavailable"
    if not np.isfinite(target_close) or target_close <= 0:
        return "target_close_unavailable"
    if int(target_minutes) < target_required:
        return "target_day_minutes_insufficient"
    return ""


def universe_manifest_rows(
    date: pd.Timestamp,
    codes: List[str],
    included: np.ndarray,
    reasons: List[str],
    history_count: np.ndarray,
    previous_count: np.ndarray,
    target_count: np.ndarray,
    evaluation_eligible: np.ndarray,
    evaluation_reasons: List[str],
    cutoff: str,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": str(pd.Timestamp(date).date()),
            "stock_code": codes,
            "included_flag": np.asarray(included, dtype=bool),
            "exclusion_reason": reasons,
            "minute_count_history": np.asarray(history_count, dtype=int),
            "minute_count_previous_session": np.asarray(previous_count, dtype=int),
            "minute_count_target_day": np.asarray(target_count, dtype=int),
            "evaluation_eligible_flag": np.asarray(evaluation_eligible, dtype=bool),
            "evaluation_exclusion_reason": evaluation_reasons,
            "data_cutoff_timestamp": cutoff,
            "universe_type": UNIVERSE_TYPE,
        }
    )
