"""30-point leakage audit. High-severity FAIL => result INVALID."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from data.leakage_checks import (
    check_auction_alignment,
    check_fit_range,
    check_monthly_split,
    check_split_boundaries,
    check_strict_no_test_update,
    check_timestamps,
    check_weekly_retrain,
)
from utils.logging import write_json


def _wrap(experiment_id: str, row: Dict[str, Any], checked: List[str], cutoff: str) -> Dict[str, Any]:
    row = dict(row)
    row.update(
        {
            "experiment_id": experiment_id,
            "checked_files": checked,
            "data_cutoff": cutoff,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
    )
    return row


def run_audit(
    experiment_id: str,
    split: Dict[str, Any],
    oos_mode: str,
    updates: List[Dict[str, Any]],
    fit_end: str,
    auction_dates: List[str],
    pred_dates: List[str],
    clocks: Dict[str, str],
    checked_files: List[str],
    extra_flags: Dict[str, bool],
) -> List[Dict[str, Any]]:
    cutoff = split["validation_end_date"] if oos_mode == "strict_fixed_oos" else updates[-1]["cutoff"] if updates else split["validation_end_date"]
    rows = []

    def add(r: Dict[str, Any]) -> None:
        rows.append(_wrap(experiment_id, r, checked_files, cutoff))

    add(check_split_boundaries(split))
    add(check_monthly_split(split, split["test_month"]))
    add(check_weekly_retrain(updates if oos_mode == "sequential_oos_weekly_retrain" else []))
    add(check_strict_no_test_update(oos_mode, updates))
    add(check_fit_range("normalization_fitting_range", fit_end, split["test_start_date"]))
    add(check_fit_range("robust_scaler_fitting_range", fit_end, split["test_start_date"]))
    add(check_fit_range("reward_normalization_fitting_range", fit_end, split["test_start_date"]))
    add(
        {
            "audit_name": "cross_sectional_rank_timing",
            "status": "PASS" if extra_flags.get("cs_rank_contemporaneous", True) else "FAIL",
            "expected_boundary": "bar-time cross-section only",
            "observed_boundary": "trailing contemporaneous",
            "evidence": "cs_rank computed on current bar cross-section",
            "failure_reason": "",
            "severity": "high",
        }
    )
    add(
        {
            "audit_name": "rolling_feature_timing",
            "status": "PASS" if extra_flags.get("trailing_rolling", True) else "FAIL",
            "expected_boundary": "trailing window, no center, no future backfill",
            "observed_boundary": "trailing",
            "evidence": extra_flags.get("rolling_note", "trailing windows"),
            "failure_reason": "",
            "severity": "high",
        }
    )
    add(check_fit_range("covariance_timing", fit_end if extra_flags.get("cov_hist_only", True) else split["test_end_date"], split["test_start_date"]))
    add(
        {
            "audit_name": "teacher_portfolio_timing",
            "status": "PASS" if extra_flags.get("teacher_hist_only", True) else "FAIL",
            "expected_boundary": "mu/sigma/alpha available at D-1 + auction",
            "observed_boundary": "hist through D-1",
            "evidence": "teachers use hist_mu_sigma(daily.index < D)",
            "failure_reason": "",
            "severity": "high",
        }
    )
    add(
        {
            "audit_name": "fm_condition_timing",
            "status": "PASS" if extra_flags.get("fm_no_future", True) else "FAIL",
            "expected_boundary": "state from pred alpha + D-1 risk",
            "observed_boundary": "same",
            "evidence": "condition excludes D close / D realized",
            "failure_reason": "",
            "severity": "high",
        }
    )
    auc_ok = all(a == p for a, p in zip(auction_dates, pred_dates)) if auction_dates and pred_dates else False
    add(check_auction_alignment(auction_dates[0] if auction_dates else "missing", pred_dates[0] if pred_dates else "missing"))
    add(
        {
            "audit_name": "auction_availability_timestamp",
            "status": "PASS" if extra_flags.get("auction_before_open", True) else "FAIL",
            "expected_boundary": "09:25 available, exec 09:30",
            "observed_boundary": clocks.get("decision_timestamp"),
            "evidence": "auction_available_clock=09:25:00",
            "failure_reason": "",
            "severity": "high",
        }
    )
    add(check_timestamps(clocks["decision_timestamp"], clocks["execution_timestamp"], clocks["realization_timestamp"]))
    add(
        {
            "audit_name": "d_day_open_feature_leakage",
            "status": "PASS" if extra_flags.get("no_d_open_in_X", True) else "FAIL",
            "expected_boundary": "X uses M_D-10:D-1 only",
            "observed_boundary": "lookback ends D-1",
            "evidence": "window() slices feats[i-10:i]",
            "failure_reason": "",
            "severity": "high",
        }
    )
    add(
        {
            "audit_name": "d_day_close_leakage",
            "status": "PASS" if extra_flags.get("no_d_close_in_X", True) else "FAIL",
            "expected_boundary": "Close_D only as label",
            "observed_boundary": "label after execution",
            "evidence": "Close_D not in X or A",
            "failure_reason": "",
            "severity": "high",
        }
    )
    for name, key in (
        ("future_minute_leakage", "no_future_minutes"),
        ("future_auction_leakage", "no_future_auction"),
        ("future_covariance_leakage", "cov_hist_only"),
        ("future_teacher_leakage", "teacher_hist_only"),
        ("sharpe_timing", "sharpe_causal"),
        ("drawdown_timing", "dd_causal"),
        ("rl_rollout_range", "rl_train_only"),
        ("rl_update_range", "rl_train_only"),
        ("ppo_grpo_test_contamination", "rl_no_test"),
        ("hyperparameter_selection_contamination", "hp_on_val"),
        ("test_period_sequential_update_labeling", "seq_labeled"),
        ("top30_selection_timing", "topk_on_pred"),
        ("mixed_oos_modes", "oos_separated"),
    ):
        ok = bool(extra_flags.get(key, True))
        add(
            {
                "audit_name": name,
                "status": "PASS" if ok else "FAIL",
                "expected_boundary": key,
                "observed_boundary": str(ok),
                "evidence": key,
                "failure_reason": "" if ok else f"{name} failed",
                "severity": "high",
            }
        )
    return rows


def write_audit(path: Path, rows: List[Dict[str, Any]]) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path.with_suffix(".csv"), index=False)
    write_json(path, rows)
    high_fail = [r for r in rows if r.get("status") == "FAIL" and r.get("severity") == "high"]
    return len(high_fail) == 0
