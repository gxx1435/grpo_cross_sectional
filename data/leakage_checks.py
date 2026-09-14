"""Atomic leakage checks used by evaluation/leakage_audit.py."""

from __future__ import annotations

from typing import Any, Dict, Sequence

import pandas as pd


def _row(
    name: str,
    ok: bool,
    *,
    expected: Any,
    observed: Any,
    evidence: str,
    severity: str,
    extra: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    out = {
        "audit_name": name,
        "status": "PASS" if ok else "FAIL",
        "expected_boundary": expected,
        "observed_boundary": observed,
        "evidence": evidence,
        "failure_reason": "" if ok else evidence,
        "severity": severity,
    }
    if extra:
        out.update(extra)
    return out


def check_split_boundaries(split: Dict[str, Any]) -> Dict[str, Any]:
    tr = [pd.Timestamp(x) for x in split["train_days"]]
    va = [pd.Timestamp(x) for x in split["val_days"]]
    te = [pd.Timestamp(x) for x in split["test_days"]]
    issues = []
    if tr and va and max(tr) >= min(va):
        issues.append("train overlaps validation")
    if va and te and max(va) >= min(te):
        issues.append("validation overlaps test")
    if tr and te and max(tr) >= min(te):
        issues.append("train overlaps test")
    return _row(
        "train_val_test_boundary",
        len(issues) == 0,
        expected=f"train<{split['validation_start_date']}<test {split['test_start_date']}",
        observed=f"n_train={len(tr)} n_val={len(va)} n_test={len(te)}",
        evidence="; ".join(issues) if issues else "disjoint chronological splits",
        severity="high",
    )


def check_monthly_split(split: Dict[str, Any], test_month: str) -> Dict[str, Any]:
    ok = split["test_month"] == test_month
    return _row(
        "monthly_split_correctness",
        ok,
        expected=test_month,
        observed=split["test_month"],
        evidence="test month label matches requested T",
        severity="high",
    )


def check_strict_no_test_update(oos_mode: str, updates: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if oos_mode == "strict_fixed_oos":
        bad = [u for u in updates if u.get("includes_test_realized")]
        return _row(
            "strict_fixed_oos_isolation",
            len(bad) == 0,
            expected="no test-period parameter update",
            observed=f"{len(bad)} test-including updates",
            evidence="strict mode must freeze checkpoint before test",
            severity="high",
        )
    return _row(
        "strict_fixed_oos_isolation",
        True,
        expected="n/a for sequential mode",
        observed=oos_mode,
        evidence="sequential mode is labeled separately",
        severity="medium",
    )


def check_weekly_retrain(updates: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    bad = []
    for u in updates:
        if not u.get("includes_test_realized"):
            continue
        if u.get("label") != "weekly_retrain_after_friday_close":
            bad.append("unlabeled sequential update")
        if str(u.get("cutoff", "")).endswith("09:25") or "open" in str(u.get("cutoff", "")).lower():
            bad.append("update before Friday close")
    return _row(
        "weekly_retrain_boundary",
        len(bad) == 0,
        expected="Friday close cutoff, apply next week, labeled sequential",
        observed=f"{len(updates)} updates",
        evidence="; ".join(bad) if bad else "weekly updates labeled and after close",
        severity="high",
    )


def check_fit_range(name: str, fit_end: str, forbidden_start: str) -> Dict[str, Any]:
    ok = pd.Timestamp(fit_end) < pd.Timestamp(forbidden_start)
    return _row(
        name,
        ok,
        expected=f"fit_end < {forbidden_start}",
        observed=str(fit_end),
        evidence="train-only statistics",
        severity="high",
    )


def check_auction_alignment(auction_date: str, prediction_date: str) -> Dict[str, Any]:
    ok = str(auction_date) == str(prediction_date)
    return _row(
        "auction_date_alignment",
        ok,
        expected=prediction_date,
        observed=auction_date,
        evidence="A_D used only for the same prediction date D",
        severity="high",
    )


def check_timestamps(pred_ts: str, exec_ts: str, real_ts: str) -> Dict[str, Any]:
    ok = pd.Timestamp(pred_ts) <= pd.Timestamp(exec_ts) <= pd.Timestamp(real_ts)
    return _row(
        "prediction_execution_realization_order",
        ok,
        expected="pred <= exec <= realization",
        observed=f"{pred_ts} / {exec_ts} / {real_ts}",
        evidence="causal trading clock",
        severity="high",
    )
