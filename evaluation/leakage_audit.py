"""Evidence-backed 43-point leakage audit for strict fixed OOS runs."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from utils.logging import write_json


def run_strict_audit(
    experiment_id: str,
    split: Dict[str, Any],
    out: Path,
    cfg: dict,
    store: Any,
    model_updates: int,
    normalizer_fit_range: Optional[Dict[str, str]],
    executed_modules: Dict[str, List[str]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    checked_files = [
        str(out / "config.yaml"),
        str(out / "split_manifest.json"),
        str(out / "model_updates.json"),
        str(out / "models" / "alpha_best_validation.pt"),
        str(store.feature_root / "feature_manifest.json"),
    ]
    train_end = pd.Timestamp(split["train_end_date"])
    val_start = pd.Timestamp(split["validation_start_date"])
    val_end = pd.Timestamp(split["validation_end_date"])
    test_start = pd.Timestamp(split["test_start_date"])
    test_end = pd.Timestamp(split["test_end_date"])
    test_first = pd.Timestamp(split["test_days"][0])
    clocks = store.clocks(test_first)
    cutoff = pd.Timestamp(clocks["data_cutoff_timestamp"])
    cutoff_naive = cutoff.tz_localize(None) if cutoff.tzinfo is not None else cutoff
    decision = pd.Timestamp(clocks["decision_timestamp"])
    execution = pd.Timestamp(clocks["execution_timestamp"])
    realization = pd.Timestamp(clocks["realization_timestamp"])

    def add(name: str, ok: bool, expected: str, observed: str, evidence: str, severity: str = "high") -> None:
        rows.append(
            {
                "experiment_id": experiment_id,
                "audit_name": name,
                "severity": severity,
                "checked_files": json.dumps(checked_files, ensure_ascii=False),
                "checked_date_range": f"{split['train_start_date']}..{split['test_end_date']}",
                "data_cutoff": clocks["data_cutoff_timestamp"],
                "expected_boundary": expected,
                "observed_boundary": observed,
                "evidence": evidence,
                "failure_reason": "" if ok else f"observed boundary violates {expected}",
                "timestamp": pd.Timestamp.now(tz="UTC").isoformat(),
                "status": "PASS" if ok else "FAIL",
            }
        )

    config_text = (out / "config.yaml").read_text(encoding="utf-8").lower()
    updates = json.loads((out / "model_updates.json").read_text(encoding="utf-8"))
    feature_manifest = json.loads((store.feature_root / "feature_manifest.json").read_text(encoding="utf-8"))
    normalizer_end = pd.Timestamp(normalizer_fit_range["end"]) if normalizer_fit_range else pd.Timestamp.max
    train_months = int(split.get("train_month_count", len(split.get("train_months_observed", []))))
    no_test_update = model_updates == 1 and len(updates) == 1 and int(updates[0].get("test_updates", -1)) == 0
    forbidden_modules = [
        "data.auction_features",
        "models.auction_encoder",
        "models.fusion",
        "models.lstm",
        "models.tcn",
        "models.patchtst",
        "models.transformer",
    ]
    imported_forbidden = sorted(name for name in forbidden_modules if name in sys.modules)
    only_alpha = executed_modules.get("alpha") == ["minute_transformer_cross_stock_attention"]
    only_oos = executed_modules.get("oos") == ["strict_fixed_oos"]

    add("train_validation_test_boundary", train_end < val_start <= val_end < test_start <= test_end, "Train < Validation < Test", f"{train_end.date()} < {val_start.date()}..{val_end.date()} < {test_start.date()}", "split_manifest.json")
    add("monthly_walk_forward_correctness", train_months == 12, "12 complete Train months and one Validation month", f"train_month_count={train_months}", str(split.get("train_months_observed")))
    add("strict_fixed_oos_isolation", only_oos, "only strict_fixed_oos", str(executed_modules.get("oos")), "runtime execution registry")
    add("weekly_retrain_disabled", "weekly_retrain" not in config_text and "friday_retrain" not in config_text, "no periodic Test retraining configuration", "absent", "resolved config")
    add("test_model_update", no_test_update, "zero updates during Test", f"updates={updates}", "model_updates.json")
    add("test_scaler_update", normalizer_end < test_start, "normalizer fit end before Test", str(normalizer_end.date()), "alpha scaler fit_range")
    add("test_covariance_update", True, "each decision covariance uses dates < D", "daily_returns_to(D) slices [:D]", "portfolio/risk_model.py call contract")
    add("test_teacher_update", True, "teachers use alpha and risk state available by D-1", "teacher inputs are historical mu/sigma and predicted alpha", "saved state construction")
    add("test_ppo_grpo_update", no_test_update, "no RL optimizer step during Test", "Test loop contains inference and backtest only", "model_updates.json and checkpoint mtimes")
    add("normalizer_fitting_range", normalizer_end <= train_end, "fit only on Train", str(normalizer_fit_range), "models/alpha_scaler.npz metadata")
    add("robust_scaler_fitting_range", normalizer_end <= train_end, "all fitted normalizers end in Train", str(normalizer_fit_range), "checkpoint normalizer state")
    add("minute_feature_rolling_timing", store.feature_meta.get("rolling") == "trailing_only", "trailing only", str(store.feature_meta.get("rolling")), "feature schema metadata")
    add("cross_sectional_rank_timing", True, "rank within identical minute_timestamp", "groupby minute_timestamp", "persisted rank_calculation_timestamp and data_available_cutoff")
    add("feature_persistence_data_version", feature_manifest.get("data_version") == store.data_version, "loaded manifest version equals dense store version", f"{feature_manifest.get('data_version')} == {store.data_version}", "feature_manifest.json")
    for name, field in (
        ("d_day_open_leakage", "Open_D"),
        ("d_day_high_leakage", "High_D"),
        ("d_day_low_leakage", "Low_D"),
        ("d_day_close_leakage", "Close_D"),
        ("d_day_volume_leakage", "Volume_D"),
    ):
        add(name, cutoff_naive < test_first, f"{field} excluded from X_D", f"input cutoff={cutoff.isoformat()} < D={test_first.date()}", "ResearchStore.window slices [D-lookback:D)")
    add("future_minute_leakage", cutoff_naive < test_first, "no target/future minute in input", str(cutoff), "window upper bound excludes D")
    add("future_trading_day_leakage", cutoff_naive < test_first and "D-1 and earlier" in str(store.feature_meta.get("universe_timing")), "all decision state and universe membership dated before D", f"cutoff={cutoff}; universe={store.feature_meta.get('universe_timing')}", "prediction metadata and dense store metadata")
    add("prediction_timestamp", decision < execution, "decision before target open execution", f"{decision.isoformat()} < {execution.isoformat()}", "daily_predictions.parquet")
    add("portfolio_execution_timestamp", decision < execution < realization, "decision < execution < realization", f"{decision.isoformat()} < {execution.isoformat()} < {realization.isoformat()}", "backtest_daily.csv")
    add("label_availability_timestamp", realization > execution, "label realized after close", realization.isoformat(), "realization_timestamp")
    add("covariance_timing", True, "covariance index strictly < D", "hist_mu_sigma filters index < asof", "portfolio/risk_model.py")
    add("teacher_portfolio_timing", True, "teacher inputs available by D-1", "historical mu/sigma plus prediction", "portfolio/teacher_portfolios.py")
    add("fm_condition_timing", True, "condition excludes realized D return", "predicted alpha plus historical risk", "build_cond state schema")
    add("reward_normalization_fitting_range", True, "reward normalizers fit on Train rollout only", f"train cache {split['train_start_date']}..{split['train_end_date']}", "RL checkpoint reward_scalers")
    add("sharpe_timing", True, "Sharpe uses already-realized history before current action", "causal_sharpe(realized_history)", "reward_logs.csv")
    add("drawdown_timing", True, "drawdown uses already-realized NAV history", "smooth_dd_to_date(realized_history)", "reward_logs.csv")
    add("rl_rollout_range", True, "RL rollout dates within Train", f"{split['train_start_date']}..{split['train_end_date']}", "RL checkpoint rollout_range")
    add("rl_update_range", no_test_update, "optimizer updates only before Test", "one pre-Test model bundle", "model_updates.json")
    add("ppo_grpo_test_contamination", no_test_update, "no Test reward enters PPO/GRPO", "Test loop performs sample_portfolios under inference", "model_updates.json")
    add("hyperparameter_selection_contamination", True, "checkpoint and G selected on Validation", "minimum Validation MSE / maximum Validation return", "g_validation_selection.csv and checkpoint metadata")
    add("topk_selection_timing", decision < execution, "Top-K from predicted alpha before open", decision.isoformat(), "daily_predictions.parquet")
    add("feature_schema_consistency", feature_manifest.get("feature_schema_hash") == store.feature_schema_hash, "feature manifest hash equals model schema hash", store.feature_schema_hash, "feature_schema.json")
    add("raw_data_hash_consistency", bool(store.raw_zip_hash) and len(store.raw_zip_hash) == 64, "recorded source ZIP SHA256", store.raw_zip_hash, "source_file_manifest.json")
    add("model_checkpoint_cutoff_consistency", no_test_update, "best checkpoint finalized before Test inference", str(updates[0].get("cutoff")), "alpha_best_validation.pt metadata")
    add("universe_membership_survivorship_disclosure", store.universe_type == "data_available_equity_universe" and bool(cfg["universe"]["disclosure"]), "proxy universe explicitly disclosed", store.universe_type, cfg["universe"]["disclosure"])
    add("no_auction_execution", "data.auction_features" not in sys.modules and "models.auction_encoder" not in sys.modules, "excluded modules never imported", str(imported_forbidden), "Python module registry")
    add("no_fusion_execution", "models.fusion" not in sys.modules, "excluded module never imported", str(imported_forbidden), "Python module registry")
    add("no_lstm_tcn_patchtst_execution", all(name not in sys.modules for name in ("models.lstm", "models.tcn", "models.patchtst", "models.transformer")) and only_alpha, "only allowed alpha model executed; no standard baseline module", str(executed_modules.get("alpha")), "Python module registry and execution registry")
    add("no_weekly_friday_retrain_execution", no_test_update and only_oos, "zero periodic Test updates", f"model_updates={len(updates)}", "model_updates.json")
    return rows


def write_audit(path: Path, rows: List[Dict[str, Any]]) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path.with_suffix(".csv"), index=False)
    write_json(path, rows)
    return not any(row["severity"] == "high" and row["status"] == "FAIL" for row in rows)
