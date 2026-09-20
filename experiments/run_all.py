"""Orchestrate data preparation and all strict fixed-OOS Test months."""

from __future__ import annotations

import time
import traceback
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import pandas as pd

from data.loaders import data_contract
from data.sp500_pipeline import prepare_sp500_data
from data.splits import infer_test_months
from data.store import ResearchStore
from evaluation.sp500_reports import write_annual_report
from experiments.engine import run_month
from utils.git_info import environment_info
from utils.logging import log, write_json
from utils.runtime import estimate_asdict, estimate_resources


def _record_version_history(cfg: dict, manifest: Dict[str, Any]) -> None:
    """Append an immutable-version catalog; partition files are never overwritten."""
    processed = Path(cfg["_root"]) / cfg["paths"]["processed_dir"]
    path = processed / "feature_metadata" / "version_history.json"
    history = []
    if path.is_file():
        import json

        history = json.loads(path.read_text(encoding="utf-8"))
    version = str(manifest["data_version"])
    if not any(str(row.get("data_version")) == version for row in history):
        history.append(
            {
                "data_version": version,
                "feature_schema_hash": manifest["feature_schema_hash"],
                "feature_config_hash": manifest["feature_config_hash"],
                "code_version_hash": manifest["code_version_hash"],
                "recorded_at": pd.Timestamp.now(tz="UTC").isoformat(),
                "feature_paths": [row["feature_path"] for row in manifest.get("months", [])],
                "raw_paths": [row["raw_path"] for row in manifest.get("months", [])],
            }
        )
    write_json(path, history)
    discovered = sorted(
        {
            file.stem.removeprefix("part-")
            for root_name in ("minute_features", "raw_ohlcv")
            for file in (processed / root_name).glob("year=*/month=*/part-*.parquet")
        }
    )
    write_json(processed / "feature_metadata" / "discovered_version_tags.json", discovered)


def run_all(cfg: dict, out_root: Path, months_filter: Optional[Sequence[str]] = None, prepare_only: bool = False, force_features: bool = False) -> Dict[str, Any]:
    started = time.time()
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "data_contract").mkdir(exist_ok=True)
    (out_root / "feature_generation").mkdir(exist_ok=True)
    (out_root / "failed_experiments").mkdir(exist_ok=True)
    contract = data_contract(cfg)
    write_json(out_root / "data_contract" / "data_contract.json", contract)
    if not contract["ok"]:
        failure = {"status": "failed", "phase": "data_contract", "missing": contract["missing"]}
        write_json(out_root / "failed_experiments" / f"data_contract_{int(time.time())}.json", failure)
        raise RuntimeError(str(failure))
    log(f"data contract PASS: {len(contract['members'])} monthly files, ZIP={contract['zip_sha256']}")
    feature_started = time.time()
    manifest = prepare_sp500_data(cfg, force=force_features)
    _record_version_history(cfg, manifest)
    records = list(manifest.get("months", []))
    raw_columns = sorted(
        {
            str(column)
            for record in records
            for column in (record.get("quality", {}).get("raw_columns", []) or [])
        }
    )
    write_json(
        out_root / "data_contract" / "observed_data_contract.json",
        {
            "status": "PASS",
            "pandas_read_verified": len(records) == len(contract["members"]),
            "detected_formats": sorted({str(record.get("quality", {}).get("detected_format")) for record in records}),
            "raw_columns": raw_columns,
            "column_mappings": [record.get("quality", {}).get("column_mapping", {}) for record in records],
            "source_months": [record.get("source_month") for record in records],
            "date_start": min(str(record.get("date_start")) for record in records),
            "date_end": max(str(record.get("date_end")) for record in records),
            "row_count_raw": sum(int(record.get("row_count_raw", 0)) for record in records),
            "row_count_regular_session": sum(int(record.get("row_count_regular_session", 0)) for record in records),
            "duplicate_count": sum(int(record.get("duplicate_count", 0)) for record in records),
            "invalid_count": sum(int(record.get("invalid_count", 0)) for record in records),
            "stock_count_min": min(int(record.get("stock_count", 0)) for record in records),
            "stock_count_max": max(int(record.get("stock_count", 0)) for record in records),
            "timezone": cfg["calendar"]["timezone"],
            "data_version": manifest["data_version"],
        },
    )
    write_json(
        out_root / "feature_generation" / "status.json",
        {
            "status": "completed",
            "data_version": manifest["data_version"],
            "months": [row["source_month"] for row in manifest["months"]],
            "seconds": time.time() - feature_started,
        },
    )
    if prepare_only:
        return {"status": "prepared", "data_version": manifest["data_version"]}

    store = ResearchStore(cfg, prepare=False)
    months = infer_test_months(store.days, cfg)
    if months_filter:
        wanted = {str(value)[:7] for value in months_filter}
        months = [month for month in months if str(pd.Period(month, freq="M")) in wanted]
        if not months:
            raise RuntimeError(f"no Test month matches {sorted(wanted)}")
    write_json(out_root / "environment.json", environment_info(cfg["_root"]))
    write_json(out_root / "resource_estimate.json", estimate_asdict(estimate_resources(cfg, len(months))))
    results = []
    for position, month in enumerate(months, start=1):
        label = str(pd.Period(month, freq="M"))
        log(f"======== strict fixed OOS month {position}/{len(months)}: {label} ========")
        try:
            results.append(run_month(cfg, store, month, "strict_fixed_oos", out_root))
        except Exception as exc:
            failure = {"status": "failed", "month": label, "error": str(exc), "traceback": traceback.format_exc()}
            results.append(failure)
            log(f"FAILED {label}: {exc}")
        write_json(out_root / "run_index.json", results)
    annual = write_annual_report(out_root, cfg)
    summary = {
        "status": "completed_with_failures" if any(row.get("status") == "failed" for row in results) else "completed",
        "n_requested": len(months),
        "n_valid": sum(row.get("valid") is True for row in results),
        "n_invalid": sum(row.get("valid") is False for row in results),
        "n_failed": sum(row.get("status") == "failed" for row in results),
        "hours": (time.time() - started) / 3600.0,
        "annual": annual,
    }
    write_json(out_root / "final_summary" / "summary.json", summary)
    return summary
