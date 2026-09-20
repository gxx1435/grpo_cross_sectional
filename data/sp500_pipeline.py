"""Versioned raw/feature persistence for the S&P 500 ZIP source."""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml

from data.loaders import (
    data_contract,
    enumerate_month_members,
    hash_zip_member,
    read_month_member,
    regular_session,
)
from data.minute_features import (
    add_cross_sectional_features,
    compute_symbol_features,
    feature_names,
    finalize_feature_validity,
)
from utils.config import resolve_path
from utils.logging import log, write_json


def _sha_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)


def _code_hash(root: Path) -> str:
    rels = [
        "data/loaders.py",
        "data/minute_features.py",
        "data/sp500_pipeline.py",
        "data/store.py",
    ]
    h = hashlib.sha256()
    for rel in rels:
        p = root / rel
        h.update(rel.encode())
        if p.is_file():
            h.update(p.read_bytes())
    return h.hexdigest()


def _git_commit(root: Path) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    except Exception:
        return "unknown"


def _read_json(path: Path, default: Any) -> Any:
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _partition(root: Path, month: str) -> Path:
    year, mon = month.split("-")
    return root / f"year={year}" / f"month={mon}"


def _tail_for_next(feature_path: Path, max_window: int) -> pd.DataFrame:
    cols = [
        "stock_code",
        "minute_timestamp",
        "trading_date",
        "local_timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]
    x = pd.read_parquet(feature_path, columns=cols)
    return (
        x.sort_values(["stock_code", "minute_timestamp"], kind="mergesort")
        .groupby("stock_code", sort=False, observed=True)
        .tail(max_window)
        .reset_index(drop=True)
    )


def _quality_summary(raw: pd.DataFrame, stats: Dict[str, Any]) -> Dict[str, Any]:
    valid = raw.loc[~raw["source_invalid_flag"]]
    local_dates = valid["trading_date"].dropna()
    jumps = (
        valid.sort_values(["stock_code", "minute_timestamp"])
        .groupby("stock_code", observed=True)["close"]
        .pct_change(fill_method=None)
        .abs()
    )
    missing = raw[["open", "high", "low", "close", "volume"]].isna().mean()
    return {
        **stats,
        "row_count_valid": int(len(valid)),
        "date_start": str(local_dates.min().date()) if len(local_dates) else None,
        "date_end": str(local_dates.max().date()) if len(local_dates) else None,
        "stock_count": int(valid["stock_code"].nunique()),
        "price_jump_gt_20pct_count": int((jumps > 0.20).sum()),
        "missing_rate_by_field": {k: float(v) for k, v in missing.items()},
    }


def _compute_month_features(current: pd.DataFrame, carry: pd.DataFrame, cfg: dict) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    current = current.copy()
    current["_is_current"] = True
    if len(carry):
        carry = carry.copy()
        carry["_is_current"] = False
        combined = pd.concat([carry, current], ignore_index=True, sort=False)
    else:
        combined = current
    combined = combined.sort_values(["stock_code", "minute_timestamp"], kind="mergesort")
    parts: List[pd.DataFrame] = []
    groups = combined.groupby("stock_code", sort=False, observed=True)
    total = groups.ngroups
    for i, (_, stock) in enumerate(groups, start=1):
        calculated = compute_symbol_features(stock, cfg)
        parts.append(calculated.loc[calculated["_is_current"].fillna(False)])
        if i == 1 or i % 100 == 0 or i == total:
            log(f"    features symbols {i}/{total}")
    out = pd.concat(parts, ignore_index=True)
    out = add_cross_sectional_features(out, cfg)
    names = feature_names(cfg)
    out, rates = finalize_feature_validity(out, names)
    out.drop(columns=["_is_current"], inplace=True, errors="ignore")
    order = [
        "stock_code",
        "minute_timestamp",
        "trading_date",
        "local_timestamp",
        *names,
        "feature_valid_fraction",
        "feature_invalid_count",
        "feature_valid_mask",
        "cross_section_size",
        "cross_section_missing_count",
        "rank_calculation_timestamp",
        "data_available_cutoff",
    ]
    out = out[order].sort_values(["stock_code", "minute_timestamp"], kind="mergesort")
    for name in names:
        out[name] = out[name].astype("float32")
    return out, rates


def prepare_sp500_data(cfg: dict, force: bool = False, months: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """Validate, persist, feature-engineer, and version every requested month."""
    root = Path(cfg["_root"])
    contract = data_contract(cfg)
    if not contract["ok"]:
        raise RuntimeError(f"data contract failed: {contract['missing']}")
    zip_path = Path(contract["source_zip"])
    processed = resolve_path(cfg, cfg["paths"]["processed_dir"])
    raw_root = processed / "raw_ohlcv"
    feat_root = processed / "minute_features"
    meta_root = processed / "feature_metadata"
    for p in (raw_root, feat_root, meta_root):
        p.mkdir(parents=True, exist_ok=True)

    names = feature_names(cfg)
    schema = {
        "feature_names": names,
        "feature_count": len(names),
        "dtype": cfg["features"]["persist_dtype"],
        "rolling": "trailing_only",
        "label_columns_excluded": ["target_day_open", "target_day_high", "target_day_low", "target_day_close", "target_day_volume"],
        "vwap_definition": "trailing typical-price times volume divided by trailing volume; proxy, not trade-level VWAP",
    }
    schema_hash = _sha_text(_canonical(schema))
    feature_cfg = cfg["features"]
    feature_cfg_hash = _sha_text(_canonical(feature_cfg))
    code_hash = _code_hash(root)
    version_payload = {
        "zip_sha256": contract["zip_sha256"],
        "feature_schema_hash": schema_hash,
        "feature_config_hash": feature_cfg_hash,
        "code_version_hash": code_hash,
    }
    data_version = _sha_text(_canonical(version_payload))
    version_tag = data_version[:16]
    write_json(meta_root / "feature_schema.json", schema)
    _write_text(meta_root / "feature_schema_hash.txt", schema_hash + "\n")
    _write_text(meta_root / "feature_config_hash.txt", feature_cfg_hash + "\n")
    write_json(
        meta_root / "data_version.json",
        {
            **version_payload,
            "data_version": data_version,
            "git_commit": _git_commit(root),
            "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        },
    )

    requested = {str(x)[:7] for x in months} if months else None
    member_rows = enumerate_month_members(zip_path)
    if requested is not None:
        member_rows = [r for r in member_rows if r["source_month"] in requested]
    old_manifest = _read_json(feat_root / "feature_manifest.json", {})
    cached_by_month = {str(r["source_month"]): r for r in old_manifest.get("months", [])}
    month_records: List[Dict[str, Any]] = []
    missing_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    carry = pd.DataFrame()
    max_window = max(int(w) for w in cfg["features"]["windows"])
    failures = resolve_path(cfg, cfg["paths"]["results_dir"]) / "failed_experiments"

    for pos, source in enumerate(member_rows, start=1):
        month = source["source_month"]
        expected_hash = source["source_file_hash"]
        cached = cached_by_month.get(month) or {}
        feat_path = Path(str(cached.get("feature_path", "")))
        cache_ok = (
            not force
            and feat_path.is_file()
            and cached.get("source_file_hash") == expected_hash
            and cached.get("feature_config_hash") == feature_cfg_hash
            and cached.get("feature_schema_hash") == schema_hash
            and cached.get("code_version_hash") == code_hash
        )
        if cache_ok:
            log(f"[{pos}/{len(member_rows)}] reuse feature month {month} -> {feat_path.name}")
            month_records.append(cached)
            carry = _tail_for_next(feat_path, max_window)
            continue
        log(f"[{pos}/{len(member_rows)}] read and persist source month {month}")
        try:
            observed_hash = hash_zip_member(zip_path, source["source_file"])
            if expected_hash and observed_hash != expected_hash:
                raise RuntimeError(f"source member SHA256 mismatch expected={expected_hash} observed={observed_hash}")
            raw, read_stats = read_month_member(zip_path, source["source_file"], cfg["calendar"]["timezone"])
            quality = _quality_summary(raw, read_stats)
            raw_part = _partition(raw_root, month)
            raw_part.mkdir(parents=True, exist_ok=True)
            raw_path = raw_part / f"part-{version_tag}.parquet"
            raw.to_parquet(raw_path, index=False, compression=cfg["features"]["parquet_compression"])
            rth = regular_session(raw, cfg["calendar"]["regular_session_start"], cfg["calendar"]["regular_session_end"])
            rth = rth.loc[~rth["source_invalid_flag"]].copy()
            features, rates = _compute_month_features(rth, carry, cfg)
            feat_part = _partition(feat_root, month)
            feat_part.mkdir(parents=True, exist_ok=True)
            feature_path = feat_part / f"part-{version_tag}.parquet"
            features.to_parquet(feature_path, index=False, compression=cfg["features"]["parquet_compression"])

            generation = pd.Timestamp.now(tz="UTC").isoformat()
            record = {
                "source_month": month,
                "source_file": source["source_file"],
                "source_file_hash": observed_hash,
                "row_count_raw": int(read_stats["row_count_raw"]),
                "row_count_valid": int(quality["row_count_valid"]),
                "row_count_regular_session": int(len(rth)),
                "date_start": quality["date_start"],
                "date_end": quality["date_end"],
                "stock_count": quality["stock_count"],
                "feature_count": len(names),
                "feature_schema_hash": schema_hash,
                "feature_config_hash": feature_cfg_hash,
                "code_version_hash": code_hash,
                "data_version": data_version,
                "duplicate_count": quality["duplicate_count"],
                "invalid_count": quality["invalid_count"],
                "invalid_rate_by_feature": {k: v["invalid_rate"] for k, v in rates.items()},
                "missing_rate_by_feature": {k: v["missing_rate"] for k, v in rates.items()},
                "timezone": cfg["calendar"]["timezone"],
                "generation_timestamp": generation,
                "code_version": _git_commit(root),
                "raw_path": str(raw_path),
                "feature_path": str(feature_path),
                "quality": quality,
            }
            month_records.append(record)
            for name in names:
                missing_rows.append({"source_month": month, "feature": name, **rates[name]})
                s = features[name]
                summary_rows.append(
                    {
                        "source_month": month,
                        "feature": name,
                        "count": int(s.count()),
                        "mean": float(s.mean()),
                        "std": float(s.std(ddof=0)),
                        "min": float(s.min()),
                        "max": float(s.max()),
                    }
                )
            carry = (
                rth.sort_values(["stock_code", "minute_timestamp"], kind="mergesort")
                .groupby("stock_code", sort=False, observed=True)
                .tail(max_window)
                .reset_index(drop=True)
            )
            del raw, rth, features
        except Exception as exc:
            failures.mkdir(parents=True, exist_ok=True)
            failure = {
                "status": "failed",
                "phase": "feature_generation",
                "source_month": month,
                "source_file": source["source_file"],
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "retry_status": "not_retried",
                "timestamp": pd.Timestamp.now(tz="UTC").isoformat(),
            }
            write_json(failures / f"feature_generation_{month}_{int(time.time())}.json", failure)
            raise

        manifest = {
            "data_version": data_version,
            "feature_schema_hash": schema_hash,
            "feature_config_hash": feature_cfg_hash,
            "code_version_hash": code_hash,
            "months": sorted(month_records, key=lambda r: r["source_month"]),
        }
        write_json(feat_root / "feature_manifest.json", manifest)
        write_json(raw_root / "manifest.json", {"data_version": data_version, "months": manifest["months"]})

    manifest = {
        "data_version": data_version,
        "feature_schema_hash": schema_hash,
        "feature_config_hash": feature_cfg_hash,
        "code_version_hash": code_hash,
        "months": sorted(month_records, key=lambda r: r["source_month"]),
    }
    write_json(feat_root / "feature_manifest.json", manifest)
    write_json(raw_root / "manifest.json", {"data_version": data_version, "months": manifest["months"]})
    write_json(meta_root / "source_file_manifest.json", {"zip": contract, "months": manifest["months"]})
    if missing_rows:
        pd.DataFrame(missing_rows).to_csv(meta_root / "feature_missingness.csv", index=False)
    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(meta_root / "feature_summary.csv", index=False)
    return manifest
