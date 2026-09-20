"""Auditable pandas loader for the S&P 500 Databento ZIP delivery."""

from __future__ import annotations

import hashlib
import io
import json
import re
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd

from utils.config import resolve_path

STANDARD_COLUMNS = [
    "stock_code",
    "minute_timestamp",
    "trading_date",
    "open",
    "high",
    "low",
    "close",
    "volume",
]

IDENTIFIER_ALIASES = ("stock_code", "symbol", "ticker", "security_id", "instrument_id")
TIMESTAMP_ALIASES = ("minute_timestamp", "ts_event", "timestamp", "datetime", "date")
OHLCV_ALIASES = {
    "open": ("open", "Open", "开盘"),
    "high": ("high", "High", "最高"),
    "low": ("low", "Low", "最低"),
    "close": ("close", "Close", "收盘"),
    "volume": ("volume", "Volume", "成交量", "成交量(股)"),
}


def sha256_path(path: Path, block: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def _json_member(zf: zipfile.ZipFile, suffix: str) -> Dict[str, Any]:
    names = [n for n in zf.namelist() if n.endswith(suffix)]
    if len(names) != 1:
        raise RuntimeError(f"expected one {suffix} in ZIP, found {names}")
    return json.loads(zf.read(names[0]))


def enumerate_month_members(zip_path: Path) -> List[Dict[str, Any]]:
    """List monthly .zst members and reconcile them with the vendor manifest."""
    with zipfile.ZipFile(zip_path) as zf:
        manifest = _json_member(zf, "/manifest.json")
        expected = {
            str(row["filename"]): str(row.get("hash", "")).removeprefix("sha256:")
            for row in manifest.get("files", [])
        }
        rows: List[Dict[str, Any]] = []
        for info in zf.infolist():
            if not info.filename.lower().endswith(".zst"):
                continue
            base = Path(info.filename).name
            match = re.search(r"(20\d{2})(\d{2})\d{2}-(20\d{2})(\d{2})\d{2}", base)
            if not match:
                raise RuntimeError(f"cannot infer month from {info.filename}")
            rows.append(
                {
                    "source_month": f"{match.group(1)}-{match.group(2)}",
                    "source_file": info.filename,
                    "compressed_size": int(info.file_size),
                    "zip_compress_type": int(info.compress_type),
                    "source_file_hash": expected.get(base, ""),
                }
            )
    rows.sort(key=lambda r: (r["source_month"], r["source_file"]))
    if not rows:
        raise RuntimeError(f"no .zst files in {zip_path}")
    return rows


def hash_zip_member(zip_path: Path, member: str, block: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with zipfile.ZipFile(zip_path) as zf, zf.open(member) as fh:
        for chunk in iter(lambda: fh.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def detect_member_format(member: str, first_bytes: bytes) -> str:
    low = member.lower()
    if low.endswith((".csv.zst", ".csv.zstd")):
        return "csv"
    if first_bytes[:4] == b"PAR1":
        return "parquet"
    stripped = first_bytes.lstrip()
    if stripped.startswith((b"{", b"[")):
        return "json"
    if b"," in first_bytes.splitlines()[0]:
        return "csv"
    raise RuntimeError(f"unsupported content format for {member}")


def _choose(columns: Iterable[str], aliases: Iterable[str], field: str) -> str:
    cols = list(columns)
    by_lower = {str(c).lower(): str(c) for c in cols}
    for alias in aliases:
        if alias in cols:
            return alias
        if alias.lower() in by_lower:
            return by_lower[alias.lower()]
    raise RuntimeError(f"missing {field}; columns={cols}")


def normalize_ohlcv(raw: pd.DataFrame, timezone: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    ts_col = _choose(raw.columns, TIMESTAMP_ALIASES, "timestamp")
    id_col = _choose(raw.columns, IDENTIFIER_ALIASES, "stock identifier")
    colmap = {name: _choose(raw.columns, aliases, name) for name, aliases in OHLCV_ALIASES.items()}
    ts = pd.to_datetime(raw[ts_col], utc=True, errors="coerce")
    out = pd.DataFrame(
        {
            "stock_code": raw[id_col].astype("string").str.strip().str.upper(),
            "minute_timestamp": ts,
        }
    )
    for name, source in colmap.items():
        out[name] = pd.to_numeric(raw[source], errors="coerce")
    local = ts.dt.tz_convert(timezone)
    out["trading_date"] = local.dt.tz_localize(None).dt.normalize()
    out["local_timestamp"] = local.dt.tz_localize(None)
    for optional in ("instrument_id", "publisher_id", "rtype", "vwap", "trade_count", "amount", "session", "exchange"):
        if optional in raw.columns and optional not in out.columns:
            out[optional] = raw[optional]

    duplicate = out.duplicated(["stock_code", "minute_timestamp"], keep="first")
    bad_ts = out["minute_timestamp"].isna() | out["stock_code"].isna() | out["stock_code"].eq("")
    prices = out[["open", "high", "low", "close"]]
    bad_price = (~np.isfinite(prices)).any(axis=1) | (prices <= 0).any(axis=1)
    bad_volume = ~np.isfinite(out["volume"]) | (out["volume"] < 0)
    bad_ohlc = (out["high"] < out[["open", "close"]].max(axis=1)) | (
        out["low"] > out[["open", "close"]].min(axis=1)
    )
    invalid = bad_ts | bad_price | bad_volume | bad_ohlc
    stats = {
        "duplicate_count": int(duplicate.sum()),
        "invalid_timestamp_count": int(bad_ts.sum()),
        "invalid_price_count": int(bad_price.sum()),
        "invalid_volume_count": int(bad_volume.sum()),
        "invalid_ohlc_count": int(bad_ohlc.sum()),
        "invalid_count": int(invalid.sum()),
        "identifier_source": id_col,
        "timestamp_source": ts_col,
        "column_mapping": {"stock_code": id_col, "minute_timestamp": ts_col, **colmap},
    }
    out["source_invalid_flag"] = invalid
    out = out.loc[~duplicate].sort_values(["stock_code", "minute_timestamp"], kind="mergesort").reset_index(drop=True)
    return out, stats


def read_month_member(zip_path: Path, member: str, timezone: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Read a real ZIP member with pandas after inspecting its decompressed header."""
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(member) as probe:
            import zstandard as zstd

            reader = zstd.ZstdDecompressor().stream_reader(probe)
            head = reader.read(4096)
        fmt = detect_member_format(member, head)
        with zf.open(member) as fh:
            if fmt == "csv":
                raw = pd.read_csv(fh, compression="zstd", low_memory=False)
            elif fmt == "parquet":
                raw = pd.read_parquet(io.BytesIO(fh.read()))
            elif fmt == "json":
                raw = pd.read_json(fh, compression="zstd", lines=True)
            else:  # pragma: no cover - detect_member_format is exhaustive
                raise RuntimeError(fmt)
    out, stats = normalize_ohlcv(raw, timezone)
    stats.update({"detected_format": fmt, "raw_columns": list(raw.columns), "row_count_raw": int(len(raw))})
    return out, stats


def regular_session(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    clock = df["local_timestamp"].dt.strftime("%H:%M")
    return df.loc[(clock >= str(start)) & (clock <= str(end))].copy()


def data_contract(cfg: dict) -> Dict[str, Any]:
    zip_path = resolve_path(cfg, cfg["paths"]["source_zip"])
    processed = resolve_path(cfg, cfg["paths"]["processed_dir"])
    results = resolve_path(cfg, cfg["paths"]["results_dir"])
    missing: List[str] = []
    notes: List[str] = []
    members: List[Dict[str, Any]] = []
    if not zip_path.is_file():
        missing.append(str(zip_path))
    else:
        try:
            members = enumerate_month_members(zip_path)
            notes.append(f"ZIP contains {len(members)} monthly .zst members")
        except Exception as exc:
            missing.append(f"ZIP contract: {exc}")
    for path in (processed, results):
        try:
            path.mkdir(parents=True, exist_ok=True)
            probe = path / ".write_probe"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
        except Exception as exc:
            missing.append(f"not writable: {path}: {exc}")
    stat = zip_path.stat() if zip_path.is_file() else None
    return {
        "ok": not missing,
        "missing": missing,
        "notes": notes,
        "source_zip": str(zip_path),
        "zip_size": int(stat.st_size) if stat else None,
        "zip_mtime_ns": int(stat.st_mtime_ns) if stat else None,
        "zip_sha256": sha256_path(zip_path) if stat else None,
        "members": members,
        "standard_columns": STANDARD_COLUMNS,
        "timezone": cfg["calendar"]["timezone"],
    }
