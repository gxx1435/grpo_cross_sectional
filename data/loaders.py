"""Minute OHLCV and auction loaders. Align only on (stock_code, trading_date)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from data.universe import filename_to_code, normalize_stock_code

RAW_COLS = {
    "日期": "ts",
    "开盘": "open",
    "最高": "high",
    "最低": "low",
    "收盘": "close",
    "成交量(股)": "volume",
    "成交额(元)": "amount",
    "换手率(%)": "turnover",
}

AUCTION_COLS = {
    "日期": "trading_date",
    "代码": "raw_code",
    "集合竞价涨幅%": "auction_return_pct",
    "集合竞价成交价": "auction_price",
    "集合竞价成交量(股)": "auction_volume",
    "集合竞价成交额(元)": "auction_amount",
    "集合竞价换手率1": "auction_turnover_1",
    "集合竞价换手率2": "auction_turnover_2",
    "集合竞价量比1": "auction_vol_ratio_1",
    "集合竞价量比2": "auction_vol_ratio_2",
    "集合竞价量比3": "auction_vol_ratio_3",
}

AUCTION_FEATURE_NAMES = [
    "auction_return",
    "auction_price",
    "auction_volume",
    "auction_amount",
    "auction_turnover_1",
    "auction_turnover_2",
    "auction_vol_ratio_1",
    "auction_vol_ratio_2",
    "auction_vol_ratio_3",
]


def read_minute_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["日期"])
    df = df.rename(columns=RAW_COLS)
    df = df.set_index("ts").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    keep = [c for c in ("open", "high", "low", "close", "volume", "amount", "turnover") if c in df.columns]
    return df[keep]


def find_stock_csv(stock_code: str, pool_dirs: Sequence[Path]) -> List[Path]:
    num, ex = stock_code.split(".")
    fname = f"{ex.lower()}{num}.csv"
    found = []
    for d in pool_dirs:
        p = Path(d) / fname
        if p.is_file():
            found.append(p)
    return found


def load_stock_minutes(
    stock_code: str,
    pool_dirs: Sequence[Path],
    start: str,
    end: str,
) -> pd.DataFrame:
    parts = [read_minute_csv(p) for p in find_stock_csv(stock_code, pool_dirs)]
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts).sort_index()
    df = df[~df.index.duplicated(keep="first")]
    t0, t1 = pd.Timestamp(start), pd.Timestamp(end) + pd.Timedelta(days=1)
    return df[(df.index >= t0) & (df.index < t1)]


def load_auction_dir(path: Path) -> pd.DataFrame:
    rows = []
    for p in sorted(Path(path).glob("*.csv")):
        try:
            raw = pd.read_csv(p)
        except Exception:
            continue
        raw = raw.rename(columns=AUCTION_COLS)
        need = set(AUCTION_COLS.values())
        if not need.issubset(raw.columns):
            continue
        raw["stock_code"] = raw["raw_code"].map(lambda x: normalize_stock_code(str(x)))
        raw["trading_date"] = pd.to_datetime(raw["trading_date"].astype(str), format="%Y%m%d", errors="coerce")
        raw = raw.dropna(subset=["trading_date", "stock_code"])
        rows.append(raw)
    if not rows:
        return pd.DataFrame()
    df = pd.concat(rows, ignore_index=True)
    df["auction_return"] = pd.to_numeric(df["auction_return_pct"], errors="coerce") / 100.0
    for c in AUCTION_FEATURE_NAMES[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["auction_available_clock"] = "09:25:00"
    return df[["stock_code", "trading_date"] + AUCTION_FEATURE_NAMES + ["auction_available_clock"]]


def load_all_auctions(dirs: Sequence[Path]) -> pd.DataFrame:
    parts = [load_auction_dir(Path(d)) for d in dirs]
    parts = [p for p in parts if len(p)]
    if not parts:
        raise FileNotFoundError(f"no auction CSVs under {list(dirs)}")
    df = pd.concat(parts, ignore_index=True)
    df = df.drop_duplicates(["stock_code", "trading_date"], keep="last")
    return df.sort_values(["trading_date", "stock_code"])


def data_contract(cfg: dict) -> Dict[str, object]:
    from utils.config import resolve_path

    missing: List[str] = []
    notes: List[str] = []
    cons = resolve_path(cfg, cfg["paths"]["constituents_csv"])
    if not cons.is_file():
        missing.append(str(cons))
    pools = [resolve_path(cfg, p) for p in cfg["paths"]["minute_pools"]]
    for p in pools:
        if not p.is_dir():
            missing.append(str(p))
        else:
            n = len(list(p.glob("*.csv")))
            notes.append(f"{p}: {n} csv")
            if n == 0:
                missing.append(f"{p} (empty)")
    auc = [resolve_path(cfg, p) for p in cfg["paths"]["auction_dirs"]]
    for p in auc:
        if not p.is_dir():
            missing.append(str(p))
        else:
            n = len(list(p.glob("*.csv")))
            notes.append(f"{p}: {n} auction csv")
            if n == 0:
                missing.append(f"{p} (empty)")
    return {
        "ok": len(missing) == 0,
        "missing": missing,
        "notes": notes,
        "pools": [str(p) for p in pools],
        "auction_dirs": [str(p) for p in auc],
        "constituents": str(cons),
    }
