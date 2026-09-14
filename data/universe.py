"""CSI500 universe. Join only by stock_code, never by name."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


EX_MAP = {"sh": "SH", "sz": "SZ", "bj": "BJ"}


def normalize_stock_code(raw: str, prefix: str = "") -> str:
    s = str(raw).strip().upper().replace(" ", "")
    if "." in s:
        num, ex = s.split(".", 1)
        return f"{num.zfill(6)}.{ex[:2]}"
    if s.startswith(("SH", "SZ", "BJ")) and len(s) >= 8:
        return f"{s[2:].zfill(6)}.{s[:2]}"
    p = prefix.strip().lower()
    ex = EX_MAP.get(p, "")
    if ex:
        return f"{s.zfill(6)}.{ex}"
    if s.startswith("6") or s.startswith("9"):
        return f"{s.zfill(6)}.SH"
    if s.startswith(("0", "3")):
        return f"{s.zfill(6)}.SZ"
    return f"{s.zfill(6)}.SZ"


def filename_to_code(name: str) -> str:
    stem = Path(name).stem.lower()
    if stem.startswith(("sh", "sz", "bj")):
        return normalize_stock_code(stem[2:], stem[:2])
    return normalize_stock_code(stem)


def load_constituents(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "code" not in df.columns:
        raise RuntimeError(f"constituents missing code column: {path}")
    prefix = df["prefix"].astype(str) if "prefix" in df.columns else ""
    codes = [
        normalize_stock_code(c, p)
        for c, p in zip(df["code"].astype(str), prefix if len(prefix) else [""] * len(df))
    ]
    out = df.copy()
    out["stock_code"] = codes
    if "filename" in out.columns:
        out["filename"] = out["filename"].astype(str)
    else:
        out["filename"] = [
            ("sh" if c.endswith(".SH") else "sz" if c.endswith(".SZ") else "bj") + c.split(".")[0] + ".csv"
            for c in codes
        ]
    return out.drop_duplicates("stock_code")


def universe_codes(path: Path, n_cap: int) -> List[str]:
    df = load_constituents(path)
    return df["stock_code"].astype(str).tolist()[: int(n_cap)]


def code_filename_map(path: Path) -> Dict[str, str]:
    df = load_constituents(path)
    return dict(zip(df["stock_code"].astype(str), df["filename"].astype(str)))
