"""Experiment logging helpers."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Mapping


def log(msg: str) -> None:
    print(msg, flush=True)


class Tee:
    def __init__(self, path: Path, stream=None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.f = path.open("a", encoding="utf-8")
        self.stdout = stream if stream is not None else sys.stdout

    def write(self, x: str) -> int:
        self.stdout.write(x)
        self.f.write(x)
        self.f.flush()
        return len(x)

    def flush(self) -> None:
        self.stdout.flush()
        self.f.flush()

    def isatty(self) -> bool:
        return bool(getattr(self.stdout, "isatty", lambda: False)())

    def close(self) -> None:
        self.f.close()

    @classmethod
    def install(cls, path: Path) -> "Tee":
        """Mirror stdout and stderr to `path` while still printing in the terminal."""
        tee = cls(path, stream=sys.__stdout__)
        sys.stdout = tee  # type: ignore
        sys.stderr = tee  # type: ignore
        return tee


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str, ensure_ascii=False), encoding="utf-8")


def append_csv_row(path: Path, row: Mapping[str, Any]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.is_file()
    with path.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            w.writeheader()
        w.writerow(dict(row))
