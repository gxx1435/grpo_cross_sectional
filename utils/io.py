"""IO helpers and experiment_id generation."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Mapping


def experiment_id(name: str, test_month: str, oos_mode: str, extra: str = "") -> str:
    raw = f"{name}|{test_month}|{oos_mode}|{extra}|{int(time.time() * 1000)}"
    h = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:10]
    safe = f"{name}_{test_month}_{oos_mode}_{h}"
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in safe)


def dump_yaml_copy(cfg: Mapping[str, Any], path: Path) -> None:
    import yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(dict(cfg), allow_unicode=True, sort_keys=False), encoding="utf-8")


def file_sha256(path: Path, nbytes: int = 0) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        if nbytes > 0:
            h.update(f.read(nbytes))
        else:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    return h.hexdigest()


def schema_hash(names: list) -> str:
    return hashlib.sha256(json.dumps(list(names), ensure_ascii=False).encode("utf-8")).hexdigest()
