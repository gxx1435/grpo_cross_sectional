"""Load and merge YAML/JSON configs. No scattered hyperparameter defaults."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_FILES = (
    "configs/default.yaml",
    "configs/prediction.yaml",
    "configs/ssfm.yaml",
    "configs/rl.yaml",
    "configs/experiments.yaml",
)


def _deep_update(base: Dict[str, Any], extra: Mapping[str, Any]) -> Dict[str, Any]:
    for k, v in extra.items():
        if isinstance(v, Mapping) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = copy.deepcopy(v)
    return base


def load_yaml(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        obj = json.loads(text)
    else:
        obj = yaml.safe_load(text)
    if obj is None:
        return {}
    if not isinstance(obj, dict):
        raise TypeError(f"config {path} must be a mapping")
    return obj


def load_config(
    extra_files: Optional[Iterable[str]] = None,
    overrides: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {}
    files = list(DEFAULT_CONFIG_FILES)
    if extra_files:
        files.extend(extra_files)
    for rel in files:
        p = ROOT / rel if not Path(rel).is_absolute() else Path(rel)
        if not p.is_file():
            raise FileNotFoundError(f"missing config: {p}")
        _deep_update(cfg, load_yaml(p))
    if overrides:
        _deep_update(cfg, overrides)
    cfg["_root"] = str(ROOT)
    return cfg


def cfg_get(cfg: Mapping[str, Any], dotted: str, default: Any = None) -> Any:
    cur: Any = cfg
    for part in dotted.split("."):
        if not isinstance(cur, Mapping) or part not in cur:
            return default
        cur = cur[part]
    return cur


def refresh_prediction_train_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Re-read SFT loss knobs from prediction.yaml so the next model can change mid-run."""
    pred = load_yaml(ROOT / "configs/prediction.yaml").get("prediction") or {}
    dst = cfg.setdefault("prediction", {})
    for k in ("rank_weight", "sft_cs_zscore", "val_select"):
        if k in pred:
            dst[k] = pred[k]
    return cfg


def resolve_path(cfg: Mapping[str, Any], rel: str) -> Path:
    p = Path(rel)
    if p.is_absolute():
        return p
    return Path(cfg.get("_root", ROOT)) / rel
