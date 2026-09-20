"""Train-only feature and reward normalizers."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np


class RobustScaler:
    def __init__(self, eps: float = 1e-8, zmax: float = 5.0) -> None:
        self.eps = float(eps)
        self.zmax = float(zmax)
        self.median: Optional[np.ndarray] = None
        self.mad: Optional[np.ndarray] = None
        self.n = 0
        self.fit_range: Optional[Dict[str, str]] = None

    def fit(self, x: np.ndarray, fit_range: Optional[Dict[str, str]] = None) -> "RobustScaler":
        values = np.asarray(x, dtype=np.float64)
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        self.median = np.nanmedian(values, axis=0)
        self.mad = np.nanmedian(np.abs(values - self.median), axis=0)
        self.n = int(np.isfinite(values).sum())
        self.fit_range = fit_range
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.median is None or self.mad is None:
            raise RuntimeError("RobustScaler.transform before fit")
        z = (np.asarray(x, dtype=np.float64) - self.median) / (1.4826 * self.mad + self.eps)
        return np.clip(z, -self.zmax, self.zmax).astype(np.float32)

    def state(self) -> Dict[str, object]:
        return {
            "median": None if self.median is None else self.median.tolist(),
            "mad": None if self.mad is None else self.mad.tolist(),
            "eps": self.eps,
            "zmax": self.zmax,
            "n": self.n,
            "fit_range": self.fit_range,
            "train_only": True,
        }


class ChannelZScore:
    """Streaming per-feature mean/std fit exclusively on training as-of dates."""

    def __init__(self, eps: float = 1e-6, zmax: float = 5.0) -> None:
        self.eps = float(eps)
        self.zmax = float(zmax)
        self.count_x: Optional[np.ndarray] = None
        self.mean_x: Optional[np.ndarray] = None
        self.m2_x: Optional[np.ndarray] = None
        self.n_days = 0
        self.fit_range: Optional[Dict[str, str]] = None

    def update_x(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> None:
        values = np.asarray(x, dtype=np.float64)
        if values.ndim != 3:
            raise RuntimeError("update_x expected [stocks, minutes, features]")
        f = int(values.shape[-1])
        if self.mean_x is None:
            self.count_x = np.zeros(f, dtype=np.int64)
            self.mean_x = np.zeros(f, dtype=np.float64)
            self.m2_x = np.zeros(f, dtype=np.float64)
        valid_rows = np.ones(values.shape[:2], dtype=bool) if mask is None else np.asarray(mask) > 0.5
        flat = values.reshape(-1, f)[valid_rows.reshape(-1)]
        for j in range(f):
            column = flat[:, j]
            column = column[np.isfinite(column)]
            if not len(column):
                continue
            old_n = int(self.count_x[j])
            batch_n = int(len(column))
            batch_mean = float(column.mean())
            batch_m2 = float(np.square(column - batch_mean).sum())
            if old_n == 0:
                self.count_x[j] = batch_n
                self.mean_x[j] = batch_mean
                self.m2_x[j] = batch_m2
                continue
            total = old_n + batch_n
            delta = batch_mean - float(self.mean_x[j])
            new_mean = float(self.mean_x[j]) + delta * batch_n / total
            self.m2_x[j] = float(self.m2_x[j]) + batch_m2 + delta * (batch_mean - new_mean) * old_n
            self.mean_x[j] = new_mean
            self.count_x[j] = total
        self.n_days += 1

    def _std(self) -> np.ndarray:
        if self.count_x is None or self.m2_x is None:
            raise RuntimeError("normalizer not fitted")
        return np.sqrt(np.clip(self.m2_x / np.maximum(self.count_x - 1, 1), 0.0, None)) + self.eps

    def transform_x(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        if self.mean_x is None:
            raise RuntimeError("ChannelZScore.transform_x before fit")
        z = (np.asarray(x, dtype=np.float64) - self.mean_x) / self._std()
        out = np.clip(z, -self.zmax, self.zmax).astype(np.float32)
        np.nan_to_num(out, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        if mask is not None:
            out *= (np.asarray(mask) > 0.5)[..., None]
        return out

    def state(self) -> Dict[str, object]:
        return {
            "mean": None if self.mean_x is None else self.mean_x.tolist(),
            "m2": None if self.m2_x is None else self.m2_x.tolist(),
            "count": None if self.count_x is None else self.count_x.tolist(),
            "n_days": self.n_days,
            "fit_range": self.fit_range,
            "eps": self.eps,
            "zmax": self.zmax,
            "train_only": True,
        }

    def save(self, path: Path) -> None:
        np.savez(
            Path(path),
            mean_x=self.mean_x,
            m2_x=self.m2_x,
            count_x=self.count_x,
            eps=self.eps,
            zmax=self.zmax,
            n_days=self.n_days,
            fit_start=(self.fit_range or {}).get("start", ""),
            fit_end=(self.fit_range or {}).get("end", ""),
        )

    @classmethod
    def load(cls, path: Path) -> "ChannelZScore":
        blob = np.load(Path(path), allow_pickle=True)
        obj = cls(float(blob["eps"]), float(blob["zmax"]))
        obj.mean_x = np.asarray(blob["mean_x"], dtype=np.float64)
        obj.m2_x = np.asarray(blob["m2_x"], dtype=np.float64)
        obj.count_x = np.asarray(blob["count_x"], dtype=np.int64)
        obj.n_days = int(blob["n_days"])
        obj.fit_range = {"start": str(blob["fit_start"]), "end": str(blob["fit_end"])}
        return obj
