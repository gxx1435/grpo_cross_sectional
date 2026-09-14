"""Train-only robust scalers. Test/Val never participate in fit."""

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
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        self.median = np.nanmedian(x, axis=0)
        self.mad = np.nanmedian(np.abs(x - self.median), axis=0)
        self.n = int(np.isfinite(x).sum())
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
    """Per-channel mean/std fitted on train asofs only. Val/Test only transform."""

    def __init__(self, eps: float = 1e-6, zmax: float = 5.0) -> None:
        self.eps = float(eps)
        self.zmax = float(zmax)
        self.count_x: Optional[np.ndarray] = None
        self.mean_x: Optional[np.ndarray] = None
        self.m2_x: Optional[np.ndarray] = None
        self.count_a: Optional[np.ndarray] = None
        self.mean_a: Optional[np.ndarray] = None
        self.m2_a: Optional[np.ndarray] = None
        self.n_days = 0
        self.fit_range: Optional[Dict[str, str]] = None

    @staticmethod
    def _ensure(c: int, count, mean, m2):
        if mean is None:
            return np.zeros(c, np.int64), np.zeros(c, np.float64), np.zeros(c, np.float64)
        return count, mean, m2

    @staticmethod
    def _update(flat: np.ndarray, count: np.ndarray, mean: np.ndarray, m2: np.ndarray) -> None:
        if flat.size == 0:
            return
        flat = np.asarray(flat, dtype=np.float64)
        if flat.ndim == 1:
            flat = flat.reshape(-1, 1)
        for j in range(flat.shape[1]):
            v = flat[:, j]
            v = v[np.isfinite(v)]
            if v.size == 0:
                continue
            n = int(count[j])
            nb = int(v.size)
            mb = float(v.mean())
            m2b = float(np.square(v - mb).sum())
            if n <= 0:
                count[j] = nb
                mean[j] = mb
                m2[j] = m2b
                continue
            delta = mb - float(mean[j])
            n2 = n + nb
            mean2 = float(mean[j]) + delta * nb / n2
            m2[j] = float(m2[j]) + m2b + delta * (mb - mean2) * n
            mean[j] = mean2
            count[j] = n2

    def update_x(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> None:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 3:
            raise RuntimeError("update_x expected [N, T, F]")
        f = int(x.shape[-1])
        self.count_x, self.mean_x, self.m2_x = self._ensure(f, self.count_x, self.mean_x, self.m2_x)
        m = np.ones(x.shape[:2], dtype=bool) if mask is None else np.asarray(mask) > 0.5
        self._update(x.reshape(-1, f)[m.reshape(-1)], self.count_x, self.mean_x, self.m2_x)
        self.n_days += 1

    def update_a(self, a: np.ndarray) -> None:
        a = np.asarray(a, dtype=np.float64)
        if a.ndim == 1:
            a = a.reshape(-1, 1)
        c = int(a.shape[-1])
        self.count_a, self.mean_a, self.m2_a = self._ensure(c, self.count_a, self.mean_a, self.m2_a)
        self._update(a.reshape(-1, c), self.count_a, self.mean_a, self.m2_a)

    def _std(self, count: np.ndarray, m2: np.ndarray) -> np.ndarray:
        n = np.maximum(count.astype(np.float64), 2.0)
        return np.sqrt(np.clip(m2 / (n - 1.0), 0.0, None)) + self.eps

    def transform_x(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        if self.mean_x is None:
            raise RuntimeError("ChannelZScore.transform_x before fit")
        z = (np.asarray(x, dtype=np.float64) - self.mean_x) / self._std(self.count_x, self.m2_x)
        out = np.clip(z, -self.zmax, self.zmax).astype(np.float32)
        np.nan_to_num(out, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        return out

    def transform_a(self, a: np.ndarray) -> np.ndarray:
        if self.mean_a is None:
            return np.asarray(a, dtype=np.float32)
        z = (np.asarray(a, dtype=np.float64) - self.mean_a) / self._std(self.count_a, self.m2_a)
        out = np.clip(z, -self.zmax, self.zmax).astype(np.float32)
        np.nan_to_num(out, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        return out

    def save(self, path: Path) -> None:
        np.savez(
            Path(path),
            mean_x=self.mean_x,
            m2_x=self.m2_x,
            count_x=self.count_x,
            mean_a=np.array([]) if self.mean_a is None else self.mean_a,
            m2_a=np.array([]) if self.m2_a is None else self.m2_a,
            count_a=np.array([]) if self.count_a is None else self.count_a,
            eps=self.eps,
            zmax=self.zmax,
            n_days=self.n_days,
        )

    @classmethod
    def load(cls, path: Path) -> "ChannelZScore":
        blob = np.load(Path(path), allow_pickle=True)
        obj = cls(eps=float(blob["eps"]), zmax=float(blob["zmax"]))
        obj.mean_x = np.asarray(blob["mean_x"], dtype=np.float64)
        obj.m2_x = np.asarray(blob["m2_x"], dtype=np.float64)
        obj.count_x = np.asarray(blob["count_x"], dtype=np.int64)
        ma = np.asarray(blob["mean_a"], dtype=np.float64)
        if ma.size:
            obj.mean_a = ma
            obj.m2_a = np.asarray(blob["m2_a"], dtype=np.float64)
            obj.count_a = np.asarray(blob["count_a"], dtype=np.int64)
        obj.n_days = int(blob["n_days"]) if "n_days" in blob.files else 0
        return obj
