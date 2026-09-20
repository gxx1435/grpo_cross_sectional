"""Dense causal research store backed by versioned S&P 500 feature parquet."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from data.minute_features import feature_names
from data.sp500_pipeline import prepare_sp500_data
from data.trading_calendar import session_times, stamp
from data.universe import UNIVERSE_TYPE, decision_exclusion_reason, evaluation_exclusion_reason, universe_manifest_rows
from utils.config import resolve_path
from utils.logging import log, write_json


class ResearchStore:
    def __init__(self, cfg: dict, prepare: bool = True) -> None:
        self.cfg = cfg
        self.processed = resolve_path(cfg, cfg["paths"]["processed_dir"])
        self.feature_root = self.processed / "minute_features"
        self.meta_root = self.processed / "feature_metadata"
        manifest_path = self.feature_root / "feature_manifest.json"
        if prepare and not manifest_path.is_file():
            prepare_sp500_data(cfg)
        if not manifest_path.is_file():
            raise FileNotFoundError(f"feature manifest not found: {manifest_path}")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.data_version = str(self.manifest["data_version"])
        self.feature_schema_hash = str(self.manifest["feature_schema_hash"])
        self.feature_config_hash = str(self.manifest["feature_config_hash"])
        self.feature_names = feature_names(cfg)
        self.feat_dim = len(self.feature_names)
        self.template = session_times(cfg)
        self.bars = len(self.template)
        self.lookback_days = int(cfg["calendar"]["lookback_days"])
        self.cache_dir = self.processed / "dense_cache" / self.data_version[:16]
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.meta_path = self.cache_dir / "store_meta.json"
        self.feat_path = self.cache_dir / "features.f32.dat"
        self.mask_path = self.cache_dir / "minute_mask.u8.dat"
        self.open_path = self.cache_dir / "target_open.f32.npy"
        self.close_path = self.cache_dir / "target_close.f32.npy"
        self.target_count_path = self.cache_dir / "target_minute_count.i16.npy"
        self.session_end_path = self.cache_dir / "session_end_slot.i16.npy"
        self._load_or_build()

    def _feature_paths(self) -> List[Path]:
        paths = [Path(row["feature_path"]) for row in self.manifest.get("months", [])]
        missing = [str(p) for p in paths if not p.is_file()]
        if missing:
            raise FileNotFoundError(f"missing feature partitions: {missing}")
        return paths

    def _load_or_build(self) -> None:
        needed = [self.meta_path, self.feat_path, self.mask_path, self.open_path, self.close_path, self.target_count_path, self.session_end_path]
        if all(p.is_file() for p in needed):
            meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
            if meta.get("data_version") == self.data_version and meta.get("feature_schema_hash") == self.feature_schema_hash:
                self._attach(meta)
                log(f"  dense S&P500 store cache shape={self.feats.shape} days={self.T} N={self.N} F={self.feat_dim}")
                return
        self._build()

    def _attach(self, meta: Dict[str, object]) -> None:
        self.kept = [str(x) for x in meta["stock_codes"]]
        self.codes = list(self.kept)
        self.days = [pd.Timestamp(x) for x in meta["days"]]
        self.T, self.N = len(self.days), len(self.kept)
        self.session_end_slot = np.load(self.session_end_path)
        self.feats = np.memmap(self.feat_path, dtype=np.float32, mode="r", shape=(self.T, self.bars, self.N, self.feat_dim))
        self.mask = np.memmap(self.mask_path, dtype=np.uint8, mode="r", shape=(self.T, self.bars, self.N))
        self.open = np.load(self.open_path)
        self.close = np.load(self.close_path)
        self.target_count = np.load(self.target_count_path)
        self.feature_meta = dict(meta.get("feature_meta") or {})
        self.filter_reasons = dict(meta.get("filter_reasons") or {})
        self.raw_zip_hash = str(meta.get("raw_zip_hash", ""))
        self.universe_type = UNIVERSE_TYPE

    def _build(self) -> None:
        paths = self._feature_paths()
        codes: set[str] = set()
        days: set[pd.Timestamp] = set()
        log(f"  enumerate feature partitions n={len(paths)}")
        for path in paths:
            index = pd.read_parquet(path, columns=["stock_code", "trading_date"])
            codes.update(index["stock_code"].dropna().astype(str).unique())
            days.update(pd.to_datetime(index["trading_date"], errors="coerce").dropna().dt.normalize().unique())
        kept = sorted(codes)
        day_list = sorted(pd.Timestamp(d).normalize() for d in days)
        if not kept or not day_list:
            raise RuntimeError("empty feature partitions")
        T, N, F = len(day_list), len(kept), self.feat_dim
        log(f"  build dense cache days={T} bars={self.bars} stocks={N} features={F}")
        feats = np.memmap(self.feat_path, dtype=np.float32, mode="w+", shape=(T, self.bars, N, F))
        mask = np.memmap(self.mask_path, dtype=np.uint8, mode="w+", shape=(T, self.bars, N))
        open_d = np.full((T, N), np.nan, dtype=np.float32)
        close_d = np.full((T, N), np.nan, dtype=np.float32)
        target_count = np.zeros((T, N), dtype=np.int16)
        session_end = np.full(T, self.bars - 1, dtype=np.int16)
        code_to_i = {c: i for i, c in enumerate(kept)}
        day_to_i = {d: i for i, d in enumerate(day_list)}
        start_minutes = self.template[0].hour * 60 + self.template[0].minute

        required_cols = ["stock_code", "trading_date", "local_timestamp", "feature_valid_mask", *self.feature_names]
        for pi, path in enumerate(paths, start=1):
            log(f"    dense partition {pi}/{len(paths)} {path.parent.parent.name}-{path.parent.name}")
            frame = pd.read_parquet(path, columns=required_cols)
            frame["trading_date"] = pd.to_datetime(frame["trading_date"]).dt.normalize()
            local = pd.to_datetime(frame["local_timestamp"])
            slots = (local.dt.hour * 60 + local.dt.minute - start_minutes).to_numpy(dtype=np.int16)
            valid_slot = (slots >= 0) & (slots < self.bars)
            frame = frame.loc[valid_slot].reset_index(drop=True)
            slots = slots[valid_slot]
            di = frame["trading_date"].map(day_to_i).to_numpy(dtype=np.int32)
            ni = frame["stock_code"].map(code_to_i).to_numpy(dtype=np.int32)
            values = frame[self.feature_names].to_numpy(dtype=np.float32, copy=False)
            token_ok = frame["feature_valid_mask"].fillna(False).to_numpy(dtype=bool)
            for lo in range(0, len(frame), 100_000):
                hi = min(lo + 100_000, len(frame))
                block = np.nan_to_num(values[lo:hi], nan=0.0, posinf=0.0, neginf=0.0)
                feats[di[lo:hi], slots[lo:hi], ni[lo:hi], :] = block
                mask[di[lo:hi], slots[lo:hi], ni[lo:hi]] = token_ok[lo:hi].astype(np.uint8)

            count_table = frame.groupby(["trading_date", "stock_code"], observed=True).size()
            for (day, code), count in count_table.items():
                target_count[day_to_i[pd.Timestamp(day)], code_to_i[str(code)]] = min(int(count), np.iinfo(np.int16).max)
            slot_counts = frame.groupby(["trading_date", pd.Series(slots, index=frame.index, name="slot")], observed=True).size()
            for day, counts in slot_counts.groupby(level=0):
                c = counts.droplevel(0)
                threshold = max(20, int(c.max() * 0.5))
                eligible = c[c >= threshold]
                if len(eligible):
                    session_end[day_to_i[pd.Timestamp(day)]] = int(eligible.index.max())

            open_rows = frame.loc[slots == 0, ["trading_date", "stock_code", "open"]].drop_duplicates(["trading_date", "stock_code"], keep="first")
            for row in open_rows.itertuples(index=False):
                open_d[day_to_i[pd.Timestamp(row.trading_date)], code_to_i[str(row.stock_code)]] = float(row.open)
            for day, group in frame.assign(_slot=slots).groupby("trading_date", observed=True):
                d_idx = day_to_i[pd.Timestamp(day)]
                end_slot = int(session_end[d_idx])
                close_rows = group.loc[group["_slot"] == end_slot, ["stock_code", "close"]].drop_duplicates("stock_code", keep="last")
                for row in close_rows.itertuples(index=False):
                    close_d[d_idx, code_to_i[str(row.stock_code)]] = float(row.close)
            del frame, values
            feats.flush()
            mask.flush()

        np.save(self.open_path, open_d)
        np.save(self.close_path, close_d)
        np.save(self.target_count_path, target_count)
        np.save(self.session_end_path, session_end)
        source_meta = json.loads((self.meta_root / "source_file_manifest.json").read_text(encoding="utf-8"))
        meta: Dict[str, object] = {
            "data_version": self.data_version,
            "feature_schema_hash": self.feature_schema_hash,
            "feature_config_hash": self.feature_config_hash,
            "raw_zip_hash": source_meta.get("zip", {}).get("zip_sha256", ""),
            "stock_codes": kept,
            "days": [str(d.date()) for d in day_list],
            "bars": self.bars,
            "feature_count": F,
            "feature_names": self.feature_names,
            "feature_meta": {
                "rolling": "trailing_only",
                "universe_timing": "decision universe uses D-1 and earlier only; target availability is a separate post-close evaluation flag",
                "timezone": self.cfg["calendar"]["timezone"],
                "minute_template": [f"{t.hour:02d}:{t.minute:02d}" for t in self.template],
            },
            "filter_reasons": {
                "universe_type": UNIVERSE_TYPE,
                "history_min_coverage": self.cfg["universe"]["history_min_coverage"],
                "min_history_minutes": self.cfg["universe"]["min_history_minutes"],
                "min_target_minutes": self.cfg["universe"]["min_target_minutes"],
            },
        }
        write_json(self.meta_path, meta)
        del feats, mask
        self._attach(meta)
        self._write_universe_manifest()

    def _write_universe_manifest(self) -> None:
        dest = self.meta_root / "universe_manifest.parquet"
        rows: List[pd.DataFrame] = []
        for day in self.days:
            i = self.day_index(day)
            history_count = self._history_count(i)
            required = self._required_history(i)
            included = self.decision_mask(day)
            previous_count = np.asarray(self.mask[i - 1]).sum(axis=0, dtype=np.int32) if i > 0 else np.zeros(self.N, dtype=np.int32)
            previous_required = self._previous_required(i)
            target_required = min(int(self.cfg["universe"]["min_target_minutes"]), max(1, int((int(self.session_end_slot[i]) + 1) * 0.7)))
            reasons = [
                decision_exclusion_reason(
                    self.close[i - 1, j] if i > 0 else np.nan,
                    previous_count[j],
                    previous_required,
                    history_count[j],
                    required,
                )
                for j in range(self.N)
            ]
            evaluation_reasons = [
                evaluation_exclusion_reason(self.open[i, j], self.close[i, j], self.target_count[i, j], target_required)
                for j in range(self.N)
            ]
            evaluation_eligible = included & np.asarray([reason == "" for reason in evaluation_reasons], dtype=bool)
            rows.append(
                universe_manifest_rows(
                    day,
                    self.kept,
                    included,
                    reasons,
                    history_count,
                    previous_count,
                    self.target_count[i],
                    evaluation_eligible,
                    evaluation_reasons,
                    self.clocks(day)["data_cutoff_timestamp"],
                )
            )
        pd.concat(rows, ignore_index=True).to_parquet(dest, index=False, compression="zstd")
        write_json(
            self.meta_root / "universe_manifest.json",
            {
                "universe_type": UNIVERSE_TYPE,
                "logic": "decision included_flag uses only D-1 and earlier history; post-close target label/execution availability is stored separately as evaluation_eligible_flag and never filters attention or Top-K",
                "historical_membership_available": False,
                "survivorship_bias_disclosure": self.cfg["universe"]["disclosure"],
                "rows_path": str(dest),
            },
        )

    def day_index(self, day: pd.Timestamp) -> int:
        value = pd.Timestamp(day).normalize()
        try:
            return self.days.index(value)
        except ValueError as exc:
            raise KeyError(value) from exc

    def _history_count(self, i: int) -> np.ndarray:
        if i < self.lookback_days:
            return np.zeros(self.N, dtype=np.int32)
        return np.asarray(self.mask[i - self.lookback_days : i]).sum(axis=(0, 1), dtype=np.int32)

    def _required_history(self, i: int) -> int:
        if i < self.lookback_days:
            return int(self.cfg["universe"]["min_history_minutes"])
        available = int(sum(int(x) + 1 for x in self.session_end_slot[i - self.lookback_days : i]))
        return max(int(self.cfg["universe"]["min_history_minutes"]), int(available * float(self.cfg["universe"]["history_min_coverage"])))

    def _previous_required(self, i: int) -> int:
        if i <= 0:
            return int(self.cfg["universe"]["min_target_minutes"])
        previous_bars = int(self.session_end_slot[i - 1]) + 1
        return min(int(self.cfg["universe"]["min_target_minutes"]), max(1, int(previous_bars * float(self.cfg["universe"]["history_min_coverage"]))))

    def decision_mask(self, day: pd.Timestamp) -> np.ndarray:
        """Point-in-time candidate set using no target-day field."""
        i = self.day_index(day)
        if i < self.lookback_days:
            return np.zeros(self.N, dtype=bool)
        history_count = self._history_count(i)
        previous_count = np.asarray(self.mask[i - 1]).sum(axis=0, dtype=np.int32)
        return (
            np.isfinite(self.close[i - 1])
            & (self.close[i - 1] > 0)
            & (previous_count >= self._previous_required(i))
            & (history_count >= self._required_history(i))
        )

    def evaluation_mask(self, day: pd.Timestamp) -> np.ndarray:
        """Post-close observability, for audit only and never portfolio selection."""
        i = self.day_index(day)
        target_required = min(int(self.cfg["universe"]["min_target_minutes"]), max(1, int((int(self.session_end_slot[i]) + 1) * 0.7)))
        return (
            self.decision_mask(day)
            & np.isfinite(self.open[i])
            & np.isfinite(self.close[i])
            & (self.open[i] > 0)
            & (self.close[i] > 0)
            & (self.target_count[i] >= target_required)
        )

    def valid_mask(self, day: pd.Timestamp) -> np.ndarray:
        return self.decision_mask(day)

    def window(self, day: pd.Timestamp, stock_indices: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Return X_D from D-10 through D-1; no target-day row is accessed."""
        i = self.day_index(day)
        if i < self.lookback_days:
            raise KeyError("insufficient lookback")
        idx = np.arange(self.N) if stock_indices is None else np.asarray(stock_indices, dtype=int)
        sl = np.asarray(self.feats[i - self.lookback_days : i, :, idx, :], dtype=np.float32)
        x = np.ascontiguousarray(np.transpose(sl, (2, 0, 1, 3)).reshape(len(idx), -1, self.feat_dim))
        mk = np.asarray(self.mask[i - self.lookback_days : i, :, idx], dtype=np.float32)
        m = np.ascontiguousarray(np.transpose(mk, (2, 0, 1)).reshape(len(idx), -1))
        return x, m

    def labels(self, day: pd.Timestamp) -> Tuple[np.ndarray, np.ndarray]:
        i = self.day_index(day)
        y = np.full(self.N, np.nan, dtype=np.float32)
        good = np.isfinite(self.open[i]) & np.isfinite(self.close[i]) & (self.open[i] > 0) & (self.close[i] > 0)
        y[good] = np.log(self.close[i, good].astype(np.float64) / self.open[i, good].astype(np.float64)).astype(np.float32)
        previous = np.full(self.N, np.nan, dtype=np.float32)
        if i > 0:
            good_prev = np.isfinite(self.close[i - 1]) & np.isfinite(self.close[i]) & (self.close[i - 1] > 0) & (self.close[i] > 0)
            previous[good_prev] = np.log(self.close[i, good_prev].astype(np.float64) / self.close[i - 1, good_prev].astype(np.float64)).astype(np.float32)
        return y, previous

    def daily_returns_to(self, day: pd.Timestamp) -> pd.DataFrame:
        i = self.day_index(day)
        o, c = self.open[:i].astype(np.float64), self.close[:i].astype(np.float64)
        good = np.isfinite(o) & np.isfinite(c) & (o > 0) & (c > 0)
        r = np.full_like(o, np.nan)
        r[good] = np.log(c[good] / o[good])
        return pd.DataFrame(r, index=self.days[:i], columns=self.kept)

    def prediction_meta(self, day: pd.Timestamp) -> Dict[str, object]:
        return {
            "prediction_date": str(pd.Timestamp(day).date()),
            "data_cutoff_timestamp": self.clocks(day)["data_cutoff_timestamp"],
            "join_key": "stock_code",
            "universe_type": self.universe_type,
        }

    def clocks(self, day: pd.Timestamp) -> Dict[str, str]:
        i = self.day_index(day)
        cal = self.cfg["calendar"]
        tz = cal["timezone"]
        previous = self.days[i - 1] if i > 0 else day
        prev_slot = int(self.session_end_slot[max(i - 1, 0)])
        prev_clock = self.template[prev_slot]
        cutoff = pd.Timestamp(f"{pd.Timestamp(previous).date()} {prev_clock.hour:02d}:{prev_clock.minute:02d}:59", tz=tz)
        return {
            "decision_timestamp": str(stamp(day, cal["decision_clock"], tz)),
            "execution_timestamp": str(stamp(day, cal["execution_clock"], tz)),
            "realization_timestamp": str(stamp(day, cal["realization_clock"], tz)),
            "data_cutoff_timestamp": str(cutoff),
        }
