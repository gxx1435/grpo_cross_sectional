"""Cached CSI500 panel: 240-bar days, auction A_D, executable Open/Close labels."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from data.auction_features import auction_matrix
from data.loaders import AUCTION_FEATURE_NAMES, load_all_auctions, load_stock_minutes
from data.minute_features import FEATURE_NAMES, compute_minute_features, feature_dim
from data.trading_calendar import align_day_to_template, session_times, stamp, trading_days_from_index
from data.universe import universe_codes
from experiments.progress import Progress
from utils.config import resolve_path
from utils.io import schema_hash
from utils.logging import log, write_json


class ResearchStore:
    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.cache_dir = resolve_path(cfg, cfg["paths"]["cache_dir"])
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cons = resolve_path(cfg, cfg["paths"]["constituents_csv"])
        self.codes = universe_codes(cons, int(cfg["universe"]["n_stocks_cap"]))
        self.pools = [resolve_path(cfg, p) for p in cfg["paths"]["minute_pools"]]
        self.auction_dirs = [resolve_path(cfg, p) for p in cfg["paths"]["auction_dirs"]]
        self.panel_lo = str(cfg["panel"]["start"])
        self.panel_hi = str(cfg["panel"]["end"])
        self.template = session_times(cfg)
        self.bars = int(cfg["calendar"]["bars_per_day"])
        self.lookback_days = int(cfg["calendar"]["lookback_days"])
        self.feature_names = list(FEATURE_NAMES)
        self.feat_dim = feature_dim()
        self.auction_names = list(AUCTION_FEATURE_NAMES)
        tag = f"ssfm_n{len(self.codes)}_f{self.feat_dim}_b{self.bars}"
        self.meta_path = self.cache_dir / f"{tag}_meta.json"
        self.feat_path = self.cache_dir / f"{tag}_feat.fp16.dat"
        self.mask_path = self.cache_dir / f"{tag}_mask.u8.dat"
        self.open_path = self.cache_dir / f"{tag}_open.fp32.npy"
        self.close_path = self.cache_dir / f"{tag}_close.fp32.npy"
        self.auc_path = self.cache_dir / f"{tag}_auction.fp32.npy"
        self.auc_ok_path = self.cache_dir / f"{tag}_auction_ok.u8.npy"
        self._load_or_build(tag)

    def _load_or_build(self, tag: str) -> None:
        needed = [self.meta_path, self.feat_path, self.mask_path, self.open_path, self.close_path, self.auc_path]
        if all(p.is_file() for p in needed):
            meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
            self.kept = list(meta["kept"])
            self.days = [pd.Timestamp(x) for x in meta["days"]]
            self.T = int(meta["T"])
            self.N = int(meta["N"])
            self.feat_dim = int(meta["feat_dim"])
            self.feature_names = list(meta["feature_names"])
            self.feats = np.memmap(self.feat_path, dtype=np.float16, mode="r", shape=(self.T, self.bars, self.N, self.feat_dim))
            self.mask = np.memmap(self.mask_path, dtype=np.uint8, mode="r", shape=(self.T, self.bars, self.N))
            self.open = np.load(self.open_path)
            self.close = np.load(self.close_path)
            self.auction = np.load(self.auc_path)
            self.auction_ok = np.load(self.auc_ok_path)
            self.filter_reasons = meta.get("filter_reasons", {})
            self.feature_meta = meta.get("feature_meta", {})
            self.data_version = meta.get("data_version", tag)
            log(f"  store cache {self.feats.shape} F={self.feat_dim} days={len(self.days)}")
            return
        self._build(tag)

    def _build(self, tag: str) -> None:
        log(f"  building ResearchStore F={self.feat_dim} n={len(self.codes)} …")
        frames = {c: [] for c in ("open", "high", "low", "close", "volume", "amount")}
        kept: List[str] = []
        load_bar = Progress(len(self.codes), "STORE 读分钟CSV", log_every=50)
        for i, code in enumerate(self.codes):
            df = load_stock_minutes(code, self.pools, self.panel_lo, self.panel_hi)
            if df.empty or len(df) < 800:
                load_bar.update(i + 1, f"skip {code}")
                continue
            for c in frames:
                if c not in df.columns:
                    df[c] = np.nan
                frames[c].append(df[c].rename(code))
            kept.append(code)
            load_bar.update(i + 1, f"kept={len(kept)} {code}")
        load_bar.close(f"kept={len(kept)}")
        if not kept:
            raise RuntimeError("no minute panels loaded")
        panels = {k: pd.concat(v, axis=1).sort_index() for k, v in frames.items()}
        idx = panels["close"].dropna(how="all").index
        for k in panels:
            panels[k] = panels[k].reindex(idx)
        drop = set(self.cfg["calendar"]["drop_minute_times"])
        keep_mask = np.array(
            [f"{pd.Timestamp(t).hour:02d}:{pd.Timestamp(t).minute:02d}" not in drop for t in idx],
            dtype=bool,
        )
        raw_ts = pd.DatetimeIndex(idx[keep_mask])
        o = panels["open"].loc[raw_ts, kept].to_numpy(np.float32)
        h = panels["high"].loc[raw_ts, kept].to_numpy(np.float32)
        l = panels["low"].loc[raw_ts, kept].to_numpy(np.float32)
        c = panels["close"].loc[raw_ts, kept].to_numpy(np.float32)
        v = panels["volume"].loc[raw_ts, kept].to_numpy(np.float32)
        a = panels["amount"].loc[raw_ts, kept].to_numpy(np.float32)
        days = trading_days_from_index(raw_ts)
        norms = pd.DatetimeIndex(raw_ts).normalize()
        T, n = len(days), len(kept)
        feats = np.memmap(self.feat_path, dtype=np.float16, mode="w+", shape=(T, self.bars, n, self.feat_dim))
        mask = np.memmap(self.mask_path, dtype=np.uint8, mode="w+", shape=(T, self.bars, n))
        open_d = np.zeros((T, n), dtype=np.float32)
        close_d = np.zeros((T, n), dtype=np.float32)
        full_open = panels["open"][kept]
        full_close = panels["close"][kept]
        raw_feat_path = self.cache_dir / f"{tag}_rawfeat.fp16.dat"
        raw_feats = np.memmap(raw_feat_path, dtype=np.float16, mode="w+", shape=(len(raw_ts), n, self.feat_dim))
        feat_meta = {"feature_names": self.feature_names, "feat_dim": self.feat_dim, "rolling": "trailing_only"}
        chunk = 4000
        warmup = 2500
        n_chunks = (len(raw_ts) + chunk - 1) // chunk
        feat_bar = Progress(n_chunks, "STORE 计算46维特征", log_every=1)
        for ci, t0 in enumerate(range(0, len(raw_ts), chunk)):
            t1 = min(len(raw_ts), t0 + chunk)
            left = max(0, t0 - warmup)
            full, feat_meta = compute_minute_features(
                o[left:t1], h[left:t1], l[left:t1], c[left:t1], v[left:t1], a[left:t1], raw_ts[left:t1],
            )
            raw_feats[t0:t1] = full[t0 - left :].astype(np.float16)
            del full
            feat_bar.update(ci + 1, f"bars {t1}/{len(raw_ts)}")
        feat_bar.close()
        raw_feats.flush()
        align_bar = Progress(T, "STORE 对齐240分钟模板", log_every=40)
        for di, day in enumerate(days):
            sl = np.where(norms == day)[0]
            if len(sl):
                x = np.asarray(raw_feats[sl[0] : sl[-1] + 1], dtype=np.float32)
                aligned, m = align_day_to_template(raw_ts[sl[0] : sl[-1] + 1], x, self.template)
                feats[di] = aligned.astype(np.float16)
                mask[di] = np.repeat(m[:, None], n, axis=1).astype(np.uint8)
            day_full = full_close.index.normalize() == day
            if day_full.any():
                open_d[di] = full_open.loc[day_full].iloc[0].to_numpy(np.float32)
                close_d[di] = full_close.loc[day_full].iloc[-1].to_numpy(np.float32)
            align_bar.update(di + 1, str(day.date()))
        align_bar.close()
        del raw_feats
        try:
            raw_feat_path.unlink()
        except OSError:
            pass
        feats.flush()
        mask.flush()
        np.save(self.open_path, open_d)
        np.save(self.close_path, close_d)

        log("    loading auction tables …")
        auc_df = load_all_auctions(self.auction_dirs)
        auction = np.zeros((T, n, 9), dtype=np.float32)
        auction_ok = np.zeros((T, n), dtype=np.uint8)
        auc_bar = Progress(T, "STORE 对齐集合竞价A_D", log_every=40)
        for di, day in enumerate(days):
            a_d, ok, _ = auction_matrix(auc_df, kept, day)
            auction[di] = a_d
            auction_ok[di] = ok.astype(np.uint8)
            auc_bar.update(di + 1, f"valid={int(ok.sum())}")
        auc_bar.close()
        np.save(self.auc_path, auction)
        np.save(self.auc_ok_path, auction_ok)

        self.kept = kept
        self.days = days
        self.T, self.N = T, n
        self.feats = np.memmap(self.feat_path, dtype=np.float16, mode="r", shape=(T, self.bars, n, self.feat_dim))
        self.mask = np.memmap(self.mask_path, dtype=np.uint8, mode="r", shape=(T, self.bars, n))
        self.open = open_d
        self.close = close_d
        self.auction = auction
        self.auction_ok = auction_ok
        self.feature_meta = feat_meta
        self.filter_reasons = {"min_bars": 800, "join": "stock_code"}
        self.data_version = tag
        meta = {
            "kept": kept,
            "days": [str(d.date()) for d in days],
            "T": T,
            "N": n,
            "feat_dim": self.feat_dim,
            "feature_names": self.feature_names,
            "feature_schema_hash": schema_hash(self.feature_names),
            "auction_names": self.auction_names,
            "panel_lo": self.panel_lo,
            "panel_hi": self.panel_hi,
            "bars": self.bars,
            "feature_meta": {k: v for k, v in feat_meta.items() if k != "missing_rate"},
            "missing_rate": feat_meta.get("missing_rate", {}),
            "filter_reasons": self.filter_reasons,
            "data_version": tag,
            "n_auction_rows": int(len(auc_df)),
        }
        write_json(self.meta_path, meta)
        log(f"  saved store {self.feat_path} days={T} N={n} F={self.feat_dim}")

    def day_index(self, day: pd.Timestamp) -> int:
        day = pd.Timestamp(day).normalize()
        for i, d in enumerate(self.days):
            if d == day:
                return i
        raise KeyError(day)

    def valid_mask(self, day: pd.Timestamp) -> np.ndarray:
        i = self.day_index(day)
        di = self.days.index(pd.Timestamp(day).normalize())
        if di < self.lookback_days:
            return np.zeros(self.N, dtype=bool)
        o = self.open[i]
        c = self.close[i]
        auc_ok = self.auction_ok[i].astype(bool)
        hist = self.mask[di - self.lookback_days : di]
        hist_ok = hist.reshape(self.lookback_days, self.bars, self.N).mean(axis=(0, 1)) >= 0.7
        m = (
            np.isfinite(o)
            & np.isfinite(c)
            & (o > 0)
            & (c > 0)
            & auc_ok
            & hist_ok
        )
        return m

    def window(self, day: pd.Timestamp) -> Tuple[np.ndarray, np.ndarray]:
        """X_D [N, 2400, F] and minute mask from D-10:D-1. No D-day minutes."""
        i = self.day_index(day)
        if i < self.lookback_days:
            raise KeyError("insufficient lookback")
        sl = np.asarray(self.feats[i - self.lookback_days : i], dtype=np.float32)  # [10, 240, N, F]
        x = np.ascontiguousarray(np.transpose(sl, (2, 0, 1, 3)).reshape(self.N, self.lookback_days * self.bars, self.feat_dim))
        del sl
        np.nan_to_num(x, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        np.clip(x, -30.0, 30.0, out=x)
        mk = np.asarray(self.mask[i - self.lookback_days : i], dtype=np.float32)
        m = np.ascontiguousarray(np.transpose(mk, (2, 0, 1)).reshape(self.N, -1))
        del mk
        return x, m

    def auction_d(self, day: pd.Timestamp) -> Tuple[np.ndarray, Dict[str, object]]:
        i = self.day_index(day)
        meta = {
            "auction_date": str(pd.Timestamp(day).date()),
            "prediction_date": str(pd.Timestamp(day).date()),
            "available_ts": str(stamp(day, self.cfg["calendar"]["auction_available_clock"])),
            "join_key": "(stock_code, trading_date)",
        }
        return np.asarray(self.auction[i], dtype=np.float32), meta

    def labels(self, day: pd.Timestamp) -> Tuple[np.ndarray, np.ndarray]:
        """Primary y = log(Close_D/Open_D). Aux close-to-close is not a training target."""
        i = self.day_index(day)
        o = np.clip(self.open[i].astype(np.float64), 1e-8, None)
        c = np.clip(self.close[i].astype(np.float64), 1e-8, None)
        y = np.log(c / o)
        y = np.where(np.isfinite(y), y, np.nan).astype(np.float32)
        prev = np.clip(self.close[i - 1].astype(np.float64), 1e-8, None) if i > 0 else o
        y_c2c = np.log(c / prev).astype(np.float32)
        return y, y_c2c

    def daily_returns_to(self, day: pd.Timestamp) -> pd.DataFrame:
        """Intraday log returns up to D-1 (inclusive of last completed day)."""
        i = self.day_index(day)
        o = np.clip(self.open[:i], 1e-8, None)
        c = np.clip(self.close[:i], 1e-8, None)
        r = np.log(c / o)
        return pd.DataFrame(r, index=self.days[:i], columns=self.kept)

    def clocks(self, day: pd.Timestamp) -> Dict[str, str]:
        cal = self.cfg["calendar"]
        return {
            "decision_timestamp": str(stamp(day, cal["auction_available_clock"])),
            "execution_timestamp": str(stamp(day, cal["execution_clock"])),
            "realization_timestamp": str(stamp(day, cal["realization_clock"])),
            "data_cutoff_timestamp": str(stamp(day, cal["auction_available_clock"])),
        }
