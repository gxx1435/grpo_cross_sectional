"""
HF data pipeline for the cross-sectional StockTemporalTransformer.

Pipeline overview
-----------------
1. Read every CSV under ``pool_dir`` (e.g. ``files/csi500/2025/``) and align
   their 1-minute bars onto a common timestamp index.
2. Drop the 09:30 and 15:00 auction bars per HF_EXPERIMENT_DESIGN.md §7.2.
3. Compute F=11 causal features per stock per bar (returns at multiple
   horizons, vwap-gap, range, realized vol, volume / turnover z, time-of-day).
4. Cross-sectionally z-score every feature.
5. Build SFT labels with a small linear factor model:
        score = mean of cross-sectional z-scores of
                  -ret_5  (short reversal)
                  -ret_30 (long reversal)
                  -vol_z  (volume anomaly, large vol -> low score)
                  -rv_30  (low realized vol preferred)
   then rank-normalized into [0, 1]. No future info is used.
6. Slide a window of length L over time to produce
        x_t : (N, L, F)
        y_t : (N,)             SFT label
        r_t : (N,)             realized log-return over the next H bars
                              (used by GRPO as ground-truth return).
7. Cache the result to ``pool_dir/.hf_cache/{tag}.npz`` for fast re-runs.

Public surface
--------------
    HFConfig                 : dataclass of all knobs.
    build_hf_dataset(cfg)    : returns a dict with x / y / r / timestamps.
    HFSFTDataset             : torch Dataset returning (x, y).
    HFEpisodeDataset         : torch Dataset returning M consecutive (x, r) -- for GRPO.
    split_train_val_test     : helper that purges H bars at boundaries.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


HERE = Path(__file__).resolve().parent
DEFAULT_POOL_DIR = HERE / "csi500" / "2025"

RAW_COLS_RENAME = {
    "日期": "ts",
    "开盘": "open",
    "最高": "high",
    "最低": "low",
    "收盘": "close",
    "成交量(股)": "volume",
    "成交额(元)": "amount",
    "涨跌幅(%)": "pct_change",
    "换手率(%)": "turnover",
}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class HFConfig:
    pool_dir: Path = DEFAULT_POOL_DIR
    extra_pool_dirs: Optional[List[Path]] = None  # e.g. [2026_1min] merged with pool_dir per stock
    start_day: int = 0
    num_days: int = 3
    lookback: int = 30
    horizon: int = 5
    n_stocks_max: Optional[int] = None
    stock_ids: Optional[List[str]] = None   # explicit universe (overrides n_stocks_max order)
    drop_open_bar: bool = True
    drop_close_bar: bool = True
    cache_dir: Path = HERE / "csi500" / ".hf_cache"
    cache_tag: Optional[str] = None
    val_ratio: float = 0.34
    test_ratio: float = 0.34
    feature_names: List[str] = field(default_factory=lambda: [
        "ret_1", "ret_5", "ret_30", "vwap_gap", "range_norm",
        "rv_30", "vol_z", "turn_z", "tod_sin", "tod_cos", "bars_since_open",
    ])
    use_regime_features: bool = False
    regime_lookback_days: int = 30          # prior trading days for regime at each bar
    bars_per_day: int = 240

    @property
    def feat_dim(self) -> int:
        n = len(self.feature_names)
        if self.use_regime_features:
            n += len(REGIME_FEATURE_NAMES)
        return n


REGIME_FEATURE_NAMES = [
    "regime_bull",
    "regime_bear",
    "regime_high_vol",
    "regime_low_vol",
    "regime_mkt_ret_1m",
    "regime_mkt_vol_1m",
]


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------
def _read_one_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["日期"])
    df = df.rename(columns=RAW_COLS_RENAME).set_index("ts").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    return df[["open", "high", "low", "close", "volume", "amount", "turnover"]]


def _pool_dirs(cfg_or_dir, extra: Optional[List[Path]] = None) -> List[Path]:
    if isinstance(cfg_or_dir, HFConfig):
        dirs = [cfg_or_dir.pool_dir]
        if cfg_or_dir.extra_pool_dirs:
            dirs.extend(cfg_or_dir.extra_pool_dirs)
        return dirs
    dirs = [Path(cfg_or_dir)]
    if extra:
        dirs.extend(extra)
    return dirs


def _read_stock_merged(stock_id: str, pool_dirs: List[Path]) -> pd.DataFrame:
    """Load and concatenate one symbol across multiple pool directories."""
    frames: List[pd.DataFrame] = []
    for p in pool_dirs:
        path = p / f"{stock_id}.csv"
        if path.is_file():
            frames.append(_read_one_csv(path))
    if not frames:
        raise FileNotFoundError(f"No CSV for {stock_id} under {pool_dirs}")
    df = pd.concat(frames).sort_index()
    return df[~df.index.duplicated(keep="first")]


def _master_trading_days(pool_dirs: List[Path], stock_id: Optional[str] = None) -> pd.DatetimeIndex:
    if stock_id is not None:
        base_df = _read_stock_merged(stock_id, pool_dirs)
    else:
        csv_files = sorted(pool_dirs[0].glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files under {pool_dirs[0]}")
        base_df = _read_stock_merged(csv_files[0].stem, pool_dirs)
    return pd.DatetimeIndex(sorted(set(base_df.index.normalize())))


def _select_date_range(dates: pd.DatetimeIndex, start: int, k: int) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(dates[start: start + k])


def trading_day_index(
    pool_dir: Path,
    date: str,
    extra_pool_dirs: Optional[List[Path]] = None,
) -> int:
    """Map a calendar date string to its 0-based index on the unified pool calendar."""
    pool_dirs = _pool_dirs(pool_dir, extra_pool_dirs)
    dates = _master_trading_days(pool_dirs)
    target = pd.Timestamp(date).normalize()
    if target not in dates:
        prior = dates[dates <= target]
        if len(prior) == 0:
            raise ValueError(f"No trading day on or before {date} in {pool_dirs}")
        target = prior[-1]
    return int(dates.get_loc(target))


def list_trading_days(
    pool_dir: Path,
    start_date: str,
    end_date: str,
    extra_pool_dirs: Optional[List[Path]] = None,
) -> pd.DatetimeIndex:
    """Return trading days in ``[start_date, end_date]`` inclusive."""
    dates = _master_trading_days(_pool_dirs(pool_dir, extra_pool_dirs))
    s, e = pd.Timestamp(start_date).normalize(), pd.Timestamp(end_date).normalize()
    return pd.DatetimeIndex(dates[(dates >= s) & (dates <= e)])


def _intraday_filter(ts: pd.DatetimeIndex, drop_open: bool, drop_close: bool) -> np.ndarray:
    keep = np.ones(len(ts), dtype=bool)
    if drop_open:
        keep &= ~((ts.hour == 9) & (ts.minute == 30))
    if drop_close:
        keep &= ~((ts.hour == 15) & (ts.minute == 0))
    return keep


def _build_panel(cfg: HFConfig) -> Tuple[np.ndarray, pd.DatetimeIndex, List[str]]:
    pool_dirs = _pool_dirs(cfg)
    if cfg.stock_ids:
        stock_ids_req = list(cfg.stock_ids)
    else:
        csv_files = sorted(pool_dirs[0].glob("*.csv"))
        if cfg.n_stocks_max is not None:
            csv_files = csv_files[: cfg.n_stocks_max]
        stock_ids_req = [f.stem for f in csv_files]
    if not stock_ids_req:
        raise FileNotFoundError(f"No stocks resolved under {pool_dirs}")

    master_dates = _master_trading_days(pool_dirs)
    target_dates = _select_date_range(master_dates, cfg.start_day, cfg.num_days)
    if len(target_dates) == 0:
        raise ValueError(
            f"No dates available with start_day={cfg.start_day}, num_days={cfg.num_days} "
            f"in {pool_dirs}"
        )
    base_df = _read_stock_merged(stock_ids_req[0], pool_dirs)
    base_df = base_df[base_df.index.normalize().isin(target_dates)]
    keep = _intraday_filter(base_df.index, cfg.drop_open_bar, cfg.drop_close_bar)
    master_ts = base_df.index[keep]

    cols = ["open", "high", "low", "close", "volume", "amount", "turnover"]
    panels: List[np.ndarray] = []
    stock_ids: List[str] = []

    for sid in stock_ids_req:
        df = _read_stock_merged(sid, pool_dirs)
        df = df.reindex(master_ts)
        panels.append(df[cols].to_numpy(dtype=np.float64))
        stock_ids.append(sid)

    panel = np.stack(panels, axis=1)  # (T, N, C)
    return panel, master_ts, stock_ids


# ---------------------------------------------------------------------------
# Feature engineering (causal, no future info)
# ---------------------------------------------------------------------------
def _shift_along_time(a: np.ndarray, k: int) -> np.ndarray:
    out = np.full_like(a, np.nan, dtype=np.float64)
    if k > 0:
        out[k:] = a[:-k]
    return out


def _rolling_mean(a: np.ndarray, w: int) -> np.ndarray:
    out = np.full_like(a, np.nan, dtype=np.float64)
    csum = np.nancumsum(np.nan_to_num(a, nan=0.0), axis=0)
    cnt = np.cumsum((~np.isnan(a)).astype(np.float64), axis=0)
    if w < 1:
        return out
    csum_lag = np.concatenate([np.zeros((w, *a.shape[1:])), csum[:-w]], axis=0) if w <= len(a) else np.zeros_like(csum)
    cnt_lag = np.concatenate([np.zeros((w, *a.shape[1:])), cnt[:-w]], axis=0) if w <= len(a) else np.zeros_like(cnt)
    win_sum = csum - csum_lag
    win_cnt = cnt - cnt_lag
    out[w - 1:] = win_sum[w - 1:] / np.where(win_cnt[w - 1:] > 0, win_cnt[w - 1:], np.nan)
    return out


def _rolling_std(a: np.ndarray, w: int) -> np.ndarray:
    mu = _rolling_mean(a, w)
    sq = _rolling_mean(a ** 2, w)
    var = sq - mu ** 2
    var = np.where(var < 0, 0, var)
    return np.sqrt(var)


def _time_of_day_features(ts: pd.DatetimeIndex) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    minutes = ts.hour * 60 + ts.minute
    morning_start = 9 * 60 + 30  # 09:30
    morning_end = 11 * 60 + 30   # 11:30
    afternoon_start = 13 * 60     # 13:00
    afternoon_end = 15 * 60       # 15:00
    bars_since_open = np.where(
        minutes < afternoon_start,
        minutes - morning_start,
        minutes - afternoon_start + (morning_end - morning_start),
    ).astype(np.float64)
    total_minutes = (morning_end - morning_start) + (afternoon_end - afternoon_start)
    norm = bars_since_open / max(total_minutes, 1)
    tod_sin = np.sin(2 * math.pi * norm)
    tod_cos = np.cos(2 * math.pi * norm)
    return tod_sin, tod_cos, bars_since_open


def _compute_exposure_features(panel: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Causal exposure proxies per bar per stock.

    size_z : cross-sectional z-score of log(amount)
    beta_z : cross-sectional z-score of rolling 30-bar beta vs equal-weight market
    """
    amount = panel[..., 5]
    close = panel[..., 3]
    log_close = np.log(np.where(close > 0, close, np.nan))
    ret_1 = log_close - _shift_along_time(log_close, 1)

    log_amt = np.log(np.where(amount > 0, amount, np.nan))
    size_z = _cross_section_zscore(log_amt)

    # market return: equal-weight cross section
    mkt = np.nanmean(ret_1, axis=1, keepdims=True)  # (T, 1)
    mkt = np.broadcast_to(mkt, ret_1.shape).copy()
    # rolling cov/var for beta (30 bars, causal)
    w = 30
    beta = np.zeros_like(ret_1)
    for t in range(w, len(ret_1)):
        rs = ret_1[t - w: t]          # (w, N)
        ms = mkt[t - w: t]            # (w, N)
        ms1 = ms.mean(axis=1, keepdims=True)
        rs_c = rs - rs.mean(axis=0, keepdims=True)
        ms_c = ms - ms1
        cov = (rs_c * ms_c).mean(axis=0)
        var = (ms_c ** 2).mean(axis=0) + 1e-8
        beta[t] = cov / var
    beta_z = _cross_section_zscore(beta)
    return (
        np.nan_to_num(size_z, nan=0.0).astype(np.float32),
        np.nan_to_num(beta_z, nan=0.0).astype(np.float32),
    )


def _cross_section_zscore(x: np.ndarray) -> np.ndarray:
    """Standardize across the N axis (axis=1). x shape (T, N)."""
    mu = np.nanmean(x, axis=1, keepdims=True)
    sigma = np.nanstd(x, axis=1, keepdims=True) + 1e-8
    z = (x - mu) / sigma
    return np.nan_to_num(z, nan=0.0)


def _compute_features(
    panel: np.ndarray, ts: pd.DatetimeIndex, cfg: HFConfig
) -> np.ndarray:
    open_, high, low, close, volume, amount, turnover = [panel[..., i] for i in range(7)]

    log_close = np.log(np.where(close > 0, close, np.nan))
    ret_1 = log_close - _shift_along_time(log_close, 1)
    ret_5 = log_close - _shift_along_time(log_close, 5)
    ret_30 = log_close - _shift_along_time(log_close, 30)

    rolling_amt = _rolling_mean(amount, 5)
    rolling_vol = _rolling_mean(volume, 5)
    vwap_5 = rolling_amt / np.where(rolling_vol > 0, rolling_vol, np.nan)
    vwap_gap = np.log(np.where(close > 0, close, np.nan) / np.where(vwap_5 > 0, vwap_5, np.nan))

    range_norm = (high - low) / (np.where(close > 0, _shift_along_time(close, 1), np.nan) + 1e-8)
    rv_30 = _rolling_mean(ret_1 ** 2, 30)

    vol_mu = _rolling_mean(volume, 30)
    vol_sigma = _rolling_std(volume, 30) + 1e-8
    vol_z = (volume - vol_mu) / vol_sigma

    turn_mu = _rolling_mean(turnover, 30)
    turn_sigma = _rolling_std(turnover, 30) + 1e-8
    turn_z = (turnover - turn_mu) / turn_sigma

    tod_sin, tod_cos, bars_since_open = _time_of_day_features(ts)
    bcast = np.broadcast_to
    n_stocks = panel.shape[1]
    tod_sin = bcast(tod_sin[:, None], (len(ts), n_stocks)).copy()
    tod_cos = bcast(tod_cos[:, None], (len(ts), n_stocks)).copy()
    bars_since_open = bcast(bars_since_open[:, None], (len(ts), n_stocks)).copy()
    bars_since_open = bars_since_open / 240.0

    raw = {
        "ret_1": ret_1,
        "ret_5": ret_5,
        "ret_30": ret_30,
        "vwap_gap": vwap_gap,
        "range_norm": range_norm,
        "rv_30": rv_30,
        "vol_z": vol_z,
        "turn_z": turn_z,
    }
    cs_z = {k: _cross_section_zscore(v) for k, v in raw.items()}
    cs_z["tod_sin"] = tod_sin
    cs_z["tod_cos"] = tod_cos
    cs_z["bars_since_open"] = bars_since_open

    feats = np.stack([cs_z[name] for name in cfg.feature_names], axis=-1)  # (T, N, F)
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return feats


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------
def _linear_factor_label(feats: np.ndarray, cfg: HFConfig) -> np.ndarray:
    """
    Build SFT labels by averaging the cross-sectional z-scores of a few
    classic linear factors (signs picked so "high score == long candidate"):

        +(-ret_5)   reversal_5
        +(-ret_30)  reversal_30
        +(-vol_z)   low-volume preference
        +(-rv_30)   low-vol preference

    Returns rank-normalized scores in [0, 1] of shape (T, N).
    """
    name_to_idx = {n: i for i, n in enumerate(cfg.feature_names)}
    f5 = -feats[..., name_to_idx["ret_5"]]
    f30 = -feats[..., name_to_idx["ret_30"]]
    fv = -feats[..., name_to_idx["vol_z"]]
    fr = -feats[..., name_to_idx["rv_30"]]
    composite = (f5 + f30 + fv + fr) / 4.0  # already cross-section z-scored

    order = composite.argsort(axis=1).argsort(axis=1).astype(np.float32)
    n = composite.shape[1]
    rank = order / max(n - 1, 1)
    return rank


def _compute_market_regime_features(
    panel: np.ndarray,
    lookback_bars: int,
) -> np.ndarray:
    """
    Causal market-regime features broadcast to every stock.

    At bar *t*, uses equal-weight market 1-bar returns over the prior
    ``lookback_bars`` window (≈30 trading days when configured that way).
    Does **not** extend the training calendar — only reads history already
    present in ``panel``.

    Returns (T, N, 6): bull/bear, high_vol/low_vol, mkt_ret_z, mkt_vol_z.
    """
    close = panel[..., 3]
    log_close = np.log(np.where(close > 0, close, np.nan))
    ret_1 = log_close - _shift_along_time(log_close, 1)
    mkt_ret = np.nan_to_num(np.nanmean(ret_1, axis=1), nan=0.0).astype(np.float64)

    T = mkt_ret.shape[0]
    lb = max(int(lookback_bars), 1)
    idx = np.arange(T, dtype=np.int64)
    win_start = np.maximum(0, idx - lb + 1)
    win_len = np.maximum(idx - win_start + 1, 1).astype(np.float64)

    csum = np.concatenate([[0.0], np.cumsum(mkt_ret)])
    csum2 = np.concatenate([[0.0], np.cumsum(mkt_ret ** 2)])
    win_sum = csum[idx + 1] - csum[win_start]
    win_sum2 = csum2[idx + 1] - csum2[win_start]
    cum_ret = win_sum
    win_mean = win_sum / win_len
    win_var = np.maximum(win_sum2 / win_len - win_mean ** 2, 0.0)
    roll_vol = np.sqrt(win_var)
    roll_vol = np.where(win_len > 1, roll_vol, 0.0)

    n = np.arange(1, T + 1, dtype=np.float64)
    ret_mu = np.cumsum(cum_ret) / n
    ret_var = np.maximum(np.cumsum(cum_ret ** 2) / n - ret_mu ** 2, 1e-16)
    ret_sd = np.sqrt(ret_var)
    vol_mu = np.cumsum(roll_vol) / n
    vol_var = np.maximum(np.cumsum(roll_vol ** 2) / n - vol_mu ** 2, 1e-16)
    vol_sd = np.sqrt(vol_var)

    ret_z = (cum_ret - ret_mu) / ret_sd
    vol_z = (roll_vol - vol_mu) / vol_sd

    bull = 1.0 / (1.0 + np.exp(-ret_z * 2.0))
    bear = 1.0 - bull
    high_vol = 1.0 / (1.0 + np.exp(-vol_z * 2.0))
    low_vol = 1.0 - high_vol

    n_stocks = panel.shape[1]
    regime = np.stack([bull, bear, high_vol, low_vol, ret_z, vol_z], axis=-1)  # (T, 6)
    regime = np.nan_to_num(regime, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    regime = np.broadcast_to(regime[:, None, :], (T, n_stocks, regime.shape[-1])).copy()
    return regime


def _forward_returns(panel: np.ndarray, horizon: int) -> np.ndarray:
    """Realized log-return over the next ``horizon`` bars, per stock."""
    close = panel[..., 3]
    log_close = np.log(np.where(close > 0, close, np.nan))
    fwd = np.full_like(log_close, np.nan, dtype=np.float64)
    fwd[:-horizon] = log_close[horizon:] - log_close[:-horizon]
    fwd = np.nan_to_num(fwd, nan=0.0, posinf=0.0, neginf=0.0)
    return fwd.astype(np.float32)


# ---------------------------------------------------------------------------
# Main builder + cache
# ---------------------------------------------------------------------------
def build_hf_dataset(cfg: HFConfig) -> Dict[str, np.ndarray]:
    """Build (and cache) the HF dataset for ``cfg``."""
    cfg.cache_dir.mkdir(parents=True, exist_ok=True)
    sid_tag = ""
    if cfg.stock_ids:
        sid_tag = f"_ids{len(cfg.stock_ids)}"
    regime_tag = "_regime" if cfg.use_regime_features else ""
    tag = cfg.cache_tag or (
        f"{cfg.pool_dir.name}_s{cfg.start_day}_d{cfg.num_days}"
        f"_L{cfg.lookback}_H{cfg.horizon}"
        f"_n{cfg.n_stocks_max if cfg.n_stocks_max else 'full'}"
        f"{sid_tag}{regime_tag}"
    )
    cache_path = cfg.cache_dir / f"{tag}.npz"
    meta_path = cfg.cache_dir / f"{tag}_meta.txt"

    if cache_path.is_file():
        with np.load(cache_path, allow_pickle=True) as z:
            return {k: z[k] for k in z.files}

    panel, ts, stock_ids = _build_panel(cfg)
    feats = _compute_features(panel, ts, cfg)
    if cfg.use_regime_features:
        lb = cfg.regime_lookback_days * cfg.bars_per_day
        regime = _compute_market_regime_features(panel, lb)
        feats = np.concatenate([feats, regime], axis=-1)
    sft = _linear_factor_label(feats, cfg)
    fwd = _forward_returns(panel, cfg.horizon)
    size_z, beta_z = _compute_exposure_features(panel)

    valid_start = max(cfg.lookback - 1, 30)
    valid_end = len(ts) - cfg.horizon

    ts_ns = pd.DatetimeIndex(ts).asi8 if ts.dtype == "datetime64[ns]" else \
        np.asarray(ts.to_numpy().astype("datetime64[ns]"), dtype="int64")
    out = {
        "x_seq": feats,                 # (T, N, F)
        "sft_label": sft,               # (T, N)
        "fwd_return": fwd,              # (T, N)
        "size_z": size_z,               # (T, N)
        "beta_z": beta_z,               # (T, N)
        "valid_start": np.int64(valid_start),
        "valid_end": np.int64(valid_end),
        "timestamps_ns": ts_ns,
        "stock_ids": np.array(stock_ids, dtype=object),
    }
    np.savez_compressed(cache_path, **out)
    meta_path.write_text(
        f"pool_dir   = {cfg.pool_dir}\n"
        f"num_days   = {cfg.num_days}\n"
        f"lookback   = {cfg.lookback}\n"
        f"horizon    = {cfg.horizon}\n"
        f"n_stocks   = {feats.shape[1]}\n"
        f"feat_dim   = {feats.shape[2]}\n"
        f"timesteps  = {feats.shape[0]}\n"
        f"valid_range= [{valid_start}, {valid_end})\n"
        f"timestamps = {ts[0]}  ..  {ts[-1]}\n"
    )
    return out


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------
class HFSFTDataset(Dataset):
    """Sliding-window cross sections for SFT.

    Each item is:
        x : (N, L, F)
        y : (N,)        rank-normalized linear-factor label in [0, 1].
    """

    def __init__(
        self,
        bundle: Dict[str, np.ndarray],
        lookback: int,
        indices: Optional[np.ndarray] = None,
    ) -> None:
        self.x_seq = bundle["x_seq"]
        self.sft_label = bundle["sft_label"]
        self.lookback = lookback
        valid_start = int(bundle["valid_start"])
        valid_end = int(bundle["valid_end"])
        all_indices = np.arange(valid_start, valid_end, dtype=np.int64)
        self.indices = all_indices if indices is None else indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        t = int(self.indices[idx])
        L = self.lookback
        window = self.x_seq[t - L + 1: t + 1]            # (L, N, F)
        x = np.transpose(window, (1, 0, 2)).copy()       # (N, L, F)
        y = self.sft_label[t]                            # (N,)
        return torch.from_numpy(x), torch.from_numpy(y).float()


class HFEpisodeDataset(Dataset):
    """Episodes of M consecutive cross sections for GRPO.

    Each item is:
        x_episode : (M, N, L, F)   features for M consecutive bars
        r_episode : (M, N)         realized forward returns at those bars
        size_z    : (M, N)         size exposure (optional in bundle)
        beta_z    : (M, N)         beta exposure (optional in bundle)
    """

    def __init__(
        self,
        bundle: Dict[str, np.ndarray],
        lookback: int,
        episode_len: int,
        indices: Optional[np.ndarray] = None,
    ) -> None:
        self.x_seq = bundle["x_seq"]
        self.fwd_return = bundle["fwd_return"]
        self.size_z = bundle.get("size_z")
        self.beta_z = bundle.get("beta_z")
        self.lookback = lookback
        self.episode_len = episode_len
        valid_start = int(bundle["valid_start"])
        valid_end = int(bundle["valid_end"]) - episode_len + 1
        all_indices = np.arange(valid_start, max(valid_end, valid_start + 1), dtype=np.int64)
        self.indices = all_indices if indices is None else indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        t0 = int(self.indices[idx])
        L = self.lookback
        M = self.episode_len
        x_eps: List[np.ndarray] = []
        for t in range(t0, t0 + M):
            window = self.x_seq[t - L + 1: t + 1]
            x_eps.append(np.transpose(window, (1, 0, 2)))
        x = np.stack(x_eps, axis=0).copy()                # (M, N, L, F)
        r = self.fwd_return[t0: t0 + M].copy()            # (M, N)
        if self.size_z is not None and self.beta_z is not None:
            sz = self.size_z[t0: t0 + M].copy()
            bz = self.beta_z[t0: t0 + M].copy()
            return (
                torch.from_numpy(x),
                torch.from_numpy(r),
                torch.from_numpy(sz),
                torch.from_numpy(bz),
            )
        return torch.from_numpy(x), torch.from_numpy(r)


# ---------------------------------------------------------------------------
# Time-aware split helpers
# ---------------------------------------------------------------------------
def bar_indices_for_dates(
    bundle: Dict[str, np.ndarray],
    dates: List[str],
) -> np.ndarray:
    """Bar indices whose calendar date falls in ``dates`` (within valid range)."""
    ts = pd.to_datetime(np.asarray(bundle["timestamps_ns"], dtype="int64"), unit="ns")
    date_set = set(dates)
    valid_start = int(bundle["valid_start"])
    valid_end = int(bundle["valid_end"])
    out: List[int] = []
    for t in range(valid_start, valid_end):
        if ts[t].strftime("%Y-%m-%d") in date_set:
            out.append(t)
    return np.asarray(out, dtype=np.int64)


def split_train_val_test(
    bundle: Dict[str, np.ndarray],
    val_ratio: float,
    test_ratio: float,
    embargo: int = 0,
    candidate_indices: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """Produce non-overlapping index arrays with embargo between splits."""
    valid_start = int(bundle["valid_start"])
    valid_end = int(bundle["valid_end"])
    if candidate_indices is not None:
        cand = np.asarray(candidate_indices, dtype=np.int64)
        all_idx = cand[(cand >= valid_start) & (cand < valid_end)]
    else:
        all_idx = np.arange(valid_start, valid_end, dtype=np.int64)

    n = len(all_idx)
    n_test = max(1, int(n * test_ratio))
    n_val = max(1, int(n * val_ratio))
    n_train = n - n_val - n_test - 2 * embargo
    if n_train <= 0:
        raise ValueError(
            "Not enough samples after split. Reduce ratios, lookback, or horizon."
        )

    train_idx = all_idx[:n_train]
    val_idx = all_idx[n_train + embargo: n_train + embargo + n_val]
    test_idx = all_idx[n_train + 2 * embargo + n_val:]
    return {"train": train_idx, "val": val_idx, "test": test_idx}


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    cfg = HFConfig(num_days=3, lookback=30, horizon=5, n_stocks_max=20)
    bundle = build_hf_dataset(cfg)
    print(f"x_seq        : {bundle['x_seq'].shape}")
    print(f"sft_label    : {bundle['sft_label'].shape}, range "
          f"[{bundle['sft_label'].min():.3f}, {bundle['sft_label'].max():.3f}]")
    print(f"fwd_return   : {bundle['fwd_return'].shape}, std "
          f"{bundle['fwd_return'].std():.4f}")
    print(f"valid range  : [{int(bundle['valid_start'])}, {int(bundle['valid_end'])})")

    splits = split_train_val_test(bundle, val_ratio=0.2, test_ratio=0.2, embargo=cfg.horizon)
    print("splits       :", {k: len(v) for k, v in splits.items()})

    ds = HFSFTDataset(bundle, lookback=cfg.lookback, indices=splits["train"])
    x, y = ds[0]
    print(f"sft sample   : x {tuple(x.shape)}, y {tuple(y.shape)}")

    eps = HFEpisodeDataset(bundle, lookback=cfg.lookback, episode_len=12, indices=splits["train"])
    if len(eps) > 0:
        xe, re = eps[0]
        print(f"episode      : x {tuple(xe.shape)}, r {tuple(re.shape)}")
