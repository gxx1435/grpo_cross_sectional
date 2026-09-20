"""Strictly trailing minute features for the S&P 500 experiment."""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

BASE_FEATURES = ["open", "high", "low", "close", "volume"]
CANDLE_FEATURES = [
    "high_tail",
    "low_tail",
    "tail_adj_ret",
    "hammer",
    "volatility",
]
BAR_SHAPE_FEATURES = [
    "high_low_range",
    "open_close_return",
    "body_ratio",
    "upper_shadow_ratio",
    "lower_shadow_ratio",
    "close_position",
]
VOLUME_PRICE_FEATURES = [
    "log_volume",
    "volume_change",
    "vwap_deviation",
    "amihud",
    "volume_weighted_return",
]
CALENDAR_FEATURES = [
    "day_of_week_sin",
    "day_of_week_cos",
    "is_monday",
    "is_friday",
]


def feature_names(cfg: dict) -> List[str]:
    windows = [int(w) for w in cfg["features"]["windows"]]
    fcfg = cfg["features"]
    out = list(BASE_FEATURES)
    for prefix in (
        "o2c",
        "h2c",
        "l2c",
        "v2c",
        "ma2c",
        "ma2v",
        "zscore",
        "o2c_per_std",
        "v2c_per_std",
        "volat_per_std",
    ):
        out.extend(f"{prefix}_w{w}" for w in windows)
    out.extend(CANDLE_FEATURES)
    out.extend(f"vwap_ratio_w{int(a)}_w{int(b)}" for a, b in cfg["features"]["vwap_pairs"])
    out.append("log_return_1")
    out.extend(f"return_{int(w)}" for w in fcfg["base_return_windows"])
    out.extend(BAR_SHAPE_FEATURES)
    out.extend(f"vol_{int(w)}" for w in fcfg["base_volatility_windows"])
    out.extend(VOLUME_PRICE_FEATURES[:2])
    out.extend(f"relative_volume_{int(w)}" for w in fcfg["relative_volume_windows"])
    out.extend(VOLUME_PRICE_FEATURES[2:])
    out.extend(f"price_ma{int(w)}_deviation" for w in fcfg["price_ma_windows"])
    ma_short, ma_long = (int(w) for w in fcfg["ma_spread_pair"])
    out.append(f"ma{ma_short}_ma{ma_long}")
    out.extend(str(name) for name in fcfg["cross_sectional_rank_sources"])
    out.extend(CALENDAR_FEATURES)
    if len(out) != len(set(out)):
        raise RuntimeError("feature schema contains duplicate names")
    return out


def feature_dim(cfg: dict) -> int:
    return len(feature_names(cfg))


def _safe_div(num: pd.Series, den: pd.Series, eps: float) -> pd.Series:
    good = den.abs() > eps
    out = pd.Series(np.nan, index=num.index, dtype="float64")
    out.loc[good] = num.loc[good] / den.loc[good]
    return out


def compute_symbol_features(frame: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Compute one symbol in timestamp order; every rolling window is trailing."""
    x = frame.sort_values("minute_timestamp", kind="mergesort").copy()
    derived: Dict[str, pd.Series] = {}
    eps = float(cfg["features"]["epsilon"])
    windows = [int(w) for w in cfg["features"]["windows"]]
    o, h, l, c, v = (x[k].astype("float64") for k in ("open", "high", "low", "close", "volume"))
    typical = (h + l + c) / 3.0
    pv = typical * v
    rolling_vwap: Dict[int, pd.Series] = {}
    for w in windows:
        count = c.rolling(w, min_periods=w).count()
        open_w = o.shift(w - 1).where(count == w)
        high_w = h.rolling(w, min_periods=w).max()
        low_w = l.rolling(w, min_periods=w).min()
        ma_w = c.rolling(w, min_periods=w).mean()
        std_w = c.rolling(w, min_periods=w).std(ddof=0)
        vol_sum = v.rolling(w, min_periods=w).sum()
        vwap_w = _safe_div(pv.rolling(w, min_periods=w).sum(), vol_sum, eps)
        rolling_vwap[w] = vwap_w
        derived[f"o2c_w{w}"] = (_safe_div(c, open_w, eps) - 1.0) * 1e4
        derived[f"h2c_w{w}"] = (_safe_div(c, high_w, eps) - 1.0) * 1e4
        derived[f"l2c_w{w}"] = (_safe_div(c, low_w, eps) - 1.0) * 1e4
        derived[f"v2c_w{w}"] = (_safe_div(c, vwap_w, eps) - 1.0) * 1e4
        derived[f"ma2c_w{w}"] = (_safe_div(c, ma_w, eps) - 1.0) * 1e4
        derived[f"ma2v_w{w}"] = (_safe_div(vwap_w, ma_w, eps) - 1.0) * 1e4
        derived[f"zscore_w{w}"] = _safe_div(std_w, ma_w, eps) * 1e4
        derived[f"o2c_per_std_w{w}"] = _safe_div(c - open_w, std_w, eps)
        derived[f"v2c_per_std_w{w}"] = _safe_div(c - vwap_w, std_w, eps)
        derived[f"volat_per_std_w{w}"] = _safe_div(h - l, std_w, eps)

    derived["high_tail"] = _safe_div(h - pd.concat([c, o], axis=1).max(axis=1), o, eps) * 1e4
    derived["low_tail"] = _safe_div(pd.concat([c, o], axis=1).min(axis=1) - l, o, eps) * 1e4
    derived["tail_adj_ret"] = _safe_div(2.0 * c - h - l, o, eps) * 1e4
    derived["hammer"] = _safe_div(o + c, h + l, eps) * 1e4
    derived["volatility"] = _safe_div(h - l, h + l, eps) * 2e4
    for a, b in cfg["features"]["vwap_pairs"]:
        a, b = int(a), int(b)
        derived[f"vwap_ratio_w{a}_w{b}"] = _safe_div(rolling_vwap[a], rolling_vwap[b], eps) * 1e4

    logc = np.log(c.where(c > 0))
    r1 = logc.diff()
    derived["log_return_1"] = r1
    for w in (int(value) for value in cfg["features"]["base_return_windows"]):
        derived[f"return_{w}"] = c.pct_change(w, fill_method=None)
    full = h - l
    derived["high_low_range"] = _safe_div(full, c, eps)
    derived["open_close_return"] = _safe_div(c, o, eps) - 1.0
    derived["body_ratio"] = _safe_div((c - o).abs(), full, eps)
    derived["upper_shadow_ratio"] = _safe_div(h - pd.concat([o, c], axis=1).max(axis=1), full, eps)
    derived["lower_shadow_ratio"] = _safe_div(pd.concat([o, c], axis=1).min(axis=1) - l, full, eps)
    derived["close_position"] = _safe_div(c - l, full, eps)
    for w in (int(value) for value in cfg["features"]["base_volatility_windows"]):
        derived[f"vol_{w}"] = r1.rolling(w, min_periods=w).std(ddof=0)
    derived["log_volume"] = np.log1p(v.where(v >= 0))
    derived["volume_change"] = derived["log_volume"].diff()
    for w in (int(value) for value in cfg["features"]["relative_volume_windows"]):
        derived[f"relative_volume_{w}"] = _safe_div(v, v.rolling(w, min_periods=w).mean(), eps)
    derived["vwap_deviation"] = _safe_div(c, typical, eps) - 1.0
    amount_proxy = typical * v
    derived["amihud"] = _safe_div(r1.abs(), amount_proxy, eps)
    volume_window = int(cfg["features"]["volume_weighted_return_volume_window"])
    derived["volume_weighted_return"] = r1 * derived[f"relative_volume_{volume_window}"]
    for w in (int(value) for value in cfg["features"]["price_ma_windows"]):
        ma = c.rolling(w, min_periods=w).mean()
        derived[f"price_ma{w}_deviation"] = _safe_div(c, ma, eps) - 1.0
    ma_short_window, ma_long_window = (int(value) for value in cfg["features"]["ma_spread_pair"])
    ma_short = c.rolling(ma_short_window, min_periods=ma_short_window).mean()
    ma_long = c.rolling(ma_long_window, min_periods=ma_long_window).mean()
    derived[f"ma{ma_short_window}_ma{ma_long_window}"] = _safe_div(ma_short, ma_long, eps) - 1.0
    dow = x["trading_date"].dt.dayofweek.astype("float64")
    derived["day_of_week_sin"] = np.sin(2 * np.pi * dow / 5.0)
    derived["day_of_week_cos"] = np.cos(2 * np.pi * dow / 5.0)
    derived["is_monday"] = (dow == 0).astype("float64")
    derived["is_friday"] = (dow == 4).astype("float64")
    return pd.concat([x, pd.DataFrame(derived, index=x.index)], axis=1)


def add_cross_sectional_features(frame: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    x = frame.copy()
    group = x.groupby("minute_timestamp", sort=False, observed=True)
    # The mapping is persisted in the feature-config hash and therefore cannot
    # change silently between feature versions.
    mapping = cfg["features"]["cross_sectional_rank_sources"]
    for dst, src in mapping.items():
        x[dst] = group[src].rank(method="average", pct=True, na_option="keep")
    size = group["stock_code"].transform("count")
    present = group["return_30"].transform("count")
    x["cross_section_size"] = size.astype("int32")
    x["cross_section_missing_count"] = (size - present).astype("int32")
    x["rank_calculation_timestamp"] = x["minute_timestamp"]
    x["data_available_cutoff"] = x["minute_timestamp"]
    return x


def finalize_feature_validity(frame: pd.DataFrame, names: Sequence[str]) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    vals = frame[list(names)].replace([np.inf, -np.inf], np.nan)
    finite = np.isfinite(vals.to_numpy(dtype=np.float64, copy=False))
    validity = pd.DataFrame(
        {
            "feature_valid_fraction": finite.mean(axis=1).astype("float32"),
            "feature_invalid_count": (~finite).sum(axis=1).astype("int16"),
            "feature_valid_mask": finite.all(axis=1),
        },
        index=frame.index,
    )
    metadata = frame.drop(columns=list(names), errors="ignore")
    x = pd.concat([metadata, vals, validity], axis=1)
    rates: Dict[str, Dict[str, float]] = {}
    for j, name in enumerate(names):
        rates[name] = {
            "missing_rate": float(vals[name].isna().mean()),
            "invalid_rate": float((~finite[:, j]).mean()),
        }
    return x, rates
