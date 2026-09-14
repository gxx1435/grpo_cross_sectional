from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from evaluation.analysis_spec import FALLBACK_COLOR, color_for


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.font_manager as fm
    import matplotlib.pyplot as plt

    plt.rcParams["axes.unicode_minus"] = False
    for fp in (
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\msyh.ttf",
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\simsun.ttc",
    ):
        if Path(fp).is_file():
            try:
                fm.fontManager.addfont(fp)
                name = fm.FontProperties(fname=fp).get_name()
                plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
                plt.rcParams["font.family"] = "sans-serif"
            except Exception:
                continue
            break
    return plt


def _save(fig, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    _plt().close(fig)


def _style_axes(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)


def plot_series(
    path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
    expected: Sequence[str],
    series: Dict[str, Tuple[np.ndarray, np.ndarray]],
    empty_ylim: Tuple[float, float] = (0.0, 1.0),
) -> None:
    """Draw axes + fixed-color legend. Missing series stay dashed in the legend."""
    try:
        plt = _plt()
    except Exception as e:
        print(f"plot skip: {e}", flush=True)
        return
    fig, ax = plt.subplots(figsize=(11, 5))
    _style_axes(ax, title, xlabel, ylabel)
    any_data = False
    for name in expected:
        c = color_for(name)
        xy = series.get(name)
        if xy is None or len(xy[0]) == 0:
            continue
        x, y = xy
        ax.plot(x, y, color=c, lw=1.5, label=name)
        any_data = True
    if not any_data:
        ax.set_xlim(pd.Timestamp("2025-05-01"), pd.Timestamp("2026-05-08"))
        ax.set_ylim(*empty_ylim)
        ax.text(0.5, 0.5, "尚无数据（轴与颜色已固定）", transform=ax.transAxes, ha="center", va="center", alpha=0.45)
    elif any(ax.get_legend_handles_labels()[1]):
        ax.legend(fontsize=7, loc="best", ncol=2)
    _save(fig, path)


def plot_grouped_bars(
    path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
    labels: Sequence[str],
    values: Sequence[Optional[float]],
    colors: Optional[Sequence[str]] = None,
) -> None:
    try:
        plt = _plt()
    except Exception as e:
        print(f"plot skip: {e}", flush=True)
        return
    fig, ax = plt.subplots(figsize=(11, 4.8))
    _style_axes(ax, title, xlabel, ylabel)
    xs = np.arange(len(labels))
    heights = []
    cols = []
    for i, v in enumerate(values):
        ok = True
        try:
            if v is None or str(v).strip() in ("", "-", "nan", "None"):
                ok = False
            elif not np.isfinite(float(v)):
                ok = False
        except (TypeError, ValueError):
            ok = False
        if not ok:
            heights.append(0.0)
            cols.append("#D9D9D9")
        else:
            heights.append(float(v))
            cols.append(colors[i] if colors is not None else color_for(labels[i]))
    ax.bar(xs, heights, color=cols, edgecolor="#333333", linewidth=0.4)
    ax.set_xticks(xs)
    ax.set_xticklabels(list(labels), rotation=35, ha="right")
    def _ok(v):
        try:
            return v is not None and str(v).strip() not in ("", "-", "nan", "None") and np.isfinite(float(v))
        except (TypeError, ValueError):
            return False

    if not any(_ok(v) for v in values):
        ax.set_ylim(0.0, 1.0)
        ax.text(0.5, 0.55, "尚无数据（轴与颜色已固定）", transform=ax.transAxes, ha="center", va="center", alpha=0.45)
    _save(fig, path)


def plot_hist_or_empty(values: np.ndarray, path: Path, title: str, xlabel: str = "取值", ylabel: str = "频数", bins: int = 40, color: str = FALLBACK_COLOR) -> None:
    try:
        plt = _plt()
    except Exception as e:
        print(f"plot skip: {e}", flush=True)
        return
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    fig, ax = plt.subplots(figsize=(8, 4))
    _style_axes(ax, title, xlabel, ylabel)
    if v.size == 0:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.text(0.5, 0.5, "尚无数据（轴与颜色已固定）", transform=ax.transAxes, ha="center", va="center", alpha=0.45)
    else:
        ax.hist(v, bins=bins, color=color, alpha=0.85, edgecolor="white")
    _save(fig, path)


def _alias_candidates(name: str) -> List[str]:
    name = str(name)
    out = [name]
    if name.endswith("_top30"):
        out.append(name[: -len("_top30")])
    else:
        out.append(f"{name}_top30")
    return out


def series_from_daily(df: pd.DataFrame, ycol: str, expected: Sequence[str], cum: bool = False) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    if df is None or df.empty or ycol not in df.columns or "model" not in df.columns:
        return out
    grouped = {str(n): g for n, g in df.groupby(df["model"].astype(str))}
    counts = {n: len(g) for n, g in grouped.items()}
    names = list(expected) if expected else list(grouped)
    for name in names:
        cands = [c for c in _alias_candidates(name) if c in grouped]
        if not cands:
            continue
        pick = max(cands, key=lambda c: counts.get(c, 0))
        g = grouped[pick]
        g = g.sort_values("date") if "date" in g.columns else g
        y = pd.to_numeric(g[ycol], errors="coerce").to_numpy(dtype=np.float64)
        if cum:
            y = np.cumsum(np.nan_to_num(y, nan=0.0))
        if "date" in g.columns:
            x = pd.to_datetime(g["date"], errors="coerce").to_numpy()
        else:
            x = np.arange(len(y))
        out[str(name)] = (x, y)
    return out


def plot_cum(df: pd.DataFrame, path: Path, title: str, focus: Optional[List[str]] = None) -> None:
    expected = list(focus) if focus else (sorted(df["model"].astype(str).unique()) if df is not None and len(df) and "model" in df.columns else [])
    series = series_from_daily(df if df is not None else pd.DataFrame(), "net_return", expected, cum=True)
    plot_series(path, title, "交易日", "累计净收益（Σ net_return）", expected, series, empty_ylim=(-0.1, 0.1))


def plot_metric_ts(df: pd.DataFrame, path: Path, metric: str, title: str, group: str = "model", expected: Optional[Sequence[str]] = None, ylabel: Optional[str] = None) -> None:
    names = list(expected) if expected is not None else (
        sorted(df[group].astype(str).unique()) if df is not None and len(df) and group in df.columns else []
    )
    series: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    if df is not None and len(df) and metric in df.columns and group in df.columns:
        tmp = df.rename(columns={group: "model"})
        if "date" not in tmp.columns and "t" in tmp.columns:
            tmp = tmp.assign(date=pd.to_datetime(tmp["t"], unit="s", errors="coerce"))
        series = series_from_daily(tmp, metric, names, cum=False)
    plot_series(path, title, "交易日", ylabel or metric, names, series)


def plot_bars(df: pd.DataFrame, path: Path, x: str, y: str, title: str, expected: Optional[Sequence[str]] = None) -> None:
    labels = list(expected) if expected is not None else (df[x].astype(str).tolist() if df is not None and len(df) and x in df.columns else [])
    lookup = {}
    if df is not None and len(df) and x in df.columns and y in df.columns:
        for _, row in df.iterrows():
            lookup[str(row[x])] = row[y]
    values = [lookup.get(lab) for lab in labels]
    plot_grouped_bars(path, title, x, y, labels, values, [color_for(lab) for lab in labels])


def plot_hist(values: np.ndarray, path: Path, title: str, bins: int = 40) -> None:
    plot_hist_or_empty(values, path, title, bins=bins)


def plot_epoch_lines(
    path: Path,
    title: str,
    ylabel: str,
    val: pd.DataFrame,
    metric: str,
    expected: Sequence[str],
) -> None:
    """One line per model across epoch 1–3. Missing models stay dashed in the legend."""
    try:
        plt = _plt()
    except Exception as e:
        print(f"plot skip: {e}", flush=True)
        return
    fig, ax = plt.subplots(figsize=(11, 4.8))
    _style_axes(ax, title, "epoch", ylabel)
    any_data = False
    for name in expected:
        c = color_for(name)
        xs, ys = [], []
        if val is not None and len(val) and "model" in val.columns and metric in val.columns:
            hit = val[val["model"].astype(str) == str(name)].copy()
            if "epoch" in hit.columns and len(hit):
                hit["epoch"] = pd.to_numeric(hit["epoch"], errors="coerce")
                hit[metric] = pd.to_numeric(hit[metric], errors="coerce")
                hit = hit.dropna(subset=["epoch", metric]).sort_values("epoch")
                if "n" in hit.columns:
                    hit["_n"] = pd.to_numeric(hit["n"], errors="coerce").fillna(0)
                    hit = hit.sort_values(["epoch", "_n"]).drop_duplicates("epoch", keep="last")
                xs = hit["epoch"].to_numpy()
                ys = hit[metric].to_numpy(dtype=np.float64)
        if len(xs) == 0:
            continue
        ax.plot(xs, ys, color=c, lw=1.6, marker="o", label=name)
        any_data = True
    ax.set_xticks([1, 2, 3])
    if not any_data:
        ax.set_ylim(0.0, 1.0)
        ax.text(0.5, 0.5, "尚无数据（轴与颜色已固定）", transform=ax.transAxes, ha="center", va="center", alpha=0.45)
    ax.legend(fontsize=7, loc="best", ncol=2)
    _save(fig, path)


def plot_clustered_bars(
    path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
    groups: Sequence[str],
    series_names: Sequence[str],
    values: Sequence[Sequence[Optional[float]]],
    colors: Optional[Sequence[str]] = None,
) -> None:
    """groups on x (e.g. months), series = models with fixed colors."""
    try:
        plt = _plt()
    except Exception as e:
        print(f"plot skip: {e}", flush=True)
        return
    fig, ax = plt.subplots(figsize=(11, 4.8))
    _style_axes(ax, title, xlabel, ylabel)
    n_g = max(len(groups), 1)
    n_s = max(len(series_names), 1)
    width = min(0.8 / n_s, 0.18)
    xs = np.arange(n_g)
    any_data = False

    def _ok(v):
        try:
            return v is not None and str(v).strip() not in ("", "-", "nan", "None") and np.isfinite(float(v))
        except (TypeError, ValueError):
            return False

    for i, name in enumerate(series_names):
        c = colors[i] if colors is not None else color_for(name)
        row = values[i] if i < len(values) else [None] * n_g
        heights = []
        for v in row:
            if _ok(v):
                heights.append(float(v))
                any_data = True
            else:
                heights.append(np.nan)
        ax.bar(xs + (i - (n_s - 1) / 2) * width, heights, width=width, color=c, edgecolor="#333333", linewidth=0.3, label=name)
    ax.set_xticks(xs)
    ax.set_xticklabels(list(groups), rotation=0, ha="center")
    if not any_data:
        ax.set_ylim(0.0, 1.0)
        ax.text(0.5, 0.55, "尚无数据（轴与颜色已固定）", transform=ax.transAxes, ha="center", va="center", alpha=0.45)
    ax.legend(fontsize=7, loc="best", ncol=2)
    _save(fig, path)
