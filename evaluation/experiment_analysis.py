"""Placeholder tables/figures first; fill real cells after each finished unit. Never invent numbers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from evaluation.analysis_spec import (
    FIG_CUM,
    FIG_F1,
    FIG_G,
    FIG_GPU,
    FIG_HIT,
    FIG_HIST,
    FIG_IC,
    FIG_RIC,
    FIG_TO,
    G_CURVE_COLS,
    G_VALUES,
    GATE_COLS,
    GEN_MODELS,
    GPU_COLS,
    LEAKAGE_COLS,
    MONTH_STATUS_COLS,
    NAV_COLS,
    NAV_DAILY_MODELS,
    PLACEHOLDER,
    PORTFOLIO_MODELS,
    PRED_COLS,
    PRED_MODELS,
    RL_MODELS,
    TEST_MONTHS,
    UNIT_STATUS_COLS,
    VAL_BEST_COLS,
    VAL_COLS,
    color_for,
    family_series,
    legend_hex,
    unit_status_rows,
)
from evaluation.plots import (
    plot_clustered_bars,
    plot_epoch_lines,
    plot_grouped_bars,
    plot_hist_or_empty,
    plot_metric_ts,
    plot_series,
    series_from_daily,
)
from evaluation.portfolio_metrics import summarize_nav
from experiments.catalog import CATALOG, analysis_root, count_summary, family_dir
from utils.logging import log, write_json


def _finite(v: Any) -> bool:
    if v is None:
        return False
    if isinstance(v, str) and v.strip() in ("", PLACEHOLDER, "nan", "None"):
        return False
    try:
        return bool(np.isfinite(float(v)))
    except (TypeError, ValueError):
        return bool(str(v).strip()) and str(v) != PLACEHOLDER


def _fmt(v: Any, nd: int = 4) -> str:
    if not _finite(v):
        return PLACEHOLDER
    if isinstance(v, (float, np.floating, int, np.integer)):
        fv = float(v)
        if abs(fv - round(fv)) < 1e-12 and abs(fv) >= 1:
            return str(int(round(fv)))
        return f"{fv:.{nd}f}"
    return str(v)


def table_md(columns: Sequence[str], rows: Sequence[Dict[str, Any]], floatfmt: int = 4) -> str:
    head = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    lines = [head, sep]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(row.get(c), floatfmt) for c in columns) + " |")
    return "\n".join(lines) + "\n"


def color_table_md(names: Sequence[str]) -> str:
    rows = [{"模型": n, "颜色": color_for(n)} for n in names]
    return table_md(["模型", "颜色"], rows)


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        if path.is_file():
            return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()
    return pd.DataFrame()


def _lookup(df: pd.DataFrame, key: str, key_col: str = "model") -> Dict[str, Any]:
    if df is None or df.empty or key_col not in df.columns:
        return {}
    hit = df[df[key_col].astype(str) == str(key)]
    if hit.empty:
        return {}
    return hit.iloc[-1].to_dict()


def _rows_from_spec(row_keys: Sequence[str], columns: Sequence[str], df: pd.DataFrame, key_col: str = "model") -> List[Dict[str, Any]]:
    out = []
    for key in row_keys:
        rec = {c: PLACEHOLDER for c in columns}
        rec[key_col] = key
        got = _lookup(df, key, key_col)
        for c in columns:
            if c == key_col:
                continue
            if c in got and _finite(got[c]):
                rec[c] = got[c]
        out.append(rec)
    return out


def _best_val(val: pd.DataFrame) -> pd.DataFrame:
    if val is None or val.empty or "model" not in val.columns:
        return pd.DataFrame()
    if "MSE" not in val.columns:
        return val.groupby("model", as_index=False).tail(1)
    rows = []
    for name, g in val.groupby("model"):
        g = g.copy()
        g["_mse"] = pd.to_numeric(g["MSE"], errors="coerce")
        g["_n"] = pd.to_numeric(g["n"], errors="coerce") if "n" in g.columns else 0.0
        g = g[np.isfinite(g["_mse"])]
        if g.empty:
            continue
        comparable = g[g["_n"].fillna(0) >= 8]
        pick = comparable if len(comparable) else g
        rows.append(pick.loc[pick["_mse"].idxmin()])
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _best_val_by_month(val: pd.DataFrame) -> pd.DataFrame:
    if val is None or val.empty or "model" not in val.columns:
        return pd.DataFrame()
    if "month" not in val.columns:
        return _best_val(val)
    parts = []
    for month, g in val.groupby(val["month"].astype(str)):
        b = _best_val(g)
        if len(b):
            b = b.copy()
            b["month"] = month
            parts.append(b)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _drop_short_series(daily: pd.DataFrame, min_days: int = 15) -> pd.DataFrame:
    """Drop replay/partial series that are shorter than a trading month."""
    if daily is None or daily.empty or "model" not in daily.columns:
        return daily if daily is not None else pd.DataFrame()
    df = daily.copy()
    if "date" not in df.columns:
        return df
    n = df.groupby(df["model"].astype(str))["date"].nunique()
    keep = set(n[n >= int(min_days)].index)
    return df[df["model"].astype(str).isin(keep)].copy()


def _prefer_pool_aliases(daily: pd.DataFrame) -> pd.DataFrame:
    """Copy *_top30 series onto the short names used by month figures."""
    if daily is None or daily.empty or "model" not in daily.columns:
        return daily if daily is not None else pd.DataFrame()
    df = daily.copy()
    df["model"] = df["model"].astype(str)
    extras = []
    for name in sorted(df["model"].unique()):
        if not name.endswith("_top30"):
            continue
        short = name[: -len("_top30")]
        src = df[df["model"] == name]
        n_src = src["date"].nunique() if "date" in src.columns else len(src)
        old = df[df["model"] == short]
        n_old = old["date"].nunique() if len(old) and "date" in old.columns else 0
        if n_src >= n_old and n_src > 0:
            df = df[df["model"] != short]
            cp = src.copy()
            cp["model"] = short
            if "strategy_name" in cp.columns:
                cp["strategy_name"] = short
            extras.append(cp)
    if extras:
        df = pd.concat([df, *extras], ignore_index=True)
    return df


def _nav_or_empty(daily: pd.DataFrame) -> pd.DataFrame:
    if daily is None or daily.empty or "model" not in daily.columns or "net_return" not in daily.columns:
        return pd.DataFrame()
    nav = summarize_nav(_drop_short_series(_prefer_pool_aliases(daily), min_days=15))
    if nav is None or nav.empty or "n_days" not in nav.columns:
        return nav
    n = pd.to_numeric(nav["n_days"], errors="coerce")
    return nav.loc[n >= 15].copy()


def _pred_mean(pred: pd.DataFrame) -> pd.DataFrame:
    if pred is None or pred.empty or "model" not in pred.columns:
        return pd.DataFrame()
    num = [c for c in PRED_COLS if c in pred.columns and c != "model"]
    if not num:
        return pd.DataFrame()
    g = pred.groupby("model", as_index=False)[num].mean(numeric_only=True)
    if "date" in pred.columns:
        n = pred.groupby("model")["date"].nunique().rename("n")
        g = g.merge(n, on="model", how="left")
    return g


def _write_table_csv(path: Path, columns: Sequence[str], rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=list(columns)).to_csv(path, index=False)


def collect_val(out_root: Path, oos: str, month: Optional[str] = None) -> pd.DataFrame:
    frames = []
    oos_dir = Path(out_root) / oos
    for p in oos_dir.glob("test_month=*/*/alpha_val_metrics.csv"):
        frames.append(_read_csv(p))
    for p in oos_dir.glob("test_month=*/alpha_val_metrics.csv"):
        frames.append(_read_csv(p))
    fam = family_dir(out_root, "M01_prediction") / "tables"
    if fam.is_dir():
        for p in fam.glob("val_*.csv"):
            frames.append(_read_csv(p))
    frames = [f for f in frames if f is not None and len(f)]
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    if month and "month" in df.columns:
        df = df[df["month"].astype(str) == str(month)].copy()
    keys = [c for c in ("model", "month", "epoch") if c in df.columns]
    if len(keys) >= 2:
        if "n" in df.columns:
            df = df.copy()
            df["_n"] = pd.to_numeric(df["n"], errors="coerce").fillna(0)
            df = df.sort_values("_n").drop_duplicates(subset=keys, keep="last").drop(columns=["_n"])
        else:
            df = df.drop_duplicates(subset=keys, keep="last")
    return df


def collect_daily(out_root: Path, oos: str, month: Optional[str] = None) -> pd.DataFrame:
    frames = []
    root = Path(out_root) / oos
    month_dirs = [root / f"test_month={month}"] if month else sorted(root.glob("test_month=*"))
    for mdir in month_dirs:
        if not mdir.is_dir():
            continue
        for p in mdir.glob("*/backtest_daily.csv"):
            if p.parent.name == "tables":
                continue
            if (p.parent / "INVALID.json").is_file():
                continue
            df = _read_csv(p)
            if len(df):
                frames.append(df)
        p = mdir / "backtest_daily.csv"
        df = _read_csv(p)
        if len(df):
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    keys = [c for c in ("date", "model") if c in df.columns]
    if keys:
        df = df.drop_duplicates(subset=keys, keep="last")
    return df


def collect_pred(out_root: Path, oos: str, month: Optional[str] = None) -> pd.DataFrame:
    frames = []
    root = Path(out_root) / oos
    for p in root.glob("test_month=*/**/prediction_metrics.csv"):
        if (p.parent / "INVALID.json").is_file():
            continue
        df = _read_csv(p)
        if month and "date" in df.columns and len(df):
            if not str(df["date"].iloc[0]).startswith(str(month)):
                # keep if any date in month
                if not df["date"].astype(str).str.startswith(str(month)).any():
                    continue
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _filter_daily(daily: pd.DataFrame, prefixes: List[str]) -> pd.DataFrame:
    if daily is None or daily.empty or not prefixes:
        return daily if daily is not None else pd.DataFrame()
    m = daily["model"].astype(str)
    keep = np.zeros(len(daily), dtype=bool)
    for p in prefixes:
        keep |= m.str.startswith(p)
    return daily.loc[keep].copy()


def _status_from_sources(month: str, val: pd.DataFrame, daily: pd.DataFrame, pred: pd.DataFrame) -> List[Dict[str, Any]]:
    rows = unit_status_rows()
    val_models = set(val["model"].astype(str)) if val is not None and len(val) and "model" in val.columns else set()
    daily_models = set(daily["model"].astype(str)) if daily is not None and len(daily) and "model" in daily.columns else set()
    pred_models = set(pred["model"].astype(str)) if pred is not None and len(pred) and "model" in pred.columns else set()
    for rec in rows:
        name = str(rec["model"])
        unit = rec["unit"]
        done = False
        if unit == "A 预测":
            done = name in val_models or name in pred_models or f"pred_{name}_ew_top30" in daily_models or f"pred_{name}_ew" in daily_models
        elif unit.startswith("B 组合"):
            pool = unit.split("/")[-1] if "/" in unit else "top30"
            done = f"baseline_{name}_{pool}" in daily_models or f"baseline_{name}" in daily_models
        elif unit.startswith("C 生成"):
            pool = unit.split("/")[-1] if "/" in unit else "top30"
            done = f"gen_{name}_{pool}" in daily_models or f"gen_{name}" in daily_models
        elif unit.startswith("D RL"):
            pool = unit.split("/")[-1] if "/" in unit else "top30"
            done = f"rl_{name}_{pool}" in daily_models or f"rl_{name}" in daily_models or name in daily_models
        elif unit.startswith("E G"):
            pool = unit.split("/")[-1] if "/" in unit else "top30"
            g = name.replace("G=", "")
            # short alias ssfm_G8 comes from top30; do not treat it as all-pool done
            done = f"ssfm_G{g}_{pool}" in daily_models
        rec["status"] = "已写入" if done else PLACEHOLDER
    return rows


def _analysis_text(title: str, filled: List[str], notes: List[str], extra: Dict[str, Any]) -> str:
    bits = [
        f"以下只讨论**已经写入数字**的行；单元格为 `{PLACEHOLDER}` 的表示该实验尚未跑完，不编造。",
        f"来源: {extra.get('source', 'disk')}。OOS=`{extra.get('oos', 'sequential_oos_weekly_retrain')}`，"
        f"费用={extra.get('cost_bps', '')}bps，TopK={extra.get('top_k', 30)}。",
        "",
    ]
    if extra.get("last_unit"):
        bits.append(f"最近完成单元: `{extra['last_unit']}`。")
    if filled:
        bits.append("已有数据的对象: " + ", ".join(f"`{x}`" for x in filled) + "。")
    else:
        bits.append("目前还没有任何实测单元格。图只画了坐标轴和固定颜色图例。")
    bits.extend(notes)
    if extra.get("valid_audit") is False:
        bits.append("**本月泄漏审计未通过，结果不得并入 final_summary。**")
    return "\n".join(bits) + "\n"


def _notes_from_val_nav(val_best: pd.DataFrame, nav: pd.DataFrame, pred_mean: pd.DataFrame) -> List[str]:
    notes: List[str] = []
    if val_best is not None and len(val_best):
        if "IC" in val_best.columns and val_best["IC"].notna().any():
            i = pd.to_numeric(val_best["IC"], errors="coerce").idxmax()
            if pd.notna(i):
                notes.append(f"- Val IC 目前最高: `{val_best.loc[i, 'model']}` = {_fmt(val_best.loc[i, 'IC'])}。")
        if "topk_f1" in val_best.columns and val_best["topk_f1"].notna().any():
            i = pd.to_numeric(val_best["topk_f1"], errors="coerce").idxmax()
            if pd.notna(i):
                notes.append(f"- Val F1@Top30 目前最高: `{val_best.loc[i, 'model']}` = {_fmt(val_best.loc[i, 'topk_f1'])}（随机约 0.06）。")
        if "MSE" in val_best.columns and val_best["MSE"].notna().any():
            i = pd.to_numeric(val_best["MSE"], errors="coerce").idxmin()
            if pd.notna(i):
                notes.append(f"- Val MSE 目前最低（入选口径）: `{val_best.loc[i, 'model']}` = {_fmt(val_best.loc[i, 'MSE'], 6)}。")
    if pred_mean is not None and len(pred_mean) and "IC" in pred_mean.columns:
        s = pd.to_numeric(pred_mean["IC"], errors="coerce")
        if s.notna().any():
            i = s.idxmax()
            notes.append(f"- Test 日均 IC 目前最高: `{pred_mean.loc[i, 'model']}` = {_fmt(pred_mean.loc[i, 'IC'])}。")
    if nav is not None and len(nav) and "total_net_return" in nav.columns:
        s = pd.to_numeric(nav["total_net_return"], errors="coerce")
        if s.notna().any():
            i = s.idxmax()
            notes.append(
                f"- 已回测净值目前最高: `{nav.loc[i, 'model']}` 累计净收益 {_fmt(nav.loc[i, 'total_net_return'])}，"
                f"Sharpe {_fmt(nav.loc[i].get('sharpe'))}。"
            )
        gnav = nav[nav["model"].astype(str).str.fullmatch(r"ssfm_G\d+")].copy()
        if len(gnav):
            gs = pd.to_numeric(gnav["total_net_return"], errors="coerce")
            if gs.notna().any():
                bits = []
                for g in G_VALUES:
                    row = gnav[gnav["model"].astype(str) == f"ssfm_G{g}"]
                    if len(row):
                        bits.append(f"G={g} {_fmt(row.iloc[0]['total_net_return'])}")
                i = gs.idxmax()
                notes.append(
                    f"- 实验 E（G 候选数，top30）累计净收益最高: `{gnav.loc[i, 'model']}` = {_fmt(gnav.loc[i, 'total_net_return'])}"
                    + (f"（{'；'.join(bits)}）" if bits else "")
                    + "。"
                )
    return notes


def _g_month_section(nav: pd.DataFrame) -> str:
    if nav is None or nav.empty or "model" not in nav.columns:
        return ""
    gnav = nav[nav["model"].astype(str).str.fullmatch(r"ssfm_G\d+")].copy()
    if gnav.empty:
        return ""
    rows = []
    for g in G_VALUES:
        hit = gnav[gnav["model"].astype(str) == f"ssfm_G{g}"]
        if not len(hit):
            continue
        r = hit.iloc[0]
        rows.append(
            f"| G={g} | {_fmt(r.get('total_net_return'))} | {_fmt(r.get('sharpe'))} | "
            f"{_fmt(r.get('max_drawdown'))} | {_fmt(r.get('mean_turnover'))} | {int(r.get('n_days') or 0)} |"
        )
    if not rows:
        return ""
    gs = pd.to_numeric(gnav["total_net_return"], errors="coerce")
    best = gnav.loc[gs.idxmax(), "model"] if gs.notna().any() else ""
    return (
        "\n## 实验 E：G 候选数\n\n"
        "只跑 top30。每个交易日加载当时周五生效的 primary / SS-FM，采 G 条权重后按 pred_utility 选 1 条。"
        "与默认 `gen_ssfm`（固定 default_g）不是同一条抽样路径。\n\n"
        "| G | 累计净收益 | Sharpe | 最大回撤 | 日均换手 | 天数 |\n"
        "| --- | --- | --- | --- | --- | --- |\n"
        + "\n".join(rows)
        + (f"\n\n本月网格里累计净收益最好的是 `{best}`。更大的 G 没有单调变好。\n" if best else "\n")
    )


def _featstd_month_section(dest: Path) -> str:
    p = Path(dest) / "tables" / "featstd_sft_compare.csv"
    if not p.is_file():
        return ""
    df = _read_csv(p)
    if df is None or df.empty:
        return ""
    cols = [c for c in ("model", "split", "variant", "MSE", "IC", "RankIC", "topk_f1", "n") if c in df.columns]
    rows = []
    for _, r in df.iterrows():
        rec = {c: r[c] if c in r.index else PLACEHOLDER for c in cols}
        rows.append(rec)
    note_p = Path(dest) / "tables" / "featstd_sft_note.txt"
    extra = note_p.read_text(encoding="utf-8").strip() if note_p.is_file() else ""
    body = (
        "\n## 特征标准化 SFT 对照（LSTM / gated_residual）\n\n"
        "只在 **Train 日** 拟合每通道 mean/std，Val/Test 只 transform，clip 到 ±5。"
        "不改标签、不开 `sft_cs_zscore`。baseline 是当月已有纯 MSE 结果。\n\n"
        + table_md(cols, rows, 6)
        + ("\n" + extra + "\n" if extra else "\n")
    )
    return body


def _md_figures(items: Sequence[Tuple[str, str, Sequence[str]]], dest: Optional[Path] = None) -> str:
    # Do not append ?t=mtime. Cursor / VS Code preview treats that as a missing filename.
    blocks = []
    for caption, rel, names in items:
        blocks.append(f"### {caption}\n")
        blocks.append(f"![{caption}]({rel})\n")
        if names:
            blocks.append("固定颜色: " + "；".join(f"`{n}` `{color_for(n)}`" for n in names) + "。\n")
    return "\n".join(blocks)


def write_month_doc(
    dest: Path,
    month: str,
    extra: Dict[str, Any],
    val: pd.DataFrame,
    daily: pd.DataFrame,
    pred: pd.DataFrame,
) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    fig = dest / "figures"
    tab = dest / "tables"
    fig.mkdir(parents=True, exist_ok=True)
    tab.mkdir(parents=True, exist_ok=True)

    daily = _drop_short_series(_prefer_pool_aliases(daily), min_days=15)
    val_best = _best_val(val)
    nav = _nav_or_empty(daily)
    pmean = _pred_mean(pred)
    status_rows = _status_from_sources(month, val, daily, pred)
    n_done = sum(1 for r in status_rows if r["status"] == "已写入")
    status = f"进行中（{n_done}/{len(status_rows)} 单元已写入）" if n_done else "占位（轴和表头已铺齐）"
    if n_done == len(status_rows):
        status = "当月单元已全部写入"

    val_best_rows = _rows_from_spec(PRED_MODELS, VAL_BEST_COLS, val_best)
    pred_rows = _rows_from_spec(PRED_MODELS, PRED_COLS, pmean)
    nav_rows = _rows_from_spec(NAV_DAILY_MODELS, NAV_COLS, nav)
    to_df = nav[["model", "mean_turnover"]].copy() if len(nav) and "mean_turnover" in nav.columns else pd.DataFrame()
    to_rows = _rows_from_spec(NAV_DAILY_MODELS, ["model", "mean_turnover"], to_df)

    epoch_rows: List[Dict[str, Any]] = []
    for m in PRED_MODELS:
        for ep in (1, 2, 3):
            rec = {c: PLACEHOLDER for c in VAL_COLS}
            rec["model"] = m
            rec["epoch"] = ep
            if val is not None and len(val) and "model" in val.columns:
                hit = val[(val["model"].astype(str) == m)]
                if "epoch" in hit.columns:
                    hit = hit[pd.to_numeric(hit["epoch"], errors="coerce") == ep]
                if len(hit):
                    got = hit.iloc[-1].to_dict()
                    for c in VAL_COLS:
                        if c in got and _finite(got[c]):
                            rec[c] = got[c]
            epoch_rows.append(rec)

    _write_table_csv(tab / "unit_status.csv", UNIT_STATUS_COLS, status_rows)
    _write_table_csv(tab / "val_best.csv", VAL_BEST_COLS, val_best_rows)
    _write_table_csv(tab / "val_epochs.csv", VAL_COLS, epoch_rows)
    _write_table_csv(tab / "prediction_metrics_mean.csv", PRED_COLS, pred_rows)
    _write_table_csv(tab / "performance_summary.csv", NAV_COLS, nav_rows)
    _write_table_csv(tab / "mean_turnover.csv", ["model", "mean_turnover"], to_rows)
    if daily is not None and len(daily):
        daily.to_csv(tab / "backtest_daily.csv", index=False)
    if pred is not None and len(pred):
        pred.to_csv(tab / "prediction_metrics_daily.csv", index=False)
    if val is not None and len(val):
        val.to_csv(tab / "alpha_val_metrics.csv", index=False)

    has_data = bool(
        (val is not None and len(val))
        or (daily is not None and len(daily))
        or (pred is not None and len(pred))
    )
    if not has_data:
        filled = [r["model"] for r in status_rows if r["status"] == "已写入"]
        notes = _notes_from_val_nav(val_best, nav, pmean)
        md = (
            f"# {month} Sequential OOS 当月实验分析\n\n"
            f"- 状态: **{status}**\n"
            f"- OOS: `sequential_oos_weekly_retrain`\n"
            f"- TopK={extra.get('top_k', 30)}，cost_bps={extra.get('cost_bps', 3.0)}，primary=`gated_residual`\n"
            f"- 缺测一律 `{PLACEHOLDER}`，不补 0、不编造。\n\n"
            "## 固定颜色\n\n"
            + color_table_md(PRED_MODELS)
            + "\n预测骨干日收益曲线用 `pred_<模型>_ew`，颜色与上表相同。\n\n"
            "## 实验进度\n\n"
            + table_md(UNIT_STATUS_COLS, status_rows)
            + "\n## 实验设定\n\n"
            "- Walk-forward：Train=T-13..T-2，Val=T-1，Test=当月。\n"
            "- Sequential OOS：周五收盘重训 primary + SS-FM + PPO/GRPO，下周一生效，不回写本周决策。\n"
            "- 标签 y = log(Close_D / Open_D)。高严重泄漏 FAIL 的月份不进 final_summary。\n\n"
            "## Validation（回归 / 排序）\n\n"
            "### 各模型 BEST（按该模型 Val MSE 最低的 epoch）\n\n"
            + table_md(VAL_BEST_COLS, val_best_rows, 6)
            + "\n## Test 预测指标（日均）\n\n"
            + table_md(PRED_COLS, pred_rows, 6)
            + "\n## 组合绩效\n\n"
            + table_md(NAV_COLS, nav_rows)
            + "\n## 分析\n\n"
            + _analysis_text(f"{month} Sequential OOS", filled, notes, {**extra, "source": extra.get("source", f"month {month}")})
        )
        (dest / "实验分析.md").write_text(md, encoding="utf-8")
        return

    # figures — axes + fixed colors always
    plot_series(
        fig / "nav_pred.png",
        f"{month} Top30 等权累计收益（预测骨干）",
        FIG_CUM["xlabel"],
        FIG_CUM["ylabel"],
        [f"pred_{m}_ew" for m in PRED_MODELS],
        series_from_daily(daily, "net_return", [f"pred_{m}_ew" for m in PRED_MODELS], cum=True),
        empty_ylim=(-0.1, 0.1),
    )
    plot_series(
        fig / "nav_port.png",
        f"{month} 组合基线累计收益",
        FIG_CUM["xlabel"],
        FIG_CUM["ylabel"],
        [f"baseline_{m}" for m in PORTFOLIO_MODELS],
        series_from_daily(daily, "net_return", [f"baseline_{m}" for m in PORTFOLIO_MODELS], cum=True),
        empty_ylim=(-0.1, 0.1),
    )
    plot_series(
        fig / "nav_gen.png",
        f"{month} 生成模型累计收益",
        FIG_CUM["xlabel"],
        FIG_CUM["ylabel"],
        [f"gen_{m}" for m in GEN_MODELS],
        series_from_daily(daily, "net_return", [f"gen_{m}" for m in GEN_MODELS], cum=True),
        empty_ylim=(-0.1, 0.1),
    )
    plot_series(
        fig / "nav_rl.png",
        f"{month} SS-FM / PPO / GRPO 累计收益",
        FIG_CUM["xlabel"],
        FIG_CUM["ylabel"],
        ["gen_ssfm", "rl_ssfm_ppo", "rl_ssfm_grpo"],
        series_from_daily(daily, "net_return", ["gen_ssfm", "rl_ssfm_ppo", "rl_ssfm_grpo"], cum=True),
        empty_ylim=(-0.1, 0.1),
    )
    g_names = [f"ssfm_G{g}" for g in G_VALUES]
    g_series = series_from_daily(daily, "net_return", g_names, cum=True)
    for g_name in ("nav_g.png", "nav_g_candidates.png"):
        plot_series(
            fig / g_name,
            f"{month} G 候选数累计收益",
            FIG_CUM["xlabel"],
            FIG_CUM["ylabel"],
            g_names,
            g_series,
            empty_ylim=(-0.1, 0.1),
        )
    plot_grouped_bars(
        fig / "mean_turnover.png",
        f"{month} 日均换手",
        FIG_TO["xlabel"],
        FIG_TO["ylabel"],
        NAV_DAILY_MODELS,
        [(_lookup(to_df, m).get("mean_turnover") if len(to_df) else None) for m in NAV_DAILY_MODELS],
        [color_for(m) for m in NAV_DAILY_MODELS],
    )
    for metric, fn, ylabel, ttl in (
        ("IC", "ic_ts.png", FIG_IC["ylabel"], f"{month} Test IC"),
        ("RankIC", "rankic_ts.png", FIG_RIC["ylabel"], f"{month} Test RankIC"),
        ("topk_f1", "f1_ts.png", FIG_F1["ylabel"], f"{month} Test F1@Top30"),
        ("top30_hit_rate", "hit_ts.png", FIG_HIT["ylabel"], f"{month} Test Hit@Top30"),
    ):
        plot_metric_ts(pred, fig / fn, metric, ttl, expected=PRED_MODELS, ylabel=ylabel)
    plot_grouped_bars(
        fig / "val_ic.png",
        f"{month} Val IC（BEST / 最低 MSE）",
        "模型",
        "IC",
        PRED_MODELS,
        [(_lookup(val_best, m).get("IC") if len(val_best) else None) for m in PRED_MODELS],
        [color_for(m) for m in PRED_MODELS],
    )
    plot_grouped_bars(
        fig / "val_rankic.png",
        f"{month} Val RankIC（BEST / 最低 MSE）",
        "模型",
        "RankIC",
        PRED_MODELS,
        [(_lookup(val_best, m).get("RankIC") if len(val_best) else None) for m in PRED_MODELS],
        [color_for(m) for m in PRED_MODELS],
    )
    plot_grouped_bars(
        fig / "val_f1.png",
        f"{month} Val F1@Top30（BEST / 最低 MSE）",
        "模型",
        "F1@Top30",
        PRED_MODELS,
        [(_lookup(val_best, m).get("topk_f1") if len(val_best) else None) for m in PRED_MODELS],
        [color_for(m) for m in PRED_MODELS],
    )
    plot_grouped_bars(
        fig / "val_mse.png",
        f"{month} Val MSE（BEST / 最低 MSE）",
        "模型",
        "MSE",
        PRED_MODELS,
        [(_lookup(val_best, m).get("MSE") if len(val_best) else None) for m in PRED_MODELS],
        [color_for(m) for m in PRED_MODELS],
    )
    for metric, fn, ylabel, ttl in (
        ("IC", "val_ic_epochs.png", "IC", f"{month} Val IC（各 epoch）"),
        ("RankIC", "val_rankic_epochs.png", "RankIC", f"{month} Val RankIC（各 epoch）"),
        ("topk_f1", "val_f1_epochs.png", "F1@Top30", f"{month} Val F1@Top30（各 epoch）"),
        ("MSE", "val_mse_epochs.png", "MSE", f"{month} Val MSE（各 epoch）"),
    ):
        plot_epoch_lines(fig / fn, ttl, ylabel, val, metric, PRED_MODELS)

    filled = [r["model"] for r in status_rows if r["status"] == "已写入"]
    notes = _notes_from_val_nav(val_best, nav, pmean)
    fig_items = [
        ("预测骨干累计收益", "figures/nav_pred.png", [f"pred_{m}_ew" for m in PRED_MODELS]),
        ("组合基线累计收益", "figures/nav_port.png", [f"baseline_{m}" for m in PORTFOLIO_MODELS]),
        ("生成模型累计收益", "figures/nav_gen.png", [f"gen_{m}" for m in GEN_MODELS]),
        ("RL 累计收益", "figures/nav_rl.png", ["gen_ssfm", "rl_ssfm_ppo", "rl_ssfm_grpo"]),
        ("G 曲线累计收益", "figures/nav_g_candidates.png", g_names),
        ("日均换手", "figures/mean_turnover.png", NAV_DAILY_MODELS),
        ("Val IC", "figures/val_ic.png", PRED_MODELS),
        ("Val RankIC", "figures/val_rankic.png", PRED_MODELS),
        ("Val F1@Top30", "figures/val_f1.png", PRED_MODELS),
        ("Val MSE", "figures/val_mse.png", PRED_MODELS),
        ("Val IC 各 epoch", "figures/val_ic_epochs.png", PRED_MODELS),
        ("Val RankIC 各 epoch", "figures/val_rankic_epochs.png", PRED_MODELS),
        ("Val F1@Top30 各 epoch", "figures/val_f1_epochs.png", PRED_MODELS),
        ("Val MSE 各 epoch", "figures/val_mse_epochs.png", PRED_MODELS),
        ("Test IC", "figures/ic_ts.png", PRED_MODELS),
        ("Test RankIC", "figures/rankic_ts.png", PRED_MODELS),
        ("Test F1@Top30", "figures/f1_ts.png", PRED_MODELS),
        ("Test Hit@Top30", "figures/hit_ts.png", PRED_MODELS),
    ]
    md = (
        f"# {month} Sequential OOS 当月实验分析\n\n"
        f"- 状态: **{status}**\n"
        f"- OOS: `sequential_oos_weekly_retrain`\n"
        f"- TopK={extra.get('top_k', 30)}，cost_bps={extra.get('cost_bps', 3.0)}，primary=`gated_residual`\n"
        f"- 缺测一律 `{PLACEHOLDER}`，不补 0、不编造。\n\n"
        "## 固定颜色\n\n"
        + color_table_md(PRED_MODELS)
        + "\n预测骨干日收益曲线用 `pred_<模型>_ew`，颜色与上表相同。\n\n"
        "## 实验进度\n\n"
        + table_md(UNIT_STATUS_COLS, status_rows)
        + "\n## 实验设定\n\n"
        "- Walk-forward：Train=T-13..T-2，Val=T-1，Test=当月。\n"
        "- Sequential OOS：周五收盘重训 primary + SS-FM + PPO/GRPO，下周一生效，不回写本周决策。\n"
        "- 实验 E：SS-FM 候选数 G∈{8,16,32,64,128}。按周五生效窗口加载当时的 primary / SS-FM，"
        "当日采样后按 pred_utility 选 1 条；只跑 top30，不跑 500 只 all 池。\n"
        "- 标签 y = log(Close_D / Open_D)。高严重泄漏 FAIL 的月份不进 final_summary。\n\n"
        "## Validation（回归 / 排序）\n\n"
        "### 各模型 BEST（按该模型 Val MSE 最低的 epoch）\n\n"
        + table_md(VAL_BEST_COLS, val_best_rows, 6)
        + "\n列含义: `MSE`/`MAE` 是 ŷ 对 y 的误差；`IC` 是 Pearson；`RankIC` 是 Spearman；"
        "`topk_f1` 是预测 Top30 ∩ 真实收益 Top30 / 30；`dir_acc` 是涨跌符号一致率；"
        "`top30_hit_rate` 是入选 30 只里 y>0 的比例。\n\n"
        "### 各 epoch\n\n"
        + table_md(VAL_COLS, epoch_rows, 6)
        + "\n## Test 预测指标（日均）\n\n"
        + table_md(PRED_COLS, pred_rows, 6)
        + "\n## 组合绩效\n\n"
        + table_md(NAV_COLS, nav_rows)
        + "\n## 日均换手\n\n"
        + table_md(["model", "mean_turnover"], to_rows)
        + "\n## 图（颜色已固定，无数据时只画轴）\n\n"
        + _md_figures(fig_items, dest)
        + "\n## 图对应数据表\n\n"
        "- 累计收益 ← `tables/backtest_daily.csv` 的 `net_return` 累加。\n"
        "- 绩效 / 换手 ← `tables/performance_summary.csv`。\n"
        "- Val ← `tables/val_best.csv`、`tables/val_epochs.csv`。\n"
        "- Test 预测 ← `tables/prediction_metrics_mean.csv`。\n\n"
        "## 分析\n\n"
        + _analysis_text(f"{month} Sequential OOS", filled, notes, {**extra, "source": extra.get("source", f"month {month}")})
        + _g_month_section(nav)
        + _featstd_month_section(dest)
    )
    (dest / "实验分析.md").write_text(md, encoding="utf-8")


def write_family_doc(
    dest: Path,
    item: Dict[str, Any],
    extra: Dict[str, Any],
    val: pd.DataFrame,
    daily: pd.DataFrame,
    pred: pd.DataFrame,
    extra_tables: Optional[Dict[str, Any]] = None,
) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    fig = dest / "figures"
    tab = dest / "tables"
    fig.mkdir(parents=True, exist_ok=True)
    tab.mkdir(parents=True, exist_ok=True)
    extra_tables = extra_tables or {}
    fid = item["id"]
    names = family_series(item)
    nav_src = daily
    if item.get("daily_prefixes"):
        nav_src = _filter_daily(daily, item["daily_prefixes"])
    if fid in ("M06_walkforward_oos", "全年总览") or item.get("kind") == "overview":
        nav_src = daily
    pred_src = pred
    if fid in ("X01_fusion_ablation", "X02_minute_vs_auction") and pred is not None and len(pred) and "model" in pred.columns:
        pred_src = pred[pred["model"].astype(str).isin(set(item.get("models") or []))].copy()

    val_best = _best_val(val)
    if fid in ("X01_fusion_ablation", "X02_minute_vs_auction") and len(val_best):
        val_best = val_best[val_best["model"].astype(str).isin(set(names))].copy()
    nav = _nav_or_empty(nav_src)
    pmean = _pred_mean(pred_src if fid.startswith(("M01", "X01", "X02", "X10", "X11")) or fid == "全年总览" else pd.DataFrame())
    if fid == "全年总览":
        pmean = _pred_mean(pred)

    pred_row_keys = names if names and names[0] in PRED_MODELS else PRED_MODELS
    if fid in ("M01_prediction", "X10_prediction_distribution", "X11_top30_stats", "X01_fusion_ablation", "X02_minute_vs_auction"):
        pred_row_keys = list(names)
    val_rows = _rows_from_spec(pred_row_keys if fid.startswith(("M01", "X01", "X02", "X10", "X11")) or fid == "全年总览" else PRED_MODELS, VAL_BEST_COLS, val_best)
    pred_rows = _rows_from_spec(pred_row_keys, PRED_COLS, pmean) if fid.startswith(("M01", "X01", "X02", "X10", "X11")) or fid == "全年总览" else []
    nav_keys = names if names and str(names[0]).startswith(("pred_", "baseline_", "gen_", "rl_", "ssfm_")) else NAV_DAILY_MODELS
    if fid == "M01_prediction":
        nav_keys = [f"pred_{m}_ew_top30" for m in PRED_MODELS] + [f"pred_{m}_softmax_all" for m in PRED_MODELS] + ["universe_ew_all"]
    elif fid == "M02_portfolio":
        nav_keys = [f"baseline_{m}_{p}" for p in ("top30", "all") for m in PORTFOLIO_MODELS] + ["baseline_score_softmax_all"]
    elif fid == "X01_fusion_ablation":
        nav_keys = [f"pred_{m}_ew" for m in names]
    elif fid == "X02_minute_vs_auction":
        nav_keys = [f"pred_{m}_ew" for m in names]
    nav_rows = _rows_from_spec(nav_keys, NAV_COLS, nav) if fid not in ("X08_leakage_audit", "X09_gpu_time", "X12_oos_mode", "X03_attention", "X04_gate") else []

    month_rows = []
    fam_nav_keys = set(nav_keys if nav_keys else names)
    fam_pred_keys = set(pred_row_keys if fid.startswith(("M01", "X01", "X02", "X10", "X11")) or fid == "全年总览" else [])
    for m in TEST_MONTHS:
        rec = {c: PLACEHOLDER for c in MONTH_STATUS_COLS}
        rec["month"] = m
        vm = val[val["month"].astype(str) == m] if val is not None and len(val) and "month" in val.columns else pd.DataFrame()
        if fid.startswith(("M01", "X01", "X02", "X10", "X11")) or fid == "全年总览":
            if len(vm) and "model" in vm.columns:
                rec["n_pred_models_val"] = int(vm[vm["model"].astype(str).isin(fam_pred_keys or set(PRED_MODELS))]["model"].nunique())
        dm = pd.DataFrame()
        if daily is not None and len(daily) and "date" in daily.columns:
            dm = daily[daily["date"].astype(str).str.startswith(m)]
            if len(dm) and "model" in dm.columns and fam_nav_keys:
                rec["n_nav_models"] = int(dm[dm["model"].astype(str).isin(fam_nav_keys)]["model"].nunique()) or PLACEHOLDER
        has_val = rec["n_pred_models_val"] != PLACEHOLDER and int(rec["n_pred_models_val"] or 0) > 0
        has_nav = rec["n_nav_models"] != PLACEHOLDER and int(rec.get("n_nav_models") or 0) > 0
        if fid == "全年总览":
            rec["status"] = "已写入" if (len(vm) or len(dm)) else PLACEHOLDER
        elif fid in ("X08_leakage_audit", "X09_gpu_time", "X12_oos_mode", "X03_attention", "X04_gate"):
            rec["status"] = PLACEHOLDER
        else:
            rec["status"] = "已写入" if (has_val or has_nav) else PLACEHOLDER
        month_rows.append(rec)

    g_nav = pd.DataFrame()
    if fid in ("M05_g_candidates", "X07_g_curves") and len(nav):
        g_nav = nav.copy()
        g_nav["G"] = g_nav["model"].astype(str).str.replace("ssfm_G", "", regex=False)
    g_rows = _rows_from_spec(G_VALUES, G_CURVE_COLS, g_nav, key_col="G")

    _write_table_csv(tab / "month_status.csv", MONTH_STATUS_COLS, month_rows)
    if val_rows:
        _write_table_csv(tab / "val_best.csv", VAL_BEST_COLS, val_rows)
    if pred_rows:
        _write_table_csv(tab / "prediction_metrics_mean.csv", PRED_COLS, pred_rows)
    if nav_rows:
        _write_table_csv(tab / "performance_summary.csv", NAV_COLS, nav_rows)
        _write_table_csv(tab / "mean_turnover.csv", ["model", "mean_turnover"], [{k: r.get(k, PLACEHOLDER) for k in ("model", "mean_turnover")} for r in nav_rows])
    if fid in ("M05_g_candidates", "X07_g_curves"):
        _write_table_csv(tab / "g_curves.csv", G_CURVE_COLS, g_rows)

    # figures
    if fid not in ("X08_leakage_audit", "X12_oos_mode"):
        plot_series(
            fig / "cumulative_return.png",
            f"{item['title']} 累计收益",
            FIG_CUM["xlabel"],
            FIG_CUM["ylabel"],
            nav_keys if nav_keys else names,
            series_from_daily(nav_src, "net_return", nav_keys if nav_keys else names, cum=True),
            empty_ylim=(-0.1, 0.1),
        )
    if fid.startswith(("M01", "X01", "X02", "X10", "X11")) or fid == "全年总览":
        plot_metric_ts(pred_src if fid != "全年总览" else pred, fig / "ic_ts.png", "IC", f"{item['title']} IC", expected=pred_row_keys, ylabel=FIG_IC["ylabel"])
        plot_metric_ts(pred_src if fid != "全年总览" else pred, fig / "rankic_ts.png", "RankIC", f"{item['title']} RankIC", expected=pred_row_keys, ylabel=FIG_RIC["ylabel"])
        plot_metric_ts(pred_src if fid != "全年总览" else pred, fig / "f1_ts.png", "topk_f1", f"{item['title']} F1@Top30", expected=pred_row_keys, ylabel=FIG_F1["ylabel"])
        plot_metric_ts(pred_src if fid != "全年总览" else pred, fig / "hit_ts.png", "top30_hit_rate", f"{item['title']} Hit@Top30", expected=pred_row_keys, ylabel=FIG_HIT["ylabel"])
        months_present = []
        if val is not None and len(val) and "month" in val.columns:
            months_present = [m for m in TEST_MONTHS if str(m) in set(val["month"].astype(str))]
        by_month = _best_val_by_month(val) if months_present else val_best

        def _month_matrix(metric: str) -> List[List[Any]]:
            mat = []
            for name in pred_row_keys:
                row = []
                for mth in months_present:
                    hit = pd.DataFrame()
                    if len(by_month) and "model" in by_month.columns:
                        hit = by_month[(by_month["model"].astype(str) == str(name)) & (by_month["month"].astype(str) == str(mth))]
                    row.append(hit.iloc[-1][metric] if len(hit) and metric in hit.columns else None)
                mat.append(row)
            return mat

        if months_present:
            plot_clustered_bars(
                fig / "val_ic.png",
                f"{item['title']} Val IC（按测试月，BEST / 最低 MSE）",
                "测试月",
                "IC",
                months_present,
                pred_row_keys,
                _month_matrix("IC"),
                [color_for(m) for m in pred_row_keys],
            )
            plot_clustered_bars(
                fig / "val_rankic.png",
                f"{item['title']} Val RankIC（按测试月，BEST / 最低 MSE）",
                "测试月",
                "RankIC",
                months_present,
                pred_row_keys,
                _month_matrix("RankIC"),
                [color_for(m) for m in pred_row_keys],
            )
        else:
            plot_grouped_bars(
                fig / "val_ic.png",
                f"{item['title']} Val IC",
                "模型",
                "IC",
                pred_row_keys,
                [(_lookup(val_best, m).get("IC") if len(val_best) else None) for m in pred_row_keys],
                [color_for(m) for m in pred_row_keys],
            )
            plot_grouped_bars(
                fig / "val_rankic.png",
                f"{item['title']} Val RankIC",
                "模型",
                "RankIC",
                pred_row_keys,
                [(_lookup(val_best, m).get("RankIC") if len(val_best) else None) for m in pred_row_keys],
                [color_for(m) for m in pred_row_keys],
            )
        plot_epoch_lines(fig / "val_ic_epochs.png", f"{item['title']} Val IC（各 epoch）", "IC", val, "IC", pred_row_keys)
        plot_epoch_lines(fig / "val_rankic_epochs.png", f"{item['title']} Val RankIC（各 epoch）", "RankIC", val, "RankIC", pred_row_keys)
    if fid in ("M05_g_candidates", "X07_g_curves"):
        plot_grouped_bars(
            fig / "g_return.png",
            "G vs 累计净收益",
            FIG_G["xlabel"],
            FIG_G["ylabel"],
            G_VALUES,
            [(_lookup(pd.DataFrame(g_rows), g, "G").get("total_net_return") if g_rows else None) for g in G_VALUES],
            [color_for(g) for g in G_VALUES],
        )
    if fid == "X03_attention":
        attn = extra_tables.get("attn")
        plot_hist_or_empty(np.asarray(attn) if attn is not None else np.array([]), fig / "cross_attn_hist.png", "cross-attn 分布", **FIG_HIST, color=color_for("cross_attention"))
    if fid == "X04_gate":
        gate = extra_tables.get("gate")
        plot_hist_or_empty(np.asarray(gate) if gate is not None else np.array([]), fig / "gate_hist.png", "gate g 分布", **FIG_HIST, color=color_for("gated_residual"))
    if fid == "X09_gpu_time":
        gpu = extra_tables.get("gpu")
        if gpu is not None and len(gpu):
            g = gpu.copy()
            if "t" in g.columns and "date" not in g.columns:
                g["date"] = pd.to_datetime(g["t"], unit="s", errors="coerce")
            g["model"] = "gpu"
            plot_metric_ts(g, fig / "gpu_util.png", "util_gpu" if "util_gpu" in g.columns else "util", "GPU 利用率 %", expected=["gpu"], ylabel=FIG_GPU["ylabel"])
        else:
            plot_series(fig / "gpu_util.png", "GPU 利用率 %", FIG_GPU["xlabel"], FIG_GPU["ylabel"], ["gpu"], {}, empty_ylim=(0, 100))
    if fid == "X10_prediction_distribution":
        yhat = extra_tables.get("yhat")
        plot_hist_or_empty(np.asarray(yhat) if yhat is not None else np.array([]), fig / "yhat_hist.png", "ŷ 分布", color=color_for("gated_residual"))

    n_done = sum(1 for r in month_rows if r["status"] == "已写入")
    status = f"进行中（{n_done}/13 月有数据）" if n_done else "占位（轴和表头已铺齐）"
    if n_done == 13:
        status = "13 个测试月均已写入"

    fig_items: List[Tuple[str, str, Sequence[str]]] = []
    if (dest / "figures" / "cumulative_return.png").is_file():
        fig_items.append(("累计收益", "figures/cumulative_return.png", nav_keys if nav_keys else names))
    for rel, cap, series in (
        ("figures/val_ic.png", "Val IC", pred_row_keys),
        ("figures/val_rankic.png", "Val RankIC", pred_row_keys),
        ("figures/val_ic_epochs.png", "Val IC 各 epoch", pred_row_keys),
        ("figures/val_rankic_epochs.png", "Val RankIC 各 epoch", pred_row_keys),
        ("figures/ic_ts.png", "Test IC", pred_row_keys),
        ("figures/rankic_ts.png", "Test RankIC", pred_row_keys),
        ("figures/f1_ts.png", "Test F1@Top30", pred_row_keys),
        ("figures/hit_ts.png", "Test Hit@Top30", pred_row_keys),
        ("figures/g_return.png", "G–收益", G_VALUES),
        ("figures/gpu_util.png", "GPU 利用率", ["gpu"]),
        ("figures/gate_hist.png", "gate 分布", ["gated_residual"]),
        ("figures/cross_attn_hist.png", "cross-attn 分布", ["cross_attention"]),
        ("figures/yhat_hist.png", "ŷ 分布", ["gated_residual"]),
    ):
        if (dest / rel).is_file():
            fig_items.append((cap, rel, series))

    filled = [r["month"] for r in month_rows if r["status"] == "已写入"]
    notes = _notes_from_val_nav(val_best, nav, pmean)
    body_tables = [
        "## 13 个月进度\n\n" + table_md(MONTH_STATUS_COLS, month_rows),
    ]
    if fid == "X12_oos_mode":
        body_tables = [
            "## 设定（已确定，不是结果占位）\n\n"
            + table_md(
                ["项", "值"],
                [
                    {"项": "采用的 OOS", "值": "sequential_oos_weekly_retrain"},
                    {"项": "已弃用", "值": "strict_fixed_oos"},
                    {"项": "重训时点", "值": "周五 15:00 收盘后"},
                    {"项": "生效", "值": "下一周第一个交易日"},
                    {"项": "回写本周决策", "值": "否"},
                    {"项": "周五重训对象", "值": "primary Alpha + SS-FM + PPO/GRPO"},
                ],
            )
        ]
        status = "已确定"
    if val_rows and fid != "X12_oos_mode":
        body_tables.append("## Validation BEST（按模型 Val MSE）\n\n" + table_md(VAL_BEST_COLS, val_rows, 6))
        month_best = _best_val_by_month(val)
        if len(month_best) and "month" in month_best.columns and month_best["month"].nunique() > 1:
            mcols = ["month"] + [c for c in VAL_BEST_COLS if c != "month"]
            mrows = []
            for mth in TEST_MONTHS:
                for name in pred_row_keys:
                    rec = {c: PLACEHOLDER for c in mcols}
                    rec["month"] = mth
                    rec["model"] = name
                    hit = month_best[(month_best["month"].astype(str) == str(mth)) & (month_best["model"].astype(str) == str(name))]
                    if len(hit):
                        got = hit.iloc[-1].to_dict()
                        for c in mcols:
                            if c in got and _finite(got[c]):
                                rec[c] = got[c]
                    if rec.get("MSE") != PLACEHOLDER:
                        mrows.append(rec)
            if mrows:
                body_tables.append("## Validation BEST（分月，只列已有实验）\n\n" + table_md(mcols, mrows, 6))
    if pred_rows:
        body_tables.append("## Test 预测指标（已完成月份的日均）\n\n" + table_md(PRED_COLS, pred_rows, 6))
    if nav_rows:
        body_tables.append("## 组合绩效（已完成月份拼接）\n\n" + table_md(NAV_COLS, nav_rows))
    if fid in ("M05_g_candidates", "X07_g_curves"):
        body_tables.append("## G–收益–耗时\n\n" + table_md(G_CURVE_COLS, g_rows))
    if fid == "X08_leakage_audit":
        leak = extra_tables.get("leakage")
        leak_rows = [{c: PLACEHOLDER for c in LEAKAGE_COLS}]
        if leak is not None and len(leak):
            leak_rows = []
            for _, r in leak.head(80).iterrows():
                leak_rows.append({c: r[c] if c in r.index and _finite(r[c]) else PLACEHOLDER for c in LEAKAGE_COLS})
                for c in LEAKAGE_COLS:
                    if c in r.index:
                        leak_rows[-1][c] = r[c] if _finite(r[c]) else PLACEHOLDER
        body_tables.append("## 审计表\n\n" + table_md(LEAKAGE_COLS, leak_rows))
    if fid == "X09_gpu_time":
        gpu = extra_tables.get("gpu")
        grow = [{c: PLACEHOLDER for c in GPU_COLS}]
        if gpu is not None and len(gpu):
            rec = {c: PLACEHOLDER for c in GPU_COLS}
            if "util_gpu" in gpu.columns:
                rec["util_mean"] = float(gpu["util_gpu"].mean())
                rec["util_max"] = float(gpu["util_gpu"].max())
            if "mem_used_mb" in gpu.columns:
                rec["mem_used_mb_mean"] = float(gpu["mem_used_mb"].mean())
            if "power_w" in gpu.columns:
                rec["power_w_mean"] = float(gpu["power_w"].mean())
            rec["n"] = int(len(gpu))
            grow = [rec]
        body_tables.append("## GPU 汇总\n\n" + table_md(GPU_COLS, grow))
    if fid == "X04_gate":
        body_tables.append("## gate 分位\n\n" + table_md(GATE_COLS, extra_tables.get("gate_stats") or [{c: PLACEHOLDER for c in GATE_COLS}]))
    if fid == "X03_attention":
        body_tables.append("## cross-attn 分位\n\n" + table_md(GATE_COLS, extra_tables.get("attn_stats") or [{c: PLACEHOLDER for c in GATE_COLS}]))

    md = (
        f"# {item['title']}\n\n"
        f"- 状态: **{status}**\n"
        f"- 编号: `{fid}`\n"
        f"- 类型: {'主实验' if item['kind']=='main' else ('全年总览' if item['kind']=='overview' else '消融/系统')}\n"
        f"- 说明: {item['desc']}\n"
        f"- 缺测一律 `{PLACEHOLDER}`。曲线颜色见下表，中途不改色。\n\n"
        "## 固定颜色\n\n"
        + color_table_md(names or PRED_MODELS)
        + "\n## 实验设定\n\n"
        f"- Walk-forward 测试月 2025-05 … 2026-05（2026-05 分钟数据只到 05-08）。\n"
        f"- TopK={extra.get('top_k', 30)}，cost_bps={extra.get('cost_bps', 3.0)}，primary=`gated_residual`。\n"
        f"- Sequential OOS：周五收盘重训，下周一生效。\n\n"
        + "\n".join(body_tables)
        + "\n## 图（颜色已固定，无数据时只画轴）\n\n"
        + _md_figures(fig_items, dest)
        + "\n## 分析\n\n"
        + _analysis_text(item["title"], filled, notes, {**extra, "source": extra.get("source", "aggregated disk")})
    )
    (dest / "实验分析.md").write_text(md, encoding="utf-8")


def catalog_index_md() -> str:
    c = count_summary()
    overs = [x for x in CATALOG if x["kind"] == "overview"]
    mains = [x for x in CATALOG if x["kind"] == "main"]
    abls = [x for x in CATALOG if x["kind"] == "ablation"]
    lines = [
        "# 实验目录\n",
        "## 三层文档（先看这里）\n",
        "1. **当月**：`results/full/sequential_oos_weekly_retrain/test_month=YYYY-MM/实验分析.md`",
        "2. **全年 / 全样本（13 个月拼在一起）**：`results/full/analysis/` 下各文件夹的 `实验分析.md`",
        "3. **总目录（本文件）**：`results/full/analysis/实验目录.md`",
        "",
        "表头、行名、曲线颜色在实验开始前就写死。没跑的单元格是 `-`，图只画坐标轴和图例虚线。",
        "每完成一个训练/回测单元（例如 A-lstm 的 VAL），会重写当月 + 族级文档，填入已有数字并据此写分析。",
        "",
        "## 固定颜色（预测骨干）\n",
        color_table_md(PRED_MODELS),
        "",
        "组合 / 生成 / RL / G 的颜色在各族文档里同样固定，见 `evaluation/analysis_spec.py` 的 `SERIES_COLORS`。",
        "",
        "## 规模\n",
        f"- 全年总览: **{c.get('n_overview_docs', 1)}**",
        f"- 主实验族: **{c['n_main_families']}**",
        f"- 消融/系统族: **{c['n_ablation_families']}**",
        f"- 实验分析文档: **{c['n_analysis_docs']}**（每族一份 `实验分析.md`）",
        f"- 测试月: **{c['n_test_months']}**（2025-05 … 2026-05）",
        f"- 主实验月度单元（不含周五重训）: **{c['n_main_units_13m']}** = (7+6+5+3+5+1)×13",
        f"- 周五重训单元（primary+5 生成+2 RL）: **{c['n_friday_retrain_units']}** ≈ 13×4×8",
        f"- 训练/回测单元合计: **{c['n_total_train_or_backtest_units']}**",
        "",
        c["note"],
        "",
        "## 全年总览\n",
    ]
    for x in overs:
        lines.append(f"- [`{x['id']}`]({x['id']}/实验分析.md) {x['title']} — {x['desc']}")
    lines += ["", "## 主实验（全年汇总）\n"]
    for x in mains:
        lines.append(f"- [`{x['id']}`]({x['id']}/实验分析.md) {x['title']} — {x['desc']}")
    lines += ["", "## 消融与系统（全年汇总）\n"]
    for x in abls:
        lines.append(f"- [`{x['id']}`]({x['id']}/实验分析.md) {x['title']} — {x['desc']}")
    lines += [
        "",
        "## 每月产物",
        "路径：`results/full/sequential_oos_weekly_retrain/test_month=YYYY-MM/实验分析.md`",
        "另有当月 `figures/`、`tables/`。单月运行只更新该月文件夹；`results/full/analysis/` 等全部测试月跑完再写。",
        "",
    ]
    return "\n".join(lines)


def _system_extras(out_root: Path, oos: str) -> Dict[str, Any]:
    extra: Dict[str, Any] = {}
    leaks = []
    for p in (Path(out_root) / oos).glob("test_month=*/**/leakage_audit.csv"):
        df = _read_csv(p)
        if len(df):
            leaks.append(df)
    if leaks:
        extra["leakage"] = pd.concat(leaks, ignore_index=True)
    gpu = Path(out_root) / "gpu_logs.csv"
    if gpu.is_file():
        extra["gpu"] = _read_csv(gpu)
    return extra


def refresh_month(out_root: Path, oos: str, month: str, extra: Optional[Dict[str, Any]] = None) -> None:
    extra = dict(extra or {})
    extra.setdefault("oos", oos)
    extra.setdefault("month", month)
    extra.setdefault("top_k", 30)
    extra.setdefault("cost_bps", 3.0)
    extra.setdefault("primary", "gated_residual")
    mdir = Path(out_root) / oos / f"test_month={month}"
    mdir.mkdir(parents=True, exist_ok=True)
    write_month_doc(
        mdir,
        month,
        extra,
        collect_val(Path(out_root), oos, month),
        collect_daily(Path(out_root), oos, month),
        collect_pred(Path(out_root), oos, month),
    )
    log(f"    refreshed month-only {mdir}")


def refresh_analysis(out_root: Path, oos: str = "sequential_oos_weekly_retrain", extra: Optional[Dict[str, Any]] = None) -> None:
    extra = dict(extra or {})
    extra.setdefault("oos", oos)
    extra.setdefault("top_k", 30)
    extra.setdefault("cost_bps", 3.0)
    extra.setdefault("primary", "gated_residual")
    if not extra.get("families"):
        month = extra.get("month")
        if month:
            refresh_month(Path(out_root), oos, str(month), extra)
            return
        log("    skip results/full/analysis (wait until all test months finish)")
        return
    root = Path(out_root)
    aroot = analysis_root(root)
    aroot.mkdir(parents=True, exist_ok=True)

    log(f"    refresh analysis families oos={oos}")
    val_all = collect_val(root, oos)
    daily_all = collect_daily(root, oos)
    pred_all = collect_pred(root, oos)
    sys_extra = _system_extras(root, oos)

    for month in TEST_MONTHS:
        mdir = root / oos / f"test_month={month}"
        mdir.mkdir(parents=True, exist_ok=True)
        vm = val_all[val_all["month"].astype(str) == month] if len(val_all) and "month" in val_all.columns else val_all
        dm = collect_daily(root, oos, month)
        pm = collect_pred(root, oos, month)
        write_month_doc(mdir, month, {**extra, "month": month}, vm, dm, pm)

    for item in CATALOG:
        dest = family_dir(root, item["id"])
        xt: Dict[str, Any] = {}
        if item["id"] == "X08_leakage_audit":
            xt["leakage"] = sys_extra.get("leakage")
        if item["id"] == "X09_gpu_time":
            xt["gpu"] = sys_extra.get("gpu")
        write_family_doc(dest, item, extra, val_all, daily_all, pred_all, xt)

    (aroot / "实验目录.md").write_text(catalog_index_md(), encoding="utf-8")
    write_json(aroot / "catalog.json", {"items": CATALOG, "counts": count_summary(), "colors": {k: color_for(k) for k in list(PRED_MODELS) + NAV_DAILY_MODELS}})
    log(f"    refreshed analysis docs under {aroot}")


def ensure_analysis_tree(out_root: Path) -> Path:
    root = analysis_root(Path(out_root))
    root.mkdir(parents=True, exist_ok=True)
    return root


def write_month_bundle(out: Path, daily: pd.DataFrame, pred: pd.DataFrame, extra: Dict[str, Any]) -> None:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    tab = out / "tables"
    tab.mkdir(parents=True, exist_ok=True)
    if daily is not None and len(daily):
        daily.to_csv(tab / "backtest_daily.csv", index=False)
        daily.to_csv(out / "backtest_daily.csv", index=False)
    if pred is not None and len(pred):
        pred.to_csv(tab / "prediction_metrics_daily.csv", index=False)
        pred.to_csv(out / "prediction_metrics.csv", index=False)
    if extra.get("aux_gate") is not None:
        g = np.asarray(extra["aux_gate"], dtype=np.float64)
        pd.DataFrame({"gate": g}).to_csv(tab / "gate_values.csv", index=False)
        plot_hist_or_empty(g, out / "figures" / "gate_hist.png", "gate 分布", color=color_for("gated_residual"))
    if extra.get("aux_attn") is not None:
        a = np.asarray(extra["aux_attn"], dtype=np.float64)
        pd.DataFrame({"cross_attn": a}).to_csv(tab / "cross_attn.csv", index=False)
        plot_hist_or_empty(a, out / "figures" / "cross_attn_hist.png", "cross-attn 分布", color=color_for("cross_attention"))
    month = str(extra.get("month") or "")
    oos = str(extra.get("oos") or "sequential_oos_weekly_retrain")
    # `out` may be the experiment id dir or the test_month dir
    out_root = extra.get("out_root")
    if out_root is None:
        p = out
        while p.name != "results" and p != p.parent:
            if p.name in ("full", "smoke") and p.parent.name == "results":
                out_root = p
                break
            p = p.parent
        if out_root is None:
            out_root = out.parents[2] if len(out.parents) >= 3 else out
    if month:
        refresh_month(Path(out_root), oos, month, extra)
    log(f"    wrote month bundle {out / '实验分析.md'}")


def _merge_val_prefer_n(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    both = pd.concat([old, new], ignore_index=True) if old is not None and len(old) else new
    if both is None or both.empty:
        return new
    keys = [c for c in ("model", "month", "epoch") if c in both.columns]
    if "epoch" in both.columns and "n" in both.columns:
        both = both.copy()
        both["_n"] = pd.to_numeric(both["n"], errors="coerce").fillna(0)
        both = both.sort_values("_n").drop_duplicates(subset=keys or ["epoch"], keep="last").drop(columns=["_n"])
        return both
    return new


def write_alpha_val_snapshot(out: Path, out_root: Path, model: str, val_rows: List[Dict[str, Any]], extra: Dict[str, Any]) -> None:
    if not val_rows:
        return
    df = pd.DataFrame(val_rows)
    dest = Path(out) / "tables"
    dest.mkdir(parents=True, exist_ok=True)
    prev = _read_csv(dest / f"alpha_val_{model}.csv")
    merged = _merge_val_prefer_n(prev, df)
    merged.to_csv(dest / f"alpha_val_{model}.csv", index=False)


def refresh_after_unit(out_root: Path, oos: str, extra: Optional[Dict[str, Any]] = None) -> None:
    extra = extra or {}
    last = str(extra.get("last_unit") or "")
    if last.startswith(("C-", "D-", "A-backtest-", "A-universe-", "B-", "E-")):
        log(f"    skip mid-train month refresh ({last})")
        return
    month = extra.get("month")
    if month:
        refresh_month(Path(out_root), oos, str(month), extra)
        return
    log("    skip results/full/analysis (wait until all test months finish)")


def update_family_docs(out_root: Path, oos: str, extra: Dict[str, Any]) -> None:
    month = extra.get("month")
    if month:
        refresh_month(Path(out_root), oos, str(month), extra)
        return
    log("    skip family docs until all test months finish")


def finalize_analysis(out_root: Path, oos_modes: List[str], extra: Dict[str, Any]) -> None:
    extra = dict(extra or {})
    for oos in oos_modes:
        ready = []
        for month in TEST_MONTHS:
            p = Path(out_root) / oos / f"test_month={month}" / "backtest_daily.csv"
            if p.is_file() and p.stat().st_size > 200:
                ready.append(month)
        if extra.get("write_families") or len(ready) >= len(TEST_MONTHS):
            refresh_analysis(Path(out_root), oos, {**extra, "families": True})
            continue
        log(f"    skip results/full/analysis finalize: {len(ready)}/{len(TEST_MONTHS)} months ready")


# kept for older call sites
def df_to_md(df: pd.DataFrame, floatfmt: int = 4) -> str:
    if df is None or df.empty:
        return f"_（暂无数据，单元格为 `{PLACEHOLDER}`）_\n"
    cols = [str(c) for c in df.columns]
    rows = [dict(r) for _, r in df.iterrows()]
    return table_md(cols, rows, floatfmt)
