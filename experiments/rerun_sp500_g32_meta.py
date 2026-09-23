#!/usr/bin/env python3
"""SP500 G=32 Meta-selection + RL ablation (freeze Alpha / FM / SS-FM).

1) FM & SS-FM: sample G=32, append teacher closed-form (MVO / MaxSharpe /
   Risk Parity), then pred_utility Meta selection.
2) Teachers: same-day closed-form baselines for curves.
3) RL: cand-PPO (SS-FM candidates as features) + GRPO (SS-FM init), G=32.

Writes results/S&P500/strict_fixed_oos_g32_meta/ and refreshes yearly report.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch

from backtest.drawdown import hard_mdd
from data.splits import split_for_test_month
from evaluation.analysis_spec import SERIES_COLORS, color_for
from evaluation.plots import plot_clustered_bars, plot_grouped_bars, plot_series
from experiments.engine import (
    _load_module_state,
    _new_gen_models,
    backtest_items,
    build_state_cache,
    pool_is_compact,
    pool_k,
    sample_model,
    select_candidate,
    select_candidate_meta,
)
from flow_matching.ssfm import sample_ss_fm_mixed
from models.alpha_predictor import build_predictor
from portfolio.constraints import apply_valid_mask
from portfolio.teacher_portfolios import TEACHER_NAMES
from rl.ablation_train import train_grpo_ssfm_init, train_ppo_candidate
from rl.candidate_policy import sample_portfolios_from_state
from rl.ppo import sample_portfolios
from utils.config import load_config, make_store
from utils.gpu import require_cuda
from utils.logging import log, write_json
from utils.seed import set_seed

MONTHS = [
    "2025-06",
    "2025-07",
    "2025-08",
    "2025-09",
    "2025-10",
    "2025-11",
    "2025-12",
    "2026-01",
    "2026-02",
    "2026-03",
    "2026-04",
    "2026-05",
]

G_DEFAULT = 32
POOL = "top30"  # default; overridden by --pool in main()
OOS = "strict_fixed_oos"


def _find_exp(month_root: Path) -> Path:
    cands = sorted(month_root.glob("sp500_*"))
    if not cands:
        raise FileNotFoundError(f"no sp500 exp under {month_root}")
    return cands[0]


def _src_month(src_root: Path, month: str) -> Path:
    p = src_root / OOS / OOS / f"test_month={month}"
    if p.is_dir():
        return p
    return src_root / OOS / f"test_month={month}"


def _load_alpha(cfg, store, ckpt: Path, device):
    name = str(cfg["prediction"]["primary_model"])
    model = build_predictor(name, cfg, store.feat_dim).to(device)
    _load_module_state(model, ckpt, device)
    model.eval()
    return model


def _alpha_pool(it) -> np.ndarray:
    return np.asarray(it["alpha"][it["idx"]], dtype=np.float64)


def _backtest_gen_meta(model, name: str, test_cache, cfg, device, g: int, model_tag: str) -> pd.DataFrame:
    ws = []
    for it in test_cache:
        cond = torch.from_numpy(it["cond"]).to(device)
        with torch.no_grad():
            arr = sample_model(name, model, cond, g, cfg).detach().cpu().numpy()
        _, wsel = select_candidate_meta(arr, _alpha_pool(it), it["sigma"], cfg, teachers=it.get("teachers"))
        ws.append(apply_valid_mask(wsel, it["valid"][it["idx"]]))
    return backtest_items(test_cache, ws, cfg, model_tag, OOS, g)


def _backtest_teachers(test_cache, cfg) -> List[pd.DataFrame]:
    out = []
    for tn in TEACHER_NAMES:
        ws = []
        for it in test_cache:
            w = np.asarray(it["teachers"][tn], dtype=np.float64)
            ws.append(apply_valid_mask(w, it["valid"][it["idx"]]))
        out.append(backtest_items(test_cache, ws, cfg, f"baseline_{tn}_{POOL}", OOS, 1))
    return out


def _backtest_ppo_cand(pol, ss_fm, test_cache, cfg, device, model_name: str, g: int) -> pd.DataFrame:
    n_steps = int(cfg["ssfm"]["n_sample_steps"])
    g_cand = int(cfg["rl"].get("candidate_group_size") or g)
    ws = []
    for it in test_cache:
        cond = torch.from_numpy(it["cond"]).to(device)
        with torch.no_grad():
            cands = sample_ss_fm_mixed(ss_fm, cond, g_cand, n_steps)
        state = pol.encode(cond, cands)
        w, _, _ = sample_portfolios_from_state(pol, state, g, float(cfg["rl"]["noise_std"]))
        arr = w.detach().cpu().numpy()
        _, wsel = select_candidate(arr, _alpha_pool(it), it["sigma"], cfg)
        ws.append(apply_valid_mask(wsel, it["valid"][it["idx"]]))
    return backtest_items(test_cache, ws, cfg, model_name, OOS, g)


def _backtest_grpo(pol, test_cache, cfg, device, model_name: str, g: int) -> pd.DataFrame:
    ws = []
    for it in test_cache:
        cond = torch.from_numpy(it["cond"]).to(device)
        w, _, _ = sample_portfolios(pol, cond, g, float(cfg["rl"]["noise_std"]))
        arr = w.detach().cpu().numpy()
        _, wsel = select_candidate(arr, _alpha_pool(it), it["sigma"], cfg)
        ws.append(apply_valid_mask(wsel, it["valid"][it["idx"]]))
    return backtest_items(test_cache, ws, cfg, model_name, OOS, g)


def run_month(cfg, store, device, month: str, src_root: Path, dest_root: Path, g: int) -> Path:
    src_month = _src_month(src_root, month)
    src_exp = _find_exp(src_month)
    ckpt = src_exp / "checkpoints"
    alpha_pt = ckpt / f"alpha_{cfg['prediction']['primary_model']}.pt"
    ssfm_pt = ckpt / f"gen_ssfm_{POOL}.pt"
    fm_pt = ckpt / f"gen_standard_fm_{POOL}.pt"
    for p in (alpha_pt, ssfm_pt, fm_pt):
        if not p.is_file():
            raise FileNotFoundError(p)

    dest_month = dest_root / OOS / f"test_month={month}"
    dest_exp = dest_month / f"{src_exp.name}_g32_meta"
    dest_ckpt = dest_exp / "checkpoints"
    dest_ckpt.mkdir(parents=True, exist_ok=True)
    for p in (alpha_pt, ssfm_pt, fm_pt):
        shutil.copy2(p, dest_ckpt / p.name)

    split = split_for_test_month(store.days, pd.Timestamp(f"{month}-01"), int(cfg["walkforward"]["train_offset_months"]))
    k = pool_k(store, cfg, POOL)
    compact = pool_is_compact(POOL)
    alpha = _load_alpha(cfg, store, alpha_pt, device)
    gens = _new_gen_models(cfg, device, k, compact)
    ssfm, fm = gens["ssfm"], gens["standard_fm"]
    _load_module_state(ssfm, ssfm_pt, device)
    _load_module_state(fm, fm_pt, device)
    ssfm.eval()
    fm.eval()

    cut = split["validation_end_date"]
    log(f"==== {month} train cache pool={POOL} K={k} G={g} ====")
    train_cache = build_state_cache(store, alpha, split["train_days"], cfg, device, pool=POOL, cutoff=cut)
    log(f"==== {month} test cache ====")
    test_cache = build_state_cache(store, alpha, split["test_days"], cfg, device, pool=POOL, cutoff=None)
    if not train_cache or not test_cache:
        raise RuntimeError(f"empty cache {month}")

    cfg_rl = dict(cfg)
    cfg_rl["rl"] = dict(cfg["rl"])
    cfg_rl["rl"]["group_size"] = int(g)
    cfg_rl["rl"]["candidate_group_size"] = int(g)
    cfg_rl["rl"]["bc_coef"] = 0.0

    parts: List[pd.DataFrame] = []
    meta = {
        "month": month,
        "pool": POOL,
        "g": g,
        "meta_selection": "generative G samples ∪ {mvo,max_sharpe,risk_parity} → pred_utility",
        "source": str(src_exp),
    }

    log(f"==== {month} Teachers ====")
    for df in _backtest_teachers(test_cache, cfg):
        parts.append(df)

    log(f"==== {month} FM G={g} Meta ====")
    t0 = time.time()
    df_fm = _backtest_gen_meta(fm, "standard_fm", test_cache, cfg, device, g, f"fm_G{g}_meta_{POOL}")
    parts.append(df_fm)
    meta["fm_meta_sec"] = time.time() - t0

    log(f"==== {month} SS-FM G={g} Meta ====")
    t0 = time.time()
    df_ss = _backtest_gen_meta(ssfm, "ssfm", test_cache, cfg, device, g, f"ssfm_G{g}_meta_{POOL}")
    parts.append(df_ss)
    meta["ssfm_meta_sec"] = time.time() - t0

    log(f"==== {month} PPO cand-feature G={g} ====")
    t0 = time.time()
    pol_ppo = train_ppo_candidate(ssfm, train_cache, cfg_rl, device, k=k, compact=compact)
    torch.save(pol_ppo.state_dict(), dest_ckpt / f"rl_ppo_cand_{POOL}.pt")
    df_ppo = _backtest_ppo_cand(pol_ppo, ssfm, test_cache, cfg_rl, device, f"rl_ssfm_ppo_cand_G{g}_{POOL}", g)
    parts.append(df_ppo)
    meta["ppo_sec"] = time.time() - t0

    log(f"==== {month} GRPO SSFM-init G={g} ====")
    t0 = time.time()
    pol_grpo = train_grpo_ssfm_init(ssfm, train_cache, cfg_rl, device, k=k, compact=compact)
    torch.save(pol_grpo.state_dict(), dest_ckpt / f"rl_grpo_init_{POOL}.pt")
    df_grpo = _backtest_grpo(pol_grpo, test_cache, cfg_rl, device, f"rl_ssfm_grpo_init_G{g}_{POOL}", g)
    parts.append(df_grpo)
    meta["grpo_sec"] = time.time() - t0

    daily = pd.concat(parts, ignore_index=True)
    daily.to_csv(dest_exp / "backtest_daily.csv", index=False)
    dest_month.mkdir(parents=True, exist_ok=True)
    daily.to_csv(dest_month / "backtest_daily.csv", index=False)
    write_json(dest_exp / "meta.json", meta)
    log(f"DONE {month} hours={sum(float(meta.get(k, 0) or 0) for k in ('fm_meta_sec','ssfm_meta_sec','ppo_sec','grpo_sec'))/3600:.2f}")
    return dest_exp


# ----- report -----


def _rename_map(g: int, pool: str) -> dict:
    return {
        f"baseline_mvo_{pool}": "MVO",
        f"baseline_max_sharpe_{pool}": "MaxSharpe",
        f"baseline_risk_parity_{pool}": "Risk Parity",
        f"fm_G{g}_meta_{pool}": f"FM G={g} Meta",
        f"ssfm_G{g}_meta_{pool}": f"SS-FM G={g} Meta",
        f"rl_ssfm_ppo_cand_G{g}_{pool}": f"SS-FM+PPO (cand) G={g}",
        f"rl_ssfm_grpo_init_G{g}_{pool}": f"SS-FM+GRPO (SSFM init) G={g}",
    }


def _summarize(daily: pd.DataFrame, names: List[str]) -> pd.DataFrame:
    rows = []
    for name in names:
        g = daily[daily["name"] == name].sort_values("date")
        if g.empty:
            continue
        r = g["net_return"].to_numpy(float)
        mu = float(np.nanmean(r))
        sd = float(np.nanstd(r, ddof=0)) + 1e-12
        rows.append(
            dict(
                model=name,
                total_net_return=float(np.nansum(r)),
                ann_return=float(mu * 252),
                sharpe=float(np.sqrt(252) * mu / sd),
                max_drawdown=hard_mdd(r),
                mean_turnover=float(g["turnover"].mean()) if "turnover" in g else float("nan"),
                n_days=len(r),
                return_agg="simple_sum",
            )
        )
    return pd.DataFrame(rows)


def _fmt(x, nd=4):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "-"
    return f"{float(x):.{nd}f}"


def write_report(dest_root: Path, year_csv: Path, g: int = G_DEFAULT, pool: str = POOL, out_tag: str = "strict_fixed_oos_top30_g32_meta") -> None:
    fig = dest_root / "analysis" / "figures"
    tab = dest_root / "analysis" / "tables"
    fig.mkdir(parents=True, exist_ok=True)
    tab.mkdir(parents=True, exist_ok=True)
    rename = _rename_map(g, pool)

    daily = pd.read_csv(year_csv)
    daily["date"] = pd.to_datetime(daily["date"])
    daily["name"] = daily["model"].astype(str).map(lambda x: rename.get(x, x))
    daily = daily.drop_duplicates(["date", "name"], keep="last")
    months = [m for m in MONTHS if daily["date"].astype(str).str.startswith(m).any()]

    fm_lab = f"FM G={g} Meta"
    ss_lab = f"SS-FM G={g} Meta"
    ppo_lab = f"SS-FM+PPO (cand) G={g}"
    grpo_lab = f"SS-FM+GRPO (SSFM init) G={g}"

    cm = {
        "MVO": color_for("baseline_mvo"),
        "MaxSharpe": color_for("baseline_max_sharpe"),
        "Risk Parity": color_for("baseline_risk_parity"),
        fm_lab: "#17BECF",
        ss_lab: color_for("gen_ssfm"),
        ppo_lab: "#E45756",
        grpo_lab: "#4C78A8",
    }
    for k, v in cm.items():
        SERIES_COLORS[k] = v

    def bars_cum(names: List[str], pref: str, title: str):
        names = [n for n in names if n in set(daily["name"])]
        nav = _summarize(daily, names).set_index("model")
        series = {}
        for n in names:
            gg = daily[daily["name"] == n].sort_values("date")
            series[n] = (gg["date"].to_numpy(), np.cumsum(gg["net_return"].to_numpy(float)))
        colors = [cm.get(n, "#888") for n in names]
        plot_series(
            fig / f"{pref}_cumulative_return.png",
            f"{title}：累计净收益（单利 Σ）",
            "交易日",
            "累计净收益（Σ net_return，单利）",
            names,
            series,
        )
        plot_grouped_bars(
            fig / f"{pref}_total_return.png",
            f"{title}：累计净收益（单利）",
            "模型",
            "累计净收益（单利）",
            names,
            [float(nav.loc[n, "total_net_return"]) for n in names],
            colors,
        )
        plot_grouped_bars(
            fig / f"{pref}_sharpe.png",
            f"{title}：Sharpe",
            "模型",
            "Sharpe",
            names,
            [float(nav.loc[n, "sharpe"]) for n in names],
            colors,
        )
        plot_grouped_bars(
            fig / f"{pref}_max_drawdown.png",
            f"{title}：最大回撤",
            "模型",
            "Max Drawdown",
            names,
            [float(nav.loc[n, "max_drawdown"]) for n in names],
            colors,
        )
        plot_grouped_bars(
            fig / f"{pref}_turnover.png",
            f"{title}：日均换手",
            "模型",
            "日均换手",
            names,
            [float(nav.loc[n, "mean_turnover"]) for n in names],
            colors,
        )
        month_vals = []
        for n in names:
            vals = []
            for m in months:
                s = daily[(daily["name"] == n) & (daily["date"].astype(str).str.startswith(m))]["net_return"]
                vals.append(float(s.sum()) if len(s) else None)
            month_vals.append(vals)
        plot_clustered_bars(
            fig / f"{pref}_monthly_return.png",
            f"{title}：分月净收益（单利）",
            "测试月",
            "当月Σ",
            months,
            names,
            month_vals,
            colors,
        )
        return nav.reset_index(), month_vals

    teach_names = ["MVO", "MaxSharpe", "Risk Parity", fm_lab, ss_lab]
    rl_names = [ss_lab, ppo_lab, grpo_lab]
    nav_t, mv_t = bars_cum(teach_names, "g32_meta_teacher_vs_gen", f"Top30 G={g} Meta：Teacher vs FM/SS-FM")
    nav_rl, mv_rl = bars_cum(rl_names, "g32_meta_rl_vs_ssfm", f"Top30 G={g} Meta：SS-FM vs PPO/GRPO")
    nav_t.to_csv(tab / "g32_meta_teacher_vs_gen.csv", index=False)
    nav_rl.to_csv(tab / "g32_meta_rl_vs_ssfm.csv", index=False)

    def table_md(nav_df, order):
        L = [
            "| model | total_net_return（单利） | ann_return | sharpe | max_drawdown | mean_turnover | n_days |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        ix = nav_df.set_index("model")
        for m in order:
            if m not in ix.index:
                continue
            r = ix.loc[m]
            L.append(
                f"| {m} | {_fmt(r.total_net_return)} | {_fmt(r.ann_return)} | {_fmt(r.sharpe)} | "
                f"{_fmt(r.max_drawdown)} | {_fmt(r.mean_turnover)} | {int(r.n_days)} |"
            )
        return "\n".join(L)

    def month_md(names, month_vals):
        hdr = "| month | " + " | ".join(names) + " |"
        sep = "| --- | " + " | ".join(["---"] * len(names)) + " |"
        lines = [hdr, sep]
        for i, m in enumerate(months):
            lines.append("| " + m + " | " + " | ".join(_fmt(month_vals[j][i]) for j in range(len(names))) + " |")
        return "\n".join(lines)

    n_days = int(nav_t["n_days"].max()) if len(nav_t) else 0
    fig_rel = f"../../{out_tag}/analysis/figures"
    L = []
    a = L.append
    a(f"## Top30 G={g} Meta 选仓：Teacher vs FM / SS-FM；SS-FM vs PPO / GRPO")
    a("")
    a(
        f"设定：冻结 Top30 实验 Alpha + FM + SS-FM；配权池 **Top30**；**G={g}**。"
        "生成模型每日采样后，**将 Teacher 闭式解（MVO / MaxSharpe / Risk Parity）并入候选集**，"
        "再按 `pred_utility`（`w·α̂ − ½λ wΣw`）做 Meta 选仓。"
        f"RL：`SS-FM+PPO(cand特征)` 与 `SS-FM+GRPO(SSFM init)`，`G={g}`，无 online BC。"
        f"拼接 OOS {months[0]}…{months[-1]}，约 **{n_days}** 日。**收益口径：单利 Σ**。"
    )
    a("")
    a(f"产物：`results/S&P500/{out_tag}/`。")
    a("")
    a(f"### Teacher vs FM / SS-FM（Top30 G={g} Meta）")
    a("")
    a(table_md(nav_t, teach_names))
    a("")
    a(month_md(teach_names, mv_t))
    a("")
    for title, fn in [
        (f"Top30 G={g} Meta Teacher vs FM/SS-FM 累计净值（单利）", "g32_meta_teacher_vs_gen_cumulative_return.png"),
        (f"Top30 G={g} Meta Teacher vs FM/SS-FM 累计净收益（单利柱）", "g32_meta_teacher_vs_gen_total_return.png"),
        (f"Top30 G={g} Meta Teacher vs FM/SS-FM Sharpe", "g32_meta_teacher_vs_gen_sharpe.png"),
        (f"Top30 G={g} Meta Teacher vs FM/SS-FM 最大回撤", "g32_meta_teacher_vs_gen_max_drawdown.png"),
        (f"Top30 G={g} Meta Teacher vs FM/SS-FM 日均换手", "g32_meta_teacher_vs_gen_turnover.png"),
        (f"Top30 G={g} Meta Teacher vs FM/SS-FM 分月收益（单利）", "g32_meta_teacher_vs_gen_monthly_return.png"),
    ]:
        a(f"#### {title}")
        a("")
        a(f"![{title}]({fig_rel}/{fn})")
        a("")

    a(f"### SS-FM G={g} Meta vs PPO / GRPO")
    a("")
    a("- **PPO**：SS-FM 候选作特征，再输出权重。")
    a("- **GRPO**：从 SS-FM 蒸馏初始化 WeightPolicy，再 GRPO（无 online BC）。")
    a("")
    a(table_md(nav_rl, rl_names))
    a("")
    a(month_md(rl_names, mv_rl))
    a("")
    for title, fn in [
        (f"Top30 G={g} Meta SS-FM vs PPO/GRPO 累计净值（单利）", "g32_meta_rl_vs_ssfm_cumulative_return.png"),
        (f"Top30 G={g} Meta SS-FM vs PPO/GRPO 累计净收益（单利柱）", "g32_meta_rl_vs_ssfm_total_return.png"),
        (f"Top30 G={g} Meta SS-FM vs PPO/GRPO Sharpe", "g32_meta_rl_vs_ssfm_sharpe.png"),
        (f"Top30 G={g} Meta SS-FM vs PPO/GRPO 最大回撤", "g32_meta_rl_vs_ssfm_max_drawdown.png"),
        (f"Top30 G={g} Meta SS-FM vs PPO/GRPO 日均换手", "g32_meta_rl_vs_ssfm_turnover.png"),
        (f"Top30 G={g} Meta SS-FM vs PPO/GRPO 分月收益（单利）", "g32_meta_rl_vs_ssfm_monthly_return.png"),
    ]:
        a(f"#### {title}")
        a("")
        a(f"![{title}]({fig_rel}/{fn})")
        a("")

    ix_rl = nav_rl.set_index("model")
    rank_t = " > ".join(
        [f"{r.model} `{r.total_net_return:.4f}`" for _, r in nav_t.sort_values("total_net_return", ascending=False).iterrows()]
    )
    pure = float(ix_rl.loc[ss_lab, "total_net_return"]) if ss_lab in ix_rl.index else float("nan")
    ppo = float(ix_rl.loc[ppo_lab, "total_net_return"]) if ppo_lab in ix_rl.index else float("nan")
    grpo = float(ix_rl.loc[grpo_lab, "total_net_return"]) if grpo_lab in ix_rl.index else float("nan")

    analysis = "\n".join(
        [
            f"### Top30 G={g} Meta 选仓结论",
            f"- Teacher vs 生成（单利排序）：{rank_t}。",
            f"- SS-FM G={g} Meta `{pure:.4f}`；cand-PPO `{ppo:.4f}`（{'胜' if ppo > pure else '负'}）；"
            f"GRPO-init `{grpo:.4f}`（{'胜' if grpo > pure else '负'}）。",
            "- Meta = 生成样本 ∪ Teacher 闭式解再 `pred_utility`；若选中 Teacher，当日权重可与基线 Teacher 相同。",
            "",
        ]
    )

    section = "\n".join(L) + "\n"
    marker = f"## Top30 G={g} Meta 选仓：Teacher vs FM / SS-FM；SS-FM vs PPO / GRPO\n"
    amark = f"### Top30 G={g} Meta 选仓结论\n"
    report_paths = [
        Path("results/S&P500/strict_fixed_oos/final_summary/S&P500全年实验分析.md"),
        Path("results/S&P500/strict_fixed_oos/analysis/S&P500全年实验分析.md"),
        Path("results/S&P500/SP500_top30全年实验分析.md"),
        Path("results/S&P500/strict_fixed_oos_top30/final_summary/SP500_top30全年实验分析.md"),
    ]
    for path in report_paths:
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        import re

        # figure relative path depends on report location
        if "strict_fixed_oos_top30/final_summary" in str(path) or str(path).endswith("SP500_top30全年实验分析.md") and "final_summary" not in str(path):
            # results/S&P500/SP500_top30... → out_tag/analysis/figures
            local_section = section.replace(fig_rel, f"{out_tag}/analysis/figures")
            if "final_summary" in str(path):
                local_section = section.replace(fig_rel, f"../../{out_tag}/analysis/figures")
        else:
            local_section = section

        if marker in text:
            s = text.index(marker)
            m = re.search(r"\n## ", text[s + len(marker) :])
            e = s + len(marker) + m.start() if m else len(text)
            text = text[:s] + local_section + text[e:]
        else:
            for anchor in ("## 附录：", "## 分析\n"):
                if anchor in text and anchor.startswith("## 附录"):
                    text = text.replace(anchor, local_section + anchor, 1)
                    break
            else:
                text = text.rstrip() + "\n\n" + local_section

        if amark in text:
            a0 = text.index(amark)
            a_rest = text[a0 + 1 :]
            a_end = None
            for stop in ("\n### ", "\n## "):
                j = a_rest.find(stop, 5)
                if j >= 0:
                    a_end = a0 + 1 + j if a_end is None else min(a_end, a0 + 1 + j)
            text = text[:a0] + analysis + (text[a_end:] if a_end else "")
        else:
            text = text.rstrip() + "\n\n" + analysis
        path.write_text(text, encoding="utf-8")
        log(f"updated report {path}")

    (dest_root / "analysis" / "G32_Meta选仓报告片段.md").write_text(section + "\n" + analysis, encoding="utf-8")


def main():
    global POOL
    ap = argparse.ArgumentParser()
    ap.add_argument("--g", type=int, default=G_DEFAULT)
    ap.add_argument("--pool", default="top30", choices=["top30", "all"])
    ap.add_argument("--out-tag", default="strict_fixed_oos_top30_g32_meta")
    ap.add_argument("--src-tag", default="strict_fixed_oos_top30")
    ap.add_argument("--months", default=",".join(MONTHS))
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()
    g = int(args.g)
    POOL = str(args.pool)
    months = [m.strip() for m in args.months.split(",") if m.strip()]

    set_seed(42, False, True)
    dest_root = Path("results/S&P500") / args.out_tag
    dest_root.mkdir(parents=True, exist_ok=True)
    year_csv = dest_root / "backtest_daily_year.csv"
    log_path = dest_root / "run_all.log"

    if args.report_only:
        if not year_csv.is_file():
            raise SystemExit(f"missing {year_csv}")
        write_report(dest_root, year_csv, g=g, pool=POOL, out_tag=args.out_tag)
        return

    import os

    os.environ.setdefault("GRPO_MARKET", "sp500")
    cfg = load_config()
    # Top30 allocation
    cfg.setdefault("portfolio", {})
    if POOL == "top30":
        cfg["portfolio"]["allocation_pools"] = ["top30"]
        cfg.setdefault("matrix", {})
        cfg["matrix"]["allocation_pools"] = ["top30"]
    device = require_cuda()
    store = make_store(cfg)
    src_root = Path("results/S&P500") / args.src_tag

    t_all = time.time()
    frames = []
    with open(log_path, "a", encoding="utf-8") as flog:

        def _log(msg: str):
            log(msg)
            flog.write(msg + "\n")
            flog.flush()

        _log(f"CONFIG pool={POOL} g={g} src={src_root} out={dest_root}")
        for month in months:
            _log(f"==== TOP30 G={g} META month={month} ====")
            t0 = time.time()
            exp = run_month(cfg, store, device, month, src_root, dest_root, g)
            df = pd.read_csv(exp / "backtest_daily.csv")
            frames.append(df)
            _log(f"DONE {month} hours={(time.time()-t0)/3600:.2f}")

        year = pd.concat(frames, ignore_index=True)
        year.to_csv(year_csv, index=False)
        _log(f"REQUESTED MONTHS SP500-TOP30-G{g}-META DONE hours={(time.time()-t_all)/3600:.2f} out={dest_root}")

    write_report(dest_root, year_csv, g=g, pool=POOL, out_tag=args.out_tag)


if __name__ == "__main__":
    main()
