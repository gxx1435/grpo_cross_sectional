#!/usr/bin/env python3
"""SP500 Top30 Diffusion Meta stability: 20 seeds × G-grid × 4 aggregations.

Freeze latest Alpha+Diffusion (no training).
Meta pool = G Diffusion samples ∪ Teacher (MVO/MaxSharpe/RP).
For each G: 20 fixed random noise seeds; daily noise = hash(date, seed).
Aggregations share the same Meta pool within (day, G, seed):
  pred_utility / mean / topk / softmax.
"""

from __future__ import annotations

import argparse
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Tuple

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
)
from flow_matching.diffusion import sample_diffusion
from models.alpha_predictor import build_predictor
from portfolio.constraints import apply_valid_mask
from rl.stable_meta import (
    aggregate_candidates,
    build_meta_pool_np,
    date_seed,
)
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

G_LIST = [8, 32, 64, 128]
MODES = [
    ("pred_utility", "Pred-Utility"),
    ("mean", "Mean"),
    ("topk", "Top-k"),
    ("softmax", "Softmax"),
]
POOL = "top30"
OOS = "strict_fixed_oos"
TOP_K = 3
SOFTMAX_T = 1.0
N_SEEDS = 20
MASTER_SEED = 42


def make_noise_seeds(n: int = N_SEEDS, master: int = MASTER_SEED) -> List[int]:
    """Use consecutive seeds {1,2,...,n} (master unused; kept for CLI compat)."""
    return list(range(1, int(n) + 1))


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


def _model_id(g: int, mode: str, seed: int) -> str:
    return f"diff_G{g}_meta_{mode}_s{seed}_{POOL}"


def _model_label(g: int, mode_key: str) -> str:
    nice = dict(MODES).get(mode_key, mode_key)
    return f"Diff Meta G={g} {nice}"


def run_month(
    cfg,
    store,
    device,
    month: str,
    src_root: Path,
    dest_root: Path,
    noise_seeds: List[int],
) -> Path:
    src_month = _src_month(src_root, month)
    src_exp = _find_exp(src_month)
    ckpt = src_exp / "checkpoints"
    alpha_pt = ckpt / f"alpha_{cfg['prediction']['primary_model']}.pt"
    diff_pt = ckpt / f"gen_diffusion_{POOL}.pt"
    for p in (alpha_pt, diff_pt):
        if not p.is_file():
            raise FileNotFoundError(p)

    dest_month = dest_root / OOS / f"test_month={month}"
    dest_exp = dest_month / f"{src_exp.name}_diff_stability_meta_seeds"
    dest_exp.mkdir(parents=True, exist_ok=True)

    split = split_for_test_month(store.days, pd.Timestamp(f"{month}-01"), int(cfg["walkforward"]["train_offset_months"]))
    k = pool_k(store, cfg, POOL)
    compact = pool_is_compact(POOL)
    alpha = _load_alpha(cfg, store, alpha_pt, device)
    diff = _new_gen_models(cfg, device, k, compact)["diffusion"]
    _load_module_state(diff, diff_pt, device)
    diff.eval()
    for p in diff.parameters():
        p.requires_grad_(False)

    log(f"==== {month} test cache pool={POOL} K={k} Diffusion Meta n_seeds={len(noise_seeds)} (freeze) ====")
    test_cache = build_state_cache(store, alpha, split["test_days"], cfg, device, pool=POOL, cutoff=None)
    if not test_cache:
        raise RuntimeError(f"empty cache {month}")

    ra = float(cfg["ssfm"]["risk_aversion"])
    parts: List[pd.DataFrame] = []
    t0 = time.time()

    # key: (g, mode, seed) -> list of daily weights
    keys = [(g, m, s) for g in G_LIST for m, _ in MODES for s in noise_seeds]
    weights_by_key: Dict[Tuple[int, str, int], List[np.ndarray]] = {k: [] for k in keys}

    for it in test_cache:
        cond = torch.from_numpy(it["cond"]).to(device)
        a = np.asarray(it["alpha"][it["idx"]], dtype=np.float64)
        sig = np.asarray(it["sigma"], dtype=np.float64)
        valid = it["valid"][it["idx"]]
        teachers = it.get("teachers")
        for g in G_LIST:
            for seed in noise_seeds:
                # day-varying but seed-controlled noise
                day_seed = date_seed(it["asof"], int(seed))
                with torch.no_grad():
                    ws = sample_diffusion(diff, cond, g, seed=day_seed).detach().cpu().numpy()
                pool = build_meta_pool_np(ws, teachers)
                for mode, _ in MODES:
                    w = aggregate_candidates(pool, a, sig, ra, mode=mode, top_k=TOP_K, temperature=SOFTMAX_T)
                    weights_by_key[(g, mode, seed)].append(apply_valid_mask(w, valid))

    for g, mode, seed in keys:
        mid = _model_id(g, mode, seed)
        df = backtest_items(test_cache, weights_by_key[(g, mode, seed)], cfg, mid, OOS, g)
        df["noise_seed"] = int(seed)
        df["g"] = int(g)
        df["agg"] = mode
        parts.append(df)

    daily = pd.concat(parts, ignore_index=True)
    dest_month.mkdir(parents=True, exist_ok=True)
    daily.to_csv(dest_exp / "backtest_daily.csv", index=False)
    daily.to_csv(dest_month / "backtest_daily.csv", index=False)
    write_json(
        dest_exp / "meta.json",
        {
            "month": month,
            "pool": POOL,
            "g_list": G_LIST,
            "modes": [m for m, _ in MODES],
            "top_k": TOP_K,
            "softmax_t": SOFTMAX_T,
            "n_seeds": len(noise_seeds),
            "noise_seeds": noise_seeds,
            "daily_noise": "hash(date, seed)",
            "meta": True,
            "generator": "diffusion",
            "teachers": ["mvo", "max_sharpe", "risk_parity"],
            "source": str(src_exp),
            "train": False,
            "sec": time.time() - t0,
        },
    )
    n_var = len(G_LIST) * len(MODES) * len(noise_seeds)
    log(f"DONE {month} hours={(time.time()-t0)/3600:.2f} variants={n_var}")
    return dest_exp


def _metrics_one(r: np.ndarray, turnover: np.ndarray) -> dict:
    r = np.asarray(r, dtype=np.float64)
    mu = float(np.nanmean(r))
    sd = float(np.nanstd(r, ddof=0)) + 1e-12
    return dict(
        total_net_return=float(np.nansum(r)),
        ann_return=float(mu * 252),
        sharpe=float(np.sqrt(252) * mu / sd),
        max_drawdown=hard_mdd(r),
        mean_turnover=float(np.nanmean(turnover)),
        n_days=int(len(r)),
    )


def _fmt(x, nd=4):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "-"
    return f"{float(x):.{nd}f}"


def _fmt_pm(mu, sd, nd=4):
    if mu is None or not np.isfinite(mu):
        return "-"
    if sd is None or not np.isfinite(sd):
        return _fmt(mu, nd)
    return f"{float(mu):.{nd}f}±{float(sd):.{nd}f}"


def write_report(dest_root: Path, year_csv: Path, out_tag: str, noise_seeds: List[int]) -> None:
    fig = dest_root / "analysis" / "figures"
    tab = dest_root / "analysis" / "tables"
    fig.mkdir(parents=True, exist_ok=True)
    tab.mkdir(parents=True, exist_ok=True)

    daily = pd.read_csv(year_csv)
    daily["date"] = pd.to_datetime(daily["date"])
    if "noise_seed" not in daily.columns:
        # parse from model name ..._s{seed}_top30
        daily["noise_seed"] = daily["model"].astype(str).str.extract(r"_s(\d+)_")[0].astype(int)
        daily["g"] = daily["model"].astype(str).str.extract(r"_G(\d+)_")[0].astype(int)
        daily["agg"] = daily["model"].astype(str).str.extract(r"_meta_([a-z_]+)_s")[0]

    months = [m for m in MONTHS if daily["date"].astype(str).str.startswith(m).any()]
    seed_rows = []
    for g in G_LIST:
        for mode, nice in MODES:
            for seed in noise_seeds:
                mid = _model_id(g, mode, seed)
                sub = daily[daily["model"] == mid].sort_values("date")
                if sub.empty:
                    # fallback filter
                    sub = daily[(daily["g"] == g) & (daily["agg"] == mode) & (daily["noise_seed"] == seed)].sort_values("date")
                if sub.empty:
                    continue
                met = _metrics_one(sub["net_return"].to_numpy(float), sub["turnover"].to_numpy(float) if "turnover" in sub else np.zeros(len(sub)))
                seed_rows.append(dict(g=g, agg=mode, agg_label=nice, noise_seed=seed, **met))
    seed_df = pd.DataFrame(seed_rows)
    seed_df.to_csv(tab / "diff_meta_stability_seeds_raw.csv", index=False)

    # mean±std across seeds
    summary_rows = []
    for g in G_LIST:
        for mode, nice in MODES:
            sub = seed_df[(seed_df["g"] == g) & (seed_df["agg"] == mode)]
            if sub.empty:
                continue
            summary_rows.append(
                dict(
                    g=g,
                    agg=mode,
                    agg_label=nice,
                    n_seeds=len(sub),
                    total_mean=float(sub["total_net_return"].mean()),
                    total_std=float(sub["total_net_return"].std(ddof=0)),
                    sharpe_mean=float(sub["sharpe"].mean()),
                    sharpe_std=float(sub["sharpe"].std(ddof=0)),
                    turnover_mean=float(sub["mean_turnover"].mean()),
                    turnover_std=float(sub["mean_turnover"].std(ddof=0)),
                    mdd_mean=float(sub["max_drawdown"].mean()),
                    mdd_std=float(sub["max_drawdown"].std(ddof=0)),
                )
            )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(tab / "diff_meta_stability_seeds_summary.csv", index=False)

    # pivots mean / std
    pivot_mu = summary.pivot(index="g", columns="agg_label", values="total_mean").reindex(G_LIST)
    pivot_sd = summary.pivot(index="g", columns="agg_label", values="total_std").reindex(G_LIST)
    pivot_mu.index = [f"G={g}" for g in pivot_mu.index]
    pivot_sd.index = [f"G={g}" for g in pivot_sd.index]
    pivot_mu.to_csv(tab / "diff_meta_stability_seeds_pivot_return_mean.csv")
    pivot_sd.to_csv(tab / "diff_meta_stability_seeds_pivot_return_std.csv")

    mode_colors = {
        "Pred-Utility": color_for("gen_diffusion"),
        "Mean": "#E45756",
        "Top-k": "#4C78A8",
        "Softmax": "#F58518",
    }
    labels = [dict(MODES)[m] for m, _ in MODES]
    # grouped bars of mean return by G (clustered by mode) — use one chart per metric with all G×mode
    names = [f"G={g} {nice}" for g in G_LIST for _, nice in MODES]
    means = []
    stds = []
    colors = []
    for g in G_LIST:
        for mode, nice in MODES:
            row = summary[(summary["g"] == g) & (summary["agg"] == mode)]
            means.append(float(row["total_mean"].iloc[0]) if len(row) else float("nan"))
            stds.append(float(row["total_std"].iloc[0]) if len(row) else float("nan"))
            colors.append(mode_colors[nice])
            SERIES_COLORS[f"G={g} {nice}"] = mode_colors[nice]

    pref = "diff_meta_stability_seeds"
    title = f"Top30 Diffusion Meta stability：G×聚合（{len(noise_seeds)} seeds）"
    plot_grouped_bars(
        fig / f"{pref}_total_return_mean.png",
        f"{title} 累计净收益 mean（单利）",
        "模型",
        "累计净收益 mean",
        names,
        means,
        colors,
    )
    plot_grouped_bars(
        fig / f"{pref}_total_return_std.png",
        f"{title} 累计净收益 std across seeds",
        "模型",
        "std",
        names,
        stds,
        colors,
    )
    sharpe_means = []
    for g in G_LIST:
        for mode, nice in MODES:
            row = summary[(summary["g"] == g) & (summary["agg"] == mode)]
            sharpe_means.append(float(row["sharpe_mean"].iloc[0]) if len(row) else float("nan"))
    plot_grouped_bars(
        fig / f"{pref}_sharpe_mean.png",
        f"{title} Sharpe mean",
        "模型",
        "Sharpe mean",
        names,
        sharpe_means,
        colors,
    )

    # For each mode: G on x, mean return bar (one series per... actually plot G means for one mode via grouped)
    for mode, nice in MODES:
        ns = [f"G={g}" for g in G_LIST]
        vals = [float(summary[(summary["g"] == g) & (summary["agg"] == mode)]["total_mean"].iloc[0]) for g in G_LIST]
        plot_grouped_bars(
            fig / f"{pref}_{mode}_return_mean_by_G.png",
            f"{title}｜{nice} 累计净收益 mean by G",
            "G",
            "累计净收益 mean",
            ns,
            vals,
            [mode_colors[nice]] * len(ns),
        )

    # Seed-level cumulative for G=32 Pred-Utility (illustrative)
    g_show, mode_show = 32, "pred_utility"
    series = {}
    sn = []
    for seed in noise_seeds[: min(8, len(noise_seeds))]:  # plot up to 8 seeds to avoid clutter
        mid = _model_id(g_show, mode_show, seed)
        gg = daily[daily["model"] == mid].sort_values("date")
        if gg.empty:
            continue
        name = f"seed={seed}"
        sn.append(name)
        series[name] = (gg["date"].to_numpy(), np.cumsum(gg["net_return"].to_numpy(float)))
        SERIES_COLORS[name] = color_for("gen_diffusion")
    if sn:
        plot_series(
            fig / f"{pref}_G{g_show}_pred_utility_seeds_cum.png",
            f"{title} G={g_show} Pred-Utility 多 seed 累计（示意）",
            "交易日",
            "累计净收益",
            sn,
            series,
        )

    n_days = int(seed_df["n_days"].max()) if len(seed_df) else 0
    L = []
    a = L.append
    a(f"## Top30 Diffusion Meta stability：G×聚合（{len(noise_seeds)} seeds）")
    a("")
    a(
        "设定：冻结 Top30 最新 Alpha+Diffusion（**不训练**）；配权 **Top30**；"
        "**Meta 候选 = G 条 Diffusion ∪ Teacher（MVO/MaxSharpe/RP）**；"
        f"**G ∈ {{{', '.join(str(x) for x in G_LIST)}}}**；"
        f"每组 G 使用 **seed ∈ {{1..{len(noise_seeds)}}}** 作初始化噪声；"
        "每日 `noise = hash(date, seed)`。"
        "同 (日,G,seed) 只建一次 Meta 池，四种聚合共享。"
    )
    a("")
    a(
        f"聚合：Pred-Utility / Mean / Top-k={TOP_K} / Softmax(τ={SOFTMAX_T})。"
        "下表为 **across seeds 的 mean±std（单利）**。"
    )
    a("")
    a(f"产物：`results/S&P500/{out_tag}/`。约 **{n_days}** 日 × {len(noise_seeds)} seeds。")
    a("")
    a("### 累计净收益 mean±std（单利）")
    a("")
    a("| G | " + " | ".join(labels) + " |")
    a("| --- | " + " | ".join(["---"] * len(labels)) + " |")
    for g in G_LIST:
        cells = []
        for mode, nice in MODES:
            row = summary[(summary["g"] == g) & (summary["agg"] == mode)]
            if row.empty:
                cells.append("-")
            else:
                cells.append(_fmt_pm(float(row["total_mean"].iloc[0]), float(row["total_std"].iloc[0])))
        a("| G=" + str(g) + " | " + " | ".join(cells) + " |")
    a("")
    a("### Sharpe mean±std")
    a("")
    a("| G | " + " | ".join(labels) + " |")
    a("| --- | " + " | ".join(["---"] * len(labels)) + " |")
    for g in G_LIST:
        cells = []
        for mode, nice in MODES:
            row = summary[(summary["g"] == g) & (summary["agg"] == mode)]
            if row.empty:
                cells.append("-")
            else:
                cells.append(_fmt_pm(float(row["sharpe_mean"].iloc[0]), float(row["sharpe_std"].iloc[0])))
        a("| G=" + str(g) + " | " + " | ".join(cells) + " |")
    a("")
    a("### 日均换手 mean±std")
    a("")
    a("| G | " + " | ".join(labels) + " |")
    a("| --- | " + " | ".join(["---"] * len(labels)) + " |")
    for g in G_LIST:
        cells = []
        for mode, nice in MODES:
            row = summary[(summary["g"] == g) & (summary["agg"] == mode)]
            if row.empty:
                cells.append("-")
            else:
                cells.append(_fmt_pm(float(row["turnover_mean"].iloc[0]), float(row["turnover_std"].iloc[0])))
        a("| G=" + str(g) + " | " + " | ".join(cells) + " |")
    a("")
    for title_i, fn in [
        (f"{title} 累计净收益 mean", f"{pref}_total_return_mean.png"),
        (f"{title} 累计净收益 std", f"{pref}_total_return_std.png"),
        (f"{title} Sharpe mean", f"{pref}_sharpe_mean.png"),
    ]:
        a(f"#### {title_i}")
        a("")
        a(f"![{title_i}](PLACEHOLDER/{fn})")
        a("")
    for mode, nice in MODES:
        a(f"#### {title}｜{nice} by G")
        a("")
        a(f"![{nice}](PLACEHOLDER/{pref}_{mode}_return_mean_by_G.png)")
        a("")
    a(f"#### {title} G={g_show} Pred-Utility 多 seed 累计（示意）")
    a("")
    a(f"![seeds](PLACEHOLDER/{pref}_G{g_show}_pred_utility_seeds_cum.png)")
    a("")

    # best by mean total
    if len(summary):
        best = summary.loc[summary["total_mean"].idxmax()]
        worst_std = summary.loc[summary["total_std"].idxmax()]
        analysis = "\n".join(
            [
                f"### Top30 Diffusion Meta stability（{len(noise_seeds)} seeds）结论",
                f"- 最高 mean 单利：Meta G={int(best.g)} {best.agg_label} = "
                f"`{_fmt_pm(best.total_mean, best.total_std)}`（Sharpe `{_fmt_pm(best.sharpe_mean, best.sharpe_std)}`）。",
                f"- 最大跨 seed 方差：G={int(worst_std.g)} {worst_std.agg_label} std=`{_fmt(worst_std.total_std)}`。",
                f"- seed ∈ {{1..{len(noise_seeds)}}}；daily `hash(date, seed)`；候选含 Teacher；不训练。",
                "",
            ]
        )
    else:
        analysis = f"### Top30 Diffusion Meta stability（{len(noise_seeds)} seeds）结论\n- 无有效结果。\n"

    section_tmpl = "\n".join(L) + "\n"
    targets = [
        (Path("results/S&P500/SP500_top30全年实验分析.md"), f"{out_tag}/analysis/figures"),
        (Path("results/S&P500/strict_fixed_oos_top30/final_summary/SP500_top30全年实验分析.md"), f"../../{out_tag}/analysis/figures"),
        (Path("results/S&P500/strict_fixed_oos/final_summary/S&P500全年实验分析.md"), f"../../{out_tag}/analysis/figures"),
        (Path("results/S&P500/strict_fixed_oos/analysis/S&P500全年实验分析.md"), f"../../{out_tag}/analysis/figures"),
    ]
    marker = f"## Top30 Diffusion Meta stability：G×聚合（{len(noise_seeds)} seeds）\n"
    amark = f"### Top30 Diffusion Meta stability（{len(noise_seeds)} seeds）结论\n"

    for path, fig_rel in targets:
        if not path.is_file():
            continue
        section = section_tmpl.replace("PLACEHOLDER", fig_rel)
        text = path.read_text(encoding="utf-8")
        if marker in text:
            s = text.index(marker)
            m = re.search(r"\n## ", text[s + len(marker) :])
            e = s + len(marker) + m.start() if m else len(text)
            text = text[:s] + section + text[e:]
        else:
            text = text.rstrip() + "\n\n" + section
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

    (dest_root / "analysis" / "diff_meta_stability_seeds报告片段.md").write_text(
        section_tmpl.replace("PLACEHOLDER", "figures") + "\n" + analysis, encoding="utf-8"
    )
    write_json(dest_root / "analysis" / "noise_seeds.json", {"master_seed": MASTER_SEED, "noise_seeds": noise_seeds})


def main():
    global POOL, TOP_K, SOFTMAX_T, N_SEEDS, MASTER_SEED
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="top30")
    ap.add_argument("--out-tag", default="strict_fixed_oos_top30_diff_stability_meta_seeds")
    ap.add_argument("--src-tag", default="strict_fixed_oos_top30")
    ap.add_argument("--months", default=",".join(MONTHS))
    ap.add_argument("--top-k", type=int, default=TOP_K)
    ap.add_argument("--softmax-t", type=float, default=SOFTMAX_T)
    ap.add_argument("--n-seeds", type=int, default=N_SEEDS)
    ap.add_argument("--master-seed", type=int, default=MASTER_SEED)
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()
    POOL = str(args.pool)
    TOP_K = int(args.top_k)
    SOFTMAX_T = float(args.softmax_t)
    N_SEEDS = int(args.n_seeds)
    MASTER_SEED = int(args.master_seed)
    months = [m.strip() for m in args.months.split(",") if m.strip()]
    noise_seeds = make_noise_seeds(N_SEEDS, MASTER_SEED)

    set_seed(MASTER_SEED, False, True)
    dest_root = Path("results/S&P500") / args.out_tag
    dest_root.mkdir(parents=True, exist_ok=True)
    year_csv = dest_root / "backtest_daily_year.csv"
    log_path = dest_root / "run_all.log"

    if args.report_only:
        write_report(dest_root, year_csv, args.out_tag, noise_seeds)
        return

    os.environ.setdefault("GRPO_MARKET", "sp500")
    cfg = load_config()
    cfg.setdefault("portfolio", {})
    cfg["portfolio"]["allocation_pools"] = [POOL]
    cfg.setdefault("matrix", {})
    cfg["matrix"]["allocation_pools"] = [POOL]
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

        _log(
            f"CONFIG pool={POOL} Meta=True G={G_LIST} modes={[m for m,_ in MODES]} "
            f"n_seeds={N_SEEDS} master_seed={MASTER_SEED} seeds={noise_seeds} "
            f"top_k={TOP_K} softmax_t={SOFTMAX_T} train=False out={dest_root}"
        )
        for month in months:
            _log(f"==== TOP30 Diff-Meta-stability-seeds month={month} ====")
            t0 = time.time()
            exp = run_month(cfg, store, device, month, src_root, dest_root, noise_seeds)
            frames.append(pd.read_csv(exp / "backtest_daily.csv"))
            _log(f"DONE {month} hours={(time.time()-t0)/3600:.2f}")

        year = pd.concat(frames, ignore_index=True)
        year.to_csv(year_csv, index=False)
        _log(
            f"REQUESTED MONTHS SP500-TOP30-DIFF-META-STABILITY-SEEDS DONE "
            f"hours={(time.time()-t_all)/3600:.2f} out={dest_root}"
        )

    write_report(dest_root, year_csv, args.out_tag, noise_seeds)


if __name__ == "__main__":
    main()
