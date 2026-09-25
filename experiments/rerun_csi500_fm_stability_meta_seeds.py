#!/usr/bin/env python3
"""CSI500 Top30 FM Meta stability: 20 seeds × G-grid × 4 aggregations.

Freeze latest Alpha + standard FM (no training).
Meta pool = G FM samples ∪ Teacher (MVO/MaxSharpe/RP).
G ∈ {8,32,64,128}; seed ∈ {1..20}; daily noise = hash(date, seed).
Aggregations share the same Meta pool within (day, G, seed):
  pred_utility / mean / topk / softmax.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

from backtest.drawdown import hard_mdd
from data.splits import split_for_test_month
from evaluation.analysis_spec import SERIES_COLORS, color_for
from evaluation.plots import plot_grouped_bars, plot_series
from experiments.engine import (
    _load_module_state,
    _new_gen_models,
    backtest_items,
    build_state_cache,
    pool_is_compact,
    pool_k,
)
from experiments.reuse_ablation_alpha import load_frozen_ablation_alpha
from flow_matching.standard_fm import sample_std_fm
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
    "2025-05",
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
    return list(range(1, int(n) + 1))


def _find_exp(month_root: Path) -> Path:
    cands = sorted(month_root.glob("csi500_*"))
    if not cands:
        raise FileNotFoundError(f"no csi500 exp under {month_root}")
    return cands[0]


def _src_month(src_root: Path, month: str) -> Path:
    p = src_root / OOS / OOS / f"test_month={month}"
    if p.is_dir():
        return p
    return src_root / OOS / f"test_month={month}"


def _model_id(g: int, mode: str, seed: int) -> str:
    return f"fm_G{g}_meta_{mode}_s{seed}_{POOL}"


def run_month(
    cfg,
    store,
    device,
    month: str,
    src_root: Path,
    dest_root: Path,
    noise_seeds: List[int],
    g_list: List[int],
) -> Path:
    src_month = _src_month(src_root, month)
    src_exp = _find_exp(src_month)
    ckpt = src_exp / "checkpoints"
    fm_pt = ckpt / f"gen_standard_fm_{POOL}.pt"
    if not fm_pt.is_file():
        raise FileNotFoundError(fm_pt)

    dest_month = dest_root / OOS / f"test_month={month}"
    dest_exp = dest_month / f"{src_exp.name}_fm_stability_meta_seeds"
    dest_exp.mkdir(parents=True, exist_ok=True)

    split = split_for_test_month(store.days, pd.Timestamp(f"{month}-01"), int(cfg["walkforward"]["train_offset_months"]))
    k = pool_k(store, cfg, POOL)
    compact = pool_is_compact(POOL)
    alpha, alpha_meta = load_frozen_ablation_alpha(cfg, store, month, device)
    write_json(dest_exp / "alpha_reuse_meta.json", alpha_meta if isinstance(alpha_meta, dict) else {"meta": str(alpha_meta)})
    fm = _new_gen_models(cfg, device, k, compact)["standard_fm"]
    _load_module_state(fm, fm_pt, device)
    fm.eval()
    for p in fm.parameters():
        p.requires_grad_(False)

    log(f"==== {month} test cache pool={POOL} K={k} FM Meta n_seeds={len(noise_seeds)} (freeze) ====")
    test_cache = build_state_cache(store, alpha, split["test_days"], cfg, device, pool=POOL, cutoff=None)
    if not test_cache:
        raise RuntimeError(f"empty cache {month}")

    n_steps = int(cfg["standard_fm"]["n_sample_steps"])
    ra = float(cfg["ssfm"]["risk_aversion"])
    parts: List[pd.DataFrame] = []
    t0 = time.time()

    keys = [(g, m, s) for g in g_list for m, _ in MODES for s in noise_seeds]
    weights_by_key: Dict[Tuple[int, str, int], List[np.ndarray]] = {k: [] for k in keys}

    for it in test_cache:
        cond = torch.from_numpy(it["cond"]).to(device)
        a = np.asarray(it["alpha"][it["idx"]], dtype=np.float64)
        sig = np.asarray(it["sigma"], dtype=np.float64)
        valid = it["valid"][it["idx"]]
        teachers = it.get("teachers")
        for g in g_list:
            for seed in noise_seeds:
                day_seed = date_seed(it["asof"], int(seed))
                with torch.no_grad():
                    ws = sample_std_fm(fm, cond, g, n_steps, seed=day_seed).detach().cpu().numpy()
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
            "g_list": g_list,
            "modes": [m for m, _ in MODES],
            "top_k": TOP_K,
            "softmax_t": SOFTMAX_T,
            "n_seeds": len(noise_seeds),
            "noise_seeds": noise_seeds,
            "daily_noise": "hash(date, seed)",
            "meta": True,
            "teachers": ["mvo", "max_sharpe", "risk_parity"],
            "generator": "standard_fm",
            "source": str(src_exp),
            "train": False,
            "sec": time.time() - t0,
        },
    )
    n_var = len(g_list) * len(MODES) * len(noise_seeds)
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


def write_report(dest_root: Path, year_csv: Path, out_tag: str, noise_seeds: List[int], g_list: List[int]) -> None:
    fig = dest_root / "analysis" / "figures"
    tab = dest_root / "analysis" / "tables"
    fig.mkdir(parents=True, exist_ok=True)
    tab.mkdir(parents=True, exist_ok=True)

    daily = pd.read_csv(year_csv)
    daily["date"] = pd.to_datetime(daily["date"])
    if "noise_seed" not in daily.columns:
        daily["noise_seed"] = daily["model"].astype(str).str.extract(r"_s(\d+)_")[0].astype(int)
        daily["g"] = daily["model"].astype(str).str.extract(r"_G(\d+)_")[0].astype(int)
        daily["agg"] = daily["model"].astype(str).str.extract(r"_meta_([a-z_]+)_s")[0]

    seed_rows = []
    for g in g_list:
        for mode, nice in MODES:
            for seed in noise_seeds:
                mid = _model_id(g, mode, seed)
                sub = daily[daily["model"] == mid].sort_values("date")
                if sub.empty:
                    sub = daily[(daily["g"] == g) & (daily["agg"] == mode) & (daily["noise_seed"] == seed)].sort_values("date")
                if sub.empty:
                    continue
                met = _metrics_one(
                    sub["net_return"].to_numpy(float),
                    sub["turnover"].to_numpy(float) if "turnover" in sub else np.zeros(len(sub)),
                )
                seed_rows.append(dict(g=g, agg=mode, agg_label=nice, noise_seed=seed, **met))
    seed_df = pd.DataFrame(seed_rows)
    seed_df.to_csv(tab / "fm_meta_stability_seeds_raw.csv", index=False)

    summary_rows = []
    for g in g_list:
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
                    total_min=float(sub["total_net_return"].min()),
                    total_max=float(sub["total_net_return"].max()),
                    sharpe_mean=float(sub["sharpe"].mean()),
                    sharpe_std=float(sub["sharpe"].std(ddof=0)),
                    turnover_mean=float(sub["mean_turnover"].mean()),
                    turnover_std=float(sub["mean_turnover"].std(ddof=0)),
                    mdd_mean=float(sub["max_drawdown"].mean()),
                    mdd_std=float(sub["max_drawdown"].std(ddof=0)),
                )
            )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(tab / "fm_meta_stability_seeds_summary.csv", index=False)

    pivot_mu = summary.pivot(index="g", columns="agg_label", values="total_mean").reindex(g_list)
    pivot_sd = summary.pivot(index="g", columns="agg_label", values="total_std").reindex(g_list)
    pivot_mu.index = [f"G={g}" for g in pivot_mu.index]
    pivot_sd.index = [f"G={g}" for g in pivot_sd.index]
    labels = [nice for _, nice in MODES]
    pivot_mu = pivot_mu.reindex(columns=labels)
    pivot_sd = pivot_sd.reindex(columns=labels)
    pivot_mu.to_csv(tab / "fm_meta_stability_seeds_pivot_return_mean.csv")
    pivot_sd.to_csv(tab / "fm_meta_stability_seeds_pivot_return_std.csv")

    mode_colors = {
        "Pred-Utility": color_for("gen_standard_fm"),
        "Mean": "#E45756",
        "Top-k": "#4C78A8",
        "Softmax": "#F58518",
    }
    names = [f"G={g} {nice}" for g in g_list for _, nice in MODES]
    means, stds, colors = [], [], []
    for g in g_list:
        for mode, nice in MODES:
            row = summary[(summary["g"] == g) & (summary["agg"] == mode)]
            means.append(float(row["total_mean"].iloc[0]) if len(row) else float("nan"))
            stds.append(float(row["total_std"].iloc[0]) if len(row) else float("nan"))
            colors.append(mode_colors[nice])
            SERIES_COLORS[f"G={g} {nice}"] = mode_colors[nice]

    pref = "fm_meta_stability_seeds"
    title = f"CSI500 FM Meta stability：G×聚合（{len(noise_seeds)} seeds）"
    plot_grouped_bars(fig / f"{pref}_total_return_mean.png", f"{title} 累计净收益 mean（单利）", "模型", "累计净收益 mean", names, means, colors)
    plot_grouped_bars(fig / f"{pref}_total_return_std.png", f"{title} 累计净收益 std across seeds", "模型", "std", names, stds, colors)
    sharpe_means = []
    for g in g_list:
        for mode, nice in MODES:
            row = summary[(summary["g"] == g) & (summary["agg"] == mode)]
            sharpe_means.append(float(row["sharpe_mean"].iloc[0]) if len(row) else float("nan"))
    plot_grouped_bars(fig / f"{pref}_sharpe_mean.png", f"{title} Sharpe mean", "模型", "Sharpe mean", names, sharpe_means, colors)

    g_show, mode_show = 32, "pred_utility"
    series, sn = {}, []
    for seed in noise_seeds[: min(8, len(noise_seeds))]:
        mid = _model_id(g_show, mode_show, seed)
        gg = daily[daily["model"] == mid].sort_values("date")
        if gg.empty:
            continue
        name = f"seed={seed}"
        sn.append(name)
        series[name] = (gg["date"].to_numpy(), np.cumsum(gg["net_return"].to_numpy(float)))
        SERIES_COLORS[name] = color_for("gen_standard_fm")
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
    a(f"# CSI500 FM Meta stability：G×聚合（{len(noise_seeds)} seeds）")
    a("")
    a(
        "设定：冻结 CSI500 Top30 最新 Alpha+FM（**不训练**）；"
        "**Meta 候选 = G 条 FM ∪ Teacher（MVO/MaxSharpe/RP）**；"
        f"**G ∈ {{{', '.join(str(x) for x in g_list)}}}**；"
        f"**seed ∈ {{1..{len(noise_seeds)}}}**；每日 `noise = hash(date, seed)`。"
        "同 (日,G,seed) 只建一次 Meta 池，四种聚合共享。"
    )
    a("")
    a(
        f"聚合：Pred-Utility / Mean / Top-k={TOP_K} / Softmax(τ={SOFTMAX_T})。"
        "下表为 **across seeds 的 mean±std（单利）**。"
    )
    a("")
    a(f"产物：`results/CSI500/{out_tag}/`。约 **{n_days}** 日 × {len(noise_seeds)} seeds。")
    a("")
    a("| G | Pred-Utility | Mean | Top-k | Softmax |")
    a("| --- | --- | --- | --- | --- |")
    for g in g_list:
        cells = []
        for mode, nice in MODES:
            row = summary[(summary["g"] == g) & (summary["agg"] == mode)]
            if row.empty:
                cells.append("-")
            else:
                cells.append(_fmt_pm(float(row["total_mean"].iloc[0]), float(row["total_std"].iloc[0])))
        a(f"| G={g} | " + " | ".join(cells) + " |")
    a("")
    a("| G | agg | n | total mean±std | min | max | sharpe mean±std | turnover mean |")
    a("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for _, r in summary.iterrows():
        a(
            f"| {int(r.g)} | {r.agg_label} | {int(r.n_seeds)} | "
            f"{_fmt_pm(r.total_mean, r.total_std)} | {_fmt(r.total_min)} | {_fmt(r.total_max)} | "
            f"{_fmt_pm(r.sharpe_mean, r.sharpe_std)} | {_fmt(r.turnover_mean)} |"
        )
    a("")
    for title_i, fn in [
        ("累计净收益 mean", f"{pref}_total_return_mean.png"),
        ("累计净收益 std", f"{pref}_total_return_std.png"),
        ("Sharpe mean", f"{pref}_sharpe_mean.png"),
        (f"G={g_show} Pred-Utility 多seed累计", f"{pref}_G{g_show}_pred_utility_seeds_cum.png"),
    ]:
        a(f"### {title_i}")
        a("")
        a(f"![{title_i}](figures/{fn})")
        a("")

    # best cell
    if len(summary):
        best = summary.sort_values("total_mean", ascending=False).iloc[0]
        a("### 结论")
        a("")
        a(
            f"- across-seed 均值最高：`G={int(best.g)} {best.agg_label}` = "
            f"`{_fmt_pm(best.total_mean, best.total_std)}`（range [{_fmt(best.total_min)}, {_fmt(best.total_max)}]）。"
        )
        a("")

    text = "\n".join(L)
    (dest_root / "analysis" / "CSI500_FM_Meta_stability报告.md").write_text(text, encoding="utf-8")
    Path("results/CSI500/CSI500_FM_Meta_stability实验报告.md").write_text(
        text.replace("(figures/", f"({out_tag}/analysis/figures/"), encoding="utf-8"
    )
    log("wrote FM Meta stability report")


def main():
    global POOL, N_SEEDS, MASTER_SEED, G_LIST
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="top30")
    ap.add_argument("--out-tag", default="strict_fixed_oos_fm_stability_meta_seeds")
    ap.add_argument("--src-tag", default="strict_fixed_oos")
    ap.add_argument("--months", default=",".join(MONTHS))
    ap.add_argument("--g-list", default=",".join(str(x) for x in G_LIST))
    ap.add_argument("--n-seeds", type=int, default=N_SEEDS)
    ap.add_argument("--master-seed", type=int, default=MASTER_SEED)
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()

    POOL = str(args.pool)
    N_SEEDS = int(args.n_seeds)
    MASTER_SEED = int(args.master_seed)
    G_LIST = [int(x.strip()) for x in args.g_list.split(",") if x.strip()]
    months = [m.strip() for m in args.months.split(",") if m.strip()]
    noise_seeds = make_noise_seeds(N_SEEDS, MASTER_SEED)

    set_seed(MASTER_SEED, False, True)
    dest_root = Path("results/CSI500") / args.out_tag
    dest_root.mkdir(parents=True, exist_ok=True)
    year_csv = dest_root / "backtest_daily_year.csv"
    log_path = dest_root / "run_all.log"

    if args.report_only:
        write_report(dest_root, year_csv, args.out_tag, noise_seeds, G_LIST)
        return

    os.environ.setdefault("GRPO_MARKET", "csi500")
    cfg = load_config()
    cfg.setdefault("portfolio", {})
    cfg["portfolio"]["allocation_pools"] = [POOL]
    cfg.setdefault("matrix", {})
    cfg["matrix"]["allocation_pools"] = [POOL]
    device = require_cuda()
    store = make_store(cfg)
    src_root = Path("results/CSI500") / args.src_tag

    t_all = time.time()
    frames = []
    with open(log_path, "a", encoding="utf-8") as flog:

        def _log(msg: str):
            log(msg)
            flog.write(msg + "\n")
            flog.flush()

        _log(
            f"CONFIG pool={POOL} g_list={G_LIST} n_seeds={N_SEEDS} seeds={noise_seeds} "
            f"modes={[m for m,_ in MODES]} Meta FM freeze out={dest_root}"
        )
        for month in months:
            month_csv = dest_root / OOS / f"test_month={month}" / "backtest_daily.csv"
            if month_csv.is_file():
                _log(f"SKIP {month}")
                frames.append(pd.read_csv(month_csv))
                continue
            _log(f"==== CSI FM-Meta-stability month={month} ====")
            t0 = time.time()
            exp = run_month(cfg, store, device, month, src_root, dest_root, noise_seeds, G_LIST)
            frames.append(pd.read_csv(exp / "backtest_daily.csv"))
            _log(f"DONE {month} hours={(time.time()-t0)/3600:.2f}")

        year = pd.concat(frames, ignore_index=True)
        year.to_csv(year_csv, index=False)
        write_report(dest_root, year_csv, args.out_tag, noise_seeds, G_LIST)
        _log(
            f"REQUESTED MONTHS CSI500-FM-META-STABILITY-SEEDS DONE "
            f"hours={(time.time()-t_all)/3600:.2f} out={dest_root}"
        )


if __name__ == "__main__":
    main()
