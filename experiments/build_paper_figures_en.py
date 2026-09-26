#!/usr/bin/env python3
"""Rebuild paper figures with English titles/axes and clean legends (no G=32 / Meta)."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "paper_figures_en"


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.unicode_minus": False,
            "figure.dpi": 130,
            "savefig.dpi": 140,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 8,
        }
    )
    return plt


def clean_label(name: str) -> str:
    s = str(name)
    # normalize separators
    s = s.replace("_", " ").replace("-", " ")
    # drop protocol noise
    for pat in [
        r"\bG\s*=?\s*\d+\b",
        r"\bmeta\b",
        r"\bMeta\b",
        r"\bsoftmax\b",
        r"\btop30\b",
        r"\ball\b",
        r"\bstrict[_\s]?fixed[_\s]?oos\b",
        r"\br06\b",
        r"\b\(default\)\b",
        r"\bdefault\b",
        r"\bpure\b",
        r"\bPure\b",
        r"\b旧\b",
    ]:
        s = re.sub(pat, " ", s, flags=re.I)
    s = re.sub(r"\s+", " ", s).strip(" _-/")

    # canonical map by tokens
    low = s.lower().replace(" ", "")
    mapping = [
        ("diffusion", "Diffusion"),
        ("standardfm", "FM"),
        ("genstandardfm", "FM"),
        ("fm", "FM"),
        ("gaussian", "Gaussian"),
        ("mlp", "MLP"),
        ("ssfm", "SS-FM"),
        ("gen_ssfm", "SS-FM"),
        ("genssfm", "SS-FM"),
        ("riskparity", "Risk Parity"),
        ("risk_parity", "Risk Parity"),
        ("maxsharpe", "Max Sharpe"),
        ("max_sharpe", "Max Sharpe"),
        ("mvo", "MVO"),
        ("rlssfmppocand", "SS-FM + PPO (cand)"),
        ("ppo_cand", "SS-FM + PPO (cand)"),
        ("ppocand", "SS-FM + PPO (cand)"),
        ("rlssfmppo", "SS-FM + PPO"),
        ("ppo", "SS-FM + PPO"),
        ("rlssfmgrpoinit", "SS-FM + GRPO (init)"),
        ("grpo_init", "SS-FM + GRPO (init)"),
        ("grpoinit", "SS-FM + GRPO (init)"),
        ("rlssfmgrpo", "SS-FM + GRPO"),
        ("grpo", "SS-FM + GRPO"),
        ("teacheraware", "Teacher-aware"),
        ("teacher_aware", "Teacher-aware"),
        ("random", "Random"),
        ("pure", "Pure"),
    ]
    # prefer longer / more specific matches via ordered checks on original lower
    raw = str(name).lower()
    if "ppo_cand" in raw or "ppo (cand)" in raw or "ppocand" in raw.replace("_", ""):
        return "SS-FM + PPO (cand)"
    if "grpo_init" in raw or "grpo (init)" in raw or "grpo (ssfm init)" in raw or "ssfm init" in raw:
        return "SS-FM + GRPO (init)"
    if "ppo" in raw and "ssfm" in raw.replace("-", ""):
        return "SS-FM + PPO"
    if "grpo" in raw and "ssfm" in raw.replace("-", ""):
        return "SS-FM + GRPO"
    if "risk" in raw and "parity" in raw:
        return "Risk Parity"
    if "max" in raw and "sharpe" in raw:
        return "Max Sharpe"
    if re.search(r"\bmvo\b", raw):
        return "MVO"
    if "diffusion" in raw:
        return "Diffusion"
    if "gaussian" in raw:
        return "Gaussian"
    if re.search(r"(^|[^a-z])mlp([^a-z]|$)", raw) or raw.startswith("mlp"):
        return "MLP"
    if "standard_fm" in raw or "gen_standard_fm" in raw or re.search(r"(^|[^a-z])fm([^a-z]|$)", raw):
        if "ssfm" not in raw.replace("-", "") and "ss-fm" not in raw:
            return "FM"
    if "ssfm" in raw.replace("-", "") or "ss-fm" in raw:
        if "ppo" not in raw and "grpo" not in raw:
            return "SS-FM"
    if "teacher" in raw and "aware" in raw:
        return "Teacher-aware"
    if raw.strip() in {"random", "sampling random"}:
        return "Random"
    if "pure" in raw and "ssfm" not in raw.replace("-", ""):
        return "Pure"
    # fallback cleanup
    s = re.sub(r"\bgen\b", "", s, flags=re.I)
    s = re.sub(r"\brl\b", "", s, flags=re.I)
    s = re.sub(r"\s+", " ", s).strip()
    return s or str(name)


COLORS = {
    "Diffusion": "#E45756",
    "FM": "#4C78A8",
    "SS-FM": "#72B7B2",
    "MLP": "#F58518",
    "Gaussian": "#B279A2",
    "MVO": "#54A24B",
    "Max Sharpe": "#EECA3B",
    "Risk Parity": "#B279A2",
    "SS-FM + PPO": "#E45756",
    "SS-FM + PPO (cand)": "#FF9D98",
    "SS-FM + GRPO": "#4C78A8",
    "SS-FM + GRPO (init)": "#9ECAE9",
    "Random": "#4C78A8",
    "Teacher-aware": "#E45756",
    "Pure": "#4C78A8",
}


def color(label: str) -> str:
    return COLORS.get(label, "#7F7F7F")


def save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    _plt().close(fig)


def plot_cum_from_daily(
    path: Path,
    daily: pd.DataFrame,
    model_col: str,
    keep: Optional[Sequence[str]] = None,
    title: str = "Cumulative net return",
    order: Optional[Sequence[str]] = None,
) -> None:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    df = daily.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["_lab"] = df[model_col].map(clean_label)
    if keep is not None:
        keep_set = set(keep)
        df = df[df["_lab"].isin(keep_set)]
    labs = list(order) if order else sorted(df["_lab"].unique())
    for lab in labs:
        sub = df[df["_lab"] == lab].sort_values("date")
        if sub.empty:
            continue
        y = sub["net_return"].astype(float).cumsum().to_numpy()
        ax.plot(sub["date"], y, lw=1.8, color=color(lab), label=lab)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative net return")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", frameon=False, ncol=2)
    save(fig, path)


def plot_bars(
    path: Path,
    labels: Sequence[str],
    values: Sequence[float],
    title: str,
    ylabel: str,
    xlabel: str = "",
    errs: Optional[Sequence[float]] = None,
) -> None:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    xs = np.arange(len(labels))
    cols = [color(clean_label(l)) for l in labels]
    ax.bar(xs, values, color=cols, yerr=errs, capsize=3 if errs is not None else 0, alpha=0.9)
    ax.set_xticks(xs)
    ax.set_xticklabels([clean_label(l) for l in labels], rotation=20, ha="right")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.3)
    save(fig, path)


def plot_lines_xy(
    path: Path,
    series: Dict[str, Tuple[np.ndarray, np.ndarray]],
    title: str,
    xlabel: str,
    ylabel: str,
    order: Optional[Sequence[str]] = None,
) -> None:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    keys = list(order) if order else list(series.keys())
    for k in keys:
        if k not in series:
            continue
        x, y = series[k]
        ax.plot(x, y, lw=1.8, marker="o", color=color(clean_label(k)), label=clean_label(k))
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    save(fig, path)


def plot_seed_cum(
    path: Path,
    daily: pd.DataFrame,
    title: str,
    g: int = 32,
    agg: str = "pred_utility",
) -> None:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    df = daily.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df[(df["g"].astype(int) == g) & (df["agg"] == agg)]
    seeds = sorted(df["noise_seed"].astype(int).unique())
    mats = []
    dates = None
    for s in seeds:
        sub = df[df["noise_seed"].astype(int) == s].sort_values("date")
        if sub.empty:
            continue
        y = sub["net_return"].astype(float).cumsum().to_numpy()
        dates = sub["date"].to_numpy()
        ax.plot(dates, y, color="#9ECAE9", lw=0.9, alpha=0.55)
        mats.append(y)
    if mats and dates is not None:
        M = np.vstack(mats)
        ax.plot(dates, M.mean(axis=0), color="#4C78A8", lw=2.4, label="Mean across seeds")
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative net return")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    save(fig, path)


def plot_mean_std_by_g(
    path: Path,
    summary: pd.DataFrame,
    title: str,
    agg: str = "pred_utility",
) -> None:
    sub = summary[summary["agg"] == agg].sort_values("g")
    plot_lines_xy(
        path,
        {
            "Mean": (sub["g"].to_numpy(), sub["total_mean"].to_numpy()),
        },
        title=title,
        xlabel="Number of candidates",
        ylabel="Total net return (mean across seeds)",
    )
    # redraw with error bars
    plt = _plt()
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.errorbar(
        sub["g"].to_numpy(),
        sub["total_mean"].to_numpy(),
        yerr=sub["total_std"].to_numpy(),
        fmt="-o",
        color="#4C78A8",
        capsize=4,
        lw=1.8,
    )
    ax.set_title(title)
    ax.set_xlabel("Number of candidates")
    ax.set_ylabel("Total net return (mean ± std)")
    ax.grid(True, alpha=0.3)
    save(fig, path)


def retitle_existing_png(src: Path, dst: Path, title: str) -> None:
    """Keep image body; replace top title strip with English title."""
    plt = _plt()
    from PIL import Image

    img = Image.open(src).convert("RGB")
    w, h = img.size
    # crop original title band (~9%)
    crop = img.crop((0, int(0.09 * h), w, h))
    fig_h = 5.6
    fig_w = fig_h * (w / (h * 0.91))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.imshow(crop)
    ax.set_title(title)
    ax.axis("off")
    save(fig, dst)


def load_daily(path: Path, usecols: Optional[List[str]] = None) -> pd.DataFrame:
    cols = usecols or ["date", "net_return", "model"]
    # allow extra filter cols if present
    header = pd.read_csv(path, nrows=0).columns.tolist()
    use = [c for c in cols if c in header]
    for extra in ("noise_seed", "g", "agg", "strategy_name"):
        if extra in header and extra not in use:
            use.append(extra)
    return pd.read_csv(path, usecols=use)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"Writing English paper figures -> {OUT}")

    # ---- Exp1 coverage / diversity ----
    s1 = pd.read_csv(
        ROOT
        / "results/S&P500/strict_fixed_oos_top30_ssfm_teacher_exps_s50/exp1_diagnose/tables/exp1_summary_by_G.csv"
    )
    plot_lines_xy(
        OUT / "exp1_coverage_by_candidates.png",
        {"SS-FM": (s1["g"].to_numpy(), s1["coverage_mean"].to_numpy())},
        title="Teacher coverage rate vs number of candidates",
        xlabel="Number of candidates",
        ylabel="Coverage rate",
    )
    plot_lines_xy(
        OUT / "exp1_diversity_by_candidates.png",
        {"SS-FM": (s1["g"].to_numpy(), s1["diversity_mean"].to_numpy())},
        title="Pairwise diversity vs number of candidates",
        xlabel="Number of candidates",
        ylabel="Mean pairwise L1 diversity",
    )
    retitle_existing_png(
        ROOT
        / "results/S&P500/strict_fixed_oos_top30_ssfm_teacher_exps_s50/exp1_diagnose/figures/exp1_clr_pca_G32.png",
        OUT / "exp1_clr_pca.png",
        "CLR–PCA of SS-FM samples and teachers",
    )

    # ---- Exp2 ----
    e2 = pd.read_csv(
        ROOT
        / "results/S&P500/strict_fixed_oos_top30_ssfm_teacher_exps_s50/exp2_sampling/tables/exp2_summary.csv"
    )
    plt = _plt()
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    for samp, lab in [("random", "Random"), ("teacher_aware", "Teacher-aware")]:
        sub = e2[(e2["sampling"] == samp) & (e2["agg"] == "pred_utility")].sort_values("g")
        ax.errorbar(
            sub["g"],
            sub["total_mean"],
            yerr=sub["total_std"],
            fmt="-o",
            label=lab,
            color=color(lab),
            capsize=3,
            lw=1.8,
        )
    ax.set_title("OOS total return by sampling rule (Pred-Utility)")
    ax.set_xlabel("Number of candidates")
    ax.set_ylabel("Total net return (mean ± std)")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    save(fig, OUT / "exp2_pred_utility_total_mean.png")

    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    width = 0.35
    cov = e2.drop_duplicates(["sampling", "g"])[["sampling", "g", "coverage_mean"]]
    gs = sorted(cov["g"].unique())
    xs = np.arange(len(gs))
    for i, (samp, lab) in enumerate([("random", "Random"), ("teacher_aware", "Teacher-aware")]):
        vals = [
            float(cov[(cov.sampling == samp) & (cov.g == g)]["coverage_mean"].iloc[0])
            for g in gs
        ]
        ax.bar(xs + (i - 0.5) * width, vals, width=width, label=lab, color=color(lab), alpha=0.9)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(g) for g in gs])
    ax.set_title("Teacher coverage by sampling rule")
    ax.set_xlabel("Number of candidates")
    ax.set_ylabel("Coverage rate")
    ax.legend(frameon=False)
    ax.grid(True, axis="y", alpha=0.3)
    save(fig, OUT / "exp2_coverage_by_sampling.png")

    # ---- Exp3 four-way bars at fixed candidate count 32 ----
    e3 = pd.read_csv(
        ROOT
        / "results/S&P500/strict_fixed_oos_top30_ssfm_teacher_exps_s50/exp3_retrain/tables/exp3_summary.csv"
    )
    sub = e3[(e3["g"] == 32) & (e3["agg"] == "pred_utility")].copy()
    labels = []
    vals = []
    errs = []
    for _, r in sub.iterrows():
        m = "Pure" if r["model_tag"] == "pure" else "Teacher-aware model"
        s = "Random" if r["sampling"] == "random" else "Teacher-aware sampling"
        labels.append(f"{m}\n+ {s}")
        vals.append(float(r["total_mean"]))
        errs.append(float(r["total_std"]))
    plt = _plt()
    fig, ax = plt.subplots(figsize=(10.0, 5.0))
    xs = np.arange(len(labels))
    ax.bar(xs, vals, yerr=errs, capsize=4, color=["#4C78A8", "#E45756", "#72B7B2", "#F58518"][: len(labels)], alpha=0.9)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_title("Four-way comparison under Pred-Utility")
    ax.set_ylabel("Total net return (mean ± std)")
    ax.grid(True, axis="y", alpha=0.3)
    save(fig, OUT / "exp3_pred_utility_four_way.png")

    # ---- CSI Pure SS-FM vs default gens ----
    d = load_daily(ROOT / "results/CSI500/strict_fixed_oos_ssfm_g32_vs_default_gens/analysis/tables/daily.csv")
    plot_cum_from_daily(
        OUT / "csi_ssfm_vs_default_gens_cum.png",
        d,
        "model",
        title="CSI500: cumulative net return of generative models",
        order=["SS-FM", "FM", "Gaussian", "MLP", "Diffusion"],
    )

    # ---- CSI FM vs SS-FM across candidate counts (monthly → cum) ----
    monthly = pd.read_csv(ROOT / "results/CSI500/strict_fixed_oos/analysis/全年总览/tables/fm_ssfm_g_monthly.csv")
    plt = _plt()
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    # Only a few clean series; label by family + candidate count number only
    for col, lab in [
        ("SS-FM G=8", "SS-FM · 8"),
        ("SS-FM G=32", "SS-FM · 32"),
        ("SS-FM G=128", "SS-FM · 128"),
        ("FM G=16", "FM · 16"),
        ("FM G=32", "FM · 32"),
        ("FM G=128", "FM · 128"),
    ]:
        if col not in monthly.columns:
            continue
        y = monthly[col].astype(float).cumsum().to_numpy()
        x = pd.to_datetime(monthly["month"] + "-01")
        ax.plot(x, y, lw=1.8, label=lab)
    ax.set_title("CSI500: FM vs SS-FM across candidate counts")
    ax.set_xlabel("Month")
    ax.set_ylabel("Cumulative net return")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, ncol=2)
    save(fig, OUT / "csi_fm_vs_ssfm_candidates_cum.png")

    # ---- CSI Meta gens / Softmax / Meta vs default ----
    meta_daily = load_daily(ROOT / "results/CSI500/strict_fixed_oos_g32_meta/backtest_daily_year.csv")
    plot_cum_from_daily(
        OUT / "csi_gens_pred_utility_cum.png",
        meta_daily[meta_daily["model"].astype(str).str.contains("diffusion|fm_|gaussian|mlp|ssfm", case=False, regex=True)],
        "model",
        title="CSI500: generative models under Pred-Utility aggregation",
        order=["Diffusion", "Gaussian", "FM", "SS-FM", "MLP"],
    )
    # teachers only panel reused later

    soft = load_daily(ROOT / "results/CSI500/strict_fixed_oos_g32_meta_softmax/backtest_daily_year.csv")
    plot_cum_from_daily(
        OUT / "csi_gens_softmax_cum.png",
        soft,
        "model",
        title="CSI500: generative models under Softmax aggregation",
        order=["Diffusion", "FM", "SS-FM", "MLP", "Gaussian"],
    )

    meta_vs = load_daily(
        ROOT / "results/CSI500/strict_fixed_oos_ssfm_g32_meta_vs_default_gens/analysis/tables/csi_ssfm_meta_vs_default_gens_daily.csv"
    )
    plot_cum_from_daily(
        OUT / "csi_ssfm_with_teachers_vs_default_gens_cum.png",
        meta_vs,
        "model",
        title="CSI500: SS-FM with teachers vs default generative models",
        order=["FM", "Gaussian", "MLP", "Diffusion", "SS-FM"],
    )

    sp_soft = load_daily(ROOT / "results/S&P500/strict_fixed_oos_top30_g32_meta_softmax/backtest_daily_year.csv")
    plot_cum_from_daily(
        OUT / "sp_gens_softmax_cum.png",
        sp_soft,
        "model",
        title="S&P500: generative models under Softmax aggregation",
        order=["SS-FM", "FM", "MLP", "Gaussian", "Diffusion"],
    )

    # ---- Stability seed curves / mean±std ----
    plot_seed_cum(
        OUT / "csi_diffusion_seed_cum_pred_utility.png",
        load_daily(
            ROOT / "results/CSI500/strict_fixed_oos_diff_stability_meta_seeds/backtest_daily_year.csv",
            ["date", "net_return", "model", "noise_seed", "g", "agg"],
        ),
        title="CSI500 Diffusion: cumulative return across seeds (Pred-Utility)",
    )
    plot_mean_std_by_g(
        OUT / "csi_diffusion_no_teacher_return_by_candidates.png",
        pd.read_csv(
            ROOT / "results/CSI500/strict_fixed_oos_diff_stability_seeds/analysis/tables/diff_stability_seeds_summary.csv"
        ),
        title="CSI500 Diffusion (samples only): return vs candidates",
    )
    plot_mean_std_by_g(
        OUT / "csi_ssfm_no_teacher_return_by_candidates.png",
        pd.read_csv(
            ROOT / "results/CSI500/strict_fixed_oos_ssfm_stability_seeds/analysis/tables/ssfm_stability_seeds_summary.csv"
        ),
        title="CSI500 SS-FM (samples only): return vs candidates",
    )
    plot_seed_cum(
        OUT / "sp_ssfm_seed_cum_pred_utility.png",
        load_daily(
            ROOT / "results/S&P500/strict_fixed_oos_top30_ssfm_stability_meta_seeds_rerun/backtest_daily_year.csv",
            ["date", "net_return", "model", "noise_seed", "g", "agg"],
        ),
        title="S&P500 SS-FM: cumulative return across seeds (Pred-Utility)",
    )

    # ---- Teachers vs gens ----
    year = load_daily(ROOT / "results/CSI500/strict_fixed_oos/final_summary/strict_fixed_oos/backtest_daily.csv")
    teacher_models = [
        "baseline_mvo_top30",
        "baseline_max_sharpe_top30",
        "baseline_risk_parity_top30",
        "ssfm_G32_top30",
    ]
    plot_cum_from_daily(
        OUT / "csi_teachers_vs_ssfm_cum.png",
        year[year["model"].isin(teacher_models)],
        "model",
        title="CSI500: teachers vs SS-FM",
        order=["SS-FM", "Risk Parity", "Max Sharpe", "MVO"],
    )

    sp_meta = load_daily(ROOT / "results/S&P500/strict_fixed_oos_top30_g32_meta/backtest_daily_year.csv")
    plot_cum_from_daily(
        OUT / "sp_teachers_vs_gens_cum.png",
        sp_meta[
            sp_meta["model"].isin(
                [
                    "baseline_mvo_top30",
                    "baseline_max_sharpe_top30",
                    "baseline_risk_parity_top30",
                    "fm_G32_meta_top30",
                    "ssfm_G32_meta_top30",
                ]
            )
        ],
        "model",
        title="S&P500: teachers vs generative models",
        order=["MVO", "SS-FM", "Max Sharpe", "FM", "Risk Parity"],
    )

    # ---- RL CSI ----
    rl_g32 = load_daily(ROOT / "results/CSI500/strict_fixed_oos_rl_g32/backtest_daily_year.csv")
    pure = year[year["model"] == "ssfm_G32_top30"].copy()
    rl_plot = pd.concat([pure, rl_g32], ignore_index=True)
    plot_cum_from_daily(
        OUT / "csi_rl_vs_ssfm_cum.png",
        rl_plot,
        "model",
        title="CSI500: SS-FM vs RL policies",
        order=["SS-FM + PPO", "SS-FM + GRPO", "SS-FM"],
    )

    abl = load_daily(ROOT / "results/CSI500/strict_fixed_oos_rl_ablation_g32/backtest_daily_year.csv")
    # also pull old PPO/GRPO from rl_g32 and pure
    abl_plot = pd.concat(
        [
            pure,
            rl_g32,
            abl,
        ],
        ignore_index=True,
    )
    plot_cum_from_daily(
        OUT / "csi_rl_ablation_cum.png",
        abl_plot,
        "model",
        title="CSI500: RL ablation",
        order=["SS-FM + PPO", "SS-FM + PPO (cand)", "SS-FM + GRPO", "SS-FM + GRPO (init)", "SS-FM"],
    )

    # ---- SP RL ----
    plot_cum_from_daily(
        OUT / "sp_rl_vs_ssfm_cum.png",
        sp_meta[
            sp_meta["model"].isin(
                ["ssfm_G32_meta_top30", "rl_ssfm_ppo_cand_G32_top30", "rl_ssfm_grpo_init_G32_top30"]
            )
        ],
        "model",
        title="S&P500: SS-FM vs RL policies",
        order=["SS-FM + PPO (cand)", "SS-FM", "SS-FM + GRPO (init)"],
    )

    # r06 multi-seed mean curve + seed curves
    seed_frames = []
    for seed in (42, 43, 44):
        p = ROOT / f"results/S&P500/strict_fixed_oos_top30_rl_r06_stability/seeds/seed={seed}/backtest_daily_year.csv"
        sdf = load_daily(p)
        sdf["seed"] = seed
        seed_frames.append(sdf)
    r06 = pd.concat(seed_frames, ignore_index=True)
    r06["date"] = pd.to_datetime(r06["date"])
    r06["_lab"] = r06["model"].map(clean_label)

    plt = _plt()
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    for lab in ["SS-FM + PPO (cand)", "SS-FM + GRPO (init)", "SS-FM"]:
        mats = []
        dates = None
        for seed, g in r06.groupby("seed"):
            sub = g[g["_lab"] == lab].sort_values("date")
            if sub.empty:
                continue
            y = sub["net_return"].astype(float).cumsum().to_numpy()
            dates = sub["date"].to_numpy()
            mats.append(y)
        if mats and dates is not None:
            M = np.vstack(mats)
            ax.plot(dates, M.mean(axis=0), lw=2.2, color=color(lab), label=lab)
            ax.fill_between(dates, M.min(axis=0), M.max(axis=0), color=color(lab), alpha=0.15)
    ax.set_title("S&P500 RL stability: mean cumulative return across seeds")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative net return")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    save(fig, OUT / "sp_rl_stability_mean_cum.png")

    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    for seed, g in r06.groupby("seed"):
        for lab, ls in [("SS-FM + PPO (cand)", "-"), ("SS-FM", "--")]:
            sub = g[g["_lab"] == lab].sort_values("date")
            if sub.empty:
                continue
            y = sub["net_return"].astype(float).cumsum().to_numpy()
            ax.plot(
                sub["date"],
                y,
                ls=ls,
                lw=1.5,
                color=color(lab),
                alpha=0.85,
                label=f"{lab} · seed {seed}",
            )
    ax.set_title("S&P500 RL stability: per-seed cumulative return")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative net return")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=7, ncol=2)
    save(fig, OUT / "sp_rl_stability_seeds_cum.png")

    print("Done. Figures:")
    for p in sorted(OUT.glob("*.png")):
        print(" ", p.name)


if __name__ == "__main__":
    main()
