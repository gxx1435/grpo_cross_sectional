#!/usr/bin/env python3
"""SP500 Top30 RL-r06 multi-seed stability: SS-FM Meta vs cand-PPO vs GRPO-init.

Same protocol as rerun_sp500_rl_r06.py (G=32, Meta pool, reward-r06, bc_coef=0),
but re-runs the full year under several global RNG seeds and reports mean±std.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from backtest.drawdown import hard_mdd
from evaluation.analysis_spec import SERIES_COLORS, color_for
from evaluation.plots import plot_grouped_bars, plot_series
from experiments.rerun_sp500_rl_r06 import (
    G_DEFAULT,
    MONTHS,
    POOL as POOL_DEFAULT,
    REWARD,
    run_month,
)
from utils.config import load_config, make_store
from utils.gpu import require_cuda
from utils.logging import log, write_json
from utils.seed import set_seed

# Import mutates module-level POOL in r06; keep a local handle.
import experiments.rerun_sp500_rl_r06 as r06

DEFAULT_SEEDS = [42, 43, 44]


def _summarize_one(daily: pd.DataFrame, model: str) -> dict:
    g = daily[daily["model"] == model].sort_values("date")
    r = g["net_return"].to_numpy(float)
    mu = float(np.nanmean(r)) if len(r) else float("nan")
    sd = float(np.nanstd(r, ddof=0)) + 1e-12 if len(r) else float("nan")
    return dict(
        model=model,
        total_net_return=float(np.nansum(r)) if len(r) else float("nan"),
        ann_return=float(mu * 252) if len(r) else float("nan"),
        sharpe=float(np.sqrt(252) * mu / sd) if len(r) else float("nan"),
        max_drawdown=hard_mdd(r) if len(r) else float("nan"),
        mean_turnover=float(g["turnover"].mean()) if len(g) and "turnover" in g else float("nan"),
        n_days=int(len(r)),
    )


def _fmt(x, nd=4):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "-"
    return f"{float(x):.{nd}f}"


def write_stability_report(dest_root: Path, raw: pd.DataFrame, g: int, seeds: List[int], out_tag: str) -> None:
    fig = dest_root / "analysis" / "figures"
    tab = dest_root / "analysis" / "tables"
    fig.mkdir(parents=True, exist_ok=True)
    tab.mkdir(parents=True, exist_ok=True)

    rename = {
        f"ssfm_G{g}_meta_top30": f"SS-FM G={g} Meta",
        f"rl_ssfm_ppo_cand_G{g}_r06_top30": f"SS-FM+PPO (cand) G={g} r06",
        f"rl_ssfm_grpo_init_G{g}_r06_top30": f"SS-FM+GRPO (SSFM init) G={g} r06",
    }
    names = list(rename.values())
    order = list(rename.keys())

    raw = raw.copy()
    raw["label"] = raw["model"].map(rename)
    raw.to_csv(tab / "rl_r06_stability_raw.csv", index=False)

    rows = []
    for mid, label in zip(order, names):
        sub = raw[raw["model"] == mid]
        rows.append(
            dict(
                model=label,
                n_seeds=int(len(sub)),
                total_mean=float(sub["total_net_return"].mean()),
                total_std=float(sub["total_net_return"].std(ddof=0)),
                total_min=float(sub["total_net_return"].min()),
                total_max=float(sub["total_net_return"].max()),
                sharpe_mean=float(sub["sharpe"].mean()),
                sharpe_std=float(sub["sharpe"].std(ddof=0)),
                mdd_mean=float(sub["max_drawdown"].mean()),
                turnover_mean=float(sub["mean_turnover"].mean()),
                turnover_std=float(sub["mean_turnover"].std(ddof=0)),
            )
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(tab / "rl_r06_stability_summary.csv", index=False)

    pivot = raw.pivot_table(index="seed", columns="label", values="total_net_return", aggfunc="first")
    pivot = pivot.reindex(columns=names)
    pivot.to_csv(tab / "rl_r06_stability_pivot_return.csv")

    cm = {
        names[0]: color_for("gen_ssfm"),
        names[1]: "#E45756",
        names[2]: "#4C78A8",
    }
    for k, v in cm.items():
        SERIES_COLORS[k] = v
    colors = [cm[n] for n in names]

    title = f"Top30 G={g} RL-r06 多seed稳定性（n={len(seeds)}）"
    pref = "rl_r06_stability"
    plot_grouped_bars(
        fig / f"{pref}_total_mean.png",
        f"{title} 累计净收益 mean",
        "模型",
        "累计净收益 mean",
        names,
        [float(summary.loc[summary.model == n, "total_mean"].iloc[0]) for n in names],
        colors,
    )
    # error-bar style via clustered: mean as primary; also dump seed curves if year dailies exist
    seed_dirs = sorted((dest_root / "seeds").glob("seed=*"))
    series = {}
    for sd in seed_dirs:
        seed = int(sd.name.split("=")[1])
        ycsv = sd / "backtest_daily_year.csv"
        if not ycsv.is_file():
            continue
        d = pd.read_csv(ycsv)
        d["date"] = pd.to_datetime(d["date"])
        for mid, label in zip(order, names):
            gg = d[d["model"] == mid].sort_values("date")
            if gg.empty:
                continue
            key = f"{label} s{seed}"
            series[key] = (gg["date"].to_numpy(), np.cumsum(gg["net_return"].to_numpy(float)))
            SERIES_COLORS[key] = cm[label]
    if series:
        plot_series(
            fig / f"{pref}_cumulative_seeds.png",
            f"{title} 各seed累计净收益",
            "交易日",
            "累计净收益（Σ）",
            list(series.keys()),
            series,
        )

    L = []
    a = L.append
    a(f"## Top30 G={g} RL-r06 多seed稳定性：SS-FM vs PPO / GRPO")
    a("")
    a(
        f"设定：冻结 Top30 Alpha+SS-FM；**G={g}**；Pure SS-FM=Meta（样本∪Teacher→`pred_utility`）；"
        f"PPO=cand；GRPO=SSFM init；`bc_coef=0`；reward-r06；**seeds={seeds}**。"
    )
    a("")
    a(f"产物：`results/S&P500/{out_tag}/`。指标为全年单利 Σ 的跨seed统计。")
    a("")
    a("| model | n | total mean±std | min | max | sharpe mean±std | turnover mean |")
    a("| --- | --- | --- | --- | --- | --- | --- |")
    for _, r in summary.iterrows():
        a(
            f"| {r.model} | {int(r.n_seeds)} | {_fmt(r.total_mean)}±{_fmt(r.total_std)} | "
            f"{_fmt(r.total_min)} | {_fmt(r.total_max)} | "
            f"{_fmt(r.sharpe_mean)}±{_fmt(r.sharpe_std)} | {_fmt(r.turnover_mean)} |"
        )
    a("")
    a("| seed | " + " | ".join(names) + " |")
    a("| --- | " + " | ".join(["---"] * len(names)) + " |")
    for seed, row in pivot.iterrows():
        a("| " + str(int(seed)) + " | " + " | ".join(_fmt(row[n]) for n in names) + " |")
    a("")
    a(f"#### {title} 累计净收益 mean")
    a("")
    a(f"![{title} mean](figures/{pref}_total_mean.png)")
    a("")
    if series:
        a(f"#### {title} 各seed累计曲线")
        a("")
        a(f"![{title} seeds](figures/{pref}_cumulative_seeds.png)")
        a("")

    # ranking stability
    ranks = pivot.rank(axis=1, ascending=False)
    win = (ranks == 1).sum()
    a("### 稳定性结论")
    a("")
    best = summary.sort_values("total_mean", ascending=False).iloc[0]
    a(
        f"- 跨seed均值最高：`{best.model}` = `{best.total_mean:.4f}±{best.total_std:.4f}` "
        f"（range [{best.total_min:.4f}, {best.total_max:.4f}]）。"
    )
    for n in names:
        a(f"- `{n}` 夺冠次数：{int(win.get(n, 0))}/{len(seeds)}。")
    a(
        f"- 波动（std/|mean|）："
        + "；".join(
            f"{r.model}={abs(r.total_std)/max(abs(r.total_mean),1e-9):.2f}"
            for _, r in summary.iterrows()
        )
        + "。"
    )
    a("")

    text = "\n".join(L)
    (dest_root / "analysis" / "rl_r06_stability报告.md").write_text(text, encoding="utf-8")
    (Path("results/S&P500") / f"SP500_top30_rl_r06_stability实验报告.md").write_text(
        text.replace("(figures/", f"({out_tag}/analysis/figures/"),
        encoding="utf-8",
    )
    log(f"wrote stability report → {dest_root / 'analysis' / 'rl_r06_stability报告.md'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--g", type=int, default=G_DEFAULT)
    ap.add_argument("--pool", default=POOL_DEFAULT)
    ap.add_argument("--out-tag", default="strict_fixed_oos_top30_rl_r06_stability")
    ap.add_argument("--src-tag", default="strict_fixed_oos_top30")
    ap.add_argument("--months", default=",".join(MONTHS))
    ap.add_argument("--seeds", default=",".join(str(s) for s in DEFAULT_SEEDS))
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()

    r06.POOL = str(args.pool)
    g = int(args.g)
    months = [m.strip() for m in args.months.split(",") if m.strip()]
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    dest_root = Path("results/S&P500") / args.out_tag
    dest_root.mkdir(parents=True, exist_ok=True)
    seeds_root = dest_root / "seeds"
    seeds_root.mkdir(parents=True, exist_ok=True)
    log_path = dest_root / "run_all.log"
    raw_path = dest_root / "analysis" / "tables" / "rl_r06_stability_raw.csv"

    if args.report_only:
        raw = pd.read_csv(raw_path)
        write_stability_report(dest_root, raw, g, seeds, args.out_tag)
        return

    os.environ.setdefault("GRPO_MARKET", "sp500")
    cfg = load_config()
    cfg.setdefault("portfolio", {})
    cfg["portfolio"]["allocation_pools"] = [r06.POOL]
    cfg.setdefault("matrix", {})
    cfg["matrix"]["allocation_pools"] = [r06.POOL]
    device = require_cuda()
    store = make_store(cfg)
    src_root = Path("results/S&P500") / args.src_tag

    model_ids = [
        f"ssfm_G{g}_meta_{r06.POOL}",
        f"rl_ssfm_ppo_cand_G{g}_r06_{r06.POOL}",
        f"rl_ssfm_grpo_init_G{g}_r06_{r06.POOL}",
    ]

    def _collect_raw(done_seeds: List[int]) -> pd.DataFrame:
        rows: List[dict] = []
        for s in done_seeds:
            ycsv = seeds_root / f"seed={s}" / "backtest_daily_year.csv"
            daily = pd.read_csv(ycsv)
            for mid in model_ids:
                row = _summarize_one(daily, mid)
                row["seed"] = int(s)
                rows.append(row)
        return pd.DataFrame(rows)

    def _emit_table(raw: pd.DataFrame, done_seeds: List[int], _log) -> None:
        (dest_root / "analysis" / "tables").mkdir(parents=True, exist_ok=True)
        raw.to_csv(raw_path, index=False)
        write_stability_report(dest_root, raw, g, done_seeds, args.out_tag)
        # compact console / log table
        rename = {
            f"ssfm_G{g}_meta_{r06.POOL}": "SS-FM Meta",
            f"rl_ssfm_ppo_cand_G{g}_r06_{r06.POOL}": "SS-FM+PPO",
            f"rl_ssfm_grpo_init_G{g}_r06_{r06.POOL}": "SS-FM+GRPO",
        }
        pivot = raw.assign(label=raw["model"].map(rename)).pivot_table(
            index="seed", columns="label", values="total_net_return", aggfunc="first"
        )
        cols = [c for c in ["SS-FM Meta", "SS-FM+PPO", "SS-FM+GRPO"] if c in pivot.columns]
        pivot = pivot.reindex(columns=cols)
        _log(f"---- interim table after seeds={done_seeds} ----")
        _log(pivot.to_string(float_format=lambda x: f"{x:.4f}"))
        means = pivot.mean()
        stds = pivot.std(ddof=0)
        _log(
            "mean±std | "
            + " | ".join(f"{c}: {means[c]:.4f}±{stds[c]:.4f}" for c in cols)
        )
        interim_md = dest_root / "analysis" / f"interim_after_seed_{done_seeds[-1]}.md"
        lines = [
            f"# Interim after seed={done_seeds[-1]} (done={done_seeds})",
            "",
            "| seed | " + " | ".join(cols) + " |",
            "| --- | " + " | ".join(["---"] * len(cols)) + " |",
        ]
        for seed_i, row in pivot.iterrows():
            lines.append("| " + str(int(seed_i)) + " | " + " | ".join(f"{float(row[c]):.4f}" for c in cols) + " |")
        lines.extend(["", "| model | mean | std |", "| --- | --- | --- |"])
        for c in cols:
            lines.append(f"| {c} | {means[c]:.4f} | {stds[c]:.4f} |")
        interim_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
        _log(f"wrote {interim_md}")

    t_all = time.time()
    done_seeds: List[int] = []
    with open(log_path, "a", encoding="utf-8") as flog:

        def _log(msg: str):
            log(msg)
            flog.write(msg + "\n")
            flog.flush()

        _log(
            f"CONFIG pool={r06.POOL} g={g} reward={REWARD} seeds={seeds} "
            f"months={len(months)} out={dest_root}"
        )
        for seed in seeds:
            seed_dest = seeds_root / f"seed={seed}"
            seed_dest.mkdir(parents=True, exist_ok=True)
            year_csv = seed_dest / "backtest_daily_year.csv"
            if year_csv.is_file():
                _log(f"SKIP seed={seed} (year csv exists)")
            else:
                set_seed(seed, False, True)
                _log(f"==== SEED {seed} start ====")
                t_seed = time.time()
                frames = []
                for month in months:
                    _log(f"==== seed={seed} month={month} ====")
                    t0 = time.time()
                    exp = run_month(cfg, store, device, month, src_root, seed_dest, g)
                    frames.append(pd.read_csv(exp / "backtest_daily.csv"))
                    _log(f"DONE seed={seed} {month} hours={(time.time()-t0)/3600:.2f}")
                daily = pd.concat(frames, ignore_index=True)
                daily.to_csv(year_csv, index=False)
                write_json(
                    seed_dest / "meta.json",
                    {"seed": seed, "g": g, "pool": r06.POOL, "reward": dict(REWARD), "months": months},
                )
                _log(f"DONE SEED {seed} hours={(time.time()-t_seed)/3600:.2f}")

            done_seeds.append(int(seed))
            raw = _collect_raw(done_seeds)
            _emit_table(raw, done_seeds, _log)

        _log(
            f"REQUESTED MONTHS SP500-TOP30-RL-R06-STABILITY DONE "
            f"hours={(time.time()-t_all)/3600:.2f} seeds={seeds} out={dest_root}"
        )


if __name__ == "__main__":
    main()
