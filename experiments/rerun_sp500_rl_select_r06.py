#!/usr/bin/env python3
"""SP500 Top30 RL select ablation: SS-FM Meta vs select-PPO vs GRPO-init.

Reward (composite, robust-z):
  r = 0.6 z(net) + 0.3 z(causal Sharpe) - 0.01 z(turnover) - 0.01 z(smooth MDD)

Pure SS-FM: G=32 Meta (samples ∪ teachers → pred_utility).
PPO: discrete select over same Meta pool (no new weight generation).
GRPO: SS-FM distill init; bc_coef=0.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import time
from pathlib import Path
from typing import List

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
from models.alpha_predictor import build_predictor
from portfolio.constraints import apply_valid_mask
from rl.ablation_train import _meta_candidate_pool, train_grpo_ssfm_init, train_ppo_select
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
POOL = "top30"
OOS = "strict_fixed_oos"

REWARD = {
    "mode": "composite",
    "w_return": 0.6,
    "w_sharpe": 0.3,
    "w_turnover": 0.01,
    "w_smooth_mdd": 0.01,
}


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


def _apply_reward(cfg: dict) -> dict:
    out = dict(cfg)
    out["reward"] = {**(cfg.get("reward") or {}), **REWARD}
    return out


def run_month(cfg, store, device, month: str, src_root: Path, dest_root: Path, g: int) -> Path:
    src_month = _src_month(src_root, month)
    src_exp = _find_exp(src_month)
    ckpt = src_exp / "checkpoints"
    alpha_pt = ckpt / f"alpha_{cfg['prediction']['primary_model']}.pt"
    ssfm_pt = ckpt / f"gen_ssfm_{POOL}.pt"
    for p in (alpha_pt, ssfm_pt):
        if not p.is_file():
            raise FileNotFoundError(p)

    dest_month = dest_root / OOS / f"test_month={month}"
    dest_exp = dest_month / f"{src_exp.name}_rl_select_r06"
    dest_ckpt = dest_exp / "checkpoints"
    dest_ckpt.mkdir(parents=True, exist_ok=True)
    for p in (alpha_pt, ssfm_pt):
        shutil.copy2(p, dest_ckpt / p.name)

    cfg = _apply_reward(cfg)
    split = split_for_test_month(store.days, pd.Timestamp(f"{month}-01"), int(cfg["walkforward"]["train_offset_months"]))
    k = pool_k(store, cfg, POOL)
    compact = pool_is_compact(POOL)
    alpha = _load_alpha(cfg, store, alpha_pt, device)
    ssfm = _new_gen_models(cfg, device, k, compact)["ssfm"]
    _load_module_state(ssfm, ssfm_pt, device)
    ssfm.eval()
    for p in ssfm.parameters():
        p.requires_grad_(False)

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
    cfg_rl["reward"] = dict(cfg["reward"])

    parts: List[pd.DataFrame] = []
    meta = {
        "month": month,
        "pool": POOL,
        "g": g,
        "reward": dict(REWARD),
        "source": str(src_exp),
        "ssfm": "G samples ∪ teachers → pred_utility (Meta)",
        "ppo": "discrete select over Meta pool (no new weights)",
        "grpo": "ssfm-init",
    }

    log(f"==== {month} SS-FM G={g} Meta ====")
    t0 = time.time()
    ws = []
    for it in test_cache:
        cond = torch.from_numpy(it["cond"]).to(device)
        with torch.no_grad():
            arr = sample_model("ssfm", ssfm, cond, g, cfg).detach().cpu().numpy()
        _, wsel = select_candidate_meta(arr, _alpha_pool(it), it["sigma"], cfg, teachers=it.get("teachers"))
        ws.append(apply_valid_mask(wsel, it["valid"][it["idx"]]))
    df_ss = backtest_items(test_cache, ws, cfg, f"ssfm_G{g}_meta_{POOL}", OOS, g)
    parts.append(df_ss)
    meta["ssfm_sec"] = time.time() - t0

    log(f"==== {month} PPO select reward={REWARD} ====")
    t0 = time.time()
    n_steps = int(cfg["ssfm"]["n_sample_steps"])
    pol_ppo = train_ppo_select(ssfm, train_cache, cfg_rl, device, k=k, compact=compact, include_teachers=True)
    torch.save(pol_ppo.state_dict(), dest_ckpt / f"rl_ppo_select_{POOL}.pt")
    ws = []
    for it in test_cache:
        cond = torch.from_numpy(it["cond"]).to(device)
        with torch.no_grad():
            pool = _meta_candidate_pool(ssfm, cond, g, n_steps, it, device, include_teachers=True)
            _, wsel_t = pol_ppo.select(cond, pool)
        wsel = wsel_t.detach().cpu().numpy()
        ws.append(apply_valid_mask(wsel, it["valid"][it["idx"]]))
    df_ppo = backtest_items(test_cache, ws, cfg, f"rl_ssfm_ppo_select_G{g}_r06_{POOL}", OOS, g)
    parts.append(df_ppo)
    meta["ppo_sec"] = time.time() - t0

    log(f"==== {month} GRPO SSFM-init reward={REWARD} ====")
    t0 = time.time()
    pol_grpo = train_grpo_ssfm_init(ssfm, train_cache, cfg_rl, device, k=k, compact=compact)
    torch.save(pol_grpo.state_dict(), dest_ckpt / f"rl_grpo_init_{POOL}.pt")
    ws = []
    for it in test_cache:
        cond = torch.from_numpy(it["cond"]).to(device)
        w, _, _ = sample_portfolios(pol_grpo, cond, g, float(cfg_rl["rl"]["noise_std"]))
        arr = w.detach().cpu().numpy()
        _, wsel = select_candidate(arr, _alpha_pool(it), it["sigma"], cfg)
        ws.append(apply_valid_mask(wsel, it["valid"][it["idx"]]))
    df_grpo = backtest_items(test_cache, ws, cfg, f"rl_ssfm_grpo_init_G{g}_r06_{POOL}", OOS, g)
    parts.append(df_grpo)
    meta["grpo_sec"] = time.time() - t0

    daily = pd.concat(parts, ignore_index=True)
    daily.to_csv(dest_exp / "backtest_daily.csv", index=False)
    dest_month.mkdir(parents=True, exist_ok=True)
    daily.to_csv(dest_month / "backtest_daily.csv", index=False)
    write_json(dest_exp / "meta.json", meta)
    log(f"DONE {month} hours={(meta['ssfm_sec']+meta['ppo_sec']+meta['grpo_sec'])/3600:.2f}")
    return dest_exp


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


def write_report(dest_root: Path, year_csv: Path, g: int, out_tag: str) -> None:
    fig = dest_root / "analysis" / "figures"
    tab = dest_root / "analysis" / "tables"
    fig.mkdir(parents=True, exist_ok=True)
    tab.mkdir(parents=True, exist_ok=True)

    rename = {
        f"ssfm_G{g}_meta_{POOL}": f"SS-FM G={g} Meta",
        f"rl_ssfm_ppo_select_G{g}_r06_{POOL}": f"SS-FM+PPO (select) G={g} r06",
        f"rl_ssfm_grpo_init_G{g}_r06_{POOL}": f"SS-FM+GRPO (SSFM init) G={g} r06",
    }
    daily = pd.read_csv(year_csv)
    daily["date"] = pd.to_datetime(daily["date"])
    daily["name"] = daily["model"].astype(str).map(lambda x: rename.get(x, x))
    daily = daily.drop_duplicates(["date", "name"], keep="last")
    months = [m for m in MONTHS if daily["date"].astype(str).str.startswith(m).any()]

    names = [
        f"SS-FM G={g} Meta",
        f"SS-FM+PPO (select) G={g} r06",
        f"SS-FM+GRPO (SSFM init) G={g} r06",
    ]
    cm = {
        names[0]: color_for("gen_ssfm"),
        names[1]: "#E45756",
        names[2]: "#4C78A8",
    }
    for k, v in cm.items():
        SERIES_COLORS[k] = v

    nav = _summarize(daily, names).set_index("model")
    series = {}
    for n in names:
        gg = daily[daily["name"] == n].sort_values("date")
        series[n] = (gg["date"].to_numpy(), np.cumsum(gg["net_return"].to_numpy(float)))
    colors = [cm[n] for n in names]
    pref = "rl_select_r06_vs_ssfm"
    title = f"Top30 G={g} reward-r06 select-PPO：SS-FM vs PPO/GRPO"
    plot_series(fig / f"{pref}_cumulative_return.png", f"{title} 累计净收益（单利）", "交易日", "累计净收益（Σ）", names, series)
    plot_grouped_bars(fig / f"{pref}_total_return.png", f"{title} 累计净收益（单利）", "模型", "累计净收益", names, [float(nav.loc[n, "total_net_return"]) for n in names], colors)
    plot_grouped_bars(fig / f"{pref}_sharpe.png", f"{title} Sharpe", "模型", "Sharpe", names, [float(nav.loc[n, "sharpe"]) for n in names], colors)
    plot_grouped_bars(fig / f"{pref}_max_drawdown.png", f"{title} 最大回撤", "模型", "MDD", names, [float(nav.loc[n, "max_drawdown"]) for n in names], colors)
    plot_grouped_bars(fig / f"{pref}_turnover.png", f"{title} 日均换手", "模型", "换手", names, [float(nav.loc[n, "mean_turnover"]) for n in names], colors)
    month_vals = []
    for n in names:
        vals = []
        for m in months:
            s = daily[(daily["name"] == n) & (daily["date"].astype(str).str.startswith(m))]["net_return"]
            vals.append(float(s.sum()) if len(s) else None)
        month_vals.append(vals)
    plot_clustered_bars(fig / f"{pref}_monthly_return.png", f"{title} 分月净收益（单利）", "测试月", "当月Σ", months, names, month_vals, colors)
    nav.reset_index().to_csv(tab / "rl_select_r06_vs_ssfm.csv", index=False)

    n_days = int(nav["n_days"].max())
    L = []
    a = L.append
    a(f"## Top30 G={g} RL reward-r06 select-PPO：SS-FM vs PPO / GRPO")
    a("")
    a(
        f"设定：冻结 Top30 Alpha+SS-FM；配权 **Top30**；**G={g}**。"
        "Pure SS-FM 为 Meta 选仓（样本 ∪ Teacher 闭式解 → `pred_utility`）。"
        "**PPO=离散 select**（同一 Meta 候选池上打分选一条，**不再生成新权重**）；"
        "GRPO=SSFM init；`bc_coef=0`。"
    )
    a("")
    a(
        "**Reward（composite，robust-z）**：`r = 0.6·z(net return) + 0.3·z(因果 Sharpe)"
        " − 0.01·z(换手) − 0.01·z(smooth MDD)`。"
    )
    a("")
    a(f"产物：`results/S&P500/{out_tag}/`。拼接约 **{n_days}** 日。**单利 Σ**。")
    a("")
    a("| model | total_net_return（单利） | ann_return | sharpe | max_drawdown | mean_turnover | n_days |")
    a("| --- | --- | --- | --- | --- | --- | --- |")
    for n in names:
        r = nav.loc[n]
        a(
            f"| {n} | {_fmt(r.total_net_return)} | {_fmt(r.ann_return)} | {_fmt(r.sharpe)} | "
            f"{_fmt(r.max_drawdown)} | {_fmt(r.mean_turnover)} | {int(r.n_days)} |"
        )
    a("")
    hdr = "| month | " + " | ".join(names) + " |"
    sep = "| --- | " + " | ".join(["---"] * len(names)) + " |"
    a(hdr)
    a(sep)
    for i, m in enumerate(months):
        a("| " + m + " | " + " | ".join(_fmt(month_vals[j][i]) for j in range(len(names))) + " |")
    a("")
    for title_i, fn in [
        (f"{title} 累计净值（单利）", f"{pref}_cumulative_return.png"),
        (f"{title} 累计净收益柱（单利）", f"{pref}_total_return.png"),
        (f"{title} Sharpe", f"{pref}_sharpe.png"),
        (f"{title} 最大回撤", f"{pref}_max_drawdown.png"),
        (f"{title} 日均换手", f"{pref}_turnover.png"),
        (f"{title} 分月收益（单利）", f"{pref}_monthly_return.png"),
    ]:
        a(f"#### {title_i}")
        a("")
        a(f"![{title_i}](PLACEHOLDER/{fn})")
        a("")

    pure = float(nav.loc[names[0], "total_net_return"])
    ppo = float(nav.loc[names[1], "total_net_return"])
    grpo = float(nav.loc[names[2], "total_net_return"])
    analysis = "\n".join(
        [
            f"### Top30 G={g} RL reward-r06 select-PPO 结论",
            f"- 单利：SS-FM Meta `{pure:.4f}`；select-PPO `{ppo:.4f}`（{'胜' if ppo > pure else '负'}）；"
            f"GRPO-init `{grpo:.4f}`（{'胜' if grpo > pure else '负'}）。",
            f"- 换手：SS-FM `{float(nav.loc[names[0],'mean_turnover']):.4f}` → "
            f"PPO `{float(nav.loc[names[1],'mean_turnover']):.4f}` / "
            f"GRPO `{float(nav.loc[names[2],'mean_turnover']):.4f}`。",
            "- PPO 动作 = 在 Meta 候选池上离散选仓（argmax 推理），与自造权重的 cand-PPO 对照。",
            "",
        ]
    )
    section_tmpl = "\n".join(L) + "\n"

    targets = [
        (Path("results/S&P500/SP500_top30全年实验分析.md"), f"{out_tag}/analysis/figures"),
        (Path("results/S&P500/strict_fixed_oos_top30/final_summary/SP500_top30全年实验分析.md"), f"../../{out_tag}/analysis/figures"),
        (Path("results/S&P500/strict_fixed_oos/final_summary/S&P500全年实验分析.md"), f"../../{out_tag}/analysis/figures"),
        (Path("results/S&P500/strict_fixed_oos/analysis/S&P500全年实验分析.md"), f"../../{out_tag}/analysis/figures"),
    ]
    marker = f"## Top30 G={g} RL reward-r06 select-PPO：SS-FM vs PPO / GRPO\n"
    amark = f"### Top30 G={g} RL reward-r06 select-PPO 结论\n"

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

    (dest_root / "analysis" / "rl_select_r06报告片段.md").write_text(
        section_tmpl.replace("PLACEHOLDER", "figures") + "\n" + analysis, encoding="utf-8"
    )


def main():
    global POOL
    ap = argparse.ArgumentParser()
    ap.add_argument("--g", type=int, default=G_DEFAULT)
    ap.add_argument("--pool", default="top30")
    ap.add_argument("--out-tag", default="strict_fixed_oos_top30_rl_select_r06")
    ap.add_argument("--src-tag", default="strict_fixed_oos_top30")
    ap.add_argument("--months", default=",".join(MONTHS))
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()
    POOL = str(args.pool)
    g = int(args.g)
    months = [m.strip() for m in args.months.split(",") if m.strip()]

    set_seed(42, False, True)
    dest_root = Path("results/S&P500") / args.out_tag
    dest_root.mkdir(parents=True, exist_ok=True)
    year_csv = dest_root / "backtest_daily_year.csv"
    log_path = dest_root / "run_all.log"

    if args.report_only:
        write_report(dest_root, year_csv, g, args.out_tag)
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

        _log(f"CONFIG pool={POOL} g={g} reward={REWARD} ppo=select out={dest_root}")
        for month in months:
            _log(f"==== TOP30 RL-select-r06 month={month} ====")
            t0 = time.time()
            exp = run_month(cfg, store, device, month, src_root, dest_root, g)
            frames.append(pd.read_csv(exp / "backtest_daily.csv"))
            _log(f"DONE {month} hours={(time.time()-t0)/3600:.2f}")

        year = pd.concat(frames, ignore_index=True)
        year.to_csv(year_csv, index=False)
        _log(f"REQUESTED MONTHS SP500-TOP30-RL-SELECT-R06 DONE hours={(time.time()-t_all)/3600:.2f} out={dest_root}")

    write_report(dest_root, year_csv, g, args.out_tag)


if __name__ == "__main__":
    main()
