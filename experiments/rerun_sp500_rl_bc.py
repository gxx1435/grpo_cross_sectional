#!/usr/bin/env python3
"""Re-train SP500 SS-FM+RL only (PPO/GRPO), freeze alpha + gen_ssfm checkpoints.

Uses lowered turnover/MDD reward weights and online BC toward SS-FM samples
(see configs/sp500.yaml rl.* / reward.*). Writes RL daily series under a new
out-tag, then merges into the original strict_fixed_oos year for the report.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from data.splits import split_for_test_month
from experiments.engine import (
    _load_module_state,
    _new_gen_models,
    backtest_items,
    build_state_cache,
    distill_and_rl,
    pool_is_compact,
    pool_k,
    sample_portfolios,
    select_candidate,
)
from models.alpha_predictor import build_predictor
from portfolio.constraints import apply_valid_mask
from utils.config import load_config, make_store, resolve_path
from utils.gpu import require_cuda
from utils.logging import log, write_json

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


def _find_exp_dir(month_root: Path) -> Path:
    cands = sorted(month_root.glob("sp500_*"))
    if not cands:
        raise FileNotFoundError(f"no sp500 exp dir under {month_root}")
    return cands[0]


def _default_g(cfg: dict, exp_dir: Path) -> int:
    meta = exp_dir / "environment.json"
    if meta.is_file():
        try:
            obj = json.loads(meta.read_text(encoding="utf-8"))
            g = int((obj.get("ssfm") or {}).get("default_g") or 0)
            if g > 0:
                return g
        except Exception:
            pass
    return int(cfg["ssfm"].get("default_g") or 16)


def _load_alpha(cfg: dict, store, ckpt: Path, device) -> torch.nn.Module:
    name = str(cfg["prediction"]["primary_model"])
    model = build_predictor(name, cfg, store.feat_dim).to(device)
    _load_module_state(model, ckpt, device)
    model.eval()
    return model


def _backtest_rl(pol, test_cache, cfg, device, model_name: str, oos: str, g: int) -> pd.DataFrame:
    ws = []
    for it in test_cache:
        cond = torch.from_numpy(it["cond"]).to(device)
        w, _, _ = sample_portfolios(pol, cond, g, float(cfg["rl"]["noise_std"]))
        arr = w.detach().cpu().numpy()
        j, wsel = select_candidate(arr, it["alpha"][it["idx"]], it["sigma"], cfg)
        ws.append(apply_valid_mask(wsel, it["valid"][it["idx"]]))
    return backtest_items(test_cache, ws, cfg, model_name, oos, g)


def run_month_rl(
    cfg: dict,
    store,
    device,
    month: str,
    src_root: Path,
    dest_root: Path,
    pool: str = "all",
) -> Path:
    oos = "strict_fixed_oos"
    src_month = src_root / oos / f"test_month={month}"
    src_exp = _find_exp_dir(src_month)
    src_ckpt = src_exp / "checkpoints"
    ssfm_pt = src_ckpt / f"gen_ssfm_{pool}.pt"
    alpha_pt = src_ckpt / "alpha_standard_transformer.pt"
    if not ssfm_pt.is_file():
        raise FileNotFoundError(ssfm_pt)
    if not alpha_pt.is_file():
        raise FileNotFoundError(alpha_pt)

    dest_month = dest_root / oos / f"test_month={month}"
    dest_exp = dest_month / f"{src_exp.name}_rl_bc"
    dest_ckpt = dest_exp / "checkpoints"
    dest_ckpt.mkdir(parents=True, exist_ok=True)

    # freeze-copy alpha + ssfm (no retrain)
    shutil.copy2(ssfm_pt, dest_ckpt / ssfm_pt.name)
    shutil.copy2(alpha_pt, dest_ckpt / alpha_pt.name)
    split = split_for_test_month(
        store.days,
        pd.Timestamp(f"{month}-01"),
        int(cfg["walkforward"]["train_offset_months"]),
    )
    # JSON-safe copy of split (dates as strings)
    split_json = {}
    for k, v in split.items():
        if isinstance(v, list):
            split_json[k] = [str(pd.Timestamp(x).date()) for x in v]
        elif hasattr(v, "isoformat"):
            split_json[k] = str(pd.Timestamp(v).date())
        else:
            split_json[k] = v
    write_json(dest_exp / "split_manifest.json", {**split_json, "source_exp": str(src_exp)})

    k = pool_k(store, cfg, pool)
    compact = pool_is_compact(pool)
    alpha = _load_alpha(cfg, store, alpha_pt, device)
    gen_all = _new_gen_models(cfg, device, k, compact)
    ssfm = gen_all["ssfm"]
    _load_module_state(ssfm, ssfm_pt, device)
    ssfm.eval()

    cut = split["validation_end_date"]
    log(f"==== {month} build train cache pool={pool} K={k} ====")
    train_cache = build_state_cache(store, alpha, split["train_days"], cfg, device, pool=pool, cutoff=cut)
    if not train_cache:
        raise RuntimeError(f"empty train cache {month}")
    log(f"==== {month} build test cache ====")
    test_cache = build_state_cache(store, alpha, split["test_days"], cfg, device, pool=pool, cutoff=None)
    if not test_cache:
        raise RuntimeError(f"empty test cache {month}")

    g = _default_g(cfg, src_exp)
    algos = [str(a) for a in cfg["rl"]["algorithms"]]
    daily_parts: List[pd.DataFrame] = []
    meta = {
        "month": month,
        "pool": pool,
        "k": k,
        "compact": compact,
        "default_g": g,
        "reward": dict(cfg["reward"]),
        "rl": {
            "bc_coef": cfg["rl"].get("bc_coef"),
            "prev_mode": cfg["rl"].get("prev_mode"),
            "noise_std": cfg["rl"].get("noise_std"),
            "epochs": cfg["rl"].get("epochs"),
            "w_turnover": cfg["reward"].get("w_turnover"),
            "w_smooth_mdd": cfg["reward"].get("w_smooth_mdd"),
        },
        "source_ssfm": str(ssfm_pt),
        "algos": {},
    }
    for algo in algos:
        log(f"==== {month} distill+RL {algo} bc_coef={cfg['rl'].get('bc_coef')} prev={cfg['rl'].get('prev_mode')} ====")
        t0 = time.time()
        pol = distill_and_rl(ssfm, train_cache, cfg, device, algo, k=k, compact=compact)
        torch.save(pol.state_dict(), dest_ckpt / f"rl_{algo}_{pool}.pt")
        df = _backtest_rl(pol, test_cache, cfg, device, f"rl_ssfm_{algo}_{pool}", oos, g)
        daily_parts.append(df)
        meta["algos"][algo] = {"wall_sec": time.time() - t0, "n_days": int(len(df))}
        log(f"    {algo} done days={len(df)} wall={time.time()-t0:.1f}s")

    daily = pd.concat(daily_parts, ignore_index=True)
    daily.to_csv(dest_exp / "backtest_daily.csv", index=False)
    daily.to_csv(dest_month / "backtest_daily.csv", index=False)
    write_json(dest_exp / "rl_bc_meta.json", meta)
    # keep a pointer to original Pure SS-FM / gen series for merge
    write_json(dest_month / "source_exp.json", {"src_exp": str(src_exp), "src_month": str(src_month)})
    return dest_exp


def merge_rl_into_original(src_root: Path, rl_root: Path, months: List[str]) -> None:
    """Replace rl_ssfm_* rows in original monthly + final_summary backtest_daily."""
    oos = "strict_fixed_oos"
    for month in months:
        src_month = src_root / oos / f"test_month={month}"
        rl_month = rl_root / oos / f"test_month={month}"
        rl_daily = pd.read_csv(rl_month / "backtest_daily.csv")
        src_exp = _find_exp_dir(src_month)
        for path in (src_exp / "backtest_daily.csv", src_month / "backtest_daily.csv"):
            if not path.is_file():
                continue
            old = pd.read_csv(path)
            keep = old[~old["model"].astype(str).str.startswith("rl_ssfm_")].copy()
            new = pd.concat([keep, rl_daily], ignore_index=True)
            new.to_csv(path, index=False)
            # tables mirror
            tab = path.parent / "tables" / "backtest_daily.csv"
            if tab.parent.is_dir():
                new.to_csv(tab, index=False)
            log(f"    merged RL -> {path}")

    # rebuild stitched final_summary daily from monthly originals
    frames = []
    for month in months:
        p = src_root / oos / f"test_month={month}" / "backtest_daily.csv"
        if p.is_file():
            frames.append(pd.read_csv(p))
    if not frames:
        return
    all_daily = pd.concat(frames, ignore_index=True)
    keys = [c for c in ("date", "model") if c in all_daily.columns]
    if keys:
        all_daily = all_daily.drop_duplicates(subset=keys, keep="last")
    out = src_root / "final_summary" / oos
    out.mkdir(parents=True, exist_ok=True)
    all_daily.to_csv(out / "backtest_daily.csv", index=False)
    from backtest.metrics import summarize_nav

    nav = summarize_nav(all_daily).sort_values("total_net_return", ascending=False)
    nav.to_csv(out / "performance_summary.csv", index=False)
    log(f"    wrote {out / 'performance_summary.csv'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", nargs="*", default=MONTHS)
    ap.add_argument("--src-tag", default="strict_fixed_oos")
    ap.add_argument("--out-tag", default="strict_fixed_oos_rl_bc")
    ap.add_argument("--reward-mode", default=None, help="composite | sharpe_only")
    ap.add_argument("--merge-only", action="store_true")
    ap.add_argument("--skip-merge", action="store_true")
    args = ap.parse_args()

    overrides = {}
    if args.reward_mode:
        mode = str(args.reward_mode).strip().lower()
        rem = {"mode": mode}
        if mode in ("sharpe", "sharpe_only", "single_sharpe"):
            rem.update({"w_return": 0.0, "w_sharpe": 1.0, "w_turnover": 0.0, "w_smooth_mdd": 0.0})
        overrides["reward"] = rem
    cfg = load_config(overrides=overrides if overrides else None)
    assert str(cfg["experiment"]["market"]).lower() in ("sp500", "s&p500", "spx")
    results_dir = resolve_path(cfg, cfg["paths"]["results_dir"])
    src_root = results_dir / args.src_tag
    dest_root = results_dir / args.out_tag
    dest_root.mkdir(parents=True, exist_ok=True)
    months = list(args.months)

    log(
        f"SP500 RL rerun | mode={cfg['reward'].get('mode', 'composite')} "
        f"TO/MDD={cfg['reward'].get('w_turnover')}/{cfg['reward'].get('w_smooth_mdd')} "
        f"bc_coef={cfg['rl'].get('bc_coef')} prev={cfg['rl'].get('prev_mode')} noise={cfg['rl'].get('noise_std')} "
        f"epochs={cfg['rl'].get('epochs')} months={len(months)} out={args.out_tag}"
    )
    write_json(
        dest_root / "rerun_config.json",
        {
            "reward": cfg["reward"],
            "rl": {k: cfg["rl"].get(k) for k in ("bc_coef", "prev_mode", "noise_std", "epochs", "group_size", "distill_steps", "teacher_bc_samples")},
            "months": months,
            "src_tag": args.src_tag,
            "out_tag": args.out_tag,
        },
    )

    if not args.merge_only:
        device = require_cuda()
        store = make_store(cfg)
        t0 = time.time()
        for month in months:
            run_month_rl(cfg, store, device, month, src_root, dest_root, pool="all")
        log(f"REQUESTED MONTHS RL-BC DONE hours={(time.time()-t0)/3600:.2f} out={dest_root}")

    if not args.skip_merge:
        merge_rl_into_original(src_root, dest_root, months)
        log(f"merged RL series into {src_root}")


if __name__ == "__main__":
    main()
