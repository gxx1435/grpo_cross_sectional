"""Run experiment E (SS-FM candidate count G) for one test month.

Uses already-trained weekly checkpoints. Does not retrain SFT / C / D.
Only top30. Writes ssfm_G{8,16,32,64,128}_top30 for all TEST days, then
refreshes that month's 实验分析.md only.
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch

from data.splits import sequential_retrain_windows, split_for_test_month
from data.store import ResearchStore
from evaluation.experiment_analysis import refresh_month
from experiments.engine import (
    _load_module_state,
    _new_gen_models,
    backtest_items,
    make_alloc_item,
    predict_day,
    select_candidate,
)
from flow_matching.samplers import sample_and_audit
from flow_matching.ssfm import sample_ss_fm_mixed
from models.alpha_predictor import build_predictor
from portfolio.constraints import apply_valid_mask
from utils.config import load_config, refresh_prediction_train_cfg
from utils.gpu import configure_gpu
from utils.io import dump_yaml_copy, experiment_id
from utils.logging import log, write_json
from utils.seed import set_seed


MONTH_ROOT = Path("results/full/sequential_oos_weekly_retrain")
OOS = "sequential_oos_weekly_retrain"
PRIMARY = "gated_residual"

# Friday-close packs already on disk for 2025-05. New weights apply next week.
# 5/6–5/16: pre-test pack (no dedicated 5/9 gen; do not use 5/16 weights on 5/12–5/16).
# 5/19–5/23: 5/16 Friday pack.
# 5/26–5/30: 5/23 SFT alpha + 5/23 SS-FM.
CKPT_PRE = MONTH_ROOT / "test_month=2025-05/csi500_ssfm_research_2025-05_sequential_oos_weekly_retrain_4b4cbd6111/checkpoints"
CKPT_0516 = MONTH_ROOT / "test_month=2025-05/csi500_ssfm_research_2025-05_sequential_oos_weekly_retrain_f3acf63371/checkpoints"
CKPT_SFT_0523 = MONTH_ROOT / "test_month=2025-05/csi500_ssfm_research_2025-05_sequential_oos_weekly_retrain_df27c3d6f0/checkpoints"
CKPT_SSFM_0523 = MONTH_ROOT / "test_month=2025-05/csi500_ssfm_research_2025-05_sequential_oos_weekly_retrain_c518cbda4e/checkpoints"


def _pack_for_day(day: pd.Timestamp) -> dict:
    d = pd.Timestamp(day).normalize()
    if d < pd.Timestamp("2025-05-19"):
        return {"label": "pre_test", "alpha": CKPT_PRE, "ssfm": CKPT_PRE}
    if d < pd.Timestamp("2025-05-26"):
        return {"label": "friday_0516", "alpha": CKPT_0516, "ssfm": CKPT_0516}
    return {"label": "friday_0523", "alpha": CKPT_SFT_0523, "ssfm": CKPT_SSFM_0523}


def _load_alpha(cfg: dict, feat_dim: int, ckpt: Path, device) -> torch.nn.Module:
    local = dict(cfg)
    refresh_prediction_train_cfg(local)
    model = build_predictor(PRIMARY, local, feat_dim).to(device)
    src = ckpt / f"alpha_{PRIMARY}.pt"
    if not src.is_file():
        raise FileNotFoundError(src)
    _load_module_state(model, src, device)
    model.eval()
    return model


def _load_ssfm(cfg: dict, ckpt: Path, device, k: int) -> torch.nn.Module:
    src = ckpt / "gen_ssfm_top30.pt"
    if not src.is_file():
        raise FileNotFoundError(src)
    model = _new_gen_models(cfg, device, k, compact=False)["ssfm"]
    _load_module_state(model, src, device)
    model.eval()
    return model


def _merge_month_root(month_dir: Path, g_daily: pd.DataFrame) -> None:
    root_csv = month_dir / "backtest_daily.csv"
    if root_csv.is_file():
        old = pd.read_csv(root_csv)
        old = old[~old["model"].astype(str).str.startswith("ssfm_G")].copy()
        out = pd.concat([old, g_daily], ignore_index=True)
    else:
        out = g_daily
    out.to_csv(root_csv, index=False)
    log(f"merged {g_daily['model'].nunique()} G series into {root_csv}")


def run_month_g(cfg: dict, month: str) -> Path:
    set_seed(int(cfg["experiment"]["seed"]), False, True)
    configure_gpu(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    store = ResearchStore(cfg)
    split = split_for_test_month(store.days, pd.Timestamp(f"{month}-01"), int(cfg["walkforward"]["train_offset_months"]))
    windows = sequential_retrain_windows(split)
    days = list(split["test_days"])
    g_grid = [int(g) for g in cfg["matrix"]["E_g"]]
    top_k = int(cfg["portfolio"]["top_k"])
    n_steps = int(cfg["ssfm"]["n_sample_steps"])
    month_dir = MONTH_ROOT / f"test_month={month}"
    exp_id = experiment_id(cfg["experiment"]["name"], month, OOS, extra="g_grid_only")
    out = month_dir / exp_id
    out.mkdir(parents=True, exist_ok=True)
    dump_yaml_copy(cfg, out / "config.yaml")
    write_json(
        out / "g_grid_meta.json",
        {
            "month": month,
            "g_grid": g_grid,
            "n_test_days": len(days),
            "windows": [
                {"cutoff": w.get("cutoff"), "effective_from": w.get("effective_from"), "label": w.get("label")}
                for w in windows
            ],
            "ckpt_map": {
                "pre_test_<2025-05-19": str(CKPT_PRE),
                "friday_0516_<2025-05-26": str(CKPT_0516),
                "friday_0523_alpha": str(CKPT_SFT_0523),
                "friday_0523_ssfm": str(CKPT_SSFM_0523),
            },
            "device": str(device),
        },
    )
    log(f"E-G only {month} days={len(days)} G={g_grid} device={device} out={out.name}")

    items = []
    loaded_alpha_key = None
    loaded_ssfm_key = None
    alpha = None
    ssfm = None
    for day in days:
        pack = _pack_for_day(day)
        akey, skey = str(pack["alpha"]), str(pack["ssfm"])
        if akey != loaded_alpha_key:
            if alpha is not None:
                del alpha
            alpha = _load_alpha(cfg, store.feat_dim, pack["alpha"], device)
            loaded_alpha_key = akey
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        if skey != loaded_ssfm_key:
            if ssfm is not None:
                del ssfm
            ssfm = _load_ssfm(cfg, pack["ssfm"], device, top_k)
            loaded_ssfm_key = skey
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        pred, valid, ameta = predict_day(alpha, store, day, device)
        y, _ = store.labels(day)
        item = make_alloc_item(store, day, pred, valid, y, ameta, cfg, "top30")
        item["ckpt_label"] = pack["label"]
        items.append(item)
        del pred
        gc.collect()
        log(f"  cache {pd.Timestamp(day).date()} {pack['label']} n={len(items)}")

    daily_all = []
    for g in g_grid:
        t0 = time.time()
        ws = []
        for it in items:
            pack = _pack_for_day(it["asof"])
            skey = str(pack["ssfm"])
            if skey != loaded_ssfm_key:
                if ssfm is not None:
                    del ssfm
                ssfm = _load_ssfm(cfg, pack["ssfm"], device, top_k)
                loaded_ssfm_key = skey
            cond = torch.from_numpy(it["cond"]).to(device)
            w, _ = sample_and_audit(lambda: sample_ss_fm_mixed(ssfm, cond, int(g), n_steps))
            arr = w.detach().cpu().numpy()
            _, wsel = select_candidate(arr, it["alpha"][it["idx"]], it["sigma"], cfg)
            ws.append(apply_valid_mask(wsel, it["valid"][it["idx"]]))
            del cond, w
        df = backtest_items(items, ws, cfg, f"ssfm_G{g}_top30", OOS, int(g))
        df["sampling_time_sec"] = time.time() - t0
        daily_all.append(df)
        log(f"  G={g} days={df['date'].nunique() if 'date' in df.columns else len(df)} wall={time.time()-t0:.1f}s")
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    g_daily = pd.concat(daily_all, ignore_index=True)
    g_daily.to_csv(out / "backtest_daily.csv", index=False)
    (out / "README.txt").write_text(
        "Experiment E only: SS-FM candidate grid G=8/16/32/64/128 on top30.\n"
        "Weekly checkpoints, no SFT/C/D retraining.\n",
        encoding="utf-8",
    )
    _merge_month_root(month_dir, g_daily)
    refresh_month(
        Path("results/full"),
        OOS,
        month,
        {"source": f"run_g_grid_month {month} weekly SS-FM", "last_unit": "E-G top30"},
    )
    log(f"done G-grid {month}: {g_daily.groupby('model')['date'].nunique().to_dict()}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--month", default="2025-05")
    args = ap.parse_args()
    if args.month != "2025-05":
        raise SystemExit("this runner currently maps 2025-05 weekly checkpoints only")
    cfg = load_config()
    run_month_g(cfg, args.month)


if __name__ == "__main__":
    main()
