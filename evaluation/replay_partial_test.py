"""Replay finished Sequential-OOS test days onto disk so figures can fill in."""

from __future__ import annotations

import gc
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch

from data.splits import sequential_retrain_windows, split_for_test_month
from data.store import ResearchStore
from evaluation.experiment_analysis import refresh_analysis
from evaluation.prediction_metrics import daily_ic, ranking_classification_metrics, top30_stats
from experiments.engine import (
    _load_module_state,
    _new_gen_models,
    backtest_items,
    cond_dim,
    make_alloc_item,
    predict_loaded,
    sample_model,
    select_candidate,
)
from flow_matching.samplers import sample_and_audit
from models.alpha_predictor import build_predictor
from portfolio.constraints import apply_valid_mask
from portfolio.topk import select_topk
from rl.ppo import WeightPolicy, sample_portfolios
from utils.config import load_config, refresh_prediction_train_cfg
from utils.logging import log
from utils.seed import set_seed


PRED_MODELS = [
    "lstm",
    "tcn",
    "standard_transformer",
    "patchtst",
    "minute_only",
    "concat",
    "cross_attention",
    "gated_residual",
]
GEN_NAMES = ("mlp", "gaussian", "diffusion", "standard_fm", "ssfm")
DONE_UNTIL = "2025-05-16"
INIT_CKPT = Path(
    "results/full/sequential_oos_weekly_retrain/test_month=2025-05/"
    "csi500_ssfm_research_2025-05_sequential_oos_weekly_retrain_4b4cbd6111/checkpoints"
)
FRIDAY_CKPT = Path(
    "results/full/sequential_oos_weekly_retrain/test_month=2025-05/"
    "csi500_ssfm_research_2025-05_sequential_oos_weekly_retrain_f3acf63371/checkpoints"
)
MONTH_DIR = Path("results/full/sequential_oos_weekly_retrain/test_month=2025-05")


def _load_alpha(name: str, cfg: dict, feat_dim: int, ckpt: Path, device) -> torch.nn.Module:
    local = dict(cfg)
    refresh_prediction_train_cfg(local)
    model = build_predictor(name, local, feat_dim).to(device)
    src = ckpt / f"alpha_{name}.pt"
    if name == "minute_only" and not src.is_file():
        src = ckpt / "alpha_standard_transformer.pt"
    _load_module_state(model, src, device)
    model.eval()
    return model


def _load_alloc(ckpt: Path, cfg: dict, device, k: int) -> dict:
    gen = _new_gen_models(cfg, device, k, compact=False)
    for n, model in gen.items():
        _load_module_state(model, ckpt / f"gen_{n}_top30.pt", device)
        model.eval()
    rl = {}
    cd = cond_dim(k, compact=False)
    for algo in cfg["rl"]["algorithms"]:
        pol = WeightPolicy(cd, k, int(cfg["rl"]["hidden"])).to(device)
        _load_module_state(pol, ckpt / f"rl_{algo}_top30.pt", device)
        pol.eval()
        rl[str(algo)] = pol
    return {"gen": gen, "rl": rl}


def _alias_daily(df: pd.DataFrame, src: str, aliases: list[str]) -> list[pd.DataFrame]:
    out = [df]
    for name in aliases:
        if name == src:
            continue
        cp = df.copy()
        cp["model"] = name
        cp["strategy_name"] = name
        out.append(cp)
    return out


def main() -> None:
    cfg = load_config()
    set_seed(int(cfg["experiment"]["seed"]), False, True)
    device = torch.device("cpu")
    store = ResearchStore(cfg)
    split = split_for_test_month(store.days, pd.Timestamp("2025-05-01"), int(cfg["walkforward"]["train_offset_months"]))
    windows = sequential_retrain_windows(split)
    week2_from = None
    for w in windows:
        if w.get("cutoff") == "2025-05-09" and w.get("effective_from"):
            week2_from = pd.Timestamp(w["effective_from"])
            break
    if week2_from is None:
        week2_from = pd.Timestamp("2025-05-12")
    days = [d for d in split["test_days"] if pd.Timestamp(d) <= pd.Timestamp(DONE_UNTIL)]
    log(f"replay {len(days)} days through {DONE_UNTIL}; week2 from {week2_from.date()} device={device}")

    top_k = int(cfg["portfolio"]["top_k"])
    default_g = int(cfg["ssfm"]["default_g"])
    primary = cfg["prediction"]["primary_model"]
    pred_rows = []
    items_by_model = {n: [] for n in PRED_MODELS}
    primary_items = []

    init_alphas = {n: _load_alpha(n, cfg, store.feat_dim, INIT_CKPT, device) for n in PRED_MODELS}
    friday_primary = _load_alpha(primary, cfg, store.feat_dim, FRIDAY_CKPT, device)
    init_alloc = _load_alloc(INIT_CKPT, cfg, device, top_k)
    friday_alloc = _load_alloc(FRIDAY_CKPT, cfg, device, top_k)

    for day in days:
        use_friday = pd.Timestamp(day) >= week2_from
        alphas = dict(init_alphas)
        if use_friday:
            alphas[primary] = friday_primary
        alloc = friday_alloc if use_friday else init_alloc
        y, y_c2c = store.labels(day)
        x, msk = store.window(day)
        a, ameta = store.auction_d(day)
        valid = store.valid_mask(day)
        xt = torch.from_numpy(np.nan_to_num(x)).to(device)
        at = torch.from_numpy(np.nan_to_num(a)).to(device)
        mt = torch.from_numpy(msk).to(device)
        del x, msk, a
        item_primary = None
        for name in PRED_MODELS:
            pred, pmeta = predict_loaded(alphas[name], xt, at, mt, valid, ameta, device)
            idx = select_topk(pred, valid, top_k)
            ic = daily_ic(pred[valid], y[valid])
            rk = ranking_classification_metrics(pred[valid], y[valid], top_k=top_k)
            tstat = top30_stats(pred, y, idx, valid)
            item = make_alloc_item(store, day, pred, valid, y, pmeta, cfg, "top30")
            item["y_c2c"] = y_c2c
            item["model"] = name
            items_by_model[name].append(item)
            pred_rows.append(
                {
                    "date": str(pd.Timestamp(day).date()),
                    "model": name,
                    **ic,
                    **rk,
                    **tstat,
                    "valid_universe_size": int(valid.sum()),
                    "ckpt": "friday_0509" if use_friday and name == primary else "init",
                }
            )
            if name == primary:
                item_primary = item
        del xt, at, mt
        gc.collect()
        assert item_primary is not None
        primary_items.append(item_primary)
        log(f"  {pd.Timestamp(day).date()} {'week2' if use_friday else 'week1'} done")

    daily_all = []
    for name, items in items_by_model.items():
        ws = [np.ones(top_k) / top_k for _ in items]
        df = backtest_items(items, ws, cfg, f"pred_{name}_ew_top30", "sequential_oos_weekly_retrain", 1)
        daily_all.extend(_alias_daily(df, f"pred_{name}_ew_top30", [f"pred_{name}_ew"]))

    for bname in cfg["matrix"]["B_portfolio"]:
        ws = [it["baselines"][bname] for it in primary_items]
        df = backtest_items(primary_items, ws, cfg, f"baseline_{bname}_top30", "sequential_oos_weekly_retrain", 1)
        daily_all.extend(_alias_daily(df, f"baseline_{bname}_top30", [f"baseline_{bname}"]))

    for gname, gmodel in friday_alloc["gen"].items():
        # per-day alloc pack: week1 init, week2 friday
        ws = []
        for it in primary_items:
            pack = friday_alloc if pd.Timestamp(it["asof"]) >= week2_from else init_alloc
            model = pack["gen"][gname]
            cond = torch.from_numpy(it["cond"]).to(device)
            w, _ = sample_and_audit(lambda m=model, c=cond: sample_model(gname, m, c, default_g, cfg))
            arr = w.detach().cpu().numpy()
            _, wsel = select_candidate(arr, it["alpha"][it["idx"]], it["sigma"], cfg)
            ws.append(apply_valid_mask(wsel, it["valid"][it["idx"]]))
        df = backtest_items(primary_items, ws, cfg, f"gen_{gname}_top30", "sequential_oos_weekly_retrain", default_g)
        daily_all.extend(_alias_daily(df, f"gen_{gname}_top30", [f"gen_{gname}"]))

    for algo in cfg["rl"]["algorithms"]:
        ws = []
        for it in primary_items:
            pack = friday_alloc if pd.Timestamp(it["asof"]) >= week2_from else init_alloc
            pol = pack["rl"][str(algo)]
            cond = torch.from_numpy(it["cond"]).to(device)
            w, _, _ = sample_portfolios(pol, cond, default_g, float(cfg["rl"]["noise_std"]))
            arr = w.detach().cpu().numpy()
            _, wsel = select_candidate(arr, it["alpha"][it["idx"]], it["sigma"], cfg)
            ws.append(apply_valid_mask(wsel, it["valid"][it["idx"]]))
        df = backtest_items(primary_items, ws, cfg, f"rl_ssfm_{algo}_top30", "sequential_oos_weekly_retrain", default_g)
        daily_all.extend(_alias_daily(df, f"rl_ssfm_{algo}_top30", [f"rl_ssfm_{algo}"]))

    pred = pd.DataFrame(pred_rows)
    daily = pd.concat(daily_all, ignore_index=True)
    MONTH_DIR.mkdir(parents=True, exist_ok=True)
    pred.to_csv(MONTH_DIR / "prediction_metrics.csv", index=False)
    daily.to_csv(MONTH_DIR / "backtest_daily.csv", index=False)
    log(f"wrote {len(pred)} pred rows / {daily['model'].nunique()} nav series / {pred['date'].nunique()} days")
    refresh_analysis(Path("results/full"), "sequential_oos_weekly_retrain", {"source": "replay_partial_test 2025-05-06..16"})
    log("figures refreshed")


if __name__ == "__main__":
    main()
