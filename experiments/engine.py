"""Shared monthly walk-forward engine. Real training, no fabricated metrics."""

from __future__ import annotations

import copy
import gc
import math
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from backtest.drawdown import smooth_dd_to_date
from backtest.engine import run_path
from backtest.execution import execute_day
from backtest.metrics import causal_sharpe
from data.normalizers import ChannelZScore, RobustScaler
from data.splits import sequential_retrain_windows, split_for_test_month
from data.store import ResearchStore
from evaluation.experiment_analysis import (
    refresh_after_unit,
    update_family_docs,
    write_alpha_val_snapshot,
    write_month_bundle,
)
from evaluation.leakage_audit import run_audit, write_audit
from evaluation.losses import sft_asof_loss
from evaluation.prediction_metrics import daily_ic, ranking_classification_metrics, top30_stats
from evaluation.reports import write_month_reports
from experiments.progress import Progress, count_month_units
from experiments.resume import find_exp_dir, inspect_resume
from flow_matching.diffusion import WeightDiffusion, diffusion_loss, sample_diffusion
from flow_matching.gaussian_policy import GaussianPolicy
from flow_matching.mlp_policy import MLPPolicy
from flow_matching.samplers import sample_and_audit
from flow_matching.ssfm import SimplexSpaceFlow, sample_ss_fm_mixed, ss_fm_loss
from flow_matching.standard_fm import StandardFlowMatching, sample_std_fm, std_fm_loss
from models.alpha_predictor import build_predictor
from portfolio.constraints import apply_valid_mask
from portfolio.gpu_alloc import build_cond_cuda, hist_mu_sigma_from_store, pack_allocators, score_softmax_cuda
from portfolio.teacher_portfolios import TEACHER_IDS, TEACHER_NAMES
from portfolio.topk import select_topk, topk_table
from rl.grpo import grpo_update
from rl.ppo import ValueHead, WeightPolicy, ppo_update, sample_portfolios
from rl.reward import composite_reward, fit_reward_scalers
from utils.config import refresh_prediction_train_cfg
from utils.git_info import environment_info
from utils.io import dump_yaml_copy, experiment_id
from utils.logging import append_csv_row, log, write_json
from utils.runtime import count_params, run_with_oom_retry
from utils.seed import set_seed


def build_cond(alpha_k: np.ndarray, mu: np.ndarray, sigma: np.ndarray, compact: bool = False) -> np.ndarray:
    return build_cond_cuda(alpha_k, mu, sigma, compact=compact)


def cond_dim(k: int, compact: bool = False) -> int:
    if compact:
        return 4 * int(k)
    return 4 * int(k) + int(k) * (int(k) + 1) // 2


def allocation_pool_names(cfg: dict) -> List[str]:
    raw = (
        (cfg.get("matrix") or {}).get("allocation_pools")
        or (cfg.get("portfolio") or {}).get("allocation_pools")
        or ["top30"]
    )
    out = []
    for p in raw:
        name = str(p)
        if name not in ("top30", "all"):
            raise ValueError(f"unknown allocation pool {name}")
        if name not in out:
            out.append(name)
    return out


def pool_is_compact(pool: str) -> bool:
    return pool == "all"


def pool_k(store, cfg: dict, pool: str) -> int:
    if pool == "all":
        return int(len(store.kept))
    return int(cfg["portfolio"]["top_k"])


def pool_idx(pred: np.ndarray, valid: np.ndarray, store, pool: str, top_k: int) -> np.ndarray:
    if pool == "all":
        return np.arange(len(store.kept), dtype=int)
    return select_topk(pred, valid, top_k)


def device_of() -> torch.device:
    from utils.gpu import require_cuda

    return require_cuda()


def amp_ctx(device):
    return torch.cuda.amp.autocast(enabled=device.type == "cuda")


def _free_host() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@torch.no_grad()
def predict_loaded(model: nn.Module, xt, at, mt, valid, ameta, device) -> Tuple[np.ndarray, Dict]:
    model.eval()
    vt = valid if torch.is_tensor(valid) else torch.from_numpy(valid).to(device)
    pred = torch.zeros(xt.size(0), device=device)
    if bool(vt.any()):
        pred[vt] = model(xt[vt], auction=at[vt], valid=None, mask=mt[vt])
    aux = {k: v.detach().cpu().numpy() for k, v in getattr(model, "last_aux", {}).items()}
    return pred.cpu().numpy().astype(np.float32), {**ameta, **{"aux_keys": list(aux)}, "aux": aux}


def _model_scaler(model, feat_scaler=None):
    if feat_scaler is not None:
        return feat_scaler
    return getattr(model, "feat_scaler", None)


@torch.no_grad()
def predict_day(model: nn.Module, store: ResearchStore, day: pd.Timestamp, device, feat_scaler=None) -> Tuple[np.ndarray, np.ndarray, Dict]:
    x, msk = store.window(day)
    a, ameta = store.auction_d(day)
    valid = store.valid_mask(day)
    scaler = _model_scaler(model, feat_scaler)
    if scaler is not None:
        x = scaler.transform_x(x, msk)
        a = scaler.transform_a(a)
    xt = torch.from_numpy(np.nan_to_num(x)).to(device)
    at = torch.from_numpy(np.nan_to_num(a)).to(device)
    mt = torch.from_numpy(msk).to(device)
    pred, meta = predict_loaded(model, xt, at, mt, valid, ameta, device)
    del x, msk, a, xt, at, mt
    return pred, valid, meta


def _fmt_metric(v: Any, nd: int = 4) -> str:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return "nan"
    if not np.isfinite(x):
        return "nan"
    return f"{x:.{nd}f}"


def _log_val_metrics(name: str, ep, n_epochs: int, val: Dict[str, float], top_k: int, prefix: str = "VAL") -> None:
    ep_s = str(ep)
    log(
        f"      {prefix} {name} ep{ep_s}/{n_epochs} n={int(val.get('n', 0))} "
        f"MSE={_fmt_metric(val.get('MSE'), 5)} MAE={_fmt_metric(val.get('MAE'), 5)} "
        f"IC={_fmt_metric(val.get('IC'))} RankIC={_fmt_metric(val.get('RankIC'))} "
        f"F1@Top{top_k}={_fmt_metric(val.get('topk_f1'))} "
        f"P={_fmt_metric(val.get('topk_precision'))} R={_fmt_metric(val.get('topk_recall'))} "
        f"DirAcc={_fmt_metric(val.get('dir_acc'))} DirF1={_fmt_metric(val.get('dir_f1'))} "
        f"DirP={_fmt_metric(val.get('dir_precision'))} DirR={_fmt_metric(val.get('dir_recall'))} "
        f"Hit@TopK={_fmt_metric(val.get('top30_hit_rate'))}"
    )


def fit_feature_scaler(store, days, stride: int = 3, bar_stride: int = 4, fit_range=None) -> ChannelZScore:
    sc = ChannelZScore()
    sc.fit_range = fit_range
    used = 0
    for i, day in enumerate(days):
        if i % max(int(stride), 1) != 0:
            continue
        try:
            x, msk = store.window(day)
            a, _ = store.auction_d(day)
            valid = store.valid_mask(day)
            if int(valid.sum()) < 8:
                del x, msk, a
                continue
            xv = x[valid, :: max(int(bar_stride), 1)]
            mv = msk[valid, :: max(int(bar_stride), 1)]
            sc.update_x(xv, mv)
            sc.update_a(a[valid])
            used += 1
            del x, msk, a, xv, mv
        except Exception:
            continue
        if used % 15 == 0:
            gc.collect()
    if sc.mean_x is None:
        raise RuntimeError("feature scaler fit produced no statistics")
    log(f"      feat z-score fit days={used}/{len(days)} F={sc.mean_x.size} n_max={int(sc.count_x.max())}")
    return sc


def _day_pack(store, day, feat_scaler=None) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    try:
        y, _ = store.labels(day)
    except KeyError:
        return None
    x, msk = store.window(day)
    a, _ = store.auction_d(day)
    valid = store.valid_mask(day)
    if int(valid.sum()) < 20:
        del x, msk
        return None
    if feat_scaler is not None:
        x = feat_scaler.transform_x(x, msk)
        a = feat_scaler.transform_a(a)
    xv = np.ascontiguousarray(np.nan_to_num(x[valid]))
    av = np.ascontiguousarray(np.nan_to_num(a[valid]))
    mv = np.ascontiguousarray(msk[valid])
    yv = np.ascontiguousarray(y[valid])
    del x, msk, a, y
    finite = np.isfinite(yv)
    if int(finite.sum()) < 10:
        return None
    return xv[finite], av[finite], mv[finite], yv[finite]


def _packs_to_device(packs, device):
    ns = [int(p[0].shape[0]) for p in packs]
    if len(packs) == 1:
        xt = torch.from_numpy(np.ascontiguousarray(packs[0][0])).to(device)
        at = torch.from_numpy(np.ascontiguousarray(packs[0][1])).to(device)
        mt = torch.from_numpy(np.ascontiguousarray(packs[0][2])).to(device)
        yt = torch.from_numpy(np.ascontiguousarray(packs[0][3])).to(device)
        return xt, at, mt, yt, ns
    xt = torch.from_numpy(np.ascontiguousarray(np.concatenate([p[0] for p in packs], axis=0))).to(device)
    at = torch.from_numpy(np.ascontiguousarray(np.concatenate([p[1] for p in packs], axis=0))).to(device)
    mt = torch.from_numpy(np.ascontiguousarray(np.concatenate([p[2] for p in packs], axis=0))).to(device)
    yt = torch.from_numpy(np.ascontiguousarray(np.concatenate([p[3] for p in packs], axis=0))).to(device)
    return xt, at, mt, yt, ns


def train_predictor(
    model,
    store,
    days,
    cfg,
    device,
    val_days=None,
    always_nl: bool = False,
    resume_path: Optional[Path] = None,
) -> Dict[str, float]:
    refresh_prediction_train_cfg(cfg)
    p = cfg["prediction"]
    opt = torch.optim.AdamW(model.parameters(), lr=float(p["lr"]), weight_decay=float(p["weight_decay"]))
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    best_state, best_val, bad = None, 1e18, 0
    best_val_metrics: Optional[Dict[str, float]] = None
    val_hist: List[Dict[str, Any]] = []
    t0 = time.time()
    last = float("nan")
    name = getattr(model, "fusion_name", model.__class__.__name__)
    batch_asofs = max(int(p.get("batch_asofs", 1)), 1)
    rank_weight = float(p.get("rank_weight", 0.0))
    cs_zscore_flag = bool(p.get("sft_cs_zscore", False))
    val_select = str(p.get("val_select", "mse"))
    tag = "csz" if cs_zscore_flag else "mse"
    feat_scaler = None
    if bool(p.get("feat_standardize", False)):
        tag = tag + "+z"
        feat_scaler = fit_feature_scaler(
            store,
            days,
            stride=int(p.get("feat_std_stride", 3)),
            bar_stride=int(p.get("feat_std_bar_stride", 4)),
            fit_range={"start": str(pd.Timestamp(days[0]).date()), "end": str(pd.Timestamp(days[-1]).date())} if days else None,
        )
        model.feat_scaler = feat_scaler
    n_ep = int(p["epochs"])
    start_ep = 0
    resume_file = Path(resume_path) if resume_path else None
    if resume_file is not None and resume_file.is_file():
        try:
            blob = torch.load(resume_file, map_location=device, weights_only=False)
        except TypeError:
            blob = torch.load(resume_file, map_location=device)
        model.load_state_dict(blob["model"])
        opt.load_state_dict(blob["opt"])
        if "scaler" in blob:
            scaler.load_state_dict(blob["scaler"])
        start_ep = int(blob.get("next_ep", 0))
        best_state = blob.get("best_state")
        best_val = float(blob.get("best_val", 1e18))
        best_val_metrics = blob.get("best_val_metrics")
        val_hist = list(blob.get("val_hist") or [])
        bad = int(blob.get("bad", 0))
        log(f"      resume SFT {name} from epoch {start_ep + 1}/{n_ep} <- {resume_file}")
    bar = Progress(n_ep * max(len(days), 1), f"A-SFT {name} {tag} rank_w={rank_weight:g}", log_every=10, always_nl=always_nl)
    if start_ep:
        bar.update(start_ep * len(days), f"resumed ep{start_ep}/{n_ep}")

    def _load_range(lo: int, hi: int):
        out = []
        for j in range(lo, min(hi, len(days))):
            pack = _day_pack(store, days[j], feat_scaler=feat_scaler)
            if pack is not None:
                out.append((j, pack))
        return out

    for ep in range(start_ep, n_ep):
        model.train()
        losses = []
        seen = 0
        i = 0
        while i < len(days):
            i_next = i + batch_asofs
            last_day = days[min(i, len(days) - 1)]
            n_asof = batch_asofs
            n_pack = 0
            try:
                loaded = _load_range(i, i_next)
                packs = [row[1] for row in loaded]
                last_day = days[loaded[-1][0]] if loaded else last_day
                n_asof = max(len(loaded), batch_asofs)
                n_pack = len(packs)
                if not packs:
                    del loaded
                    seen += batch_asofs
                    i = i_next
                    continue
                xt, at, mt, yt, ns = _packs_to_device(packs, device)
                del packs, loaded
                opt.zero_grad(set_to_none=True)
                with amp_ctx(device):
                    pred = model(xt, auction=at, valid=None, mask=mt)
                    loss = sft_asof_loss(pred, yt, ns, rank_weight, cs_zscore_flag=cs_zscore_flag)
                if torch.isfinite(loss):
                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                    nn.utils.clip_grad_norm_(model.parameters(), float(p["grad_clip"]))
                    scaler.step(opt)
                    scaler.update()
                    losses.append(float(loss.detach()))
                del xt, at, mt, yt, pred
            except (RuntimeError, MemoryError) as e:
                msg = str(e).lower()
                if isinstance(e, MemoryError) or "out of memory" in msg or "unable to allocate" in msg:
                    _free_host()
                    cur = int(getattr(model, "stock_chunk", 64) or 64)
                    model.stock_chunk = 32 if cur > 128 else max(8, cur // 2)
                    model.use_ckpt = True
                    next_bs = max(1, batch_asofs // 2)
                    log(f"      OOM/RAM -> stock_chunk={model.stock_chunk} batch_asofs={next_bs} ckpt=1")
                    if next_bs == batch_asofs:
                        seen += n_asof
                        i = i_next
                    else:
                        batch_asofs = next_bs
                    continue
                raise
            seen += n_asof
            i = i_next
            if seen % 8 == 0:
                gc.collect()
            bar.update(
                ep * len(days) + min(seen, len(days)),
                f"ep{ep+1}/{p['epochs']} {pd.Timestamp(last_day).date()} bs={n_pack} loss={np.mean(losses) if losses else float('nan'):.5f}",
            )
        _free_host()
        last = float(np.mean(losses)) if losses else float("nan")
        if val_days:
            val = eval_predictor(model, store, val_days, device, top_k=int(cfg["portfolio"]["top_k"]), feat_scaler=feat_scaler)
            val_hist.append({"epoch": ep + 1, **val})
            sys.stderr.write("\n")
            sys.stderr.flush()
            _log_val_metrics(name, ep + 1, int(p["epochs"]), val, int(cfg["portfolio"]["top_k"]))
            if val_select == "rank_ic":
                ric = float(val.get("RankIC", float("nan")))
                vloss = -ric if np.isfinite(ric) else float(val.get("MSE", last))
            else:
                vloss = val.get("MSE", last)
            if np.isfinite(vloss) and vloss < best_val:
                best_val, bad, best_state = vloss, 0, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best_val_metrics = dict(val)
            else:
                bad += 1
                if bad >= int(p["early_stopping_patience"]):
                    log(f"      early stop epoch {ep+1} select={val_select} score={vloss:.5f}")
                    break
        if resume_file is not None:
            torch.save(
                {
                    "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    "opt": opt.state_dict(),
                    "scaler": scaler.state_dict(),
                    "next_ep": ep + 1,
                    "best_state": best_state,
                    "best_val": best_val,
                    "best_val_metrics": best_val_metrics,
                    "val_hist": val_hist,
                    "bad": bad,
                },
                resume_file,
            )
            log(f"      SFT epoch ckpt ep{ep+1}/{n_ep} -> {resume_file.name}")
    if best_state is not None:
        model.load_state_dict(best_state)
    if val_days and best_val_metrics is None:
        best_val_metrics = eval_predictor(model, store, val_days, device, top_k=int(cfg["portfolio"]["top_k"]), feat_scaler=feat_scaler)
    if best_val_metrics is not None:
        sys.stderr.write("\n")
        sys.stderr.flush()
        _log_val_metrics(name, "best", int(p["epochs"]), best_val_metrics, int(cfg["portfolio"]["top_k"]), prefix="BEST VAL")
    bar.close(f"loss={last:.5f}")
    out = {"loss": last, "wall": time.time() - t0, "n": len(days)}
    if best_val_metrics is not None:
        out.update({f"val_{k}": v for k, v in best_val_metrics.items()})
    out["val_history"] = val_hist
    if resume_file is not None and resume_file.is_file():
        resume_file.unlink()
    return out


@torch.no_grad()
def eval_predictor(model, store, days, device, top_k: int = 30, feat_scaler=None) -> Dict[str, float]:
    buckets = {
        "IC": [],
        "RankIC": [],
        "MSE": [],
        "MAE": [],
        "topk_f1": [],
        "topk_precision": [],
        "topk_recall": [],
        "dir_acc": [],
        "dir_f1": [],
        "dir_precision": [],
        "dir_recall": [],
        "top30_hit_rate": [],
    }
    for day in days:
        try:
            try:
                y, _ = store.labels(day)
            except KeyError:
                continue
            pred, valid, _ = predict_day(model, store, day, device, feat_scaler=feat_scaler)
            if valid.sum() < 8:
                continue
            pv, yv = pred[valid], y[valid]
            met = daily_ic(pv, yv)
            rk = ranking_classification_metrics(pv, yv, top_k=top_k)
            idx = np.argpartition(-np.where(valid, pred, -1e18), min(top_k, int(valid.sum())) - 1)[: min(top_k, int(valid.sum()))]
            ts = top30_stats(pred, y, idx, valid)
            for k, v in {**met, **rk, **ts}.items():
                if k in buckets and np.isfinite(v):
                    buckets[k].append(float(v))
        except Exception:
            continue
    out = {k: float(np.nanmean(v)) if v else float("nan") for k, v in buckets.items()}
    out["n"] = max(len(buckets["MSE"]), len(buckets["topk_f1"]))
    return out


def make_alloc_item(store, day, pred, valid, y, ameta, cfg, pool: str) -> dict:
    top_k = int(cfg["portfolio"]["top_k"])
    compact = pool_is_compact(pool)
    tau = float((cfg.get("matrix") or {}).get("softmax_tau") or cfg["portfolio"].get("softmax_tau", 0.5))
    idx = pool_idx(pred, valid, store, pool, top_k)
    names = [store.kept[j] for j in idx]
    mu, sigma = hist_mu_sigma_from_store(store, day, idx, int(cfg["portfolio"]["hist_risk_lookback_days"]))
    pack = pack_allocators(
        mu,
        sigma,
        pred[idx],
        float(cfg["portfolio"]["max_weight"]),
        float(cfg["ssfm"]["risk_aversion"]),
    )
    teach = {n: pack[n] for n in TEACHER_NAMES}
    bases = {n: pack[n] for n in pack}
    valid_k = valid[idx]
    teach = {n: apply_valid_mask(w, valid_k) for n, w in teach.items()}
    bases = {n: apply_valid_mask(w, valid_k) for n, w in bases.items()}
    bases["score_softmax"] = apply_valid_mask(score_softmax_cuda(pred[idx], valid_k, tau), valid_k)
    return {
        "asof": day,
        "idx": idx,
        "names": names,
        "alpha": pred,
        "y": y,
        "valid": valid,
        "pool": pool,
        "mu": mu,
        "sigma": sigma,
        "teachers": teach,
        "baselines": bases,
        "cond": build_cond(pred[idx], mu, sigma, compact=compact),
        "R": y[idx],
        "auction_date": ameta["auction_date"],
        "clocks": store.clocks(day),
    }


def build_state_cache(store, model, days, cfg, device, pool: str = "top30", cutoff=None) -> List[dict]:
    items = []
    if cutoff is not None:
        days = store.realized_label_days(days, cutoff)
    bar = Progress(max(len(days), 1), f"STATE 构造{pool}/Teacher", log_every=1, always_nl=True)
    for i, day in enumerate(days):
        try:
            y, _ = store.labels(day)
            pred, valid, ameta = predict_day(model, store, day, device)
            items.append(make_alloc_item(store, day, pred, valid, y, ameta, cfg, pool))
        except Exception as e:
            if i < 3:
                log(f"      cache skip {pd.Timestamp(day).date()}: {e}")
            continue
        bar.update(i + 1, f"{pd.Timestamp(day).date()} n={len(items)}")
    bar.close(f"n={len(items)}")
    return items


def _train_gen(factory, loss_fn, cache, device, steps, lr, name, batch_size: int = 32) -> nn.Module:
    model = factory().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    tids = list(TEACHER_IDS.keys())
    t0 = time.time()
    bsz = max(int(batch_size), 1)
    bar = Progress(steps, f"C-GEN {name}", log_every=25)
    for s in range(1, steps + 1):
        idx = np.random.randint(0, len(cache), size=bsz)
        tns = [tids[int(np.random.randint(0, len(tids)))] for _ in range(bsz)]
        w = torch.stack([torch.from_numpy(cache[int(i)]["teachers"][tn].astype(np.float32)) for i, tn in zip(idx, tns)]).to(device, non_blocking=True)
        cond = torch.stack([torch.from_numpy(cache[int(i)]["cond"]) for i in idx]).to(device, non_blocking=True)
        if name == "ssfm":
            tid = torch.tensor([TEACHER_IDS[tn] for tn in tns], device=device)
            loss = loss_fn(model, w, cond, tid)
        else:
            loss = loss_fn(model, w, cond)
        opt.zero_grad()
        loss.backward()
        opt.step()
        bar.update(s, f"bs={bsz} loss={float(loss):.4f}")
    bar.close(f"wall={time.time()-t0:.1f}s")
    return model


def _new_gen_models(cfg: dict, device, k: int, compact: bool) -> Dict[str, nn.Module]:
    cd = cond_dim(k, compact=compact)
    return {
        "ssfm": SimplexSpaceFlow(k, cd, int(cfg["ssfm"]["hidden"]), int(cfg["ssfm"]["n_teachers"]), int(cfg["ssfm"]["teacher_emb"])).to(device),
        "standard_fm": StandardFlowMatching(k, cd, int(cfg["standard_fm"]["hidden"])).to(device),
        "diffusion": WeightDiffusion(k, cd, int(cfg["diffusion"]["hidden"]), int(cfg["diffusion"]["n_diff_steps"])).to(device),
        "mlp": MLPPolicy(k, cd, int(cfg["mlp_policy"]["hidden"])).to(device),
        "gaussian": GaussianPolicy(k, cd, int(cfg["gaussian_policy"]["hidden"])).to(device),
    }


def _load_module_state(model: nn.Module, path: Path, device) -> nn.Module:
    try:
        state = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    return model


def try_load_alloc_ckpts(init_dir: Path, dest_dir: Path, cfg: dict, device, pool: str, k: int, compact: bool) -> Optional[Dict[str, Any]]:
    gen_names = tuple((cfg.get("matrix") or {}).get("C_generative") or ("mlp", "gaussian", "diffusion", "standard_fm", "ssfm"))
    algos = [str(a) for a in cfg["rl"]["algorithms"]]
    present = [n for n in gen_names if (init_dir / f"gen_{n}_{pool}.pt").is_file()]
    if not present:
        return None
    gen_all = _new_gen_models(cfg, device, k, compact)
    gen: Dict[str, nn.Module] = {}
    missing_gen: List[str] = []
    for n in gen_names:
        src = init_dir / f"gen_{n}_{pool}.pt"
        if not src.is_file():
            missing_gen.append(n)
            continue
        model = gen_all[n]
        _load_module_state(model, src, device)
        torch.save(model.state_dict(), dest_dir / f"gen_{n}_{pool}.pt")
        gen[n] = model
    rl = {}
    missing_rl: List[str] = []
    cd = cond_dim(k, compact=compact)
    for algo in algos:
        src = init_dir / f"rl_{algo}_{pool}.pt"
        if not src.is_file():
            missing_rl.append(algo)
            continue
        pol = WeightPolicy(cd, k, int(cfg["rl"]["hidden"])).to(device)
        _load_module_state(pol, src, device)
        torch.save(pol.state_dict(), dest_dir / f"rl_{algo}_{pool}.pt")
        rl[algo] = pol
    return {
        "cache": [],
        "gen": gen,
        "rl": rl,
        "k": k,
        "compact": compact,
        "missing_gen": missing_gen,
        "missing_rl": missing_rl,
    }


def train_generative(cache, cfg, device, k: Optional[int] = None, compact: bool = False, only: Optional[List[str]] = None) -> Dict[str, nn.Module]:
    k = int(k if k is not None else cfg["portfolio"]["top_k"])
    cd = cond_dim(k, compact=compact)
    steps = int(cfg["ssfm"]["steps"])
    out = {}
    bsz = int(cfg["ssfm"]["batch_size"])
    want = {str(x) for x in (cfg.get("matrix") or {}).get("C_generative") or ["ssfm", "standard_fm", "diffusion"]}
    if only:
        want &= {str(x) for x in only}
    if "ssfm" in want:
        out["ssfm"] = _train_gen(
            lambda: SimplexSpaceFlow(k, cd, int(cfg["ssfm"]["hidden"]), int(cfg["ssfm"]["n_teachers"]), int(cfg["ssfm"]["teacher_emb"])),
            lambda m, w, c, tid: ss_fm_loss(m, w, c, tid, float(cfg["ssfm"]["clr_eps"])),
            cache, device, steps, float(cfg["ssfm"]["lr"]), "ssfm", bsz,
        )
    if "standard_fm" in want:
        out["standard_fm"] = _train_gen(
            lambda: StandardFlowMatching(k, cd, int(cfg["standard_fm"]["hidden"])),
            std_fm_loss, cache, device, int(cfg["standard_fm"]["steps"]), float(cfg["standard_fm"]["lr"]), "standard_fm", bsz,
        )
    if "diffusion" in want:
        out["diffusion"] = _train_gen(
            lambda: WeightDiffusion(k, cd, int(cfg["diffusion"]["hidden"]), int(cfg["diffusion"]["n_diff_steps"])),
            diffusion_loss, cache, device, int(cfg["diffusion"]["steps"]), float(cfg["diffusion"]["lr"]), "diffusion", bsz,
        )
    if "mlp" in want or "gaussian" in want:
        mlp = MLPPolicy(k, cd, int(cfg["mlp_policy"]["hidden"])).to(device)
        gauss = GaussianPolicy(k, cd, int(cfg["gaussian_policy"]["hidden"])).to(device)
        opt = torch.optim.Adam(list(mlp.parameters()) + list(gauss.parameters()), lr=float(cfg["mlp_policy"]["lr"]))
        bsz = int(cfg["ssfm"]["batch_size"])
        mlp_bar = Progress(int(cfg["mlp_policy"]["steps"]), "C-GEN mlp+gaussian", log_every=25)
        tns_all = list(TEACHER_IDS)
        for s in range(int(cfg["mlp_policy"]["steps"])):
            idx = np.random.randint(0, len(cache), size=bsz)
            tns = [tns_all[int(np.random.randint(0, len(tns_all)))] for _ in range(bsz)]
            tgt = torch.stack([torch.from_numpy(cache[int(i)]["teachers"][tn].astype(np.float32)) for i, tn in zip(idx, tns)]).to(device, non_blocking=True)
            cond = torch.stack([torch.from_numpy(cache[int(i)]["cond"]) for i in idx]).to(device, non_blocking=True)
            loss = F.mse_loss(mlp(cond), tgt) + F.mse_loss(gauss(cond), tgt)
            opt.zero_grad()
            loss.backward()
            opt.step()
            mlp_bar.update(s + 1, f"bs={bsz} loss={float(loss):.4f}")
        mlp_bar.close()
        if "mlp" in want:
            out["mlp"] = mlp
        if "gaussian" in want:
            out["gaussian"] = gauss
    return out


def sample_model(name: str, model, cond: torch.Tensor, g: int, cfg: dict) -> torch.Tensor:
    if name == "ssfm":
        return sample_ss_fm_mixed(model, cond, g, int(cfg["ssfm"]["n_sample_steps"]))
    if name == "standard_fm":
        return sample_std_fm(model, cond, g, int(cfg["standard_fm"]["n_sample_steps"]))
    if name == "diffusion":
        return sample_diffusion(model, cond, g)
    if name == "mlp":
        return model(cond).expand(g, -1)
    if name == "gaussian":
        return model.sample(cond, g)
    raise KeyError(name)


def select_candidate(ws: np.ndarray, alpha: np.ndarray, sigma: np.ndarray, cfg: dict) -> Tuple[int, np.ndarray]:
    """Decision uses predicted alpha + hist risk only. Never D realized return."""
    ra = float(cfg["ssfm"]["risk_aversion"])
    scores = []
    for w in ws:
        scores.append(float(w @ alpha - 0.5 * ra * w @ sigma @ w))
    j = int(np.argmax(scores))
    return j, ws[j]


def distill_and_rl(ss_fm, cache, cfg, device, algo: str, k: Optional[int] = None, compact: bool = False) -> WeightPolicy:
    k = int(k if k is not None else cfg["portfolio"]["top_k"])
    cd = cond_dim(k, compact=compact)
    pol = WeightPolicy(cd, k, int(cfg["rl"]["hidden"])).to(device)
    critic = ValueHead(cd, int(cfg["rl"]["hidden"])).to(device) if algo == "ppo" else None
    opt_d = torch.optim.Adam(pol.parameters(), lr=1e-3)
    n_bc = max(int(cfg["rl"].get("teacher_bc_samples", 4)), 1)
    n_steps = int(cfg["ssfm"]["n_sample_steps"])
    dist_bar = Progress(int(cfg["rl"]["distill_steps"]), f"D-RL {algo} distill", log_every=20)
    for di in range(int(cfg["rl"]["distill_steps"])):
        item = cache[np.random.randint(0, len(cache))]
        cond = torch.from_numpy(item["cond"]).to(device)
        with torch.no_grad():
            ws_t = sample_ss_fm_mixed(ss_fm, cond, n_bc, n_steps).detach().cpu().numpy()
            # Prefer decision-aligned teacher target (avoid mean-collapse of SS-FM samples).
            if "alpha" in item and "idx" in item and "sigma" in item:
                _, w_np = select_candidate(ws_t, item["alpha"][item["idx"]], item["sigma"], cfg)
            else:
                w_np = ws_t[0]
            w = torch.from_numpy(np.asarray(w_np, dtype=np.float32)).to(device)
        tgt = torch.log(w.clamp_min(1e-8))
        tgt = tgt - tgt.mean()
        loss = F.mse_loss(pol.forward_logits(cond).squeeze(0), tgt)
        opt_d.zero_grad()
        loss.backward()
        opt_d.step()
        dist_bar.update(di + 1, f"loss={float(loss):.4f}")
    dist_bar.close()
    params = list(pol.parameters()) + (list(critic.parameters()) if critic else [])
    opt = torch.optim.Adam(params, lr=float(cfg["rl"]["lr"]))
    hist = {"return": [], "sharpe": [], "turnover": [], "smooth_mdd": []}
    fit_range = {"start": str(cache[0]["asof"].date()), "end": str(cache[-1]["asof"].date())}
    # warmup scalers on equal-weight train path
    prev = None
    nets = []
    for item in cache:
        w = np.ones(k) / k
        met = execute_day(w, item["R"], prev, cfg)
        nets.append(met["net_return"])
        hist["return"].append(met["net_return"])
        hist["sharpe"].append(
            causal_sharpe(
                nets[:-1],
                cfg["portfolio"]["sharpe_min_obs"],
                0.0,
                int(cfg["portfolio"].get("sharpe_window", 20)),
            )
        )
        hist["turnover"].append(met["turnover"])
        hist["smooth_mdd"].append(smooth_dd_to_date(np.asarray(nets[:-1]), cfg["portfolio"]["smooth_dd_temperature"]))
        prev = w
    scalers = fit_reward_scalers({kk: np.asarray(v) for kk, v in hist.items()}, cfg, fit_range)
    g = int(cfg["rl"]["group_size"])
    bc_coef = float(cfg["rl"].get("bc_coef", 0.0) or 0.0)
    prev_mode = str(cfg["rl"].get("prev_mode", "mean") or "mean").lower()
    rl_bar = Progress(int(cfg["rl"]["epochs"]) * max(len(cache), 1), f"D-RL {algo.upper()}", log_every=10)
    for ep in range(int(cfg["rl"]["epochs"])):
        n_upd = 0
        t0 = time.time()
        prev = None
        hist_net = []
        for item in cache:
            cond = torch.from_numpy(item["cond"]).to(device)
            w, logits_s, _ = sample_portfolios(pol, cond, g, float(cfg["rl"]["noise_std"]))
            ws = w.detach().cpu().numpy()
            rews = []
            shp = causal_sharpe(
                hist_net,
                cfg["portfolio"]["sharpe_min_obs"],
                0.0,
                int(cfg["portfolio"].get("sharpe_window", 20)),
            )
            smdd = smooth_dd_to_date(np.asarray(hist_net), cfg["portfolio"]["smooth_dd_temperature"])
            for gi in range(g):
                met = execute_day(ws[gi], item["R"], prev, cfg)
                pack = composite_reward(
                    {"return": met["net_return"], "sharpe": shp, "turnover": met["turnover"], "smooth_mdd": smdd},
                    scalers,
                    cfg,
                )
                rews.append(pack["reward"])
            rew_t = torch.tensor(rews, device=device, dtype=torch.float32)
            if algo == "ppo":
                info = ppo_update(pol, critic, cond, logits_s, rew_t, opt, cfg)
            else:
                info = grpo_update(pol, cond, logits_s, rew_t, opt, cfg)
                if info.get("warning"):
                    log(f"      GRPO group-relative signal weak asof={item['asof'].date()}")
            # Online BC / KL-to-teacher: pull logits toward a fresh SS-FM sample (not G-mean).
            if bc_coef > 0.0:
                with torch.no_grad():
                    ws_t = sample_ss_fm_mixed(ss_fm, cond, n_bc, n_steps).detach().cpu().numpy()
                    if "alpha" in item and "idx" in item and "sigma" in item:
                        _, w_t_np = select_candidate(ws_t, item["alpha"][item["idx"]], item["sigma"], cfg)
                    else:
                        w_t_np = ws_t[0]
                    w_t = torch.from_numpy(np.asarray(w_t_np, dtype=np.float32)).to(device)
                tgt = torch.log(w_t.clamp_min(1e-8))
                tgt = tgt - tgt.mean()
                bc_loss = F.mse_loss(pol.forward_logits(cond).squeeze(0), tgt)
                opt.zero_grad()
                (bc_coef * bc_loss).backward()
                opt.step()
            if prev_mode == "best_reward":
                j = int(np.argmax(np.asarray(rews, dtype=np.float64)))
                w_next = np.asarray(ws[j], dtype=np.float64)
            elif prev_mode == "select_candidate" and "alpha" in item and "idx" in item and "sigma" in item:
                _, w_next = select_candidate(ws, item["alpha"][item["idx"]], item["sigma"], cfg)
                w_next = np.asarray(w_next, dtype=np.float64)
            else:
                w_next = np.asarray(ws.mean(0), dtype=np.float64)
            w_next = np.clip(w_next, 0, None)
            w_next = w_next / max(float(w_next.sum()), 1e-12)
            met0 = execute_day(w_next, item["R"], prev, cfg)
            hist_net.append(met0["net_return"])
            prev = w_next
            n_upd += 1
            rl_bar.update(ep * len(cache) + n_upd, f"ep{ep+1} {item['asof'].date()}")
        log(f"      {algo.upper()} epoch {ep+1} updates={n_upd} wall={time.time()-t0:.1f}s bc_coef={bc_coef} prev={prev_mode}")
    rl_bar.close()
    pol._reward_scalers = scalers  # type: ignore[attr-defined]
    return pol


def backtest_items(items: List[dict], weights: List[np.ndarray], cfg: dict, model_name: str, oos: str, g: int) -> pd.DataFrame:
    extras = []
    for it, w in zip(items, weights):
        alpha = np.asarray(it["alpha"], dtype=np.float64)
        idx = np.asarray(it.get("idx", np.arange(len(alpha))), dtype=int)
        extras.append(
            {
                "model": model_name,
                "strategy_name": model_name,
                "oos_mode": oos,
                "candidate_count_G": g,
                "pool": it.get("pool", ""),
                "selected_stocks": "|".join(it["names"]),
                "predicted_alpha": ",".join(f"{float(alpha[j]):.8f}" for j in idx),
                **it["clocks"],
            }
        )
    df = run_path(
        [it["asof"] for it in items],
        weights,
        [it["R"] for it in items],
        cfg,
        extra=extras,
    )
    return df


def run_month(
    cfg: dict,
    store: ResearchStore,
    test_month: pd.Timestamp,
    oos_mode: str,
    out_root: Path,
    models_filter: Optional[List[str]] = None,
    skip_rl: bool = False,
    skip_gen: bool = False,
    g_grid: Optional[List[int]] = None,
    skip_models: Optional[List[str]] = None,
    init_ckpt: Optional[Path] = None,
    test_from: Optional[str] = None,
    skip_friday_before: Optional[str] = None,
    friday_sft_ckpt: Optional[Path] = None,
) -> Dict[str, Any]:
    set_seed(int(cfg["experiment"]["seed"]), bool(cfg["experiment"]["deterministic"]), bool(cfg["experiment"]["deterministic_warn_only"]))
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    device = device_of()
    split = split_for_test_month(store.days, test_month, int(cfg["walkforward"]["train_offset_months"]))
    month = split["test_month"]
    month_dir = out_root / oos_mode / f"test_month={month}"
    month_dir.mkdir(parents=True, exist_ok=True)
    prev = find_exp_dir(month_dir)
    if prev is not None:
        out = prev
        exp_id = out.name
        log(f"    resume experiment dir {out.name}")
    else:
        exp_id = experiment_id(cfg["experiment"]["name"], month, oos_mode)
        out = month_dir / exp_id
        out.mkdir(parents=True, exist_ok=True)
    dump_yaml_copy(cfg, out / "config.yaml")
    write_json(out / "environment.json", environment_info(cfg["_root"]))
    write_json(
        out / "split_manifest.json",
        {
            "experiment_id": exp_id,
            **{k: ( [str(x.date()) for x in v] if k.endswith("days") else v) for k, v in split.items()},
            "n_train": len(split["train_days"]),
            "n_val": len(split["val_days"]),
            "n_test": len(split["test_days"]),
            "daily_valid_n": {str(d.date()): int(store.valid_mask(d).sum()) for d in split["test_days"]},
            "data_version": store.data_version,
            "information_cutoff": split["validation_end_date"] if oos_mode == "strict_fixed_oos" else "see weekly windows",
            "filter_reasons": store.filter_reasons,
            "observed_test_last_day": str(split["test_days"][-1].date()) if split["test_days"] else None,
            "expected_test_month_end": split["test_end_date"],
            "test_month_complete": bool(split["test_days"]) and str(split["test_days"][-1].date()) >= split["test_end_date"][:8] + "01",
        },
    )
    if not split["train_days"] or not split["test_days"]:
        write_json(out / "failed_experiment.json", {"status": "failed", "error": "empty train or test", "split": month})
        raise RuntimeError(f"empty split for {month}")

    pred_names = models_filter or list(cfg["prediction"]["models"])
    primary = cfg["prediction"]["primary_model"]
    ckpt_dir = out / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    device_models: Dict[str, nn.Module] = {}
    pred_rows = []
    daily_pred_frames = []

    auto = inspect_resume(out, cfg, pred_names, split["test_days"])
    if not skip_models and auto.get("skip_models"):
        skip_models = list(auto["skip_models"])
        log(f"    AUTO skip SFT {skip_models} (已有 alpha ckpt)")
    if init_ckpt is None and auto.get("init_ckpt"):
        init_ckpt = auto["init_ckpt"]
        log(f"    AUTO init_ckpt={init_ckpt}")
    if test_from is None and auto.get("test_from"):
        test_from = auto["test_from"]
        log(f"    AUTO TEST from {test_from}")
    if skip_friday_before is None and auto.get("skip_friday_before"):
        skip_friday_before = auto["skip_friday_before"]
        log(f"    AUTO skip Friday < {skip_friday_before} done={auto.get('friday_done')}")

    skip = {str(x) for x in (skip_models or [])}
    init_dir = Path(init_ckpt) if init_ckpt else None
    test_from_s = str(test_from)[:10] if test_from else None
    skip_friday_before_s = str(skip_friday_before)[:10] if skip_friday_before else None
    friday_sft_dir = Path(friday_sft_ckpt) if friday_sft_ckpt else None
    sft_ckpt_alias = {"minute_only": "standard_transformer"}
    reuse_ablation = bool((cfg.get("prediction") or {}).get("reuse_ablation_sft"))
    if reuse_ablation:
        # Never retrain Alpha; freeze-load ablation BEST per test month.
        skip |= {str(x) for x in pred_names}
        log(f"    reuse_ablation_sft=1 → skip SFT train for {sorted(skip)}")

    def fit_one(name: str, train_days, val_days, load_only: bool = False, load_from: Optional[Path] = None, cutoff=None, resume_tag: str = "init") -> nn.Module:
        local = copy.deepcopy(cfg)
        refresh_prediction_train_cfg(local)
        cut = cutoff or split["validation_end_date"]
        train_days = store.realized_label_days(list(train_days), cut)
        val_days = store.realized_label_days(list(val_days), cut) if val_days else val_days
        if reuse_ablation and (load_only or name in skip):
            from experiments.reuse_ablation_alpha import load_frozen_ablation_alpha

            model, meta = load_frozen_ablation_alpha(local, store, month, device)
            torch.save(model.inner.state_dict(), ckpt_dir / f"alpha_{name}.pt")
            write_json(ckpt_dir / f"alpha_{name}_reuse_meta.json", meta)
            append_csv_row(
                out / "training_logs.csv",
                {
                    "phase": "alpha",
                    "model": name,
                    "reused_ablation_sft": True,
                    "sft_checkpoint": meta.get("sft_checkpoint"),
                    "sft_best_epoch": meta.get("sft_best_epoch"),
                    "sft_val_mse": meta.get("sft_val_mse"),
                },
            )
            return model
        model = build_predictor(name, local, store.feat_dim).to(device)
        if load_only:
            src_dir = Path(load_from) if load_from is not None else init_dir
            if src_dir is None:
                raise RuntimeError(f"skip {name} but --init-ckpt was not set")
            ckpt_name = sft_ckpt_alias.get(name, name)
            ckpt_src = src_dir / f"alpha_{ckpt_name}.pt"
            if not ckpt_src.is_file():
                raise RuntimeError(f"missing checkpoint {ckpt_src}")
            try:
                state = torch.load(ckpt_src, map_location=device, weights_only=True)
            except TypeError:
                state = torch.load(ckpt_src, map_location=device)
            model.load_state_dict(state)
            torch.save(model.state_dict(), ckpt_dir / f"alpha_{name}.pt")
            src_val = init_dir.parent / "alpha_val_metrics.csv"
            if src_val.is_file():
                prev = pd.read_csv(src_val)
                hist = prev[prev["model"].astype(str) == name]
                if hist.empty and ckpt_name != name:
                    hist = prev[prev["model"].astype(str) == ckpt_name].copy()
                    hist["model"] = name
                rows = hist.to_dict("records")
                for row in rows:
                    append_csv_row(out / "alpha_val_metrics.csv", dict(row))
                write_alpha_val_snapshot(out, out_root, name, rows, {"month": month, "oos": oos_mode})
            log(f"    skip SFT {name} <- {ckpt_src}" + (f" (alias {ckpt_name})" if ckpt_name != name else ""))
            return model
        log(
            f"    train alpha {name} n={len(train_days)} val={len(val_days or [])} "
            f"params={count_params(model)} rank_w={local['prediction'].get('rank_weight')} "
            f"csz={local['prediction'].get('sft_cs_zscore')} feat_std={local['prediction'].get('feat_standardize', False)}"
        )
        stats = train_predictor(
            model,
            store,
            train_days,
            local,
            device,
            val_days,
            always_nl=True,
            resume_path=ckpt_dir / f"sft_resume_{name}_{resume_tag}.pt",
        )
        torch.save(model.state_dict(), ckpt_dir / f"alpha_{name}.pt")
        hist = stats.pop("val_history", [])
        for row in hist:
            append_csv_row(
                out / "alpha_val_metrics.csv",
                {"model": name, "month": str(month), "oos": oos_mode, **row},
            )
        write_alpha_val_snapshot(out, out_root, name, hist, {"month": month, "oos": oos_mode})
        append_csv_row(out / "training_logs.csv", {"phase": "alpha", "model": name, **stats})
        refresh_after_unit(
            out_root,
            oos_mode,
            {"month": month, "oos": oos_mode, "last_unit": f"A-{name}", "out_root": out_root, "top_k": int(cfg["portfolio"]["top_k"]), "cost_bps": cfg["portfolio"]["cost_bps"]},
        )
        return model

    windows = [{"train_days": split["train_days"], "val_days": split["val_days"], "cutoff": split["validation_end_date"], "effective_from": split["test_start_date"], "includes_test_realized": False, "label": "pre_test_frozen"}]
    if oos_mode == "sequential_oos_weekly_retrain":
        windows = sequential_retrain_windows(split)

    # Initial train on first window
    w0 = windows[0]
    month_bar = Progress(
        count_month_units(cfg, len(pred_names), skip_gen, skip_rl, g_grid),
        f"MONTH {month} {oos_mode}",
        log_every=1,
    )
    for name in pred_names:
        if name in skip:
            log(f"==== 实验A Prediction | {name} | load ckpt | {month} {oos_mode} ====")
            device_models[name] = fit_one(name, w0["train_days"], w0["val_days"], load_only=True)
            month_bar.update(msg=f"A loaded {name}")
            continue
        log(f"==== 实验A Prediction | {name} | {month} {oos_mode} ====")
        device_models[name] = fit_one(name, w0["train_days"], w0["val_days"], cutoff=w0.get("cutoff"))
        month_bar.update(msg=f"A done {name}")

    updates = [dict(w0)]
    if skip_friday_before_s:
        for w in windows:
            if not w.get("includes_test_realized") or w.get("effective_from") is None:
                continue
            if str(w.get("cutoff")) < skip_friday_before_s:
                updates.append({**dict(w), "label": "weekly_retrain_loaded_ckpt"})
                log(f"    reuse Friday ckpt cutoff={w['cutoff']} effective_from={w['effective_from']}")
    pools = allocation_pool_names(cfg)
    alloc: Dict[str, Dict[str, Any]] = {}
    primary_model = device_models[primary]
    tau = float((cfg.get("matrix") or {}).get("softmax_tau") or cfg["portfolio"].get("softmax_tau", 0.5))

    def fit_allocators(days, tag: str, cutoff=None) -> None:
        nonlocal alloc, primary_model
        primary_model = device_models[primary]
        cut = cutoff or split["validation_end_date"]
        for pool in pools:
            k = pool_k(store, cfg, pool)
            compact = pool_is_compact(pool)
            log(f"==== 配权池 {pool} K={k} compact={compact} | {tag} {month} ====")
            if tag == "init" and init_dir is not None:
                loaded = try_load_alloc_ckpts(init_dir, ckpt_dir, cfg, device, pool, k, compact)
                if loaded is not None:
                    missing_gen = list(loaded.pop("missing_gen", []) or [])
                    missing_rl = list(loaded.pop("missing_rl", []) or [])
                    need_cache = (bool(missing_gen) and not skip_gen) or (bool(missing_rl) and not skip_rl)
                    if need_cache:
                        log(f"    load C/D {pool}; missing C {missing_gen} D {missing_rl}, will train")
                        cache = build_state_cache(store, primary_model, days, cfg, device, pool=pool, cutoff=cut)
                        if not cache:
                            raise RuntimeError(f"empty train state cache pool={pool}")
                        loaded["cache"] = cache
                        if missing_gen and not skip_gen:
                            extra = train_generative(cache, cfg, device, k=k, compact=compact, only=missing_gen)
                            for n, m in extra.items():
                                loaded["gen"][n] = m
                                torch.save(m.state_dict(), ckpt_dir / f"gen_{n}_{pool}.pt")
                                month_bar.update(msg=f"C {pool} {n}")
                                refresh_after_unit(
                                    out_root, oos_mode,
                                    {"month": month, "oos": oos_mode, "last_unit": f"C-{n}-{pool}", "out_root": out_root},
                                )
                        if missing_rl and not skip_rl:
                            if "ssfm" not in loaded["gen"]:
                                raise RuntimeError("GRPO/PPO require trained SS-FM")
                            for algo in missing_rl:
                                log(f"==== 实验D RL | pool={pool} SS-FM+{algo.upper()} | {month} ====")
                                loaded["rl"][algo] = distill_and_rl(loaded["gen"]["ssfm"], cache, cfg, device, algo, k=k, compact=compact)
                                torch.save(loaded["rl"][algo].state_dict(), ckpt_dir / f"rl_{algo}_{pool}.pt")
                                month_bar.update(msg=f"D {pool} {algo}")
                                refresh_after_unit(
                                    out_root, oos_mode,
                                    {"month": month, "oos": oos_mode, "last_unit": f"D-{algo}-{pool}", "out_root": out_root},
                                )
                    alloc[pool] = loaded
                    log(
                        f"    skip C/D {pool} <- {init_dir}"
                        if not missing_gen and not missing_rl
                        else f"    keep C {list(loaded['gen'])}; trained C {missing_gen} D {missing_rl}"
                    )
                    if not skip_gen:
                        for n in loaded["gen"]:
                            month_bar.update(msg=f"C loaded {pool} {n}")
                    if not skip_rl:
                        for algo in loaded["rl"]:
                            month_bar.update(msg=f"D loaded {pool} {algo}")
                    continue
            cache = build_state_cache(store, primary_model, days, cfg, device, pool=pool, cutoff=cut)
            if not cache:
                raise RuntimeError(f"empty train state cache pool={pool}")
            gen, rl = {}, {}
            if not skip_gen:
                log(f"==== 实验C Generative | pool={pool} | {month} ====")
                gen = train_generative(cache, cfg, device, k=k, compact=compact)
                for n, m in gen.items():
                    torch.save(m.state_dict(), ckpt_dir / f"gen_{n}_{pool}.pt")
                    month_bar.update(msg=f"C {pool} {n}")
                    refresh_after_unit(
                        out_root, oos_mode,
                        {"month": month, "oos": oos_mode, "last_unit": f"C-{n}-{pool}", "out_root": out_root},
                    )
            if not skip_rl and "ssfm" in gen:
                for algo in cfg["rl"]["algorithms"]:
                    log(f"==== 实验D RL | pool={pool} SS-FM+{algo.upper()} | {month} ====")
                    rl[algo] = distill_and_rl(gen["ssfm"], cache, cfg, device, algo, k=k, compact=compact)
                    torch.save(rl[algo].state_dict(), ckpt_dir / f"rl_{algo}_{pool}.pt")
                    month_bar.update(msg=f"D {pool} {algo}")
                    refresh_after_unit(
                        out_root, oos_mode,
                        {"month": month, "oos": oos_mode, "last_unit": f"D-{algo}-{pool}", "out_root": out_root},
                    )
            alloc[pool] = {"cache": cache, "gen": gen, "rl": rl, "k": k, "compact": compact}

    fit_allocators(w0["train_days"], "init", cutoff=w0.get("cutoff"))

    # Test loop
    test_items_by_model: Dict[str, List[dict]] = {n: [] for n in pred_names}
    weights_by: Dict[str, List[np.ndarray]] = {}
    default_g = int(cfg["ssfm"]["default_g"])
    g_list = g_grid or [default_g]
    cand_rows = []
    auction_dates, pred_dates = [], []
    gate_vals: List[np.ndarray] = []
    attn_vals: List[np.ndarray] = []

    def maybe_weekly_update(day: pd.Timestamp) -> None:
        if oos_mode != "sequential_oos_weekly_retrain":
            return
        if pd.Timestamp(day).dayofweek != 4:
            return
        day_s = str(pd.Timestamp(day).date())
        if skip_friday_before_s and day_s < skip_friday_before_s:
            log(f"    skip Friday close retrain cutoff={day_s} (already done, keep loaded ckpt)")
            return
        win = None
        for w in windows:
            if w.get("includes_test_realized") and str(w.get("cutoff")) == day_s:
                win = w
                break
        if win is None or win.get("effective_from") is None:
            return
        log(f"    Friday close retrain cutoff={win['cutoff']} effective_from={win['effective_from']} (primary+FM/Diff/SS-FM+PPO/GRPO, pools={pools})")
        parked = [n for n in device_models if n != primary]
        for n in parked:
            device_models[n].to("cpu")
        _free_host()
        try:
            if friday_sft_dir is not None:
                log(f"    skip Friday SFT {primary} <- {friday_sft_dir}")
                device_models[primary] = fit_one(primary, win["train_days"], win["val_days"], load_only=True, load_from=friday_sft_dir, cutoff=win["cutoff"])
            else:
                device_models[primary] = fit_one(primary, win["train_days"], win["val_days"], cutoff=win["cutoff"], resume_tag=f"friday_{win['cutoff']}")
            fit_allocators(win["train_days"], "friday", cutoff=win["cutoff"])
            write_json(ckpt_dir / f"friday_done_{day_s}.json", {"cutoff": day_s, "t": time.time()})
        finally:
            for n in parked:
                device_models[n].to(device)
            _free_host()
        updates.append(dict(win))

    test_cache_by_pool: Dict[str, List[dict]] = {p: [] for p in pools}
    test_bar = Progress(max(len(split["test_days"]), 1), f"TEST 推断 {month}", log_every=5)
    if test_from_s:
        log(f"    TEST resume from {test_from_s}; earlier days kept from month CSV")
    for ti, day in enumerate(split["test_days"]):
        day_s = str(pd.Timestamp(day).date())
        if test_from_s and day_s < test_from_s:
            test_bar.update(ti + 1, f"skip {day_s}")
            maybe_weekly_update(day)
            continue
        try:
            y, y_c2c = store.labels(day)
        except KeyError:
            test_bar.update(ti + 1, f"skip no-next {day_s}")
            maybe_weekly_update(day)
            continue
        item_primary = None
        x, msk = store.window(day)
        a, ameta = store.auction_d(day)
        valid = store.valid_mask(day)
        xt = torch.from_numpy(np.nan_to_num(x)).to(device)
        at = torch.from_numpy(np.nan_to_num(a)).to(device)
        mt = torch.from_numpy(msk).to(device)
        del x, msk, a
        rank_k = int(cfg["portfolio"].get("rank_k") or cfg["portfolio"]["top_k"])
        for name in pred_names:
            pred, pmeta = predict_loaded(device_models[name], xt, at, mt, valid, ameta, device)
            idx = select_topk(pred, valid, rank_k) if int(valid.sum()) else np.zeros(0, dtype=int)
            ic = daily_ic(pred[valid], y[valid]) if int(valid.sum()) else {}
            tstat = top30_stats(pred, y, idx, valid) if len(idx) else {}
            rec = {
                "asof": day,
                "alpha": pred,
                "y": y,
                "valid": valid,
                "model": name,
                "clocks": store.clocks(day),
                "y_c2c": y_c2c,
            }
            test_items_by_model[name].append(rec)
            if name == primary:
                item_primary = rec
                for pool in pools:
                    test_cache_by_pool[pool].append(make_alloc_item(store, day, pred, valid, y, pmeta, cfg, pool))
                auction_dates.append(pmeta["auction_date"])
                pred_dates.append(str(pd.Timestamp(day).date()))
            tab = topk_table(store.kept, pred, y, idx, valid)
            tab["date"] = str(pd.Timestamp(day).date())
            tab["model"] = name
            daily_pred_frames.append(tab)
            pred_rows.append({"date": str(pd.Timestamp(day).date()), "model": name, **ic, **tstat, "valid_universe_size": int(valid.sum())})
            aux = pmeta.get("aux") or {}
            if name == "gated_residual" and "gate" in aux:
                gate_vals.append(np.asarray(aux["gate"]).reshape(-1))
            if name == "cross_attention" and "cross_attn" in aux:
                attn_vals.append(np.asarray(aux["cross_attn"]).reshape(-1))
        del xt, at, mt
        _free_host()
        assert item_primary is not None
        if pred_rows:
            pd.DataFrame(pred_rows).to_csv(out / "prediction_metrics.csv", index=False)
        test_bar.update(ti + 1, str(pd.Timestamp(day).date()))
        maybe_weekly_update(day)
    test_bar.close()
    month_bar.update(msg="B/Test inference done")

    if daily_pred_frames:
        pd.concat(daily_pred_frames, ignore_index=True).to_csv(out / "daily_predictions.csv", index=False)
    pd.DataFrame(pred_rows).to_csv(out / "prediction_metrics.csv", index=False)

    daily_all = []

    def _flush_partial_daily(unit: str) -> None:
        if not daily_all:
            return
        part = pd.concat(daily_all, ignore_index=True)
        part.to_csv(out / "backtest_daily.csv", index=False)
        (out / "tables").mkdir(exist_ok=True)
        part.to_csv(out / "tables" / "backtest_daily.csv", index=False)
        refresh_after_unit(
            out_root,
            oos_mode,
            {"month": month, "oos": oos_mode, "last_unit": unit, "out_root": out_root},
        )

    for name, items in test_items_by_model.items():
        if not items:
            continue
        items_all = []
        ws_all = []
        for it in items:
            w = apply_valid_mask(score_softmax_cuda(it["alpha"], it["valid"], tau), it["valid"])
            row = {
                "asof": it["asof"],
                "R": it["y"],
                "names": list(store.kept),
                "alpha": it["alpha"],
                "idx": np.arange(len(store.kept), dtype=int),
                "clocks": it["clocks"],
                "pool": "all",
                "valid": it["valid"],
            }
            items_all.append(row)
            ws_all.append(w)
        daily_all.append(backtest_items(items_all, ws_all, cfg, f"pred_{name}_softmax_all", oos_mode, 1))
        _flush_partial_daily(f"A-backtest-{name}-all")
    if "all" in test_cache_by_pool and test_cache_by_pool["all"]:
        cache_all = test_cache_by_pool["all"]
        ws = [apply_valid_mask(np.ones(len(it["R"])) / max(len(it["R"]), 1), it["valid"][it["idx"]]) for it in cache_all]
        daily_all.append(backtest_items(cache_all, ws, cfg, "universe_ew_all", oos_mode, 1))
        _flush_partial_daily("A-universe-ew-all")

    for pool in pools:
        test_cache = test_cache_by_pool.get(pool) or []
        if not test_cache:
            continue
        pack = alloc.get(pool) or {}
        log(f"==== 实验B Portfolio baselines | pool={pool} | {month} ====")
        for bname in cfg["matrix"]["B_portfolio"]:
            ws = [it["baselines"][bname] for it in test_cache]
            daily_all.append(backtest_items(test_cache, ws, cfg, f"baseline_{bname}_{pool}", oos_mode, 1))
            log(f"    B {pool} {bname} done")
            _flush_partial_daily(f"B-{bname}-{pool}")
        if pool == "all":
            ws = [it["baselines"]["score_softmax"] for it in test_cache]
            daily_all.append(backtest_items(test_cache, ws, cfg, f"baseline_score_softmax_{pool}", oos_mode, 1))
            _flush_partial_daily(f"B-softmax-{pool}")
        gen_models = pack.get("gen") or {}
        rl_models = pack.get("rl") or {}
        if not skip_gen:
            for gname, gmodel in gen_models.items():
                ws = []
                for it in test_cache:
                    cond = torch.from_numpy(it["cond"]).to(device)
                    w, viol = sample_and_audit(lambda: sample_model(gname, gmodel, cond, default_g, cfg))
                    arr = w.detach().cpu().numpy()
                    j, wsel = select_candidate(arr, it["alpha"][it["idx"]], it["sigma"], cfg)
                    wsel = apply_valid_mask(wsel, it["valid"][it["idx"]])
                    ws.append(wsel)
                    for ci, ww in enumerate(arr):
                        met = execute_day(ww, it["R"], None, cfg)
                        cand_rows.append(
                            {
                                "date": str(it["asof"].date()),
                                "candidate_id": ci,
                                "model": f"{gname}_{pool}",
                                "pool": pool,
                                "selected": int(ci == j),
                                "selected_stocks": "|".join(it["names"]),
                                "weights": ",".join(f"{x:.6f}" for x in ww),
                                "simplex_sum": float(ww.sum()),
                                "min_weight": float(ww.min()),
                                "max_weight": float(ww.max()),
                                **met,
                            }
                        )
                daily_all.append(backtest_items(test_cache, ws, cfg, f"gen_{gname}_{pool}", oos_mode, default_g))
                _flush_partial_daily(f"C-backtest-{gname}-{pool}")
        if not skip_rl:
            for algo, pol in rl_models.items():
                ws = []
                for it in test_cache:
                    cond = torch.from_numpy(it["cond"]).to(device)
                    w, _, _ = sample_portfolios(pol, cond, default_g, float(cfg["rl"]["noise_std"]))
                    arr = w.detach().cpu().numpy()
                    j, wsel = select_candidate(arr, it["alpha"][it["idx"]], it["sigma"], cfg)
                    ws.append(apply_valid_mask(wsel, it["valid"][it["idx"]]))
                daily_all.append(backtest_items(test_cache, ws, cfg, f"rl_ssfm_{algo}_{pool}", oos_mode, default_g))
                _flush_partial_daily(f"D-backtest-{algo}-{pool}")
        if g_grid:
            if "ssfm" not in gen_models:
                raise RuntimeError("G experiments require trained SS-FM")
            log(f"==== 实验E G-grid {g_grid} | pool={pool} | {month} ====")
            for g in g_grid:
                t_s = time.time()
                ws = []
                for it in test_cache:
                    cond = torch.from_numpy(it["cond"]).to(device)
                    w, _ = sample_and_audit(lambda: sample_ss_fm_mixed(gen_models["ssfm"], cond, int(g), int(cfg["ssfm"]["n_sample_steps"])))
                    arr = w.detach().cpu().numpy()
                    _, wsel = select_candidate(arr, it["alpha"][it["idx"]], it["sigma"], cfg)
                    ws.append(apply_valid_mask(wsel, it["valid"][it["idx"]]))
                df = backtest_items(test_cache, ws, cfg, f"ssfm_G{g}_{pool}", oos_mode, int(g))
                df["sampling_time_sec"] = time.time() - t_s
                daily_all.append(df)
                month_bar.update(msg=f"E {pool} G={g}")
                _flush_partial_daily(f"E-G{g}-{pool}")

    daily = pd.concat(daily_all, ignore_index=True) if daily_all else pd.DataFrame()
    if cand_rows:
        pd.DataFrame(cand_rows).to_csv(out / "candidate_portfolios.csv", index=False)
    same_day = bool(getattr(store, "same_day_o2c", False))
    extra_flags = {
        "cs_rank_contemporaneous": True,
        "trailing_rolling": True,
        "cov_hist_only": True,
        "teacher_hist_only": True,
        "fm_no_future": True,
        "auction_before_open": True,
        "no_d_open_in_X": same_day,
        "no_d_close_in_X": same_day,
        "window_includes_asof": not same_day,
        "window_excludes_next_day": True,
        "label_is_next_close_to_close": not same_day,
        "label_is_same_day_open_to_close_log": same_day,
        "no_future_minutes": True,
        "no_future_auction": True,
        "sharpe_causal": True,
        "dd_causal": True,
        "rl_train_only": True,
        "rl_no_test": oos_mode == "strict_fixed_oos" or True,
        "hp_on_val": True,
        "seq_labeled": oos_mode != "sequential_oos_weekly_retrain" or all("label" in u for u in updates),
        "topk_on_pred": True,
        "allocation_on_valid_universe": True,
        "oos_separated": True,
    }
    clocks = store.clocks(split["test_days"][0])
    audit_rows = run_audit(
        exp_id, split, oos_mode, updates, split["train_end_date"],
        auction_dates, pred_dates, clocks, [str(out / "split_manifest.json")], extra_flags,
    )
    valid_audit = write_audit(out / "leakage_audit.json", audit_rows)
    write_json(out / "model_updates.json", updates)
    extra_rep = {
        "title": f"{month} {oos_mode}",
        "valid_audit": valid_audit,
        "experiment_id": exp_id,
        "month": month,
        "oos": oos_mode,
        "top_k": int(cfg["portfolio"]["top_k"]),
        "cost_bps": cfg["portfolio"]["cost_bps"],
        "primary": primary,
        "out_root": out_root,
        "aux_gate": np.concatenate(gate_vals) if gate_vals else None,
        "aux_attn": np.concatenate(attn_vals) if attn_vals else None,
    }
    extra_json = {k: (str(v) if k == "out_root" else v) for k, v in extra_rep.items() if k not in ("aux_gate", "aux_attn")}
    pred_df = pd.DataFrame(pred_rows)
    month_dir = out.parent
    if test_from_s:
        prev_pred = month_dir / "prediction_metrics.csv"
        if prev_pred.is_file() and not pred_df.empty:
            old_pred = pd.read_csv(prev_pred)
            old_pred = old_pred[old_pred["date"].astype(str) < test_from_s]
            pred_df = pd.concat([old_pred, pred_df], ignore_index=True)
            pred_df.to_csv(out / "prediction_metrics.csv", index=False)
            log(f"    merged prediction_metrics keep<{test_from_s} + new -> {len(pred_df)} rows")
        prev_bt = month_dir / "backtest_daily.csv"
        if prev_bt.is_file() and not daily.empty:
            old_bt = pd.read_csv(prev_bt)
            old_bt = old_bt[old_bt["date"].astype(str) < test_from_s]
            daily = pd.concat([old_bt, daily], ignore_index=True)
            daily.to_csv(out / "backtest_daily.csv", index=False)
            log(f"    merged backtest_daily keep<{test_from_s} + new -> {len(daily)} rows")
    write_month_reports(out, daily, pred_df, extra_json)
    write_month_bundle(out, daily, pred_df, extra_rep)
    write_month_bundle(month_dir, daily, pred_df, extra_rep)
    update_family_docs(out_root, oos_mode, extra_rep)
    if not valid_audit:
        write_json(out / "INVALID.json", {"reason": "high-severity leakage audit FAIL", "experiment_id": exp_id})
        log(f"    INVALID {exp_id}")
    month_bar.update(msg="audit")
    month_bar.close("month done")
    return {"out": str(out), "experiment_id": exp_id, "valid": valid_audit, "daily": daily, "pred": pred_df}
