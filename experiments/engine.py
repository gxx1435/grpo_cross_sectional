"""Strict fixed-OOS monthly engine using one minute-attention alpha model."""

from __future__ import annotations

import gc
import json
import subprocess
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from backtest.drawdown import smooth_dd_to_date
from backtest.engine import run_path
from backtest.execution import execute_day_by_id
from backtest.metrics import causal_sharpe, summarize_nav
from data.normalizers import ChannelZScore
from data.splits import split_for_test_month
from data.store import ResearchStore
from evaluation.leakage_audit import run_strict_audit, write_audit
from evaluation.prediction_metrics import daily_ic, ranking_classification_metrics, top30_stats
from evaluation.sp500_reports import write_month_report
from flow_matching.diffusion import WeightDiffusion, diffusion_loss, sample_diffusion
from flow_matching.gaussian_policy import GaussianPolicy
from flow_matching.mlp_policy import MLPPolicy
from flow_matching.samplers import sample_and_audit
from flow_matching.ssfm import SimplexSpaceFlow, sample_ss_fm_mixed, ss_fm_loss
from flow_matching.standard_fm import StandardFlowMatching, sample_std_fm, std_fm_loss
from models.alpha_predictor import STRATEGY_NAME, build_predictor
from portfolio.baselines import all_baselines
from portfolio.constraints import apply_valid_mask
from portfolio.risk_model import hist_mu_sigma
from portfolio.teacher_portfolios import TEACHER_IDS, teachers
from portfolio.topk import select_topk
from rl.grpo import grpo_update
from rl.ppo import ValueHead, WeightPolicy, ppo_update, sample_portfolios
from rl.reward import composite_reward, fit_reward_scalers
from utils.git_info import environment_info
from utils.gpu_monitor import GpuMonitor
from utils.io import dump_yaml_copy, experiment_id, file_sha256
from utils.logging import log, write_json
from utils.runtime import count_params
from utils.seed import set_seed


def device_of() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def amp_context(device: torch.device):
    return torch.amp.autocast(device_type="cuda", enabled=device.type == "cuda")


def _free() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _append_union_csv(path: Path, row: Dict[str, Any]) -> None:
    """Append while preserving a valid union schema across heterogeneous phases."""
    path.parent.mkdir(parents=True, exist_ok=True)
    old = pd.read_csv(path) if path.is_file() and path.stat().st_size else pd.DataFrame()
    pd.concat([old, pd.DataFrame([row])], ignore_index=True, sort=False).to_csv(path, index=False)


def _gpu_row(phase: str) -> Dict[str, Any]:
    row: Dict[str, Any] = {"timestamp": pd.Timestamp.now(tz="UTC").isoformat(), "phase": phase}
    if torch.cuda.is_available():
        row.update(
            {
                "gpu_model": torch.cuda.get_device_name(0),
                "gpu_total_memory": int(torch.cuda.get_device_properties(0).total_memory),
                "peak_allocated_memory": int(torch.cuda.max_memory_allocated()),
                "peak_reserved_memory": int(torch.cuda.max_memory_reserved()),
            }
        )
        try:
            text = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits", "-i", "0"],
                text=True,
                timeout=5,
            ).strip().split(",")
            row["gpu_utilization"] = float(text[0])
            row["gpu_memory_used_mib"] = float(text[1])
        except Exception:
            row["gpu_utilization"] = np.nan
    return row


def _checkpoint_payload(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[Any],
    normalizer: Optional[ChannelZScore],
    metadata: Dict[str, Any],
    extra_state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "optimizer_state": None if optimizer is None else optimizer.state_dict(),
        "scheduler_state": None if scheduler is None else scheduler.state_dict(),
        "normalizer_state": None if normalizer is None else normalizer.state(),
        "metadata": _jsonable(metadata),
        "extra_state": extra_state or {},
    }


def _save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[Any],
    normalizer: Optional[ChannelZScore],
    metadata: Dict[str, Any],
    extra_state: Optional[Dict[str, Any]] = None,
) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(_checkpoint_payload(model, optimizer, scheduler, normalizer, metadata, extra_state), path)
    return file_sha256(path)


def _build_optimizer(parameters: Iterable[torch.nn.Parameter], section: Dict[str, Any]) -> torch.optim.Optimizer:
    name = str(section["optimizer"]).lower()
    kwargs = {"lr": float(section["lr"]), "weight_decay": float(section.get("weight_decay", 0.0))}
    if name == "adam":
        return torch.optim.Adam(parameters, **kwargs)
    if name == "adamw":
        return torch.optim.AdamW(parameters, **kwargs)
    raise RuntimeError(f"unsupported optimizer {section['optimizer']}")


def fit_feature_scaler(store: ResearchStore, days: Sequence[pd.Timestamp], cfg: dict) -> ChannelZScore:
    scaler_cfg = cfg["prediction"]["feature_scaler"]
    scaler = ChannelZScore(float(scaler_cfg["epsilon"]), float(scaler_cfg["zmax"]))
    stride = max(1, int(cfg["prediction"]["feat_std_stride"]))
    bar_stride = max(1, int(cfg["prediction"]["feat_std_bar_stride"]))
    used = 0
    for pos, day in enumerate(days):
        if pos % stride:
            continue
        valid = store.valid_mask(day)
        idx = np.flatnonzero(valid)
        if len(idx) < int(cfg["portfolio"]["top_k"]):
            continue
        x, minute_mask = store.window(day, idx)
        scaler.update_x(x[:, ::bar_stride], minute_mask[:, ::bar_stride])
        used += 1
        del x, minute_mask
    if scaler.mean_x is None:
        raise RuntimeError("train-only feature scaler fit produced no statistics")
    scaler.fit_range = {"start": str(pd.Timestamp(days[0]).date()), "end": str(pd.Timestamp(days[-1]).date())}
    log(f"    feature scaler fit on {used} train dates, F={len(scaler.mean_x)}")
    return scaler


def _day_pack(store: ResearchStore, day: pd.Timestamp, scaler: ChannelZScore) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    valid = store.valid_mask(day)
    idx = np.flatnonzero(valid)
    if len(idx) < int(store.cfg["portfolio"]["top_k"]):
        return None
    y, _ = store.labels(day)
    finite = np.isfinite(y[idx])
    idx = idx[finite]
    if len(idx) < 8:
        return None
    x, minute_mask = store.window(day, idx)
    x = scaler.transform_x(x, minute_mask)
    return x, minute_mask, y[idx].astype(np.float32), idx


@torch.no_grad()
def predict_day(
    model: nn.Module,
    store: ResearchStore,
    day: pd.Timestamp,
    device: torch.device,
    scaler: ChannelZScore,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any], float]:
    valid = store.valid_mask(day)
    idx = np.flatnonzero(valid)
    full = np.full(store.N, np.nan, dtype=np.float32)
    if len(idx) < 8:
        return full, valid, store.prediction_meta(day), 0.0
    x, minute_mask = store.window(day, idx)
    x = scaler.transform_x(x, minute_mask)
    xt = torch.from_numpy(x).to(device, non_blocking=True)
    mt = torch.from_numpy(minute_mask).to(device, non_blocking=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    model.eval()
    with amp_context(device):
        output = model(xt, valid=None, mask=mt)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    full[idx] = output.detach().float().cpu().numpy()
    aux = {k: v.detach().float().cpu().numpy() for k, v in getattr(model, "last_aux", {}).items()}
    meta = {**store.prediction_meta(day), "aux": aux}
    del x, minute_mask, xt, mt, output
    return full, valid, meta, elapsed


@torch.no_grad()
def evaluate_alpha(model: nn.Module, store: ResearchStore, days: Sequence[pd.Timestamp], device: torch.device, scaler: ChannelZScore, top_k: int) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    for day in days:
        try:
            y, _ = store.labels(day)
            pred, valid, _, inference = predict_day(model, store, day, device, scaler)
            if int(valid.sum()) < max(8, top_k):
                continue
            idx = select_topk(pred, valid, top_k)
            metrics = {**daily_ic(pred[valid], y[valid]), **ranking_classification_metrics(pred[valid], y[valid], top_k), **top30_stats(pred, y, idx, valid)}
            rows.append({"date": str(pd.Timestamp(day).date()), **metrics, "valid_universe_size": int(valid.sum()), "inference_time_sec": inference})
        except Exception as exc:
            log(f"    validation skip {pd.Timestamp(day).date()}: {exc}")
    keys = ["IC", "RankIC", "MSE", "MAE", "top30_mean_true", "top30_excess", "top30_hit_rate", "inference_time_sec", "valid_universe_size"]
    summary = {k: float(pd.DataFrame(rows)[k].mean()) if rows and k in rows[0] else float("nan") for k in keys}
    summary["n"] = len(rows)
    return summary, rows


def train_alpha(
    store: ResearchStore,
    train_days: Sequence[pd.Timestamp],
    val_days: Sequence[pd.Timestamp],
    cfg: dict,
    device: torch.device,
    out: Path,
    metadata: Dict[str, Any],
    gpu_monitor: Optional[GpuMonitor] = None,
) -> Tuple[nn.Module, ChannelZScore, Dict[str, Any]]:
    p = cfg["prediction"]
    model = build_predictor(STRATEGY_NAME, cfg, store.feat_dim).to(device)
    scaler = fit_feature_scaler(store, train_days, cfg)
    scaler.save(out / "models" / "alpha_scaler.npz")
    optimizer = _build_optimizer(model.parameters(), p)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, int(p["epochs"])),
        eta_min=float(p["scheduler"]["min_lr"]),
    )
    amp_scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    loss_cfg = p["prediction_loss"]
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    if str(loss_cfg["type"]).lower() == "huber":
        criterion = nn.HuberLoss(delta=float(loss_cfg["delta"]))
    elif str(loss_cfg["type"]).lower() == "mse":
        criterion = nn.MSELoss()
    else:
        raise RuntimeError(f"unsupported prediction loss {loss_cfg}")
    best_score = float("inf")
    best_epoch = 0
    best_sha = ""
    bad = 0
    train_rows: List[Dict[str, Any]] = []
    val_rows: List[Dict[str, Any]] = []
    oom_retries = 0
    started = time.time()
    for epoch in range(1, int(p["epochs"]) + 1):
        model.train()
        losses: List[float] = []
        epoch_start = time.time()
        for position, day in enumerate(train_days, start=1):
            pack = _day_pack(store, day, scaler)
            if pack is None:
                continue
            x, minute_mask, y, _ = pack
            retry = 0
            while True:
                xt = mt = yt = prediction = loss = None
                optimizer.zero_grad(set_to_none=True)
                try:
                    xt = torch.from_numpy(x).to(device, non_blocking=True)
                    mt = torch.from_numpy(minute_mask).to(device, non_blocking=True)
                    yt = torch.from_numpy(y).to(device, non_blocking=True)
                    with amp_context(device):
                        prediction = model(xt, valid=None, mask=mt)
                        loss = criterion(prediction, yt)
                    amp_scaler.scale(loss).backward()
                    amp_scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), float(p["grad_clip"]))
                    amp_scaler.step(optimizer)
                    amp_scaler.update()
                    losses.append(float(loss.detach().cpu()))
                    break
                except torch.OutOfMemoryError:
                    retry += 1
                    oom_retries += 1
                    if gpu_monitor is not None:
                        gpu_monitor.record_oom()
                    optimizer.zero_grad(set_to_none=True)
                    xt = mt = yt = prediction = loss = None
                    _free()
                    old = model.stock_chunk
                    model.stock_chunk = max(8, old // 2)
                    model.use_ckpt = True
                    _append_union_csv(
                        out / "runtime_logs.csv",
                        {
                            "phase": "alpha_oom_retry",
                            "old_stock_chunk": old,
                            "new_stock_chunk": model.stock_chunk,
                            "retry_count": retry,
                            "gradient_checkpointing": True,
                            "date": str(day.date()),
                        },
                    )
                    if retry > int(cfg["gpu"]["oom_max_retries"]) or (old == 8 and retry > 1):
                        raise
                finally:
                    del xt, mt, yt, prediction, loss
            del x, minute_mask, y
            if position == 1 or position % 10 == 0 or position == len(train_days):
                log(f"    alpha epoch {epoch}/{p['epochs']} date {position}/{len(train_days)} loss={np.mean(losses):.6f}")
        scheduler.step()
        summary, detail = evaluate_alpha(model, store, val_days, device, scaler, int(cfg["portfolio"]["top_k"]))
        score = float(summary["MSE"])
        row = {
            "phase": "alpha",
            "epoch": epoch,
            "train_loss": float(np.mean(losses)) if losses else np.nan,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "epoch_time_sec": time.time() - epoch_start,
            **{f"validation_{k}": v for k, v in summary.items()},
        }
        train_rows.append(row)
        for value in detail:
            val_rows.append({"epoch": epoch, **value})
        _save_checkpoint(
            out / "models" / "alpha_last.pt",
            model,
            optimizer,
            scheduler,
            scaler,
            {**metadata, "checkpoint_role": "last", "epoch": epoch, "validation": summary, "amp_scaler_state": amp_scaler.state_dict()},
        )
        if np.isfinite(score) and score < best_score:
            best_score, best_epoch, bad = score, epoch, 0
            best_sha = _save_checkpoint(
                out / "models" / "alpha_best_validation.pt",
                model,
                optimizer,
                scheduler,
                scaler,
                {**metadata, "checkpoint_role": "best_validation", "epoch": epoch, "selection_rule": "minimum validation MSE", "validation": summary, "amp_scaler_state": amp_scaler.state_dict()},
            )
        else:
            bad += 1
        if bad >= int(p["early_stopping_patience"]):
            break
    if not (out / "models" / "alpha_best_validation.pt").is_file():
        raise RuntimeError("alpha training produced no finite validation checkpoint")
    blob = torch.load(out / "models" / "alpha_best_validation.pt", map_location=device, weights_only=False)
    model.load_state_dict(blob["model_state"])
    pd.DataFrame(train_rows).to_csv(out / "training_logs.csv", index=False)
    pd.DataFrame(val_rows).to_csv(out / "alpha_validation_logs.csv", index=False)
    result = {
        "best_epoch": best_epoch,
        "best_validation_mse": best_score,
        "best_checkpoint_id": best_sha,
        "checkpoint_selection_rule": "minimum validation MSE",
        "parameter_count": count_params(model),
        "train_days_sampled": len(train_days),
        "validation_days": len(val_days),
        "wall_time_sec": time.time() - started,
        "initial_stock_chunk": int(p["stock_chunk"]),
        "final_stock_chunk": int(model.stock_chunk),
        "batch_asofs": int(p["batch_asofs"]),
        "gradient_accumulation": 1,
        "gradient_checkpointing": bool(model.use_ckpt),
        "oom_retries": int(oom_retries),
    }
    write_json(out / "alpha_training_metadata.json", result)
    return model, scaler, result


def build_cond(alpha: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    def z(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        return (values - values.mean()) / (values.std() + 1e-6)

    upper = sigma[np.triu_indices(sigma.shape[0])].astype(np.float32)
    upper = upper / (upper.std() + 1e-6)
    return np.concatenate([z(alpha), z(mu), z(np.sqrt(np.clip(np.diag(sigma), 1e-12, None))), upper]).astype(np.float32)


def cond_dim(k: int) -> int:
    return 3 * int(k) + int(k) * (int(k) + 1) // 2


def make_state(store: ResearchStore, day: pd.Timestamp, pred: np.ndarray, valid: np.ndarray, y: np.ndarray, cfg: dict) -> Dict[str, Any]:
    top_k = int(cfg["portfolio"]["top_k"])
    idx = select_topk(pred, valid, top_k)
    day_i = store.day_index(day)
    execution_mask = np.isfinite(store.open[day_i, idx]) & (store.open[day_i, idx] > 0)
    if not execution_mask.any():
        raise RuntimeError("causal Top-K has no security executable at the target open")
    target_y = y[idx].astype(np.float32)
    missing_close = execution_mask & ~np.isfinite(target_y)
    if missing_close.any():
        unavailable = [store.kept[j] for j in idx[missing_close]]
        raise RuntimeError(f"executed causal Top-K contains unavailable close labels: {unavailable}")
    # Open availability is observed only at the execution timestamp.  It never
    # changes Alpha, attention, Top-K, candidate scores, or policy selection;
    # unavailable orders receive zero weight and remaining orders renormalize.
    target_y = np.nan_to_num(target_y, nan=0.0)
    names = [store.kept[j] for j in idx]
    history = store.daily_returns_to(day)
    risk_cfg = cfg["portfolio"]["risk_model"]
    mu, sigma = hist_mu_sigma(history, day, names, int(cfg["portfolio"]["hist_risk_lookback_days"]), float(risk_cfg["covariance_epsilon"]))
    teacher = teachers(mu, sigma, float(cfg["portfolio"]["max_weight"]), float(cfg["ssfm"]["risk_aversion"]), risk_cfg)
    baseline = all_baselines(mu, sigma, pred[idx], float(cfg["portfolio"]["max_weight"]), risk_cfg)
    teacher = {name: apply_valid_mask(weight, execution_mask) for name, weight in teacher.items()}
    baseline = {name: apply_valid_mask(weight, execution_mask) for name, weight in baseline.items()}
    return {
        "asof": pd.Timestamp(day),
        "idx": idx,
        "names": names,
        "alpha": pred[idx].astype(np.float32),
        "y": target_y,
        "execution_mask": execution_mask,
        "execution_exclusion_count": int((~execution_mask).sum()),
        "mu": mu,
        "sigma": sigma,
        "teachers": teacher,
        "baselines": baseline,
        "cond": build_cond(pred[idx], mu, sigma),
        "clocks": store.clocks(day),
    }


def build_state_cache(store: ResearchStore, model: nn.Module, scaler: ChannelZScore, days: Sequence[pd.Timestamp], cfg: dict, device: torch.device, label: str) -> List[Dict[str, Any]]:
    cache: List[Dict[str, Any]] = []
    for position, day in enumerate(days, start=1):
        try:
            y, _ = store.labels(day)
            pred, valid, _, _ = predict_day(model, store, day, device, scaler)
            if int(valid.sum()) >= int(cfg["portfolio"]["top_k"]):
                cache.append(make_state(store, day, pred, valid, y, cfg))
        except Exception as exc:
            log(f"    {label} state skip {pd.Timestamp(day).date()}: {exc}")
        if position == 1 or position % 10 == 0 or position == len(days):
            log(f"    {label} state {position}/{len(days)} kept={len(cache)}")
    return cache


def _batch(cache: Sequence[Dict[str, Any]], batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    picks = np.random.randint(0, len(cache), size=max(1, int(batch_size)))
    teacher_names = list(TEACHER_IDS)
    chosen = [teacher_names[int(np.random.randint(0, len(teacher_names)))] for _ in picks]
    weights = torch.stack([torch.from_numpy(cache[int(i)]["teachers"][name].astype(np.float32)) for i, name in zip(picks, chosen)]).to(device)
    cond = torch.stack([torch.from_numpy(cache[int(i)]["cond"]) for i in picks]).to(device)
    teacher_id = torch.tensor([TEACHER_IDS[name] for name in chosen], device=device)
    return weights, cond, teacher_id


@torch.no_grad()
def _validation_gen_loss(model: nn.Module, cache: Sequence[Dict[str, Any]], loss_name: str, cfg: dict, device: torch.device) -> float:
    if not cache:
        return float("nan")
    torch.manual_seed(int(cfg["experiment"]["seed"]) + 911)
    losses: List[float] = []
    for item in cache[: min(16, len(cache))]:
        cond = torch.from_numpy(item["cond"]).unsqueeze(0).to(device)
        if loss_name == "ssfm":
            for teacher_name, teacher_id in TEACHER_IDS.items():
                weight = torch.from_numpy(item["teachers"][teacher_name].astype(np.float32)).unsqueeze(0).to(device)
                losses.append(float(ss_fm_loss(model, weight, cond, torch.tensor([teacher_id], device=device), float(cfg["ssfm"]["clr_eps"]))))
        else:
            weight = torch.from_numpy(item["teachers"]["risk_parity"].astype(np.float32)).unsqueeze(0).to(device)
            fn = {"standard_fm": std_fm_loss, "diffusion": diffusion_loss}[loss_name]
            losses.append(float(fn(model, weight, cond)))
    return float(np.mean(losses))


def _train_flow_model(
    name: str,
    model: nn.Module,
    loss_fn: Callable[..., torch.Tensor],
    train_cache: Sequence[Dict[str, Any]],
    val_cache: Sequence[Dict[str, Any]],
    cfg: dict,
    device: torch.device,
    out: Path,
    metadata: Dict[str, Any],
) -> nn.Module:
    section = cfg["ssfm"] if name == "ssfm" else cfg[name]
    steps = int(section["steps"])
    optimizer = _build_optimizer(model.parameters(), section)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(steps, 1), eta_min=float(section["scheduler"]["min_lr"]))
    best = float("inf")
    interval = int(cfg["ssfm"]["validation_interval"])
    for step in range(1, steps + 1):
        weights, cond, teacher_id = _batch(train_cache, int(cfg["ssfm"]["batch_size"]), device)
        loss = loss_fn(model, weights, cond, teacher_id) if name == "ssfm" else loss_fn(model, weights, cond)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), float(section["grad_clip"]))
        optimizer.step()
        scheduler.step()
        if step == 1 or step % 20 == 0 or step == steps:
            log(f"    {name} step {step}/{steps} loss={float(loss.detach()):.6f}")
        if step % interval == 0 or step == steps:
            val_loss = _validation_gen_loss(model, val_cache, name, cfg, device)
            _append_union_csv(out / "training_logs.csv", {"phase": name, "step": step, "train_loss": float(loss.detach()), "validation_loss": val_loss})
            _save_checkpoint(out / "models" / f"{name}_last.pt", model, optimizer, scheduler, None, {**metadata, "checkpoint_role": "last", "step": step, "validation_loss": val_loss})
            if np.isfinite(val_loss) and val_loss < best:
                best = val_loss
                _save_checkpoint(out / "models" / f"{name}_best_validation.pt", model, optimizer, scheduler, None, {**metadata, "checkpoint_role": "best_validation", "step": step, "selection_rule": "minimum validation imitation loss", "validation_loss": val_loss})
    best_path = out / "models" / f"{name}_best_validation.pt"
    if best_path.is_file():
        model.load_state_dict(torch.load(best_path, map_location=device, weights_only=False)["model_state"])
    return model


def _train_direct_policy(
    name: str,
    model: nn.Module,
    train_cache: Sequence[Dict[str, Any]],
    val_cache: Sequence[Dict[str, Any]],
    cfg: dict,
    device: torch.device,
    out: Path,
    metadata: Dict[str, Any],
) -> nn.Module:
    section = cfg[f"{name}_policy"]
    steps = int(section["steps"])
    optimizer = _build_optimizer(model.parameters(), section)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(steps, 1), eta_min=float(section["scheduler"]["min_lr"]))
    best = float("inf")
    interval = int(cfg["ssfm"]["validation_interval"])
    for step in range(1, steps + 1):
        weights, cond, _ = _batch(train_cache, int(cfg["ssfm"]["batch_size"]), device)
        if name == "mlp":
            loss = F.mse_loss(model(cond), weights)
        elif name == "gaussian":
            loss = float(section["nll_weight"]) * model.nll(cond, weights)
        else:  # pragma: no cover - closed internal registry
            raise KeyError(name)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), float(section["grad_clip"]))
        optimizer.step()
        scheduler.step()
        if step % interval == 0 or step == steps:
            with torch.no_grad():
                val_losses = []
                for item in val_cache[: min(32, len(val_cache))]:
                    validation_cond = torch.from_numpy(item["cond"]).to(device)
                    target = torch.from_numpy(item["teachers"]["risk_parity"].astype(np.float32)).to(device)
                    val_losses.append(float(F.mse_loss(model(validation_cond).squeeze(0), target)))
            val_loss = float(np.mean(val_losses))
            _append_union_csv(out / "training_logs.csv", {"phase": name, "step": step, "train_loss": float(loss.detach()), "validation_loss": val_loss})
            _save_checkpoint(out / "models" / f"{name}_last.pt", model, optimizer, scheduler, None, {**metadata, "checkpoint_role": "last", "step": step, "validation_loss": val_loss})
            if val_loss < best:
                best = val_loss
                _save_checkpoint(out / "models" / f"{name}_best_validation.pt", model, optimizer, scheduler, None, {**metadata, "checkpoint_role": "best_validation", "step": step, "selection_rule": "minimum validation imitation MSE", "validation_loss": val_loss})
    model.load_state_dict(torch.load(out / "models" / f"{name}_best_validation.pt", map_location=device, weights_only=False)["model_state"])
    return model


def train_generative(train_cache: Sequence[Dict[str, Any]], val_cache: Sequence[Dict[str, Any]], cfg: dict, device: torch.device, out: Path, metadata: Dict[str, Any]) -> Dict[str, nn.Module]:
    k = int(cfg["portfolio"]["top_k"])
    cd = cond_dim(k)
    models: Dict[str, nn.Module] = {}
    ssfm = SimplexSpaceFlow(k, cd, int(cfg["ssfm"]["hidden"]), int(cfg["ssfm"]["n_teachers"]), int(cfg["ssfm"]["teacher_emb"])).to(device)
    models["ssfm"] = _train_flow_model("ssfm", ssfm, lambda m, w, c, tid: ss_fm_loss(m, w, c, tid, float(cfg["ssfm"]["clr_eps"])), train_cache, val_cache, cfg, device, out, metadata)
    standard = StandardFlowMatching(k, cd, int(cfg["standard_fm"]["hidden"])).to(device)
    models["standard_fm"] = _train_flow_model("standard_fm", standard, std_fm_loss, train_cache, val_cache, cfg, device, out, metadata)
    diffusion = WeightDiffusion(k, cd, int(cfg["diffusion"]["hidden"]), int(cfg["diffusion"]["n_diff_steps"])).to(device)
    models["diffusion"] = _train_flow_model("diffusion", diffusion, diffusion_loss, train_cache, val_cache, cfg, device, out, metadata)

    mlp = MLPPolicy(k, cd, int(cfg["mlp_policy"]["hidden"])).to(device)
    gaussian_cfg = cfg["gaussian_policy"]
    gaussian = GaussianPolicy(k, cd, int(gaussian_cfg["hidden"]), float(gaussian_cfg["min_std"]), float(gaussian_cfg["max_std"])).to(device)
    models["mlp"] = _train_direct_policy("mlp", mlp, train_cache, val_cache, cfg, device, out, metadata)
    models["gaussian"] = _train_direct_policy("gaussian", gaussian, train_cache, val_cache, cfg, device, out, metadata)
    return models


@torch.no_grad()
def sample_model(name: str, model: nn.Module, cond: torch.Tensor, count: int, cfg: dict) -> torch.Tensor:
    if name == "ssfm":
        return sample_ss_fm_mixed(model, cond, count, int(cfg["ssfm"]["n_sample_steps"]))
    if name == "standard_fm":
        return sample_std_fm(model, cond, count, int(cfg["standard_fm"]["n_sample_steps"]))
    if name == "diffusion":
        return sample_diffusion(model, cond, count)
    if name == "mlp":
        return model(cond).expand(count, -1)
    if name == "gaussian":
        return model.sample(cond, count)
    raise KeyError(name)


def decision_scores(weights: np.ndarray, alpha: np.ndarray, sigma: np.ndarray, cfg: dict) -> np.ndarray:
    risk = float(cfg["ssfm"]["risk_aversion"])
    return np.asarray([float(w @ alpha - 0.5 * risk * w @ sigma @ w) for w in weights])


def select_candidate(weights: np.ndarray, item: Dict[str, Any], cfg: dict) -> Tuple[int, np.ndarray, np.ndarray]:
    scores = decision_scores(weights, item["alpha"], item["sigma"], cfg)
    selected = int(np.argmax(scores))
    return selected, apply_valid_mask(weights[selected], item["execution_mask"]), scores


def _policy_validation_score(policy: WeightPolicy, cache: Sequence[Dict[str, Any]], device: torch.device, cfg: dict) -> float:
    returns: List[float] = []
    previous = None
    previous_names = None
    with torch.no_grad():
        for item in cache:
            cond = torch.from_numpy(item["cond"]).to(device)
            planned = policy(cond).squeeze(0).cpu().numpy()
            weight = apply_valid_mask(planned, item["execution_mask"])
            met = execute_day_by_id(weight, item["y"], item["names"], previous, previous_names, cfg)
            returns.append(met["net_return"])
            previous = weight
            previous_names = item["names"]
    return float(np.mean(returns)) if returns else float("nan")


def train_rl(
    ssfm: SimplexSpaceFlow,
    train_cache: Sequence[Dict[str, Any]],
    val_cache: Sequence[Dict[str, Any]],
    cfg: dict,
    device: torch.device,
    out: Path,
    metadata: Dict[str, Any],
    algorithm: str,
) -> WeightPolicy:
    k = int(cfg["portfolio"]["top_k"])
    cd = cond_dim(k)
    policy = WeightPolicy(cd, k, int(cfg["rl"]["hidden"])).to(device)
    critic = ValueHead(cd, int(cfg["rl"]["hidden"])).to(device) if algorithm == "ppo" else None
    distill_opt = _build_optimizer(policy.parameters(), cfg["rl"])
    for _ in range(int(cfg["rl"]["distill_steps"])):
        item = train_cache[int(np.random.randint(0, len(train_cache)))]
        cond = torch.from_numpy(item["cond"]).to(device)
        with torch.no_grad():
            target_w = sample_ss_fm_mixed(ssfm, cond, 8, int(cfg["ssfm"]["n_sample_steps"])).mean(0)
        target_logits = torch.log(target_w.clamp_min(1e-8))
        target_logits = target_logits - target_logits.mean()
        loss = F.mse_loss(policy.forward_logits(cond).squeeze(0), target_logits)
        distill_opt.zero_grad(set_to_none=True)
        loss.backward()
        distill_opt.step()

    params = list(policy.parameters()) + ([] if critic is None else list(critic.parameters()))
    optimizer = _build_optimizer(params, cfg["rl"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, int(cfg["rl"]["epochs"])), eta_min=float(cfg["rl"]["scheduler"]["min_lr"]))
    hist = {"return": [], "sharpe": [], "turnover": [], "smooth_mdd": []}
    prev = None
    prev_names = None
    net_history: List[float] = []
    for item in train_cache:
        weight = np.ones(k, dtype=np.float64) / k
        weight = apply_valid_mask(weight, item["execution_mask"])
        met = execute_day_by_id(weight, item["y"], item["names"], prev, prev_names, cfg)
        hist["return"].append(met["net_return"])
        hist["sharpe"].append(causal_sharpe(net_history, int(cfg["portfolio"]["sharpe_min_obs"]), 0.0))
        hist["turnover"].append(met["turnover"])
        hist["smooth_mdd"].append(smooth_dd_to_date(np.asarray(net_history), float(cfg["portfolio"]["smooth_dd_temperature"])))
        net_history.append(met["net_return"])
        prev = weight
        prev_names = item["names"]
    fit_range = {"start": str(train_cache[0]["asof"].date()), "end": str(train_cache[-1]["asof"].date())}
    reward_scalers = fit_reward_scalers({k: np.asarray(v) for k, v in hist.items()}, cfg, fit_range)
    best = -float("inf")
    best_path = out / "models" / f"{algorithm}_best_validation.pt"
    reward_rows: List[Dict[str, Any]] = []
    for epoch in range(1, int(cfg["rl"]["epochs"]) + 1):
        prev = None
        prev_names = None
        realized: List[float] = []
        for item in train_cache:
            cond = torch.from_numpy(item["cond"]).to(device)
            weights, sampled_logits, _ = sample_portfolios(policy, cond, int(cfg["rl"]["group_size"]), float(cfg["rl"]["noise_std"]))
            planned_candidates = weights.detach().cpu().numpy()
            candidates = np.vstack([apply_valid_mask(candidate, item["execution_mask"]) for candidate in planned_candidates])
            sharpe = causal_sharpe(realized, int(cfg["portfolio"]["sharpe_min_obs"]), 0.0)
            drawdown = smooth_dd_to_date(np.asarray(realized), float(cfg["portfolio"]["smooth_dd_temperature"]))
            rewards: List[float] = []
            reward_packs: List[Dict[str, Any]] = []
            for candidate in candidates:
                met = execute_day_by_id(candidate, item["y"], item["names"], prev, prev_names, cfg)
                pack = composite_reward({"return": met["net_return"], "sharpe": sharpe, "turnover": met["turnover"], "smooth_mdd": drawdown}, reward_scalers, cfg)
                rewards.append(float(pack["reward"]))
                reward_packs.append(pack)
            reward_tensor = torch.tensor(rewards, device=device, dtype=torch.float32)
            if algorithm == "ppo":
                update = ppo_update(policy, critic, cond, sampled_logits, reward_tensor, optimizer, cfg)
            else:
                update = grpo_update(policy, cond, sampled_logits, reward_tensor, optimizer, cfg)
            mean_weight = planned_candidates.mean(axis=0)
            mean_weight /= max(mean_weight.sum(), 1e-12)
            mean_weight = apply_valid_mask(mean_weight, item["execution_mask"])
            realized_met = execute_day_by_id(mean_weight, item["y"], item["names"], prev, prev_names, cfg)
            realized.append(realized_met["net_return"])
            prev = mean_weight
            prev_names = item["names"]
            update_fields = {f"update_{key}": value for key, value in update.items() if isinstance(value, (int, float, bool))}
            for candidate_id, (reward, pack) in enumerate(zip(rewards, reward_packs)):
                row = {
                    "phase": algorithm,
                    "epoch": epoch,
                    "date": str(item["asof"].date()),
                    "candidate_id": candidate_id,
                    "reward": reward,
                    "mean_group_reward": float(np.mean(rewards)),
                    **update_fields,
                }
                for component, values in pack["components"].items():
                    row[f"{component}_raw"] = values["raw"]
                    row[f"{component}_z"] = values["z"]
                    row[f"{component}_z_clipped"] = values["z_clipped"]
                row["reward_realization_timestamp"] = item["clocks"]["realization_timestamp"]
                reward_rows.append(row)
        scheduler.step()
        validation_score = _policy_validation_score(policy, val_cache, device, cfg)
        _append_union_csv(out / "training_logs.csv", {"phase": algorithm, "epoch": epoch, "validation_mean_net_return": validation_score})
        extra = {"critic_state": None if critic is None else {k: v.detach().cpu() for k, v in critic.state_dict().items()}, "reward_scalers": {k: v.state() for k, v in reward_scalers.items()}}
        _save_checkpoint(out / "models" / f"{algorithm}_last.pt", policy, optimizer, scheduler, None, {**metadata, "checkpoint_role": "last", "epoch": epoch, "validation_mean_net_return": validation_score, "rollout_range": fit_range}, extra)
        if np.isfinite(validation_score) and validation_score > best:
            best = validation_score
            _save_checkpoint(best_path, policy, optimizer, scheduler, None, {**metadata, "checkpoint_role": "best_validation", "epoch": epoch, "selection_rule": "maximum validation mean net return", "validation_mean_net_return": validation_score, "rollout_range": fit_range}, extra)
    if not best_path.is_file():
        raise RuntimeError(f"{algorithm} produced no validation checkpoint")
    policy.load_state_dict(torch.load(best_path, map_location=device, weights_only=False)["model_state"])
    existing = pd.read_csv(out / "reward_logs.csv") if (out / "reward_logs.csv").is_file() else pd.DataFrame()
    pd.concat([existing, pd.DataFrame(reward_rows)], ignore_index=True).to_csv(out / "reward_logs.csv", index=False)
    return policy


def select_g_on_validation(ssfm: nn.Module, cache: Sequence[Dict[str, Any]], cfg: dict, device: torch.device, values: Sequence[int], out: Path) -> int:
    rows: List[Dict[str, Any]] = []
    for count in values:
        net: List[float] = []
        previous = None
        previous_names = None
        violations: List[float] = []
        started = time.perf_counter()
        for item in cache:
            cond = torch.from_numpy(item["cond"]).to(device)
            sampled, violation = sample_and_audit(lambda: sample_ss_fm_mixed(ssfm, cond, int(count), int(cfg["ssfm"]["n_sample_steps"])))
            array = sampled.detach().cpu().numpy()
            _, selected, _ = select_candidate(array, item, cfg)
            met = execute_day_by_id(selected, item["y"], item["names"], previous, previous_names, cfg)
            net.append(met["net_return"])
            previous = selected
            previous_names = item["names"]
            violations.append(float(max(violation.get("sum_abs_err", 0.0), max(0.0, -float(violation.get("min_weight", 0.0))))))
        rows.append({"G": int(count), "validation_mean_net_return": float(np.mean(net)), "sampling_time_sec": time.perf_counter() - started, "constraint_violation": float(np.max(violations)) if violations else np.nan})
    table = pd.DataFrame(rows)
    table.to_csv(out / "g_validation_selection.csv", index=False)
    if table["validation_mean_net_return"].notna().any():
        return int(table.loc[table["validation_mean_net_return"].idxmax(), "G"])
    return int(cfg["ssfm"]["default_g"])


def _candidate_rows(
    item: Dict[str, Any],
    weights: np.ndarray,
    scores: np.ndarray,
    selected: int,
    strategy: str,
    checkpoint_id: str,
    cfg: dict,
    candidate_count: int,
    previous_weight: Optional[np.ndarray] = None,
    previous_names: Optional[Sequence[str]] = None,
    sampling_time_sec: float = float("nan"),
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    state_summary = json.dumps(
        {
            "alpha_mean": float(item["alpha"].mean()),
            "alpha_std": float(item["alpha"].std()),
            "risk_trace": float(np.trace(item["sigma"])),
            "execution_exclusion_count": int(item["execution_exclusion_count"]),
        },
        sort_keys=True,
    )
    for candidate_id, planned_weight in enumerate(weights):
        weight = apply_valid_mask(planned_weight, item["execution_mask"])
        realized = execute_day_by_id(weight, item["y"], item["names"], previous_weight, previous_names, cfg)
        rows.append(
            {
                "date": str(item["asof"].date()),
                "candidate_id": candidate_id,
                "selected_stocks": "|".join(item["names"]),
                "weights": ",".join(f"{x:.10g}" for x in weight),
                "planned_weights": ",".join(f"{x:.10g}" for x in planned_weight),
                "execution_exclusion_count": int(item["execution_exclusion_count"]),
                "state_summary": state_summary,
                "policy_log_probability": np.nan,
                "decision_score": float(scores[candidate_id]),
                "simplex_sum": float(weight.sum()),
                "min_weight": float(weight.min()),
                "max_weight": float(weight.max()),
                "constraint_violation": float(max(abs(weight.sum() - 1.0), max(0.0, -float(weight.min())))),
                "selection_flag": int(candidate_id == selected),
                "strategy_name": strategy,
                "candidate_count_G": int(candidate_count),
                "sampling_time_sec": float(sampling_time_sec),
                "model_checkpoint_id": checkpoint_id,
                "data_cutoff_timestamp": item["clocks"]["data_cutoff_timestamp"],
                "gross_return": realized["gross_return"],
                "net_return": realized["net_return"],
                "turnover": realized["turnover"],
                "transaction_cost": realized["transaction_cost"],
                "realization_timestamp": item["clocks"]["realization_timestamp"],
            }
        )
    return rows


def _candidate_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if frame.empty:
        return pd.DataFrame()
    for (strategy, date), group in frame.groupby(["strategy_name", "date"], observed=True):
        array = np.vstack(
            [np.fromstring(str(value), sep=",", dtype=np.float64) for value in group["weights"]]
        )
        centre = array.mean(axis=0, keepdims=True)
        rows.append(
            {
                "strategy_name": strategy,
                "date": date,
                "candidate_diversity": float(np.linalg.norm(array - centre, axis=1).mean()),
                "reward_diversity": float(group["net_return"].std(ddof=0)),
                "constraint_violation": float(group["constraint_violation"].max()),
                "sampling_time_sec": float(group["sampling_time_sec"].mean()),
                "candidate_count_G": int(group["candidate_count_G"].iloc[0]),
            }
        )
    return (
        pd.DataFrame(rows)
        .groupby("strategy_name", as_index=False, observed=True)
        .agg(
            candidate_diversity=("candidate_diversity", "mean"),
            reward_diversity=("reward_diversity", "mean"),
            constraint_violation=("constraint_violation", "max"),
            sampling_time_sec=("sampling_time_sec", "mean"),
            candidate_count_G=("candidate_count_G", "max"),
        )
    )


def backtest_items(items: Sequence[Dict[str, Any]], weights: Sequence[np.ndarray], cfg: dict, strategy: str, count: int) -> pd.DataFrame:
    return run_path(
        [item["asof"] for item in items],
        list(weights),
        [item["y"] for item in items],
        cfg,
        extra=[
            {
                "model": strategy,
                "strategy_name": strategy,
                "oos_mode": "strict_fixed_oos",
                "candidate_count_G": int(count),
                "selected_stocks": "|".join(item["names"]),
                **item["clocks"],
            }
            for item in items
        ],
        asset_ids=[item["names"] for item in items],
    )


def _weights_table(items: Sequence[Dict[str, Any]], weights: Sequence[np.ndarray], strategy: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for item, weight in zip(items, weights):
        for code, value in zip(item["names"], weight):
            rows.append({"date": str(item["asof"].date()), "strategy_name": strategy, "stock_code": code, "weight": float(value), "data_cutoff_timestamp": item["clocks"]["data_cutoff_timestamp"]})
    return pd.DataFrame(rows)


def run_month(cfg: dict, store: ResearchStore, test_month: pd.Timestamp, oos_mode: str, out_root: Path) -> Dict[str, Any]:
    if oos_mode != "strict_fixed_oos":
        raise RuntimeError("only strict_fixed_oos is enabled")
    set_seed(int(cfg["experiment"]["seed"]), bool(cfg["experiment"]["deterministic"]), bool(cfg["experiment"]["deterministic_warn_only"]))
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    device = device_of()
    split = split_for_test_month(store.days, test_month, int(cfg["walkforward"]["train_offset_months"]))
    month = str(split["test_month"])
    exp_id = experiment_id(cfg["experiment"]["name"], month, oos_mode)
    out = out_root / oos_mode / f"test_month={month}" / f"experiment_id={exp_id}"
    for sub in ("models", "plots", "failures"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    dump_yaml_copy(cfg, out / "config.yaml")
    env = environment_info(cfg["_root"])
    write_json(out / "environment.json", env)
    if int(split["train_month_count"]) != int(cfg["walkforward"]["train_months"]):
        raise RuntimeError(f"month {month} has {split['train_month_count']} observed train months, expected 12")
    if not split["val_days"] or not split["test_days"]:
        raise RuntimeError(f"empty validation or test split for {month}")
    split_json = {k: ([str(x.date()) for x in v] if k.endswith("days") else v) for k, v in split.items()}
    test_period = pd.Period(test_month, freq="M")
    observed_period = pd.Period(max(store.days), freq="M")
    observed_test_end = pd.Timestamp(split["test_days"][-1]).normalize()
    # A later observed month proves that this source partition is closed.  For
    # the final observed month, require coverage into the last three business
    # days; this correctly marks the source's truncated 2026-05 partition.
    near_month_end = observed_test_end >= (test_period.end_time.normalize() - pd.offsets.BDay(3))
    test_month_complete = bool(test_period < observed_period or near_month_end)
    split_json.update(
        {
            "experiment_id": exp_id,
            "n_train": len(split["train_days"]),
            "n_validation": len(split["val_days"]),
            "n_test": len(split["test_days"]),
            "data_version": store.data_version,
            "information_cutoff": split["validation_end_date"],
            "test_observed_end": str(split["test_days"][-1].date()),
            "test_calendar_month_complete": test_month_complete,
        }
    )
    write_json(out / "split_manifest.json", split_json)
    write_json(out / "feature_schema.json", {"feature_names": store.feature_names, "feature_count": store.feat_dim, "feature_schema_hash": store.feature_schema_hash})
    metadata = {
        "experiment_id": exp_id,
        "test_month": month,
        "oos_mode": oos_mode,
        "data_version": store.data_version,
        "raw_data_hash": store.raw_zip_hash,
        "feature_schema_hash": store.feature_schema_hash,
        "feature_config_hash": store.feature_config_hash,
        "train_boundary": [split["train_start_date"], split["train_end_date"]],
        "validation_boundary": [split["validation_start_date"], split["validation_end_date"]],
        "test_boundary": [split["test_start_date"], split["test_end_date"]],
        "git_commit": env["git_commit"],
        "git_dirty": env.get("git_dirty"),
        "git_diff_sha256": env.get("git_diff_sha256"),
        "config": cfg,
    }
    runtime_start = time.time()
    gpu_monitor = GpuMonitor(out / "gpu_monitor_samples.csv", interval=float(cfg["gpu"]["monitor_interval_sec"]))
    gpu_monitor.start()
    gpu_monitor_stopped = False
    _append_union_csv(out / "gpu_logs.csv", _gpu_row("start"))
    try:
        train_stride = max(1, int(cfg["runtime"]["train_day_stride"]))
        state_stride = max(1, int(cfg["runtime"]["state_day_stride"]))
        train_days = list(split["train_days"])[::train_stride]
        state_days = list(split["train_days"])[::state_stride]
        val_days = list(split["val_days"])
        phase = time.time()
        alpha, normalizer, alpha_meta = train_alpha(store, train_days, val_days, cfg, device, out, metadata, gpu_monitor)
        _append_union_csv(
            out / "runtime_logs.csv",
            {
                "phase": "alpha_training",
                "seconds": time.time() - phase,
                **{key: alpha_meta[key] for key in ("initial_stock_chunk", "final_stock_chunk", "batch_asofs", "gradient_accumulation", "gradient_checkpointing", "oom_retries")},
            },
        )
        _append_union_csv(out / "gpu_logs.csv", _gpu_row("alpha_trained"))

        phase = time.time()
        train_cache = build_state_cache(store, alpha, normalizer, state_days, cfg, device, "train")
        val_cache = build_state_cache(store, alpha, normalizer, val_days, cfg, device, "validation")
        if not train_cache or not val_cache:
            raise RuntimeError("empty train/validation portfolio state cache")
        _append_union_csv(out / "runtime_logs.csv", {"phase": "portfolio_state_generation", "seconds": time.time() - phase, "train_states": len(train_cache), "validation_states": len(val_cache)})

        phase = time.time()
        generative = train_generative(train_cache, val_cache, cfg, device, out, metadata)
        _append_union_csv(out / "runtime_logs.csv", {"phase": "generative_training", "seconds": time.time() - phase})
        phase = time.time()
        policies = {algorithm: train_rl(generative["ssfm"], train_cache, val_cache, cfg, device, out, metadata, algorithm) for algorithm in cfg["rl"]["algorithms"]}
        _append_union_csv(out / "runtime_logs.csv", {"phase": "rl_training", "seconds": time.time() - phase})
        _append_union_csv(out / "gpu_logs.csv", _gpu_row("all_models_trained"))
        g_values = [int(x) for x in cfg["matrix"]["E_g"]]
        selected_g = select_g_on_validation(generative["ssfm"], val_cache, cfg, device, g_values, out)
        write_json(out / "g_selection.json", {"selected_g": selected_g, "selection_split": "validation", "selection_rule": "maximum validation mean net return", "candidates": g_values})

        alpha_checkpoint_id = file_sha256(out / "models" / "alpha_best_validation.pt")
        pred_rows: List[Dict[str, Any]] = []
        pred_detail: List[pd.DataFrame] = []
        test_items: List[Dict[str, Any]] = []
        attention_rows: List[Dict[str, Any]] = []
        temporal_vectors: List[np.ndarray] = []
        phase = time.time()
        for position, day in enumerate(split["test_days"], start=1):
            y, _ = store.labels(day)
            prediction, valid, pred_meta, inference = predict_day(alpha, store, day, device, normalizer)
            if int(valid.sum()) < int(cfg["portfolio"]["top_k"]):
                log(f"    test date {day.date()} skipped valid={int(valid.sum())}")
                continue
            selected = select_topk(prediction, valid, int(cfg["portfolio"]["top_k"]))
            metrics = {**daily_ic(prediction[valid], y[valid]), **top30_stats(prediction, y, selected, valid)}
            quantiles = np.nanquantile(prediction[valid], [0.01, 0.25, 0.5, 0.75, 0.99])
            pred_rows.append(
                {
                    "date": str(day.date()),
                    **metrics,
                    "topk_mean_true_return": metrics["top30_mean_true"],
                    "topk_excess_return_vs_universe": metrics["top30_excess"],
                    "topk_hit_rate": metrics["top30_hit_rate"],
                    "prediction_mean": float(np.nanmean(prediction[valid])),
                    "prediction_std": float(np.nanstd(prediction[valid])),
                    **{f"prediction_q{int(q*100):02d}": float(v) for q, v in zip([0.01, 0.25, 0.5, 0.75, 0.99], quantiles)},
                    "valid_universe_size": int(valid.sum()),
                    "inference_time_sec": inference,
                }
            )
            ranks = np.full(store.N, np.nan)
            order = np.flatnonzero(valid)[np.argsort(-prediction[valid])]
            ranks[order] = np.arange(1, len(order) + 1)
            clocks = store.clocks(day)
            detail = pd.DataFrame(
                {
                    "date": str(day.date()),
                    "stock_code": store.kept,
                    "predicted_alpha": prediction,
                    "true_intraday_return": y,
                    "rank": ranks,
                    "selected_flag": np.isin(np.arange(store.N), selected),
                    "execution_eligible_flag": np.isfinite(store.open[store.day_index(day)]) & (store.open[store.day_index(day)] > 0),
                    "valid_universe_size": int(valid.sum()),
                    "decision_timestamp": clocks["decision_timestamp"],
                    "data_cutoff_timestamp": clocks["data_cutoff_timestamp"],
                    "model_checkpoint_id": alpha_checkpoint_id,
                    "oos_mode": oos_mode,
                    "feature_schema_hash": store.feature_schema_hash,
                }
            )
            pred_detail.append(detail.loc[valid].copy())
            test_items.append(make_state(store, day, prediction, valid, y, cfg))
            aux = pred_meta.get("aux", {})
            temporal = aux.get("temporal_patch_attention")
            if temporal is not None:
                temporal_vectors.append(np.asarray(temporal, dtype=np.float64))
            attention_rows.append(
                {
                    "date": str(day.date()),
                    "temporal_attention_entropy": float(np.asarray(aux.get("temporal_attention_entropy", np.nan))),
                    "temporal_attention_concentration": float(np.asarray(aux.get("temporal_attention_concentration", np.nan))),
                    "temporal_mask_rate": float(np.asarray(aux.get("temporal_mask_rate", np.nan))),
                    "cross_stock_attention_entropy": float(np.asarray(aux.get("cross_stock_attention_entropy", np.nan))),
                    "cross_stock_attention_concentration": float(np.asarray(aux.get("cross_stock_attention_concentration", np.nan))),
                    "cross_stock_mask_rate": float(np.asarray(aux.get("cross_stock_mask_rate", np.nan))),
                    "universe_size": int(valid.sum()),
                }
            )
            log(f"    fixed Test inference {position}/{len(split['test_days'])} {day.date()} valid={int(valid.sum())}")
        if not test_items:
            raise RuntimeError("no valid Test dates")
        pd.DataFrame(pred_rows).to_csv(out / "prediction_metrics.csv", index=False)
        pd.concat(pred_detail, ignore_index=True).to_parquet(out / "daily_predictions.parquet", index=False, compression="zstd")
        pd.DataFrame(attention_rows).to_csv(out / "attention_statistics.csv", index=False)
        if temporal_vectors:
            mean_attention = np.vstack(temporal_vectors).mean(axis=0)
            patch_size = int(cfg["prediction"]["patch"])
            bars = int(cfg["calendar"]["bars_per_day"])
            pd.DataFrame(
                {
                    "patch_index": np.arange(len(mean_attention)),
                    "history_day_index": (np.arange(len(mean_attention)) * patch_size) // bars,
                    "minute_start_in_day": (np.arange(len(mean_attention)) * patch_size) % bars,
                    "mean_attention": mean_attention,
                }
            ).to_csv(out / "temporal_attention_summary.csv", index=False)
        _append_union_csv(out / "runtime_logs.csv", {"phase": "test_inference", "seconds": time.time() - phase, "test_dates": len(test_items)})

        daily_frames: List[pd.DataFrame] = []
        weight_frames: List[pd.DataFrame] = []
        candidate_rows: List[Dict[str, Any]] = []
        strategy_labels = {
            "equal_weight": "Equal Weight",
            "mvo": "MVO",
            "max_sharpe": "Maximum Sharpe",
            "risk_parity": "Risk Parity",
            "black_litterman": "Black-Litterman",
            "kelly": "Kelly",
            "mlp": "MLP Policy",
            "gaussian": "Gaussian Policy",
            "diffusion": "Diffusion",
            "standard_fm": "Standard FM",
            "ssfm": "SS-FM",
        }
        for key in cfg["matrix"]["B_portfolio"]:
            weights = [item["baselines"][key] for item in test_items]
            label = strategy_labels[key]
            daily_frames.append(backtest_items(test_items, weights, cfg, label, 1))
            weight_frames.append(_weights_table(test_items, weights, label))
        checkpoint_paths = {
            "ssfm": out / "models" / "ssfm_best_validation.pt",
            "standard_fm": out / "models" / "standard_fm_best_validation.pt",
            "diffusion": out / "models" / "diffusion_best_validation.pt",
            "mlp": out / "models" / "mlp_best_validation.pt",
            "gaussian": out / "models" / "gaussian_best_validation.pt",
        }
        for name, model in generative.items():
            selected_weights: List[np.ndarray] = []
            previous_weight = None
            previous_names = None
            checkpoint_id = file_sha256(checkpoint_paths[name])
            for item in test_items:
                cond = torch.from_numpy(item["cond"]).to(device)
                sampling_started = time.perf_counter()
                sampled, _ = sample_and_audit(lambda n=name, m=model, c=cond: sample_model(n, m, c, selected_g, cfg))
                sampling_latency = time.perf_counter() - sampling_started
                array = sampled.detach().cpu().numpy()
                chosen, weight, scores = select_candidate(array, item, cfg)
                selected_weights.append(weight)
                candidate_rows.extend(
                    _candidate_rows(
                        item,
                        array,
                        scores,
                        chosen,
                        strategy_labels[name],
                        checkpoint_id,
                        cfg,
                        selected_g,
                        previous_weight,
                        previous_names,
                        sampling_latency,
                    )
                )
                previous_weight = weight
                previous_names = item["names"]
            daily_frames.append(backtest_items(test_items, selected_weights, cfg, strategy_labels[name], selected_g))
            weight_frames.append(_weights_table(test_items, selected_weights, strategy_labels[name]))
        for algorithm, policy in policies.items():
            weights: List[np.ndarray] = []
            previous_weight = None
            previous_names = None
            label = f"SS-FM + {algorithm.upper()}"
            checkpoint_id = file_sha256(out / "models" / f"{algorithm}_best_validation.pt")
            for item in test_items:
                cond = torch.from_numpy(item["cond"]).to(device)
                sampling_started = time.perf_counter()
                sampled, _, logp = sample_portfolios(policy, cond, selected_g, float(cfg["rl"]["noise_std"]))
                sampling_latency = time.perf_counter() - sampling_started
                array = sampled.detach().cpu().numpy()
                chosen, weight, scores = select_candidate(array, item, cfg)
                weights.append(weight)
                rows = _candidate_rows(item, array, scores, chosen, label, checkpoint_id, cfg, selected_g, previous_weight, previous_names, sampling_latency)
                log_values = logp.detach().cpu().numpy()
                for row, value in zip(rows, log_values):
                    row["policy_log_probability"] = float(value)
                candidate_rows.extend(rows)
                previous_weight = weight
                previous_names = item["names"]
            daily_frames.append(backtest_items(test_items, weights, cfg, label, selected_g))
            weight_frames.append(_weights_table(test_items, weights, label))
        for count in g_values:
            weights = []
            previous_weight = None
            previous_names = None
            started = time.perf_counter()
            for item in test_items:
                cond = torch.from_numpy(item["cond"]).to(device)
                sampling_started = time.perf_counter()
                sampled, _ = sample_and_audit(lambda c=cond, g=count: sample_ss_fm_mixed(generative["ssfm"], c, g, int(cfg["ssfm"]["n_sample_steps"])))
                sampling_latency = time.perf_counter() - sampling_started
                array = sampled.detach().cpu().numpy()
                chosen, weight, scores = select_candidate(array, item, cfg)
                weights.append(weight)
                candidate_rows.extend(
                    _candidate_rows(
                        item,
                        array,
                        scores,
                        chosen,
                        f"SS-FM G={count}",
                        file_sha256(out / "models" / "ssfm_best_validation.pt"),
                        cfg,
                        count,
                        previous_weight,
                        previous_names,
                        sampling_latency,
                    )
                )
                previous_weight = weight
                previous_names = item["names"]
            frame = backtest_items(test_items, weights, cfg, f"SS-FM G={count}", count)
            frame["sampling_time_sec"] = (time.perf_counter() - started) / max(len(test_items), 1)
            daily_frames.append(frame)
            weight_frames.append(_weights_table(test_items, weights, f"SS-FM G={count}"))

        daily = pd.concat(daily_frames, ignore_index=True)
        weights_df = pd.concat(weight_frames, ignore_index=True)
        candidates_df = pd.DataFrame(candidate_rows)
        daily.to_csv(out / "backtest_daily.csv", index=False)
        daily.to_parquet(out / "daily_portfolios.parquet", index=False, compression="zstd")
        weights_df.to_parquet(out / "portfolio_weights.parquet", index=False, compression="zstd")
        candidates_df.to_parquet(out / "candidate_portfolios.parquet", index=False, compression="zstd")
        performance = summarize_nav(daily)
        performance.to_csv(out / "performance_summary.csv", index=False)
        candidate_summary = _candidate_summary(candidates_df)
        candidate_summary.to_csv(out / "candidate_summary.csv", index=False)
        g_test_summary = candidate_summary[candidate_summary["strategy_name"].astype(str).str.startswith("SS-FM G=")].merge(
            performance,
            left_on="strategy_name",
            right_on="model",
            how="left",
        )
        g_test_summary.to_csv(out / "g_test_summary.csv", index=False)
        write_json(out / "performance_summary.json", {"strategies": performance.to_dict("records"), "selected_g": selected_g})
        write_json(out / "model_updates.json", [{"phase": "pre_test", "cutoff": split["validation_end_date"], "effective_from": split["test_start_date"], "test_updates": 0}])

        audit_rows = run_strict_audit(
            experiment_id=exp_id,
            split=split_json,
            out=out,
            cfg=cfg,
            store=store,
            model_updates=1,
            normalizer_fit_range=normalizer.fit_range,
            executed_modules={"alpha": [STRATEGY_NAME], "oos": ["strict_fixed_oos"]},
        )
        valid_audit = write_audit(out / "leakage_audit.json", audit_rows)
        _append_union_csv(out / "gpu_logs.csv", _gpu_row("complete"))
        gpu_summary = gpu_monitor.stop()
        gpu_monitor_stopped = True
        write_json(out / "gpu_summary.json", gpu_summary)
        _append_union_csv(out / "runtime_logs.csv", {"phase": "total", "seconds": time.time() - runtime_start, **gpu_summary})
        report_meta = {
            **metadata,
            **alpha_meta,
            "selected_g": selected_g,
            "valid_audit": valid_audit,
            "test_dates_executed": len(test_items),
            "average_universe_size": float(pd.DataFrame(pred_rows)["valid_universe_size"].mean()),
            "total_runtime_sec": time.time() - runtime_start,
        }
        write_month_report(out, report_meta, cfg)
        if not valid_audit:
            write_json(out / "INVALID.json", {"experiment_id": exp_id, "reason": "high-severity leakage audit failure"})
        return {"out": str(out), "experiment_id": exp_id, "valid": valid_audit, "month": month, "status": "ok" if valid_audit else "invalid_audit"}
    except Exception as exc:
        if not gpu_monitor_stopped:
            gpu_summary = gpu_monitor.stop()
            write_json(out / "gpu_summary.json", gpu_summary)
        failure = {
            "status": "failed",
            "experiment_id": exp_id,
            "month": month,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "attempted_fix": "OOM fallback reduces stock chunk and enables gradient checkpointing; other failures are preserved without fabricated output",
            "retry_status": "failed",
            "timestamp": pd.Timestamp.now(tz="UTC").isoformat(),
        }
        write_json(out / "failures" / "pipeline_failure.json", failure)
        write_json(out_root / "failed_experiments" / f"{exp_id}.json", failure)
        raise
