"""Resource estimates, OOM retry, parameter counts."""

from __future__ import annotations

import time
import traceback
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, Optional

import torch


@dataclass
class ResourceEstimate:
    params: int
    trainable_params: int
    batch_size: int
    grad_accum: int
    est_vram_gb: float
    est_gpu_util_pct: float
    est_sec_per_epoch: float
    est_total_hours: float
    est_infer_sec: float
    est_sample_sec: float
    est_disk_gb: float
    est_gpu_hours: float
    notes: str = ""


def count_params(model: torch.nn.Module) -> Dict[str, int]:
    n = sum(p.numel() for p in model.parameters())
    t = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"params": int(n), "trainable_params": int(t)}


def estimate_resources(cfg: Dict[str, Any], n_months: int = 13) -> ResourceEstimate:
    pred = cfg["prediction"]
    d = int(pred["d_model"])
    patch = int(pred["patch"])
    lookback = int(pred["lookback_bars"])
    n_stocks = int(cfg["universe"]["n_stocks_cap"])
    epochs = int(pred["epochs"])
    top_k = int(cfg["portfolio"]["top_k"])
    g = int(cfg["rl"]["group_size"])
    patches = lookback // max(patch, 1)
    params = feat_dim_guess(cfg) * patch * d + d * d * 12 + top_k * 128 * 6 + 250_000
    act_gb = n_stocks * patches * d * 2 / 1e9 * 3
    gpu_mem = 8.0
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    vram = min(gpu_mem * 0.9, 1.0 + act_gb + params * 4 / 1e9 * 3)
    sec_ep = n_stocks / 500.0 * (lookback / 2400.0) * 220 * 1.6
    n_pred = len(pred["models"])
    n_gen = 5
    n_rl = 2
    n_pools = max(len((cfg.get("matrix") or {}).get("allocation_pools") or ["top30"]), 1)
    oos = len(cfg["oos_modes"])
    n_friday = 4 if "sequential_oos_weekly_retrain" in cfg.get("oos_modes", []) else 0
    init_s = sec_ep * epochs * n_months * n_pred + n_pools * (180 * n_months * n_gen + 300 * n_months * n_rl)
    # Friday close retrain: primary alpha + generative + PPO/GRPO only, both pools
    retrain_s = n_friday * n_months * (sec_ep * epochs + n_pools * (180 * n_gen + 300 * n_rl))
    if "strict_fixed_oos" in cfg.get("oos_modes", []) and n_friday:
        init_s *= oos
    elif "strict_fixed_oos" in cfg.get("oos_modes", []):
        init_s *= max(oos, 1)
    total_h = (init_s + retrain_s) / 3600.0
    notes = (
        f"OOS={cfg.get('oos_modes')} | {n_months} months | pools={n_pools} "
        f"(top30+all uses compact cond on K=500) | "
        f"Friday retrain ~{n_friday}/month (primary+SS-FM+RL per pool) | "
        f"SFT rank_weight={pred.get('rank_weight', 0)} csz={pred.get('sft_cs_zscore', True)} val_select={pred.get('val_select', 'mse')} | "
        "AMP + multi-asof batch + cudnn.benchmark (high_utilization)"
    )
    if gpu_mem >= 30:
        notes = "RTX 32GB class: AMP on, allow grad accum / checkpoint on OOM"
    return ResourceEstimate(
        params=int(params),
        trainable_params=int(params),
        batch_size=int(pred["batch_asofs"]),
        grad_accum=1,
        est_vram_gb=round(float(vram), 2),
        est_gpu_util_pct=82.0 if cfg.get("gpu", {}).get("high_utilization", True) else (72.0 if gpu_mem < 16 else 80.0),
        est_sec_per_epoch=round(sec_ep, 1),
        est_total_hours=round(total_h, 1),
        est_infer_sec=round(0.6 * lookback / 2400.0 * n_stocks / 500.0, 2),
        est_sample_sec=round(0.05 * g, 3),
        est_disk_gb=round(n_months * oos * 0.25 + 3.0, 2),
        est_gpu_hours=round(total_h * 1.1, 1),
        notes=notes,
    )


def feat_dim_guess(cfg: Dict[str, Any]) -> int:
    f = cfg["prediction"].get("feat_dim", "auto")
    return 46 if f == "auto" else int(f)


def is_oom(err: BaseException) -> bool:
    msg = str(err).lower()
    return "out of memory" in msg or "cuda" in msg and "oom" in msg


def run_with_oom_retry(
    fn: Callable[[], Any],
    *,
    reduce_chunk: Optional[Dict[str, Any]] = None,
    max_retries: int = 4,
    on_event: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Any:
    last = None
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except RuntimeError as e:
            last = e
            if not is_oom(e) or attempt >= max_retries:
                raise
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            ev = {
                "event": "oom",
                "attempt": attempt + 1,
                "error": str(e),
                "traceback": traceback.format_exc(),
                "t": time.time(),
            }
            if reduce_chunk is not None and "stock_chunk" in reduce_chunk:
                old = int(reduce_chunk["stock_chunk"])
                reduce_chunk["stock_chunk"] = max(8, old // 2)
                ev["old_stock_chunk"] = old
                ev["new_stock_chunk"] = reduce_chunk["stock_chunk"]
                reduce_chunk["gradient_checkpointing"] = True
            if on_event:
                on_event(ev)
    raise last  # type: ignore[misc]


def estimate_asdict(est: ResourceEstimate) -> Dict[str, Any]:
    return asdict(est)
