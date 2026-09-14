"""GPU throughput knobs. Seed is kept; full cudnn determinism is dropped when targeting high util."""

from __future__ import annotations

import os
from typing import Any, Dict

import torch

from utils.logging import log


def configure_gpu(cfg: Dict[str, Any]) -> Dict[str, Any]:
    gpu = cfg.get("gpu", {})
    info: Dict[str, Any] = {"cuda": bool(torch.cuda.is_available())}
    if not torch.cuda.is_available():
        return info
    high = bool(gpu.get("high_utilization", True))
    if bool(gpu.get("allow_tf32", True)):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
        info["tf32"] = True
    bench = bool(gpu.get("cudnn_benchmark", high))
    torch.backends.cudnn.benchmark = bench
    if bench:
        torch.backends.cudnn.deterministic = False
        try:
            torch.use_deterministic_algorithms(False)
        except Exception:
            pass
    info["cudnn_benchmark"] = bool(torch.backends.cudnn.benchmark)
    poll = float(gpu.get("progress_gpu_poll_sec", 30.0))
    os.environ["SSFM_GPU_POLL_SEC"] = str(poll)
    try:
        from experiments.progress import set_gpu_poll_interval

        set_gpu_poll_interval(poll)
    except Exception:
        pass
    torch.cuda.empty_cache()
    free, total = torch.cuda.mem_get_info()
    info["free_gb"] = round(free / 1e9, 2)
    info["total_gb"] = round(total / 1e9, 2)
    if high:
        log(
            f"GPU high-util: benchmark={bench} tf32={info.get('tf32', False)} "
            f"stock_chunk={cfg['prediction'].get('stock_chunk')} "
            f"batch_asofs={cfg['prediction'].get('batch_asofs')} "
            f"ckpt={gpu.get('gradient_checkpointing')} poll={poll}s "
            f"device={torch.cuda.get_device_name(0)} free={info['free_gb']}GB/{info['total_gb']}GB"
        )
    if free < 2.0e9:
        raise RuntimeError(
            f"GPU free memory only {info['free_gb']}GB; stop leftover python/run_all processes and retry"
        )
    return info
