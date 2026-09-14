"""Execute the full A–F matrix with monthly walk-forward and separated OOS modes."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTHONUNBUFFERED", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from experiments.run_all import run_all
from utils.config import load_config, resolve_path
from utils.gpu import configure_gpu
from utils.gpu_monitor import GpuMonitor
from utils.logging import Tee, log
from utils.runtime import estimate_asdict, estimate_resources
from utils.seed import set_seed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="1 real month, fewer steps (still trains, never fabricates)")
    ap.add_argument("--months", nargs="*", default=None, help="只跑这些测试月，例如 2025-06")
    ap.add_argument("--skip-models", nargs="*", default=None, help="这些 SFT 模型从 --init-ckpt 加载，不再训")
    ap.add_argument("--init-ckpt", default=None, help="已完成 SFT 的 checkpoints 目录")
    ap.add_argument("--test-from", default=None, help="TEST 从该日开始推断，更早的日期沿用当月 CSV")
    ap.add_argument("--skip-friday-before", default=None, help="该日之前的周五重训跳过，沿用 --init-ckpt")
    ap.add_argument("--friday-sft-ckpt", default=None, help="周五 primary SFT 已完成时直接加载，只补 SS-FM/RL")
    args = ap.parse_args()
    cfg = load_config()
    set_seed(int(cfg["experiment"]["seed"]), bool(cfg["experiment"]["deterministic"]), bool(cfg["experiment"]["deterministic_warn_only"]))
    configure_gpu(cfg)
    tag = "smoke" if args.smoke else "full"
    out = resolve_path(cfg, cfg["paths"]["results_dir"]) / tag
    out.mkdir(parents=True, exist_ok=True)
    tee = Tee(out / "run.log")
    sys.stdout = tee  # type: ignore
    sys.stderr = tee  # type: ignore
    n_run = 1 if args.smoke else (len(args.months) if args.months else 13)
    est = estimate_resources(cfg, n_run)
    log(f"out={out}")
    log(
        f"CSI500 SS-FM | F=auto | Top{cfg['portfolio']['top_k']} | pools={cfg.get('matrix', {}).get('allocation_pools')} | "
        f"rank_w={cfg['prediction'].get('rank_weight')} csz={cfg['prediction'].get('sft_cs_zscore', False)} | "
        f"Test filter={args.months or (cfg['walkforward']['test_start']+'..'+cfg['walkforward']['test_end'])}"
        + (f" | test_from={args.test_from}" if args.test_from else "")
        + (f" | skip_friday_before={args.skip_friday_before}" if args.skip_friday_before else "")
        + (f" | friday_sft_ckpt={args.friday_sft_ckpt}" if args.friday_sft_ckpt else "")
    )
    log(f"EST ~{est.est_total_hours}h wall / {est.est_gpu_hours} GPU-h | util~{est.est_gpu_util_pct}% VRAM~{est.est_vram_gb}GB | {est.notes}")
    gpu = GpuMonitor(out / "gpu_logs.csv", float(cfg["gpu"]["monitor_interval_sec"]))
    gpu.start()
    try:
        run_all(
            cfg,
            out,
            smoke=args.smoke,
            months_filter=args.months,
            skip_models=args.skip_models,
            init_ckpt=args.init_ckpt,
            test_from=args.test_from,
            skip_friday_before=args.skip_friday_before,
            friday_sft_ckpt=args.friday_sft_ckpt,
        )
    finally:
        log("GPU " + str(gpu.stop()))


if __name__ == "__main__":
    main()
