from __future__ import annotations

import time
from pathlib import Path

from data.loaders import data_contract
from data.splits import month_starts
from data.store import ResearchStore
from evaluation.experiment_analysis import ensure_analysis_tree, finalize_analysis
from evaluation.reports import merge_and_compare
from experiments.engine import run_month
from experiments.catalog import count_summary
from experiments.progress import Progress
from utils.git_info import environment_info
from utils.logging import log, write_json
from utils.runtime import estimate_asdict, estimate_resources


def run_all(cfg: dict, out_root: Path, smoke: bool = False, months_filter=None, skip_models=None, init_ckpt=None, test_from=None, skip_friday_before=None, friday_sft_ckpt=None) -> None:
    t0 = time.time()
    contract = data_contract(cfg)
    write_json(out_root / "data_contract.json", contract)
    if not contract["ok"]:
        write_json(out_root / "failed_experiment.json", {"status": "failed", "missing": contract["missing"]})
        raise RuntimeError(f"data contract failed: {contract['missing']}")
    log("data contract PASS")
    for n in contract["notes"]:
        log(f"  {n}")

    months = month_starts(cfg["walkforward"]["test_start"], cfg["walkforward"]["test_end"])
    if months_filter:
        want = {str(x)[:7] for x in months_filter}
        months = [m for m in months if m.strftime("%Y-%m") in want]
        if not months:
            raise RuntimeError(f"no walk-forward months match {sorted(want)}")
        log(f"MONTH FILTER: 只跑 { [m.strftime('%Y-%m') for m in months] }，跑完后停下，不自动开下个月、不做全年合并结论。")
    if smoke:
        months = months[:1]
        cfg["prediction"]["epochs"] = 1
        cfg["ssfm"]["steps"] = 20
        cfg["standard_fm"]["steps"] = 20
        cfg["diffusion"]["steps"] = 20
        cfg["mlp_policy"]["steps"] = 20
        cfg["gaussian_policy"]["steps"] = 20
        cfg["rl"]["epochs"] = 1
        cfg["rl"]["distill_steps"] = 10
        log("SMOKE: 1 test month, reduced steps — still real training, not fabricated")

    est = estimate_resources(cfg, n_months=len(months))
    write_json(out_root / "resource_estimate.json", estimate_asdict(est))
    write_json(out_root / "environment.json", environment_info(cfg["_root"]))
    log("=== RESOURCE ESTIMATE (Sequential OOS only) ===")
    for k, v in estimate_asdict(est).items():
        log(f"  {k}: {v}")

    analysis_root = ensure_analysis_tree(out_root)
    cnt = count_summary()
    log(f"analysis placeholders -> {analysis_root}")
    log(
        f"实验规模: 主实验族 {cnt['n_main_families']} + 消融/系统 {cnt['n_ablation_families']} "
        f"= {cnt['n_analysis_docs']} 份分析文档; "
        f"13月主单元 {cnt['n_main_units_13m']} + 周五重训 {cnt['n_friday_retrain_units']} "
        f"= {cnt['n_total_train_or_backtest_units']} 个训练/回测单元"
    )
    log("building / loading ResearchStore …")
    store = ResearchStore(cfg)
    write_json(out_root / "feature_schema.json", {"names": store.feature_names, "F": store.feat_dim, "meta": store.feature_meta})

    oos_modes = list(cfg["oos_modes"])
    total = len(months) * len(oos_modes)
    log("======== 实验矩阵 ========")
    log(f"  F-OOS: {oos_modes}")
    log(f"  月份: {[m.strftime('%Y-%m') for m in months]}  共{len(months)}个月")
    log("  A-Prediction: " + ", ".join(cfg["prediction"]["models"]))
    log("  B-Portfolio: " + ", ".join(cfg["matrix"]["B_portfolio"]))
    log("  C-Generative: " + ", ".join(cfg["matrix"]["C_generative"]))
    log("  D-RL: " + ", ".join(cfg["matrix"]["D_rl"]))
    log("  E-G: " + str(cfg["matrix"]["E_g"]))
    log("  OOS 仅 Sequential Weekly Retrain；禁止与已弃用的 Strict 混合。")
    prog = Progress(total, "F-OOS 月度总进度", log_every=1)
    results = []
    for oos in oos_modes:
        for m in months:
            tag = f"{oos} {m.strftime('%Y-%m')}"
            log(f"======== 实验F {tag} ========")
            try:
                rec = run_month(
                    cfg, store, m, oos, out_root,
                    g_grid=list(cfg["matrix"]["E_g"]),
                    skip_models=skip_models,
                    init_ckpt=init_ckpt,
                    test_from=test_from,
                    skip_friday_before=skip_friday_before,
                    friday_sft_ckpt=friday_sft_ckpt,
                )
                rec["status"] = "ok" if rec["valid"] else "invalid_audit"
                rec["month"] = m.strftime("%Y-%m")
                rec["oos"] = oos
                results.append(rec)
            except Exception as e:
                import traceback
                err = {"status": "failed", "error": str(e), "traceback": traceback.format_exc(), "month": m.strftime("%Y-%m"), "oos": oos}
                write_json(out_root / oos / f"test_month={m.strftime('%Y-%m')}" / "failed_experiment.json", err)
                log(f"FAILED {tag}: {e}")
                results.append(err)
            prog.update(msg=tag)
    write_json(out_root / "run_index.json", [{k: v for k, v in r.items() if k not in ("daily", "pred")} for r in results])

    final = out_root / "final_summary"
    final.mkdir(parents=True, exist_ok=True)
    for oos in oos_modes:
        paths = list((out_root / oos).glob("test_month=*/**/backtest_daily.csv"))
        valid_paths = []
        for p in paths:
            if (p.parent / "INVALID.json").is_file():
                continue
            valid_paths.append(p)
        merge_and_compare(valid_paths, final / oos, f"{oos} valid only")
    write_json(
        final / "summary.json",
        {
            "hours": (time.time() - t0) / 3600.0,
            "n_ok": sum(1 for r in results if r.get("status") == "ok"),
            "n_invalid": sum(1 for r in results if r.get("status") == "invalid_audit"),
            "n_failed": sum(1 for r in results if r.get("status") == "failed"),
            "note": "strict_fixed_oos and sequential_oos_weekly_retrain are never merged",
        },
    )
    finalize_analysis(out_root, oos_modes, {"oos": ",".join(oos_modes), "top_k": cfg["portfolio"]["top_k"], "cost_bps": cfg["portfolio"]["cost_bps"], "primary": cfg["prediction"]["primary_model"]})
    prog.close("requested months done")
    log(f"REQUESTED MONTHS DONE hours={(time.time()-t0)/3600:.2f} out={out_root}")
    if months_filter:
        log("停在这里：尚未跑后续月份，也没有把全年总览当成完整 OOS 年。需要的话再开下个月。")
