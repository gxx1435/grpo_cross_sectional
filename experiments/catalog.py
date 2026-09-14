"""Canonical experiment families: main matrix + ablations. One analysis doc per family."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

CATALOG: List[Dict[str, Any]] = [
    {
        "id": "全年总览",
        "kind": "overview",
        "title": "全年总览：2025-05 至 2026-05 全部实验",
        "models": [],
        "daily_prefixes": [],
        "n_units_per_month": 0,
        "desc": "13 个测试月拼接后的全样本总分析：所有主实验 + 消融的指标表、图和结论。不是单月切片。",
    },
    {
        "id": "M01_prediction",
        "kind": "main",
        "title": "主实验 A：Alpha 预测骨干对比",
        "models": ["lstm", "tcn", "standard_transformer", "patchtst", "minute_only", "concat", "cross_attention", "gated_residual"],
        "daily_prefixes": ["pred_"],
        "n_units_per_month": 8,
        "desc": "8 个预测模型（含 PatchTST），月度 walk-forward 训练 + validation（含 F1）+ Top30 等权回测。",
    },
    {
        "id": "M02_portfolio",
        "kind": "main",
        "title": "主实验 B：组合构建基线对比",
        "models": ["equal_weight", "mvo", "max_sharpe", "risk_parity", "black_litterman", "kelly"],
        "daily_prefixes": ["baseline_"],
        "n_units_per_month": 6,
        "desc": "同一 primary Alpha（gated_residual）Top30 上的 6 种组合器。",
    },
    {
        "id": "M03_generative",
        "kind": "main",
        "title": "主实验 C：生成权重模型对比",
        "models": ["mlp", "gaussian", "diffusion", "standard_fm", "ssfm"],
        "daily_prefixes": ["gen_"],
        "n_units_per_month": 5,
        "desc": "MLP / Gaussian / Diffusion / Standard FM / SS-FM 在 simplex 上生成候选权重。",
    },
    {
        "id": "M04_rl",
        "kind": "main",
        "title": "主实验 D：SS-FM + PPO / GRPO",
        "models": ["ssfm", "ssfm_ppo", "ssfm_grpo"],
        "daily_prefixes": ["rl_", "gen_ssfm"],
        "n_units_per_month": 3,
        "desc": "冻结 Alpha，SS-FM 蒸馏后 PPO / GRPO；对照纯 SS-FM。",
    },
    {
        "id": "M05_g_candidates",
        "kind": "main",
        "title": "主实验 E：候选数 G",
        "models": ["8", "16", "32", "64", "128"],
        "daily_prefixes": ["ssfm_G"],
        "n_units_per_month": 5,
        "desc": "SS-FM 采样候选数 G∈{8,16,32,64,128}，看收益与采样耗时。",
    },
    {
        "id": "M06_walkforward_oos",
        "kind": "main",
        "title": "主实验 F：Sequential OOS Weekly Retrain 走步",
        "models": ["sequential_oos_weekly_retrain"],
        "daily_prefixes": [],
        "n_units_per_month": 1,
        "desc": "仅 Sequential OOS：周五收盘后用已实现数据重训 primary+SS-FM+RL，下周一生效。13 个测试月。",
    },
    {
        "id": "X01_fusion_ablation",
        "kind": "ablation",
        "title": "消融：融合方式",
        "models": ["minute_only", "concat", "cross_attention", "gated_residual"],
        "daily_prefixes": ["pred_minute_only", "pred_concat", "pred_cross_attention", "pred_gated_residual"],
        "n_units_per_month": 4,
        "desc": "分钟-only / concat / 单向 cross-attn / 门控残差 H_hist + g⊙H_auction。",
    },
    {
        "id": "X02_minute_vs_auction",
        "kind": "ablation",
        "title": "消融：分钟 vs 集合竞价",
        "models": ["minute_only", "gated_residual"],
        "daily_prefixes": ["pred_minute_only", "pred_gated_residual"],
        "n_units_per_month": 2,
        "desc": "有无 A_D 竞价增量信息。",
    },
    {
        "id": "X03_attention",
        "kind": "ablation",
        "title": "消融：Cross-Attention 诊断",
        "models": ["cross_attention"],
        "daily_prefixes": ["pred_cross_attention"],
        "n_units_per_month": 1,
        "desc": "Q=H_hist, K=V=H_auction 的注意力统计，不编造热力图。",
    },
    {
        "id": "X04_gate",
        "kind": "ablation",
        "title": "消融：门控分布",
        "models": ["gated_residual"],
        "daily_prefixes": ["pred_gated_residual"],
        "n_units_per_month": 1,
        "desc": "g 的均值/分位，检验竞价通道是否被学到。",
    },
    {
        "id": "X05_fm_vs_diffusion",
        "kind": "ablation",
        "title": "消融：Flow Matching vs Diffusion",
        "models": ["standard_fm", "diffusion", "ssfm"],
        "daily_prefixes": ["gen_standard_fm", "gen_diffusion", "gen_ssfm"],
        "n_units_per_month": 3,
        "desc": "标准 FM、扩散、单纯形 SS-FM 对照。",
    },
    {
        "id": "X06_ppo_vs_grpo",
        "kind": "ablation",
        "title": "消融：PPO vs GRPO",
        "models": ["ppo", "grpo"],
        "daily_prefixes": ["rl_ssfm_ppo", "rl_ssfm_grpo"],
        "n_units_per_month": 2,
        "desc": "同初始化、同奖励，只换更新算法。",
    },
    {
        "id": "X07_g_curves",
        "kind": "ablation",
        "title": "消融：G–收益–耗时曲线",
        "models": ["8", "16", "32", "64", "128"],
        "daily_prefixes": ["ssfm_G"],
        "n_units_per_month": 5,
        "desc": "从主实验 E 抽出 G 曲线与采样时间。",
    },
    {
        "id": "X08_leakage_audit",
        "kind": "ablation",
        "title": "系统：Leakage Audit",
        "models": [],
        "daily_prefixes": [],
        "n_units_per_month": 1,
        "desc": "高低严重级泄漏审计；FAIL 则该月 INVALID，不进入 final_summary。",
    },
    {
        "id": "X09_gpu_time",
        "kind": "ablation",
        "title": "系统：GPU 利用率与墙钟",
        "models": [],
        "daily_prefixes": [],
        "n_units_per_month": 1,
        "desc": "nvidia-smi 实测，不编造利用率。",
    },
    {
        "id": "X10_prediction_distribution",
        "kind": "ablation",
        "title": "诊断：预测分布",
        "models": [],
        "daily_prefixes": ["pred_"],
        "n_units_per_month": 1,
        "desc": "ŷ 与 y 的分布、符号、分位。",
    },
    {
        "id": "X11_top30_stats",
        "kind": "ablation",
        "title": "诊断：Top30 命中与超额",
        "models": [],
        "daily_prefixes": ["pred_"],
        "n_units_per_month": 1,
        "desc": "Hit@TopK、超额收益、覆盖重叠。",
    },
    {
        "id": "X12_oos_mode",
        "kind": "ablation",
        "title": "设定：OOS 模式说明",
        "models": ["sequential_oos_weekly_retrain"],
        "daily_prefixes": [],
        "n_units_per_month": 1,
        "desc": "Strict Fixed OOS 已弃用，只保留 Sequential Weekly Retrain。",
    },
]


def n_test_months() -> int:
    return 13


def count_summary() -> Dict[str, Any]:
    n_months = n_test_months()
    mains = [c for c in CATALOG if c["kind"] == "main"]
    abls = [c for c in CATALOG if c["kind"] == "ablation"]
    overviews = [c for c in CATALOG if c["kind"] == "overview"]
    # Independent train/backtest units (not counting Friday retrains)
    units = sum(int(c["n_units_per_month"]) for c in mains) * n_months
    friday_extra = n_months * 4 * (1 + 5 + 2)  # primary alpha + 5 gen + PPO/GRPO
    return {
        "n_main_families": len(mains),
        "n_ablation_families": len(abls),
        "n_overview_docs": len(overviews),
        "n_analysis_docs": len(CATALOG),
        "n_test_months": n_months,
        "n_main_units_13m": units,
        "n_friday_retrain_units": friday_extra,
        "n_total_train_or_backtest_units": units + friday_extra,
        "note": "分析文档按实验族计（18 份），不是按每个月每个模型各写一份。",
    }


def analysis_root(out_root: Path) -> Path:
    return Path(out_root) / "analysis"


def family_dir(out_root: Path, exp_id: str) -> Path:
    return analysis_root(out_root) / exp_id
