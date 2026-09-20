"""Fixed table headers, row order, and series colors for analysis docs.

Missing cells stay as "-". Curve colors do not change when new data arrives.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

TEST_MONTHS: List[str] = [
    "2025-05",
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

PRED_MODELS: List[str] = [
    "lstm",
    "tcn",
    "standard_transformer",
    "patchtst",
    "minute_only",
    "concat",
    "cross_attention",
    "gated_residual",
]
PORTFOLIO_MODELS: List[str] = [
    "equal_weight",
    "mvo",
    "max_sharpe",
    "risk_parity",
    "black_litterman",
    "kelly",
]
GEN_MODELS: List[str] = ["mlp", "gaussian", "diffusion", "standard_fm", "ssfm"]
RL_MODELS: List[str] = ["ssfm", "ssfm_ppo", "ssfm_grpo"]
G_VALUES: List[str] = ["8", "16", "32", "64", "128"]

NAV_DAILY_MODELS: List[str] = (
    [f"pred_{m}_ew_top30" for m in PRED_MODELS]
    + [f"pred_{m}_ew" for m in PRED_MODELS]
    + [f"pred_{m}_softmax_all" for m in PRED_MODELS]
    + ["universe_ew_all"]
    + [f"baseline_{m}_{p}" for p in ("top30", "all") for m in PORTFOLIO_MODELS]
    + [f"baseline_{m}" for m in PORTFOLIO_MODELS]
    + ["baseline_score_softmax_all"]
    + [f"gen_{m}_{p}" for p in ("top30", "all") for m in GEN_MODELS]
    + [f"gen_{m}" for m in GEN_MODELS]
    + [f"rl_ssfm_{a}_{p}" for p in ("top30", "all") for a in ("ppo", "grpo")]
    + ["rl_ssfm_ppo", "rl_ssfm_grpo"]
    + [f"ssfm_G{g}_{p}" for p in ("top30", "all") for g in G_VALUES]
    + [f"ssfm_G{g}" for g in G_VALUES]
)

NAV_COLS: List[str] = [
    "model",
    "total_net_return",
    "ann_return",
    "sharpe",
    "max_drawdown",
    "mean_turnover",
    "mean_cost",
    "n_days",
]
PRED_COLS: List[str] = [
    "model",
    "MSE",
    "MAE",
    "IC",
    "RankIC",
    "topk_f1",
    "topk_precision",
    "topk_recall",
    "dir_acc",
    "dir_f1",
    "dir_precision",
    "dir_recall",
    "top30_hit_rate",
    "top30_excess",
    "n",
]
VAL_COLS: List[str] = [
    "model",
    "epoch",
    "MSE",
    "MAE",
    "IC",
    "RankIC",
    "topk_f1",
    "topk_precision",
    "topk_recall",
    "dir_acc",
    "dir_f1",
    "dir_precision",
    "dir_recall",
    "top30_hit_rate",
    "n",
]
VAL_BEST_COLS: List[str] = [c for c in VAL_COLS if c != "epoch"]
MONTH_STATUS_COLS: List[str] = ["month", "status", "n_pred_models_val", "n_nav_models", "leakage_audit"]
UNIT_STATUS_COLS: List[str] = ["unit", "model", "status"]
LEAKAGE_COLS: List[str] = ["check", "severity", "passed", "detail"]
GPU_COLS: List[str] = ["util_mean", "util_max", "mem_used_mb_mean", "power_w_mean", "n"]
GATE_COLS: List[str] = ["mean", "p25", "p50", "p75", "min", "max", "n"]
G_CURVE_COLS: List[str] = ["G", "total_net_return", "sharpe", "max_drawdown", "mean_turnover", "sampling_time_sec", "n_days"]

# Vega / Tableau-like palette. One color per logical series, reused across docs.
SERIES_COLORS: Dict[str, str] = {
    "lstm": "#4C78A8",
    "tcn": "#F58518",
    "standard_transformer": "#E45756",
    "patchtst": "#D37295",
    "minute_only": "#72B7B2",
    "concat": "#54A24B",
    "cross_attention": "#EECA3B",
    "gated_residual": "#B279A2",
    "equal_weight": "#4C78A8",
    "mvo": "#F58518",
    "max_sharpe": "#E45756",
    "risk_parity": "#72B7B2",
    "black_litterman": "#54A24B",
    "kelly": "#B279A2",
    "mlp": "#9D755D",
    "gaussian": "#BAB0AC",
    "diffusion": "#F2CF5B",
    "standard_fm": "#17BECF",
    "ssfm": "#B279A2",
    "ssfm_ppo": "#E45756",
    "ssfm_grpo": "#4C78A8",
    "ppo": "#E45756",
    "grpo": "#4C78A8",
    "8": "#9D755D",
    "16": "#BAB0AC",
    "32": "#4C78A8",
    "64": "#F58518",
    "128": "#E45756",
    "gpu": "#4C78A8",
}

# daily model name → color (same hue as the logical series)
for _m in PRED_MODELS:
    SERIES_COLORS[f"pred_{_m}_ew"] = SERIES_COLORS[_m]
for _m in PORTFOLIO_MODELS:
    SERIES_COLORS[f"baseline_{_m}"] = SERIES_COLORS[_m]
for _m in GEN_MODELS:
    SERIES_COLORS[f"gen_{_m}"] = SERIES_COLORS[_m]
SERIES_COLORS["rl_ssfm_ppo"] = SERIES_COLORS["ssfm_ppo"]
SERIES_COLORS["rl_ssfm_grpo"] = SERIES_COLORS["ssfm_grpo"]
SERIES_COLORS["gen_ssfm"] = SERIES_COLORS["ssfm"]
for _g in G_VALUES:
    SERIES_COLORS[f"ssfm_G{_g}"] = SERIES_COLORS[_g]

FALLBACK_COLOR = "#7F7F7F"
PLACEHOLDER = "-"

# Human-readable strategy names used by the S&P 500 reports.  Keep these as
# aliases of the existing logical series so monthly/annual charts and the
# legacy Top30 charts use the same stable palette rather than silently falling
# back to grey.
DISPLAY_NAME_ALIASES: Dict[str, str] = {
    "Equal Weight": "equal_weight",
    "MVO": "mvo",
    "Maximum Sharpe": "max_sharpe",
    "Risk Parity": "risk_parity",
    "Black-Litterman": "black_litterman",
    "Kelly": "kelly",
    "MLP Policy": "mlp",
    "Gaussian Policy": "gaussian",
    "Gaussian + MLP": "gaussian",
    "Diffusion": "diffusion",
    "Standard FM": "standard_fm",
    "SS-FM": "ssfm",
    "SS-FM + PPO": "ssfm_ppo",
    "SS-FM + GRPO": "ssfm_grpo",
}

FIG_CUM = dict(xlabel="交易日", ylabel="累计净收益（Σ net_return）")
FIG_IC = dict(xlabel="交易日", ylabel="IC（Pearson）")
FIG_RIC = dict(xlabel="交易日", ylabel="RankIC（Spearman）")
FIG_F1 = dict(xlabel="交易日", ylabel="F1@Top30")
FIG_HIT = dict(xlabel="交易日", ylabel="Hit@Top30（次日 y>0 占比）")
FIG_TO = dict(xlabel="模型", ylabel="日均换手")
FIG_VAL = dict(xlabel="模型", ylabel="指标值")
FIG_G = dict(xlabel="G（候选数）", ylabel="累计净收益")
FIG_GPU = dict(xlabel="时间", ylabel="GPU 利用率 %")
FIG_HIST = dict(xlabel="取值", ylabel="频数")


def color_for(name: str) -> str:
    key = str(name)
    if key in DISPLAY_NAME_ALIASES:
        return color_for(DISPLAY_NAME_ALIASES[key])
    if key.startswith("SS-FM G="):
        return color_for(key.split("=", 1)[-1])
    if key in SERIES_COLORS:
        return SERIES_COLORS[key]
    for suffix in ("_top30", "_all"):
        if key.endswith(suffix):
            return color_for(key[: -len(suffix)])
    if key == "universe_ew_all":
        return SERIES_COLORS["equal_weight"]
    if key == "baseline_score_softmax_all" or key.endswith("score_softmax_all"):
        return SERIES_COLORS["gated_residual"]
    for prefix in ("pred_", "baseline_", "gen_", "rl_"):
        if key.startswith(prefix):
            rest = key[len(prefix) :]
            if rest.endswith("_ew"):
                rest = rest[: -len("_ew")]
            if rest.startswith("ssfm_"):
                rest = rest
            if rest in SERIES_COLORS:
                return SERIES_COLORS[rest]
            if rest.replace("ssfm_", "") in SERIES_COLORS:
                return SERIES_COLORS[rest.replace("ssfm_", "")]
    if key.startswith("ssfm_G"):
        g = key.replace("ssfm_G", "")
        return SERIES_COLORS.get(g, FALLBACK_COLOR)
    return FALLBACK_COLOR


def legend_hex(names: Sequence[str]) -> List[Tuple[str, str]]:
    return [(str(n), color_for(n)) for n in names]


def unit_status_rows() -> List[Dict[str, str]]:
    rows = [{"unit": "A 预测", "model": m, "status": PLACEHOLDER} for m in PRED_MODELS]
    for pool in ("top30", "all"):
        rows += [{"unit": f"B 组合/{pool}", "model": m, "status": PLACEHOLDER} for m in PORTFOLIO_MODELS]
        rows += [{"unit": f"C 生成/{pool}", "model": m, "status": PLACEHOLDER} for m in GEN_MODELS]
        rows += [{"unit": f"D RL/{pool}", "model": m, "status": PLACEHOLDER} for m in ("ssfm_ppo", "ssfm_grpo")]
        rows += [{"unit": f"E G/{pool}", "model": f"G={g}", "status": PLACEHOLDER} for g in G_VALUES]
    return rows


def family_series(item: Dict[str, Any]) -> List[str]:
    fid = str(item.get("id", ""))
    if fid == "M01_prediction" or fid == "X10_prediction_distribution" or fid == "X11_top30_stats":
        return list(PRED_MODELS)
    if fid == "M02_portfolio":
        return [f"baseline_{m}" for m in PORTFOLIO_MODELS]
    if fid == "M03_generative" or fid == "X05_fm_vs_diffusion":
        return [f"gen_{m}" for m in (GEN_MODELS if fid == "M03_generative" else ["standard_fm", "diffusion", "ssfm"])]
    if fid == "M04_rl" or fid == "X06_ppo_vs_grpo":
        return ["gen_ssfm", "rl_ssfm_ppo", "rl_ssfm_grpo"] if fid == "M04_rl" else ["rl_ssfm_ppo", "rl_ssfm_grpo"]
    if fid == "M05_g_candidates" or fid == "X07_g_curves":
        return [f"ssfm_G{g}" for g in G_VALUES]
    if fid == "X01_fusion_ablation":
        return ["minute_only", "concat", "cross_attention", "gated_residual"]
    if fid == "X02_minute_vs_auction":
        return ["minute_only", "gated_residual"]
    if fid == "X03_attention":
        return ["cross_attention"]
    if fid == "X04_gate":
        return ["gated_residual"]
    if fid in ("M06_walkforward_oos", "全年总览"):
        return list(NAV_DAILY_MODELS)
    return list(item.get("models") or [])
