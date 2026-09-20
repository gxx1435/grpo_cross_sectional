"""Chinese monthly and annual reports for the strict S&P 500 experiment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd

from backtest.metrics import summarize_nav
from evaluation.plots import plot_cum


def _read_json(path: Path, default: Any) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


def _fmt(value: Any, digits: int = 6) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NA"
    return f"{number:.{digits}f}" if np.isfinite(number) else "NA"


def _md_table(frame: pd.DataFrame, columns: List[str]) -> str:
    if frame.empty:
        return "无有效记录。"
    view = frame.reindex(columns=columns).copy()
    for column in view.columns:
        if pd.api.types.is_numeric_dtype(view[column]):
            view[column] = view[column].map(lambda x: _fmt(x, 4))
    header = "| " + " | ".join(view.columns) + " |"
    separator = "|" + "|".join(["---"] + ["---:" for _ in view.columns[1:]]) + "|"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in view.itertuples(index=False, name=None)]
    return "\n".join([header, separator, *rows])


def write_month_report(out: Path, meta: Dict[str, Any], cfg: dict) -> None:
    split = _read_json(out / "split_manifest.json", {})
    prediction = pd.read_csv(out / "prediction_metrics.csv")
    performance = pd.read_csv(out / "performance_summary.csv")
    attention = pd.read_csv(out / "attention_statistics.csv")
    audit = pd.read_csv(out / "leakage_audit.csv")
    g_selection = pd.read_csv(out / "g_validation_selection.csv")
    candidate_summary = pd.read_csv(out / "candidate_summary.csv")
    g_test = pd.read_csv(out / "g_test_summary.csv")
    gpu = _read_json(out / "gpu_summary.json", {})
    universe_path = Path(cfg["_root"]) / cfg["paths"]["processed_dir"] / "feature_metadata" / "universe_manifest.parquet"
    universe = pd.read_parquet(universe_path)
    universe = universe[(universe["date"] >= split["test_start_date"]) & (universe["date"] <= split["test_end_date"])]
    exclusions = universe.loc[~universe["included_flag"]].groupby("exclusion_reason").size().sort_values(ascending=False)
    evaluation_exclusions = universe.loc[universe["evaluation_exclusion_reason"].astype(str).ne("")].groupby("evaluation_exclusion_reason").size().sort_values(ascending=False)
    pred_mean = prediction.mean(numeric_only=True)
    valid_audit = not ((audit["severity"] == "high") & (audit["status"] == "FAIL")).any()
    perf_cols = [
        "model",
        "cumulative_return",
        "total_net_return",
        "ann_return",
        "ann_volatility",
        "sharpe",
        "sortino",
        "max_drawdown",
        "calmar",
        "win_rate",
        "average_daily_return",
        "mean_turnover",
        "total_transaction_cost",
        "candidate_feasibility",
    ]
    lines = [
        "# S&P500实验分析",
        "",
        "## 1. 实验身份",
        "",
        f"- experiment_id: `{meta['experiment_id']}`",
        f"- Test 月: `{meta['test_month']}`",
        f"- random seed: `{cfg['experiment']['seed']}`",
        f"- 数据版本: `{meta['data_version']}`",
        f"- 原始 ZIP hash: `{meta['raw_data_hash']}`",
        f"- feature schema hash: `{meta['feature_schema_hash']}`",
        f"- git commit: `{meta['git_commit']}`",
        "- OOS mode: `strict_fixed_oos`",
        "",
        "## 2. 数据与动态股票池",
        "",
        f"- Train: `{split['train_start_date']}` 至 `{split['train_end_date']}`；Validation: `{split['validation_start_date']}` 至 `{split['validation_end_date']}`；Test: `{split['test_start_date']}` 至 `{split['test_end_date']}`。",
        f"- Test 实际执行交易日: `{meta['test_dates_executed']}`；观测终点: `{split['test_observed_end']}`；自然月完整: `{split['test_calendar_month_complete']}`。",
        f"- 平均有效股票数: `{_fmt(meta['average_universe_size'], 2)}`；universe_type: `data_available_equity_universe`。",
        f"- 历史窗口: `{cfg['calendar']['lookback_days']}` 个交易日；标准分钟模板: `{cfg['calendar']['bars_per_day']}`；特征数: `{len(_read_json(out / 'feature_schema.json', {}).get('feature_names', []))}`。",
        f"- 训练日期抽样步长: `{cfg['runtime']['train_day_stride']}`；Test 始终逐交易日完整推断。",
        f"- 排除原因摘要: `{json.dumps(exclusions.to_dict(), ensure_ascii=False)}`。",
        f"- 收盘后标签/执行可用性缺口（仅审计、未用于决策池筛选）: `{json.dumps(evaluation_exclusions.to_dict(), ensure_ascii=False)}`。",
        "- 原始文件不含历史 point-in-time 成分信息，因此这里是数据可得性代理股票池，存在 survivorship/selection bias。",
        "",
        "## 3. Alpha 最终 Test OOS 结果",
        "",
        f"- best_checkpoint_id: `{meta['best_checkpoint_id']}`",
        f"- checkpoint_selection_rule: `{meta['checkpoint_selection_rule']}`",
        f"- Test IC / RankIC: `{_fmt(pred_mean.get('IC'))}` / `{_fmt(pred_mean.get('RankIC'))}`",
        f"- Test MSE / MAE: `{_fmt(pred_mean.get('MSE'))}` / `{_fmt(pred_mean.get('MAE'))}`",
        f"- Top-K 平均真实收益 / 相对股票池超额: `{_fmt(pred_mean.get('topk_mean_true_return'))}` / `{_fmt(pred_mean.get('topk_excess_return_vs_universe'))}`",
        f"- Top-K hit rate: `{_fmt(pred_mean.get('topk_hit_rate'))}`",
        f"- 预测均值 / 标准差: `{_fmt(pred_mean.get('prediction_mean'))}` / `{_fmt(pred_mean.get('prediction_std'))}`",
        f"- 参数量: `{meta['parameter_count']}`；平均推理耗时: `{_fmt(pred_mean.get('inference_time_sec'), 4)}s`；Leakage Audit: `{'PASS' if valid_audit else 'FAIL'}`。",
        f"- GPU: `{gpu.get('gpu_name', 'NA')}`；利用率均值/峰值: `{_fmt(gpu.get('util_gpu_mean'), 2)}%` / `{_fmt(gpu.get('util_gpu_max'), 2)}%`；显存峰值: `{_fmt(gpu.get('mem_used_mb_max'), 0)} MiB`；进程 RSS 峰值: `{_fmt(gpu.get('process_rss_mb_max'), 0)} MB`。",
        "",
        "以上仅为 Validation 选出的 checkpoint 在完全未参与训练、调参和选择的 Test 上的最终结果。完整逐轮日志仅存盘供审计，本报告不展示训练过程排名或曲线。",
        "",
        "## 4. Temporal Transformer / Cross-Stock Attention 分析",
        "",
        f"- 历史 patch attention entropy / concentration: `{_fmt(attention['temporal_attention_entropy'].mean())}` / `{_fmt(attention['temporal_attention_concentration'].mean())}`。",
        f"- Cross-stock attention entropy / concentration: `{_fmt(attention['cross_stock_attention_entropy'].mean())}` / `{_fmt(attention['cross_stock_attention_concentration'].mean())}`。",
        f"- 历史 minute mask rate: `{_fmt(attention['temporal_mask_rate'].mean())}`；横截面 mask rate: `{_fmt(attention['cross_stock_mask_rate'].mean())}`。",
        "- `temporal_attention_summary.csv` 保存历史交易日/日内位置的平均注意力；`attention_statistics.csv` 保存逐日横截面统计，可与 universe size 联合审计。",
        "",
        "## 5. Portfolio 结果",
        "",
        _md_table(performance, perf_cols),
        "",
        _md_table(candidate_summary, ["strategy_name", "candidate_count_G", "candidate_diversity", "reward_diversity", "constraint_violation", "sampling_time_sec"]),
        "",
        "## 6. G 数量实验",
        "",
        f"最终默认 G=`{meta['selected_g']}`，严格由 Validation 平均净收益选择；Test 回报未参与该选择。",
        "",
        _md_table(g_selection, ["G", "validation_mean_net_return", "sampling_time_sec", "constraint_violation"]),
        "",
        _md_table(g_test, ["strategy_name", "candidate_count_G", "sampling_time_sec", "candidate_diversity", "reward_diversity", "constraint_violation", "cumulative_return", "sharpe", "max_drawdown", "mean_turnover", "candidate_feasibility"]),
        "",
        "## 7. Leakage Audit",
        "",
        f"总体状态：`{'PASS' if valid_audit else 'FAIL'}`。高严重度失败数：`{int(((audit['severity'] == 'high') & (audit['status'] == 'FAIL')).sum())}`。",
        "",
        _md_table(audit[["audit_name", "status", "observed_boundary"]], ["audit_name", "status", "observed_boundary"]),
        "",
        "Test 期间 Alpha、SS-FM、PPO、GRPO、normalizer、covariance 与 teacher 均保持决策时点因果边界；若上述审计出现高严重度失败，本月状态自动记为 INVALID 且不进入全年汇总。",
        "",
    ]
    (out / "S&P500实验分析.md").write_text("\n".join(lines), encoding="utf-8")
    try:
        plot_cum(pd.read_csv(out / "backtest_daily.csv"), out / "plots" / "cumulative_return.png", f"S&P500 {meta['test_month']} strict fixed OOS")
    except Exception:
        pass


def write_annual_report(out_root: Path, cfg: dict) -> Dict[str, Any]:
    final = out_root / "final_summary"
    (final / "plots").mkdir(parents=True, exist_ok=True)
    valid_dirs: List[Path] = []
    excluded: List[Dict[str, str]] = []
    source_manifest_path = Path(cfg["_root"]) / cfg["paths"]["processed_dir"] / "feature_metadata" / "source_file_manifest.json"
    source_manifest = _read_json(source_manifest_path, {})
    source_months = sorted(source_manifest.get("months", []), key=lambda row: str(row.get("source_month", "")))
    if source_months:
        terminal = source_months[-1]
        terminal_month = str(terminal.get("source_month", ""))
        terminal_end = pd.Timestamp(terminal.get("date_end"))
        terminal_period = pd.Period(terminal_month, freq="M")
        complete = terminal_end >= (terminal_period.end_time.normalize() - pd.offsets.BDay(3))
        if not complete and terminal_month >= str(cfg["walkforward"]["test_start"]):
            excluded.append(
                {
                    "month": terminal_month,
                    "reason": f"terminal source partition is incomplete (observed through {terminal_end.date()})",
                }
            )
    # The formal run index is authoritative.  Directory mtime is not: a later
    # smoke rerun must never replace a completed formal month in the annual
    # report.  The directory scan is retained only for old runs without an
    # index.
    run_index = _read_json(out_root / "run_index.json", [])
    if run_index:
        selected_runs = [
            (str(row.get("month", "")), Path(str(row.get("out", ""))))
            for row in run_index
            if row.get("valid") is True and row.get("status") == "ok"
        ]
    else:
        selected_runs = []
        for month_root in sorted((out_root / "strict_fixed_oos").glob("test_month=*")):
            experiments = sorted(month_root.glob("experiment_id=*"), key=lambda p: p.stat().st_mtime)
            month = month_root.name.split("=", 1)[-1]
            if experiments:
                selected_runs.append((month, experiments[-1]))
            else:
                excluded.append({"month": month, "reason": "no experiment output"})

    for month, selected in sorted(selected_runs):
        if not selected.is_dir():
            excluded.append({"month": month, "reason": "indexed experiment output missing"})
            continue
        audit = selected / "leakage_audit.csv"
        if selected.joinpath("INVALID.json").is_file() or not audit.is_file():
            excluded.append({"month": month, "reason": "invalid or missing leakage audit"})
            continue
        aud = pd.read_csv(audit)
        if ((aud["severity"] == "high") & (aud["status"] == "FAIL")).any():
            excluded.append({"month": month, "reason": "high-severity audit failure"})
            continue
        valid_dirs.append(selected)

    if not valid_dirs:
        write = "# S&P500全年实验分析\n\n没有通过 leakage audit 的有效月度实验，不能生成全年绩效结论。\n"
        (final / "S&P500全年实验分析.md").write_text(write, encoding="utf-8")
        return {"valid_months": [], "excluded": excluded}

    predictions = []
    daily = []
    audits = []
    runtimes = []
    candidate_summaries = []
    attention_summaries = []
    temporal_summaries = []
    prediction_details = []
    for path in valid_dirs:
        month = path.parent.name.split("=", 1)[-1]
        p = pd.read_csv(path / "prediction_metrics.csv")
        p["test_month"] = month
        predictions.append(p)
        d = pd.read_csv(path / "backtest_daily.csv")
        d["test_month"] = month
        daily.append(d)
        a = pd.read_csv(path / "leakage_audit.csv")
        a["test_month"] = month
        audits.append(a)
        r = pd.read_csv(path / "runtime_logs.csv")
        r["test_month"] = month
        runtimes.append(r)
        c = pd.read_csv(path / "candidate_summary.csv")
        c["test_month"] = month
        candidate_summaries.append(c)
        attention = pd.read_csv(path / "attention_statistics.csv")
        attention["test_month"] = month
        attention_summaries.append(attention)
        temporal = pd.read_csv(path / "temporal_attention_summary.csv")
        temporal["test_month"] = month
        temporal_summaries.append(temporal)
        detail = pd.read_parquet(path / "daily_predictions.parquet")
        detail["test_month"] = month
        prediction_details.append(detail)
    pred = pd.concat(predictions, ignore_index=True)
    all_daily = pd.concat(daily, ignore_index=True)
    audit_all = pd.concat(audits, ignore_index=True)
    runtime_all = pd.concat(runtimes, ignore_index=True)
    candidate_all = pd.concat(candidate_summaries, ignore_index=True)
    attention_all = pd.concat(attention_summaries, ignore_index=True)
    temporal_all = pd.concat(temporal_summaries, ignore_index=True)
    prediction_detail = pd.concat(prediction_details, ignore_index=True)
    prediction_monthly = pred.groupby("test_month", as_index=False).mean(numeric_only=True)
    performance = summarize_nav(all_daily)
    prediction_monthly.to_csv(final / "annual_prediction_summary.csv", index=False)
    performance.to_csv(final / "annual_performance_summary.csv", index=False)
    performance.to_csv(final / "annual_strategy_comparison.csv", index=False)
    audit_all.groupby(["test_month", "status"], as_index=False).size().to_csv(final / "annual_leakage_audit_summary.csv", index=False)
    runtime_all.groupby(["test_month", "phase"], as_index=False).sum(numeric_only=True).to_csv(final / "annual_runtime_summary.csv", index=False)
    annual_candidates = candidate_all.groupby("strategy_name", as_index=False, observed=True).agg(
        candidate_diversity=("candidate_diversity", "mean"),
        reward_diversity=("reward_diversity", "mean"),
        constraint_violation=("constraint_violation", "max"),
        sampling_time_sec=("sampling_time_sec", "mean"),
        candidate_count_G=("candidate_count_G", "max"),
    )
    annual_candidates.to_csv(final / "annual_candidate_summary.csv", index=False)
    attention_monthly = attention_all.groupby("test_month", as_index=False).mean(numeric_only=True)
    attention_monthly.to_csv(final / "annual_attention_summary.csv", index=False)
    temporal_by_day = temporal_all.groupby("history_day_index", as_index=False)["mean_attention"].mean()
    temporal_by_day["normalized_attention"] = temporal_by_day["mean_attention"] / temporal_by_day["mean_attention"].sum()
    temporal_by_day.to_csv(final / "annual_temporal_day_attention.csv", index=False)
    temporal_by_minute = temporal_all.groupby("minute_start_in_day", as_index=False)["mean_attention"].mean()
    temporal_by_minute["normalized_attention"] = temporal_by_minute["mean_attention"] / temporal_by_minute["mean_attention"].sum()
    temporal_by_minute.to_csv(final / "annual_intraday_attention.csv", index=False)
    sensitivity = prediction_monthly[["test_month", "valid_universe_size", "IC", "RankIC", "topk_excess_return_vs_universe"]].copy()
    sensitivity["universe_size_group"] = pd.qcut(
        sensitivity["valid_universe_size"], q=min(4, len(sensitivity)), duplicates="drop"
    ).astype(str)
    universe_sensitivity = sensitivity.groupby("universe_size_group", as_index=False, observed=True).agg(
        months=("test_month", "count"),
        mean_universe_size=("valid_universe_size", "mean"),
        mean_IC=("IC", "mean"),
        mean_RankIC=("RankIC", "mean"),
        mean_topk_excess=("topk_excess_return_vs_universe", "mean"),
    )
    universe_sensitivity.to_csv(final / "annual_universe_sensitivity.csv", index=False)
    runtime_phase = runtime_all.groupby("phase", as_index=False, observed=True)["seconds"].sum()
    try:
        plot_cum(all_daily, final / "plots" / "annual_cumulative_return.png", "S&P500 strict fixed OOS")
    except Exception:
        pass

    pred_mean = pred.mean(numeric_only=True)
    ic = prediction_monthly["IC"].to_numpy(float)
    ric = prediction_monthly["RankIC"].to_numpy(float)
    strategy_cols = ["model", "ann_return", "ann_volatility", "sharpe", "max_drawdown", "calmar", "mean_turnover", "total_transaction_cost", "candidate_feasibility"]
    valid_months = [path.parent.name.split("=", 1)[-1] for path in valid_dirs]
    splits = [_read_json(path / "split_manifest.json", {}) for path in valid_dirs]
    train_range = f"{min(row['train_start_date'] for row in splits)} 至 {max(row['train_end_date'] for row in splits)}"
    validation_range = f"{min(row['validation_start_date'] for row in splits)} 至 {max(row['validation_end_date'] for row in splits)}"
    test_range = f"{min(row['test_start_date'] for row in splits)} 至 {max(row['test_end_date'] for row in splits)}"
    attention_mean = attention_all.mean(numeric_only=True)
    selected = prediction_detail["selected_flag"].astype(bool)
    execution_eligible = prediction_detail["execution_eligible_flag"].astype(bool)
    selected_execution_exclusions = int((selected & ~execution_eligible).sum())
    selected_count = int(selected.sum())
    feature_names = _read_json(valid_dirs[0] / "feature_schema.json", {}).get("feature_names", [])

    def feature_group(name: str) -> str:
        if name in {"open", "high", "low", "close", "volume"}:
            return "原始 OHLCV"
        if name.startswith("cs_rank_"):
            return "同分钟横截面排名"
        if name.startswith(("day_of_week_", "is_")):
            return "日历变量"
        if "_w" in name or name.startswith(("return_", "vol_", "relative_volume_", "price_ma")):
            return "因果滚动窗口"
        return "价格/成交量技术特征"

    feature_groups = (
        pd.Series([feature_group(str(name)) for name in feature_names], name="feature_group")
        .value_counts()
        .rename_axis("feature_group")
        .reset_index(name="feature_count")
    )
    g_annual = annual_candidates[annual_candidates["strategy_name"].str.startswith("SS-FM G=")].copy()
    g_annual["G"] = g_annual["strategy_name"].str.split("=").str[-1].astype(int)
    g_annual = g_annual.sort_values("G")
    top_intraday = temporal_by_minute.sort_values("normalized_attention", ascending=False).head(10).sort_values("minute_start_in_day")
    lines = [
        "# S&P500全年实验分析",
        "",
        "本实验未执行 sequential OOS weekly retrain。本实验没有周五重训。所有结果均为 `strict_fixed_oos`。Alpha 模型仅为 Minute Transformer + Cross-Stock Attention。未使用 auction 数据、Auction Encoder、Fusion 或 Gate。",
        "",
        "## 1. 数据覆盖",
        "",
        "- 原始数据覆盖 2024-05 至 2026-05；2026-05 的源文件只覆盖至 2026-05-08。",
        f"- 可用 Train 范围：`{train_range}`；各月均使用完整 12 个月。",
        f"- 可用 Validation 范围：`{validation_range}`；各 Test 月前一自然月用于 checkpoint/G 选择。",
        f"- 有效 Test 范围：`{test_range}`；共 `{int(pred['date'].nunique())}` 个交易日。",
        f"- 纳入汇总月份：`{', '.join(valid_months)}`。",
        f"- 剔除月份：`{json.dumps(excluded, ensure_ascii=False)}`。",
        "- universe_type 为 `data_available_equity_universe`。原始交付没有 point-in-time 历史 S&P 500 membership，因此存在 survivorship bias，不能把该代理池表述为无偏的历史指数成分回放。",
        "",
        "## 2. Alpha 总体 OOS 表现",
        "",
        f"- 平均 IC / ICIR：`{_fmt(np.nanmean(ic))}` / `{_fmt(np.nanmean(ic) / (np.nanstd(ic) + 1e-12))}`。",
        f"- 平均 RankIC / RankICIR：`{_fmt(np.nanmean(ric))}` / `{_fmt(np.nanmean(ric) / (np.nanstd(ric) + 1e-12))}`。",
        f"- MSE / MAE：`{_fmt(pred_mean.get('MSE'))}` / `{_fmt(pred_mean.get('MAE'))}`。",
        f"- Top-K 平均收益 / 超额 / hit rate：`{_fmt(pred_mean.get('topk_mean_true_return'))}` / `{_fmt(pred_mean.get('topk_excess_return_vs_universe'))}` / `{_fmt(pred_mean.get('topk_hit_rate'))}`。",
        "",
        _md_table(prediction_monthly, ["test_month", "IC", "RankIC", "MSE", "MAE", "topk_mean_true_return", "topk_excess_return_vs_universe", "topk_hit_rate"]),
        "",
        "## 3. Portfolio 策略总览",
        "",
        _md_table(performance, strategy_cols),
        "",
        "## 4. SS-FM 主线",
        "",
        "Standard FM → SS-FM → SS-FM + PPO 是主比较链，GRPO 作为并列对照。simplex feasibility、return、Sharpe、MDD、turnover、transaction cost 与运行信息见上表及年度 CSV；candidate/reward diversity 与 sampling latency 保留在各月候选表、reward logs 和 G 表中。",
        "",
        _md_table(
            annual_candidates[annual_candidates["strategy_name"].isin(["Standard FM", "SS-FM", "SS-FM + PPO", "SS-FM + GRPO"])],
            ["strategy_name", "candidate_count_G", "candidate_diversity", "reward_diversity", "constraint_violation", "sampling_time_sec"],
        ),
        "",
        "阶段总运行时间（秒）：",
        "",
        _md_table(runtime_phase, ["phase", "seconds"]),
        "",
        "### G 数量实验",
        "",
        "各月默认 G 仅由该月 Validation 平均净收益选择。下表汇总所有预先指定 G 的 Test 表现辅助诊断，不参与选择。",
        "",
        _md_table(g_annual, ["G", "candidate_diversity", "reward_diversity", "constraint_violation", "sampling_time_sec"]),
        "",
        "## 5. Alpha 解释",
        "",
        "唯一 Alpha 路径使用历史分钟 OHLCV/技术特征、Temporal Transformer 和 Cross-Stock Attention。以下统计均来自 Validation 选定且在 Test 月冻结的 checkpoint。",
        "",
        f"- 历史 patch attention entropy / concentration：`{_fmt(attention_mean.get('temporal_attention_entropy'))}` / `{_fmt(attention_mean.get('temporal_attention_concentration'))}`。",
        f"- Cross-stock attention entropy / concentration：`{_fmt(attention_mean.get('cross_stock_attention_entropy'))}` / `{_fmt(attention_mean.get('cross_stock_attention_concentration'))}`。",
        f"- 历史 minute mask rate 均值/最大值：`{_fmt(attention_all['temporal_mask_rate'].mean())}` / `{_fmt(attention_all['temporal_mask_rate'].max())}`；横截面 mask rate 均值/最大值：`{_fmt(attention_all['cross_stock_mask_rate'].mean())}` / `{_fmt(attention_all['cross_stock_mask_rate'].max())}`。",
        f"- 因果 Top-K 中因目标日开盘不可用而在执行时置零的股票记录：`{selected_execution_exclusions}/{selected_count}`；剩余股票重新归一化，计划权重和执行权重均已保留。",
        "",
        "历史交易日注意力（`history_day_index=0` 为窗口最早日）：",
        "",
        _md_table(temporal_by_day, ["history_day_index", "normalized_attention"]),
        "",
        "注意力最高的日内 30 分钟 patch 起点：",
        "",
        _md_table(top_intraday, ["minute_start_in_day", "normalized_attention"]),
        "",
        "有效股票池规模敏感性（月度四分组）：",
        "",
        _md_table(universe_sensitivity, ["universe_size_group", "months", "mean_universe_size", "mean_IC", "mean_RankIC", "mean_topk_excess"]),
        "",
        "输入特征组：",
        "",
        _md_table(feature_groups, ["feature_group", "feature_count"]),
        "",
        "注意力权重用于解释模型关注的历史时段和股票关系，不等同于单特征因果重要性；本实验未用 Test 标签拟合额外的特征归因器。分钟缺失通过 mask 显式处理；横截面 mask rate 为零，但历史分钟 mask 仍非零，股票池规模变化也会影响横截面 attention 与指标稳定性。",
        "",
        "## 6. 风险和局限",
        "",
        "- 历史 S&P 500 membership 不可得，代理股票池存在 survivorship/selection bias。",
        "- 数据覆盖与 Test 月数量有限，2026-05 还是不完整月份；分钟缺失通过 mask 显式处理，但仍可能影响结果。",
        "- 交易成本为固定费率，未完整建模开盘冲击、容量、滑点和成交概率；回测假设能在目标日开盘执行。",
        "- 深度模型、生成策略与多组 G 比较存在过拟合和多重比较风险；RL 结果依赖 reward 定义与 Train-only robust normalization。",
        "- 不进行周五重训降低了 Test 污染风险，但也牺牲月内适应性。strict fixed OOS 的优点是边界清晰，局限是面对 regime shift 时模型整月固定。",
        "- Test-period model update 数为零；所有模型、normalizer、风险状态与策略参数在对应 Test 月内冻结。",
        "",
    ]
    (final / "S&P500全年实验分析.md").write_text("\n".join(lines), encoding="utf-8")
    return {"valid_months": valid_months, "excluded": excluded}
