# CSI500 实验执行状态

## 设定

- Alpha：**冻结复用** `results_alpha_prediction_ablation` 的 `transformer_gated_residual_auction` BEST
  - 含 **9 维集合竞价** Gate Residual Fusion
  - **不重训 SFT**
- Store：AblationStore F=62，同日 `log(C/O)`，分钟窗 `M_{D-10:D-1}`
- OOS：仅 `strict_fixed_oos`，Top30，G∈{8..128}
- 输出：`results/CSI500/strict_fixed_oos/`

## 状态

- **13/13 测试月已完成**（墙钟约 1.63h）
- 全年报告：[`strict_fixed_oos/final_summary/CSI500全年实验分析.md`](strict_fixed_oos/final_summary/CSI500全年实验分析.md)
- 分析树：`strict_fixed_oos/analysis/`（含 `全年总览/` 图）

## 备注

- 跑时泄漏审计曾因 ablation `clocks` 缺 `decision_timestamp` 把月份标成 failed；回测 CSV 完整。适配器已补齐时钟字段。
- `final_summary` 中 `performance_summary.csv` / `backtest_daily.csv` 已按 `(date, model)` 去重（n_days=245）。
