# S&P500 实验执行状态

## 设定

- Alpha：`standard_transformer`（分钟 + Cross-Stock Attention）
- OOS：仅 `strict_fixed_oos`，配权=全股池
- Test：2025-06 … 2026-05

## RL-BC 复跑（已完成）

- 冻结原月 `gen_ssfm_all.pt`，**不重训** Alpha / SS-FM
- `w_turnover=w_smooth_mdd=0.05`；`bc_coef=0.2`；`prev_mode=best_reward`；`noise_std=0.25`；`epochs=3`
- 中间产物：`results/S&P500/strict_fixed_oos_rl_bc/`（约 2.47h）
- 已写回正本 RL 日收益并刷新全年报告

## 报告

- [`strict_fixed_oos/final_summary/S&P500全年实验分析.md`](strict_fixed_oos/final_summary/S&P500全年实验分析.md)
