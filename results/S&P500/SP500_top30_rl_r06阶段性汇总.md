# SP500 Top30 RL reward-r06 阶段性汇总（2025-06 … 2026-03）

- **性质**：部分月份已跑完的中期报告，**不是**最终全年报告。
- **已完成**：2025-06, 2025-07, 2025-08, 2025-09, 2025-10, 2025-11, 2025-12, 2026-01, 2026-02, 2026-03（10/12）。
- **设定**：Top30；G=32；SS-FM=Meta（样本∪Teacher→pred_utility）；PPO=cand 特征；GRPO=SSFM init；`bc_coef=0`。
- **Reward**：`r = 0.6·z(net) + 0.3·z(因果 Sharpe) − 0.01·z(换手) − 0.01·z(smooth MDD)`。
- **单利 Σ**；已拼接约 **209** 日。
- 最终全年章节名：`Top30 G=32 RL reward-r06：SS-FM vs PPO / GRPO`（与本文件不重名）。

## 对照表

| model | total_net_return（单利） | ann_return | sharpe | max_drawdown | mean_turnover | n_days |
| --- | --- | --- | --- | --- | --- | --- |
| SS-FM G=32 Meta | 0.2918 | 0.3519 | 1.8508 | 0.1187 | 0.7950 | 209 |
| SS-FM+PPO (cand) G=32 r06 | 0.2111 | 0.2545 | 1.5100 | 0.0720 | 0.1698 | 209 |
| SS-FM+GRPO (SSFM init) G=32 r06 | 0.0476 | 0.0574 | 0.4495 | 0.0707 | 0.1753 | 209 |

| month | SS-FM G=32 Meta | SS-FM+PPO (cand) G=32 r06 | SS-FM+GRPO (SSFM init) G=32 r06 |
| --- | --- | --- | --- |
| 2025-06 | 0.0013 | 0.0752 | 0.0207 |
| 2025-07 | 0.0630 | -0.0084 | -0.0134 |
| 2025-08 | 0.1250 | 0.1047 | 0.0578 |
| 2025-09 | 0.0021 | 0.0171 | 0.0105 |
| 2025-10 | -0.0164 | 0.0698 | 0.0075 |
| 2025-11 | 0.0334 | 0.0103 | 0.0008 |
| 2025-12 | 0.0131 | -0.0339 | -0.0300 |
| 2026-01 | 0.1214 | 0.0335 | 0.0333 |
| 2026-02 | 0.0271 | -0.0360 | 0.0201 |
| 2026-03 | -0.0782 | -0.0214 | -0.0596 |

### 累计净值（单利）

![累计净值（单利）](strict_fixed_oos_top30_rl_r06/partial_summary/figures/partial_rl_r06_cumulative_return.png)

### 累计净收益柱（单利）

![累计净收益柱（单利）](strict_fixed_oos_top30_rl_r06/partial_summary/figures/partial_rl_r06_total_return.png)

### Sharpe

![Sharpe](strict_fixed_oos_top30_rl_r06/partial_summary/figures/partial_rl_r06_sharpe.png)

### 最大回撤

![最大回撤](strict_fixed_oos_top30_rl_r06/partial_summary/figures/partial_rl_r06_max_drawdown.png)

### 日均换手

![日均换手](strict_fixed_oos_top30_rl_r06/partial_summary/figures/partial_rl_r06_turnover.png)

### 分月收益（单利）

![分月收益（单利）](strict_fixed_oos_top30_rl_r06/partial_summary/figures/partial_rl_r06_monthly_return.png)

## 简要结论（阶段性）

- 单利：SS-FM Meta `0.2918`；cand-PPO `0.2111`（负）；GRPO `0.0476`（负）。
- 换手：SS-FM `0.7950` → PPO `0.1698` / GRPO `0.1753`。
- 覆盖至 `2026-03`；剩余月份跑完后写入全年报告对应章节。
