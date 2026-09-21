# CSI500 数据契约

- 分钟：`/home/tom/Documents/csi500/{2024,2025,2026_…/2026/1分钟}`
- **集合竞价（必须）**：`/home/tom/Documents/csi500/集合竞价_{2024,2025,2026_截止到8月31}`（9 维，09:25 可用）
- 成分：`files/csi500_constituents.csv`
- Alpha Store：复用 ablation 缓存 `alpha_prediction_ablation/ablation_data/cache/`（T=565, N=500, **F=62**）

## 协议（本实验）

| 项 | 设定 |
| --- | --- |
| Alpha | **冻结复用** `results_alpha_prediction_ablation` 的 `transformer_gated_residual_auction` BEST（含竞价门控，**不重训**） |
| 分钟窗 | `M_{D-10:D-1}`，不含 D 日盘中 |
| 竞价 | D 日 09:25 九维 Auction → Gate Residual Fusion |
| 标签 | `y = log(Close_D / Open_D)` |
| 选股 | 按 ŷ Top30 |
| OOS | **仅** `strict_fixed_oos` |
| Test | 2025-05 … 2026-05（13 月） |
| Teacher | MVO / MaxSharpe / RiskParity（Kelly 仅 baseline） |
| 生成 / RL / G | MLP·Gaussian·FM·Diffusion·SS-FM；PPO/GRPO；G∈{8..128} |

## 输出

`results/CSI500/strict_fixed_oos/`
