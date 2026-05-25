# Walk-forward GRPO vs Equal-Weight：差距为何这么大？

> 实验目录：`files/experiments/walkforward_202604/`  
> 对比图：`backtest/walkforward_curves.png`  
> 数据截止：2026-05-08（原始数据无 05-11～05-14）

---

## 1. 结论先行

**GRPO +30.7% 并不差，但 Equal-Weight (EW) +69.3% 在这个设定下“天然占优”。**

差距大的**首要原因不是模型完全失效**，而是：

| 维度 | GRPO 策略 | Equal-Weight 基准 |
|------|-----------|-------------------|
| 仓位结构 | **多空对冲**（Top5 做多 + Bottom5 做空，净暴露≈0） | **100% 多头**（30 只等权） |
| 收益来源 | 横截面 **相对排序**（alpha） | **市场方向 + 分散**（beta） |
| 股票池 | 2026 年 YTD **涨幅最好的 Top30** | 同一池子 |
| 测试区间 | 2026-04-07 ~ 05-08，整体偏 **多头行情** | 同上 |

在「强势动量股 + 上涨市」里，**100% 多头等权几乎必然跑赢 dollar-neutral 多空**——这不是 GRPO 独有现象，是策略类型与基准不匹配。

---

## 2. 实验设定回顾

### 2.1 Walk-forward 流程

- 7 个 fold，每个 fold：**3 天训练 → 3 天 OOS**
- 训练：SFT → GRPO（reward 在验证集上 auto-learn）
- OOS：用该 fold 的 `grpo_model.pt` 预测后 3 天
- 全部中间模型保存在 `folds/fold_XXX_.../grpo_model.pt`

### 2.2 股票池

- CSI500 中按 **2026-01-05 ~ 2026-03-31 累计涨幅** 选 Top30
- 代表：sh600487、sh688525、sz002432 等（见 `universe/top30_ytd2026.csv`）
- 这是**动量/强势**组合，4～5 月继续上涨的 prior 很强

### 2.3 GRPO 仓位构造（Long-Short TopK）

```
Top 5  得分最高 → 各 +10% 多头  （合计 +50% gross long）
Bottom 5 得分最低 → 各 -10% 空头 （合计 -50% gross short）
中间 20 只 → 权重 0
净暴露 ≈ 0（dollar-neutral）
```

EW 基准：30 只各 **+3.33%**，**100% 多头**，无空头。

---

## 3. 观测数据

### 3.1 拼接 OOS 总体（~1 个月，4774 bars）

| 指标 | GRPO (Long-Short) | Equal-Weight |
|------|-------------------|--------------|
| 总收益 | **+30.70%** | **+69.27%** |
| 年化 Sharpe | +6.17 | +12.80 |
| Max Drawdown | 14.37% | （见曲线） |
| 单 bar 平均收益 | +0.587 bps | +1.125 bps |
| Hit rate（bar>0） | 51.9% | 52.8% |

**Hit rate 几乎一样**（~52%），说明 GRPO 并不是“方向全错”，而是**赚的时候赚得少、结构性地放弃了 beta**。

### 3.2 各 Fold 三段 OOS 收益（单 fold 内复利）

| Fold | OOS 日期 | GRPO | EW | GRPO − EW |
|------|----------|------|-----|-----------|
| 0 | 04-07 ~ 04-09 | +9.6% | +21.3% | −11.7% |
| 1 | 04-10 ~ 04-14 | −4.3% | +7.4% | −11.6% |
| 2 | 04-15 ~ 04-17 | +8.9% | +8.8% | ≈0 |
| 3 | 04-20 ~ 04-22 | +7.5% | +17.5% | −10.0% |
| 4 | 04-23 ~ 04-27 | **+10.0%** | **−8.3%** | **+18.3%** ✓ |
| 5 | 04-28 ~ 04-30 | −2.3% | +0.1% | −2.4% |
| 6 | 05-06 ~ 05-08 | −1.0% | +10.8% | −11.7% |

- **7 个 fold 里 GRPO 只在 fold 4 明显跑赢 EW**
- 其余 6 个 fold 多数落后 **~10–12%/3天**，与多空结构在上涨段吃不到 beta 一致
- fold 4 市场分化大，多空排序策略相对受益

---

## 4. 差距大的五个原因（按重要性排序）

### 原因 ①：策略类型不对等 — 多空 vs 纯多头

这是**最根本**的原因。

- EW：市场涨 1%，30 只平均涨，组合约涨 1%（100% 暴露）
- GRPO Long-Short：市场涨 1%，若多空腿涨幅相近，组合收益 ≈ **0**（净暴露为 0）

在 Top30 强势股的上涨市里：

- **多头腿**（Top5）赚钱
- **空头腿**（Bottom5）也在涨 → 做空 **持续亏损**
- 两者部分抵消，只剩横截面 alpha

实测权重结构：

```
GRPO: gross_long = 50%, gross_short = 50%, net ≈ 0
EW:   gross_long = 100%, gross_short = 0%
```

→ EW 吃满 **beta**；GRPO 只赚 **ranking spread**。

### 原因 ②：股票池是“2026 年最强动量股”

Top30 按 YTD 涨幅选取，本身处于**强者恒强**阶段：

- EW 持有全部 30 只上涨股 → 全面受益
- GRPO 做空的 Bottom5 仍是 Top30 里的“相对弱者”，**绝对收益仍可能为正**
- 做空相对弱者 ≠ 做空下跌股，在动量池里空头 leg 是 drag

### 原因 ③：Reward 优化目标与 EW 基准不一致

Auto-learn 在每个 fold 验证集上 grid-search，7 个 fold **全部收敛到同一组**：

```json
{
  "ic_weight": 1.5,
  "sharpe_weight": 1.2,
  "turnover_weight": 0.2,
  "...": "penalty = 0.2 for all risk terms"
}
```

含义：

- 高度奖励 **IC（排序相关性）** 和 **Sharpe**
- **低惩罚** turnover / drawdown / 暴露 → 允许较激进的排序
- 没有任何项直接优化 **“相对 EW 的超额”** 或 **“绝对多头收益”**
- 训练信号是“把权重和收益排序对齐”，不是“跑赢等权基准”

因此 GRPO 学的是**好的 ranker**，不一定是**好的 long-only portfolio manager**。

### 原因 ④：Hit rate 接近，但 payoff 不对称

| 指标 | GRPO | EW |
|------|------|-----|
| Bar 命中率 | 51.9% | 52.8% |

命中率几乎相同 → 模型排序并非随机。

但 GRPO 的 PnL 结构是：

```
portfolio_return = Σ w_i × r_i
                 = 0.5 × (top5 avg ret) − 0.5 × (bottom5 avg ret)
```

当 **top5 和 bottom5 都在涨**，且 bottom5 涨得不少时，alpha 被吃掉。

EW 的 PnL：

```
portfolio_return = (1/30) × Σ r_i  > 0  （整体上涨时）
```

→ **同样的“猜对率”，不同的 payoff 函数**。

### 原因 ⑤：Fold 拼接复利放大视觉差距

7 段 OOS 首尾拼接后做 `(1+r).cumprod()`：

- GRPO 最终 +30.7%（已有 alpha，但被结构限制）
- EW 最终 +69.3%（beta 全程叠加）

绝对差距 ~38.6 pp，部分来自**连续复利**而非单 fold 平均差。

---

## 5. 这不是“GRPO 训练失败”的证据

支持 GRPO 仍有效的信号：

1. **拼接 OOS 仍 +30.7%**，Sharpe +6.17 — 多空策略里不算差
2. **Fold 4** 在 EW 亏损 (−8.3%) 时 GRPO 赚 +10.0% → 排序在分化行情有用
3. **Fold 2** 与 EW 打平 (+8.9% vs +8.8%)
4. Bar 命中率与 EW 相当，说明不是纯噪声

不支持“GRPO 全面优于 EW”：

- 在强势动量池 + 上涨段，**long-only 基准太强**
- 当前对比是 **apples-to-oranges**

---

## 6. 更合理的对比方式（后续实验建议）

若目标是“GRPO 能否打败 EW”，需要至少其一：

| 方案 | 说明 |
|------|------|
| **A. Long-only TopK** | Top5 各 20%，不做空 → 与 EW 同为多头，比排序能力 |
| **B. 多空 + 市场中性基准** | 基准改为「Top5 多头 − Bottom5 等权空头」而非 EW |
| **C. 全市场随机 30 只** | 降低动量池 beta，多空更公平 |
| **D. Reward 加 alpha-vs-EW 项** | `reward += λ × (ret_policy − ret_EW)` |
| **E. 报告 IR / 信息比率** | 对多空策略报告 rank-IC、top-bottom spread，而非 NAV vs EW |

---

## 7. 与 Empirical Reward 重跑的关系

已启动独立实验目录（不与本文档所在 run 混放）：

```
files/experiments/walkforward_202604_empirical/
```

经验值 reward（固定，不做 grid-search）：

| 成分 | 权重 |
|------|------|
| IC | 1.0 |
| Sharpe | 0.8 |
| Turnover 惩罚 | 0.3 |
| Downside 惩罚 | 0.4 |
| MaxDD 惩罚 | 0.5 |
| Size 暴露惩罚 | 0.2 |
| Beta 暴露惩罚 | 0.2 |
| HHI 集中度惩罚 | 0.1 |

**预期**：换 reward 权重可能改善 GRPO 的 **ranking / 风控**，但**无法消除多空 vs 100% 多头的结构性差距**。若 empirical run 仍大幅落后 EW，主因仍是 §4 原因 ①②，而非 reward 系数 alone。

---

## 8. 一句话总结

> **GRPO 在做一个 dollar-neutral 的多空排序策略；EW 在做一个 100% 做多的动量组合。在 2026 年 Top30 强势股的上涨段，后者吃 beta、前者吃 alpha——基准选错了，差距大是预期内的，不是 bug。**

---

*文档生成依据：`walkforward_202604` 实验 `stitched_bar_returns.csv`、`fold_summary.csv`、各 fold `reward_weights.json` 及 `portfolio.py` 权重结构分析。*
