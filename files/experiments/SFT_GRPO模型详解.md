# Walk-Forward 实验：SFT 与 GRPO 模型技术详解

> 文档基于当前代码库（`files/` 目录）整理，涵盖 walk-forward 实验（含 `walkforward_longonly_r30_semiannual`、`walkforward_longonly_r100_semiannual` 等）实际调用的训练与推理路径。
>
> 生成时间：2026-05-22

---

## 1. 总体流程

每个 fold 按以下顺序执行（见 `walkforward.py::run_walkforward`）：

```
原始 1min CSV
    ↓  hf_data.build_hf_dataset
特征 x (T,N,F) + SFT标签 y + 前向收益 r + 暴露因子
    ↓  train_sft (SFT)
best_sft_model.pt  ——  模仿线性因子 teacher 的排序
    ↓  calibrate_reward (auto 模式) 或固定 empirical 权重
reward_weights.json
    ↓  train_fold_grpo (GRPO)
grpo_model.pt  ——  在 SFT 初始化上，用 RL 目标微调打分器
    ↓  backtest_fold_oos
oos_bar_returns.csv → 拼接 walkforward 曲线
```

**关键设计**：SFT 与 GRPO **共用同一个** `StockTemporalTransformer` 网络；GRPO 不更换结构，只改变优化目标与动作空间（通过打分 → 组合权重）。

---

## 2. 共享骨干网络：`StockTemporalTransformer`

**源码**：`files/stock_transformer.py`

### 2.1 任务定义

| 项目 | 说明 |
|------|------|
| 输入 | `x`: `(N, L, F)` 或 batch `(B, N, L, F)` |
| 输出 | `scores`: `(N,)` 或 `(B, N)`，经 Sigmoid 约束在 **[0, 1]** |
| 含义 | 每个 bar 对 N 只股票的**横截面相对打分**（越高越倾向做多/Top-K） |

其中：
- **N** = 股票池大小（random30=30，random100=100，Top30=30 等）
- **L** = lookback = **30** 根 1min bar（约 30 分钟历史）
- **F** = feat_dim = **11** 维特征

**参数量**：F=11 时约 **72,129** 个可训练参数（实验 log 中可见）。

### 2.2 两阶段 Transformer 结构

```
输入 x: (B, N, L, F)
         │
         ▼ reshape → (B·N, L, F)
    ┌─────────────────────────────────────┐
    │ Stage 1: 时序编码（每只股票独立）      │
    │  • input_proj: Linear(F→64)+LN+GELU │
    │  • Learnable PE (max_len=30)        │
    │  • TransformerEncoder × 1 layer     │
    │    - d_model=64, nhead=4            │
    │    - ffn_dim=128, GELU, norm_first  │
    │  • 取最后一根 bar 的 token 表示      │
    └─────────────────────────────────────┘
         │ h: (B, N, 64)
         ▼
    ┌─────────────────────────────────────┐
    │ Stage 2: 横截面编码（N 只股票互 attend）│
    │  • TransformerEncoder × 1 layer     │
    │    （股票维度作为 sequence length）   │
    └─────────────────────────────────────┘
         │
         ▼
    output_head: LN → Linear(64→32) → GELU → Dropout
                 → Linear(32→1) → Sigmoid
         │
         ▼
    scores: (B, N) ∈ [0,1]
```

**设计要点**：
1. **Stage 1** 在 `(L)` 维度做 self-attention，提取单只股票短窗时序模式。
2. 池化策略：只用 **最后一根 bar**（`h[:, -1, :]`），代表「当前时刻」状态。
3. **Stage 2** 在 `(N)` 维度做 self-attention，让模型感知同 bar 其他股票的相对信息（纯 cross-sectional context）。
4. 输出 Sigmoid 使分数有界，便于 BCE 监督与后续 Top-K 排序。

### 2.3 默认超参（SFT / GRPO / Backtest 一致）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `d_model` | 64 | 隐层维度 |
| `num_heads` | 4 | 注意力头数（每头 16 维） |
| `ffn_dim` | 128 | FFN 中间层 |
| `dropout` | 0.1 | Dropout |
| `max_lookback` | 30 | PE 与输入窗长上限 |

权重初始化：Linear 用 Xavier uniform；PE embedding 用 N(0, 0.02)。

---

## 3. 数据管道与 Label 设计

**源码**：`files/hf_data.py`

### 3.1 原始数据

- 来源：CSI500 成分股 **1 分钟 OHLCV** CSV（`files/csi500/2025/` + 可选 `2026_1min/` 合并）
- 对齐：多股票统一到公共时间戳索引
- 过滤：去掉 **09:30** 与 **15:00** 集合竞价 bar（每交易日有效 bar ≈ **240** 根）
- 年化 bar 数：`BARS_PER_YEAR = 240 × 244 = 58,560`（`backtest.py`）

### 3.2 11 维因果特征（F=11）

所有原始特征先按 bar **横截面 z-score**（跨 N 只股票），再堆叠：

| # | 名称 | 计算（因果，无未来信息） |
|---|------|--------------------------|
| 1 | `ret_1` | 1-bar 对数收益 |
| 2 | `ret_5` | 5-bar 对数收益 |
| 3 | `ret_30` | 30-bar 对数收益 |
| 4 | `vwap_gap` | log(close / vwap_5) |
| 5 | `range_norm` | (high-low) / 前 bar close |
| 6 | `rv_30` | 30-bar 滚动 realized variance |
| 7 | `vol_z` | 成交量相对 30-bar 均值/std |
| 8 | `turn_z` | 换手率相对 30-bar 均值/std |
| 9 | `tod_sin` | 日内时间 sin 编码 |
| 10 | `tod_cos` | 日内时间 cos 编码 |
| 11 | `bars_since_open` | 距开盘 bar 数 / 240 |

额外暴露因子（**仅 GRPO reward 惩罚用**，不进模型输入）：
- `size_z`：log(amount) 横截面 z-score
- `beta_z`：30-bar rolling beta 相对等权市场的 z-score

### 3.3 SFT Label（Teacher）

**函数**：`_linear_factor_label`

Teacher 为**手工线性因子复合**，不含未来信息：

```
composite_t,n = mean( -ret_5, -ret_30, -vol_z, -rv_30 )   # 各因子已是 CS z-score
rank_t,n      = argsort(composite) 的归一化秩 / (N-1)     →  ∈ [0, 1]
```

**经济含义（高 label = 更倾向做多）**：

| 因子 | 符号 | 偏好 |
|------|------|------|
| 短期反转 | `-ret_5` | 近期跌得多 → 分高 |
| 长期反转 | `-ret_30` | 同上，更长窗口 |
| 低量 | `-vol_z` | 异常低成交量 → 分高 |
| 低波 | `-rv_30` | 低 realized vol → 分高 |

Label 是 **rank-normalized** 到 [0,1]，而非原始 z-score；SFT 学的是**排序结构**。

### 3.4 GRPO 用的 Ground-Truth 收益

**函数**：`_forward_returns`

```
r_t,n = log(close_{t+H}) - log(close_t)     H = horizon = 5 bar
```

- 每个 bar 对应未来 **5 根 bar** 的实现 log return
- GRPO episode 中 `r_episode: (M, N)` 即 M 个连续 bar 的前向收益
- **注意**：label `y` 与 `r` 不同源——SFT 模仿因子 teacher，GRPO 直接优化组合在 `r` 上的表现

### 3.5 样本构造

**SFT 样本**（`HFSFTDataset`）：
```
x: (N, L, F)  = x_seq[t-L+1 : t+1] 转置
y: (N,)       = sft_label[t]
有效 t ∈ [valid_start, valid_end)
valid_start = max(L-1, 30)
valid_end   = T - H
```

**GRPO Episode**（`HFEpisodeDataset`）：
```
x_episode: (M, N, L, F)   M = episode_len = 12 连续 bar
r_episode: (M, N)
size_z / beta_z: (M, N)   可选，供 reward 惩罚
```

### 3.6 时间切分（防泄漏）

**函数**：`split_train_val_test(bundle, val_ratio, test_ratio, embargo)`

```
train | --embargo(H)-- | val | --embargo(H)-- | test
```

| 阶段 | SFT 默认 | GRPO (walkforward) |
|------|----------|---------------------|
| val_ratio | 0.20 | 0.25 |
| test_ratio | 0.20 | 0.10 |
| embargo | H=5 bars | H=5 bars |

SFT 选 **val rank_ic 最高** 的 checkpoint；GRPO 训练集 episode 来自 train split。

---

## 4. SFT（Supervised Fine-Tuning）阶段

**源码**：`files/train_sft.py`  
**Walk-forward 调用**：每 fold 独立 SFT（非 warm_chain 模式），`epochs=4`

### 4.1 训练配置（实验实际值 vs 默认）

| 参数 | 实验 (walkforward) | SFTConfig 默认 |
|------|-------------------|----------------|
| `epochs` | **4** | 8 |
| `batch_size` | 16 | 16 |
| `lr` | 5e-4 | 5e-4 |
| `weight_decay` | 1e-2 | 1e-2 |
| `warmup_ratio` | 0.05 | 0.05 |
| `grad_clip` | 1.0 | 1.0 |
| `mse_weight` | 0.5 | 0.5 |
| `rank_weight` | 0.2 | 0.2 |
| 优化器 | AdamW (β1=0.9, β2=0.999) | 同左 |
| LR 调度 | Linear warmup + cosine decay | 同左 |

半年度 fold（~146 交易日 ≈ 35,040 bar）典型样本量：**train ≈ 20,907**（log 可见，随 fold 略有变化）。

### 4.2 损失函数

**总损失**（`sft_loss`）：

\[
\mathcal{L}_{\text{SFT}} = \underbrace{\text{BCE}(s, y)}_{\text{点估计}} + 0.5 \cdot \underbrace{\text{MSE}(s, y)}_{\text{回归}} + 0.2 \cdot \underbrace{\mathcal{L}_{\text{pairwise}}(s, y)}_{\text{排序}}
\]

其中 `s = clamp(scores, ε, 1-ε)`，`y = clamp(labels, 0, 1)`。

**Pairwise Rank Loss**（`pairwise_rank_loss`）：

对任意股票对 (i, j)：
```
若 y_i > y_j  →  希望 s_i > s_j
loss_ij = softplus( -(s_i - s_j) * sign(y_i - y_j) )
```
只对 label 不相等的对计算，平均。

**直觉**：BCE 拟合 [0,1] 绝对值；MSE 加强数值精度；pairwise 直接优化排序一致性。

### 4.3 训练指标

| 指标 | 定义 |
|------|------|
| IC | batch 内 Pearson(scores, labels)，对 batch 维平均 |
| Rank IC | Pearson(rank(scores), rank(labels)) |
| 模型选择 | **验证集 Rank IC 最大** → `best_sft_model.pt` |

实验 log 典型值（in-sample，teacher 较简单）：val rank_ic ≈ **0.997+**（接近完美拟合 teacher 排序）。

### 4.4 产出文件

```
folds/fold_XXX/
  best_sft_model.pt    # state_dict，供 GRPO 初始化
  last_sft_model.pt
  sft_log.csv          # epoch, train/val loss, ic, rank_ic, lr
```

---

## 5. GRPO 阶段（Walk-Forward 实验路径）

**源码**：`files/walkforward.py`（`train_fold_grpo`, `grpo_step_longshort`）  
**Reward**：`files/reward_model.py`（`episode_reward_v2`）

> **注意**：`files/train_grpo_trl.py` 中另有一套基于 **Dirichlet 单纯形采样** 的 GRPO 实现，**walk-forward 实验未使用**；实验用的是 **Gaussian 扰动 score + Top-K 离散权重** 路径。

### 5.1 训练配置（实验实际值）

| 参数 | 实验值 | 说明 |
|------|--------|------|
| `epochs` | **2** | 每 fold GRPO epoch 数 |
| `batch_size` | **2** | episode batch |
| `num_generations` | **4** | 每 episode 采样 G=4 条 rollout |
| `episode_len` | **12** | 每条 rollout 连续 12 bar |
| `lr` | **1e-5** | 远低于 SFT |
| `weight_decay` | 1e-2 | |
| `warmup_ratio` | 0.05 | |
| `grad_clip` | 1.0 | |
| `epsilon` | 0.2 | PPO clip |
| `beta` | 0.04 | KL(ref‖policy) 系数 |
| `scale_rewards` | `"group"` | 组内标准化 advantage |
| 初始化 | `best_sft_model.pt` | 同 fold SFT |
| Reference policy | SFT 权重的 **frozen copy** | GRPO 全程不更新 |

每 epoch 步数 ≈ `ceil(n_train_episodes / 2)`；半年度 fold 典型 **~11,325 steps/epoch × 2 epoch ≈ 22,650 steps**（与 `grpo_log.csv` 行数一致）。

### 5.2 动作空间：从 Score 到组合权重

**源码**：`files/portfolio.py`

GRPO 中 policy 输出 `scores: (M, N)`，对 **每个 bar** 独立构造权重：

#### Long-Only Top5（当前 R30/R100 实验）

```
portfolio_mode = "long_only"
top_k = 5, bottom_k = 0, long_gross = 1.0

按 score 降序排名：
  Top 5 股票：各 +20%（共 100% 多头）
  其余：0
```

#### Long-Short Top5/Bot5（如 walkforward_202604）

```
long_gross = 0.5, short_gross = 0.5
Top 5: +10% each；Bottom 5: -10% each； dollar-neutral
```

#### Rollout 随机性（探索）

```python
noisy_scores[g] = scores + N(0, score_noise_std²),  score_noise_std = 0.05
weights[g] = scores_to_portfolio_weights(noisy_scores[g], ...)
```

- 对 G=4 条 generation 各采样一组高斯噪声
- 噪声后的 score 再 Top-K 离散化 → 不同 rollout 得到不同组合
- **Log-prob**：因子化高斯密度 `_gaussian_log_prob`，对 N 维 score 求和

### 5.3 Episode Reward（多目标）

**函数**：`episode_reward_v2`（`reward_model.py`）

输入：
- `weights`: (G, M, N)
- `realized_returns`: (M, N)
- 可选 `size_z`, `beta_z`: (M, N)

**分量**：

| 分量 | 公式 | 方向 |
|------|------|------|
| IC | 每 bar Pearson(w, r) 对 M 平均 | 越大越好 |
| Sharpe | mean(port_ret) / std(port_ret)，port_ret = Σ w·r | 越大越好 |
| Turnover | mean_t ‖w_t - w_{t-1}‖₁ | 惩罚 |
| Downside | mean(min(port_ret,0)²) | 惩罚 |
| Max DD | max drawdown of cumsum(port_ret) | 惩罚 |
| Size exposure | \|corr(w, size_z)\| 均值 | 惩罚 |
| Beta exposure | \|corr(w, beta_z)\| 均值 | 惩罚 |
| HHI | Herfindahl of \|w\| / sum(\|w\|) | 惩罚集中度 |

**总 reward**：

\[
R = w_{ic}\cdot IC + w_{sh}\cdot Sharpe - w_{to}\cdot TO - w_{ds}\cdot DS - w_{dd}\cdot MDD - w_{sz}\cdot |corr_{size}| - w_{\beta}\cdot |corr_{\beta}| - w_{hhi}\cdot HHI
\]

### 5.4 Auto-Learn Reward 权重

**模式**：`reward_mode = "auto"`（R30/R100 实验）

1. 用 **SFT 模型**（尚未 GRPO）在 **val episodes** 上生成确定性权重
2. `search_reward_weights` 网格搜索：

| 搜索维度 | 候选值 |
|----------|--------|
| `ic_weight` | 0.5, 1.0, 1.5 |
| `sharpe_weight` | 0.5, 0.8, 1.2 |
| `penalty`（turnover/downside/max_dd 等） | 0.2, 0.4, 0.6 |

惩罚项派生：
```
turnover_weight = downside_weight = max_dd_weight = pen_w
size_exp_weight = beta_exp_weight = pen_w * 0.5
hhi_weight = pen_w * 0.25
```

3. 选 **验证集平均 composite score 最高** 的组合 → 写入 `reward_weights.json`

**R30 fold0 实测学习结果**：
```json
{
  "ic_weight": 1.5,
  "sharpe_weight": 1.2,
  "turnover_weight": 0.2,
  "downside_weight": 0.2,
  "max_dd_weight": 0.2,
  "size_exp_weight": 0.1,
  "beta_exp_weight": 0.1,
  "hhi_weight": 0.05
}
```

**Empirical 模式**（部分旧实验）：固定 `ic=1.0, sharpe=0.8, penalties=0.3/0.4/0.5/...`，无网格搜索。

### 5.5 GRPO 损失（TRL 兼容形式）

**函数**：`grpo_loss`（`train_grpo_trl.py`，walkforward 复用）

对每个 episode，G 条 rollout 的 **episode 级 log-prob**（M bar 求和）：

**Advantage**（组内标准化）：
```
A_g = (R_g - mean(R)) / (std(R) + ε)     scale_rewards = "group"
```

**Policy gradient（PPO clip）**：
```
ratio_g = exp(log π_new - log π_old)
L_pg = -mean( min(ratio·A, clip(ratio, 1-ε, 1+ε_high)·A) )
```

**KL 惩罚**（k3 estimator，对 reference policy）：
```
L_kl = mean( exp(log π_ref - log π_new) - (log π_ref - log π_new) - 1 )
```

**总损失**：
```
L = L_pg + β · L_kl,    β = 0.04
```

Reference policy 固定为 SFT 初始化副本，防止 GRPO 偏离过远。

### 5.6 产出文件

```
folds/fold_XXX/
  reward_weights.json
  grpo_model.pt
  grpo_log.csv    # epoch, step, loss, reward_mean, ic_mean, sharpe_mean, max_dd_mean, lr
```

---

## 6. OOS 回测（推理阶段）

**源码**：`walkforward.py::backtest_fold_oos` + `backtest.py::simulate`

### 6.1 推理逻辑

与 GRPO 训练 **不同**：OOS **无噪声**，直接用模型 score → Top-K 权重：

```python
scores = model(x_t)                              # (N,)
w = scores_to_portfolio_weights(scores, ls_cfg, portfolio_mode)  # 确定性 Top5
port_ret_t = dot(w, fwd_return_t)
```

### 6.2 交易成本与再平衡

| 参数 | 值 |
|------|-----|
| `fee_bps` | 5 bps（单边） |
| `rebalance_every` | 5 bar（= horizon H） |
| 成本 | `fee * ‖w_t - w_{t-1}‖₁`（再平衡时） |

### 6.3 Baseline

| baseline_mode | 组合 |
|---------------|------|
| `long_only`（R30/R100） | N 只股票 **等权 100% 多头** |
| `ls_equal`（部分 LS 实验） | 固定 index Top5 多 + Bot5 空，50/50 gross |

---

## 7. Walk-Forward 调度（半年度实验）

**schedule_mode**：`monthly_semiannual`

| Fold | 训练窗（约） | OOS |
|------|-------------|-----|
| 0 | 2025-06 ~ 2025-12（~146 日） | 2026-01 |
| 1 | 2025-07 ~ 2026-01 | 2026-02 |
| 2 | 2025-08 ~ 2026-02 | 2026-03 |
| 3 | 2025-09 ~ 2026-03 | 2026-04 |

**R30 vs R100 唯一区别**：`stock_ids` 数量（30 vs 100，seed=42 随机抽样）；其余 hyperparam 相同。

**Long-only 实验统一设置**：

```python
WalkForwardConfig(
    portfolio_mode="long_only",
    baseline_mode="long_only",
    top_k=5, bottom_k=0,
    reward_mode="auto",
    lookback=30, horizon=5,
    sft_epochs=4, grpo_epochs=2,
    grpo_generations=4, episode_len=12,
    schedule_mode="monthly_semiannual",
    oos_stitch_start="2026-01", oos_stitch_end="2026-04",
)
```

---

## 8. 与 standalone `train_grpo_trl.py` 的差异

| 项目 | Walk-forward 实验 | `train_grpo_trl.py` 独立脚本 |
|------|---------------------|------------------------------|
| 动作分布 | Gaussian 扰动 score → Top-K | Dirichlet(α=1+10·score) 连续权重 |
| Reward | `episode_reward_v2` 8 分量 | `episode_reward` 3 分量 (IC+Sharpe-DD) |
| Reward 权重 | auto-learn / empirical | 固定 cfg 权重 |
| 组合 | Top-K 离散 | 单纯形连续采样 |
| 使用场景 | **所有 walkforward 实验** | Demo / 遗留路径 |

---

## 9. 代码索引

| 模块 | 文件 | 职责 |
|------|------|------|
| 网络 | `stock_transformer.py` | StockTemporalTransformer |
| 数据 | `hf_data.py` | 特征、label、episode 数据集 |
| SFT | `train_sft.py` | 监督训练与损失 |
| GRPO (WF) | `walkforward.py` | fold 调度、GRPO step、回测 |
| GRPO (TRL) | `train_grpo_trl.py` | Dirichlet 版 GRPO（非 WF 主路径） |
| Reward | `reward_model.py` | 多目标 reward + 网格搜索 |
| 组合 | `portfolio.py` | Top-K / LS 权重映射 |
| 回测 | `backtest.py` | simulate、年化指标 |

---

## 10. 参数量与计算复杂度直觉

- **参数量 ~72K**：极小模型，CPU 可训；瓶颈在 **GRPO 逐步 rollout**（每 step：forward × (B episodes × G generations × M bars)）。
- **GRPO step 数**主要由 **train episode 数量**决定，与 N（30 vs 100）**线性相关**（每 forward 处理 N 只股票）。
- **SFT step 数**由 bar 数决定，与 N 线性相关；半年度窗 >> 3 日窗，故 SFT/GRPO 均远慢于早期 3d→3d 实验。

---

## 11. 设计总结

1. **SFT** 将 Transformer 校准为「线性多因子 reversal/low-vol teacher」的排序器，提供稳定初始化。
2. **GRPO** 在同一打分网络上，用 **带噪 Top-K 组合** 作为 action，以 **IC + Sharpe − 多项风险惩罚** 为 reward，PPO-clip + KL 约束微调。
3. **Auto reward** 在 val 上为每个 fold 自适应 penalty 强度，避免手工调参。
4. **OOS** 用确定性 Top-K，与 GRPO 探索期分离，评估真实可交易规则。
5. 横截面 Transformer 让每只股票打分依赖同 bar 其他股票 context，适合 relative ranking 任务。
