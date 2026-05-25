# 高频截面实验设计（3 日 × 1 分钟 × 中证 500）

本文针对当前仓库的设定：

- **股票池**：`files/csi500/`（中证 500，N ≈ 500）
- **数据**：1 分钟 K 线（OHLCV + 涨跌幅 + 换手率 + 流通股本 + 总股本）
- **训练规模**：3 个交易日（约 720 ~ 726 根 bar）
- **建模骨架**：`StockTransformer`（单层截面 Transformer，输入 N×F → 输出 N 个 [0,1] 分数）+ SFT + GRPO（Dirichlet 组合权重）

写完以后照着第 9 节 "落地配方" 直接跑就行。

---

## 1. 总体目标与建模框架

把高频"选股+组合"问题切成两层：

| 层 | 模型 | 输入 | 输出 | 目标 |
|---|---|---|---|---|
| 截面打分 | StockTransformer | (N, F)：N 只股票当前 bar 的特征 | (N,) 分数 ∈ [0,1] | 让分数与未来 bar 截面收益的"排名/方向"一致 |
| 组合权重 | Dirichlet(α=base+scale·score) | 上一层的分数 | (N,) 权重 ∈ simplex | 用 GRPO 直接最大化交易奖励 |

**为什么是截面而不是单股**：1 分钟数据信噪比极低，单只股票绝对收益预测几乎是噪声，但**横截面相对强弱**（"今天哪些股票比平均强"）的可预测性显著更高且更稳定，这也是为什么模型设计本身就是 cross-sectional Transformer。

**为什么 SFT + GRPO 两段**：

- SFT 阶段把模型先校到一个"分数高 ↔ 未来截面收益高"的合理流形上（监督，梯度稳）。
- GRPO 阶段才真正引入**交易约束**（成本、换手、不可交易日内 stops），让模型在"交易奖励"这个真实目标上微调，不再仅仅最大化 IC。
- 不直接用 RL 从零训：3 天数据搞 RL 必然过拟合。

---

## 2. 数据规模与样本构造

### 2.1 一个交易日有多少根 1 分钟 bar

A 股 1 分钟 K 线（你的 CSV 里时间戳是 bar 起始）：

```
09:30 (开盘集合竞价撮合)
09:31, 09:32, ..., 11:30   → 121 根
13:00, 13:01, ..., 15:00   → 121 根（含 14:57-15:00 收盘集合竞价）
合计：约 242 根
```

3 天 ≈ **726 根 bar**。如果按 "每根 bar 是一个截面样本"，那训练集合就是 726 个 (N, F) → (N,) 的 (input, label) pair —— **这是这套实验唯一的样本数**，远小于我们日频研究的体量，必须用相应的策略来防过拟合（见 §5、§6）。

### 2.2 一个样本 = 一个 (时间戳, 股票池) 截面

形式定义：

```
样本 t  ─────────────────────────────────────────────
  时间戳:  t                       （1 分钟 bar 的开始时间）
  股票池:  P_t  ⊆ CSI500           （在 t 时刻可交易的成分股）
  输入 x:  (|P_t|, F) 的特征矩阵     （仅使用 ≤ t 的信息）
  标签 y:  (|P_t|,)  的截面排名      （来自 (t, t+H] 区间的实现收益）
```

**P_t 必须随时间变化**（停牌、ST、上市/退市、涨跌停锁单等），不能假设 N 恒等于 500。`StockTransformer` 已经支持任意 N，把这个池子动态化即可（见 §7.3）。

### 2.3 lookback W 与预测 horizon H 的选择

固定符号：W = 计算特征用的回看 bar 数；H = 预测的未来 bar 数。

| 选择 | 适用场景 | 注意 |
|---|---|---|
| W = 30, H = 5  | 默认起手；30 分钟回看预测后 5 分钟 | 样本损失 35 根/日 |
| W = 60, H = 1  | 极短期 alpha    | H=1 噪声极大，IC 通常 < 1%，靠 GRPO 打 |
| W = 60, H = 30 | 半小时级别 momentum/反转 | label 跨段时谨慎（见 §5） |

实际可用样本数 ≈ 3 × (242 - W - H)。W=30, H=5 时约 621 个截面。

---

## 3. 特征工程（针对 1 分钟 OHLCV）

CSV 字段：日期, 开, 高, 低, 收, 成交量(股), 成交额(元), 涨跌(元), 涨跌幅(%), 换手率(%), 流通股本(股), 总股本(股)。

下面每一组特征**都先按股票算出原始值，再做截面 z-score 或 rank**——不做截面标准化的 HF 特征基本无法用。

### 3.1 价格 / 收益类（动量与反转混合）

| 特征 | 公式 | 备注 |
|---|---|---|
| `ret_k` | $r_t^{(k)} = \ln(\mathrm{close}_t / \mathrm{close}_{t-k})$，k ∈ {1, 3, 5, 15, 30} | 多 horizon 动量 |
| `vwap_gap` | $\ln(\mathrm{close}_t / \mathrm{vwap}_{[t-W,t]})$ | "近 W 分钟成本-当前价"差 |
| `range_norm` | $(\mathrm{high}_t - \mathrm{low}_t) / \mathrm{close}_{t-1}$ | 单 bar 真实振幅 |
| `gap_open` | $\ln(\mathrm{open}_t / \mathrm{close}_{t-1})$ | 开盘跳空（含日内段间） |

经验：`ret_5` 和 `ret_30` 在 A 股 1 分钟级一般是**反转**信号，`ret_1` 多数时候是噪声；GRPO 阶段会自己学权重。

### 3.2 成交量 / 资金流类

| 特征 | 公式 | 备注 |
|---|---|---|
| `vol_z` | (vol_t - mean(vol_{[t-W,t]})) / std | 异常放量 |
| `turn_z` | 换手率截面 z 化 | 流动性指标 |
| `amount_per_trade` | 成交额 / 估算笔数 | 笔均成交，反映大小单 |
| `signed_vol` | sign(close-open) × vol | "上涨成交"的近似 |
| `flow_imb` | rolling sum of `signed_vol` over W | 资金流动量 |

### 3.3 波动 / 微结构类

| 特征 | 公式 |
|---|---|
| `rv_W` | $\sum_{s=t-W+1}^{t} (r_s)^2$（已实现波动率） |
| `bv_W` | bipower variation $\sum |r_s||r_{s-1}|$（剔跳跃后的波动） |
| `jump_W` | rv - bv（跳跃强度近似） |
| `kyle_lambda` | $|r_t| / \log(1+\mathrm{vol}_t)$ rolling | 价格冲击近似 |
| `acvr_1` | rolling Corr(r_t, r_{t-1}) | 一阶自相关，反转/动量识别 |

### 3.4 日内时间编码（很重要）

3 天 × 240 bar 的样本里，**时间-of-day** 是几乎免费的强解释变量：开盘 5 分钟、午后第一根、收盘前 5 分钟的统计性质完全不同。

- `tod_sin = sin(2π · bar_idx / 242)`、`tod_cos = cos(...)`：周期编码
- `is_open_5 / is_close_5`：前 5 / 后 5 bar 的 0-1 标记
- `bars_since_open`、`bars_to_close`：线性距离

3 天太短不要做日历特征（day-of-week / month）。

### 3.5 截面统计（cross-sectional reduce）

任意 §3.1–§3.3 的特征都可加一份"该股票相对截面的位置"作为额外列：

```python
x_cs[:, i] = (x[:, i] - x[:, i].mean()) / (x[:, i].std() + 1e-8)
x_rank[:, i] = x[:, i].argsort().argsort() / (N - 1)
```

经验：截面 z-score 比绝对值好用很多，但**不能再叠加 BatchNorm**（重复了）。

### 3.6 不要碰的"未来泄漏"陷阱

- 涨跌幅(%) 当根 bar：若用 close-of-bar，下一根 bar 之前不可见，OK；用更细分钟的 close 拼出来的"涨跌"则要小心。
- 换手率：CSV 里给的是当根 bar 内的换手率，用 t 时刻特征时只能引用 ≤ t 的 bar。
- 总股本/流通股本：这俩在日内一般不变，可作为静态特征但**用 t 时刻的值**（除权除息日跳变）。

---

## 4. 标签设计

### 4.1 原始 label：未来 H 根 bar 的实现收益

$$ \tilde{y}_{i,t} = \ln\frac{P_{i, t+H}}{P_{i, t}} $$

**$P$ 用什么价**：

| 价 | 性质 | 何时用 |
|---|---|---|
| `close_{t+H}`              | 简单，但你 t 时刻不能在 close 价交易 | label 估计 |
| `vwap[t+1, t+H]`           | "下一根开盘买进、第 H 根 bar 内卖出" 的近似 | 更接近真实可执行 |
| `(close_{t+H} - close_t) / close_t` | 同上但简单百分比 | 一般场景 |

**强烈推荐用 `vwap`-类标签做 GRPO 阶段**，因为它把"下一根 bar 才能交易"这个约束内化进了标签。

### 4.2 SFT 阶段标签：截面排名归一化到 [0,1]

```python
# 一个截面内：
y_rank = raw_returns.argsort().argsort()           # 0..N-1
y_label = y_rank / (N - 1)                          # ∈ [0, 1]
```

这样：

- 可以直接送给 `train_sft.py` 里既有的 BCE+MSE+pairwise rank loss。
- 截面内不存在"今天大盘整体涨/跌"这种信号，模型只学相对强弱（这是 HF 截面唯一稳定的信号）。

### 4.3 GRPO 阶段奖励：直接是交易收益

`train_grpo.py` 里的 `default_reward_fn` 当前是 `w·labels + entropy_bonus`，但跑 HF 时**应该把 labels 换成真实下一段 vwap 收益**，并加上交易成本与换手罚项。一个干净的初版：

```python
def hf_reward_fn(weights, features, labels, prev_weights, config, **_):
    # labels = next-H-bar log return (vector over N)
    portfolio_return = (weights * labels).sum(dim=-1)
    cost = config.fee_bps * 1e-4 * (weights - prev_weights).abs().sum(dim=-1)
    entropy = -(weights * (weights + 1e-8).log()).sum(dim=-1) / math.log(weights.size(-1))
    return portfolio_return - cost + config.entropy_bonus * entropy
```

第 9 节给出对接代码。`prev_weights` 用上一根 bar 的权重传进来即可。

### 4.4 多 horizon 联合标签（加性）

3 天数据下样本太少，SFT 阶段建议同时拟合多个 horizon 的 rank label：

```
loss_total = α₁·L_rank(H=1) + α₂·L_rank(H=5) + α₃·L_rank(H=15)
```

相当于用同一份输入构造多个监督信号，能显著改善样本效率。

---

## 5. 数据切分与防穿越（小数据下尤其关键）

### 5.1 推荐切分（3 天）

```
day 1   ──────────────────────  train
day 2   ──────  val (用于 early-stop / 选 hyperparam)
day 3   ──────────────────────  test (从不参与调参)
```

**关键约束**（Lopez de Prado 的 purge & embargo）：

- **purge**：训练样本的 label 不能跨过验证起点，丢弃任何 label 区间 (t, t+H] 越过 day 1 收盘的样本。
- **embargo**：在 train 末端再丢掉 H 根 bar，防止 label-feature 交叠泄漏。

样本损失大约 (W + H) × 3 ≈ 100 ~ 150 根 bar，对 720 来说不可忽略，但比泄漏导致的乐观偏差好太多。

### 5.2 Walk-forward（更稳健，但 3 天偏短）

```
fold 1:  train=day1               val=day2_AM   test=day2_PM
fold 2:  train=day1+day2          test=day3
最终报告 test 平均
```

适合做最终 HF 实验的 reporting；hyperparam 选择仍然只在 val（不能用 fold 2 的 test 调参）。

### 5.3 截面标准化的"穿越"陷阱

**绝对禁止**用整段时间的均值方差做标准化。正确做法：

- 跨股票方向（截面方向）的 z / rank：永远只用当根 bar 自己的 N 只股票，不会泄漏。
- 跨时间方向（rolling）的 z / rank：必须用因果 rolling，统计窗口严格在 t 时刻之前。

`pandas.DataFrame.rolling(W).mean()` 默认就是因果的，但 `expanding()` 在 train/val 之间会泄漏，要分段做。

---

## 6. 评估指标

### 6.1 模型质量（训练时盯着）

| 指标 | 含义 | 目标 |
|---|---|---|
| **IC**（Pearson）        | 截面预测分 vs 实现收益的相关系数 | 单截面通常 0.01–0.05 已可观 |
| **RankIC**（Spearman）   | rank-based IC，更鲁棒 | HF 报告主用这个 |
| **ICIR**                | mean(IC) / std(IC)，跨 bar 稳定性 | > 0.1 即有信号 |
| **Top-Bottom spread**    | 多空十分位收益差（毛收益） | 看绝对效益 |
| **Hit rate**             | sign(score) 与 sign(realized) 一致比例 | > 50.5% 已比较有戏 |

**重点**：HF 看 IC**单值大小没那么重要，看 ICIR、看分位数稳定性**——3 天能稳就行。

### 6.2 策略质量（GRPO 阶段才管用）

| 指标 | 计算 |
|---|---|
| 净收益（after cost）   | $\sum_t (w_t \cdot r_{t,t+H}) - c \sum_t \|w_t - w_{t-H}\|_1$ |
| 年化 Sharpe           | √(252·N_per_day) × mean / std of bar returns |
| 最大回撤              | 累积净值的回撤序列最大值 |
| 换手率（双边/日）      | $\frac{1}{T}\sum_t \|w_t - w_{t-1}\|_1 / 2 \times 240$ |
| 持仓集中度 HHI        | $\sum_i w_i^2$，越小越分散 |

成本率初版：双边 0.13%（A 股 0.05% 印花税卖出 + 0.025%×2 佣金 + 0.005% 过户费），HF 真实成本会更高。

---

## 7. 高频特有的工程坑

### 7.1 涨跌停 / 停牌

- 不能买入：当根 bar 已封涨停（next-bar open ≥ close × 1.099）或停牌。
- 不能卖出：当根已封跌停或停牌。
- 简易判定：`涨跌幅(%) ≥ 9.9` 且 `成交量很小` 视为封板；`成交量(股) == 0` 视为停牌。
- GRPO 实现：把不可交易的股票从该截面 P_t 中**剔除**或对其 score 加 mask（最简单）。

### 7.2 集合竞价 bar 的特殊处理

- 09:30 那根 bar：开盘集合竞价的撮合价 + 09:30:00–09:30:59 的连续撮合，价量混合。
- 14:57–15:00：集合竞价的最后 3 分钟，A 股不能撤单。
- 建议：HF 阶段**直接丢弃** 09:30 和 15:00 这两根 bar 作为决策点，但仍可以作为特征 lookback。

### 7.3 池子在每根 bar 都不一样（动态 universe）

伪代码：

```python
def universe_at(t):
    s = stocks[(stocks.list_date <= t) & ((stocks.delist_date.isna()) | (stocks.delist_date > t))]
    s = s[~s.is_st_at(t)]
    s = s[~s.is_halted_at(t)]
    return s
```

`StockTransformer` 已经天然支持任意 N，但 dataloader 端必须按真实 P_t 切片输入。

### 7.4 隔夜：3 天的 day-1-close → day-2-open 的跳空

label 计算时**不要跨夜**：H 根 bar 的 horizon 不要跨过 11:30→13:00（中午休市）和 15:00→次日 09:30。否则学到的是隔夜 alpha，不是 HF。

### 7.5 时间戳对齐

CSV 里的时间戳是 bar 起始。计算 t 时刻特征**只能用 ≤ t-1 的 bar**（因为 t 这根 bar 在 t 这一刻还没结束）；下一可交易 bar 是 t+1。

具体：

```
特征时刻       t-1 这根 bar 已收，可用
决策时刻       t          （bar t 刚开始）
最早成交时刻   t          （以 bar t 的成交执行）→ 但这是非因果的，常用 t+1 vwap
最早可观测 label  t+H+1  （bar t+H 收完才知道）
```

把这条画在墙上，不然 HF 各种隐性泄漏会让你 IC 看着很高、实盘变 0。

---

## 8. 实验阶梯（建议顺序）

1. **Sanity zero-baseline**：等权多头（每股票 1/N），看 cost 后纯收益和 Sharpe。这是任何模型都必须打过的下限。
2. **简单截面 IC baseline**：把 `ret_5` 截面 rank 直接当分数（不训练），评估 IC、RankIC、Top-Bottom。如果信号很弱，先怀疑数据，再上模型。
3. **SFT only**：跑 `train_sft.py`，盯 RankIC 在 val 上是否高于 baseline。能高 0.005~0.01 已经是好结果。
4. **GRPO with cost=0**：先关掉成本项，验证 RL 流程没 bug、reward 在涨。
5. **GRPO with full cost + turnover penalty**：加上交易摩擦，再观察 net Sharpe。
6. **消融**：分别去掉 §3 中的特征组（价格 / 量 / 波动 / 时间），看哪类特征贡献最大。
7. **Horizon 扫描**：固定其他配置，跑 H ∈ {1, 3, 5, 15, 30}，比 ICIR 与 net Sharpe。
8. **池子扫描**：CSI 300（蓝筹）vs CSI 500（中盘）vs 全市场。HF alpha 通常在 CSI 500 / CSI 1000 上更显著。

每一步都把 train/val/test 三档指标都报一遍，注意 train→val 衰减比例（gap > 50% 一般是过拟合）。

---

## 9. 落地配方（接入现有代码）

### 9.1 数据 loader 草稿

新建 `files/hf_data.py`：

```python
import pandas as pd, numpy as np, torch
from pathlib import Path

POOL_DIR = Path("files/csi500/2025")  # 或 csi500/2026_1min

def load_pool(pool_dir=POOL_DIR):
    """读所有股票 1 分钟 CSV，返回 dict[stock_id] -> DataFrame, 索引为时间戳."""
    out = {}
    for csv in pool_dir.glob("*.csv"):
        df = pd.read_csv(csv, parse_dates=["日期"]).set_index("日期")
        df = df[~df.index.duplicated(keep="first")].sort_index()
        out[csv.stem] = df  # csv.stem 形如 "sh600004"
    return out

def build_panel(pool, dates):
    """切出 dates 这几天，对齐时间戳，返回 (T, N, F_raw) ndarray + 列名."""
    days_data = []
    for d in dates:
        # 取每天 09:31~11:30 + 13:01~15:00 共 240 根 bar；丢 09:30, 15:00
        ...
    panel = np.stack(days_data, axis=0)  # (T, N, F_raw)
    return panel, list(pool.keys())

def make_features(panel, W=30):
    """从原始 panel 算出 §3 各特征，截面 z-score，返回 (T, N, F)."""
    ...

def make_labels(panel, H=5):
    """vwap-based 下 H 根 bar 收益的截面 rank ∈ [0,1]，返回 (T, N)."""
    ...

def hf_samples(pool_dir, W=30, H=5):
    pool = load_pool(pool_dir)
    panel, ids = build_panel(pool, dates=sorted_unique_dates(pool))
    X = make_features(panel, W)  # (T, N, F)
    Y = make_labels(panel, H)    # (T, N)
    valid = slice(W, X.shape[0] - H)
    return [(torch.from_numpy(X[t]).float(),
             torch.from_numpy(Y[t]).float())
            for t in range(*valid.indices(X.shape[0]))]
```

### 9.2 接到 SFT

`train_sft.py` 里 `load_samples()` 改成：

```python
def load_samples():
    from hf_data import hf_samples
    return hf_samples("files/csi500/2025", W=30, H=5)
```

`Config.feat_dim` 改成 `make_features` 实际产出的 F；`Config.num_stocks` 用不上（按真实 N 走）。

### 9.3 接到 GRPO（带成本/换手的奖励）

`train_grpo.py` 里加上一个 HF 版本的 reward，并在 trainer 里维护"上一权重"：

```python
def hf_reward_fn(weights, features, labels, config, prev_weights=None, **_):
    pr = (weights * labels.unsqueeze(1)).sum(dim=-1)
    if prev_weights is None:
        cost = 0.0
    else:
        cost = config.fee_bps * 1e-4 * (weights - prev_weights.unsqueeze(1)).abs().sum(dim=-1)
    eps = 1e-8
    H = -(weights * (weights + eps).log()).sum(dim=-1) / math.log(weights.size(-1))
    return pr - cost + config.entropy_bonus * H
```

`GRPOConfig` 加 `fee_bps: float = 13.0`（双边 13 bps，初版偏严，可调）。

### 9.4 训练命令（顺序）

```bash
# 0. 池子（已经做过，跳过）
python3 files/build_csi500_pool.py

# 1. SFT
python3 files/train_sft.py
# → checkpoints/best_sft_model.pt

# 2. GRPO
python3 files/train_grpo.py
# → checkpoints/last_grpo_model.pt + logs/grpo_log.csv
```

---

## 10. 一句话总结

**3 天 × 1 分钟 × 中证 500** 的 HF 截面实验，能站得住的关键是三件事：

1. 把"截面"做扎实——所有特征在送进模型前都做截面 z/rank。
2. 把"因果"做严——purge + embargo + 不跨夜 + 不用未来 bar。
3. 用 SFT 把模型校到合理流形，再用 GRPO 直接打交易奖励（含成本和换手）。

模型本身（单层 Transformer）足够了；3 天数据下千万**别加层数、别加大 d_model**——你的瓶颈不是模型容量，是样本量与防穿越。
