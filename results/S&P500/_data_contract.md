# S&P500 数据契约（实测）

## 源目录

`/home/tom/Documents/SP500/equs_mini_with_adj_factors_202405_202605/`

## 文件

- 25 个月：`equs-mini-YYYYMMDD-YYYYMMDD.ohlcv-1m.with_adj.parquet`
- 覆盖：2024-05-01 … 2026-05-10（末月不完整）
- 辅表：`daily_adj_factors.parquet`、`databento_daily_close.parquet`、`symbol_map.csv`、`README.json`

## 字段（实测 schema）

| 列 | 类型 | 含义 |
| --- | --- | --- |
| ts_event | string (UTC) | 分钟 bar 时间 |
| session_date | timestamp[ns] | 美东日历日 |
| symbol | string | ticker（BRK.B 等） |
| open/high/low/close/volume | float/int | **未复权**成交价量 |
| adj_factor | float | Yahoo AdjClose / Databento日收 |
| open_adj…close_adj | float | 复权价 |

## 会话

- 原始含盘前盘后（ET 约 7–19 时）
- 本实验 **RTH**：`[09:30, 16:00)` America/New_York → 目标 **390** bars/日（动态以实际对齐为准）
- 模型输入用 `*_adj`；volume 用原始 volume（因子含分红，不缩放 volume）

## Universe

- **类型**：proxy_current_constituents_backfilled（来自 Databento job metadata / 出现过的 symbol）
- **披露**：非官方 point-in-time 成分；存在 **survivorship bias**，已写入 split/universe manifest

## 标签与窗口

- `y(i,D) = log(Close_adj(i,D) / Open_adj(i,D))`
- 预测 D：特征仅 `M_{D-L:D-1}`（不含 D 日分钟）
- OOS：`strict_fixed_oos` only；Test 2025-06…2026-05

## 禁用

Auction / Fusion / Gate / LSTM / TCN / PatchTST / Friday retrain / sequential OOS
