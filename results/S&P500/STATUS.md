# S&P500 实验执行状态

更新时间：以 `run_all.log` / `heartbeat.json` 为准。

## 已完成

| 项 | 状态 |
| --- | --- |
| 数据契约扫描 | OK（25 月 parquet，`*_adj` + `adj_factor`） |
| Store 构建 | OK：`T=507 N=499 F=44 bars=390`，缓存 `files/sp500/.cache/`（约 75min） |
| 配置 / 入口 | OK：`GRPO_MARKET=sp500` → `configs/sp500.yaml` |
| 训练启动 | **进行中**：`results/S&P500/strict_fixed_oos/` |

## 正在跑

```bash
GRPO_MARKET=sp500 python run_all_experiments.py --out-tag strict_fixed_oos --no-watchdog
```

- GPU：RTX 3090
- 预估墙钟：~14.4h（12 个 Test 月）
- OOS：仅 `strict_fixed_oos`
- Alpha：`standard_transformer` + Cross-Stock Attention
- 矩阵：EW/MVO/MaxSharpe/RP/BL/Kelly + MLP/Gaussian/Diffusion/FM/SSFM + PPO/GRPO + G∈{8,16,32,64,128}

## 监控

```bash
tail -f results/S\&P500/run_all.log
cat results/S\&P500/strict_fixed_oos/heartbeat.json
```

## 已知披露

- Universe：**proxy**（非官方 PIT 成分）→ survivorship bias
- 面板自 2024-05：Train 最早日需 lookback，已过滤不足窗口的 asof 日
- 末月数据至 2026-05-08（非整月）

## 产物路径（完成后）

- 月度：`results/S&P500/strict_fixed_oos/strict_fixed_oos/test_month=YYYY-MM/<exp_id>/`
- 全年：`results/S&P500/final_summary/S&P500全年实验分析.md`（finalize 后）
