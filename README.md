# grpo_cross_sectional

S&P 500 分钟级 OHLCV 严格 monthly walk-forward 研究流水线：

`D-10..D-1 minute features -> Minute Transformer -> Cross-Stock Attention -> D open-to-close alpha -> Top-K -> portfolio generation -> SS-FM -> PPO/GRPO`

本分支只允许 `strict_fixed_oos`。每个 Test 月使用连续 12 个月 Train、前一月 Validation，并在整个 Test 月冻结模型、scaler、风险状态训练逻辑和策略参数。原始数据中没有 point-in-time 历史指数成分，因此动态股票池明确标记为 `data_available_equity_universe`。

## 环境

推荐 Python 3.11、PyTorch 2.8 + CUDA 12.8；其余版本锁定在 `requirements.txt`。在本机完成实验时使用：

```bash
/home/zouli/tmp/milestone4_torch280/bin/python -m pytest -q tests/test_sp500_strict.py
```

## 数据

默认源文件：

```text
/home/zouli/projects/research/data/_temp/EQUS-20260919-6L9X397UJK.zip
```

流水线用 pandas 逐月读取 ZIP 内 CSV-Zstandard，标准化为 `stock_code, minute_timestamp, trading_date, open, high, low, close, volume`，持久化至 `data/processed/sp500/`。版本复用同时校验 source member、feature config、feature schema 和代码 hash。

## 执行

激活包含依赖的 Python 环境后，以下命令均可运行：

```bash
python run_all_experiments.py
python train_prediction.py --test-month 2025-06
python train_ssfm.py --test-month 2025-06
python train_rl.py --test-month 2025-06
python run_backtest.py --test-month 2025-06
```

仅准备和核验真实数据：

```bash
python run_all_experiments.py --prepare-only
```

开发期真实 smoke run（仍读取真实数据、训练真实模型，不生成模拟结果）：

```bash
python run_all_experiments.py --smoke --months 2025-06
```

所有数据契约、月度模型、预测、组合、回测、审计和报告写入 `results/S&P500/`。
