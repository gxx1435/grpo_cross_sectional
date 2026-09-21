# S&P500 工程适配清单

## 复用（as-is）

- `models/transformer.py`, `cross_stock_attention.py`, `alpha_predictor.py`（`standard_transformer`）
- `flow_matching/*`, `rl/*`, `backtest/*`, `portfolio/*`
- `data/splits.py`（strict_fixed_oos）
- `experiments/engine.py`, `run_all.py`（经 `make_store`）
- `evaluation/leakage_audit.py`（auction 对齐用同日占位）

## 新增

- `sp500/store.py`, `sp500/features.py`
- `configs/sp500.yaml`
- `results/S&P500/_data_contract.md`
- `utils.config.make_store` + `GRPO_MARKET=sp500` 加载 overlay

## 修改

- `data/trading_calendar.py`：`session_mode=us_rth`
- `data/loaders.data_contract`：SP500 parquet 契约
- `train_prediction.py` / `train_ssfm.py` / `train_rl.py` / `run_backtest.py` / `experiments/run_all.py`：`make_store`

## 禁用

- Auction / Fusion / Gate / LSTM / TCN / PatchTST
- Friday / sequential OOS
- overnight_* 特征

## 运行

```bash
export GRPO_MARKET=sp500
export PYTHONUNBUFFERED=1
python run_all_experiments.py --out-tag strict_fixed_oos --no-watchdog
# 或分入口：
python train_prediction.py --test-month 2025-06 --oos strict_fixed_oos
python train_ssfm.py --test-month 2025-06 --oos strict_fixed_oos
python train_rl.py --test-month 2025-06 --oos strict_fixed_oos
python run_backtest.py --test-month 2025-06 --oos strict_fixed_oos
```
