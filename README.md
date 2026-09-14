# grpo_cross_sectional

中证 500 分钟级截面选股：SFT 预测、SS-FM 组合生成、PPO / GRPO 强化学习，按月 walk-forward。

## Layout

```
project/
├── configs/
│   ├── default.yaml
│   ├── prediction.yaml
│   ├── ssfm.yaml
│   ├── rl.yaml
│   └── experiments.yaml
├── data/
├── models/
├── portfolio/
├── flow_matching/
├── rl/
├── backtest/
├── evaluation/
├── utils/
├── experiments/
├── train_prediction.py
├── train_ssfm.py
├── train_rl.py
├── run_backtest.py
└── run_all_experiments.py
```

行情数据、旧 `files/` 流水线和 `results/` 只留在本地，不进 GitHub。

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install torch pandas numpy pyyaml matplotlib
```

## Run

```bash
python run_all_experiments.py
python train_prediction.py
python train_ssfm.py
python train_rl.py
python run_backtest.py
```
