# grpo_cross_sectional

Walk-forward cross-sectional stock selection with SFT + GRPO (Transformer / PatchTST), plus linear-factor baselines.

## Layout

```
grpo_cross_sectional/
├── run_walkforward_*.py      # experiment entry scripts
├── files/
│   ├── *.py                  # training, data, backtest pipeline
│   ├── README.md             # minute-bar data format (local only, not in git)
│   ├── csi500/               # CSI500 subset (local, gitignored)
│   └── experiments/          # experiment READMEs (outputs gitignored)
└── README.md
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch pandas numpy matplotlib
```

Place minute-bar CSVs under `files/csi500/` as described in `files/README.md`.

## Run examples

```bash
# Transformer L1d walk-forward (long-only, random 30)
python3 run_walkforward_longonly_r30_semiannual_L1d.py

# Linear factor baseline (same schedule, no training)
python3 run_walkforward_longonly_r30_semiannual_L1d_linear.py
```

See `files/experiments/SFT_GRPO模型详解.md` for model and pipeline details.
