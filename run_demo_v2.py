"""
End-to-end run #2: 200 stocks, 30-day training, 10-day OOS backtest.

Why this run is set up for *stability* (not raw return)
-------------------------------------------------------
1) Universe x10 vs the demo (30 -> 200 stocks): much better cross-sectional
   diversification, smaller idiosyncratic blow-ups in single names.
2) Training window x3 (10 -> 30 days): more cross sections to learn from
   so the SFT scorer fits real factor structure rather than a single regime.
3) Smoother policy:
   - Lower alpha_scale (10 -> 6)  => less concentrated Dirichlet => weights
     stay closer to uniform => realized portfolio variance drops a lot.
   - Higher KL beta in GRPO (0.04 -> 0.10) => the RL update can't drift far
     from the SFT base, which is the main source of training noise.
4) Reward weighted toward Sharpe & low drawdown:
   ic_weight 1.0, sharpe_weight 1.0 (was 0.5), dd_weight 1.0 (was 0.5).
5) Less frequent rebalance (every 5 -> every 10 bars) => half the turnover,
   half the cost-driven noise floor; cleaner OOS PnL curve.
6) All artifacts go to NEW directories so the previous demo is preserved:
        files/checkpoints_v2/   files/logs_v2/   files/backtest_out_v2/
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, "files")
warnings.filterwarnings("ignore")

from train_sft import SFTConfig, train_sft
from train_grpo_trl import GRPOTrainConfig, train_grpo
from backtest import BacktestConfig, backtest
from plot_backtest import main as plot_backtest


HERE = Path(__file__).resolve().parent
FILES = HERE / "files"
POOL = FILES / "csi500" / "2026_1min"

# === fresh artifact dirs (do NOT touch the v1 ones) ===
CKPT_DIR = FILES / "checkpoints_v2"
LOG_DIR = FILES / "logs_v2"
OUT_DIR = FILES / "backtest_out_v2"
for d in (CKPT_DIR, LOG_DIR, OUT_DIR):
    d.mkdir(parents=True, exist_ok=True)

SFT_CKPT = CKPT_DIR / "best_sft_model.pt"
GRPO_CKPT = CKPT_DIR / "last_grpo_model.pt"

# === experiment shape ===
N_STOCKS = 200
TRAIN_DAYS = 30
BT_DAYS = 10
LOOKBACK = 30
HORIZON = 5

# === stability-tuned policy knobs (shared across SFT / GRPO / backtest) ===
ALPHA_BASE = 1.0
ALPHA_SCALE = 6.0     # was 10.0 in v1 -> smoother Dirichlet -> less concentrated weights

# === stability-tuned model size (a touch wider for 200 stocks) ===
D_MODEL = 96
NUM_HEADS = 4
FFN_DIM = 192
DROPOUT = 0.15        # mild bump for a larger universe

print("=" * 88, flush=True)
print(
    f"[1/4] SFT  -- 2026 first {TRAIN_DAYS} days, {N_STOCKS} stocks "
    f"(L={LOOKBACK}, H={HORIZON})",
    flush=True,
)
print("=" * 88, flush=True)

if SFT_CKPT.is_file():
    print(f"  -> SFT checkpoint already exists at {SFT_CKPT}, skipping SFT.", flush=True)
else:
    t = time.time()
    train_sft(SFTConfig(
        pool_dir=POOL,
        start_day=0,
        num_days=TRAIN_DAYS,
        n_stocks_max=N_STOCKS,
        lookback=LOOKBACK,
        horizon=HORIZON,

        d_model=D_MODEL, num_heads=NUM_HEADS, ffn_dim=FFN_DIM, dropout=DROPOUT,

        epochs=10, batch_size=16, lr=4e-4, weight_decay=2e-2,
        warmup_ratio=0.05, grad_clip=1.0,

        val_ratio=0.15, test_ratio=0.15,
        mse_weight=0.5, rank_weight=0.3,

        ckpt_dir=CKPT_DIR, log_dir=LOG_DIR,
    ))
    print(f"[SFT done in {time.time()-t:.1f}s]", flush=True)

print("\n" + "=" * 88, flush=True)
print(
    "[2/4] GRPO -- IC + 1.0*Sharpe - 1.0*MaxDD, 8 generations, KL beta=0.10 "
    "(light touch on top of strong SFT)",
    flush=True,
)
print("=" * 88, flush=True)

# Notes on GRPO compute:
#   With N=200 stocks the cross-section attention is ~44x heavier than the
#   N=30 demo, so we deliberately keep GRPO as a *light* polish on top of the
#   already-strong SFT scorer (test rank_ic > 0.99):
#     - episode_len 24 -> 12        (half the cross sections per step)
#     - epochs 4 -> 2               (cosine schedule still tapers)
#     - num_generations 12 -> 8     (cheaper sampling + log-prob)
#     - val/test_ratio 0.15 -> 0.4  (sub-sample training episodes; we don't
#                                     actually iterate val/test during GRPO,
#                                     this just shrinks the train index set)
#     - strong KL anchor beta=0.10  (keeps policy near the SFT base)
t = time.time()
train_grpo(GRPOTrainConfig(
    pool_dir=POOL,
    start_day=0,
    num_days=TRAIN_DAYS,
    n_stocks_max=N_STOCKS,
    lookback=LOOKBACK,
    horizon=HORIZON,
    episode_len=12,

    d_model=D_MODEL, num_heads=NUM_HEADS, ffn_dim=FFN_DIM, dropout=DROPOUT,

    sft_ckpt=SFT_CKPT,
    alpha_base=ALPHA_BASE, alpha_scale=ALPHA_SCALE,

    epochs=2, batch_size=4, lr=1e-5, weight_decay=1e-2,
    warmup_ratio=0.05, grad_clip=1.0,

    num_generations=8, beta=0.10, epsilon=0.2, temperature=1.0,
    scale_rewards="group",

    ic_weight=1.0, sharpe_weight=1.0, dd_weight=1.0,

    val_ratio=0.4, test_ratio=0.4,
    ckpt_dir=CKPT_DIR, log_dir=LOG_DIR,
))
print(f"[GRPO done in {time.time()-t:.1f}s]", flush=True)

print("\n" + "=" * 88, flush=True)
print(
    f"[3/4] Backtest -- next {BT_DAYS} OOS days, "
    f"rebalance every 10 bars, fee 5bps",
    flush=True,
)
print("=" * 88, flush=True)

t = time.time()
bt_cfg = BacktestConfig(
    pool_dir=POOL,
    start_day=TRAIN_DAYS,
    num_days=BT_DAYS,
    n_stocks_max=N_STOCKS,
    lookback=LOOKBACK,
    horizon=HORIZON,

    d_model=D_MODEL, num_heads=NUM_HEADS, ffn_dim=FFN_DIM, dropout=DROPOUT,
    max_lookback=LOOKBACK,

    alpha_base=ALPHA_BASE, alpha_scale=ALPHA_SCALE,

    fee_bps=5.0, rebalance_every=10,

    sft_ckpt=SFT_CKPT, grpo_ckpt=GRPO_CKPT,
    output_dir=OUT_DIR,
)
backtest(bt_cfg)
print(f"[Backtest done in {time.time()-t:.1f}s]", flush=True)

print("\n" + "=" * 88, flush=True)
print("[4/4] Plot -- NAV curve + rolling Sharpe + summary bars", flush=True)
print("=" * 88, flush=True)

t = time.time()
plot_backtest(bt_cfg)
print(f"[Plot done in {time.time()-t:.1f}s]", flush=True)

print("\n" + "=" * 88, flush=True)
print("All v2 artifacts saved under:", flush=True)
print(f"  ckpts  : {CKPT_DIR}", flush=True)
print(f"  logs   : {LOG_DIR}", flush=True)
print(f"  output : {OUT_DIR}", flush=True)
print("=" * 88, flush=True)
