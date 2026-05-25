"""End-to-end demo: train SFT + GRPO on first 10 days, backtest on next 5 days."""

import sys, time, warnings
from pathlib import Path

sys.path.insert(0, "files")
warnings.filterwarnings("ignore")

from train_sft import SFTConfig, train_sft
from train_grpo_trl import GRPOTrainConfig, train_grpo
from backtest import BacktestConfig, backtest


POOL = Path("files/csi500/2026_1min")
N = 30
TRAIN_DAYS = 10
BT_DAYS = 5

print("=" * 80, flush=True)
print(f"[1/3] SFT  -- 2026 first {TRAIN_DAYS} days, {N} stocks", flush=True)
print("=" * 80, flush=True)

t = time.time()
train_sft(SFTConfig(
    pool_dir=POOL, start_day=0, num_days=TRAIN_DAYS, n_stocks_max=N,
    lookback=30, horizon=5,
    d_model=64, num_heads=4, ffn_dim=128,
    epochs=6, batch_size=16, lr=5e-4,
))
print(f"[SFT done in {time.time()-t:.1f}s]", flush=True)

print("\n" + "=" * 80, flush=True)
print(f"[2/3] GRPO -- IC + 0.5*Sharpe - 0.5*MaxDD, 8 generations", flush=True)
print("=" * 80, flush=True)

t = time.time()
train_grpo(GRPOTrainConfig(
    pool_dir=POOL, start_day=0, num_days=TRAIN_DAYS, n_stocks_max=N,
    lookback=30, horizon=5, episode_len=24,
    d_model=64, num_heads=4, ffn_dim=128,
    epochs=3, batch_size=4, lr=2e-5,
    num_generations=8, beta=0.04, epsilon=0.2,
    ic_weight=1.0, sharpe_weight=0.5, dd_weight=0.5,
))
print(f"[GRPO done in {time.time()-t:.1f}s]", flush=True)

print("\n" + "=" * 80, flush=True)
print(f"[3/3] Backtest -- next {BT_DAYS} trading days OOS", flush=True)
print("=" * 80, flush=True)

t = time.time()
backtest(BacktestConfig(
    pool_dir=POOL, start_day=TRAIN_DAYS, num_days=BT_DAYS, n_stocks_max=N,
    lookback=30, horizon=5,
    d_model=64, num_heads=4, ffn_dim=128, max_lookback=30,
    fee_bps=5.0, rebalance_every=5,
))
print(f"[Backtest done in {time.time()-t:.1f}s]", flush=True)
