"""
Post-training evaluation for the *first* GRPO checkpoint.

Background
----------
The first GRPO training (artifact: ``files/checkpoints/last_grpo_model.pt``)
was launched by ``run_demo.py`` on the first 10 trading days of
``files/csi500/2026_1min`` with 30 stocks.

    train days : 2026-01-05 .. 2026-01-16 (start_day=0, num_days=10)
    cutoff     : 2026-01-16
    next 3 d   : 2026-01-19 .. 2026-01-21 (start_day=10, num_days=3)

This script:

1. Builds features for the 3 trading days after the cutoff via
   ``hf_data.build_hf_dataset``.
2. Loads ``checkpoints/last_grpo_model.pt``, sweeps the policy bar-by-bar
   (deterministic Dirichlet mean weights), and simulates a portfolio with
   the same knobs the demo used (rebalance every 5 bars, 5 bps one-side).
3. Plots cumulative NAV + rolling Sharpe against the equal-weight
   baseline using a *compressed* x-axis (bar index, with day separators),
   so the overnight 16:00-09:30 gap and the 11:30-13:00 lunch break are
   visually removed.
4. Randomly samples 3 cross sections from those 3 days, computes for each:
       - Spearman/Pearson rank-IC between scores and forward returns
       - Direction-accuracy (sign of score-0.5 vs sign of fwd return)
       - Top-10 vs bottom-10 average forward return
   and dumps a per-stock comparison table.
5. Prints a structured summary at the end.

Outputs land in ``files/backtest_out_post_train/``.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
FILES = HERE / "files"
sys.path.insert(0, str(FILES))

from backtest import (  # noqa: E402  -- sys.path manipulation above
    BARS_PER_DAY,
    BARS_PER_YEAR,
    BacktestConfig,
    _load_policy,
    annualized_stats,
    run_policy_sweep,
    scores_to_weights,
    simulate,
)
from hf_data import HFConfig, build_hf_dataset  # noqa: E402


# ---------------------------------------------------------------------------
# Knobs (match run_demo.py / files/checkpoints exactly)
# ---------------------------------------------------------------------------
POOL_DIR = FILES / "csi500" / "2026_1min"
N_STOCKS = 30
LOOKBACK = 30
HORIZON = 5
START_DAY = 10            # first OOS trading day index (post-cutoff)
NUM_DAYS = 3              # 3 trading days right after the cutoff
REBALANCE_EVERY = 5
FEE_BPS = 5.0

CKPT_DIR = FILES / "checkpoints"
OUT_DIR = FILES / "backtest_out_post_train"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RNG = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _ts_from_int64(arr: np.ndarray) -> pd.DatetimeIndex:
    """Robust int64 -> DatetimeIndex.

    Older caches (built under pandas>=3) accidentally stored timestamps in
    microseconds while the field name says nanoseconds. Detect the unit by
    looking at the magnitude: 2025/2026 dates are ~1.7e18 ns or ~1.7e15 us.
    Anything below ~1e17 must therefore be us (or older).
    """
    a = np.asarray(arr, dtype="int64")
    max_abs = int(np.abs(a).max()) if a.size else 0
    if max_abs >= 10**17:
        unit = "ns"
    elif max_abs >= 10**14:
        unit = "us"
    elif max_abs >= 10**11:
        unit = "ms"
    else:
        unit = "s"
    return pd.to_datetime(a, unit=unit)


def rolling_sharpe(net_pr: np.ndarray, window: int, ann: int) -> np.ndarray:
    if len(net_pr) < window:
        return np.full(len(net_pr), np.nan)
    s = pd.Series(net_pr)
    mu = s.rolling(window).mean()
    sd = s.rolling(window).std(ddof=0)
    return (math.sqrt(ann) * mu / (sd + 1e-12)).to_numpy()


def spearman_ic(scores: np.ndarray, rets: np.ndarray) -> float:
    """Rank-based Pearson == Spearman correlation."""
    if scores.size < 3:
        return float("nan")
    rs = np.asarray(pd.Series(scores).rank().to_numpy(), dtype=np.float64).copy()
    rr = np.asarray(pd.Series(rets).rank().to_numpy(), dtype=np.float64).copy()
    rs = rs - rs.mean()
    rr = rr - rr.mean()
    den = (np.linalg.norm(rs) * np.linalg.norm(rr)) + 1e-12
    return float((rs * rr).sum() / den)


def pearson_ic(scores: np.ndarray, rets: np.ndarray) -> float:
    if scores.size < 3:
        return float("nan")
    a = scores.astype(np.float64) - float(scores.mean())
    b = rets.astype(np.float64) - float(rets.mean())
    den = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float((a * b).sum() / den)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    torch.manual_seed(42)

    cfg = BacktestConfig(
        pool_dir=POOL_DIR,
        start_day=START_DAY,
        num_days=NUM_DAYS,
        n_stocks_max=N_STOCKS,
        lookback=LOOKBACK,
        horizon=HORIZON,
        d_model=64, num_heads=4, ffn_dim=128, dropout=0.1, max_lookback=LOOKBACK,
        alpha_base=1.0, alpha_scale=10.0,
        fee_bps=FEE_BPS, rebalance_every=REBALANCE_EVERY,
        sft_ckpt=CKPT_DIR / "best_sft_model.pt",
        grpo_ckpt=CKPT_DIR / "last_grpo_model.pt",
        output_dir=OUT_DIR,
    )

    hf_cfg = HFConfig(
        pool_dir=cfg.pool_dir,
        start_day=cfg.start_day,
        num_days=cfg.num_days,
        lookback=cfg.lookback,
        horizon=cfg.horizon,
        n_stocks_max=cfg.n_stocks_max,
    )
    bundle = build_hf_dataset(hf_cfg)
    feat_dim = bundle["x_seq"].shape[-1]
    n_stocks = bundle["x_seq"].shape[1]
    stock_ids = list(bundle["stock_ids"])

    model, ckpt_label = _load_policy(cfg, feat_dim)
    print("=" * 80)
    print(" Post-training evaluation -- 3 trading days AFTER the cutoff")
    print("=" * 80)
    print(f" universe   : N={n_stocks} CSI500 stocks (first {N_STOCKS} of pool)")
    print(f" data shape : F={feat_dim} L={cfg.lookback} H={cfg.horizon}")
    print(f" date range : start_day={cfg.start_day}, num_days={cfg.num_days}")
    print(f"              {pd.to_datetime(bundle['timestamps_ns'][0])}"
          f" .. {pd.to_datetime(bundle['timestamps_ns'][-1])}")
    print(f" policy     : {ckpt_label}")
    print(f" rebalance  : every {cfg.rebalance_every} bars  | fee {cfg.fee_bps}bps")
    print("-" * 80)

    # ── 1) Sweep policy + simulate -----------------------------------------
    weights, returns, ts_idx = run_policy_sweep(cfg, bundle, model)
    sim_pol = simulate(weights, returns, cfg)
    eq_w = np.full_like(weights, 1.0 / n_stocks, dtype=np.float32)
    sim_eq = simulate(eq_w, returns, cfg)

    ts_all = _ts_from_int64(bundle["timestamps_ns"])
    ts = ts_all[ts_idx]
    pol_pr = sim_pol["net_pr"]
    eq_pr = sim_eq["net_pr"]

    pol_nav = (1.0 + pol_pr).cumprod()
    eq_nav = (1.0 + eq_pr).cumprod()

    # ── 2) Stats ----------------------------------------------------------
    win = min(120, max(20, len(pol_pr) // 5))
    pol_sh = rolling_sharpe(pol_pr, win, BARS_PER_YEAR)
    eq_sh = rolling_sharpe(eq_pr, win, BARS_PER_YEAR)
    stats_pol = annualized_stats(pol_pr, BARS_PER_YEAR)
    stats_eq = annualized_stats(eq_pr, BARS_PER_YEAR)

    # Daily aggregation (sum of log-like net returns is fine for tiny sizes)
    bars_df = pd.DataFrame({
        "ts": ts,
        "policy_ret": pol_pr,
        "policy_nav": pol_nav,
        "ew_ret": eq_pr,
        "ew_nav": eq_nav,
        "policy_rolling_sharpe": pol_sh,
        "ew_rolling_sharpe": eq_sh,
    })
    bars_df.to_csv(OUT_DIR / "bar_returns.csv", index=False)

    daily_df = (bars_df.assign(date=bars_df["ts"].dt.normalize())
                       .groupby("date")
                       .agg(policy_daily_ret=("policy_ret",
                                              lambda x: (1 + x.values).prod() - 1),
                            ew_daily_ret=("ew_ret",
                                          lambda x: (1 + x.values).prod() - 1))
                       .reset_index())
    daily_df["policy_cum_nav"] = (1.0 + daily_df["policy_daily_ret"]).cumprod()
    daily_df["ew_cum_nav"]     = (1.0 + daily_df["ew_daily_ret"]).cumprod()
    daily_df.to_csv(OUT_DIR / "daily_returns.csv", index=False)

    print("=== headline stats (3-day OOS, 1-min bars) ===")
    print(f"  policy  : bars={int(stats_pol['bars'])} "
          f"total={stats_pol['total_return']*100:+.3f}% "
          f"sharpe(ann)={stats_pol['annualized_sharpe']:+.2f} "
          f"MDD={stats_pol['max_drawdown']*100:.3f}% "
          f"hit={stats_pol['hit_rate']*100:.1f}%")
    print(f"  EW base : bars={int(stats_eq['bars'])} "
          f"total={stats_eq['total_return']*100:+.3f}% "
          f"sharpe(ann)={stats_eq['annualized_sharpe']:+.2f} "
          f"MDD={stats_eq['max_drawdown']*100:.3f}% "
          f"hit={stats_eq['hit_rate']*100:.1f}%")
    print()
    print("=== daily returns ===")
    print(daily_df.to_string(index=False))
    print("-" * 80)

    # ── 3) Plot (compressed bar-index x-axis so non-trading time is gone) ─
    bar_idx = np.arange(len(pol_pr))
    dates_only = ts.normalize()
    day_change = np.r_[True, dates_only.values[1:] != dates_only.values[:-1]]
    day_starts = bar_idx[day_change]
    day_labels = ts[day_starts].strftime("%m-%d")

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]})

    ax = axes[0]
    ax.plot(bar_idx, pol_nav, color="#d6336c", lw=1.7, label="GRPO policy")
    ax.plot(bar_idx, eq_nav,  color="#1864ab", lw=1.4, label="Equal-weight")
    ax.axhline(1.0, color="#aaaaaa", lw=0.8, ls="--")
    for d in day_starts[1:]:
        ax.axvline(d, color="#999999", lw=0.7, ls=":")
        ax.text(d, ax.get_ylim()[1], "  new day", color="#666",
                fontsize=8, va="top")
    ax.set_ylabel("Cumulative NAV  (start = 1)")
    ax.set_title(
        f"Post-train OOS (3 days after cutoff)  --  "
        f"first GRPO ckpt, N={n_stocks}, 1-min bars, "
        f"{cfg.fee_bps:.0f}bps/rebal x{cfg.rebalance_every}\n"
        f"policy  total={stats_pol['total_return']*100:+.3f}%  "
        f"Sharpe={stats_pol['annualized_sharpe']:+.2f}     |     "
        f"EW  total={stats_eq['total_return']*100:+.3f}%  "
        f"Sharpe={stats_eq['annualized_sharpe']:+.2f}"
    )
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(bar_idx, pol_sh, color="#d6336c", lw=1.2, label="GRPO policy")
    ax.plot(bar_idx, eq_sh,  color="#1864ab", lw=1.0, label="Equal-weight")
    ax.axhline(0.0, color="#aaaaaa", lw=0.8, ls="--")
    for d in day_starts[1:]:
        ax.axvline(d, color="#999999", lw=0.7, ls=":")
    ax.set_ylabel(f"Rolling Sharpe (win={win} bars)")
    ax.set_xlabel("Trading bar index  (overnight + lunch gaps removed)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_xticks(day_starts)
    ax.set_xticklabels(day_labels, rotation=0)

    fig.tight_layout()
    p_curves = OUT_DIR / "post_train_curves.png"
    fig.savefig(p_curves, dpi=130)
    plt.close(fig)
    print(f"  saved -> {p_curves}")

    # ── 4) Random sample 3 cross sections, check direction accuracy --------
    valid_start = int(bundle["valid_start"])
    valid_end   = int(bundle["valid_end"])
    x_seq = bundle["x_seq"]
    fwd   = bundle["fwd_return"]
    L     = cfg.lookback
    ts_full = ts_all

    sample_t = sorted(int(t) for t in
                      RNG.choice(np.arange(valid_start, valid_end),
                                 size=3, replace=False))

    section_results = []
    section_tables  = []

    with torch.no_grad():
        for k, t in enumerate(sample_t, start=1):
            window = x_seq[t - L + 1: t + 1]
            x = np.transpose(window, (1, 0, 2)).copy()
            x_t = torch.from_numpy(x).float()

            scores  = model(x_t)
            weights = scores_to_weights(scores, cfg.alpha_base, cfg.alpha_scale)
            scores_np  = scores.numpy()
            weights_np = weights.numpy()
            fwd_np = fwd[t].astype(np.float64)

            section_df = (pd.DataFrame({
                "stock_id": stock_ids,
                "score":    scores_np,
                "weight":   weights_np,
                "fwd_return_H": fwd_np,
            })
                .sort_values("score", ascending=False)
                .reset_index(drop=True))
            section_df.insert(0, "rank_by_score", np.arange(1, len(section_df) + 1))
            section_df["rank_by_return"] = (
                section_df["fwd_return_H"].rank(ascending=False, method="min")
                .astype(int))

            # Direction-accuracy: pick the top-half vs bottom-half by score
            # and check that top-half average forward return > bottom-half.
            half = max(1, len(section_df) // 2)
            top_avg_ret = section_df.head(half)["fwd_return_H"].mean()
            bot_avg_ret = section_df.tail(half)["fwd_return_H"].mean()
            top_minus_bot = top_avg_ret - bot_avg_ret

            # Per-stock hit (above-median score -> above-median return)
            above_med_score = scores_np >= np.median(scores_np)
            above_med_ret   = fwd_np >= np.median(fwd_np)
            hit_rate = float((above_med_score == above_med_ret).mean())

            ic_spear = spearman_ic(scores_np, fwd_np)
            ic_pear  = pearson_ic(scores_np, fwd_np)

            # Top-10 vs bottom-10
            top10_ret = section_df.head(10)["fwd_return_H"].mean()
            bot10_ret = section_df.tail(10)["fwd_return_H"].mean()

            section_results.append({
                "k": k,
                "t_idx": t,
                "timestamp": ts_full[t],
                "ic_spearman": ic_spear,
                "ic_pearson":  ic_pear,
                "hit_rate":    hit_rate,
                "top_half_ret":  top_avg_ret,
                "bot_half_ret":  bot_avg_ret,
                "top_minus_bot": top_minus_bot,
                "top10_ret":  top10_ret,
                "bot10_ret":  bot10_ret,
            })
            section_tables.append(section_df.assign(section=k, timestamp=ts_full[t]))

    sec_summary = pd.DataFrame(section_results)
    sec_summary.to_csv(OUT_DIR / "sampled_sections_summary.csv", index=False)
    pd.concat(section_tables).to_csv(
        OUT_DIR / "sampled_sections_perstock.csv", index=False)

    print()
    print("=" * 80)
    print(" Random sample of 3 cross sections  --  direction accuracy")
    print("=" * 80)
    for r in section_results:
        ts_s = r["timestamp"].strftime("%Y-%m-%d %H:%M")
        print(f"\n[section {r['k']}]  t={r['t_idx']}  {ts_s}")
        print(f"   IC (spearman) : {r['ic_spearman']:+.4f}")
        print(f"   IC (pearson)  : {r['ic_pearson']:+.4f}")
        print(f"   above-median hit rate : {r['hit_rate']*100:.1f}%   "
              f"(naive random = 50%)")
        print(f"   top-half avg fwd-ret  : {r['top_half_ret']*1e4:+.2f} bps")
        print(f"   bot-half avg fwd-ret  : {r['bot_half_ret']*1e4:+.2f} bps")
        print(f"   top - bot              : {r['top_minus_bot']*1e4:+.2f} bps  "
              + ("(direction correct)" if r['top_minus_bot'] > 0
                 else "(direction WRONG)"))
        print(f"   top10 - bot10          : "
              f"{(r['top10_ret']-r['bot10_ret'])*1e4:+.2f} bps")
        # Top-3 stocks for context
        top3 = section_tables[r["k"] - 1].head(3)[
            ["rank_by_score", "stock_id", "score", "weight",
             "fwd_return_H", "rank_by_return"]
        ]
        bot3 = section_tables[r["k"] - 1].tail(3)[
            ["rank_by_score", "stock_id", "score", "weight",
             "fwd_return_H", "rank_by_return"]
        ]
        print("   top-3 by score:")
        print(top3.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
        print("   bottom-3 by score:")
        print(bot3.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))

    print()
    print("=" * 80)
    print(" Aggregated section diagnostics (means across the 3 samples)")
    print("=" * 80)
    print(f"   mean IC (spearman)         : {sec_summary['ic_spearman'].mean():+.4f}")
    print(f"   mean IC (pearson)          : {sec_summary['ic_pearson'].mean():+.4f}")
    print(f"   mean above-median hit rate : "
          f"{sec_summary['hit_rate'].mean()*100:.1f}%")
    print(f"   mean top-half - bot-half   : "
          f"{(sec_summary['top_half_ret']-sec_summary['bot_half_ret']).mean()*1e4:+.2f} bps")

    print()
    print(f"  results saved under {OUT_DIR}")


if __name__ == "__main__":
    main()
