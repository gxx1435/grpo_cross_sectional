"""
Plot OOS backtest curves: cumulative return + rolling Sharpe.

Reuses ``backtest.py`` machinery: re-runs the policy sweep + simulate
(both fast, ~2s on CPU) so we have bar-level series in memory, then
renders matplotlib PNGs into ``files/backtest_out/``.

Outputs
-------
- backtest_curves.png       cumulative NAV + rolling Sharpe (2 subplots)
- backtest_summary_bar.png  side-by-side bar chart of headline stats
- bar_returns.csv           bar-level net returns + NAV (for downstream use)

Run
---
    python3 files/plot_backtest.py
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from backtest import (
    BARS_PER_YEAR,
    BacktestConfig,
    _load_policy,
    annualized_stats,
    run_policy_sweep,
    simulate,
)
from hf_data import HFConfig, build_hf_dataset


HERE = Path(__file__).resolve().parent


def rolling_sharpe(net_pr: np.ndarray, window: int, ann: int) -> np.ndarray:
    """Rolling annualized Sharpe with a centered window (NaN at the edges)."""
    if len(net_pr) < window:
        return np.full(len(net_pr), np.nan)
    s = pd.Series(net_pr)
    mu = s.rolling(window).mean()
    sd = s.rolling(window).std(ddof=0)
    sharpe = math.sqrt(ann) * mu / (sd + 1e-12)
    return sharpe.to_numpy()


def main(cfg: BacktestConfig | None = None) -> None:
    if cfg is None:
        # Match the actual demo run: 1-week OOS, 30 stocks.
        cfg = BacktestConfig(start_day=10, num_days=5, n_stocks_max=30)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

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

    model, ckpt_label = _load_policy(cfg, feat_dim)
    print(f"policy   : {ckpt_label}")
    print(f"shape    : N={n_stocks} F={feat_dim} L={cfg.lookback}")
    print(f"window   : start_day={cfg.start_day} num_days={cfg.num_days}")

    weights, returns, ts_idx = run_policy_sweep(cfg, bundle, model)

    sim_pol = simulate(weights, returns, cfg)
    eq_w = np.full_like(weights, 1.0 / n_stocks, dtype=np.float32)
    sim_eq = simulate(eq_w, returns, cfg)

    ts = pd.to_datetime(bundle["timestamps_ns"][ts_idx])
    pol_pr = sim_pol["net_pr"]
    eq_pr = sim_eq["net_pr"]

    pol_nav = (1.0 + pol_pr).cumprod()
    eq_nav = (1.0 + eq_pr).cumprod()

    # rolling Sharpe; window ~ half a trading day (120 bars).
    win = min(120, max(20, len(pol_pr) // 5))
    pol_sh = rolling_sharpe(pol_pr, win, BARS_PER_YEAR)
    eq_sh = rolling_sharpe(eq_pr, win, BARS_PER_YEAR)

    stats_pol = annualized_stats(pol_pr, BARS_PER_YEAR)
    stats_eq = annualized_stats(eq_pr, BARS_PER_YEAR)

    # ------------------------------------------------------------------
    # Persist bar-level series for reproducibility
    # ------------------------------------------------------------------
    bars_df = pd.DataFrame({
        "ts": ts,
        "policy_ret": pol_pr,
        "policy_nav": pol_nav,
        "ew_ret": eq_pr,
        "ew_nav": eq_nav,
        "policy_rolling_sharpe": pol_sh,
        "ew_rolling_sharpe": eq_sh,
    })
    bars_df.to_csv(cfg.output_dir / "bar_returns.csv", index=False)
    print(f"saved   : {cfg.output_dir / 'bar_returns.csv'}")

    # ------------------------------------------------------------------
    # Plot 1: cumulative NAV + rolling Sharpe
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]})

    ax = axes[0]
    ax.plot(ts, pol_nav, color="#d6336c", lw=1.6, label=f"GRPO policy")
    ax.plot(ts, eq_nav, color="#1864ab", lw=1.4, label="Equal-weight")
    ax.axhline(1.0, color="#aaaaaa", lw=0.8, ls="--")
    ax.set_ylabel("Cumulative NAV (start = 1)")
    ax.set_title(
        f"OOS backtest -- {n_stocks} CSI500 stocks, 1-min bars, "
        f"{cfg.fee_bps:.1f}bps/rebal x{cfg.rebalance_every} | "
        f"policy total={stats_pol['total_return']*100:+.2f}%, "
        f"Sharpe={stats_pol['annualized_sharpe']:.2f} | "
        f"EW total={stats_eq['total_return']*100:+.2f}%, "
        f"Sharpe={stats_eq['annualized_sharpe']:.2f}"
    )
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(ts, pol_sh, color="#d6336c", lw=1.2, label=f"GRPO policy")
    ax.plot(ts, eq_sh, color="#1864ab", lw=1.0, label="Equal-weight")
    ax.axhline(0.0, color="#aaaaaa", lw=0.8, ls="--")
    ax.set_ylabel(f"Rolling Sharpe (win={win} bars)")
    ax.set_xlabel("Time")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    fig.autofmt_xdate()

    fig.tight_layout()
    out_curves = cfg.output_dir / "backtest_curves.png"
    fig.savefig(out_curves, dpi=130)
    plt.close(fig)
    print(f"saved   : {out_curves}")

    # ------------------------------------------------------------------
    # Plot 1b: same NAV / rolling Sharpe but with a *compressed* x-axis
    # (bar index instead of calendar time).
    #
    # On a 1-min trading bar series only ~17% of calendar time is actual
    # trading (4h/day out of 24h, plus the 11:30-13:00 lunch break and the
    # 09:30/15:00 auctions are dropped). Plotting against `ts` therefore
    # spends most of the x-axis drawing flat overnight segments and squeezes
    # every trading session into a tiny sliver. Using bar index gives every
    # traded minute equal x-axis space and the overnight gap collapses to a
    # single tick (visible as a small jump in NAV instead of a long flat).
    # ------------------------------------------------------------------
    bar_idx = np.arange(len(pol_pr))

    # Find the first bar of each new trading day for x-tick labels, plus
    # vertical separators that show where the overnight gap lives now.
    dates_only = ts.normalize()
    day_change = np.r_[True, dates_only.values[1:] != dates_only.values[:-1]]
    day_starts = bar_idx[day_change]
    day_labels = ts[day_starts].strftime("%m-%d")

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]})

    ax = axes[0]
    ax.plot(bar_idx, pol_nav, color="#d6336c", lw=1.6, label="GRPO policy")
    ax.plot(bar_idx, eq_nav, color="#1864ab", lw=1.4, label="Equal-weight")
    ax.axhline(1.0, color="#aaaaaa", lw=0.8, ls="--")
    for d in day_starts[1:]:
        ax.axvline(d, color="#cccccc", lw=0.7, ls=":")
    ax.set_ylabel("Cumulative NAV (start = 1)")
    ax.set_title(
        f"OOS backtest (compressed time) -- {n_stocks} CSI500 stocks, "
        f"1-min bars, {cfg.fee_bps:.1f}bps/rebal x{cfg.rebalance_every} | "
        f"policy total={stats_pol['total_return']*100:+.2f}%, "
        f"Sharpe={stats_pol['annualized_sharpe']:.2f} | "
        f"EW total={stats_eq['total_return']*100:+.2f}%, "
        f"Sharpe={stats_eq['annualized_sharpe']:.2f}"
    )
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(bar_idx, pol_sh, color="#d6336c", lw=1.2, label="GRPO policy")
    ax.plot(bar_idx, eq_sh, color="#1864ab", lw=1.0, label="Equal-weight")
    ax.axhline(0.0, color="#aaaaaa", lw=0.8, ls="--")
    for d in day_starts[1:]:
        ax.axvline(d, color="#cccccc", lw=0.7, ls=":")
    ax.set_ylabel(f"Rolling Sharpe (win={win} bars)")
    ax.set_xlabel("Trading bar index (overnight gaps removed)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    axes[1].set_xticks(day_starts)
    axes[1].set_xticklabels(day_labels, rotation=30, ha="right")

    fig.tight_layout()
    out_curves_c = cfg.output_dir / "backtest_curves_compressed.png"
    fig.savefig(out_curves_c, dpi=130)
    plt.close(fig)
    print(f"saved   : {out_curves_c}")

    # ------------------------------------------------------------------
    # Plot 2: bar chart of headline stats
    # ------------------------------------------------------------------
    fig, axb = plt.subplots(1, 3, figsize=(11, 4))
    labels = ["GRPO", "EW"]
    color = ["#d6336c", "#1864ab"]

    axb[0].bar(labels, [stats_pol["total_return"] * 100,
                        stats_eq["total_return"] * 100],
               color=color)
    axb[0].set_title("Total return (%)")
    axb[0].grid(True, alpha=0.3, axis="y")

    axb[1].bar(labels, [stats_pol["annualized_sharpe"],
                        stats_eq["annualized_sharpe"]],
               color=color)
    axb[1].set_title("Annualized Sharpe")
    axb[1].grid(True, alpha=0.3, axis="y")

    axb[2].bar(labels, [stats_pol["max_drawdown"] * 100,
                        stats_eq["max_drawdown"] * 100],
               color=color)
    axb[2].set_title("Max drawdown (%)")
    axb[2].grid(True, alpha=0.3, axis="y")

    for ax in axb:
        for p in ax.patches:
            h = p.get_height()
            ax.annotate(f"{h:.2f}",
                        xy=(p.get_x() + p.get_width() / 2, h),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", va="bottom", fontsize=9)
    fig.suptitle("Headline stats: GRPO policy vs equal-weight baseline")
    fig.tight_layout()
    out_bar = cfg.output_dir / "backtest_summary_bar.png"
    fig.savefig(out_bar, dpi=130)
    plt.close(fig)
    print(f"saved   : {out_bar}")

    # ------------------------------------------------------------------
    # Console summary
    # ------------------------------------------------------------------
    print("\n=== headline stats ===")
    print(f"  policy : total={stats_pol['total_return']*100:+.2f}%  "
          f"sharpe={stats_pol['annualized_sharpe']:.2f}  "
          f"MDD={stats_pol['max_drawdown']*100:.2f}%  "
          f"hit={stats_pol['hit_rate']*100:.1f}%")
    print(f"  EW     : total={stats_eq['total_return']*100:+.2f}%  "
          f"sharpe={stats_eq['annualized_sharpe']:.2f}  "
          f"MDD={stats_eq['max_drawdown']*100:.2f}%  "
          f"hit={stats_eq['hit_rate']*100:.1f}%")


if __name__ == "__main__":
    main()
