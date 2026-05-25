"""
Three-way NAV comparison: Equal-weight vs PatchTST GRPO vs Transformer GRPO.

Reads stitched OOS bar returns from the 202604 small-window regime experiments
(same universe / schedule; only model_type differs).

Output:
  files/experiments/model_compare_r30_regime_202604/three_way_curves.png
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "files"))
from backtest import BARS_PER_YEAR, annualized_stats

ROOT = Path(__file__).resolve().parent / "files" / "experiments"
PATCHTST = ROOT / "walkforward_patchtst_r30_regime_202604" / "backtest" / "stitched_bar_returns.csv"
TRANSFORMER = ROOT / "walkforward_transformer_r30_regime_202604" / "backtest" / "stitched_bar_returns.csv"
OUT_DIR = ROOT / "model_compare_r30_regime_202604"


def load_returns(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["ts"] = pd.to_datetime(df["ts"])
    return df


def plot_three_way(
    patchtst: pd.DataFrame,
    transformer: pd.DataFrame,
    out_path: Path,
) -> dict:
    merged = patchtst[["ts", "net_pr", "ew_net_pr"]].rename(columns={"net_pr": "patchtst_pr"})
    merged = merged.merge(
        transformer[["ts", "net_pr"]].rename(columns={"net_pr": "transformer_pr"}),
        on="ts",
        how="inner",
    )
    if len(merged) == 0:
        raise ValueError("No overlapping timestamps between PatchTST and Transformer runs.")

    bar_idx = np.arange(len(merged))
    ew_nav = (1.0 + merged["ew_net_pr"]).cumprod()
    pt_nav = (1.0 + merged["patchtst_pr"]).cumprod()
    tr_nav = (1.0 + merged["transformer_pr"]).cumprod()

    st_ew = annualized_stats(merged["ew_net_pr"].to_numpy(), BARS_PER_YEAR)
    st_pt = annualized_stats(merged["patchtst_pr"].to_numpy(), BARS_PER_YEAR)
    st_tr = annualized_stats(merged["transformer_pr"].to_numpy(), BARS_PER_YEAR)

    dates = merged["ts"].dt.normalize()
    day_change = np.r_[True, dates.values[1:] != dates.values[:-1]]
    day_starts = bar_idx[day_change]
    day_labels = merged["ts"].iloc[day_starts].dt.strftime("%m-%d").tolist()

    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.plot(bar_idx, ew_nav, color="#1864ab", lw=1.6, label="Equal-weight (long-only)")
    ax.plot(bar_idx, pt_nav, color="#d6336c", lw=1.6, label="PatchTST GRPO Top5")
    ax.plot(bar_idx, tr_nav, color="#e67700", lw=1.6, label="Transformer GRPO Top5")
    ax.axhline(1.0, color="#aaa", ls="--", lw=0.8)
    for d in day_starts[1:]:
        ax.axvline(d, color="#ccc", ls=":", lw=0.7)

    ax.set_ylabel("Cumulative NAV")
    ax.set_xlabel("Trading bar index")
    ax.set_xticks(day_starts)
    ax.set_xticklabels(day_labels, rotation=30, ha="right")
    ax.set_title(
        f"OOS 2026-04-07 ~ 2026-05-08  |  "
        f"EW {st_ew['total_return'] * 100:+.1f}% (Sharpe {st_ew['annualized_sharpe']:.1f})  "
        f"PatchTST {st_pt['total_return'] * 100:+.1f}% ({st_pt['annualized_sharpe']:.1f})  "
        f"Transformer {st_tr['total_return'] * 100:+.1f}% ({st_tr['annualized_sharpe']:.1f})"
    )
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)

    summary = pd.DataFrame([
        {"model": "equal_weight", **st_ew},
        {"model": "patchtst_grpo", **st_pt},
        {"model": "transformer_grpo", **st_tr},
    ])
    return {"bars": len(merged), "summary": summary}


if __name__ == "__main__":
    if not PATCHTST.is_file():
        raise SystemExit(f"Missing PatchTST returns: {PATCHTST}")
    if not TRANSFORMER.is_file():
        raise SystemExit(f"Missing Transformer returns: {TRANSFORMER}")

    pt = load_returns(PATCHTST)
    tr = load_returns(TRANSFORMER)
    out_png = OUT_DIR / "three_way_curves.png"
    result = plot_three_way(pt, tr, out_png)
    result["summary"].to_csv(OUT_DIR / "three_way_summary.csv", index=False)
    print(f"Saved -> {out_png}  ({result['bars']} bars)")
    print(result["summary"].to_string(index=False))
