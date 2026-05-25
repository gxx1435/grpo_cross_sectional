"""
Quick diagnostic for the per-section weight distribution.

Reads ``files/backtest_out_post_train/sampled_sections_perstock.csv`` and
- dumps three minimal (stock_id, weight) tables to the same folder
- prints distribution stats per section
- fits a few candidate parametric shapes (uniform on [a,b], exponential,
  truncated normal) by moments and reports which one fits best
- also reports the closed-form prediction implied by the policy:
      weight_i ∝ alpha_base + alpha_scale * score_i
  so a "uniform scores" assumption would give weights uniformly distributed
  on [1/S, 11/S] where S = sum(1 + 10 * score).
- saves a histogram + sorted-weight (Lorenz-style) plot per section.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT  = HERE / "files" / "backtest_out_post_train"

src = OUT / "sampled_sections_perstock.csv"
df  = pd.read_csv(src)

# --- 1. three minimal tables --------------------------------------------------
for k in (1, 2, 3):
    sub = (df[df["section"] == k]
             .sort_values("weight", ascending=False)
             .loc[:, ["stock_id", "weight"]]
             .reset_index(drop=True))
    sub.index += 1
    sub.index.name = "rank"
    fname = OUT / f"section{k}_weights.csv"
    sub.to_csv(fname)
    print(f"\n=== Section {k} (sorted by weight, desc) ===")
    print(sub.to_string(float_format=lambda v: f"{v:.6f}"))

# --- 2. distribution diagnostics ---------------------------------------------
def fit_diagnostics(w: np.ndarray, label: str) -> dict:
    w_sorted = np.sort(w)
    n = len(w_sorted)
    stats = {
        "section": label,
        "N": n,
        "min": w_sorted.min(),
        "p10": np.quantile(w_sorted, 0.10),
        "median": np.median(w_sorted),
        "mean": w_sorted.mean(),
        "p90": np.quantile(w_sorted, 0.90),
        "max": w_sorted.max(),
        "std": w_sorted.std(ddof=0),
        "skew": ((w_sorted - w_sorted.mean()) ** 3).mean()
                 / (w_sorted.std(ddof=0) ** 3 + 1e-12),
        "kurt_excess": ((w_sorted - w_sorted.mean()) ** 4).mean()
                       / (w_sorted.std(ddof=0) ** 4 + 1e-12) - 3.0,
        "HHI": float((w_sorted ** 2).sum()),
        "ratio_max_min": w_sorted.max() / max(w_sorted.min(), 1e-12),
    }
    # KS-style distance vs Uniform([min, max])
    u_cdf = (w_sorted - w_sorted.min()) / max(w_sorted.max() - w_sorted.min(), 1e-12)
    emp_cdf = np.arange(1, n + 1) / n
    stats["ks_vs_uniform"] = float(np.max(np.abs(u_cdf - emp_cdf)))
    # KS-style distance vs Exponential w/ same mean
    lam = 1.0 / max(w_sorted.mean(), 1e-12)
    exp_cdf = 1.0 - np.exp(-lam * w_sorted)
    stats["ks_vs_exponential"] = float(np.max(np.abs(exp_cdf - emp_cdf)))
    return stats


rows = []
for k in (1, 2, 3):
    w = df.loc[df["section"] == k, "weight"].to_numpy()
    rows.append(fit_diagnostics(w, f"section {k}"))
diag = pd.DataFrame(rows)
diag.to_csv(OUT / "section_weight_distribution_diag.csv", index=False)
print("\n=== distribution diagnostics ===")
print(diag.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

# --- 3. histogram + sorted ladder plot ---------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(12, 6.5))
for k in (1, 2, 3):
    w = np.sort(df.loc[df["section"] == k, "weight"].to_numpy())
    ax = axes[0, k - 1]
    ax.hist(w, bins=10, color="#d6336c", edgecolor="white", alpha=0.85)
    ax.axvline(1.0 / len(w), color="#1864ab", ls="--", lw=1.0,
               label=f"equal-weight 1/N = {1/len(w):.4f}")
    ax.set_title(f"Section {k}: weight histogram")
    ax.set_xlabel("weight")
    ax.set_ylabel("count")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1, k - 1]
    rank = np.arange(1, len(w) + 1)
    ax.plot(rank, w[::-1], "-o", color="#d6336c", ms=3, label="GRPO weight")
    a, b = w.min(), w.max()
    ax.plot(rank, np.linspace(b, a, len(w)), color="#888", ls="--",
            lw=1.0, label=f"linear ramp [{a:.4f}, {b:.4f}]")
    ax.axhline(1.0 / len(w), color="#1864ab", ls=":",
               label="equal-weight 1/N")
    ax.set_title(f"Section {k}: sorted weights (rank 1 = heaviest)")
    ax.set_xlabel("rank by score")
    ax.set_ylabel("weight")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

fig.suptitle("GRPO policy weight distribution -- 3 sampled OOS cross sections",
             fontsize=12)
fig.tight_layout()
out_png = OUT / "section_weight_distribution.png"
fig.savefig(out_png, dpi=130)
plt.close(fig)
print(f"\nsaved -> {out_png}")
