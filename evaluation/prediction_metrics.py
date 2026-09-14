from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def daily_ic(pred: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    m = np.isfinite(pred) & np.isfinite(y)
    if m.sum() < 8:
        return {"IC": np.nan, "RankIC": np.nan, "MSE": np.nan, "MAE": np.nan}
    p, t = pred[m], y[m]
    ic = float(np.corrcoef(p, t)[0, 1]) if np.std(p) > 1e-8 and np.std(t) > 1e-8 else np.nan
    ric = float(pd.Series(p).rank().corr(pd.Series(t).rank()))
    return {
        "IC": ic,
        "RankIC": ric,
        "MSE": float(np.mean((p - t) ** 2)),
        "MAE": float(np.mean(np.abs(p - t))),
    }


def ranking_classification_metrics(pred: np.ndarray, y: np.ndarray, top_k: int = 30) -> Dict[str, float]:
    """
    Daily ranking / classification metrics on valid names.

    TopK F1: predicted TopK vs realized TopK (equal K ⇒ P=R=F1=overlap/K).
    Dir F1: treat ŷ>0 as positive class vs y>0.
    """
    p = np.asarray(pred, dtype=np.float64).ravel()
    t = np.asarray(y, dtype=np.float64).ravel()
    m = np.isfinite(p) & np.isfinite(t)
    out = {
        "dir_acc": np.nan,
        "dir_precision": np.nan,
        "dir_recall": np.nan,
        "dir_f1": np.nan,
        "topk_precision": np.nan,
        "topk_recall": np.nan,
        "topk_f1": np.nan,
        "topk_overlap": np.nan,
    }
    if int(m.sum()) < max(int(top_k), 8):
        return out
    s, r = p[m], t[m]
    k = min(int(top_k), len(s))
    pred_top = set(np.argsort(s)[-k:].tolist())
    true_top = set(np.argsort(r)[-k:].tolist())
    overlap = len(pred_top & true_top)
    prec = overlap / float(k)
    rec = overlap / float(k)
    yhat = s > 0
    ypos = r > 0
    tp = float(np.sum(yhat & ypos))
    fp = float(np.sum(yhat & ~ypos))
    fn = float(np.sum(~yhat & ypos))
    dprec = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    drec = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    df1 = np.nan if not (np.isfinite(dprec) and np.isfinite(drec) and (dprec + drec) > 0) else 2 * dprec * drec / (dprec + drec)
    nz = (np.abs(s) > 1e-12) & (np.abs(r) > 1e-12)
    dir_acc = float((np.sign(s[nz]) == np.sign(r[nz])).mean()) if nz.any() else np.nan
    out.update(
        {
            "dir_acc": dir_acc,
            "dir_precision": float(dprec) if dprec == dprec else np.nan,
            "dir_recall": float(drec) if drec == drec else np.nan,
            "dir_f1": float(df1) if df1 == df1 else np.nan,
            "topk_precision": float(prec),
            "topk_recall": float(rec),
            "topk_f1": float(prec),
            "topk_overlap": float(overlap),
        }
    )
    return out


def top30_stats(pred: np.ndarray, y: np.ndarray, idx: np.ndarray, valid: np.ndarray) -> Dict[str, float]:
    yv = y[valid]
    yt = y[idx]
    hit = float(np.mean(yt > 0)) if len(yt) else np.nan
    return {
        "top30_mean_true": float(np.nanmean(yt)),
        "universe_mean_true": float(np.nanmean(yv)) if valid.any() else np.nan,
        "top30_excess": float(np.nanmean(yt) - np.nanmean(yv)) if valid.any() else np.nan,
        "top30_hit_rate": hit,
    }
