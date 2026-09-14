"""2025-05 SFT with train-only feature z-score: LSTM and gated_residual only."""

from __future__ import annotations

import argparse
import copy
import gc
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch

from data.splits import sequential_retrain_windows, split_for_test_month
from data.store import ResearchStore
from evaluation.experiment_analysis import refresh_month
from evaluation.prediction_metrics import daily_ic, ranking_classification_metrics, top30_stats
from experiments.engine import eval_predictor, predict_day, train_predictor
from models.alpha_predictor import build_predictor
from utils.config import load_config, refresh_prediction_train_cfg
from utils.gpu import configure_gpu
from utils.logging import Tee, log
from utils.runtime import count_params
from utils.seed import set_seed

MONTH = "2025-05"
OOS = "sequential_oos_weekly_retrain"
MODELS = ("lstm", "gated_residual")
MONTH_DIR = Path("results/full") / OOS / f"test_month={MONTH}"


def _baseline_row(path: Path, model: str) -> dict:
    if not path.is_file():
        return {}
    df = pd.read_csv(path)
    hit = df[df["model"].astype(str) == model]
    return hit.iloc[-1].to_dict() if len(hit) else {}


def _eval_test(model, store, days, device, top_k: int, feat_scaler=None) -> tuple[dict, pd.DataFrame]:
    rows = []
    for day in days:
        try:
            y, _ = store.labels(day)
            pred, valid, _ = predict_day(model, store, day, device, feat_scaler=feat_scaler)
            if int(valid.sum()) < 8:
                continue
            idx = np.argpartition(-np.where(valid, pred, -1e18), min(top_k, int(valid.sum())) - 1)[: min(top_k, int(valid.sum()))]
            rec = {
                "date": str(pd.Timestamp(day).date()),
                "model": getattr(model, "fusion_name", "model"),
                **daily_ic(pred[valid], y[valid]),
                **ranking_classification_metrics(pred[valid], y[valid], top_k=top_k),
                **top30_stats(pred, y, idx, valid),
            }
            rows.append(rec)
        except Exception as e:
            log(f"  test skip {pd.Timestamp(day).date()}: {e}")
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    df = pd.DataFrame(rows)
    mean = {}
    if len(df):
        for c in ("MSE", "MAE", "IC", "RankIC", "topk_f1"):
            if c in df.columns:
                mean[c] = float(pd.to_numeric(df[c], errors="coerce").mean())
        mean["n"] = int(df["date"].nunique()) if "date" in df.columns else len(df)
    return mean, df


def _train_one(name: str, cfg: dict, store, train_days, val_days, device, out_dir: Path):
    local = copy.deepcopy(cfg)
    refresh_prediction_train_cfg(local)
    local["prediction"]["feat_standardize"] = True
    local["prediction"]["rank_weight"] = 0.0
    local["prediction"]["sft_cs_zscore"] = False
    local["prediction"]["val_select"] = "mse"
    model = build_predictor(name, local, store.feat_dim).to(device)
    log(f"==== featstd SFT {name} n_train={len(train_days)} n_val={len(val_days)} params={count_params(model)}")
    stats = train_predictor(
        model,
        store,
        train_days,
        local,
        device,
        val_days,
        resume_path=out_dir / f"sft_resume_{name}.pt",
        always_nl=True,
    )
    torch.save(model.state_dict(), out_dir / f"alpha_{name}_featstd.pt")
    scaler = getattr(model, "feat_scaler", None)
    if scaler is not None:
        scaler.save(out_dir / f"feat_scaler_{name}.npz")
    hist = stats.get("val_history") or []
    if hist:
        pd.DataFrame([{"model": f"{name}_featstd", "month": MONTH, "oos": OOS, **r} for r in hist]).to_csv(
            out_dir / f"alpha_val_{name}_featstd.csv", index=False
        )
    return model, stats


def _best_val(stats: dict) -> dict:
    hist = stats.get("val_history") or []
    if not hist:
        return {k.replace("val_", ""): v for k, v in stats.items() if k.startswith("val_")}
    best = min(hist, key=lambda r: float(r.get("MSE", 1e18)))
    return dict(best)


def _pack_compare(model: str, split: str, variant: str, met: dict) -> dict:
    return {
        "model": model,
        "split": split,
        "variant": variant,
        "MSE": met.get("MSE", met.get("val_MSE")),
        "IC": met.get("IC", met.get("val_IC")),
        "RankIC": met.get("RankIC", met.get("val_RankIC")),
        "topk_f1": met.get("topk_f1", met.get("val_topk_f1")),
        "n": met.get("n", met.get("val_n")),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--month", default=MONTH)
    args = ap.parse_args()
    if args.month != MONTH:
        raise SystemExit("this runner maps 2025-05 checkpoints / docs only")
    cfg = load_config()
    cfg.setdefault("gpu", {})["progress_gpu_poll_sec"] = 5.0
    set_seed(int(cfg["experiment"]["seed"]), False, True)
    out_dir = MONTH_DIR / "featstd_sft"
    out_dir.mkdir(parents=True, exist_ok=True)
    Tee.install(out_dir / "run.log")
    configure_gpu(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    store = ResearchStore(cfg)
    split = split_for_test_month(store.days, pd.Timestamp(f"{MONTH}-01"), int(cfg["walkforward"]["train_offset_months"]))
    windows = sequential_retrain_windows(split)
    w0 = windows[0]
    top_k = int(cfg["portfolio"]["top_k"])
    tab = MONTH_DIR / "tables"
    tab.mkdir(parents=True, exist_ok=True)

    trained = {}
    val_best = {}
    for name in MODELS:
        model, stats = _train_one(name, cfg, store, w0["train_days"], w0["val_days"], device, out_dir)
        trained[name] = model
        val_best[name] = _best_val(stats)
        log(f"  {name} BEST VAL MSE={val_best[name].get('MSE')} IC={val_best[name].get('IC')} RankIC={val_best[name].get('RankIC')}")
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # gated_residual Friday retrains (same sequential OOS rule as the main run)
    weekly = {pd.Timestamp(w0["effective_from"]).normalize(): trained["gated_residual"]}
    for w in windows:
        if not w.get("includes_test_realized") or not w.get("effective_from"):
            continue
        log(f"==== featstd Friday SFT gated_residual cutoff={w['cutoff']} effective_from={w['effective_from']}")
        model, stats = _train_one("gated_residual", cfg, store, w["train_days"], w["val_days"], device, out_dir)
        torch.save(model.state_dict(), out_dir / f"alpha_gated_residual_featstd_{w['cutoff']}.pt")
        sc = getattr(model, "feat_scaler", None)
        if sc is not None:
            sc.save(out_dir / f"feat_scaler_gated_residual_{w['cutoff']}.npz")
        weekly[pd.Timestamp(w["effective_from"]).normalize()] = model
        trained["gated_residual"] = model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    cuts = sorted(weekly)
    test_rows = []
    test_means = {}
    for name in MODELS:
        if name == "lstm":
            mean, df = _eval_test(trained["lstm"], store, split["test_days"], device, top_k)
            df["model"] = "lstm_featstd"
            test_means[name] = mean
            test_rows.append(df)
        else:
            rows = []
            for day in split["test_days"]:
                d = pd.Timestamp(day).normalize()
                pick = weekly[cuts[0]]
                for c in cuts:
                    if d >= c:
                        pick = weekly[c]
                y, _ = store.labels(day)
                pred, valid, _ = predict_day(pick, store, day, device)
                if int(valid.sum()) < 8:
                    continue
                idx = np.argpartition(-np.where(valid, pred, -1e18), min(top_k, int(valid.sum())) - 1)[: min(top_k, int(valid.sum()))]
                rows.append(
                    {
                        "date": str(d.date()),
                        "model": "gated_residual_featstd",
                        **daily_ic(pred[valid], y[valid]),
                        **ranking_classification_metrics(pred[valid], y[valid], top_k=top_k),
                        **top30_stats(pred, y, idx, valid),
                    }
                )
                gc.collect()
            df = pd.DataFrame(rows)
            mean = {c: float(pd.to_numeric(df[c], errors="coerce").mean()) for c in ("MSE", "MAE", "IC", "RankIC", "topk_f1") if c in df.columns}
            mean["n"] = int(df["date"].nunique()) if len(df) else 0
            test_means[name] = mean
            test_rows.append(df)
        log(f"  {name} TEST featstd {test_means[name]}")

    if test_rows:
        pd.concat(test_rows, ignore_index=True).to_csv(tab / "featstd_prediction_metrics_daily.csv", index=False)

    base_val = _baseline_row(tab / "val_best.csv", "lstm"), _baseline_row(tab / "val_best.csv", "gated_residual")
    base_test = _baseline_row(tab / "prediction_metrics_mean.csv", "lstm"), _baseline_row(tab / "prediction_metrics_mean.csv", "gated_residual")
    cmp_rows = []
    for i, name in enumerate(MODELS):
        cmp_rows.append(_pack_compare(name, "val", "baseline", base_val[i]))
        cmp_rows.append(_pack_compare(name, "val", "featstd", val_best[name]))
        cmp_rows.append(_pack_compare(name, "test", "baseline", base_test[i]))
        cmp_rows.append(_pack_compare(name, "test", "featstd", test_means[name]))
    cmp = pd.DataFrame(cmp_rows)
    cmp.to_csv(tab / "featstd_sft_compare.csv", index=False)

    lines = ["结论（相对当月已有 baseline）:"]
    for name in MODELS:
        sub = cmp[cmp["model"] == name]
        for split_n in ("val", "test"):
            b = sub[(sub["split"] == split_n) & (sub["variant"] == "baseline")]
            f = sub[(sub["split"] == split_n) & (sub["variant"] == "featstd")]
            if not len(b) or not len(f):
                continue
            def g(df, c):
                v = pd.to_numeric(df.iloc[0].get(c), errors="coerce")
                return float(v) if pd.notna(v) else float("nan")
            bits = []
            for c, better in (("MSE", "down"), ("IC", "up"), ("RankIC", "up"), ("topk_f1", "up")):
                bv, fv = g(b, c), g(f, c)
                if not np.isfinite(bv) or not np.isfinite(fv):
                    continue
                improved = (fv < bv) if better == "down" else (fv > bv)
                bits.append(f"{c} {bv:.4g}→{fv:.4g} ({'升' if improved else '降'})")
            lines.append(f"- `{name}` {split_n}: " + "；".join(bits))
    (tab / "featstd_sft_note.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log("\n".join(lines))
    refresh_month(Path("results/full"), OOS, MONTH, {"source": "featstd SFT LSTM/gated_residual", "last_unit": "A-featstd"})
    log(f"wrote {tab / 'featstd_sft_compare.csv'}")


if __name__ == "__main__":
    main()
