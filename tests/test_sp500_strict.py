from __future__ import annotations

import zipfile

import numpy as np
import pandas as pd
import torch

from backtest.engine import run_path
from data.loaders import enumerate_month_members, read_month_member, regular_session
from data.minute_features import add_cross_sectional_features, compute_symbol_features, feature_names
from data.splits import infer_test_months, split_for_test_month
from data.store import ResearchStore
from flow_matching.simplex import tangent_project
from models.alpha_predictor import STRATEGY_NAME, build_predictor
from portfolio.constraints import cap_and_simplex
from utils.config import load_config


def test_config_enables_only_required_paths() -> None:
    cfg = load_config()
    assert cfg["oos_modes"] == ["strict_fixed_oos"]
    assert cfg["prediction"]["models"] == [STRATEGY_NAME]
    assert cfg["prediction"]["primary_model"] == STRATEGY_NAME


def test_real_zip_member_contract() -> None:
    cfg = load_config()
    members = enumerate_month_members(cfg["paths"]["source_zip"])
    assert len(members) == 25
    assert members[0]["source_month"] == "2024-05"
    with zipfile.ZipFile(cfg["paths"]["source_zip"]) as zf, zf.open(members[0]["source_file"]) as fh:
        raw = pd.read_csv(fh, compression="zstd", nrows=1000)
    assert {"ts_event", "symbol", "open", "high", "low", "close", "volume"}.issubset(raw.columns)


def test_features_are_prefix_causal_and_exclude_overnight() -> None:
    cfg = load_config()
    n = 900
    ts = pd.date_range("2025-01-02 09:30", periods=n, freq="min")
    base = pd.DataFrame(
        {
            "stock_code": "AAPL",
            "minute_timestamp": ts.tz_localize("America/New_York").tz_convert("UTC"),
            "local_timestamp": ts,
            "trading_date": ts.normalize(),
            "open": np.linspace(100, 101, n),
            "high": np.linspace(100.1, 101.1, n),
            "low": np.linspace(99.9, 100.9, n),
            "close": np.linspace(100.02, 101.02, n),
            "volume": np.arange(n) + 100,
        }
    )
    full = add_cross_sectional_features(compute_symbol_features(base, cfg), cfg)
    prefix = add_cross_sectional_features(compute_symbol_features(base.iloc[:850], cfg), cfg)
    names = feature_names(cfg)
    np.testing.assert_allclose(full.loc[:849, names], prefix[names], equal_nan=True)
    assert all("overnight" not in name for name in names)


def test_monthly_split_has_exact_twelve_train_months() -> None:
    days = pd.bdate_range("2024-05-01", "2025-06-30")
    split = split_for_test_month(days, pd.Timestamp("2025-06-01"), 13)
    assert split["train_month_count"] == 12
    assert split["train_start_date"] == "2024-05-01"
    assert split["train_end_date"] == "2025-04-30"
    assert split["validation_start_date"] == "2025-05-01"


def test_partial_terminal_month_is_not_an_eligible_test_month() -> None:
    cfg = load_config()
    days = pd.bdate_range("2024-05-01", "2026-05-08")
    months = infer_test_months(days, cfg)
    assert str(pd.Period(months[-1], freq="M")) == "2026-04"


def test_model_forward_and_attention_statistics() -> None:
    cfg = load_config(overrides={"calendar": {"lookback_days": 1, "bars_per_day": 60}, "prediction": {"patch": 10, "d_model": 16, "n_heads": 4, "n_layers_temporal": 1, "n_layers_cross": 1, "stock_chunk": 4}})
    model = build_predictor(STRATEGY_NAME, cfg, feat_dim=8)
    x = torch.randn(7, 60, 8)
    mask = torch.ones(7, 60)
    y = model(x, mask=mask)
    assert y.shape == (7,)
    assert "temporal_patch_attention" in model.last_aux
    assert "cross_stock_attention_entropy" in model.last_aux
    assert torch.isclose(model.last_aux["temporal_patch_attention"].sum(), torch.tensor(1.0), atol=1e-5)


def test_tangent_projection_sums_to_zero() -> None:
    value = tangent_project(torch.randn(11, 30))
    assert torch.allclose(value.sum(dim=-1), torch.zeros(11), atol=1e-6)


def test_turnover_aligns_changing_topk_by_security_id() -> None:
    cfg = load_config()
    result = run_path(
        list(pd.to_datetime(["2025-01-02", "2025-01-03"])),
        [np.array([1.0, 0.0]), np.array([1.0, 0.0])],
        [np.array([0.01, 0.02]), np.array([0.03, 0.04])],
        cfg,
        asset_ids=[["A", "B"], ["B", "C"]],
    )
    assert result.loc[0, "turnover"] == 0.5
    assert result.loc[1, "turnover"] == 1.0


def test_capped_simplex_preserves_sum_and_cap() -> None:
    weights = cap_and_simplex(np.array([20.0, 2.0, 1.0, -4.0, 0.5]), 0.30)
    assert np.isclose(weights.sum(), 1.0, atol=1e-10)
    assert weights.min() >= 0.0
    assert weights.max() <= 0.30 + 1e-10


def test_decision_universe_does_not_read_target_day_prices() -> None:
    store = ResearchStore.__new__(ResearchStore)
    store.N = 2
    store.lookback_days = 2
    store.days = list(pd.to_datetime(["2025-01-02", "2025-01-03", "2025-01-06"]))
    store.session_end_slot = np.array([1, 1, 1], dtype=np.int16)
    store.mask = np.ones((3, 2, 2), dtype=np.uint8)
    store.close = np.array([[10.0, 20.0], [11.0, 21.0], [np.nan, np.nan]], dtype=np.float32)
    store.open = np.full((3, 2), np.nan, dtype=np.float32)
    store.target_count = np.zeros((3, 2), dtype=np.int16)
    store.cfg = {"universe": {"min_history_minutes": 1, "min_target_minutes": 1, "history_min_coverage": 0.5}}
    np.testing.assert_array_equal(store.decision_mask(pd.Timestamp("2025-01-06")), np.array([True, True]))
    np.testing.assert_array_equal(store.evaluation_mask(pd.Timestamp("2025-01-06")), np.array([False, False]))
