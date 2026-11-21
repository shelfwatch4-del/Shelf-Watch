"""Unit tests for week-based TimeSeriesSplit, parameter sampling, and tuning."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.tune import (
    WeekTimeSeriesSplit,
    evaluate_baselines_cv,
    evaluate_trial_cv,
    sample_params,
    tune,
)


def _make_synthetic_train_frame(n_products: int = 4, n_weeks: int = 15) -> pd.DataFrame:
    """Generate deterministic synthetic training DataFrame across multiple weeks."""
    weeks = pd.date_range("2011-01-03", periods=n_weeks, freq="W-MON")
    rows: list[dict[str, object]] = []

    for p_idx in range(n_products):
        pid = f"PROD_{p_idx + 1}"
        for w_idx, week in enumerate(weeks):
            base_val = 10.0 + p_idx * 5.0 + w_idx * 2.0
            rows.append(
                {
                    "product_id": pid,
                    "week_start": week,
                    "target": float(base_val),
                    "lag_1": float(base_val - 2.0),
                    "lag_2": float(base_val - 4.0),
                    "lag_3": float(base_val - 6.0),
                    "lag_4": float(base_val - 8.0),
                    "roll_mean_4": float(max(1.0, base_val - 5.0)),
                    "roll_mean_8": float(max(1.0, base_val - 7.0)),
                    "f1": float(w_idx),
                    "f2": float(p_idx),
                }
            )

    return pd.DataFrame(rows)


def test_week_time_series_split_strictly_preceding_and_disjoint() -> None:
    """Train weeks must strictly precede validation weeks and never overlap."""
    df = _make_synthetic_train_frame(n_products=3, n_weeks=12)
    splitter = WeekTimeSeriesSplit(n_splits=4)
    splits = splitter.split(df, week_col="week_start")

    assert len(splits) == 4

    seen_val_weeks: list[pd.Timestamp] = []

    for fold_idx, (tr_idx, va_idx) in enumerate(splits):
        assert len(tr_idx) > 0
        assert len(va_idx) > 0

        tr_weeks = pd.to_datetime(df.iloc[tr_idx]["week_start"]).dt.normalize().unique()
        va_weeks = pd.to_datetime(df.iloc[va_idx]["week_start"]).dt.normalize().unique()

        # No week appears on both sides
        assert set(tr_weeks).isdisjoint(set(va_weeks))

        # Train weeks strictly precede validation weeks
        assert max(tr_weeks) < min(va_weeks)

        # Validation weeks across folds are non-overlapping in TimeSeriesSplit
        assert set(seen_val_weeks).isdisjoint(set(va_weeks))
        seen_val_weeks.extend(va_weeks)


def test_sample_params_ranges() -> None:
    """Hyperparameter random sampling must conform to specified domains."""
    rng = np.random.RandomState(42)

    # Test Ridge
    for _ in range(20):
        params, scale_col = sample_params("ridge", target_mode="raw", rng=rng)
        assert scale_col is None
        assert 1e-3 <= params["alpha"] <= 1e3

    # Test Ridge scale-aware
    for _ in range(20):
        _, scale_col = sample_params("ridge", target_mode="scale_aware", rng=rng)
        assert scale_col in ("roll_mean_4", "roll_mean_8")

    # Test Random Forest
    depth_values: set[int | None] = set()
    for _ in range(50):
        params, _ = sample_params("random_forest", target_mode="raw", rng=rng)
        assert 100 <= params["n_estimators"] <= 500
        assert 1 <= params["min_samples_leaf"] <= 20
        depth_values.add(params["max_depth"])
        if params["max_depth"] is not None:
            assert 3 <= params["max_depth"] <= 20
    assert None in depth_values  # None is a valid choice

    # Test LightGBM
    for _ in range(50):
        params, _ = sample_params("lightgbm", target_mode="raw", rng=rng)
        assert 8 <= params["num_leaves"] <= 64
        assert 0.01 <= params["learning_rate"] <= 0.2
        assert 100 <= params["n_estimators"] <= 600
        assert 5 <= params["min_child_samples"] <= 80
        assert 0.6 <= params["subsample"] <= 1.0
        assert 0.6 <= params["colsample_bytree"] <= 1.0
        assert 1e-3 <= params["reg_lambda"] <= 10.0
        if params["subsample"] < 1.0:
            assert params.get("subsample_freq") == 1


def test_evaluate_trial_scale_aware_transform_and_clip() -> None:
    """Scale-aware evaluation correctly scales by floored denominator and unscales."""
    df = _make_synthetic_train_frame(n_products=2, n_weeks=10)
    # inject zero or negative rolling mean to test floor at 1.0
    df.loc[0, "roll_mean_4"] = 0.0
    df.loc[1, "roll_mean_4"] = -5.0

    splitter = WeekTimeSeriesSplit(n_splits=3)
    splits = splitter.split(df)

    params = {"alpha": 1.0}
    fold_scores, mean_wape, val_bias, oof = evaluate_trial_cv(
        name="ridge",
        train_df=df,
        features=["lag_1", "lag_2"],
        params=params,
        target_mode="scale_aware",
        scale_col="roll_mean_4",
        splits=splits,
    )

    assert len(fold_scores) == 3
    assert all(np.isfinite(s) for s in fold_scores)
    assert np.isfinite(mean_wape)
    assert np.isfinite(val_bias)
    assert len(oof) > 0
    # Predictions must be non-negative
    assert np.all(oof["predicted"] >= 0.0)


def test_tune_fast_synthetic() -> None:
    """Fast tune execution produces valid trials, best params, and OOF predictions."""
    df = _make_synthetic_train_frame(n_products=2, n_weeks=10)
    res = tune(
        name="ridge",
        train=df,
        features=["lag_1", "lag_2"],
        target_mode="raw",
        n_trials=3,
        seed=42,
        n_splits=3,
    )

    assert res["candidate"] == "ridge_raw"
    assert "alpha" in res["best_params"]
    assert len(res["best_fold_scores"]) == 3
    assert res["best_mean_val_wape"] > 0
    assert len(res["trials_records"]) == 3

    # Exactly one trial has is_best == 1
    best_flags = [t["is_best"] for t in res["trials_records"]]
    assert sum(best_flags) == 1

    # Out of fold predictions must have required columns
    oof = res["best_oof_preds"]
    assert set(["product_id", "week_start", "actual", "predicted"]).issubset(oof.columns)
    assert np.all(np.isfinite(oof["predicted"]))


def test_evaluate_baselines_cv() -> None:
    """Baseline CV evaluation returns valid metrics for moving_avg_4 and naive."""
    df = _make_synthetic_train_frame(n_products=2, n_weeks=10)
    baselines = evaluate_baselines_cv(df, n_splits=3)

    assert "moving_avg_4" in baselines
    assert "naive" in baselines
    assert baselines["moving_avg_4"]["mean_val_wape"] > 0
    assert np.isfinite(baselines["moving_avg_4"]["val_bias"])
