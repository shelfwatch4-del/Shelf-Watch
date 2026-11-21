"""Unit tests for chronological splitting, metrics, and baseline predictors."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.train import (
    MODEL_NAMES,
    evaluate,
    fit_predict,
    make_model,
    predict_moving_avg_4,
    predict_naive,
    split_chronological,
)


def _make_feature_frame(n_train_weeks: int = 10, n_test_weeks: int = 8) -> pd.DataFrame:
    """Create a deterministic feature frame spanning train and test weeks."""
    total_weeks = n_train_weeks + n_test_weeks
    test_start = pd.Timestamp("2011-10-10")
    first_week = test_start - pd.Timedelta(weeks=n_train_weeks)
    weeks = pd.date_range(first_week, periods=total_weeks, freq="W-MON")

    rows: list[dict[str, object]] = []
    for pid in ("PROD_A", "PROD_B"):
        for idx, week in enumerate(weeks):
            rows.append(
                {
                    "product_id": pid,
                    "week_start": week,
                    "target": float(10 + idx),
                    "lag_1": float(9 + idx),
                    "lag_2": float(8 + idx),
                    "lag_3": float(7 + idx),
                    "lag_4": float(6 + idx),
                    "roll_mean_4": float(7.5 + idx),
                }
            )
    return pd.DataFrame(rows)


def test_split_chronological_strict() -> None:
    """Split must be strictly chronological with exactly 8 distinct test weeks."""
    df = _make_feature_frame(n_train_weeks=10, n_test_weeks=8)
    train, test = split_chronological(df, test_start="2011-10-10")

    train_weeks = pd.to_datetime(train["week_start"])
    test_weeks = pd.to_datetime(test["week_start"])

    assert train_weeks.max() < test_weeks.min()
    assert test_weeks.min() == pd.Timestamp("2011-10-10")
    assert test_weeks.nunique() == 8
    assert len(train) == 20
    assert len(test) == 16


def test_split_chronological_requires_eight_test_weeks() -> None:
    """Split must fail when the test period does not have 8 distinct weeks."""
    df = _make_feature_frame(n_train_weeks=10, n_test_weeks=7)
    with pytest.raises(AssertionError, match="8 distinct test weeks"):
        split_chronological(df, test_start="2011-10-10")


def test_evaluate_hand_computed() -> None:
    """Evaluate must match hand-computed MAE, RMSE, WAPE, and bias on 4 rows."""
    # Hand-computed 4-row example:
    #   y_true = [10, 0, 20, 10]  -> sum(y_true) = 40
    #   y_pred = [14, 2, 16, 12]
    #   errors (y_pred - y_true) = [+4, +2, -4, +2]
    #   |errors|                 = [ 4,  2,  4,  2] -> sum = 12
    #   errors^2                 = [16,  4, 16,  4] -> sum = 40
    #   MAE  = 12 / 4 = 3.0
    #   RMSE = sqrt(40 / 4) = sqrt(10.0)
    #   WAPE = 12 / 40 = 0.3
    #   bias = (+4 + 2 - 4 + 2) / 4 = 1.0
    y_true = np.array([10.0, 0.0, 20.0, 10.0])
    y_pred = np.array([14.0, 2.0, 16.0, 12.0])

    metrics = evaluate(y_true, y_pred)

    assert set(metrics.keys()) == {"MAE", "RMSE", "WAPE", "bias"}
    assert metrics["MAE"] == pytest.approx(3.0)
    assert metrics["RMSE"] == pytest.approx(math.sqrt(10.0))
    assert metrics["WAPE"] == pytest.approx(0.3)
    assert metrics["bias"] == pytest.approx(1.0)


def test_evaluate_all_zero_actual_wape_nan() -> None:
    """WAPE must return NaN (not raise) when all actual values are zero."""
    y_true = np.array([0.0, 0.0, 0.0, 0.0])
    y_pred = np.array([1.0, 0.0, 2.0, 1.0])

    metrics = evaluate(y_true, y_pred)

    assert np.isnan(metrics["WAPE"])
    assert metrics["MAE"] == pytest.approx(1.0)
    assert metrics["RMSE"] == pytest.approx(math.sqrt(1.5))
    assert metrics["bias"] == pytest.approx(1.0)


def test_baselines_clip_at_zero_and_naive_fallback() -> None:
    """Baselines must clip predictions at 0 and naive must fall back when lag_1 is NaN."""
    df = pd.DataFrame(
        {
            "lag_1": [5.0, np.nan, np.nan, -3.0],
            "lag_2": [4.0, 12.0, np.nan, 2.0],
            "lag_3": [3.0, 11.0, -4.0, 1.0],
            "lag_4": [2.0, 10.0, 8.0, 0.0],
            "roll_mean_4": [3.5, -1.5, 0.0, 7.25],
        }
    )

    naive_preds = predict_naive(df)
    ma4_preds = predict_moving_avg_4(df)

    np.testing.assert_allclose(naive_preds, np.array([5.0, 12.0, 0.0, 0.0]))
    np.testing.assert_allclose(ma4_preds, np.array([3.5, 0.0, 0.0, 7.25]))
    assert np.all(naive_preds >= 0.0)
    assert np.all(ma4_preds >= 0.0)


def test_make_model_invalid_name() -> None:
    """make_model must raise ValueError for unsupported model names."""
    with pytest.raises(ValueError, match="Unknown model name"):
        make_model("xgboost")


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_models_fit_and_predict_finite(model_name: str) -> None:
    """Each model must fit on a tiny frame and predict finite, clipped values."""
    features = ["f1", "f2", "f3"]
    train = pd.DataFrame(
        {
            "f1": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0],
            "f2": [10.0, 12.0, 14.0, np.nan, 18.0, 20.0],
            "f3": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
            "target": [5.0, 7.0, 9.0, 11.0, 13.0, 15.0],
        }
    )
    test = pd.DataFrame(
        {
            "f1": [np.nan, 7.0, -100.0],
            "f2": [16.0, np.nan, -200.0],
            "f3": [2.0, 3.5, -50.0],
        }
    )

    preds = fit_predict(model_name, None, train, test, features=features)

    assert preds.shape == (len(test),)
    assert np.all(np.isfinite(preds))
    assert np.all(preds >= 0.0)


def test_ridge_imputer_uses_train_median_only() -> None:
    """Ridge pipeline imputer statistics must come from train only, never test."""
    features = ["f1", "f2"]
    train = pd.DataFrame(
        {
            "f1": [2.0, 4.0, 6.0, 4.0],
            "f2": [1.0, 2.0, 3.0, 2.0],
            "target": [4.0, 8.0, 12.0, 8.0],
        }
    )
    # Test contains NaN and an extreme outlier (1000.0) that would shift a test median.
    test = pd.DataFrame(
        {
            "f1": [np.nan, 4.0, 1000.0],
            "f2": [2.0, 2.0, 1000.0],
        }
    )

    pipeline = make_model("ridge")
    pipeline.fit(train[features], train["target"])

    imputer_stats = pipeline.named_steps["imputer"].statistics_
    np.testing.assert_allclose(imputer_stats, np.array([4.0, 2.0]))

    preds = pipeline.predict(test[features])
    # Row 0 (f1=NaN, f2=2.0) must get the exact same prediction as Row 1 (f1=4.0, f2=2.0).
    assert preds[0] == pytest.approx(preds[1])


