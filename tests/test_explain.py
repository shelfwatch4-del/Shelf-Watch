"""Unit tests for feature attribution, exact decomposition, and explain helper."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.explain import (
    compute_feature_importance,
    explain_prediction,
    explain_row,
    generate_feature_importance_csv,
)
from src.train import make_model


def _generate_synthetic_data(n_samples: int = 100, n_features: int = 6, seed: int = 42) -> tuple[pd.DataFrame, pd.Series]:
    """Generate reproducible synthetic regression dataset."""
    rng = np.random.RandomState(seed)
    feature_names = [f"feat_{i}" for i in range(n_features)]
    X_mat = rng.randn(n_samples, n_features) * 5.0 + 10.0
    true_weights = rng.uniform(0.5, 3.0, size=n_features)
    y_vec = np.dot(X_mat, true_weights) + 15.0 + rng.randn(n_samples)

    df_X = pd.DataFrame(X_mat, columns=feature_names)
    s_y = pd.Series(y_vec, name="target")
    return df_X, s_y


def test_explain_row_lightgbm_sum_identity_20_rows() -> None:
    """LightGBM: base_value + sum(contributions) must equal model_output within 1e-6 on 20 rows."""
    X, y = _generate_synthetic_data(n_samples=80, n_features=5, seed=42)
    model = make_model("lightgbm", params={"n_estimators": 40, "num_leaves": 15})
    model.fit(X, y)

    rng = np.random.RandomState(123)
    random_indices = rng.choice(len(X), size=20, replace=False)

    for idx in random_indices:
        row = X.iloc[idx]
        exp = explain_row(model, row, feature_names=list(X.columns))

        base_val = exp["base_value"]
        contrib_sum = sum(exp["contributions"].values())
        model_out = exp["model_output"]

        diff = abs((base_val + contrib_sum) - model_out)
        assert diff < 1e-6, f"Row {idx}: diff {diff} exceeded 1e-6 tolerance."
        assert exp["scaling_factor"] == pytest.approx(1.0)
        assert exp["unscaled_output"] == pytest.approx(max(0.0, model_out))


def test_explain_row_ridge_pipeline_sum_identity_20_rows() -> None:
    """Ridge Pipeline: base_value + sum(contributions) must equal model_output within 1e-6 on 20 rows."""
    X, y = _generate_synthetic_data(n_samples=80, n_features=5, seed=77)
    # Inject some NaNs to test that imputer and scaler handle missing values properly
    X_with_nan = X.copy()
    X_with_nan.iloc[2, 0] = np.nan
    X_with_nan.iloc[5, 3] = np.nan
    X_with_nan.iloc[10, 1] = np.nan

    model = make_model("ridge", params={"alpha": 10.0})
    model.fit(X_with_nan, y)

    rng = np.random.RandomState(999)
    random_indices = rng.choice(len(X_with_nan), size=20, replace=False)

    for idx in random_indices:
        row = X_with_nan.iloc[idx]
        exp = explain_row(model, row, feature_names=list(X_with_nan.columns))

        base_val = exp["base_value"]
        contrib_sum = sum(exp["contributions"].values())
        model_out = exp["model_output"]

        diff = abs((base_val + contrib_sum) - model_out)
        assert diff < 1e-6, f"Row {idx}: diff {diff} exceeded 1e-6 tolerance."


def test_explain_row_scale_aware_scaling_factor() -> None:
    """Scale-aware targets return contributions in scaled space and report scaling factor."""
    X, y = _generate_synthetic_data(n_samples=40, n_features=4, seed=42)
    model = make_model("lightgbm", params={"n_estimators": 25})
    scale_denom = 12.5
    y_scaled = y / scale_denom
    model.fit(X, y_scaled)

    row = X.iloc[0]
    exp = explain_row(model, row, scale_factor=scale_denom)

    base_val = exp["base_value"]
    contrib_sum = sum(exp["contributions"].values())
    model_out = exp["model_output"]

    assert abs((base_val + contrib_sum) - model_out) < 1e-6
    assert exp["scaling_factor"] == pytest.approx(scale_denom)
    assert exp["unscaled_output"] == pytest.approx(max(0.0, model_out * scale_denom))


def test_explain_row_random_forest_pipeline_sum_identity_20_rows() -> None:
    """Random Forest Pipeline: base_value + sum(contributions) must equal model_output within 1e-6 on 20 rows."""
    X, y = _generate_synthetic_data(n_samples=80, n_features=5, seed=101)
    # Inject NaNs to test median imputer inside RF pipeline
    X_with_nan = X.copy()
    X_with_nan.iloc[3, 1] = np.nan
    X_with_nan.iloc[7, 4] = np.nan

    model = make_model("random_forest", params={"n_estimators": 25, "max_depth": 4})
    model.fit(X_with_nan, y)

    rng = np.random.RandomState(456)
    random_indices = rng.choice(len(X_with_nan), size=20, replace=False)

    for idx in random_indices:
        row = X_with_nan.iloc[idx]
        exp = explain_row(model, row, feature_names=list(X_with_nan.columns))

        base_val = exp["base_value"]
        contrib_sum = sum(exp["contributions"].values())
        model_out = exp["model_output"]

        diff = abs((base_val + contrib_sum) - model_out)
        assert diff < 1e-6, f"Row {idx}: diff {diff} exceeded 1e-6 tolerance."


def test_compute_feature_importance_schema_and_values() -> None:
    """compute_feature_importance must produce non-negative metrics and expected columns."""
    X_train, y_train = _generate_synthetic_data(n_samples=60, n_features=4, seed=1)
    X_test, _ = _generate_synthetic_data(n_samples=20, n_features=4, seed=2)

    model = make_model("lightgbm", params={"n_estimators": 20})
    model.fit(X_train, y_train)

    feature_cols = list(X_train.columns)
    fi = compute_feature_importance(model, X_test, feature_cols, model_name="lightgbm")

    expected_cols = {"model_name", "feature", "feature_group", "gain_importance", "mean_abs_contribution"}
    assert expected_cols.issubset(fi.columns)
    assert len(fi) == len(feature_cols)
    assert (fi["gain_importance"] >= 0.0).all()
    assert (fi["mean_abs_contribution"] >= 0.0).all()


def test_contracts_c16_c17_files_exist_and_valid() -> None:
    """Validate that data/processed/tuning_trials.csv and feature_importance.csv match contracts."""
    from pathlib import Path

    p_trials = Path("data/processed/tuning_trials.csv")
    assert p_trials.exists(), "data/processed/tuning_trials.csv must exist."
    df_trials = pd.read_csv(p_trials)
    expected_trials_cols = ["candidate", "trial_no", "params", "mean_val_wape", "is_best"]
    assert list(df_trials.columns) == expected_trials_cols
    assert len(df_trials) == 180

    p_fi = Path("data/processed/feature_importance.csv")
    assert p_fi.exists(), "data/processed/feature_importance.csv must exist."
    df_fi = pd.read_csv(p_fi)
    expected_fi_cols = {"model_name", "feature", "feature_group", "gain_importance", "mean_abs_contribution"}
    assert expected_fi_cols.issubset(df_fi.columns)
    assert len(df_fi) in (15, 16)


def test_explain_row_ensemble_sum_identity_20_rows() -> None:
    """Ensemble: base_value + sum(contributions) must equal model_output within 1e-6 on 20 rows."""
    feature_names = ["lag_1", "roll_mean_4", "roll_mean_8", "prod_past_mean", "yoy_mean_5w"]
    X, y = _generate_synthetic_data(n_samples=80, n_features=5, seed=333)
    X.columns = feature_names

    model = make_model("ridge", params={"alpha": 50.0})
    model.fit(X, y)

    rng = np.random.RandomState(888)
    random_indices = rng.choice(len(X), size=20, replace=False)

    for idx in random_indices:
        row = X.iloc[idx]
        exp = explain_row(model, row, feature_names=feature_names, is_ensemble=True)

        base_val = exp["base_value"]
        contrib_sum = sum(exp["contributions"].values())
        model_out = exp["model_output"]

        diff = abs((base_val + contrib_sum) - model_out)
        assert diff < 1e-6, f"Row {idx}: diff {diff} exceeded 1e-6 tolerance."

