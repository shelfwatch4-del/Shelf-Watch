"""Unit and integration tests for Task M12 result files (Contracts C2, C3, C4)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.results import (
    compute_product_sigma_error,
    generate_all_m12_results,
)


@pytest.fixture(scope="module")
def result_files() -> dict[str, pd.DataFrame]:
    """Ensure all M12 result files are generated and return their DataFrames."""
    return generate_all_m12_results()


def test_predictions_csv_schema_and_integrity(result_files: dict[str, pd.DataFrame]) -> None:
    """Validate data/processed/predictions.csv schema, candidate row counts, and non-negativity (C2)."""
    p_path = Path("data/processed/predictions.csv")
    assert p_path.exists(), "data/processed/predictions.csv must exist."

    df = pd.read_csv(p_path, dtype={"product_id": str, "model_name": str, "split": str})
    expected_cols = ["product_id", "week_start", "actual", "predicted", "model_name", "split"]
    assert list(df.columns) == expected_cols, f"Columns mismatch. Expected {expected_cols}, got {list(df.columns)}"

    # Check splits
    assert set(df["split"].unique()) == {"val", "test"}

    # Check candidates
    expected_candidates = {
        "naive",
        "moving_avg_4",
        "ridge_raw",
        "random_forest_raw",
        "lightgbm_raw",
        "ridge_scale_aware",
        "random_forest_scale_aware",
        "lightgbm_scale_aware",
        "ensemble",
    }
    assert expected_candidates.issubset(df["model_name"].unique()), "Missing candidates in predictions.csv"

    # Exactly 800 test rows per candidate (100 products * 8 test weeks)
    test_df = df[df["split"] == "test"]
    test_counts = test_df.groupby("model_name").size()
    for cand in expected_candidates:
        assert test_counts.get(cand, 0) == 800, f"Expected 800 test rows for '{cand}', got {test_counts.get(cand, 0)}"

    # 100 products represented
    assert test_df["product_id"].nunique() == 100, f"Expected 100 products, got {test_df['product_id'].nunique()}"

    # No missing or negative predictions
    assert not df["predicted"].isna().any(), "Found NaN in predicted column."
    assert (df["predicted"] >= 0.0).all(), "Found negative predicted values."
    assert not df["actual"].isna().any(), "Found NaN in actual column."
    assert (df["actual"] >= 0.0).all(), "Found negative actual values."


def test_metrics_csv_schema_and_integrity(result_files: dict[str, pd.DataFrame]) -> None:
    """Validate data/processed/metrics.csv schema, absence of MAPE, and finite values (C3)."""
    m_path = Path("data/processed/metrics.csv")
    assert m_path.exists(), "data/processed/metrics.csv must exist."

    df = pd.read_csv(m_path, dtype={"model_name": str, "split": str})
    expected_cols = ["model_name", "split", "MAE", "RMSE", "WAPE", "bias"]
    assert list(df.columns) == expected_cols, f"Columns mismatch. Expected {expected_cols}, got {list(df.columns)}"

    # Both splits present for all candidates
    assert set(df["split"].unique()) == {"val", "test"}
    assert "MAPE" not in df.columns, "MAPE is strictly forbidden."

    for col in ("MAE", "RMSE", "WAPE", "bias"):
        assert pd.api.types.is_numeric_dtype(df[col]), f"Column {col} must be numeric."
        assert not df[col].isna().any(), f"Column {col} contains NaN values."
        assert np.isfinite(df[col]).all(), f"Column {col} contains infinite values."

    # MAE, RMSE, WAPE must be strictly non-negative
    for col in ("MAE", "RMSE", "WAPE"):
        assert (df[col] >= 0.0).all(), f"Column {col} must be non-negative."


def test_next_week_forecast_csv_schema_and_constraints(result_files: dict[str, pd.DataFrame]) -> None:
    """Validate data/processed/next_week_forecast.csv against Contract C4 & Addendum C4."""
    f_path = Path("data/processed/next_week_forecast.csv")
    assert f_path.exists(), "data/processed/next_week_forecast.csv must exist."

    df = pd.read_csv(f_path, dtype={"product_id": str})
    expected_cols = [
        "product_id",
        "week_start",
        "forecast_weekly",
        "forecast_p10",
        "forecast_p90",
        "sigma_error",
        "avg_weekly_units_train",
        "last_avg_price",
    ]
    assert list(df.columns) == expected_cols, f"Columns mismatch. Expected {expected_cols}, got {list(df.columns)}"

    # Exactly 100 products for week 2011-12-05
    assert len(df) == 100, f"Expected exactly 100 rows, got {len(df)}"
    assert df["product_id"].nunique() == 100, "All product_ids must be unique."
    assert (df["week_start"] == "2011-12-05").all(), "All week_start must be '2011-12-05'."

    # Forecast constraints: 0 <= p10 <= forecast_weekly <= p90
    assert not df.isna().any().any(), "next_week_forecast.csv contains NaN values."
    assert (df["forecast_weekly"] >= 0.0).all(), "forecast_weekly has negative values."
    assert (df["forecast_p10"] >= 0.0).all(), "forecast_p10 has negative values."
    assert (df["forecast_p90"] >= 0.0).all(), "forecast_p90 has negative values."
    assert (df["forecast_p10"] <= df["forecast_weekly"]).all(), "Constraint violation: forecast_p10 > forecast_weekly"
    assert (df["forecast_weekly"] <= df["forecast_p90"]).all(), "Constraint violation: forecast_weekly > forecast_p90"

    # sigma_error >= 0.0
    assert (df["sigma_error"] >= 0.0).all(), "sigma_error must be >= 0."

    # avg_weekly_units_train > 0.0 and last_avg_price > 0.0
    assert (df["avg_weekly_units_train"] > 0.0).all(), "avg_weekly_units_train must be > 0."
    assert (df["last_avg_price"] > 0.0).all(), "last_avg_price must be > 0."


def test_sigma_error_fallback_logic() -> None:
    """compute_product_sigma_error must fall back to pooled segment sigma when obs < 6."""
    mock_oof = pd.DataFrame(
        {
            "product_id": ["P1", "P1", "P1", "P2", "P2", "P2", "P2", "P2", "P2", "P2"],
            "actual": [10.0, 10.0, 10.0, 20.0, 22.0, 19.0, 21.0, 20.0, 23.0, 18.0],
            "predicted": [12.0, 8.0, 15.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0],
            "candidate": ["ridge_raw"] * 10,
        }
    )
    mock_seg = pd.DataFrame(
        {
            "product_id": ["P1", "P2"],
            "segment": ["high", "high"],
        }
    )

    sigma_map = compute_product_sigma_error(mock_oof, mock_seg, best_candidate="ridge_raw", min_observations=6)

    # P1 has only 3 obs (< 6), so it must take the pooled segment sigma
    # P2 has 7 obs (>= 6), so it must take its own std
    assert "P1" in sigma_map
    assert "P2" in sigma_map
    assert sigma_map["P1"] > 0.0
    assert sigma_map["P2"] > 0.0
    # P2 error std: predicted - actual = [0, -2, 1, -1, 0, -3, 2], std(ddof=1) is approx 1.73
    p2_err = np.array([0.0, -2.0, 1.0, -1.0, 0.0, -3.0, 2.0])
    assert sigma_map["P2"] == pytest.approx(float(np.std(p2_err, ddof=1)))
