"""Contract schema tests for C1, C1b, C2, C3, and C4.

Validates files in data/mock/ and, whenever present, data/processed/.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

MOCK_DIR = Path("data/mock")
PROCESSED_DIR = Path("data/processed")

C1_COLUMNS = [
    "product_id",
    "description",
    "week_start",
    "units",
    "orders",
    "customers",
    "avg_price",
]
C1B_COLUMNS = ["week_start"]
C2_COLUMNS = ["product_id", "week_start", "actual", "predicted", "model_name", "split"]
C3_COLUMNS = ["model_name", "split", "MAE", "RMSE", "WAPE", "bias"]
C4_COLUMNS = [
    "product_id",
    "week_start",
    "forecast_weekly",
    "sigma_error",
    "avg_weekly_units_train",
]


def _target_dirs(filename: str) -> list[tuple[str, Path]]:
    """Return (label, directory) pairs where the target contract file exists."""
    dirs: list[tuple[str, Path]] = [("mock", MOCK_DIR)]
    if (PROCESSED_DIR / filename).exists():
        dirs.append(("processed", PROCESSED_DIR))
    return dirs


@pytest.mark.parametrize(("env_name", "data_dir"), _target_dirs("weekly_demand.parquet"))
def test_contract_c1_weekly_demand(env_name: str, data_dir: Path) -> None:
    """Validate C1 weekly_demand.parquet schema, Monday grid, and gapless ranges."""
    path = data_dir / "weekly_demand.parquet"
    assert path.exists(), f"Missing {path}"

    df = pd.read_parquet(path)
    assert not df.empty, f"{path} must not be empty"
    assert list(df.columns) == C1_COLUMNS
    assert not df.isna().any().any(), f"{path} contains null values"

    assert pd.api.types.is_string_dtype(df["product_id"])
    assert pd.api.types.is_string_dtype(df["description"])
    assert pd.api.types.is_datetime64_any_dtype(df["week_start"])
    assert pd.api.types.is_integer_dtype(df["units"])
    assert pd.api.types.is_integer_dtype(df["orders"])
    assert pd.api.types.is_integer_dtype(df["customers"])
    assert pd.api.types.is_float_dtype(df["avg_price"])

    assert (df["units"] >= 0).all()
    assert (df["orders"] >= 0).all()
    assert (df["customers"] >= 0).all()
    assert np.isfinite(df["avg_price"]).all() and (df["avg_price"] > 0).all()

    weeks = pd.to_datetime(df["week_start"])
    assert (weeks.dt.weekday == 0).all(), "All week_start values must be Mondays"
    assert not df.duplicated(subset=["product_id", "week_start"]).any()

    sorted_df = df.sort_values(["product_id", "week_start"]).reset_index(drop=True)
    assert df.reset_index(drop=True).equals(sorted_df), "Rows must be sorted by product_id, week_start"

    for product_id, group in df.groupby("product_id", sort=False):
        prod_weeks = pd.to_datetime(group["week_start"]).sort_values()
        expected = pd.date_range(prod_weeks.min(), prod_weeks.max(), freq="W-MON")
        assert list(prod_weeks) == list(expected), f"Gap found in {product_id} week range"

    n_products = df["product_id"].nunique()
    if env_name == "mock":
        assert n_products == 10
        assert df.groupby("product_id")["week_start"].nunique().eq(20).all()
    else:
        assert n_products == 100


@pytest.mark.parametrize(("env_name", "data_dir"), _target_dirs("closed_weeks.csv"))
def test_contract_c1b_closed_weeks(env_name: str, data_dir: Path) -> None:
    """Validate C1b closed_weeks.csv schema and Monday dates."""
    path = data_dir / "closed_weeks.csv"
    assert path.exists(), f"Missing {path}"

    df = pd.read_csv(path, dtype={"week_start": str})
    assert not df.empty
    assert list(df.columns) == C1B_COLUMNS
    assert not df.isna().any().any()

    weeks = pd.to_datetime(df["week_start"])
    assert (weeks.dt.weekday == 0).all()
    assert not df["week_start"].duplicated().any()
    assert "2010-12-27" in set(weeks.dt.strftime("%Y-%m-%d"))


@pytest.mark.parametrize(("env_name", "data_dir"), _target_dirs("predictions.csv"))
def test_contract_c2_predictions(env_name: str, data_dir: Path) -> None:
    """Validate C2 predictions.csv schema, types, and non-negativity."""
    path = data_dir / "predictions.csv"
    assert path.exists(), f"Missing {path}"

    df = pd.read_csv(path, dtype={"product_id": str, "model_name": str, "split": str})
    assert not df.empty
    assert list(df.columns) == C2_COLUMNS
    assert not df.isna().any().any()

    assert pd.api.types.is_string_dtype(df["product_id"])
    assert pd.api.types.is_string_dtype(df["model_name"])
    assert pd.api.types.is_string_dtype(df["split"])
    assert set(df["split"].unique()).issubset({"val", "test"})

    weeks = pd.to_datetime(df["week_start"])
    assert (weeks.dt.weekday == 0).all()

    assert pd.api.types.is_numeric_dtype(df["actual"])
    assert pd.api.types.is_numeric_dtype(df["predicted"])
    assert np.isfinite(df["actual"]).all() and (df["actual"] >= 0).all()
    assert np.isfinite(df["predicted"]).all() and (df["predicted"] >= 0).all()


@pytest.mark.parametrize(("env_name", "data_dir"), _target_dirs("metrics.csv"))
def test_contract_c3_metrics(env_name: str, data_dir: Path) -> None:
    """Validate C3 metrics.csv schema and absence of MAPE."""
    path = data_dir / "metrics.csv"
    assert path.exists(), f"Missing {path}"

    df = pd.read_csv(path, dtype={"model_name": str, "split": str})
    assert not df.empty
    assert list(df.columns) == C3_COLUMNS
    assert "MAPE" not in df.columns
    assert not df.isna().any().any()

    assert pd.api.types.is_string_dtype(df["model_name"])
    assert pd.api.types.is_string_dtype(df["split"])
    assert set(df["split"].unique()).issubset({"val", "test"})

    for col in ("MAE", "RMSE", "WAPE"):
        assert pd.api.types.is_numeric_dtype(df[col])
        assert np.isfinite(df[col]).all() and (df[col] >= 0).all()

    assert pd.api.types.is_numeric_dtype(df["bias"])
    assert np.isfinite(df["bias"]).all()


@pytest.mark.parametrize(("env_name", "data_dir"), _target_dirs("next_week_forecast.csv"))
def test_contract_c4_next_week_forecast(env_name: str, data_dir: Path) -> None:
    """Validate C4 next_week_forecast.csv schema and target week."""
    path = data_dir / "next_week_forecast.csv"
    assert path.exists(), f"Missing {path}"

    df = pd.read_csv(path, dtype={"product_id": str})
    assert not df.empty
    assert list(df.columns) == C4_COLUMNS
    assert not df.isna().any().any()
    assert not df["product_id"].duplicated().any()

    assert pd.api.types.is_string_dtype(df["product_id"])
    weeks = pd.to_datetime(df["week_start"])
    assert (weeks.dt.weekday == 0).all()
    assert (weeks.dt.strftime("%Y-%m-%d") == "2011-12-05").all()

    for col in ("forecast_weekly", "sigma_error", "avg_weekly_units_train"):
        assert pd.api.types.is_numeric_dtype(df[col])
        assert np.isfinite(df[col]).all() and (df[col] >= 0).all()

