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
C4_BASE_COLUMNS = [
    "product_id",
    "week_start",
    "forecast_weekly",
    "sigma_error",
    "avg_weekly_units_train",
]
C4_AMENDED_COLUMNS = [
    "product_id",
    "week_start",
    "forecast_weekly",
    "forecast_p10",
    "forecast_p90",
    "sigma_error",
    "avg_weekly_units_train",
    "last_avg_price",
]
C8_COLUMNS = [
    "product_id",
    "week_start",
    "gross_units",
    "cancelled_units",
    "export_units",
    "guest_units",
]
C9_COLUMNS = [
    "week_start",
    "uk_bank_holiday_count",
    "trading_days_expected",
    "is_christmas_week",
]
C10_COLUMNS = [
    "product_id",
    "week_start",
    "units",
    "orders",
]
C11_COLUMNS = [
    "product_id",
    "description",
    "adi",
    "cv2",
    "pattern_class",
    "abc_class",
    "xyz_class",
    "lifecycle_stage",
    "weeks_since_last_sale",
    "return_rate",
    "export_share",
    "top3_customer_share",
    "known_customer_coverage",
    "revenue_share",
]
C12_COLUMNS = [
    "product_id",
    "companion_id",
    "companion_description",
    "n_invoices",
    "confidence",
    "lift",
]
C14_COLUMNS = [
    "policy",
    "lead_time_weeks",
    "abc_class",
    "fill_rate",
    "stockout_weeks",
    "avg_on_hand",
    "unmet_units",
]
C15_COLUMNS = [
    "policy",
    "lead_time_weeks",
    "product_id",
    "week_start",
    "on_hand_start",
    "order_qty",
    "arrived_qty",
    "demand",
    "sold",
    "unmet",
    "on_hand_end",
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

    closed_path = data_dir / "closed_weeks.csv"
    closed_set: set[pd.Timestamp] = set()
    if closed_path.exists():
        closed_df = pd.read_csv(closed_path, dtype={"week_start": str})
        closed_set = set(pd.to_datetime(closed_df["week_start"]).dt.normalize())

    end_week = weeks.max().normalize()
    for product_id, group in df.groupby("product_id", sort=False):
        prod_weeks = pd.to_datetime(group["week_start"]).dt.normalize().sort_values()
        if closed_set:
            assert not prod_weeks.isin(closed_set).any(), f"Closed week filled for {product_id}"
        expected = [
            w for w in pd.date_range(prod_weeks.min(), end_week, freq="W-MON")
            if w not in closed_set
        ]
        assert list(prod_weeks) == expected, f"Gap or early end in {product_id} week range"

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
    assert list(df.columns) in (C4_BASE_COLUMNS, C4_AMENDED_COLUMNS)
    assert not df.isna().any().any()
    assert not df["product_id"].duplicated().any()

    assert pd.api.types.is_string_dtype(df["product_id"])
    weeks = pd.to_datetime(df["week_start"])
    assert (weeks.dt.weekday == 0).all()
    assert (weeks.dt.strftime("%Y-%m-%d") == "2011-12-05").all()

    for col in ("forecast_weekly", "sigma_error", "avg_weekly_units_train"):
        assert pd.api.types.is_numeric_dtype(df[col])
        assert np.isfinite(df[col]).all() and (df[col] >= 0).all()

    if list(df.columns) == C4_AMENDED_COLUMNS:
        for col in ("forecast_p10", "forecast_p90", "last_avg_price"):
            assert pd.api.types.is_numeric_dtype(df[col])
            assert np.isfinite(df[col]).all() and (df[col] >= 0).all()
        assert (df["forecast_p10"] <= df["forecast_weekly"]).all()
        assert (df["forecast_weekly"] <= df["forecast_p90"]).all()
        assert (df["last_avg_price"] > 0).all()


def test_contract_c5_model_files() -> None:
    """Validate Contract C5 model files in models/ directory."""
    import json
    import joblib

    models_dir = Path("models")
    best_model_path = models_dir / "best_model.joblib"
    feature_list_path = models_dir / "feature_list.json"
    config_path = models_dir / "config.json"

    assert best_model_path.exists(), "models/best_model.joblib must exist."
    assert feature_list_path.exists(), "models/feature_list.json must exist."
    assert config_path.exists(), "models/config.json must exist."

    # Validate model loadability
    model = joblib.load(best_model_path)
    assert hasattr(model, "predict"), "best_model must implement predict()."

    # Validate feature list
    with open(feature_list_path, encoding="utf-8") as f:
        features = json.load(f)
    assert isinstance(features, list), "feature_list.json must be a list."
    assert len(features) > 0, "feature_list.json cannot be empty."
    assert all(isinstance(f, str) for f in features), "All features must be strings."

    # Validate config.json
    with open(config_path, encoding="utf-8") as f:
        cfg = json.load(f)
    required_keys = {
        "model_name",
        "horizon_weeks",
        "train_end",
        "seed",
        "library_versions",
    }
    assert required_keys.issubset(cfg.keys()), f"Missing required keys in config.json: {required_keys - set(cfg.keys())}"
    assert cfg["horizon_weeks"] == 1
    assert cfg["seed"] == 42
    assert isinstance(cfg["library_versions"], dict)


@pytest.mark.parametrize(("env_name", "data_dir"), _target_dirs("enrichment_weekly.parquet"))
def test_contract_c8_enrichment_weekly(env_name: str, data_dir: Path) -> None:
    """Validate C8 enrichment_weekly.parquet schema and integrity."""
    path = data_dir / "enrichment_weekly.parquet"
    assert path.exists(), f"Missing {path}"

    df = pd.read_parquet(path)
    assert not df.empty
    assert list(df.columns) == C8_COLUMNS
    assert not df.isna().any().any()

    assert pd.api.types.is_string_dtype(df["product_id"])
    assert pd.api.types.is_datetime64_any_dtype(df["week_start"])
    weeks = pd.to_datetime(df["week_start"])
    assert (weeks.dt.weekday == 0).all(), "All week_start values must be Mondays"

    for col in ("gross_units", "cancelled_units", "export_units", "guest_units"):
        assert pd.api.types.is_integer_dtype(df[col])
        assert (df[col] >= 0).all()

    assert (df["export_units"] <= df["gross_units"]).all()
    assert (df["guest_units"] <= df["gross_units"]).all()

    # Compare with weekly_demand if available in same directory
    demand_path = data_dir / "weekly_demand.parquet"
    if demand_path.exists():
        demand_df = pd.read_parquet(demand_path)
        assert len(df) == len(demand_df), "C8 must have the exact same row count as C1"
        assert (df["product_id"] == demand_df["product_id"]).all()
        assert (df["week_start"] == demand_df["week_start"]).all()
        assert (df["gross_units"] == demand_df["units"]).all(), "gross_units must equal C1 units"


@pytest.mark.parametrize(("env_name", "data_dir"), _target_dirs("calendar_weekly.csv"))
def test_contract_c9_calendar_weekly(env_name: str, data_dir: Path) -> None:
    """Validate C9 calendar_weekly.csv schema, bank holidays, and trading days."""
    path = data_dir / "calendar_weekly.csv"
    assert path.exists(), f"Missing {path}"

    df = pd.read_csv(path)
    assert not df.empty
    assert list(df.columns) == C9_COLUMNS
    assert not df.isna().any().any()

    weeks = pd.to_datetime(df["week_start"])
    assert (weeks.dt.weekday == 0).all(), "All week_start values must be Mondays"
    assert not df["week_start"].duplicated().any()

    assert pd.api.types.is_integer_dtype(df["uk_bank_holiday_count"])
    assert pd.api.types.is_integer_dtype(df["trading_days_expected"])
    assert pd.api.types.is_integer_dtype(df["is_christmas_week"])

    assert (df["uk_bank_holiday_count"] >= 0).all()
    assert (df["trading_days_expected"] >= 0).all() and (df["trading_days_expected"] <= 7).all()
    assert set(df["is_christmas_week"].unique()).issubset({0, 1})


@pytest.mark.parametrize(("env_name", "data_dir"), _target_dirs("prior_year_weekly.parquet"))
def test_contract_c10_prior_year_weekly(env_name: str, data_dir: Path) -> None:
    """Validate C10 prior_year_weekly.parquet schema and strictly pre-2010-12-06 dates."""
    path = data_dir / "prior_year_weekly.parquet"
    assert path.exists(), f"Missing {path}"

    df = pd.read_parquet(path)
    assert not df.empty
    assert list(df.columns) == C10_COLUMNS
    assert not df.isna().any().any()

    assert pd.api.types.is_string_dtype(df["product_id"])
    assert pd.api.types.is_datetime64_any_dtype(df["week_start"])
    assert pd.api.types.is_integer_dtype(df["units"])
    assert pd.api.types.is_integer_dtype(df["orders"])

    assert (df["units"] >= 0).all()
    assert (df["orders"] >= 0).all()

    weeks = pd.to_datetime(df["week_start"])
    assert (weeks.dt.weekday == 0).all()
    assert (weeks < pd.Timestamp("2010-12-06")).all(), "Prior-year dates must be before 2010-12-06"
    assert not (weeks == pd.Timestamp("2009-12-28")).any(), "Closure week 2009-12-28 must have no rows"


@pytest.mark.parametrize(("env_name", "data_dir"), _target_dirs("product_intelligence.csv"))
def test_contract_c11_product_intelligence(env_name: str, data_dir: Path) -> None:
    """Validate C11 product_intelligence.csv schema, categorical sets, and ratio bounds."""
    path = data_dir / "product_intelligence.csv"
    assert path.exists(), f"Missing {path}"

    df = pd.read_csv(path, dtype={"product_id": str})
    assert not df.empty
    assert list(df.columns) == C11_COLUMNS
    assert not df.isna().any().any()
    assert not df["product_id"].duplicated().any()

    assert pd.api.types.is_string_dtype(df["product_id"])
    assert pd.api.types.is_string_dtype(df["description"])

    assert pd.api.types.is_numeric_dtype(df["adi"]) and (df["adi"] > 0).all()
    assert pd.api.types.is_numeric_dtype(df["cv2"]) and (df["cv2"] >= 0).all()

    assert set(df["pattern_class"].unique()).issubset({"smooth", "erratic", "intermittent", "lumpy"})
    assert set(df["abc_class"].unique()).issubset({"A", "B", "C"})
    assert set(df["xyz_class"].unique()).issubset({"X", "Y", "Z"})
    assert set(df["lifecycle_stage"].unique()).issubset({"new", "growing", "mature", "declining", "inactive"})

    assert pd.api.types.is_integer_dtype(df["weeks_since_last_sale"])
    assert (df["weeks_since_last_sale"] >= 0).all()
    assert pd.api.types.is_numeric_dtype(df["return_rate"]) and (df["return_rate"] >= 0).all()

    for col in ("export_share", "top3_customer_share", "known_customer_coverage", "revenue_share"):
        assert pd.api.types.is_numeric_dtype(df[col])
        assert (df[col] >= 0.0).all() and (df[col] <= 1.0).all()


@pytest.mark.parametrize(("env_name", "data_dir"), _target_dirs("basket_pairs.csv"))
def test_contract_c12_basket_pairs(env_name: str, data_dir: Path) -> None:
    """Validate C12 basket_pairs.csv schema and companion metrics."""
    path = data_dir / "basket_pairs.csv"
    assert path.exists(), f"Missing {path}"

    df = pd.read_csv(path, dtype={"product_id": str, "companion_id": str})
    assert not df.empty
    assert list(df.columns) == C12_COLUMNS
    assert not df.isna().any().any()

    assert pd.api.types.is_string_dtype(df["product_id"])
    assert pd.api.types.is_string_dtype(df["companion_id"])
    assert pd.api.types.is_string_dtype(df["companion_description"])
    assert (df["product_id"] != df["companion_id"]).all()

    assert pd.api.types.is_integer_dtype(df["n_invoices"])
    assert (df["n_invoices"] > 0).all()
    assert pd.api.types.is_float_dtype(df["confidence"])
    assert (df["confidence"] >= 0.0).all() and (df["confidence"] <= 1.0).all()
    assert pd.api.types.is_float_dtype(df["lift"])
    assert (df["lift"] > 0.0).all()

    # Up to 3 companions per product
    assert (df.groupby("product_id")["companion_id"].count() <= 3).all()


@pytest.mark.parametrize(("env_name", "data_dir"), _target_dirs("policy_backtest.csv"))
def test_contract_c14_policy_backtest(env_name: str, data_dir: Path) -> None:
    """Validate C14 policy_backtest.csv schema, types, and metric bounds."""
    path = data_dir / "policy_backtest.csv"
    assert path.exists(), f"Missing {path}"

    df = pd.read_csv(path)
    assert not df.empty
    assert list(df.columns) == C14_COLUMNS
    assert not df.isna().any().any()

    assert set(df["policy"].unique()).issubset(
        {"naive", "moving_avg_4", "best_model", "best_model_safety"}
    )
    assert set(df["lead_time_weeks"].unique()).issubset({1, 2, 3, 4})
    assert set(df["abc_class"].unique()).issubset({"A", "B", "C", "ALL"})

    assert pd.api.types.is_float_dtype(df["fill_rate"])
    assert (df["fill_rate"] >= 0.0).all() and (df["fill_rate"] <= 1.0).all()
    assert pd.api.types.is_integer_dtype(df["stockout_weeks"])
    assert (df["stockout_weeks"] >= 0).all()
    assert pd.api.types.is_float_dtype(df["avg_on_hand"])
    assert (df["avg_on_hand"] >= 0.0).all()
    assert pd.api.types.is_float_dtype(df["unmet_units"])
    assert (df["unmet_units"] >= 0.0).all()


@pytest.mark.parametrize(("env_name", "data_dir"), _target_dirs("policy_backtest_weekly.csv"))
def test_contract_c15_policy_backtest_weekly(env_name: str, data_dir: Path) -> None:
    """Validate C15 policy_backtest_weekly.csv schema and flow conservation."""
    path = data_dir / "policy_backtest_weekly.csv"
    assert path.exists(), f"Missing {path}"

    df = pd.read_csv(path, dtype={"product_id": str})
    assert not df.empty
    assert list(df.columns) == C15_COLUMNS
    assert not df.isna().any().any()

    # Flow conservation: sold + on_hand_end == on_hand_start + arrived_qty
    available = df["on_hand_start"] + df["arrived_qty"]
    fulfilled = df["sold"] + df["on_hand_end"]
    np.testing.assert_allclose(fulfilled.to_numpy(), available.to_numpy(), atol=1e-5)


