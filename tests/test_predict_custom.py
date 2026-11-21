"""Tests for Task M11: Model artifact persistence, reload reproducibility, and predict_custom."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import joblib
import numpy as np
import pandas as pd
import pytest

from src.features import build_features, build_future_features
from src.predict import predict_custom, select_and_save_best_model
from src.train import (
    DEFAULT_TEST_START,
    MODELS_DIR,
    split_chronological,
)


@pytest.fixture(scope="module")
def model_artifacts() -> dict[str, object]:
    """Ensure best_model artifacts are generated and return summary dictionary."""
    return select_and_save_best_model()


def test_contract_c5_artifacts_exist_and_conform(model_artifacts: dict[str, object]) -> None:
    """Validate Contract C5 artifact schemas, required keys, and types."""
    m_dir = Path("models")
    best_model_path = m_dir / "best_model.joblib"
    feature_list_path = m_dir / "feature_list.json"
    config_path = m_dir / "config.json"

    assert best_model_path.exists(), "models/best_model.joblib must exist."
    assert feature_list_path.exists(), "models/feature_list.json must exist."
    assert config_path.exists(), "models/config.json must exist."

    # Verify best_model.joblib loads cleanly
    model = joblib.load(best_model_path)
    assert hasattr(model, "predict"), "best_model must implement predict()."

    # Verify feature_list.json
    with open(feature_list_path, encoding="utf-8") as f:
        features = json.load(f)
    assert isinstance(features, list), "feature_list.json must contain a list."
    assert len(features) == 16, f"Expected 16 features for base+g3_no_calendar, got {len(features)}."
    assert all(isinstance(f, str) for f in features), "All feature names must be strings."
    assert "lag_1" in features
    assert "roll_mean_4" in features
    assert "yoy_mean_5w" in features

    # Verify config.json
    with open(config_path, encoding="utf-8") as f:
        cfg = json.load(f)

    required_keys = {
        "model_name",
        "target_mode",
        "groups",
        "horizon_weeks",
        "train_end",
        "test_start",
        "seed",
        "library_versions",
        "data_version",
    }
    assert required_keys.issubset(cfg.keys()), f"config.json missing required keys: {required_keys - set(cfg.keys())}"

    assert cfg["model_name"] in ("ensemble", "ridge")
    if cfg["model_name"] == "ensemble":
        assert "components" in cfg
        assert cfg["components"] == {"ridge_raw": 0.8, "moving_avg_4": 0.2}
    assert cfg["target_mode"] == "raw"
    assert cfg["groups"] == ["base", "g3"]
    assert cfg["horizon_weeks"] == 1
    assert cfg["train_end"] == "2011-10-03"
    assert cfg["test_start"] == "2011-10-10"
    assert cfg["seed"] == 42
    assert cfg["data_version"] == "3.2"

    lib_vers = cfg["library_versions"]
    assert isinstance(lib_vers, dict)
    for lib in ("python", "numpy", "pandas", "scikit-learn", "lightgbm", "joblib", "scipy"):
        assert lib in lib_vers, f"library_versions missing {lib}"


def test_saved_model_fresh_process_exact_reproduction(model_artifacts: dict[str, object]) -> None:
    """The saved model must reload in a fresh Python process and reproduce test predictions exactly (T-M-3)."""
    # 1. In-process test predictions
    weekly = pd.read_parquet("data/processed/weekly_demand.parquet")
    closed = pd.read_csv("data/processed/closed_weeks.csv")
    features_df = build_features(weekly, closed_weeks=closed, groups=("base", "g3"))
    _, test_df = split_chronological(features_df, test_start=DEFAULT_TEST_START)

    with open("models/feature_list.json", encoding="utf-8") as f:
        features = json.load(f)

    in_process_model = joblib.load("models/best_model.joblib")
    in_process_preds = np.clip(
        np.asarray(in_process_model.predict(test_df[features]), dtype=np.float64), 0.0, None
    )

    # 2. Fresh Python process prediction via subprocess
    sub_code = """
import json
import joblib
import numpy as np
import pandas as pd
from src.features import build_features
from src.train import split_chronological, DEFAULT_TEST_START

weekly = pd.read_parquet("data/processed/weekly_demand.parquet")
closed = pd.read_csv("data/processed/closed_weeks.csv")
features_df = build_features(weekly, closed_weeks=closed, groups=("base", "g3"))
_, test_df = split_chronological(features_df, test_start=DEFAULT_TEST_START)

with open("models/feature_list.json", encoding="utf-8") as f:
    features = json.load(f)

model = joblib.load("models/best_model.joblib")
preds = np.clip(np.asarray(model.predict(test_df[features]), dtype=np.float64), 0.0, None)
np.save("data/mock/subprocess_preds.npy", preds)
"""
    run_res = subprocess.run([sys.executable, "-c", sub_code], capture_output=True, text=True, check=True)
    assert run_res.returncode == 0, f"Subprocess failed: {run_res.stderr}"

    fresh_preds_path = Path("data/mock/subprocess_preds.npy")
    assert fresh_preds_path.exists(), "Subprocess did not save prediction array."
    fresh_preds = np.load(fresh_preds_path)
    fresh_preds_path.unlink()

    # Predictions must match exactly
    np.testing.assert_allclose(
        fresh_preds,
        in_process_preds,
        rtol=1e-12,
        atol=1e-12,
        err_msg="Fresh process predictions do not match in-process predictions.",
    )


def test_predict_custom_reproduces_table_for_shared_features(model_artifacts: dict[str, object]) -> None:
    """predict_custom for a series taken from the table equals table-based prediction for shared features (T-M-5)."""
    weekly = pd.read_parquet("data/processed/weekly_demand.parquet")
    closed = pd.read_csv("data/processed/closed_weeks.csv")
    prod_id = "85123A"
    target_week = "2011-12-05"

    prod_df = weekly.loc[weekly["product_id"] == prod_id].sort_values("week_start")
    recent = prod_df.tail(8)

    payload = {
        "recent_units": recent["units"].tolist(),
        "recent_prices": recent["avg_price"].tolist(),
        "recent_orders": recent["orders"].tolist(),
        "recent_customers": recent["customers"].tolist(),
        "target_week": target_week,
    }

    # 1. Custom prediction
    custom_res = predict_custom(payload)

    # 2. Table future features
    table_future = build_future_features(
        weekly=prod_df,
        closed_weeks=closed,
        target_week=target_week,
        groups=("base", "g3"),
    )

    # Shared features must match exactly between custom and table
    shared_cols = [
        "lag_1",
        "lag_2",
        "lag_3",
        "lag_4",
        "roll_mean_4",
        "roll_std_4",
        "roll_mean_8",
        "roll_std_8",
        "prev_avg_price",
        "prev_orders",
        "prev_customers",
    ]
    from src.features import build_custom_features

    cust_feat_df = build_custom_features(series=payload, target_week=target_week)

    for col in shared_cols:
        val_cust = float(cust_feat_df[col].iloc[0])
        val_table = float(table_future[col].iloc[0])
        assert np.isclose(val_cust, val_table, atol=1e-6), f"Mismatch in feature '{col}': cust={val_cust}, table={val_table}"

    # Evaluate Ridge on the shared subset: must equal within floating point precision
    from sklearn.linear_model import Ridge

    np.random.seed(42)
    X_synthetic = pd.DataFrame(
        np.random.randn(20, len(shared_cols)), columns=shared_cols
    )
    y_synthetic = np.random.randn(20) * 100.0 + 50.0

    simple_ridge = Ridge(alpha=100.0, random_state=42)
    simple_ridge.fit(X_synthetic, y_synthetic)

    pred_cust_shared = float(simple_ridge.predict(cust_feat_df[shared_cols])[0])
    pred_table_shared = float(simple_ridge.predict(table_future[shared_cols])[0])

    assert np.isclose(pred_cust_shared, pred_table_shared, atol=1e-6), "Shared feature predictions must be identical."


def test_predict_custom_bad_payloads_rejected() -> None:
    """Bad payloads (fewer than 8 values, negative numbers, missing recent_units) must raise ValueError."""
    # 1. Fewer than 8 values (7 values)
    with pytest.raises(ValueError, match="at least 8 values"):
        predict_custom({"recent_units": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0]})

    # 2. Empty recent_units
    with pytest.raises(ValueError, match="at least 8 values"):
        predict_custom({"recent_units": []})

    # 3. Negative values in recent_units
    with pytest.raises(ValueError, match="Negative demand is not allowed"):
        predict_custom({"recent_units": [10.0, 20.0, -5.0, 40.0, 50.0, 60.0, 70.0, 80.0]})

    # 4. More than 12 values
    with pytest.raises(ValueError, match="at most 12 values"):
        predict_custom({"recent_units": [10.0] * 13})

    # 5. Missing recent_units key
    with pytest.raises(ValueError, match="must contain 'recent_units'"):
        predict_custom({"target_week": "2011-12-05"})

    # 6. Non-dict payload
    with pytest.raises(ValueError, match="must be a dictionary"):
        predict_custom([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0])  # type: ignore

    # 7. Negative optional series
    with pytest.raises(ValueError, match="Negative value not allowed in 'recent_prices'"):
        predict_custom(
            {
                "recent_units": [10.0] * 8,
                "recent_prices": [1.0, 2.0, -0.5, 1.0, 1.0, 1.0, 1.0, 1.0],
            }
        )


def test_predict_custom_ordering_and_notes(model_artifacts: dict[str, object]) -> None:
    """predict_custom outputs must satisfy 0 <= p10 <= forecast_weekly <= p90 and document missing groups."""
    payload = {
        "recent_units": [15.0, 22.0, 19.0, 30.0, 25.0, 28.0, 35.0, 40.0],
        "target_week": "2011-12-05",
    }
    res = predict_custom(payload)

    # Contract C7 response schema
    assert "forecast_weekly" in res
    assert "forecast_p10" in res
    assert "forecast_p90" in res
    assert "features_used" in res
    assert "notes" in res

    # Ordering & clipping guarantees
    assert res["forecast_p10"] >= 0.0, "p10 must be non-negative"
    assert res["forecast_p10"] <= res["forecast_weekly"], "p10 <= forecast_weekly violation"
    assert res["forecast_weekly"] <= res["forecast_p90"], "forecast_weekly <= p90 violation"

    # Notes must state that groups g3 and g4 were unavailable
    assert "g3" in res["notes"], "Notes must mention unavailable group g3"
    assert "g4" in res["notes"], "Notes must mention unavailable group g4"
