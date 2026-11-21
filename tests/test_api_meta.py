"""tests/test_api_meta.py - Tests for Task B9 app skeleton, schemas, clean helper, and meta endpoints."""

from __future__ import annotations

import math
import os
from datetime import date, datetime
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.main import (
    REQUIRED_DATA_FILES,
    app,
    clean,
    data_store,
    load_application_data,
)


@pytest.fixture(scope="module")
def client():
    """TestClient fixture with app lifespan execution."""
    with TestClient(app) as test_client:
        yield test_client


# -----------------------------------------------------------------------------
# Unit Tests for clean(obj) Helper
# -----------------------------------------------------------------------------

def test_clean_helper_nan_and_inf():
    """Verify clean converts NaN, inf, -inf to None."""
    payload = {
        "nan_val": float("nan"),
        "inf_val": float("inf"),
        "neg_inf_val": float("-inf"),
        "np_nan": np.nan,
        "valid_val": 42.5,
    }
    cleaned = clean(payload)
    assert cleaned["nan_val"] is None
    assert cleaned["inf_val"] is None
    assert cleaned["neg_inf_val"] is None
    assert cleaned["np_nan"] is None
    assert cleaned["valid_val"] == 42.5


def test_clean_helper_numpy_types():
    """Verify clean converts numpy scalars and arrays to native Python types."""
    payload = {
        "np_int": np.int64(100),
        "np_float": np.float64(3.14159),
        "np_bool": np.bool_(True),
        "np_arr": np.array([1, 2, np.nan, 4]),
    }
    cleaned = clean(payload)
    assert isinstance(cleaned["np_int"], int)
    assert cleaned["np_int"] == 100
    assert isinstance(cleaned["np_float"], float)
    assert cleaned["np_float"] == 3.14159
    assert isinstance(cleaned["np_bool"], bool)
    assert cleaned["np_bool"] is True
    assert cleaned["np_arr"] == [1, 2, None, 4]


def test_clean_helper_dates_and_timestamps():
    """Verify clean converts dates and timestamps to ISO 8601 strings."""
    d = date(2011, 12, 5)
    dt = datetime(2011, 12, 5, 10, 30, 0)
    ts = pd.Timestamp("2011-12-05")
    payload = {"date": d, "datetime": dt, "timestamp": ts}
    cleaned = clean(payload)
    assert cleaned["date"] == "2011-12-05"
    assert cleaned["datetime"] == "2011-12-05T10:30:00"
    assert "2011-12-05" in cleaned["timestamp"]


def test_clean_helper_strips_customer_id():
    """Verify clean never exposes CustomerID in any capitalization or naming form."""
    payload = {
        "product_id": "85123A",
        "units": 50,
        "CustomerID": 12345,
        "customer_id": 67890,
        "CustomerID_alt": "allowed_key",
        "nested": {
            "customerid": 99999,
            "Customer_Id": 88888,
            "valid_field": "ok",
        },
        "items": [
            {"CustomerID": 111, "name": "Item 1"},
            {"customer_id": 222, "name": "Item 2"},
        ],
    }
    cleaned = clean(payload)
    assert "CustomerID" not in cleaned
    assert "customer_id" not in cleaned
    assert "customerid" not in cleaned["nested"]
    assert "Customer_Id" not in cleaned["nested"]
    assert cleaned["nested"]["valid_field"] == "ok"
    assert cleaned["items"][0] == {"name": "Item 1"}
    assert cleaned["items"][1] == {"name": "Item 2"}


# -----------------------------------------------------------------------------
# Endpoint Tests: GET /api/health and /health
# -----------------------------------------------------------------------------

def test_health_endpoint(client: TestClient):
    """Verify GET /api/health returns status ok and database table counts."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "database" in data
    assert data["database"]["connected"] is True
    assert "tables" in data["database"]
    assert "inventory_levels" in data["database"]["tables"]
    assert data["database"]["tables"]["inventory_levels"] >= 5


def test_health_root_alias(client: TestClient):
    """Verify GET /health root alias returns identical response to /api/health."""
    api_resp = client.get("/api/health")
    root_resp = client.get("/health")
    assert root_resp.status_code == 200
    assert root_resp.json() == api_resp.json()


# -----------------------------------------------------------------------------
# Endpoint Tests: GET /api/disclaimer
# -----------------------------------------------------------------------------

def test_disclaimer_endpoint(client: TestClient):
    """Verify GET /api/disclaimer returns the exact mandatory disclaimer text."""
    response = client.get("/api/disclaimer")
    assert response.status_code == 200
    data = response.json()
    expected_text = (
        "Inventory levels are simulated. The dataset contains sales only, "
        "not stock-on-hand. Risk flags show how the rule behaves under the "
        "stated assumptions, not real stock positions."
    )
    assert data["text"] == expected_text


# -----------------------------------------------------------------------------
# Endpoint Tests: GET /api/model-info
# -----------------------------------------------------------------------------

def test_model_info_endpoint(client: TestClient):
    """Verify GET /api/model-info returns all required contract fields."""
    response = client.get("/api/model-info")
    assert response.status_code == 200
    data = response.json()

    # Core metadata fields
    assert "model_name" in data
    assert data["model_name"] == "ensemble"
    assert "target_mode" in data
    assert "feature_groups" in data
    assert isinstance(data["feature_groups"], list)
    assert "trained_on" in data
    assert "test_start" in data

    # Headline metrics
    assert "headline_metrics" in data
    metrics = data["headline_metrics"]
    for m in ["MAE", "RMSE", "WAPE", "bias"]:
        assert m in metrics
        assert isinstance(metrics[m], (int, float))

    # Ablation entries
    assert "ablation" in data
    assert isinstance(data["ablation"], list)
    if data["ablation"]:
        entry = data["ablation"][0]
        assert "groups" in entry
        assert "val_wape" in entry
        assert "val_bias" in entry

    # Interval coverage
    assert "interval_coverage" in data
    assert "val" in data["interval_coverage"]
    assert "test" in data["interval_coverage"]
    assert isinstance(data["interval_coverage"]["val"], (int, float))
    assert isinstance(data["interval_coverage"]["test"], (int, float))

    # Policy backtest
    assert "policy_backtest" in data
    assert isinstance(data["policy_backtest"], list)
    if data["policy_backtest"]:
        pb = data["policy_backtest"][0]
        assert "policy" in pb
        assert "lead_time_weeks" in pb
        assert "fill_rate" in pb
        assert "stockout_weeks" in pb
        assert "avg_on_hand" in pb
        assert "unmet_units" in pb

    # Library versions
    assert "library_versions" in data
    assert isinstance(data["library_versions"], dict)

    # Kaggle links structure
    assert "kaggle" in data
    assert "dataset_url" in data["kaggle"]
    assert "notebook_url" in data["kaggle"]
    assert "model_url" in data["kaggle"]


def test_model_info_kaggle_env(client: TestClient):
    """Verify Kaggle URLs read from environment variables or default to empty string."""
    env_vars = {
        "KAGGLE_DATASET_URL": "https://kaggle.com/datasets/shelfwatch",
        "KAGGLE_NOTEBOOK_URL": "https://kaggle.com/code/shelfwatch-nb",
        "KAGGLE_MODEL_URL": "https://kaggle.com/models/shelfwatch-ensemble",
    }
    with patch.dict(os.environ, env_vars, clear=False):
        response = client.get("/api/model-info")
        assert response.status_code == 200
        kaggle = response.json()["kaggle"]
        assert kaggle["dataset_url"] == env_vars["KAGGLE_DATASET_URL"]
        assert kaggle["notebook_url"] == env_vars["KAGGLE_NOTEBOOK_URL"]
        assert kaggle["model_url"] == env_vars["KAGGLE_MODEL_URL"]

    # When unset, defaults to empty strings
    with patch.dict(os.environ, {"KAGGLE_DATASET_URL": "", "KAGGLE_NOTEBOOK_URL": "", "KAGGLE_MODEL_URL": ""}):
        response = client.get("/api/model-info")
        assert response.status_code == 200
        kaggle = response.json()["kaggle"]
        assert kaggle["dataset_url"] == ""
        assert kaggle["notebook_url"] == ""
        assert kaggle["model_url"] == ""


# -----------------------------------------------------------------------------
# Missing Required File Check Test
# -----------------------------------------------------------------------------

def test_missing_required_file_stops_startup(tmp_path: Path):
    """Verify load_application_data raises descriptive RuntimeError when a required file is missing."""
    # Pass an empty directory
    with pytest.raises(RuntimeError) as exc_info:
        load_application_data(base_dir=tmp_path)

    err_msg = str(exc_info.value)
    assert "Missing required file:" in err_msg
    # Ensure command name is included in message
    assert "Run '" in err_msg
