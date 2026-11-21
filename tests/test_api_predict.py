"""tests/test_api_predict.py - Tests for Task B11 /api/predict and /api/predict/custom."""

from __future__ import annotations

from typing import Any, Dict

import pytest
from fastapi.testclient import TestClient

from api.db import SessionLocal
from api.main import app
from api.models import PredictionLog


@pytest.fixture(scope="module")
def client():
    """TestClient fixture with app lifespan execution."""
    with TestClient(app) as test_client:
        yield test_client


# -----------------------------------------------------------------------------
# POST /api/predict and Root Alias /predict Tests
# -----------------------------------------------------------------------------

def test_predict_standard_success_and_interval(client: TestClient):
    """Verify standard predict returns documented fields and p10 <= forecast_weekly <= p90."""
    payload = {
        "product_id": "85123A",
        "lead_time_weeks": 1,
        "service_level": 0.95,
        "on_hand": None,
    }
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 200

    data: Dict[str, Any] = response.json()
    assert data["product_id"] == "85123A"
    assert data["week_start"] == "2011-12-05"

    # Core predictions and intervals
    assert isinstance(data["forecast_weekly"], (int, float))
    assert isinstance(data["forecast_p10"], (int, float))
    assert isinstance(data["forecast_p90"], (int, float))
    assert data["forecast_p10"] <= data["forecast_weekly"] <= data["forecast_p90"]

    # Stock, source, and risk indicators
    assert data["risk_level"] in {"HIGH", "MEDIUM", "LOW"}
    assert data["action"] in {"ORDER_NOW", "ORDER_SOON", "REVIEW_STOP", "HOLD"}
    assert isinstance(data["on_hand"], (int, float))
    assert data["on_hand_source"] in {"user", "db", "simulated"}

    # Replenishment advice
    assert isinstance(data["reorder_point"], (int, float))
    assert isinstance(data["days_of_cover"], (int, float))
    assert 0.0 <= data["stockout_probability"] <= 1.0
    assert data["recommended_order_qty"] >= 0.0
    assert data["days_to_reorder"] >= 0.0
    assert data["revenue_at_risk"] >= 0.0


def test_predict_override_changes_result(client: TestClient):
    """Verify on_hand override changes on_hand_source to 'user', alters risk level and action."""
    # Baseline with simulated / DB stock
    base_resp = client.post(
        "/api/predict",
        json={"product_id": "85123A", "lead_time_weeks": 1, "service_level": 0.95, "on_hand": None},
    )
    assert base_resp.status_code == 200
    base_data = base_resp.json()

    # Extreme high stock override -> must become LOW risk and HOLD action
    override_resp = client.post(
        "/api/predict",
        json={"product_id": "85123A", "lead_time_weeks": 1, "service_level": 0.95, "on_hand": 50000.0},
    )
    assert override_resp.status_code == 200
    override_data = override_resp.json()

    assert override_data["on_hand"] == 50000.0
    assert override_data["on_hand_source"] == "user"
    assert override_data["risk_level"] == "LOW"
    assert override_data["action"] == "HOLD"
    assert override_data["recommended_order_qty"] == 0.0
    assert override_data["stockout_probability"] == 0.0

    # Ensure override changed the outcome vs baseline (if baseline had recommended order > 0)
    assert override_data["on_hand"] != base_data["on_hand"]


def test_predict_root_alias(client: TestClient):
    """Verify POST /predict root alias behaves identically to POST /api/predict (SRS FR-A-13)."""
    payload = {
        "product_id": "22423",
        "lead_time_weeks": 2,
        "service_level": 0.95,
        "on_hand": 100.0,
    }
    api_resp = client.post("/api/predict", json=payload)
    root_resp = client.post("/predict", json=payload)

    assert root_resp.status_code == 200
    assert root_resp.json() == api_resp.json()


def test_predict_unknown_product_404(client: TestClient):
    """Verify unknown product SKU returns 404 with detail message."""
    response = client.post(
        "/api/predict",
        json={"product_id": "INVALID_PRODUCT_XYZ", "lead_time_weeks": 1, "service_level": 0.95},
    )
    assert response.status_code == 404
    assert response.json() == {"detail": "Product not found"}


def test_predict_invalid_params_422(client: TestClient):
    """Verify out-of-range parameters return HTTP 422 validation error."""
    # lead_time_weeks out of range (allowed 1-4)
    r_lt = client.post(
        "/api/predict",
        json={"product_id": "85123A", "lead_time_weeks": 5, "service_level": 0.95},
    )
    assert r_lt.status_code == 422

    # service_level out of range (allowed 0.80-0.99)
    r_sl = client.post(
        "/api/predict",
        json={"product_id": "85123A", "lead_time_weeks": 1, "service_level": 0.75},
    )
    assert r_sl.status_code == 422

    # negative on_hand
    r_oh = client.post(
        "/api/predict",
        json={"product_id": "85123A", "lead_time_weeks": 1, "service_level": 0.95, "on_hand": -5.0},
    )
    assert r_oh.status_code == 422


# -----------------------------------------------------------------------------
# POST /api/predict/custom Tests
# -----------------------------------------------------------------------------

def test_predict_custom_success_and_determinism(client: TestClient):
    """Verify custom prediction succeeds and identical payloads produce identical outputs."""
    payload = {
        "recent_units": [30, 42, 38, 51, 47, 40, 55, 60],
        "target_week": "2011-12-05",
    }
    r1 = client.post("/api/predict/custom", json=payload)
    assert r1.status_code == 200
    data1 = r1.json()

    # Verify interval property
    assert "forecast_weekly" in data1
    assert "forecast_p10" in data1
    assert "forecast_p90" in data1
    assert data1["forecast_p10"] <= data1["forecast_weekly"] <= data1["forecast_p90"]
    assert isinstance(data1["features_used"], list)
    assert len(data1["features_used"]) > 0
    assert isinstance(data1["notes"], str)

    # Determinism: same payload twice gives exact same numbers
    r2 = client.post("/api/predict/custom", json=payload)
    assert r2.status_code == 200
    assert r2.json() == data1


def test_predict_custom_rejects_7_values(client: TestClient):
    """Verify custom prediction rejects fewer than 8 values (422)."""
    payload = {"recent_units": [30, 42, 38, 51, 47, 40, 55]}  # 7 values
    resp = client.post("/api/predict/custom", json=payload)
    assert resp.status_code == 422


def test_predict_custom_rejects_13_values(client: TestClient):
    """Verify custom prediction rejects more than 12 values (422)."""
    payload = {"recent_units": [30, 42, 38, 51, 47, 40, 55, 60, 62, 58, 65, 70, 75]}  # 13 values
    resp = client.post("/api/predict/custom", json=payload)
    assert resp.status_code == 422


def test_predict_custom_rejects_negatives(client: TestClient):
    """Verify custom prediction rejects negative demand numbers (422)."""
    payload = {"recent_units": [30, 42, -10, 51, 47, 40, 55, 60]}  # 8 values with negative
    resp = client.post("/api/predict/custom", json=payload)
    assert resp.status_code == 422


# -----------------------------------------------------------------------------
# Audit Logging Test (SRS FR-A-15)
# -----------------------------------------------------------------------------

def test_predict_logs_to_database(client: TestClient):
    """Verify calls to /api/predict and /api/predict/custom write rows to prediction_log."""
    db = SessionLocal()
    try:
        initial_count = db.query(PredictionLog).count()

        # Trigger predict call
        client.post(
            "/api/predict",
            json={"product_id": "85123A", "lead_time_weeks": 1, "service_level": 0.95},
        )
        count_after_predict = db.query(PredictionLog).count()
        assert count_after_predict == initial_count + 1

        last_entry = db.query(PredictionLog).order_by(PredictionLog.id.desc()).first()
        assert last_entry.endpoint == "/api/predict"
        assert last_entry.product_id == "85123A"
        assert last_entry.inputs is not None
        assert last_entry.outputs is not None

        # Trigger custom predict call
        client.post(
            "/api/predict/custom",
            json={"recent_units": [30, 42, 38, 51, 47, 40, 55, 60]},
        )
        count_after_custom = db.query(PredictionLog).count()
        assert count_after_custom == count_after_predict + 1

        last_custom_entry = db.query(PredictionLog).order_by(PredictionLog.id.desc()).first()
        assert last_custom_entry.endpoint == "/api/predict/custom"
        assert last_custom_entry.inputs is not None
        assert last_custom_entry.outputs is not None
    finally:
        db.close()
