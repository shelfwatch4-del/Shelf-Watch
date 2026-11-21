"""tests/test_api_products_forecast.py - Tests for Task B10 /api/products and /api/forecast/{product_id}."""

from __future__ import annotations

import re
from datetime import date
from typing import Any, Dict, List

import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture(scope="module")
def client():
    """TestClient fixture with app lifespan execution."""
    with TestClient(app) as test_client:
        yield test_client


# -----------------------------------------------------------------------------
# GET /api/products Tests
# -----------------------------------------------------------------------------

def test_get_products_count_and_ordering(client: TestClient):
    """Verify GET /api/products returns exactly 100 products ordered by total_units descending."""
    response = client.get("/api/products")
    assert response.status_code == 200

    products: List[Dict[str, Any]] = response.json()
    assert len(products) == 100, f"Expected 100 products, got {len(products)}"

    # Verify fields and types
    for p in products:
        assert "product_id" in p and isinstance(p["product_id"], str)
        assert "description" in p
        assert "total_units" in p and isinstance(p["total_units"], int)
        assert p["total_units"] > 0
        assert "pattern_class" in p
        if p["pattern_class"] is not None:
            assert p["pattern_class"] in {"smooth", "erratic", "intermittent", "lumpy"}

    # Verify descending ordering by training total_units
    total_units_seq = [p["total_units"] for p in products]
    assert total_units_seq == sorted(total_units_seq, reverse=True), (
        "Products are not ordered descending by total_units"
    )


# -----------------------------------------------------------------------------
# GET /api/forecast/{product_id} Tests
# -----------------------------------------------------------------------------

def test_get_forecast_52_history_entries_and_single_null(client: TestClient):
    """Verify history has exactly 52 entries and exactly one null at closure week 2010-12-27."""
    response = client.get("/api/forecast/85123A")
    assert response.status_code == 200

    data = response.json()
    assert data["product_id"] == "85123A"
    assert data["description"] == "WHITE HANGING HEART T-LIGHT HOLDER"
    assert data["closed_weeks"] == ["2010-12-27"]

    history = data["history"]
    assert len(history) == 52, f"Expected 52 history entries, got {len(history)}"

    # Exactly one null entry on the closure week 2010-12-27
    null_entries = [h for h in history if h["units"] is None]
    assert len(null_entries) == 1, (
        f"Expected exactly 1 null history entry, got {len(null_entries)}"
    )
    assert null_entries[0]["week_start"] == "2010-12-27"

    # All other 51 entries must have non-null numeric units
    non_null_entries = [h for h in history if h["units"] is not None]
    assert len(non_null_entries) == 51
    for h in non_null_entries:
        assert isinstance(h["units"], (int, float))
        assert h["units"] >= 0


def test_get_forecast_test_predictions_and_horizon(client: TestClient):
    """Verify test_predictions has 8 test weeks from best model and horizon has valid intervals."""
    response = client.get("/api/forecast/85123A")
    assert response.status_code == 200

    data = response.json()
    test_preds = data["test_predictions"]
    assert len(test_preds) == 8, f"Expected 8 test predictions, got {len(test_preds)}"

    expected_test_weeks = [
        "2011-10-10",
        "2011-10-17",
        "2011-10-24",
        "2011-10-31",
        "2011-11-07",
        "2011-11-14",
        "2011-11-21",
        "2011-11-28",
    ]
    actual_weeks = [p["week_start"] for p in test_preds]
    assert actual_weeks == expected_test_weeks

    for p in test_preds:
        assert isinstance(p["actual"], (int, float))
        assert isinstance(p["predicted"], (int, float))
        assert p["predicted"] >= 0

    # Horizon forecast for 2011-12-05
    horizon = data["next_week_forecast"]
    assert horizon["week_start"] == "2011-12-05"
    assert isinstance(horizon["forecast_weekly"], (int, float))
    assert isinstance(horizon["forecast_p10"], (int, float))
    assert isinstance(horizon["forecast_p90"], (int, float))

    # Interval relationship: p10 <= weekly <= p90
    assert horizon["forecast_p10"] <= horizon["forecast_weekly"] <= horizon["forecast_p90"]
    assert horizon["forecast_p10"] >= 0


def test_get_forecast_unknown_product_404(client: TestClient):
    """Verify requesting an unknown product SKU returns 404 with exact detail message."""
    response = client.get("/api/forecast/NONEXISTENT_SKU_999")
    assert response.status_code == 404
    data = response.json()
    assert data == {"detail": "Product not found"}


def test_no_nan_tokens_in_json(client: TestClient):
    """Verify raw JSON text contains no NaN or nan or Infinity tokens."""
    # Test on products endpoint
    prod_resp = client.get("/api/products")
    assert "NaN" not in prod_resp.text
    assert "Infinity" not in prod_resp.text
    # Ensure word 'nan' doesn't appear as a bare literal token (regex search)
    assert not re.search(r'\b[nN]an\b', prod_resp.text)

    # Test on forecast endpoints for multiple SKUs
    for sku in ["85123A", "22423", "85099B"]:
        fc_resp = client.get(f"/api/forecast/{sku}")
        assert "NaN" not in fc_resp.text
        assert "Infinity" not in fc_resp.text
        assert not re.search(r'\b[nN]an\b', fc_resp.text)


def test_iso_dates_format(client: TestClient):
    """Verify all dates in history, closed_weeks, test_predictions, and next_week_forecast are ISO 8601."""
    response = client.get("/api/forecast/85123A")
    assert response.status_code == 200
    data = response.json()

    for cw in data["closed_weeks"]:
        d = date.fromisoformat(cw)
        assert str(d) == cw

    for h in data["history"]:
        d = date.fromisoformat(h["week_start"])
        assert str(d) == h["week_start"]

    for tp in data["test_predictions"]:
        d = date.fromisoformat(tp["week_start"])
        assert str(d) == tp["week_start"]

    d_horizon = date.fromisoformat(data["next_week_forecast"]["week_start"])
    assert str(d_horizon) == data["next_week_forecast"]["week_start"]
