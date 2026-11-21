"""Tests for Task B13: metrics, error-analysis, intelligence, logs."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from api.main import app, data_store


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as tc:
        yield tc


def test_metrics_schema_and_counts(client):
    resp = client.get("/api/metrics")
    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body, list)
    assert len(body) == len(data_store.metrics)
    assert {r["split"] for r in body} == {"val", "test"}
    for row in body:
        assert {"model_name", "split", "MAE", "RMSE", "WAPE", "bias"} <= set(row)


def test_error_analysis_sections(client):
    body = client.get("/api/error-analysis").json()
    assert set(body) == {"segments", "weeks", "products", "note"}
    ea = data_store.error_analysis
    assert len(body["weeks"]) == int((ea["level"] == "week").sum())
    assert len(body["products"]) == int((ea["level"] == "product").sum())
    assert len(body["segments"]) > 0


def test_error_analysis_missing_file(client, monkeypatch):
    monkeypatch.setattr(data_store, "error_analysis", None)
    body = client.get("/api/error-analysis").json()
    assert body["segments"] == body["weeks"] == body["products"] == []
    assert body["note"]


def test_intelligence_list_and_detail(client):
    rows = client.get("/api/intelligence").json()
    assert len(rows) == len(data_store.product_intelligence)
    pid = rows[0]["product_id"]
    detail = client.get(f"/api/intelligence/{pid}")
    assert detail.status_code == 200
    body = detail.json()
    assert str(body["profile"]["product_id"]) == str(pid)
    assert len(body["companions"]) <= 3


def test_intelligence_companions_capped_at_three(client):
    bp = data_store.basket_pairs
    pid = str(bp["product_id"].value_counts().index[0])
    body = client.get(f"/api/intelligence/{pid}").json()
    assert 1 <= len(body["companions"]) <= 3


def test_intelligence_unknown_404(client):
    assert client.get("/api/intelligence/NOPE").status_code == 404


def test_logs_paging(client):
    for _ in range(3):
        client.get("/api/disclaimer")
    full = client.get("/api/logs?limit=200").json()
    assert isinstance(full, list) and len(full) >= 2
    ids = [i["id"] for i in full]
    assert ids == sorted(ids, reverse=True)
    page = client.get("/api/logs?limit=1&offset=1").json()
    assert [i["id"] for i in page] == [ids[1]] or page[0]["id"] < ids[0]


def test_predictions_history(client):
    client.post("/api/predict/custom", json={"recent_units": [10.0] * 8})
    rows = client.get("/api/predictions/history?limit=5").json()
    assert isinstance(rows, list) and 1 <= len(rows) <= 5
    assert {"id", "endpoint", "inputs", "outputs", "created_at"} <= set(rows[0])
    ids = [r["id"] for r in rows]
    assert ids == sorted(ids, reverse=True)
    assert client.get("/api/predictions/history?limit=0").status_code == 422


def test_logs_limit_bounds(client):
    assert client.get("/api/logs?limit=201").status_code == 422
    assert client.get("/api/logs?limit=0").status_code == 422
    assert client.get("/api/logs?offset=-1").status_code == 422
    assert client.get("/api/predictions/history?limit=201").status_code == 422


def test_json_safe(client):
    for path in ("/api/metrics", "/api/error-analysis", "/api/intelligence", "/api/logs"):
        text = client.get(path).text
        assert "NaN" not in text and "Infinity" not in text
        json.loads(text)
        assert "customer_id" not in text.lower()
