"""Tests for Task B14: middleware (CORS, request id/timing, audit, API key, rate limit, 500)."""

from __future__ import annotations

import os

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api import middleware
from api.main import app, data_store
from api.middleware import register_middleware, reset_rate_limiter

KEY = os.environ["API_KEY"]  # set explicitly by tests/conftest.py


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as tc:
        yield tc


@pytest.fixture(autouse=True)
def _clean_limiter():
    reset_rate_limiter()
    yield
    reset_rate_limiter()


def _stock_body():
    return {"on_hand": 50, "lead_time_weeks": 2, "service_level": 0.95}


def _pid():
    return data_store.products[0]["product_id"]


def test_headers_on_every_response(client):
    for resp in (
        client.get("/api/health"),
        client.get("/api/products/nope-not-a-route"),  # 404
        client.put(f"/api/inventory/{_pid()}", json=_stock_body()),  # 401
    ):
        assert resp.headers["X-Request-ID"]
        assert float(resp.headers["X-Process-Time-Ms"]) >= 0


def test_write_without_key_is_401(client, monkeypatch):
    monkeypatch.setenv("API_KEY", "key-for-this-test")
    for resp in (
        client.put(f"/api/inventory/{_pid()}", json=_stock_body()),
        client.put(f"/api/inventory/{_pid()}", json=_stock_body(), headers={"X-API-Key": "wrong"}),
        client.delete(f"/api/inventory/{_pid()}"),
        client.post("/api/orders", json={"product_id": _pid(), "quantity": 5}),
        client.patch("/api/orders/1", json={"status": "Approved"}),
        client.delete("/api/orders/1"),
    ):
        assert resp.status_code == 401
        assert resp.json() == {"detail": "Invalid or missing API key"}


def test_write_with_key_succeeds_and_reads_are_open(client, monkeypatch):
    monkeypatch.setenv("API_KEY", "key-for-this-test")
    headers = {"X-API-Key": "key-for-this-test"}
    resp = client.put(f"/api/inventory/{_pid()}", json=_stock_body(), headers=headers)
    assert resp.status_code in (200, 201)
    assert client.delete(f"/api/inventory/{_pid()}", headers=headers).status_code == 204
    # reads and predictions need no key
    assert client.get("/api/inventory").status_code == 200
    assert client.get("/api/orders").status_code == 200
    assert client.post("/api/predict", json={"product_id": _pid()}).status_code == 200


def test_requests_appear_in_logs(client):
    client.get("/api/health")
    resp = client.get("/api/health")
    rid = resp.headers["X-Request-ID"]
    items = client.get("/api/logs?limit=50").json()
    assert isinstance(items, list)
    match = [r for r in items if r["request_id"] == rid]
    assert len(match) == 1
    assert match[0]["method"] == "GET"
    assert match[0]["path"] == "/api/health"
    assert match[0]["status"] == 200
    assert match[0]["duration_ms"] >= 0
    # /api/logs itself is never audited
    assert not any(r["path"] == "/api/logs" for r in items)
    assert set(match[0]) == {
        "id", "request_id", "method", "path", "status", "duration_ms", "created_at"
    }
    assert [r["id"] for r in items] == sorted((r["id"] for r in items), reverse=True)


def test_audit_row_never_stores_key_or_body(client, monkeypatch):
    from sqlalchemy import text

    from api.db import engine

    monkeypatch.setenv("API_KEY", "key-for-this-test")
    secret_body = {"on_hand": 4321, "lead_time_weeks": 2, "service_level": 0.95}
    resp = client.put(
        f"/api/inventory/{_pid()}?token=query-secret",
        json=secret_body,
        headers={"X-API-Key": "key-for-this-test"},
    )
    rid = resp.headers["X-Request-ID"]
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT * FROM api_logs WHERE request_id = :r"), {"r": rid}
        ).mappings().one()
    assert set(row) == {"id", "request_id", "method", "path", "status", "duration_ms", "created_at"}
    stored = " ".join(str(v) for v in row.values())
    assert "key-for-this-test" not in stored
    assert "4321" not in stored and "query-secret" not in stored
    client.delete(f"/api/inventory/{_pid()}", headers={"X-API-Key": "key-for-this-test"})


@pytest.mark.parametrize("value", ["", None])
def test_fail_closed_without_api_key(client, monkeypatch, value):
    if value is None:
        monkeypatch.delenv("API_KEY", raising=False)
    else:
        monkeypatch.setenv("API_KEY", value)
    for headers in ({}, {"X-API-Key": ""}, {"X-API-Key": "anything"}):
        resp = client.put(f"/api/inventory/{_pid()}", json=_stock_body(), headers=headers)
        assert resp.status_code == 401
        assert resp.json() == {"detail": "Invalid or missing API key"}


def test_audit_failure_never_breaks_request(client, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("db down")

    monkeypatch.setattr(middleware, "_write_audit", boom)
    resp = client.get("/api/health")
    assert resp.status_code == 200


def test_rate_limit_returns_429(client, monkeypatch):
    monkeypatch.setenv("RATE_LIMIT", "3")
    codes = [client.get("/api/disclaimer").status_code for _ in range(5)]
    assert codes == [200, 200, 200, 429, 429]
    over = client.get("/api/disclaimer")
    assert over.json() == {"detail": "Rate limit exceeded"}
    assert over.headers["X-Request-ID"]


def test_rate_limit_keyed_by_forwarded_address_behind_loopback(monkeypatch):
    monkeypatch.setenv("RATE_LIMIT", "2")
    with TestClient(app, client=("127.0.0.1", 50000)) as tc:
        a = {"X-Forwarded-For": "203.0.113.10, 10.0.0.1"}
        b = {"X-Forwarded-For": "203.0.113.20"}
        assert [tc.get("/api/disclaimer", headers=a).status_code for _ in range(3)] == [200, 200, 429]
        assert [tc.get("/api/disclaimer", headers=b).status_code for _ in range(2)] == [200, 200]
        assert tc.get("/api/disclaimer", headers=b).status_code == 429


def test_forwarded_header_ignored_from_non_loopback_peer(monkeypatch):
    monkeypatch.setenv("RATE_LIMIT", "2")
    with TestClient(app, client=("198.51.100.7", 50000)) as tc:
        codes = [
            tc.get("/api/disclaimer", headers={"X-Forwarded-For": f"203.0.113.{i}"}).status_code
            for i in range(3)
        ]
    assert codes == [200, 200, 429]  # spoofed header does not buy a fresh bucket


def _mini_app() -> FastAPI:
    mini = FastAPI()
    register_middleware(mini)

    @mini.get("/api/boom")
    def boom():
        raise RuntimeError("secret internal detail /etc/passwd")

    @mini.get("/api/ok")
    def ok():
        return {"ok": True}

    return mini


def test_unhandled_error_is_generic_500(monkeypatch):
    monkeypatch.setattr(middleware, "_write_audit", lambda *a, **k: None)
    with TestClient(_mini_app(), raise_server_exceptions=False) as tc:
        resp = tc.get("/api/boom")
    assert resp.status_code == 500
    assert resp.json() == {"detail": "Internal server error"}
    assert "secret" not in resp.text and "Traceback" not in resp.text
    assert set(resp.json()) == {"detail"}
    assert resp.headers["X-Request-ID"]


def test_cors_only_for_allowed_origins(monkeypatch):
    monkeypatch.setenv("ALLOWED_ORIGINS", "https://good.example, https://also.example")
    monkeypatch.setattr(middleware, "_write_audit", lambda *a, **k: None)
    with TestClient(_mini_app()) as tc:
        good = tc.get("/api/ok", headers={"Origin": "https://good.example"})
        bad = tc.get("/api/ok", headers={"Origin": "https://evil.example"})
        pre = tc.options(
            "/api/ok",
            headers={"Origin": "https://good.example", "Access-Control-Request-Method": "PUT"},
        )
    assert good.headers["access-control-allow-origin"] == "https://good.example"
    assert "access-control-allow-origin" not in bad.headers
    assert pre.status_code == 200
    assert "PUT" in pre.headers["access-control-allow-methods"]


def test_cors_empty_allows_none(monkeypatch):
    monkeypatch.setenv("ALLOWED_ORIGINS", "")
    monkeypatch.setattr(middleware, "_write_audit", lambda *a, **k: None)
    with TestClient(_mini_app()) as tc:
        resp = tc.get("/api/ok", headers={"Origin": "https://good.example"})
    assert "access-control-allow-origin" not in resp.headers
