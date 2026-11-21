"""Task B19: weekly replenishment plan workflow (/api/plans)."""

from __future__ import annotations

import csv
import io
import os

import pytest
from fastapi.testclient import TestClient

from api.db import SessionLocal
from api.main import _invalidate_risk_cache, app
from api.models import InventoryLevel, PlanLine, PurchaseOrder, ReplenishmentPlan

HEADERS = {"X-API-Key": os.environ["API_KEY"]}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as tc:
        yield tc


def _wipe_plans() -> None:
    s = SessionLocal()
    try:
        s.query(PurchaseOrder).filter(PurchaseOrder.plan_id.isnot(None)).delete()
        s.query(PlanLine).delete()
        s.query(ReplenishmentPlan).delete()
        s.commit()
    finally:
        s.close()
    _invalidate_risk_cache()


@pytest.fixture(autouse=True)
def clean_plans(client):
    _wipe_plans()
    yield
    _wipe_plans()


def _generate(client) -> dict:
    r = client.post("/api/plans/generate", headers=HEADERS)
    assert r.status_code == 201, r.text
    return r.json()


def _plan_orders(plan_id: int) -> list:
    s = SessionLocal()
    try:
        return s.query(PurchaseOrder).filter(PurchaseOrder.plan_id == plan_id).all()
    finally:
        s.close()


def test_generate_builds_draft_from_risk(client):
    plan = _generate(client)
    assert plan["status"] == "Draft"
    assert plan["orders"], "expected at least one ORDER_NOW/ORDER_SOON line"
    risk = {i["product_id"]: i for i in client.post("/api/risk", json={}).json()["items"]}
    for line in plan["orders"]:
        item = risk[line["product_id"]]
        assert item["action"] in ("ORDER_NOW", "ORDER_SOON")
        assert line["quantity"] > 0
        assert line["status"] == "Draft"
    wanted = {
        pid for pid, i in risk.items()
        if i["action"] in ("ORDER_NOW", "ORDER_SOON") and i["recommended_order_qty"] > 0
    }
    assert {l["product_id"] for l in plan["orders"]} == wanted
    assert _plan_orders(plan["id"]) == []  # Draft creates no purchase orders


def test_generate_requires_api_key(client):
    assert client.post("/api/plans/generate").status_code == 401
    assert client.patch("/api/plans/1", json={"status": "Approved"}).status_code == 401


def test_duplicate_draft_is_409(client):
    _generate(client)
    r = client.post("/api/plans/generate", headers=HEADERS)
    assert r.status_code == 409
    assert "detail" in r.json()


def test_list_and_get(client):
    plan = _generate(client)
    listing = client.get("/api/plans").json()
    assert [p["id"] for p in listing] == [plan["id"]]
    got = client.get(f"/api/plans/{plan['id']}").json()
    assert got["orders"] == plan["orders"]
    assert client.get("/api/plans/999999").status_code == 404


def test_approve_creates_pending_orders_linked_to_plan(client):
    plan = _generate(client)
    r = client.patch(f"/api/plans/{plan['id']}", json={"status": "Approved"}, headers=HEADERS)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "Approved" and body["approved_at"]
    orders = _plan_orders(plan["id"])
    assert len(orders) == len(plan["orders"])
    assert {o.status for o in orders} == {"Pending"}
    assert {(o.product_id, o.quantity) for o in orders} == {
        (l["product_id"], l["quantity"]) for l in plan["orders"]
    }
    assert all(l["id"] is not None for l in body["orders"])
    # Approved plan can't be changed again, and a new Draft may now be generated
    assert client.patch(
        f"/api/plans/{plan['id']}", json={"status": "Cancelled"}, headers=HEADERS
    ).status_code == 409
    assert client.post("/api/plans/generate", headers=HEADERS).status_code in (201,)


def test_cancel_creates_no_orders(client):
    plan = _generate(client)
    r = client.patch(f"/api/plans/{plan['id']}", json={"status": "Cancelled"}, headers=HEADERS)
    assert r.status_code == 200
    assert r.json()["status"] == "Cancelled"
    assert _plan_orders(plan["id"]) == []
    assert client.patch(
        f"/api/plans/{plan['id']}", json={"status": "Approved"}, headers=HEADERS
    ).status_code == 409


def test_patch_invalid_status_is_422(client):
    plan = _generate(client)
    r = client.patch(f"/api/plans/{plan['id']}", json={"status": "Received"}, headers=HEADERS)
    assert r.status_code == 422


def test_export_csv_columns(client):
    plan = _generate(client)
    r = client.get(f"/api/plans/{plan['id']}/export")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/csv")
    rows = list(csv.reader(io.StringIO(r.text)))
    assert rows[0] == ["product_id", "description", "quantity", "status"]
    assert len(rows) == len(plan["orders"]) + 1
    assert client.get("/api/plans/999999/export").status_code == 404


def test_receive_updates_stock_and_source_becomes_user(client):
    plan = _generate(client)
    client.patch(f"/api/plans/{plan['id']}", json={"status": "Approved"}, headers=HEADERS)
    order = client.get(f"/api/orders?status=Pending").json()
    order = next(o for o in order if o["id"] in {x.id for x in _plan_orders(plan["id"])})
    pid, qty = order["product_id"], order["quantity"]

    s = SessionLocal()
    try:
        before = s.query(InventoryLevel).filter_by(product_id=pid).first()
        before_qty = before.on_hand if before else 0
    finally:
        s.close()

    # Pending -> Approved -> Received
    assert client.patch(f"/api/orders/{order['id']}", json={"status": "Approved"}, headers=HEADERS).status_code == 200
    r = client.patch(f"/api/orders/{order['id']}", json={"status": "Received"}, headers=HEADERS)
    assert r.status_code == 200, r.text

    s = SessionLocal()
    try:
        after = s.query(InventoryLevel).filter_by(product_id=pid).first()
        assert after.on_hand == before_qty + qty
    finally:
        s.close()

    item = next(i for i in client.post("/api/risk", json={}).json()["items"] if i["product_id"] == pid)
    assert item["on_hand_source"] == "user"
    assert item["on_hand"] == before_qty + qty
