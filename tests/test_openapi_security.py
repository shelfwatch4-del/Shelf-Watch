"""tests/test_openapi_security.py - apiKey scheme and request-body examples (Task B18b).

Uses a private in-memory SQLite database (get_db overridden), like tests/test_api.py.
"""

from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from api import main
from api.db import get_db, init_db
from api.main import app, data_store
from api.middleware import requires_api_key
from api.models import Product

KEY = {"X-API-Key": os.environ["API_KEY"]}
WRITE_METHODS = ("post", "put", "patch", "delete")


@pytest.fixture(scope="module")
def client():
    engine = create_engine(
        "sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    init_db(engine)
    Session = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    def _override():
        db = Session()
        try:
            yield db
        finally:
            db.close()

    with TestClient(app) as tc:
        with Session() as s:
            for p in data_store.products:
                s.add(Product(product_id=p["product_id"], description=p["description"]))
            s.commit()
        app.dependency_overrides[get_db] = _override
        yield tc
    app.dependency_overrides.pop(get_db, None)


@pytest.fixture(scope="module")
def spec(client):
    return client.get("/openapi.json").json()


def test_security_scheme_defined(spec):
    scheme = spec["components"]["securitySchemes"]["ApiKeyAuth"]
    assert scheme["type"] == "apiKey"
    assert scheme["in"] == "header"
    assert scheme["name"] == "X-API-Key"


def test_security_on_exactly_the_write_endpoints(spec):
    protected, open_ = [], []
    for path, ops in spec["paths"].items():
        for method, op in ops.items():
            has = op.get("security") == [{"ApiKeyAuth": []}]
            (protected if has else open_).append((method.upper(), path))
            # the OpenAPI document and the middleware must agree on which routes are protected
            assert has == requires_api_key(method.upper(), path), f"{method.upper()} {path}"
    assert ("POST", "/api/orders") in protected
    assert ("PATCH", "/api/orders/{order_id}") in protected
    assert ("PUT", "/api/inventory/{product_id}") in protected
    assert ("DELETE", "/api/orders/{order_id}") in protected
    assert ("POST", "/api/risk") in open_  # a POST that does not write
    assert ("POST", "/api/plans/generate") in protected
    assert ("PATCH", "/api/plans/{plan_id}") in protected
    assert len(protected) == 7


def _example_for(spec, op):
    schema = op["requestBody"]["content"]["application/json"]["schema"]
    name = schema["$ref"].rsplit("/", 1)[1]
    component = spec["components"]["schemas"][name]
    assert "example" in component, f"{name} has no example"
    return component["example"]


def test_every_request_body_example_is_accepted(client, spec):
    """POST/PUT/PATCH every documented example (writes sent with the test key); none may be 422.
    Order matters only within this test: create the order first so PATCH has an id to use."""
    pid = data_store.products[0]["product_id"]
    cases = []
    for path, ops in spec["paths"].items():
        for method, op in ops.items():
            if "requestBody" in op:
                cases.append((method, path, op))
    assert len(cases) >= 7, "expected every body endpoint to be covered"
    cases.sort(key=lambda c: 0 if (c[0], c[1]) == ("post", "/api/orders") else 1)

    order_id = None
    seen = []
    plan_resp = client.post("/api/plans/generate", headers=KEY)  # PATCH /api/plans/{plan_id} needs a Draft plan
    assert plan_resp.status_code == 201, plan_resp.text
    plan_id = plan_resp.json()["id"]
    try:
        for method, path, op in cases:
            body = dict(_example_for(spec, op))
            if "product_id" in body:  # examples use 85123A; make sure it is a catalogued product
                body["product_id"] = pid
            url = path.replace("{product_id}", pid).replace("{order_id}", str(order_id)).replace("{plan_id}", str(plan_id))
            headers = KEY if op.get("security") else {}
            r = client.request(method.upper(), url, json=body, headers=headers)
            assert r.status_code != 422, f"{method.upper()} {path}: {r.text}"
            assert r.status_code < 500, f"{method.upper()} {path}: {r.text}"
            assert r.status_code in (200, 201), f"{method.upper()} {path}: {r.status_code} {r.text}"
            if (method, path) == ("post", "/api/orders"):
                order_id = r.json()["id"]
            seen.append((method, path))
    finally:
        if order_id is not None:
            client.patch(f"/api/orders/{order_id}", json={"status": "Cancelled"}, headers=KEY)
            client.delete(f"/api/orders/{order_id}", headers=KEY)
        client.delete(f"/api/inventory/{pid}", headers=KEY)
        from api.db import SessionLocal
        from api.models import PlanLine, PurchaseOrder, ReplenishmentPlan

        s = SessionLocal()
        try:  # remove the plan and the orders its approval created
            s.query(PurchaseOrder).filter(PurchaseOrder.plan_id == plan_id).delete()
            s.query(PlanLine).filter(PlanLine.plan_id == plan_id).delete()
            s.query(ReplenishmentPlan).filter(ReplenishmentPlan.id == plan_id).delete()
            s.commit()
        finally:
            s.close()
        main._invalidate_risk_cache()
    assert len(seen) == len(cases)


def test_body_endpoints_without_key_rejected_only_when_protected(client, spec):
    r = client.post("/api/orders", json={"product_id": "x", "quantity": 1})
    assert r.status_code == 401
    r = client.post("/api/risk", json={})
    assert r.status_code == 200
