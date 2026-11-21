"""tests/test_api.py - HTTP tests for the inventory and order CRUD routes (Task B15).

Covers the happy path, every status code (200/201/204/401/404/409/422), the order
transitions table, cache clearing on every write, and the effect of writes on /api/risk.
Uses an isolated in-memory SQLite database (get_db is overridden).
"""

from __future__ import annotations

import os
from itertools import product as cross

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from api import main
from api.db import get_db, init_db
from api.main import app, data_store
from api.models import Product

KEY = {"X-API-Key": os.environ["API_KEY"]}
STOCK = {"on_hand": 40, "lead_time_weeks": 2, "service_level": 0.95}
STATUSES = ["Pending", "Approved", "Received", "Cancelled"]
ALLOWED = {
    ("Pending", "Approved"),
    ("Pending", "Cancelled"),
    ("Approved", "Received"),
    ("Approved", "Cancelled"),
}


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

    with TestClient(app) as tc:  # lifespan loads data_store
        with Session() as s:
            for p in data_store.products:
                s.add(Product(product_id=p["product_id"], description=p["description"]))
            s.commit()
        app.dependency_overrides[get_db] = _override
        yield tc
    app.dependency_overrides.pop(get_db, None)


@pytest.fixture(autouse=True)
def _fresh_cache():
    main._invalidate_risk_cache()
    yield
    main._invalidate_risk_cache()


@pytest.fixture
def pid():
    return data_store.products[0]["product_id"]


def _risk_item(client, pid, **body):
    r = client.post("/api/risk", json=body)
    assert r.status_code == 200, r.text
    return next(i for i in r.json()["items"] if i["product_id"] == pid)


def _order(client, pid, qty=10, **extra):
    r = client.post("/api/orders", json={"product_id": pid, "quantity": qty, **extra}, headers=KEY)
    assert r.status_code == 201, r.text
    return r.json()


def _patch(client, oid, **body):
    return client.patch(f"/api/orders/{oid}", json=body, headers=KEY)


def _order_in_status(client, pid, status):
    o = _order(client, pid)
    path = {
        "Pending": [],
        "Approved": ["Approved"],
        "Received": ["Approved", "Received"],
        "Cancelled": ["Cancelled"],
    }[status]
    for step in path:
        assert _patch(client, o["id"], status=step).status_code == 200
    return o["id"]


# --------------------------------------------------------------------------- inventory


def test_inventory_happy_path(client, pid):
    r = client.put(f"/api/inventory/{pid}", json=STOCK, headers=KEY)
    assert r.status_code == 201  # new
    body = r.json()
    assert body["product_id"] == pid and body["on_hand"] == 40 and body["lead_time_weeks"] == 2
    assert set(body) == {"product_id", "on_hand", "lead_time_weeks", "service_level", "updated_at"}

    r = client.put(f"/api/inventory/{pid}", json={**STOCK, "on_hand": 55}, headers=KEY)
    assert r.status_code == 200  # replaced
    assert r.json()["on_hand"] == 55

    rows = client.get("/api/inventory").json()
    assert [x["on_hand"] for x in rows if x["product_id"] == pid] == [55]

    assert client.delete(f"/api/inventory/{pid}", headers=KEY).status_code == 204
    assert all(x["product_id"] != pid for x in client.get("/api/inventory").json())


def test_inventory_get_is_open_and_ordered(client):
    a, b = sorted(p["product_id"] for p in data_store.products[:2])
    for p in (b, a):
        assert client.put(f"/api/inventory/{p}", json=STOCK, headers=KEY).status_code == 201
    ids = [x["product_id"] for x in client.get("/api/inventory").json()]  # no key needed
    assert ids == sorted(ids) and {a, b} <= set(ids)
    for p in (a, b):
        client.delete(f"/api/inventory/{p}", headers=KEY)


def test_inventory_404(client):
    assert client.put("/api/inventory/NOPE", json=STOCK, headers=KEY).status_code == 404
    assert client.delete("/api/inventory/NOPE", headers=KEY).status_code == 404


def test_inventory_delete_without_record_is_404(client, pid):
    client.delete(f"/api/inventory/{pid}", headers=KEY)
    r = client.delete(f"/api/inventory/{pid}", headers=KEY)
    assert r.status_code == 404 and "detail" in r.json()


@pytest.mark.parametrize(
    "bad",
    [
        {"on_hand": -1},
        {"on_hand": 10.5},  # stock is whole units
        {"on_hand": "lots"},
        {"lead_time_weeks": 0},
        {"lead_time_weeks": 5},
        {"service_level": 0.5},
        {"service_level": 1.0},
    ],
)
def test_inventory_422(client, pid, bad):
    r = client.put(f"/api/inventory/{pid}", json={**STOCK, **bad}, headers=KEY)
    assert r.status_code == 422
    assert "detail" in r.json()


def test_inventory_422_missing_field(client, pid):
    assert client.put(f"/api/inventory/{pid}", json={"on_hand": 5}, headers=KEY).status_code == 422


def test_inventory_401_without_key(client, pid):
    assert client.put(f"/api/inventory/{pid}", json=STOCK).status_code == 401
    assert client.put(f"/api/inventory/{pid}", json=STOCK, headers={"X-API-Key": "x"}).status_code == 401
    assert client.delete(f"/api/inventory/{pid}").status_code == 401
    assert client.get("/api/inventory").status_code == 200


# --------------------------------------------------------------------------- orders


def test_orders_happy_path(client, pid):
    o = _order(client, pid, 25, supplier_note="rush")
    assert o["status"] == "Pending" and o["quantity"] == 25 and o["supplier_note"] == "rush"
    oid = o["id"]

    assert client.get(f"/api/orders/{oid}").json()["id"] == oid
    assert oid in [x["id"] for x in client.get("/api/orders").json()]

    assert _patch(client, oid, quantity=30).json()["quantity"] == 30
    assert _patch(client, oid, status="Approved").json()["status"] == "Approved"
    assert _patch(client, oid, status="Received").json()["status"] == "Received"

    cancelled = _order(client, pid)["id"]
    assert _patch(client, cancelled, status="Cancelled").status_code == 200
    assert client.delete(f"/api/orders/{cancelled}", headers=KEY).status_code == 204
    assert client.get(f"/api/orders/{cancelled}").status_code == 404


def test_orders_filters(client):
    p1, p2 = data_store.products[0]["product_id"], data_store.products[1]["product_id"]
    o1, o2 = _order(client, p1)["id"], _order(client, p2)["id"]
    _patch(client, o2, status="Approved")

    by_status = {x["id"] for x in client.get("/api/orders", params={"status": "Approved"}).json()}
    assert o2 in by_status and o1 not in by_status
    by_pid = client.get("/api/orders", params={"product_id": p1}).json()
    assert o1 in [x["id"] for x in by_pid] and all(x["product_id"] == p1 for x in by_pid)
    both = client.get("/api/orders", params={"status": "Approved", "product_id": p1}).json()
    assert o1 not in [x["id"] for x in both]


def test_orders_404(client, pid):
    assert client.get("/api/orders/999999").status_code == 404
    assert _patch(client, 999999, status="Approved").status_code == 404
    assert client.delete("/api/orders/999999", headers=KEY).status_code == 404
    r = client.post("/api/orders", json={"product_id": "NOPE", "quantity": 5}, headers=KEY)
    assert r.status_code == 404 and "detail" in r.json()


@pytest.mark.parametrize(
    "body",
    [
        {"quantity": 0},
        {"quantity": -3},
        {"quantity": 2.5},
        {"quantity": "ten"},
        {},  # quantity missing
    ],
)
def test_orders_create_422(client, pid, body):
    r = client.post("/api/orders", json={"product_id": pid, **body}, headers=KEY)
    assert r.status_code == 422


def test_orders_create_422_missing_product(client):
    assert client.post("/api/orders", json={"quantity": 5}, headers=KEY).status_code == 422


def test_orders_patch_422(client, pid):
    oid = _order(client, pid)["id"]
    assert _patch(client, oid).status_code == 422  # empty body
    assert _patch(client, oid, status="Shipped").status_code == 422  # not a status
    assert _patch(client, oid, quantity=0).status_code == 422
    assert client.get("/api/orders/abc").status_code == 422  # non-integer id


def test_orders_401_without_key(client, pid):
    oid = _order(client, pid)["id"]
    for resp in (
        client.post("/api/orders", json={"product_id": pid, "quantity": 5}),
        client.patch(f"/api/orders/{oid}", json={"status": "Approved"}),
        client.delete(f"/api/orders/{oid}"),
        client.post("/api/orders", json={"product_id": pid, "quantity": 5}, headers={"X-API-Key": "bad"}),
    ):
        assert resp.status_code == 401
    assert client.get(f"/api/orders/{oid}").json()["status"] == "Pending"  # untouched


def test_orders_quantity_only_while_pending_409(client, pid):
    oid = _order_in_status(client, pid, "Approved")
    assert _patch(client, oid, quantity=99).status_code == 409


@pytest.mark.parametrize("src,dst", list(cross(STATUSES, STATUSES)))
def test_transitions_table(client, pid, src, dst):
    oid = _order_in_status(client, pid, src)
    r = _patch(client, oid, status=dst)
    if (src, dst) in ALLOWED:
        assert r.status_code == 200, r.text
        assert r.json()["status"] == dst
    else:  # includes same-state and every move out of Received / Cancelled
        assert r.status_code == 409, r.text
        assert "detail" in r.json()
        assert client.get(f"/api/orders/{oid}").json()["status"] == src


@pytest.mark.parametrize("status,code", [("Pending", 204), ("Cancelled", 204), ("Approved", 409), ("Received", 409)])
def test_delete_rules(client, pid, status, code):
    oid = _order_in_status(client, pid, status)
    assert client.delete(f"/api/orders/{oid}", headers=KEY).status_code == code
    assert client.get(f"/api/orders/{oid}").status_code == (404 if code == 204 else 200)


def test_receiving_adds_to_stock(client, pid):
    client.put(f"/api/inventory/{pid}", json={**STOCK, "on_hand": 100}, headers=KEY)
    _order_in_status(client, pid, "Received")  # quantity 10
    on_hand = next(x for x in client.get("/api/inventory").json() if x["product_id"] == pid)["on_hand"]
    assert on_hand == 110
    client.delete(f"/api/inventory/{pid}", headers=KEY)


# --------------------------------------------------------------------------- risk cache


def _warm(client):
    assert client.post("/api/risk", json={}).status_code == 200
    assert main._risk_cache, "risk response should be cached"


def test_every_write_clears_the_risk_cache(client, pid):
    writes = []

    def put():
        return client.put(f"/api/inventory/{pid}", json=STOCK, headers=KEY)

    def put_again():
        return client.put(f"/api/inventory/{pid}", json={**STOCK, "on_hand": 1}, headers=KEY)

    def delete_stock():
        return client.delete(f"/api/inventory/{pid}", headers=KEY)

    state = {}

    def post_order():
        r = client.post("/api/orders", json={"product_id": pid, "quantity": 5}, headers=KEY)
        state["id"] = r.json()["id"]
        return r

    def patch_status():
        return _patch(client, state["id"], status="Approved")

    def patch_qty():
        o = _order(client, pid)
        main._invalidate_risk_cache()
        _warm(client)
        return _patch(client, o["id"], quantity=7)

    def delete_order():
        o = _order(client, pid)
        main._invalidate_risk_cache()
        _warm(client)
        return client.delete(f"/api/orders/{o['id']}", headers=KEY)

    writes = [put, put_again, delete_stock, post_order, patch_status, patch_qty, delete_order]
    for write in writes:
        _warm(client)
        resp = write()
        assert resp.status_code in (200, 201, 204), (write.__name__, resp.text)
        assert main._risk_cache == {}, f"{write.__name__} left a stale risk cache"


def test_failed_write_does_not_need_to_clear(client, pid):
    """Rejected writes change nothing, so the cached answer stays valid."""
    _warm(client)
    before = dict(main._risk_cache)
    assert client.put("/api/inventory/NOPE", json=STOCK, headers=KEY).status_code == 404
    assert client.put(f"/api/inventory/{pid}", json=STOCK).status_code == 401
    assert main._risk_cache == before


# --------------------------------------------------------------------------- effect on /api/risk


def test_stock_write_marks_source_user_in_risk(client, pid):
    assert _risk_item(client, pid)["on_hand_source"] == "simulated"

    client.put(f"/api/inventory/{pid}", json={**STOCK, "on_hand": 7}, headers=KEY)
    item = _risk_item(client, pid)
    assert item["on_hand_source"] == "user"
    assert item["on_hand"] == 7

    client.delete(f"/api/inventory/{pid}", headers=KEY)
    assert _risk_item(client, pid)["on_hand_source"] == "simulated"


def test_approved_order_raises_on_order_and_lowers_recommended_qty(client, pid):
    # Zero stock guarantees a positive recommended order quantity to reduce.
    client.put(f"/api/inventory/{pid}", json={**STOCK, "on_hand": 0}, headers=KEY)
    before = _risk_item(client, pid)
    assert before["recommended_order_qty"] > 0

    qty = 10
    oid = _order(client, pid, qty)["id"]
    _patch(client, oid, status="Approved")
    after = _risk_item(client, pid)

    assert after["on_order"] == before["on_order"] + qty
    assert after["recommended_order_qty"] < before["recommended_order_qty"]

    # Cancelling releases the open quantity again.
    _patch(client, oid, status="Cancelled")
    restored = _risk_item(client, pid)
    assert restored["on_order"] == before["on_order"]
    assert restored["recommended_order_qty"] == before["recommended_order_qty"]

    client.delete(f"/api/inventory/{pid}", headers=KEY)


# --------------------------------------------------------------------------- error shape (C7)


def _assert_string_detail(resp, status):
    assert resp.status_code == status, resp.text
    assert isinstance(resp.json()["detail"], str) and resp.json()["detail"]
    assert resp.headers["X-Request-ID"]  # middleware headers survive error paths


def test_422_bad_body_on_predict(client):
    r = client.post("/api/predict", json={"product_id": "85123A", "lead_time_weeks": "SECRET-VALUE"})
    _assert_string_detail(r, 422)
    assert r.json()["detail"].startswith("lead_time_weeks: ")
    assert "SECRET-VALUE" not in r.text  # submitted values are never echoed
    assert r.json()["errors"][0]["field"] == "lead_time_weeks"


def test_422_not_json_object(client):
    r = client.post("/api/predict", content="{not json", headers={"Content-Type": "application/json"})
    _assert_string_detail(r, 422)


def test_422_missing_required_field(client):
    r = client.post("/api/predict", json={})
    _assert_string_detail(r, 422)
    assert r.json()["detail"] == "product_id: Field required"


def test_422_multiple_errors_joined(client, pid):
    r = client.put(
        f"/api/inventory/{pid}",
        json={"on_hand": -1, "lead_time_weeks": 9, "service_level": 2},
        headers=KEY,
    )
    _assert_string_detail(r, 422)
    assert r.json()["detail"].count("; ") == 2
    assert {e["field"] for e in r.json()["errors"]} == {"on_hand", "lead_time_weeks", "service_level"}


def test_422_out_of_range_risk_param(client):
    r = client.post("/api/risk", json={"lead_time_weeks": 99})
    _assert_string_detail(r, 422)
    assert r.json()["detail"].startswith("lead_time_weeks: ")


def test_422_list_item_names_the_field(client):
    r = client.post("/api/predict/custom", json={"recent_units": [1, 2, "x", 4, 5, 6, 7, 8]})
    _assert_string_detail(r, 422)
    assert r.json()["detail"].startswith("recent_units: ")


def test_422_is_audited(client):
    from api.db import SessionLocal  # the audit middleware writes via SessionLocal, not get_db
    from api.models import ApiLog

    rid = client.post("/api/risk", json={"lead_time_weeks": 99}).headers["X-Request-ID"]
    with SessionLocal() as s:
        row = s.query(ApiLog).filter_by(request_id=rid).one()
    assert row.status == 422 and row.path == "/api/risk"


def test_404_and_405_have_string_detail(client):
    _assert_string_detail(client.get("/api/does-not-exist"), 404)
    _assert_string_detail(client.get("/api/predict"), 405)  # route exists as POST only


def test_401_409_429_have_string_detail(client, pid, monkeypatch):
    _assert_string_detail(client.put(f"/api/inventory/{pid}", json=STOCK), 401)
    oid = _order_in_status(client, pid, "Received")
    _assert_string_detail(_patch(client, oid, status="Pending"), 409)

    from api.middleware import reset_rate_limiter

    monkeypatch.setenv("RATE_LIMIT", "1")
    reset_rate_limiter()
    try:
        client.get("/api/health")
        _assert_string_detail(client.get("/api/health"), 429)
    finally:
        monkeypatch.setenv("RATE_LIMIT", "0")
        reset_rate_limiter()


# =============================================================================
# Task B17: every endpoint against CONTRACTS.md C7 + addendum (T-A-1..3, T-P-1/2)
# =============================================================================

RISK_ITEM_KEYS = {
    "product_id", "description", "forecast_weekly", "forecast_lead_time", "on_hand",
    "safety_stock", "reorder_point", "days_of_cover", "risk_level", "forecast_p10",
    "forecast_p90", "on_hand_source", "stockout_probability", "on_order",
    "recommended_order_qty", "days_to_reorder", "revenue_at_risk", "action",
}
SUMMARY_KEYS = {
    "total_revenue_at_risk", "n_order_now", "n_order_soon", "n_review_stop", "n_hold",
}
PREDICT_KEYS = {
    "product_id", "week_start", "forecast_weekly", "forecast_p10", "forecast_p90",
    "risk_level", "action", "on_hand", "on_hand_source", "reorder_point", "days_of_cover",
    "stockout_probability", "recommended_order_qty", "days_to_reorder", "revenue_at_risk",
}
CUSTOM_BODY = {"recent_units": [30, 42, 38, 51, 47, 40, 55, 60]}


def _json(resp, status=200):
    assert resp.status_code == status, resp.text
    return resp.json()


def _no_customer(obj):
    """Recursively assert no key mentions a customer identifier (T-A-3)."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            assert "customerid" not in str(k).lower().replace("_", ""), k
            _no_customer(v)
    elif isinstance(obj, list):
        for v in obj:
            _no_customer(v)


def test_a1_health_and_root_aliases(client):
    h = _json(client.get("/api/health"))
    assert h["status"] == "ok"
    assert h["database"]["connected"] is True
    assert isinstance(h["database"]["tables"], dict)
    assert _json(client.get("/health")).keys() == h.keys()


def test_a1_products(client):
    rows = _json(client.get("/api/products"))
    assert len(rows) == 100
    assert set(rows[0]) == {"product_id", "description", "total_units", "pattern_class"}
    units = [r["total_units"] for r in rows]
    assert units == sorted(units, reverse=True)


def test_a1_forecast(client, pid):
    d = _json(client.get(f"/api/forecast/{pid}"))
    assert {"product_id", "description", "closed_weeks", "history", "test_predictions",
            "next_week_forecast"} <= set(d)
    assert len(d["history"]) == 52
    closed = set(d["closed_weeks"])
    assert closed and all(h["units"] is None for h in d["history"] if h["week_start"] in closed)
    assert set(d["next_week_forecast"]) >= {
        "week_start", "forecast_weekly", "forecast_p10", "forecast_p90"}
    assert set(d["test_predictions"][0]) >= {"week_start", "actual", "predicted"}


def test_a1_predict_and_root_alias(client, pid):
    body = {"product_id": pid, "lead_time_weeks": 1, "service_level": 0.95, "on_hand": None}
    d = _json(client.post("/api/predict", json=body))
    assert PREDICT_KEYS <= set(d)
    assert _json(client.post("/predict", json=body))["product_id"] == pid


def test_p1_predict_interval_ordered(client):
    for p in data_store.products[:10]:
        d = _json(client.post("/api/predict", json={"product_id": p["product_id"]}))
        assert d["forecast_p10"] <= d["forecast_weekly"] <= d["forecast_p90"]


def test_a1_predict_custom(client):
    d = _json(client.post("/api/predict/custom", json=CUSTOM_BODY))
    assert {"forecast_weekly", "forecast_p10", "forecast_p90", "features_used", "notes"} <= set(d)
    assert d["forecast_p10"] <= d["forecast_weekly"] <= d["forecast_p90"]


@pytest.mark.parametrize(
    "units",
    [[1, 2, 3, 4, 5, 6, 7], [], [30, 42, 38, 51, 47, 40, 55, -1], [1] * 13],
)
def test_p2_predict_custom_rejects_bad_series(client, units):
    r = client.post("/api/predict/custom", json={"recent_units": units})
    assert r.status_code == 422, r.text
    assert isinstance(r.json()["detail"], str)


def test_a1_risk(client):
    d = _json(client.post("/api/risk", json={"lead_time_weeks": 1, "service_level": 0.95,
                                               "initial_stock_weeks": 3}))
    assert len(d["items"]) == 100
    assert RISK_ITEM_KEYS <= set(d["items"][0])
    assert SUMMARY_KEYS <= set(d["summary"])
    assert {i["action"] for i in d["items"]} <= {"ORDER_NOW", "ORDER_SOON", "REVIEW_STOP", "HOLD"}
    assert {i["risk_level"] for i in d["items"]} <= {"HIGH", "MEDIUM", "LOW"}


def test_a1_simulate(client):
    d = _json(client.post("/api/simulate", json={"lead_time_weeks": 2, "delay_weeks": 1,
                                                   "service_level": 0.95,
                                                   "initial_stock_weeks": 3}))
    assert {"params", "summary", "baseline_summary", "changed"} <= set(d)
    for c in d["changed"]:
        assert set(c) == {"product_id", "from", "to"}


def test_a1_sensitivity(client):
    d = _json(client.post("/api/sensitivity", json={"lead_times": [1, 2, 3, 4],
                                                      "service_levels": [0.9, 0.95, 0.99]}))
    assert len(d) == 12
    assert set(d[0]) == {"lead_time_weeks", "service_level", "n_high", "n_medium", "n_low"}


def test_a1_metrics_disclaimer_model_info(client):
    m = _json(client.get("/api/metrics"))
    assert set(m[0]) >= {"model_name", "split", "MAE", "RMSE", "WAPE", "bias"}
    assert "simulated" in _json(client.get("/api/disclaimer"))["text"].lower()
    mi = _json(client.get("/api/model-info"))
    assert {"model_name", "target_mode", "feature_groups", "trained_on", "test_start",
            "headline_metrics", "ablation", "interval_coverage", "policy_backtest",
            "library_versions", "kaggle"} <= set(mi)
    assert set(mi["kaggle"]) == {"dataset_url", "notebook_url", "model_url"}


def test_a1_intelligence(client, pid):
    rows = _json(client.get("/api/intelligence"))
    assert isinstance(rows, list) and rows
    d = _json(client.get(f"/api/intelligence/{pid}"))
    assert set(d) == {"profile", "companions"}


def test_a1_logs_and_history_and_inventory_shapes(client, pid):
    # /api/logs content is covered by test_middleware (T-MW-3) and test_api_insights.
    assert isinstance(_json(client.get("/api/logs?limit=5&offset=0")), list)
    assert isinstance(_json(client.get("/api/predictions/history?limit=5")), list)
    client.put(f"/api/inventory/{pid}", json=STOCK, headers=KEY)
    inv = _json(client.get("/api/inventory"))
    assert set(inv[0]) == {"product_id", "on_hand", "lead_time_weeks", "service_level",
                           "updated_at"}
    client.delete(f"/api/inventory/{pid}", headers=KEY)


def test_a2_unknown_product_404(client):
    assert client.get("/api/forecast/NOPE").status_code == 404
    assert client.get("/api/forecast/NOPE").json() == {"detail": "Product not found"}
    assert client.post("/api/predict", json={"product_id": "NOPE"}).status_code == 404
    assert client.get("/api/intelligence/NOPE").status_code == 404


@pytest.mark.parametrize(
    "path,body",
    [
        ("/api/risk", {"lead_time_weeks": 5}),
        ("/api/risk", {"lead_time_weeks": 0}),
        ("/api/risk", {"service_level": 0.5}),
        ("/api/risk", {"service_level": 1.0}),
        ("/api/risk", {"initial_stock_weeks": 9}),
        ("/api/simulate", {"delay_weeks": 3}),
        ("/api/simulate", {"lead_time_weeks": 9}),
    ],
)
def test_a2_out_of_range_params_422(client, path, body):
    assert client.post(path, json=body).status_code == 422


def test_a3_no_customer_id_anywhere(client, pid):
    responses = [
        client.get("/api/products"),
        client.get(f"/api/forecast/{pid}"),
        client.post("/api/predict", json={"product_id": pid}),
        client.post("/api/predict/custom", json=CUSTOM_BODY),
        client.post("/api/risk", json={}),
        client.post("/api/simulate", json={}),
        client.get("/api/intelligence"),
        client.get(f"/api/intelligence/{pid}"),
        client.get("/api/model-info"),
        client.get("/api/error-analysis"),
        client.get("/api/logs"),
    ]
    for r in responses:
        assert "customerid" not in r.text.lower().replace("_", "").replace(" ", ""), r.url
        if r.headers.get("content-type", "").startswith("application/json"):
            _no_customer(r.json())
