"""tests/test_api_client.py - scripts/api_client.py against a real uvicorn server (Tasks B18, B18b).

The app runs in a background thread on a free local port, so the demo exercises real
HTTP: sockets, headers, JSON, status codes. Every route the server handles uses a private
in-memory SQLite database (get_db is overridden), so these tests share no state with other
modules and never touch the MySQL database. Retry and timeout behaviour uses
httpx.MockTransport so no sleeping server is needed.
"""

from __future__ import annotations

import os
import socket
import threading
import time

import httpx
import pytest
import uvicorn
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from api.db import get_db, init_db
from api.main import app, data_store
from api.models import Product
from scripts import api_client
from scripts.api_client import ApiConnectionError, ApiError, ShelfwatchClient, run_demo

KEY = os.environ["API_KEY"]


@pytest.fixture(scope="module")
def base_url():
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

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        deadline = time.time() + 60
        while not server.started:
            assert thread.is_alive() and time.time() < deadline, "server failed to start"
            time.sleep(0.1)
        with Session() as sess:  # lifespan has loaded data_store by now
            for p in data_store.products:
                sess.add(Product(product_id=p["product_id"], description=p["description"]))
            sess.commit()
        app.dependency_overrides[get_db] = _override
        yield f"http://127.0.0.1:{port}"
    finally:
        app.dependency_overrides.pop(get_db, None)
        server.should_exit = True
        thread.join(timeout=10)


@pytest.fixture
def api(base_url):
    with ShelfwatchClient(base_url, KEY) as c:
        yield c


def test_demo_runs_end_to_end(api):
    lines = []
    before = api.list_inventory()
    results = run_demo(api, out=lines.append)
    assert results["health"]["status"] == "ok"
    assert results["created"]["status"] == "Pending"
    assert results["approved"]["status"] == "Approved"
    assert results["stock"]["on_hand"] == 75
    assert results["cancelled"]["status"] == "Cancelled"
    with pytest.raises(ApiError) as exc:
        api.get_order(results["created"]["id"])
    assert exc.value.status_code == 404
    assert api.list_inventory() == before  # demo left the stock table as it found it
    assert any("DELETE /api/orders" in line for line in lines)


def test_main_cli_demo(base_url, capsys):
    assert api_client.main(["demo", "--base-url", base_url, "--api-key", KEY]) == 0
    assert "demo finished" in capsys.readouterr().out


def test_read_endpoints(api):
    pid = api.products()[0]["product_id"]
    assert len(api.products()) == 100
    assert api.forecast(pid)["product_id"] == pid
    assert api.risk()["summary"]
    assert api.simulate()["changed"] is not None
    custom = api.predict_custom([10, 12, 9, 11, 13, 10, 12, 11])
    assert custom["forecast_p10"] <= custom["forecast_p90"]


def test_error_carries_status_and_detail(api):
    with pytest.raises(ApiError) as exc:
        api.forecast("NO-SUCH-SKU")
    assert exc.value.status_code == 404
    assert "not found" in exc.value.detail.lower()
    assert "404" in str(exc.value)

    with pytest.raises(ApiError) as exc:
        api.risk(lead_time_weeks=99)
    assert exc.value.status_code == 422


def test_missing_api_key_is_401(base_url, monkeypatch):
    monkeypatch.delenv("API_KEY", raising=False)
    with ShelfwatchClient(base_url, api_key="") as anon:
        with pytest.raises(ApiError) as exc:
            anon.create_order("85123A", 5)
    assert exc.value.status_code == 401


def test_wrong_api_key_is_401(base_url):
    with ShelfwatchClient(base_url, api_key="not-the-key") as bad:
        with pytest.raises(ApiError) as exc:
            bad.create_order("85123A", 5)
    assert exc.value.status_code == 401


def test_env_configuration(monkeypatch):
    monkeypatch.setenv("SHELFWATCH_URL", "http://example.test:9/")
    monkeypatch.setenv("SHELFWATCH_API_KEY", "from-env")
    c = ShelfwatchClient()
    assert c.base_url == "http://example.test:9"
    assert c.api_key == "from-env"
    assert c._http.timeout.read == 10.0
    c.close()


# -- retry policy (fake transport, no network, no waiting) -------------------

def _client_with(handler) -> ShelfwatchClient:
    http = httpx.Client(base_url="http://x", transport=httpx.MockTransport(handler))
    return ShelfwatchClient("http://x", "k", http_client=http)


@pytest.fixture
def no_backoff(monkeypatch):
    monkeypatch.setattr(api_client, "BACKOFF_SECONDS", 0)


def _counting(exc=None, statuses=None):
    """Fake transport handler: raises exc on every call, or answers from statuses in turn."""
    calls = []

    def handler(request):
        calls.append(request.method)
        if exc is not None:
            raise exc
        code = statuses[min(len(calls) - 1, len(statuses) - 1)]
        return httpx.Response(code, json={"ok": 1} if code < 400 else {"detail": f"status {code}"})

    return handler, calls


def test_get_retries_5xx_then_succeeds(no_backoff):
    handler, calls = _counting(statuses=[503, 503, 200])
    assert _client_with(handler).request("GET", "/x") == {"ok": 1}
    assert len(calls) == 3  # first attempt + 2 retries


@pytest.mark.parametrize("method", ["GET", "PUT", "DELETE"])
def test_idempotent_methods_give_up_after_two_retries_on_5xx(no_backoff, method):
    handler, calls = _counting(statuses=[500])
    with pytest.raises(ApiError) as exc:
        _client_with(handler).request(method, "/x")
    assert (exc.value.status_code, exc.value.detail, len(calls)) == (500, "status 500", 3)


@pytest.mark.parametrize("method", ["GET", "PUT", "DELETE"])
def test_idempotent_read_timeout_is_retried_twice(no_backoff, method):
    handler, calls = _counting(exc=httpx.ReadTimeout("slow"))
    with pytest.raises(ApiConnectionError):
        _client_with(handler).request(method, "/x")
    assert len(calls) == 3


@pytest.mark.parametrize("method", ["POST", "PATCH"])
def test_post_patch_read_timeout_is_not_retried(no_backoff, method):
    handler, calls = _counting(exc=httpx.ReadTimeout("slow"))
    with pytest.raises(ApiConnectionError):
        _client_with(handler).request(method, "/x", json_body={"a": 1})
    assert len(calls) == 1


@pytest.mark.parametrize("method", ["POST", "PATCH"])
def test_post_patch_5xx_is_not_retried(no_backoff, method):
    handler, calls = _counting(statuses=[503, 200])
    with pytest.raises(ApiError) as exc:
        _client_with(handler).request(method, "/x", json_body={"a": 1})
    assert exc.value.status_code == 503 and len(calls) == 1


@pytest.mark.parametrize("method", ["POST", "PATCH"])
def test_post_patch_retried_on_connection_refused(no_backoff, method):
    handler, calls = _counting(exc=httpx.ConnectError("refused"))
    with pytest.raises(ApiConnectionError):
        _client_with(handler).request(method, "/x", json_body={"a": 1})
    assert len(calls) == 3


def test_post_other_transport_errors_not_retried(no_backoff):
    handler, calls = _counting(exc=httpx.ReadError("connection reset mid-response"))
    with pytest.raises(ApiConnectionError):
        _client_with(handler).request("POST", "/x", json_body={})
    assert len(calls) == 1


def test_4xx_is_not_retried(no_backoff):
    handler, calls = _counting(statuses=[409])
    with pytest.raises(ApiError):
        _client_with(handler).request("DELETE", "/x")
    assert len(calls) == 1


# -- demo cleanup ------------------------------------------------------------

class _Boom(Exception):
    pass


def test_demo_failure_midway_restores_pre_existing_state(api, monkeypatch):
    """The cancel step fails once, after the order was approved and stock changed. The finally
    block must leave no demo order behind and the original stock record (on_hand 33) intact."""
    pid = api.products()[0]["product_id"]
    api.put_inventory(pid, on_hand=33, lead_time_weeks=3, service_level=0.9)
    inventory_before = api.list_inventory()
    orders_before = api.list_orders()

    real_update = api.update_order
    state = {"failed": False}

    def flaky_update(order_id, status=None, quantity=None):
        if status == "Cancelled" and not state["failed"]:
            state["failed"] = True
            raise _Boom("injected failure")
        return real_update(order_id, status=status, quantity=quantity)

    monkeypatch.setattr(api, "update_order", flaky_update)
    lines = []
    with pytest.raises(_Boom):
        run_demo(api, out=lines.append)
    assert state["failed"]
    assert api.list_orders() == orders_before        # demo order cancelled and deleted
    assert api.list_inventory() == inventory_before  # original stock restored
    assert any("cleanup" in line for line in lines)
    api.delete_inventory(pid)


def test_demo_failure_removes_stock_record_it_created(api, monkeypatch):
    """Failure at the very last step: the demo-created stock record must be removed."""
    pid = api.products()[0]["product_id"]
    assert all(r["product_id"] != pid for r in api.list_inventory())
    orders_before = api.list_orders()

    real_delete = api.delete_order
    state = {"failed": False}

    def flaky_delete(order_id):
        if not state["failed"]:
            state["failed"] = True
            raise _Boom("injected failure")
        return real_delete(order_id)

    monkeypatch.setattr(api, "delete_order", flaky_delete)
    with pytest.raises(_Boom):
        run_demo(api, out=lambda *_: None)
    assert state["failed"]
    assert all(r["product_id"] != pid for r in api.list_inventory())
    assert api.list_orders() == orders_before


def test_docs_render_with_tags_and_examples(base_url):
    assert httpx.get(f"{base_url}/docs").status_code == 200
    spec = httpx.get(f"{base_url}/openapi.json").json()
    for path, ops in spec["paths"].items():
        for method, op in ops.items():
            assert op.get("tags") and op.get("summary"), f"{method} {path}"
    schemas = spec["components"]["schemas"]
    for name in ("PredictRequest", "PredictCustomRequest", "RiskRequest", "SimulateRequest",
                 "InventoryCreateUpdateRequest", "OrderCreateRequest", "OrderUpdateRequest"):
        assert "example" in schemas[name], name
