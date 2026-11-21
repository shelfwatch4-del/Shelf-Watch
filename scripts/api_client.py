"""scripts/api_client.py - Python client for the Shelfwatch REST API (Task B18).

Library use::

    from scripts.api_client import ShelfwatchClient
    with ShelfwatchClient("http://127.0.0.1:8000", api_key="YOUR_API_KEY") as api:
        print(api.health())
        print(api.predict("85123A"))

Command line::

    python scripts/api_client.py demo [--base-url URL] [--api-key KEY]

Base URL and API key come from arguments or the environment
(SHELFWATCH_URL, SHELFWATCH_API_KEY; API_KEY is accepted as a fallback key).

Behaviour (SRS 7.5, FR-PC-1 to FR-PC-3):
- 10 second timeout on every request.
- Retries (up to 2, with a short back-off) depend on the method:
    GET, PUT, DELETE (idempotent): retried on 5xx, connection errors and timeouts.
    POST, PATCH (not idempotent): retried only when the connection could not be
    made at all (connection refused or connect timeout, so nothing was sent).
    Never retried on a read timeout or a 5xx, because the server may have acted.
- Any non-2xx final answer raises ApiError carrying the status code and the
  server's ``detail`` message; an unreachable server raises ApiConnectionError.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence

import httpx

DEFAULT_BASE_URL = "http://127.0.0.1:8000"
TIMEOUT_SECONDS = 10.0
MAX_RETRIES = 2  # retries after the first attempt, so at most 3 attempts in total
BACKOFF_SECONDS = 0.3
IDEMPOTENT_METHODS = frozenset({"GET", "PUT", "DELETE"})
# Raised before any request bytes leave the client, so retrying cannot duplicate work.
NOT_SENT_ERRORS = (httpx.ConnectError, httpx.ConnectTimeout)


class ShelfwatchError(Exception):
    """Base class for every error this client raises."""


class ApiError(ShelfwatchError):
    """The server answered with a non-2xx status."""

    def __init__(self, status_code: int, detail: str, method: str = "", url: str = ""):
        self.status_code = status_code
        self.detail = detail
        self.method = method
        self.url = url
        super().__init__(f"{method} {url} -> HTTP {status_code}: {detail}".strip())


class ApiConnectionError(ShelfwatchError):
    """No usable answer: connection refused, DNS failure or timeout, after all retries."""


def _detail_from(response: httpx.Response) -> str:
    """Pull the server's message out of an error body ({"detail": ...})."""
    try:
        body = response.json()
    except ValueError:
        return response.text.strip() or response.reason_phrase
    if isinstance(body, dict) and "detail" in body:
        detail = body["detail"]
        return detail if isinstance(detail, str) else json.dumps(detail)
    return json.dumps(body)


class ShelfwatchClient:
    """Thin wrapper over httpx.Client with auth, timeout, retries and clean errors."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = TIMEOUT_SECONDS,
        max_retries: int = MAX_RETRIES,
        http_client: Optional[httpx.Client] = None,
    ):
        self.base_url = (base_url or os.getenv("SHELFWATCH_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.api_key = api_key or os.getenv("SHELFWATCH_API_KEY") or os.getenv("API_KEY") or ""
        self.max_retries = max_retries
        self._http = http_client or httpx.Client(base_url=self.base_url, timeout=timeout)

    # -- plumbing ----------------------------------------------------------

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "ShelfwatchClient":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    def request(self, method: str, path: str, *, json_body: Any = None,
                params: Optional[Dict[str, Any]] = None, auth: bool = False) -> Any:
        """Send one request; return the decoded JSON body (None for 204)."""
        headers = {"Accept": "application/json"}
        if auth or self.api_key:
            headers["X-API-Key"] = self.api_key
        method = method.upper()
        idempotent = method in IDEMPOTENT_METHODS
        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            if attempt:
                time.sleep(BACKOFF_SECONDS * attempt)
            try:
                response = self._http.request(
                    method, path, json=json_body, params=params, headers=headers
                )
            except httpx.TransportError as exc:
                last_exc = exc
                if idempotent or isinstance(exc, NOT_SENT_ERRORS):
                    continue
                break  # POST/PATCH: the request may have reached the server; do not repeat it
            if response.status_code >= 500 and idempotent and attempt < self.max_retries:
                continue
            if response.status_code >= 400:
                raise ApiError(response.status_code, _detail_from(response), method, str(response.url))
            if response.status_code == 204 or not response.content:
                return None
            return response.json()
        raise ApiConnectionError(
            f"{method} {self.base_url}{path} failed after {attempt + 1} attempt(s): "
            f"{type(last_exc).__name__}: {last_exc}"
        )

    # -- meta / catalog / forecast ----------------------------------------

    def health(self) -> Dict[str, Any]:
        return self.request("GET", "/api/health")

    def products(self) -> List[Dict[str, Any]]:
        return self.request("GET", "/api/products")

    def forecast(self, product_id: str) -> Dict[str, Any]:
        return self.request("GET", f"/api/forecast/{product_id}")

    # -- prediction / risk -------------------------------------------------

    def predict(self, product_id: str, lead_time_weeks: int = 1, service_level: float = 0.95,
                on_hand: Optional[float] = None) -> Dict[str, Any]:
        body: Dict[str, Any] = {"product_id": product_id, "lead_time_weeks": lead_time_weeks,
                                "service_level": service_level}
        if on_hand is not None:
            body["on_hand"] = on_hand
        return self.request("POST", "/api/predict", json_body=body)

    def predict_custom(self, recent_units: Sequence[float], target_week: str = "2011-12-05",
                       **optional: Any) -> Dict[str, Any]:
        """optional: recent_prices, recent_orders, recent_customers (same length as units)."""
        body = {"recent_units": list(recent_units), "target_week": target_week, **optional}
        return self.request("POST", "/api/predict/custom", json_body=body)

    def risk(self, lead_time_weeks: int = 1, service_level: float = 0.95,
             initial_stock_weeks: float = 3.0) -> Dict[str, Any]:
        return self.request("POST", "/api/risk", json_body={
            "lead_time_weeks": lead_time_weeks, "service_level": service_level,
            "initial_stock_weeks": initial_stock_weeks})

    def simulate(self, lead_time_weeks: int = 2, delay_weeks: int = 1, service_level: float = 0.95,
                 initial_stock_weeks: float = 3.0) -> Dict[str, Any]:
        return self.request("POST", "/api/simulate", json_body={
            "lead_time_weeks": lead_time_weeks, "delay_weeks": delay_weeks,
            "service_level": service_level, "initial_stock_weeks": initial_stock_weeks})

    # -- inventory CRUD ----------------------------------------------------

    def list_inventory(self) -> List[Dict[str, Any]]:
        return self.request("GET", "/api/inventory")

    def put_inventory(self, product_id: str, on_hand: float, lead_time_weeks: int = 2,
                      service_level: float = 0.95) -> Dict[str, Any]:
        return self.request("PUT", f"/api/inventory/{product_id}", auth=True, json_body={
            "on_hand": on_hand, "lead_time_weeks": lead_time_weeks, "service_level": service_level})

    def delete_inventory(self, product_id: str) -> None:
        self.request("DELETE", f"/api/inventory/{product_id}", auth=True)

    # -- order CRUD --------------------------------------------------------

    def list_orders(self, status: Optional[str] = None,
                    product_id: Optional[str] = None) -> List[Dict[str, Any]]:
        params = {k: v for k, v in {"status": status, "product_id": product_id}.items() if v}
        return self.request("GET", "/api/orders", params=params)

    def get_order(self, order_id: int) -> Dict[str, Any]:
        return self.request("GET", f"/api/orders/{order_id}")

    def create_order(self, product_id: str, quantity: int,
                     supplier_note: Optional[str] = None) -> Dict[str, Any]:
        body: Dict[str, Any] = {"product_id": product_id, "quantity": quantity}
        if supplier_note:
            body["supplier_note"] = supplier_note
        return self.request("POST", "/api/orders", auth=True, json_body=body)

    def update_order(self, order_id: int, status: Optional[str] = None,
                     quantity: Optional[int] = None) -> Dict[str, Any]:
        body = {k: v for k, v in {"status": status, "quantity": quantity}.items() if v is not None}
        return self.request("PATCH", f"/api/orders/{order_id}", auth=True, json_body=body)

    def delete_order(self, order_id: int) -> None:
        self.request("DELETE", f"/api/orders/{order_id}", auth=True)


# -- demo -------------------------------------------------------------------

def run_demo(api: ShelfwatchClient, out=print) -> Dict[str, Any]:
    """Walk through health, predict, create order, approve, change stock, delete order.

    An Approved order cannot be deleted (409), so the order is cancelled first.
    Whatever happens, the finally block puts the server back as it was found: the
    product's stock record is restored (or removed if there was none) and the demo
    order is cancelled and deleted. A failing step re-raises after cleanup.
    """
    def show(title: str, payload: Any) -> None:
        out(f"\n== {title}")
        out(json.dumps(payload, indent=2, default=str) if payload is not None else "(204 No Content)")

    results: Dict[str, Any] = {}
    pid: Optional[str] = None
    previous: Optional[Dict[str, Any]] = None
    order_id: Optional[int] = None
    stock_touched = False
    try:
        results["health"] = api.health()
        show("GET /api/health", results["health"])

        pid = api.products()[0]["product_id"]
        previous = next((r for r in api.list_inventory() if r["product_id"] == pid), None)
        results["predict"] = api.predict(pid, lead_time_weeks=2)
        show(f"POST /api/predict ({pid})", results["predict"])

        order = api.create_order(pid, 120, "demo order")
        order_id = order["id"]
        results["created"] = order
        show("POST /api/orders", order)

        results["approved"] = api.update_order(order_id, status="Approved")
        show(f"PATCH /api/orders/{order_id} -> Approved", results["approved"])

        stock_touched = True
        results["stock"] = api.put_inventory(pid, on_hand=75, lead_time_weeks=2)
        show(f"PUT /api/inventory/{pid}", results["stock"])

        results["cancelled"] = api.update_order(order_id, status="Cancelled")
        show(f"PATCH /api/orders/{order_id} -> Cancelled (Approved orders cannot be deleted)",
             results["cancelled"])

        api.delete_order(order_id)
        order_id = None
        show(f"DELETE /api/orders/{results['created']['id']}", None)
    finally:
        _restore(api, out, pid, previous, order_id, stock_touched)
    return results


def _restore(api: ShelfwatchClient, out, pid: Optional[str], previous: Optional[Dict[str, Any]],
             order_id: Optional[int], stock_touched: bool) -> None:
    """Undo demo side effects. Each step is attempted even if an earlier one fails."""
    def attempt(title: str, fn) -> None:
        try:
            fn()
            out(f"\n== cleanup: {title}")
        except ShelfwatchError as exc:
            out(f"\n== cleanup FAILED: {title}: {exc}")

    if order_id is not None:
        def drop_order() -> None:
            if api.get_order(order_id)["status"] == "Approved":
                api.update_order(order_id, status="Cancelled")
            api.delete_order(order_id)  # a Received order stays: the server refuses (409)
        attempt(f"cancel and delete order {order_id}", drop_order)
    if stock_touched and pid is not None:
        if previous is None:
            attempt(f"remove stock record {pid}", lambda: api.delete_inventory(pid))
        else:
            attempt(f"restore stock record {pid}", lambda: api.put_inventory(
                pid, previous["on_hand"], previous["lead_time_weeks"], previous["service_level"]))


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Shelfwatch API client")
    parser.add_argument("command", choices=["demo"])
    parser.add_argument("--base-url", default=None, help="default: $SHELFWATCH_URL or " + DEFAULT_BASE_URL)
    parser.add_argument("--api-key", default=None, help="default: $SHELFWATCH_API_KEY")
    args = parser.parse_args(argv)
    try:
        with ShelfwatchClient(args.base_url, args.api_key) as api:
            run_demo(api)
    except ShelfwatchError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print("\ndemo finished")
    return 0


if __name__ == "__main__":
    sys.exit(main())
