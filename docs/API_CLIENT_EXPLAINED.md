# How a Python program calls a REST API

This explains [scripts/api_client.py](../scripts/api_client.py), the client for the Shelfwatch API, using its own code.

## 1. The request

A REST call is a message with four parts: a **method** (what to do), a **URL** (which resource), **headers** (metadata) and an optional **body**. The client sends one through `httpx`:

```python
response = self._http.request(method, path, json=json_body, params=params, headers=headers)
```

| Method | Meaning here | Example in the client |
|---|---|---|
| GET | read | `self.request("GET", "/api/health")` |
| POST | create / compute | `self.request("POST", "/api/risk", json_body=...)` |
| PUT | create or replace | `put_inventory` |
| PATCH | partial change | `update_order(id, status="Approved")` |
| DELETE | remove | `delete_order` |

`params=` becomes the query string (`/api/orders?status=Approved`); `json=` becomes the body.

## 2. Headers and the API key

Headers travel with every request. The client always asks for JSON, and adds the key for writes:

```python
headers = {"Accept": "application/json"}
if auth or self.api_key:
    headers["X-API-Key"] = self.api_key
```

The server's `ApiKeyMiddleware` (api/middleware.py) compares this header to its `API_KEY` setting and answers 401 if it is missing or wrong. Reads are public. The key comes from the `api_key` argument, then `SHELFWATCH_API_KEY`, then `API_KEY`; the base URL from the argument, then `SHELFWATCH_URL`, then `http://127.0.0.1:8000`. Never hard-code a key in source.

## 3. JSON in, JSON out

Passing a dict to `json=` makes httpx serialise it and set `Content-Type: application/json`. So `predict("85123A", lead_time_weeks=2)` puts this on the wire:

```json
{"product_id": "85123A", "lead_time_weeks": 2, "service_level": 0.95}
```

The server validates the body against a Pydantic model (`PredictRequest`): types and ranges (`lead_time_weeks` 1 to 4) are checked before any of our code runs. The reply is decoded with `response.json()` into ordinary dicts and lists.

## 4. Status codes

The three-digit code says how it went. Shelfwatch uses:

| Code | Meaning | Typical cause |
|---|---|---|
| 200 / 201 | OK / created | normal success (201 for a new order or stock record) |
| 204 | success, no body | DELETE; the client returns `None` |
| 401 | bad or missing API key | write without `X-API-Key` |
| 404 | not found | unknown product or order id |
| 409 | conflict | deleting an Approved order, an illegal status change |
| 422 | invalid input | out-of-range value, too few history points |
| 429 | rate limit | too many calls from one address |
| 5xx | server fault | bug or overload |

Every error body is `{"detail": "message"}`. The client turns anything 400 or above into an exception that carries both parts, so callers never inspect raw responses:

```python
raise ApiError(response.status_code, _detail_from(response), method, str(response.url))
# ApiError: DELETE http://.../api/orders/7 -> HTTP 409: Only Pending or Cancelled orders can be deleted
```

```python
try:
    api.delete_order(7)
except ApiError as err:
    if err.status_code == 409: ...
```

This is why the demo cancels an order before deleting it: deleting an Approved one returns 409.

## 5. Timeouts

Without a timeout, a hung server hangs your program forever. The client sets 10 seconds on every request (connect, read and write each):

```python
self._http = httpx.Client(base_url=self.base_url, timeout=TIMEOUT_SECONDS)
```

A timeout raises `httpx.TimeoutException`, a kind of `httpx.TransportError`, which feeds into retries.

## 6. Retries

Some failures are momentary: a server restarting, a dropped connection. But repeating a request is only safe if doing it twice has the same effect as doing it once. That property is called **idempotent**. GET (read), PUT (replace with this value) and DELETE (remove) are idempotent. POST (create) and PATCH (change) are not: if a POST creates an order and only the reply is lost, a retry creates a second order.

So the client retries by method, up to 2 extra attempts, waiting a little longer each time:

```python
IDEMPOTENT_METHODS = frozenset({"GET", "PUT", "DELETE"})
NOT_SENT_ERRORS = (httpx.ConnectError, httpx.ConnectTimeout)   # failed before anything was sent

except httpx.TransportError as exc:
    last_exc = exc
    if idempotent or isinstance(exc, NOT_SENT_ERRORS):
        continue          # safe to try again
    break                 # POST/PATCH: the server may have acted, so give up
if response.status_code >= 500 and idempotent and attempt < self.max_retries:
    continue
```

| Failure | GET, PUT, DELETE | POST, PATCH |
|---|---|---|
| Connection refused, connect timeout (nothing was sent) | retried | retried |
| Read timeout, connection reset mid-reply | retried | **not retried** |
| 5xx response | retried | **not retried** |
| 4xx response | not retried | not retried |

A 4xx is never retried: the request itself is wrong, and repeating it gives the same answer. After the last attempt a 5xx raises `ApiError` and a connection failure raises `ApiConnectionError`. A POST or PATCH that times out or gets a 5xx fails at once: check the state (for example `list_orders()`) before trying again.

## 7. Using it

```python
from scripts.api_client import ShelfwatchClient, ApiError

with ShelfwatchClient() as api:               # reads SHELFWATCH_URL / SHELFWATCH_API_KEY
    print(api.predict("85123A")["risk_level"])
    order = api.create_order("85123A", 120)
    api.update_order(order["id"], status="Approved")
```

From the command line, `python scripts/api_client.py demo` runs health, predict, create order, approve, change stock, cancel and delete order, printing each response. The steps run inside try/finally: even if one fails, the demo order is cancelled and deleted and the stock record is restored (or removed if the demo created it) before the error is re-raised. The `with` block closes the connection pool on exit.

## 8. How it is tested

[tests/test_api_client.py](../tests/test_api_client.py) starts the real app with uvicorn on a free local port in a background thread and runs the demo over genuine HTTP. Retry and timeout logic is tested with `httpx.MockTransport`, a fake server that returns scripted responses or raises scripted errors (for example: a GET that times out is called 3 times, a POST that times out is called once) with no network and no waiting.
