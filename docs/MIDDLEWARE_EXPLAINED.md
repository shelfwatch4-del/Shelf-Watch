# Middleware explained (Task B14, SRS 7.4, FR-MW-1 to FR-MW-6)

All middleware lives in [api/middleware.py](../api/middleware.py) and is attached by one call, `register_middleware(app)`, at [api/main.py:415](../api/main.py#L415). A request passes through the layers in this order (outermost first):

`CORS -> RequestContext -> RateLimit -> ApiKey -> route`

Because RequestContext sits outside RateLimit and ApiKey, even a 401 or 429 gets the timing headers and an audit row. Tests: [tests/test_middleware.py](../tests/test_middleware.py).

## 1. CORS (FR-MW-1)
Cross-origin browser requests are allowed only from origins listed in the `ALLOWED_ORIGINS` environment variable (comma-separated); an empty value allows none, and only GET, POST, PUT, PATCH and DELETE are permitted. The list is parsed by `allowed_origins()` ([middleware.py:160](../api/middleware.py#L160)) and passed to FastAPI's `CORSMiddleware` inside `register_middleware` ([middleware.py:170](../api/middleware.py#L170)), with the methods in `CORS_METHODS` ([middleware.py:30](../api/middleware.py#L30)). It is read once at start-up, so changing `ALLOWED_ORIGINS` needs a restart.

## 2. Request ID and timing (FR-MW-2)
Every request gets a fresh UUID and a stopwatch, and every response carries `X-Request-ID` and `X-Process-Time-Ms` so one call can be traced from the browser to the server log. This is `RequestContextMiddleware.dispatch` ([middleware.py:57](../api/middleware.py#L57)), where the headers are set right after `call_next`. It also catches any exception from inner layers so even a failing response carries the headers.

## 3. Audit log (FR-MW-3)
After the response is built, the same middleware writes method, path, status code and duration to the `api_logs` table for every path starting with `/api`, except `/api/logs` itself. The write is `_write_audit` ([middleware.py:47](../api/middleware.py#L47)), called from the `try/except` in `dispatch` ([middleware.py:71-78](../api/middleware.py#L71-L78)), so a database failure is only logged as a warning and never breaks the request. Only those five columns are stored (never the key, headers, query string or body). These rows are read with `GET /api/logs` ([main.py:1119](../api/main.py#L1119)); saved predictions are at `GET /api/predictions/history`.

## 4. API key (FR-MW-4)
Write requests (PUT and DELETE on `/api/inventory` and `/api/orders`, `POST /api/orders`, and any PATCH) must send `X-API-Key` equal to the `API_KEY` environment variable (there is no default), otherwise the response is 401 `{"detail": "Invalid or missing API key"}`. The rule is `requires_api_key` ([middleware.py:136](../api/middleware.py#L136)) and the check is `ApiKeyMiddleware.dispatch` ([middleware.py:147](../api/middleware.py#L147)), which compares with `hmac.compare_digest` to avoid timing leaks. If `API_KEY` is unset or empty every write is refused (fail closed). GET requests and the prediction endpoints are never checked, and the key is read per request.

## 5. Rate limit (FR-MW-5)
Each client may make `RATE_LIMIT` requests per rolling 60 seconds (default 120); the next one gets 429 `{"detail": "Rate limit exceeded"}` until old requests age out of the window. `RateLimitMiddleware.dispatch` ([middleware.py:117](../api/middleware.py#L117)) keys clients by the first `X-Forwarded-For` address when the socket peer is 127.0.0.1 or ::1 (`client_address`), otherwise by the socket address, and keeps a deque of timestamps per client in `_hits` ([middleware.py:83](../api/middleware.py#L83)), guarded by a lock. Setting `RATE_LIMIT=0` disables it, which [tests/conftest.py](../tests/conftest.py) does for the whole suite; the counters are in memory, so they reset on restart and are not shared between server processes.

## 6. Global exception handler (FR-MW-6)
Any unhandled error returns `{"detail": "Internal server error"}` with status 500 and no stack trace, while the full traceback goes only to the server log. `unhandled_exception_handler` ([middleware.py:39](../api/middleware.py#L39)) does this and is registered for `Exception` at [middleware.py:166](../api/middleware.py#L166); `RequestContextMiddleware` also calls it ([middleware.py:65](../api/middleware.py#L65)) so the 500 still gets the request headers and an audit row. Deliberate errors (`HTTPException`, validation 422) are not affected and keep their own messages.

## Settings summary

| Variable | Default | Effect |
|---|---|---|
| `ALLOWED_ORIGINS` | empty (none) | CORS allow-list, read at start-up |
| `API_KEY` | none (writes refused) | required in `X-API-Key` on write requests |
| `RATE_LIMIT` | `120` | requests per minute per client, `0` = off |

Note: the old `CORS_ORIGINS` setting and the per-route `_require_api_key` dependency were removed from `main.py`; middleware now does both jobs.
