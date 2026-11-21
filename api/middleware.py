"""api/middleware.py - Shelfwatch middleware stack (SRS 7.4, FR-MW-1 to FR-MW-6).

Call register_middleware(app) once. Order, outermost first:
CORS -> RequestContext (id, timing, audit, 500 safety net) -> RateLimit -> ApiKey -> routes.
Settings are read from the environment on each request (CORS: at registration),
so tests can change them with monkeypatch.
"""

from __future__ import annotations

import hmac
import logging
import os
import threading
import time
import uuid
from collections import defaultdict, deque
from typing import Deque, Dict, List

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger("shelfwatch.api")

DEFAULT_RATE_LIMIT = 120
RATE_WINDOW_SECONDS = 60.0
CORS_METHODS = ["GET", "POST", "PUT", "PATCH", "DELETE"]


def _json(status: int, detail: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"detail": detail})


# --- 6. Global exception handler (FR-MW-6) ---------------------------------

async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Log details server-side only; the client gets a generic message."""
    logger.error("Unhandled error on %s %s", request.method, request.url.path, exc_info=exc)
    return _json(500, "Internal server error")


# --- 6b. Validation errors: every 422 has a string detail (C7) ---------------

def _field_name(loc) -> str:
    """Last named element of the error location (skips list indexes); 'body' if none."""
    names = [str(p) for p in loc if not isinstance(p, int)]
    return names[-1] if names else "body"


async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """422 as {"detail": "field: message; ...", "errors": [{field, message}]}.

    Only the field name and pydantic's message are used; the submitted value (`input`)
    is never echoed.
    """
    errors = [
        {"field": _field_name(e.get("loc", ())), "message": str(e.get("msg", "Invalid value"))}
        for e in exc.errors()
    ]
    detail = "; ".join(f"{e['field']}: {e['message']}" for e in errors) or "Invalid request"
    return JSONResponse(status_code=422, content={"detail": detail, "errors": errors})


# --- 2 + 3. Request ID, timing, audit log (FR-MW-2, FR-MW-3) ----------------

def _write_audit(request_id: str, method: str, path: str, status: int, ms: float) -> None:
    from api.db import SessionLocal, write_log

    session = SessionLocal()
    try:
        write_log(session, request_id, method, path, status, round(ms, 2))
    finally:
        session.close()


class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        start = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception as exc:  # safety net so even a 500 carries the headers
            response = await unhandled_exception_handler(request, exc)
        ms = (time.perf_counter() - start) * 1000
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time-Ms"] = f"{ms:.1f}"

        path = request.url.path
        if path.startswith("/api") and path != "/api/logs":
            try:
                await run_in_threadpool(
                    _write_audit, request_id, request.method, path, response.status_code, ms
                )
            except Exception:
                logger.warning("Audit log write failed", exc_info=True)  # never break the request
        return response


# --- 5. Rate limit (FR-MW-5) ------------------------------------------------

_hits: Dict[str, Deque[float]] = defaultdict(deque)
_hits_lock = threading.Lock()


def reset_rate_limiter() -> None:
    with _hits_lock:
        _hits.clear()


def _rate_limit() -> int:
    try:
        return int(os.getenv("RATE_LIMIT", str(DEFAULT_RATE_LIMIT)))
    except ValueError:
        return DEFAULT_RATE_LIMIT


LOOPBACK = {"127.0.0.1", "::1"}


def client_address(request: Request) -> str:
    """Rate-limit key: first X-Forwarded-For hop, but only when the socket peer is loopback.

    A local proxy (ngrok agent, reverse proxy) sits on 127.0.0.1, so the real caller is in
    the header. From any other peer the header is client-controlled and is ignored.
    """
    peer = request.client.host if request.client else "unknown"
    if peer in LOOPBACK:
        forwarded = request.headers.get("x-forwarded-for", "")
        first = forwarded.split(",")[0].strip()
        if first:
            return first
    return peer


class RateLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        limit = _rate_limit()
        if limit <= 0:  # RATE_LIMIT=0 disables the limiter (used in tests)
            return await call_next(request)
        client = client_address(request)
        now = time.monotonic()
        with _hits_lock:
            window = _hits[client]
            while window and now - window[0] >= RATE_WINDOW_SECONDS:
                window.popleft()
            if len(window) >= limit:
                return _json(429, "Rate limit exceeded")
            window.append(now)
        return await call_next(request)


# --- 4. API key (FR-MW-4) ---------------------------------------------------

def requires_api_key(method: str, path: str) -> bool:
    path = path.rstrip("/")
    if method == "PATCH":
        return True
    if method in ("PUT", "DELETE"):
        return path.startswith("/api/inventory") or path.startswith("/api/orders")
    if method == "POST":
        return path in ("/api/orders", "/api/plans/generate")
    return False


class ApiKeyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if requires_api_key(request.method, request.url.path):
            expected = os.getenv("API_KEY", "")
            supplied = request.headers.get("x-api-key", "")
            # Fail closed: with no API_KEY configured, every write is rejected.
            if not expected or not hmac.compare_digest(supplied.encode(), expected.encode()):
                return _json(401, "Invalid or missing API key")
        return await call_next(request)


# --- 1. CORS + registration (FR-MW-1) ---------------------------------------

def allowed_origins() -> List[str]:
    return [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o.strip()]


def register_middleware(app: FastAPI) -> None:
    """Attach the full stack. add_middleware wraps, so the last added is outermost."""
    app.add_exception_handler(Exception, unhandled_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_middleware(ApiKeyMiddleware)
    app.add_middleware(RateLimitMiddleware)
    app.add_middleware(RequestContextMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins(),  # empty list = no origin allowed
        allow_methods=CORS_METHODS,
        allow_headers=["*"],
        expose_headers=["X-Request-ID", "X-Process-Time-Ms"],
    )
