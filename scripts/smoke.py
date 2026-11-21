"""scripts/smoke.py - end-to-end smoke test for a running Shelfwatch server (Task Q4).

    python scripts/smoke.py [BASE_URL] [--api-key KEY] [--timeout SECONDS]

BASE_URL defaults to SHELFWATCH_URL or http://localhost:8000, so the same script
checks a local server and a public deployment. The write key comes from --api-key,
SHELFWATCH_API_KEY, API_KEY, or an API_KEY= line in ./.env.

Prints PASS or FAIL per check and exits 1 if any check failed (2 if the server
cannot be reached at all). The only data it writes is one order, which it deletes.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple

import httpx

DEFAULT_URL = "http://localhost:8000"
FORBIDDEN = "CustomerID"
FORECAST_WEEK = "2011-12-05"


def _load_key(cli_key: Optional[str]) -> Optional[str]:
    key = cli_key or os.getenv("SHELFWATCH_API_KEY") or os.getenv("API_KEY")
    if key:
        return key
    env_file = Path(__file__).resolve().parent.parent / ".env"
    if env_file.is_file():
        for line in env_file.read_text(encoding="utf-8", errors="ignore").splitlines():
            name, _, value = line.partition("=")
            if name.strip() == "API_KEY" and value.strip():
                return value.strip().strip("\"'")
    return None


class Smoke:
    def __init__(self, base_url: str, api_key: Optional[str], timeout: float):
        self.base = base_url.rstrip("/")
        self.api_key = api_key
        self.http = httpx.Client(timeout=timeout, follow_redirects=True)
        self.results: List[Tuple[str, bool, str]] = []
        self.leaks: List[str] = []

    # -- plumbing ----------------------------------------------------------

    def call(self, method: str, path: str, *, key: bool = False, **kwargs: Any) -> httpx.Response:
        headers = {"X-API-Key": self.api_key} if key and self.api_key else {}
        resp = self.http.request(method, self.base + path, headers=headers, **kwargs)
        if FORBIDDEN.lower() in resp.text.lower():
            self.leaks.append(f"{method} {path}")
        return resp

    def check(self, name: str, fn: Callable[[], Any]) -> Any:
        """Run fn; it returns a value, or raises AssertionError / any error to fail."""
        try:
            value = fn()
        except AssertionError as exc:
            self.results.append((name, False, str(exc) or "assertion failed"))
            print(f"FAIL  {name}: {exc}")
            return None
        except httpx.HTTPError as exc:
            self.results.append((name, False, f"{type(exc).__name__}: {exc}"))
            print(f"FAIL  {name}: {type(exc).__name__}: {exc}")
            return None
        except Exception as exc:  # malformed JSON, missing key, etc.
            self.results.append((name, False, f"{type(exc).__name__}: {exc}"))
            print(f"FAIL  {name}: {type(exc).__name__}: {exc}")
            return None
        self.results.append((name, True, ""))
        print(f"PASS  {name}")
        return value if value is not None else True

    @staticmethod
    def status(resp: httpx.Response, expected: int) -> None:
        assert resp.status_code == expected, (
            f"expected HTTP {expected}, got {resp.status_code}: {resp.text[:150]}"
        )

    # -- checks ------------------------------------------------------------

    def run(self) -> bool:
        c = self.check
        call = self.call
        status = self.status

        def health():
            r = call("GET", "/api/health")
            status(r, 200)
            assert r.json().get("status") == "ok", f"status={r.json().get('status')!r}"
        c("GET /api/health is ok", health)

        def disclaimer():
            r = call("GET", "/api/disclaimer")
            status(r, 200)
            assert "simulated" in json.dumps(r.json()).lower(), "no 'simulated' in text"
        c("GET /api/disclaimer mentions simulated", disclaimer)

        def products():
            r = call("GET", "/api/products")
            status(r, 200)
            items = r.json()
            assert len(items) == 100, f"{len(items)} items, expected 100"
            units = [p["total_units"] for p in items]
            assert units == sorted(units, reverse=True), "not ordered by total_units desc"
            return items
        prods = c("GET /api/products: 100 items by total_units", products)
        pid = prods[0]["product_id"] if prods else None

        def forecast():
            assert pid, "no product id (products check failed)"
            r = call("GET", f"/api/forecast/{pid}")
            status(r, 200)
            body = r.json()
            hist = body["history"]
            assert len(hist) == 52, f"{len(hist)} history entries, expected 52"
            nulls = sum(1 for h in hist if h["units"] is None)
            assert nulls == 1, f"{nulls} null history entries, expected exactly 1"
            fc = body["next_week_forecast"]
            assert fc["week_start"] == FORECAST_WEEK, f"forecast week {fc['week_start']}"
            assert fc["forecast_p10"] <= fc["forecast_weekly"] <= fc["forecast_p90"], (
                f"p10/forecast/p90 = {fc['forecast_p10']}/{fc['forecast_weekly']}/{fc['forecast_p90']}"
            )
        c("GET /api/forecast/{first id}: 52 weeks, 1 null, p10<=fc<=p90", forecast)

        c("GET /api/forecast/NOPE is 404",
          lambda: status(call("GET", "/api/forecast/NOPE"), 404))

        def predict():
            assert pid, "no product id"
            r = call("POST", "/api/predict", json={"product_id": pid})
            status(r, 200)
            assert "forecast_weekly" in r.json(), "no forecast_weekly in response"
        c("POST /api/predict works", predict)

        def custom():
            r = call("POST", "/api/predict/custom",
                     json={"recent_units": [10, 12, 9, 14, 11, 13, 10, 12],
                           "target_week": FORECAST_WEEK})
            status(r, 200)
            assert "forecast_weekly" in r.json(), "no forecast_weekly in response"
        c("POST /api/predict/custom works", custom)

        c("POST /api/predict/custom rejects 7 values with 422",
          lambda: status(call("POST", "/api/predict/custom",
                              json={"recent_units": [10, 12, 9, 14, 11, 13, 10],
                                    "target_week": FORECAST_WEEK}), 422))

        def risk():
            r = call("POST", "/api/risk", json={"lead_time_weeks": 2})
            status(r, 200)
            s = r.json()["summary"]
            total = s["n_high"] + s["n_medium"] + s["n_low"]
            assert total == 100, f"high+medium+low = {total}, expected 100"
        c("POST /api/risk counts add up to 100", risk)

        c("POST /api/risk lead_time_weeks 9 is 422",
          lambda: status(call("POST", "/api/risk", json={"lead_time_weeks": 9}), 422))

        def simulate():
            r = call("POST", "/api/simulate", json={"lead_time_weeks": 2, "delay_weeks": 1})
            status(r, 200)
            assert "summary" in r.json(), "no summary in response"
        c("POST /api/simulate works", simulate)

        def sensitivity():
            r = call("POST", "/api/sensitivity", json={})
            status(r, 200)
            assert len(r.json()) == 12, f"{len(r.json())} rows, expected 12"
        c("POST /api/sensitivity returns 12 rows", sensitivity)

        def metrics():
            r = call("GET", "/api/metrics")
            status(r, 200)
            names = [str(m.get("model_name", "")).lower() for m in r.json()]
            assert any("baseline" in n or "naive" in n for n in names), f"models: {sorted(set(names))}"
        c("GET /api/metrics includes a baseline", metrics)

        def responds(path: str) -> Callable[[], None]:
            return lambda: status(call("GET", path), 200)
        c("GET /api/error-analysis responds", responds("/api/error-analysis"))
        c("GET /api/intelligence responds", responds("/api/intelligence"))

        # -- writes ----------------------------------------------------------
        order_body = {"product_id": pid or "", "quantity": 1, "supplier_note": "smoke test"}

        c("write without X-API-Key is 401",
          lambda: status(call("POST", "/api/orders", json=order_body), 401))

        def create_and_delete():
            assert self.api_key, "no API key (use --api-key, SHELFWATCH_API_KEY or .env)"
            r = call("POST", "/api/orders", key=True, json=order_body)
            assert r.status_code in (200, 201), f"create: HTTP {r.status_code}: {r.text[:150]}"
            order_id = r.json()["id"]
            d = call("DELETE", f"/api/orders/{order_id}", key=True)
            assert d.status_code in (200, 204), (
                f"delete order {order_id}: HTTP {d.status_code}: {d.text[:150]}"
            )
        c("write with X-API-Key succeeds (create then delete order)", create_and_delete)

        def logs():
            r = call("GET", "/api/logs", params={"limit": 200})
            status(r, 200)
            seen = {(row["method"], str(row["path"]).split("?")[0]) for row in r.json()}
            expected = [("GET", "/api/health"), ("GET", "/api/products"),
                        ("POST", "/api/risk"), ("POST", "/api/orders")]
            missing = [f"{m} {p}" for m, p in expected if (m, p) not in seen]
            assert not missing, f"not in the last 200 log rows: {', '.join(missing)}"
        c("GET /api/logs shows the requests just made", logs)

        def dashboard():
            r = call("GET", "/")
            status(r, 200)
            assert "html" in r.headers.get("content-type", "").lower(), (
                f"content-type {r.headers.get('content-type')!r}"
            )
            assert "Shelfwatch" in r.text, "'Shelfwatch' not found in page"
        c("GET / is HTML containing Shelfwatch", dashboard)

        def no_leak():
            assert not self.leaks, f"{FORBIDDEN} found in: {', '.join(self.leaks)}"
        c(f"no response contains {FORBIDDEN}", no_leak)

        passed = sum(1 for _, ok, _ in self.results if ok)
        print(f"\n{passed}/{len(self.results)} checks passed")
        return passed == len(self.results)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Shelfwatch end-to-end smoke test")
    parser.add_argument("base_url", nargs="?", default=os.getenv("SHELFWATCH_URL", DEFAULT_URL))
    parser.add_argument("--api-key", help="write key (else SHELFWATCH_API_KEY / API_KEY / .env)")
    parser.add_argument("--timeout", type=float, default=30.0, help="per-request seconds (default 30)")
    args = parser.parse_args(argv)

    print(f"Smoke test against {args.base_url}\n")
    smoke = Smoke(args.base_url, _load_key(args.api_key), args.timeout)
    try:
        try:
            smoke.http.get(smoke.base + "/api/health")
        except httpx.TransportError as exc:
            print(f"Cannot reach {smoke.base}: {type(exc).__name__}: {exc}")
            return 2
        ok = smoke.run()
    finally:
        smoke.http.close()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
