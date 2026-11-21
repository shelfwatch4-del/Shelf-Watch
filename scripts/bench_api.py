"""scripts/bench_api.py - latency benchmark for the Shelfwatch API (Task B18, NFR-1).

Measures median and 95th percentile over N calls (default 50) for:
  POST /api/risk (cold: unique parameters, cache miss every call)
  POST /api/risk (warm: identical parameters, served from the cache)
  GET  /api/forecast/{id}
  GET  /api/products
Targets: /api/risk warm < 500 ms, /api/forecast < 300 ms.

    python scripts/bench_api.py                      # starts its own local server
    python scripts/bench_api.py --base-url http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import os
import socket
import statistics
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx

TARGETS_MS = {"POST /api/risk (warm)": 500.0, "GET /api/forecast/{id}": 300.0}


def p95(values: List[float]) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(0.95 * len(ordered))) - 1)]


def time_calls(fn: Callable[[int], httpx.Response], n: int) -> List[float]:
    out = []
    for i in range(n):
        t0 = time.perf_counter()
        r = fn(i)
        out.append((time.perf_counter() - t0) * 1000)
        r.raise_for_status()
    return out


def run_bench(client: httpx.Client, n: int = 50) -> Dict[str, List[float]]:
    pid = client.get("/api/products").json()[0]["product_id"]
    warm_body = {"lead_time_weeks": 2, "service_level": 0.95, "initial_stock_weeks": 3.0}
    client.post("/api/risk", json=warm_body).raise_for_status()  # prime the cache
    return {
        "POST /api/risk (cold)": time_calls(
            lambda i: client.post("/api/risk", json={**warm_body, "initial_stock_weeks": 0.5 + 0.1 * i}), n),
        "POST /api/risk (warm)": time_calls(lambda i: client.post("/api/risk", json=warm_body), n),
        "GET /api/forecast/{id}": time_calls(lambda i: client.get(f"/api/forecast/{pid}"), n),
        "GET /api/products": time_calls(lambda i: client.get("/api/products"), n),
    }


def report(results: Dict[str, List[float]]) -> bool:
    ok = True
    print(f"{'endpoint':<26}{'median ms':>11}{'p95 ms':>10}{'target':>10}  result")
    for name, vals in results.items():
        med, hi = statistics.median(vals), p95(vals)
        target = TARGETS_MS.get(name)
        verdict = ""
        if target is not None:
            passed = hi < target
            ok &= passed
            verdict = "PASS" if passed else "FAIL"
        print(f"{name:<26}{med:>11.1f}{hi:>10.1f}{(f'<{target:.0f}' if target else '-'):>10}  {verdict}")
    return ok


def start_local_server() -> str:
    os.environ.setdefault("RATE_LIMIT", "0")
    import uvicorn
    from api.main import app

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    while not server.started:
        time.sleep(0.1)
    return f"http://127.0.0.1:{port}"


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base-url", default=None, help="default: start a local server in-process")
    ap.add_argument("--calls", type=int, default=50)
    args = ap.parse_args(argv)
    base = args.base_url or start_local_server()
    with httpx.Client(base_url=base, timeout=30) as client:
        ok = report(run_bench(client, args.calls))
    print("\nall targets met" if ok else "\nTARGET MISSED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
