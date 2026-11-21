"""scripts/doctor.py - pre-flight check for the Shelfwatch deployment (SRS 11.4, 11.7).

Usage: venv\\Scripts\\python.exe scripts\\doctor.py [--base-url http://127.0.0.1:8000]
Exit code 0 when every check passes, 1 otherwise. The API key and DB password are never printed.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FAILS: list[str] = []

# Tables the system fills at run time; they may legitimately be empty.
RUNTIME_TABLES = {"replenishment_plans", "plan_lines", "purchase_orders", "prediction_log", "api_logs"}
MODEL_FILES = ["best_model.joblib", "q10.joblib", "q90.joblib", "config.json", "feature_list.json"]


def ok(msg: str) -> None:
    print(f"[ OK ] {msg}")


def fail(msg: str) -> None:
    FAILS.append(msg)
    print(f"[FAIL] {msg}")


def check_python() -> None:
    v = sys.version_info
    if (v.major, v.minor) == (3, 11):
        ok(f"Python {v.major}.{v.minor}.{v.micro}")
    else:
        fail(f"Python {v.major}.{v.minor}.{v.micro} found; the project is pinned to 3.11 "
             "(run with venv\\Scripts\\python.exe)")


def check_packages() -> None:
    bad = []
    n = 0
    for line in (ROOT / "requirements.txt").read_text().splitlines():
        m = re.match(r"^([A-Za-z0-9_.\-]+)==(\S+)$", line.strip())
        if not m:
            continue
        n += 1
        name, want = m.groups()
        try:
            have = metadata.version(name)
        except metadata.PackageNotFoundError:
            bad.append(f"{name} missing (need {want})")
            continue
        if have != want:
            bad.append(f"{name} {have} installed, need {want}")
    if bad:
        fail("Pinned packages differ: " + "; ".join(bad))
    else:
        ok(f"{n} pinned packages match requirements.txt")


def load_env() -> dict:
    from dotenv import dotenv_values
    path = ROOT / ".env"
    if not path.is_file():
        fail(".env not found (copy .env.example to .env)")
        return {}
    env = dotenv_values(path)
    ok(".env present")
    key = (env.get("API_KEY") or "").strip()
    if not key or key == "YOUR_API_KEY":
        fail("API_KEY is not set in .env (or still the placeholder)")
    else:
        ok(f"API_KEY set ({len(key)} characters)")
    if not env.get("DATABASE_URL"):
        fail("DATABASE_URL is not set in .env")
    elif not env["DATABASE_URL"].startswith("mysql"):
        fail("DATABASE_URL is not a MySQL URL")
    if env.get("TEST_DATABASE_URL"):
        print("[WARN] TEST_DATABASE_URL is set in .env; api/db.py prefers it over DATABASE_URL. "
              "scripts\\start_server.bat blanks it; do not start uvicorn any other way.")
    return env


def check_mysql(env: dict) -> None:
    url = env.get("DATABASE_URL")
    if not url or not url.startswith("mysql"):
        return
    from sqlalchemy import create_engine, text
    try:
        eng = create_engine(url, pool_pre_ping=True)
        conn = eng.connect()
    except Exception as exc:  # the message can contain the URL; print only the type
        fail(f"MySQL not reachable ({type(exc).__name__}). Is the MySQL80 service running "
             "and the .env login correct?")
        return
    ok("MySQL reachable")

    schema = (ROOT / "sql" / "schema.sql").read_text()
    tables = re.findall(r"CREATE TABLE IF NOT EXISTS\s+(\w+)", schema)
    missing, empty = [], []
    with conn:
        existing = {r[0] for r in conn.execute(text("SHOW TABLES"))}
        for t in tables:
            if t not in existing:
                missing.append(t)
                continue
            n = conn.execute(text(f"SELECT COUNT(*) FROM `{t}`")).scalar()
            if n == 0 and t not in RUNTIME_TABLES:
                empty.append(t)
        if missing:
            fail("Tables missing: " + ", ".join(missing) + " (run scripts\\load_mysql.py)")
        if empty:
            fail("Tables empty but should have rows: " + ", ".join(empty)
                 + " (run scripts\\load_mysql.py)")
        if not missing and not empty:
            ok(f"All {len(tables)} tables of sql/schema.sql exist; "
               f"{len(tables) - len(RUNTIME_TABLES)} data tables have rows")
        if "next_week_forecast" in existing:
            n = conn.execute(text("SELECT COUNT(DISTINCT product_id) FROM next_week_forecast")).scalar()
            if n == 100:
                ok("100 products have a next_week_forecast row")
            else:
                fail(f"{n} products have a next_week_forecast row; expected 100")


def check_models() -> None:
    mdir = ROOT / "models"
    miss = [f for f in MODEL_FILES if not (mdir / f).is_file()]
    if miss:
        fail("models/ files missing: " + ", ".join(miss))
    else:
        ok("models/ files present: " + ", ".join(MODEL_FILES))
    try:
        cfg = json.loads((mdir / "config.json").read_text())
        ok(f"Served model (models/config.json): {cfg['model_name']}")
    except Exception as exc:
        fail(f"models/config.json unreadable ({type(exc).__name__})")


def check_health(base: str) -> None:
    import httpx
    try:
        r = httpx.get(base.rstrip("/") + "/health", timeout=5,
                      headers={"ngrok-skip-browser-warning": "true"})
    except Exception as exc:
        fail(f"GET {base}/health unreachable ({type(exc).__name__}); start scripts\\start_server.bat")
        return
    try:
        body = r.json()
    except Exception:
        body = {}
    if r.status_code == 200 and body.get("status") == "ok":
        ok(f"GET {base}/health -> ok")
    else:
        fail(f"GET {base}/health -> HTTP {r.status_code}, status={body.get('status')!r}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    args = ap.parse_args()
    check_python()
    check_packages()
    env = load_env()
    check_mysql(env)
    check_models()
    check_health(args.base_url)
    print()
    if FAILS:
        print(f"{len(FAILS)} check(s) FAILED.")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
