"""Task B16: dashboard served from the same app, after all API routes."""

import logging

from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.main import app, mount_dashboard

PAGES = ["index", "predict", "risk", "inventory", "orders"]


def _make_dashboard(tmp_path):
    for p in PAGES:
        (tmp_path / f"{p}.html").write_text(f"<html>{p} page</html>")
    return tmp_path


def test_missing_dashboard_warns_and_skips(tmp_path, caplog):
    other = FastAPI()
    with caplog.at_level(logging.WARNING, logger="shelfwatch.api"):
        assert mount_dashboard(other, tmp_path / "nope") is False
    assert "not found" in caplog.text


def test_root_and_pages_served(tmp_path):
    other = FastAPI()
    assert mount_dashboard(other, _make_dashboard(tmp_path)) is True
    c = TestClient(other)
    r = c.get("/")
    assert r.status_code == 200 and "index page" in r.text
    for p in PAGES:
        r = c.get(f"/{p}.html")
        assert r.status_code == 200 and f"{p} page" in r.text


def test_mount_does_not_shadow_api_routes(tmp_path, monkeypatch):
    import api.main as main

    monkeypatch.setattr(main, "DASHBOARD_DIR", tmp_path)  # GET / now serves index.html from here
    saved = list(app.router.routes)
    app.router.routes[:] = [r for r in saved if getattr(r, "name", "") != "dashboard"]
    mount_dashboard(app, _make_dashboard(tmp_path))
    try:
        with TestClient(app) as c:
            assert c.get("/api/health").status_code == 200
            assert c.get("/docs").status_code == 200
            assert c.get("/openapi.json").status_code == 200
            r = c.post("/api/risk", json={})
            assert r.status_code != 404 and r.status_code != 405
            assert "index page" in c.get("/").text
    finally:
        app.router.routes[:] = saved
