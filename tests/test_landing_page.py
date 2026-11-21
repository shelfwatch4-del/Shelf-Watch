"""Task B16-min: GET / landing page (built-in fallback, or dashboard/index.html)."""

from fastapi.testclient import TestClient

import api.main as main
from api.main import app


def test_landing_page_html(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "DASHBOARD_DIR", tmp_path)  # no index.html -> fallback
    with TestClient(app) as c:
        r = c.get("/")
        disclaimer = c.get("/api/disclaimer").json()["text"]
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/html")
        assert "Shelfwatch" in r.text
        assert "Retail Product Demand Estimation and Stockout Risk Analysis" in r.text
        assert disclaimer in r.text
        for link in ("/docs", "/health", "/api/model-info", "/api/products"):
            assert f'href="{link}"' in r.text
        assert r.text.index('href="/docs"') < r.text.index('href="/health"')
        assert "Try the API" in r.text
        assert "prototype" not in r.text
        assert "<script" not in r.text and "http://" not in r.text and "https://" not in r.text


def test_landing_page_no_secrets(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "DASHBOARD_DIR", tmp_path)
    with TestClient(app) as c:
        body = c.get("/").text
    assert "CustomerID" not in body
    import os

    key = os.getenv("API_KEY")
    if key:
        assert key not in body


def test_landing_serves_dashboard_index_when_present(monkeypatch, tmp_path):
    (tmp_path / "index.html").write_text("<html>REAL FRONTEND</html>")
    monkeypatch.setattr(main, "DASHBOARD_DIR", tmp_path)
    with TestClient(app) as c:
        r = c.get("/")
    assert r.status_code == 200 and "REAL FRONTEND" in r.text


def test_landing_not_in_schema_and_api_still_works(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "DASHBOARD_DIR", tmp_path)
    with TestClient(app) as c:
        spec = c.get("/openapi.json")
        assert spec.status_code == 200 and "/" not in spec.json()["paths"]
        assert c.get("/docs").status_code == 200
        assert c.get("/health").status_code == 200
        r = c.post("/api/risk", json={})
        assert r.status_code not in (404, 405)
