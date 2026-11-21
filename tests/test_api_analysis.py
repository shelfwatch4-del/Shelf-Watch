"""Tests for Task B20: analysis, explain, transactions endpoints and prediction_log on /api/predict.

The app is pointed (get_db override) at a private in-memory SQLite database loaded with
scripts.load_mysql, so every endpoint is proven to read from tables, not from CSV files.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from api import explain_service
from api.db import get_db
from api.main import app, data_store
from api.models import PredictionLog, Product, TransactionsTop100
from scripts.load_mysql import load_mysql
from src.features import build_features

ROOT = Path(__file__).resolve().parent.parent
TOL = 1e-6


@pytest.fixture(scope="module")
def sessionmaker_():
    engine = create_engine(
        "sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    # raw_path does not exist -> a handful of synthetic transaction lines, no big raw read
    load_mysql(engine_instance=engine, raw_path_str="data/raw/__absent__.parquet")
    return sessionmaker(bind=engine, autoflush=False)


@pytest.fixture(scope="module")
def client(sessionmaker_):
    def _db():
        s = sessionmaker_()
        try:
            yield s
        finally:
            s.close()

    with TestClient(app) as tc:
        app.dependency_overrides[get_db] = _db
        yield tc
        app.dependency_overrides.pop(get_db, None)


@pytest.fixture(scope="module")
def pid(sessionmaker_):
    with sessionmaker_() as s:
        return s.query(Product.product_id).order_by(Product.rank_train).first()[0]


def _has_source(block):
    return isinstance(block, str) and block.startswith("MySQL tables:")


# ---------------------------------------------------------------- analysis blocks


def test_lineage(client):
    body = client.get("/api/analysis/lineage").json()
    assert {d["role"] for d in body["datasets"]} == {"primary", "supporting", "calendar"}
    assert set(body["datasets"][0]) == {"name", "role", "rows_raw", "date_min", "date_max"}
    assert len(body["funnel"]) >= 8
    assert all({"step", "rows_remaining"} <= set(f) for f in body["funnel"])
    sp = body["split"]
    assert sp["train"]["last_week"] < sp["val"]["first_week"] <= sp["val"]["last_week"] < sp["test"]["first_week"]
    assert body["overlap_check"]["ok"] is True and body["overlap_check"]["shared_weeks"] == 0
    assert _has_source(body["source"])
    assert set(body["sources"]) == {"datasets", "funnel", "split", "overlap_check"}
    assert all(_has_source(v) for v in body["sources"].values())


def test_features(client):
    body = client.get("/api/analysis/features").json()
    assert body["model_name"] == data_store.config["model_name"]
    assert body["importance"] and {"feature", "group", "gain_importance", "mean_abs_contribution"} == set(body["importance"][0])
    assert {g["group"] for g in body["groups"]} == {r["group"] for r in body["importance"]}
    assert body["ablation"] and {"feature_groups", "model_name", "val_wape", "val_bias"} <= set(body["ablation"][0])
    assert _has_source(body["source"]) and all(_has_source(v) for v in body["sources"].values())
    other = client.get("/api/analysis/features?model_name=NOPE").json()
    assert other["importance"] == []


def test_tuning(client):
    body = client.get("/api/analysis/tuning").json()
    assert body["trials"] and {"candidate", "trial_no", "params", "mean_val_wape", "is_best"} == set(body["trials"][0])
    assert isinstance(body["trials"][0]["params"], dict)
    assert {c["candidate"] for c in body["candidates"]} == {t["candidate"] for t in body["trials"]}
    assert body["best_params"] and all(isinstance(v, dict) for v in body["best_params"].values())
    assert _has_source(body["source"]) and all(_has_source(v) for v in body["sources"].values())


def test_evaluation(client):
    body = client.get("/api/analysis/evaluation").json()
    assert {"metrics", "by_week", "by_segment", "by_pattern", "over_under", "coverage"} <= set(body)
    assert {m["split"] for m in body["metrics"]} == {"val", "test"}
    assert body["by_week"] and all(r["level"] == "week" for r in body["by_week"])
    assert body["by_pattern"] and all(r["level"] == "pattern_class" for r in body["by_pattern"])
    assert body["by_segment"]
    ou = body["over_under"][0]
    assert ou["over_forecast_units"] >= 0 and ou["under_forecast_units"] >= 0
    assert abs(ou["net_bias_units"] - (ou["over_forecast_units"] - ou["under_forecast_units"])) < 1e-6
    assert body["coverage"]
    assert set(body["sources"]) == {"metrics", "by_week", "by_segment", "by_pattern", "over_under", "coverage"}
    assert _has_source(body["source"]) and all(isinstance(v, str) and v for v in body["sources"].values())


def test_backtest(client, pid):
    body = client.get(f"/api/analysis/backtest?product_id={pid}").json()
    assert body["summary"] and {"policy", "lead_time_weeks", "abc_class", "fill_rate"} <= set(body["summary"][0])
    assert body["replay"] and all(r["product_id"] == pid for r in body["replay"])
    assert _has_source(body["source"]) and all(_has_source(v) for v in body["sources"].values())
    assert client.get("/api/analysis/backtest").json()["replay"] == []
    assert client.get("/api/analysis/backtest?product_id=NOPE").status_code == 404


# ---------------------------------------------------------------- explain


def test_explain_sum_identity_next_week(client, sessionmaker_):
    with sessionmaker_() as s:
        pids = [r[0] for r in s.query(Product.product_id).order_by(Product.rank_train).limit(8)]
    for p in pids:
        body = client.get(f"/api/explain/{p}").json()
        total = body["base_value"] + sum(f["contribution"] for f in body["features"])
        assert abs(total - body["model_output"]) <= TOL, p
        assert body["target_week"] == "2011-12-05"
        assert body["model_name"] == data_store.config["model_name"]
        assert body["prediction"] == max(body["model_output"], 0.0)


def test_explain_shape(client, pid):
    body = client.get(f"/api/explain/{pid}").json()
    assert {"product_id", "target_week", "model_name", "history_used", "features", "base_value",
            "model_output", "calendar", "prior_year", "interval", "caption", "scaling", "source"} <= set(body)
    assert len(body["history_used"]) == 12
    weeks = [h["week_start"] for h in body["history_used"]]
    assert weeks == sorted(weeks) and weeks[-1] < body["target_week"]
    assert (date.fromisoformat(body["target_week"]) - date.fromisoformat(weeks[0])).days == 12 * 7
    feature_list = json.loads((ROOT / "models" / "feature_list.json").read_text())
    assert [f["name"] for f in body["features"]] == feature_list
    assert all(set(f) == {"name", "group", "value", "contribution"} for f in body["features"])
    assert {"p10", "p90"} == set(body["interval"]) and body["interval"]["p10"] is not None
    assert _has_source(body["source"]) and "weekly_demand" in body["source"]
    # a scale-aware model must say how scaling is applied
    assert "standardised" in body["caption"] and "0.8" in body["caption"]
    assert len(body["prior_year"]) <= 5


def test_explain_history_week_matches_features_py(client, pid):
    target = "2011-10-10"
    body = client.get(f"/api/explain/{pid}?target_week={target}").json()
    assert abs(body["base_value"] + sum(f["contribution"] for f in body["features"]) - body["model_output"]) <= TOL

    proc = ROOT / "data" / "processed"
    weekly = pd.read_parquet(proc / "weekly_demand.parquet")
    expected = build_features(
        weekly,
        closed_weeks=pd.read_csv(proc / "closed_weeks.csv"),
        enrichment={
            "calendar": pd.read_csv(proc / "calendar_weekly.csv"),
            "prior_year": pd.read_parquet(proc / "prior_year_weekly.parquet"),
            "enrichment": pd.read_parquet(proc / "enrichment_weekly.parquet"),
        },
        groups=data_store.config["groups"],
    )
    row = expected[(expected["product_id"].astype(str) == pid) & (expected["week_start"] == pd.Timestamp(target))].iloc[0]
    for f in body["features"]:
        want = row[f["name"]]
        if pd.isna(want):
            assert f["value"] is None, f["name"]
        else:
            # weekly_demand.avg_price is DECIMAL(10,4) in the schema, so price-based features
            # carry up to 5e-5 of rounding that the parquet does not; 1e-4 relative covers it.
            assert f["value"] == pytest.approx(float(want), rel=1e-4, abs=1e-6), f["name"]


def test_explain_uses_only_weeks_before_target(client, pid):
    body = client.get(f"/api/explain/{pid}?target_week=2011-10-10").json()
    assert all(h["week_start"] < "2011-10-10" for h in body["history_used"])


def test_explain_errors(client, pid):
    assert client.get("/api/explain/NOPE").status_code == 404
    assert client.get(f"/api/explain/{pid}?target_week=2011-10-11").status_code == 422  # not a Monday
    assert client.get(f"/api/explain/{pid}?target_week=2011-12-12").status_code == 422  # beyond next week
    assert client.get(f"/api/explain/{pid}?target_week=2010-12-20").status_code == 422  # too little history
    assert client.get(f"/api/explain/{pid}?target_week=not-a-date").status_code == 422
    assert "detail" in client.get("/api/explain/NOPE").json()


def test_explain_scale_aware_caption_and_identity(sessionmaker_, pid):
    cfg = dict(data_store.config, model_name="ridge", target_mode="scale_aware", scale_col="roll_mean_8")
    with sessionmaker_() as s:
        out = explain_service.build_explanation(s, cfg, pid, date(2011, 10, 10))
    assert abs(out["base_value"] + sum(f["contribution"] for f in out["features"]) - out["model_output"]) <= TOL
    assert out["scaling"]["scale_col"] == "roll_mean_8" and out["scaling"]["scale_factor"] > 0
    assert "multiplied back by roll_mean_8" in out["caption"]
    raw_cfg = dict(cfg, target_mode="raw")
    with sessionmaker_() as s:
        raw = explain_service.build_explanation(s, raw_cfg, pid, date(2011, 10, 10))
    assert out["model_output"] == pytest.approx(raw["model_output"] * out["scaling"]["scale_factor"], rel=1e-9)


def test_explain_tree_model_identity(sessionmaker_, pid, monkeypatch):
    """LightGBM path: pred_contrib sums to the raw prediction."""
    import lightgbm as lgb

    cfg = dict(data_store.config, model_name="lightgbm", target_mode="raw")
    names = json.loads((ROOT / "models" / "feature_list.json").read_text())
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.random((200, len(names))) * 100, columns=names)
    y = X["lag_1"] * 0.5 + rng.random(200)
    lgbm = lgb.LGBMRegressor(n_estimators=20, min_child_samples=5, verbose=-1).fit(X, y)
    monkeypatch.setattr(explain_service, "_load_bundle", lambda _d: (lgbm, names))
    with sessionmaker_() as s:
        out = explain_service.build_explanation(s, cfg, pid, date(2011, 10, 10))
    assert abs(out["base_value"] + sum(f["contribution"] for f in out["features"]) - out["model_output"]) <= TOL
    assert "pred_contrib" in out["caption"]


# ---------------------------------------------------------------- transactions


def _add_lines(sessionmaker_, product_id, week, n):
    with sessionmaker_() as s:
        s.add_all(
            TransactionsTop100(
                invoice_no=f"INV{i:04d}",
                product_id=product_id,
                description="X",
                quantity=1 + i,
                quantity_capped=1 + i,
                invoice_date=pd.Timestamp(week) + pd.Timedelta(minutes=i),
                week_start=week,
                unit_price=1.25,
                country="United Kingdom",
                has_customer=1,
                is_cancelled=i % 2,
            )
            for i in range(n)
        )
        s.commit()


def test_transactions_cap_and_no_customer_fields(client, sessionmaker_, pid):
    week = date(2011, 3, 7)
    _add_lines(sessionmaker_, pid, week, 250)
    resp = client.get(f"/api/data/transactions?product_id={pid}&week_start={week}")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["lines"]) == 200
    assert body["total_lines"] == 250 and body["truncated"] is True and body["limit"] == 200
    expected_keys = {"invoice_no", "invoice_date", "quantity", "unit_price", "country", "is_cancelled"}
    assert all(set(line) == expected_keys for line in body["lines"])
    assert isinstance(body["lines"][1]["is_cancelled"], bool)
    assert [l["invoice_no"] for l in body["lines"]] == sorted(l["invoice_no"] for l in body["lines"])
    low = resp.text.lower()
    assert "customer" not in low.replace("has_customer", "")  # no customer field in the payload at all
    assert "has_customer" not in low
    assert _has_source(body["source"])


def test_transactions_below_cap_not_truncated(client, sessionmaker_, pid):
    week = date(2011, 3, 14)
    _add_lines(sessionmaker_, pid, week, 7)
    body = client.get(f"/api/data/transactions?product_id={pid}&week_start={week}").json()
    assert len(body["lines"]) == 7 and body["truncated"] is False


def test_transactions_table_has_no_customer_column(sessionmaker_):
    cols = [c["name"] for c in inspect(sessionmaker_.kw["bind"]).get_columns("transactions_top100")]
    assert not [c for c in cols if "customer" in c.lower() and c != "has_customer"]


def test_transactions_errors(client, pid):
    assert client.get("/api/data/transactions?product_id=NOPE&week_start=2011-03-07").status_code == 404
    assert client.get(f"/api/data/transactions?product_id={pid}").status_code == 422
    assert client.get(f"/api/data/transactions?product_id={pid}&week_start=garbage").status_code == 422
    assert client.get("/api/data/transactions?week_start=2011-03-07").status_code == 422
    empty = client.get(f"/api/data/transactions?product_id={pid}&week_start=2000-01-03").json()
    assert empty["lines"] == [] and empty["truncated"] is False


# ---------------------------------------------------------------- predictions


def test_predict_inserts_prediction_log_row(client, sessionmaker_, pid):
    with sessionmaker_() as s:
        before = s.query(PredictionLog).count()
    resp = client.post("/api/predict", json={"product_id": pid})
    assert resp.status_code == 200
    with sessionmaker_() as s:
        after = s.query(PredictionLog).count()
        row = s.query(PredictionLog).order_by(PredictionLog.id.desc()).first()
    assert after == before + 1
    assert row.endpoint == "/api/predict" and row.product_id == pid
    assert row.request_id == resp.headers["X-Request-ID"]
    assert row.outputs["forecast_weekly"] == pytest.approx(resp.json()["forecast_weekly"])

    hist = client.get("/api/predictions/history?limit=5").json()
    assert hist[0]["id"] == row.id and hist[0]["product_id"] == pid


def test_predict_failure_does_not_log(client, sessionmaker_):
    with sessionmaker_() as s:
        before = s.query(PredictionLog).count()
    assert client.post("/api/predict", json={"product_id": "NOPE"}).status_code == 404
    with sessionmaker_() as s:
        assert s.query(PredictionLog).count() == before


def test_json_safe(client, pid):
    for path in (
        "/api/analysis/lineage", "/api/analysis/features", "/api/analysis/tuning",
        "/api/analysis/evaluation", f"/api/analysis/backtest?product_id={pid}", f"/api/explain/{pid}",
    ):
        text = client.get(path).text
        assert "NaN" not in text and "Infinity" not in text, path
        json.loads(text)
        assert "customer_id" not in text.lower()
