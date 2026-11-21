"""api/crud_analysis.py - Read-only analysis queries (Task B20, FR-U-20, FR-M-17).

Every function reads MySQL tables through a SQLAlchemy session and never touches a CSV.
Every block carries a `source` caption naming the tables it came from.
"""

from __future__ import annotations

import json
from datetime import date, datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import case, func
from sqlalchemy.orm import Session

from api.models import (
    Ablation,
    DataSource,
    ErrorAnalysis,
    FeatureImportance,
    Metric,
    PipelineLog,
    PolicyBacktest,
    PolicyBacktestWeekly,
    Prediction,
    Product,
    PriorYearWeekly,
    TransactionsTop100,
    TuningTrial,
    WeeklyDemand,
    ClosedWeek,
)

MAX_TRANSACTION_LINES = 200


class UnknownProduct(Exception):
    """Raised when a product_id is not in the products table."""


def source_caption(*tables: str) -> str:
    return "MySQL tables: " + ", ".join(tables)


def _iso(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    return str(value)


def _as_obj(value: Any) -> Any:
    """JSON columns can come back as text (pandas to_sql loads them as strings)."""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            return value
    return value


def _f(value: Any) -> Optional[float]:
    return float(value) if value is not None else None


def require_product(db: Session, product_id: str) -> str:
    pid = str(product_id).strip()
    if db.query(Product.product_id).filter(Product.product_id == pid).first() is None:
        raise UnknownProduct(pid)
    return pid


# ---------------------------------------------------------------- lineage


def get_lineage(db: Session) -> Dict[str, Any]:
    sources = db.query(DataSource).order_by(DataSource.id).all()
    datasets = [
        {
            "name": s.name,
            "role": s.role,
            "rows_raw": s.rows_raw,
            "date_min": _iso(s.date_min),
            "date_max": _iso(s.date_max),
        }
        for s in sources
    ]
    funnel = [
        {"step": r.step, "rows_remaining": r.rows_remaining, "note": r.note}
        for r in db.query(PipelineLog).order_by(PipelineLog.step_no).all()
    ]

    # Chronological split, derived from the weeks that carry val/test predictions.
    first_week, last_week, n_weeks = db.query(
        func.min(WeeklyDemand.week_start),
        func.max(WeeklyDemand.week_start),
        func.count(func.distinct(WeeklyDemand.week_start)),
    ).one()
    closed = [_iso(r[0]) for r in db.query(ClosedWeek.week_start).order_by(ClosedWeek.week_start).all()]
    bounds = {
        name: (lo, hi, n)
        for name, lo, hi, n in db.query(
            Prediction.split_name,
            func.min(Prediction.week_start),
            func.max(Prediction.week_start),
            func.count(func.distinct(Prediction.week_start)),
        ).group_by(Prediction.split_name)
    }
    split: Dict[str, Any] = {
        "rule": "chronological, no shuffling: train < val < test",
        "first_week": _iso(first_week),
        "last_week": _iso(last_week),
        "n_weeks": n_weeks,
        "closed_weeks": closed,
        "source": source_caption("weekly_demand", "closed_weeks", "predictions"),
    }
    held_out_starts = [lo for lo, _, _ in bounds.values() if lo is not None]
    train_end = None
    if held_out_starts and first_week is not None:
        first_held_out = min(held_out_starts)
        n_train = (
            db.query(func.count(func.distinct(WeeklyDemand.week_start)))
            .filter(WeeklyDemand.week_start < first_held_out)
            .scalar()
        )
        train_last = (
            db.query(func.max(WeeklyDemand.week_start)).filter(WeeklyDemand.week_start < first_held_out).scalar()
        )
        train_end = train_last
        split["train"] = {"first_week": _iso(first_week), "last_week": _iso(train_last), "n_weeks": n_train}
    for name in ("val", "test"):
        if name in bounds:
            lo, hi, n = bounds[name]
            split[name] = {"first_week": _iso(lo), "last_week": _iso(hi), "n_weeks": n}
    split["train_end"] = _iso(train_end)

    return {
        "datasets": datasets,
        "funnel": funnel,
        "split": split,
        "overlap_check": _overlap_check(db, sources),
        "source": source_caption("data_sources", "pipeline_log", "weekly_demand", "predictions", "prior_year_weekly"),
        "sources": {
            "datasets": source_caption("data_sources"),
            "funnel": source_caption("pipeline_log"),
            "split": split["source"],
            "overlap_check": source_caption("data_sources", "weekly_demand", "prior_year_weekly"),
        },
    }


def _overlap_check(db: Session, sources: List[DataSource]) -> Dict[str, Any]:
    """Do the two retail datasets overlap, and does prior-year data leak into the demand weeks?"""
    primary = next((s for s in sources if s.role == "primary"), None)
    supporting = next((s for s in sources if s.role == "supporting"), None)
    out: Dict[str, Any] = {
        "raw_overlap_from": None,
        "raw_overlap_to": None,
        "raw_overlap_days": 0,
        "prior_year_weeks": 0,
        "demand_weeks": 0,
        "shared_weeks": 0,
        "ok": True,
    }
    if primary and supporting and primary.date_min and supporting.date_max:
        lo = max(primary.date_min, supporting.date_min or primary.date_min)
        hi = min(primary.date_max or supporting.date_max, supporting.date_max)
        if lo <= hi:
            out.update(raw_overlap_from=_iso(lo), raw_overlap_to=_iso(hi), raw_overlap_days=(hi - lo).days + 1)
    py_weeks = {r[0] for r in db.query(PriorYearWeekly.week_start).distinct()}
    wd_weeks = {r[0] for r in db.query(WeeklyDemand.week_start).distinct()}
    shared = py_weeks & wd_weeks
    out.update(
        prior_year_weeks=len(py_weeks),
        demand_weeks=len(wd_weeks),
        shared_weeks=len(shared),
        # The raw files share a few days at the year boundary; what matters is that no
        # prior-year week is also a demand week, so a feature never reads the target year.
        ok=len(shared) == 0,
        note="ok = no week_start appears in both prior_year_weekly and weekly_demand",
    )
    return out


# ---------------------------------------------------------------- features


def get_features(db: Session, served_model: Optional[str] = None, model_name: Optional[str] = None) -> Dict[str, Any]:
    available = [r[0] for r in db.query(FeatureImportance.model_name).distinct().order_by(FeatureImportance.model_name)]
    chosen = model_name or (served_model if served_model in available else (available[0] if available else None))
    rows = []
    if chosen is not None:
        rows = (
            db.query(FeatureImportance)
            .filter(FeatureImportance.model_name == chosen)
            .order_by(FeatureImportance.gain_importance.desc(), FeatureImportance.feature)
            .all()
        )
    importance = [
        {
            "feature": r.feature,
            "group": r.feature_group,
            "gain_importance": r.gain_importance,
            "mean_abs_contribution": r.mean_abs_contribution,
        }
        for r in rows
    ]
    groups: Dict[str, Dict[str, Any]] = {}
    for r in importance:
        g = groups.setdefault(r["group"] or "base", {"group": r["group"] or "base", "features": [], "gain_importance": 0.0, "mean_abs_contribution": 0.0})
        g["features"].append(r["feature"])
        g["gain_importance"] += r["gain_importance"] or 0.0
        g["mean_abs_contribution"] += r["mean_abs_contribution"] or 0.0
    ablation = [
        {
            "feature_groups": a.feature_groups,
            "model_name": a.model_name,
            "val_wape": a.val_wape,
            "val_bias": a.val_bias,
            "note": a.note,
        }
        for a in db.query(Ablation).order_by(Ablation.model_name, Ablation.feature_groups).all()
    ]
    return {
        "model_name": chosen,
        "available_models": available,
        "groups": list(groups.values()),
        "importance": importance,
        "ablation": ablation,
        "source": source_caption("feature_importance", "ablation"),
        "sources": {
            "groups": source_caption("feature_importance"),
            "importance": source_caption("feature_importance"),
            "ablation": source_caption("ablation"),
        },
    }


# ---------------------------------------------------------------- tuning


def get_tuning(db: Session) -> Dict[str, Any]:
    rows = db.query(TuningTrial).order_by(TuningTrial.candidate, TuningTrial.trial_no).all()
    trials = [
        {
            "candidate": r.candidate,
            "trial_no": r.trial_no,
            "params": _as_obj(r.params),
            "mean_val_wape": r.mean_val_wape,
            "is_best": bool(r.is_best),
        }
        for r in rows
    ]
    candidates: Dict[str, Dict[str, Any]] = {}
    best_params: Dict[str, Any] = {}
    for t in trials:
        c = candidates.setdefault(t["candidate"], {"candidate": t["candidate"], "n_trials": 0, "best_mean_val_wape": None})
        c["n_trials"] += 1
        w = t["mean_val_wape"]
        if w is not None and (c["best_mean_val_wape"] is None or w < c["best_mean_val_wape"]):
            c["best_mean_val_wape"] = w
        if t["is_best"]:
            best_params[t["candidate"]] = t["params"]
    return {
        "candidates": list(candidates.values()),
        "trials": trials,
        "best_params": best_params,
        "source": source_caption("tuning_trials"),
        "sources": {k: source_caption("tuning_trials") for k in ("candidates", "trials", "best_params")},
    }


# ---------------------------------------------------------------- evaluation


def _error_rows(db: Session, levels: List[str]) -> List[Dict[str, Any]]:
    rows = (
        db.query(ErrorAnalysis)
        .filter(ErrorAnalysis.level_name.in_(levels))
        .order_by(ErrorAnalysis.level_name, ErrorAnalysis.key_name, ErrorAnalysis.model_name)
        .all()
    )
    return [
        {
            "level": r.level_name,
            "key": r.key_name,
            "model_name": r.model_name,
            "n": r.n,
            "MAE": r.mae,
            "RMSE": r.rmse,
            "WAPE": r.wape,
            "bias": r.bias,
            "over_forecast_units": r.over_forecast_units,
            "under_forecast_units": r.under_forecast_units,
        }
        for r in rows
    ]


def get_evaluation(db: Session, fallback_coverage: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    metrics = [
        {"model_name": m.model_name, "split": m.split_name, "MAE": m.mae, "RMSE": m.rmse, "WAPE": m.wape, "bias": m.bias}
        for m in db.query(Metric).order_by(Metric.split_name, Metric.model_name).all()
    ]

    over = func.sum(case((Prediction.predicted > Prediction.actual, Prediction.predicted - Prediction.actual), else_=0.0))
    under = func.sum(case((Prediction.actual > Prediction.predicted, Prediction.actual - Prediction.predicted), else_=0.0))
    over_under = [
        {
            "model_name": name,
            "split": split_name,
            "n": n,
            "over_forecast_units": _f(o),
            "under_forecast_units": _f(u),
            "net_bias_units": _f((o or 0.0) - (u or 0.0)),
        }
        for name, split_name, n, o, u in db.query(
            Prediction.model_name, Prediction.split_name, func.count(), over, under
        )
        .group_by(Prediction.model_name, Prediction.split_name)
        .order_by(Prediction.split_name, Prediction.model_name)
        .all()
    ]

    inside = case(
        ((Prediction.actual >= Prediction.forecast_p10) & (Prediction.actual <= Prediction.forecast_p90), 1.0),
        else_=0.0,
    )
    coverage = [
        {
            "model_name": name,
            "split": split_name,
            "n": n,
            "coverage": _f(cov),
            "mean_width": _f(width),
            "target_coverage": 0.8,
        }
        for name, split_name, n, cov, width in db.query(
            Prediction.model_name,
            Prediction.split_name,
            func.count(),
            func.avg(inside),
            func.avg(Prediction.forecast_p90 - Prediction.forecast_p10),
        )
        .filter(Prediction.forecast_p10.isnot(None), Prediction.forecast_p90.isnot(None))
        .group_by(Prediction.model_name, Prediction.split_name)
        .order_by(Prediction.split_name, Prediction.model_name)
        .all()
    ]
    coverage_source = source_caption("predictions (rows with forecast_p10 and forecast_p90)")
    if not coverage and fallback_coverage:
        # predictions carries no interval columns for these rows; use the calibration
        # result the API loaded at start-up and say so in the caption.
        coverage = [
            {"model_name": None, "split": k, "n": None, "coverage": fallback_coverage[k], "mean_width": None, "target_coverage": 0.8}
            for k in ("val", "test")
            if isinstance(fallback_coverage.get(k), (int, float))
        ]
        coverage_source = "models/interval_coverage.json loaded at start-up (predictions table has no interval columns)"

    sources = {
        "metrics": source_caption("metrics"),
        "by_week": source_caption("error_analysis"),
        "by_segment": source_caption("error_analysis"),
        "by_pattern": source_caption("error_analysis"),
        "over_under": source_caption("predictions"),
        "coverage": coverage_source,
    }
    return {
        "metrics": metrics,
        "by_week": _error_rows(db, ["week"]),
        "by_segment": _error_rows(db, ["segment", "abc_class"]),
        "by_pattern": _error_rows(db, ["pattern_class"]),
        "over_under": over_under,
        "coverage": coverage,
        "source": source_caption("metrics", "error_analysis", "predictions"),
        "sources": sources,
    }


# ---------------------------------------------------------------- backtest


def get_backtest(db: Session, product_id: Optional[str] = None) -> Dict[str, Any]:
    pid = require_product(db, product_id) if product_id else None
    summary = [
        {
            "policy": r.policy,
            "lead_time_weeks": int(r.lead_time_weeks),
            "abc_class": r.abc_class,
            "fill_rate": r.fill_rate,
            "stockout_weeks": r.stockout_weeks,
            "avg_on_hand": r.avg_on_hand,
            "unmet_units": r.unmet_units,
        }
        for r in db.query(PolicyBacktest).order_by(PolicyBacktest.policy, PolicyBacktest.lead_time_weeks, PolicyBacktest.abc_class).all()
    ]
    replay: List[Dict[str, Any]] = []
    if pid:
        replay = [
            {
                "policy": r.policy,
                "lead_time_weeks": int(r.lead_time_weeks),
                "product_id": r.product_id,
                "week_start": _iso(r.week_start),
                "on_hand_start": r.on_hand_start,
                "order_qty": r.order_qty,
                "arrived_qty": r.arrived_qty,
                "demand": r.demand,
                "sold": r.sold,
                "unmet": r.unmet,
                "on_hand_end": r.on_hand_end,
            }
            for r in db.query(PolicyBacktestWeekly)
            .filter(PolicyBacktestWeekly.product_id == pid)
            .order_by(PolicyBacktestWeekly.policy, PolicyBacktestWeekly.lead_time_weeks, PolicyBacktestWeekly.week_start)
            .all()
        ]
    return {
        "product_id": pid,
        "summary": summary,
        "replay": replay,
        "source": source_caption("policy_backtest", "policy_backtest_weekly"),
        "sources": {
            "summary": source_caption("policy_backtest"),
            "replay": source_caption("policy_backtest_weekly"),
        },
    }


# ---------------------------------------------------------------- transactions


def get_transactions(db: Session, product_id: str, week_start: date) -> Dict[str, Any]:
    """Invoice lines for one product-week. Selects named columns only, so no customer
    field can ever leak (the table itself holds has_customer, never a customer id)."""
    pid = require_product(db, product_id)
    base = db.query(TransactionsTop100).filter(
        TransactionsTop100.product_id == pid, TransactionsTop100.week_start == week_start
    )
    total = base.count()
    rows = (
        db.query(
            TransactionsTop100.invoice_no,
            TransactionsTop100.invoice_date,
            TransactionsTop100.quantity,
            TransactionsTop100.unit_price,
            TransactionsTop100.country,
            TransactionsTop100.is_cancelled,
        )
        .filter(TransactionsTop100.product_id == pid, TransactionsTop100.week_start == week_start)
        .order_by(TransactionsTop100.invoice_date, TransactionsTop100.line_id)
        .limit(MAX_TRANSACTION_LINES)
        .all()
    )
    lines = [
        {
            "invoice_no": r.invoice_no,
            "invoice_date": _iso(r.invoice_date),
            "quantity": r.quantity,
            "unit_price": _f(r.unit_price),
            "country": r.country,
            "is_cancelled": bool(r.is_cancelled),
        }
        for r in rows
    ]
    return {
        "product_id": pid,
        "week_start": week_start.isoformat(),
        "lines": lines,
        "total_lines": total,
        "limit": MAX_TRANSACTION_LINES,
        "truncated": total > len(lines),
        "source": source_caption("transactions_top100"),
    }

