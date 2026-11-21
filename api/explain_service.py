"""api/explain_service.py - Per-product explanation for a target week (Task B20, FR-M-17).

The feature row is built by src.features.build_future_features from MySQL tables
(weekly_demand, closed_weeks, calendar_weekly, prior_year_weekly, enrichment_weekly),
using only weeks strictly before the target week. Contributions come from
src.explain.explain_row, whose identity base_value + sum(contributions) == model_output
(within 1e-6) is asserted there and again here.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import pandas as pd
from sqlalchemy.orm import Session

from api.crud_analysis import UnknownProduct, require_product, source_caption
from api.models import (
    CalendarWeekly,
    ClosedWeek,
    EnrichmentWeekly,
    NextWeekForecast,
    Prediction,
    PriorYearWeekly,
    WeeklyDemand,
)
from src.features import FEATURE_GROUPS, _resolve_selected_feature_cols, build_future_features

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
HISTORY_WEEKS = 12
MIN_ACTIVE_HISTORY = 8  # longest rolling window in src.features
IDENTITY_TOL = 1e-6


class ExplainError(Exception):
    """Bad target week or not enough history; the route turns it into a 422."""


@lru_cache(maxsize=2)
def _load_bundle(models_dir: str) -> Tuple[Any, Optional[List[str]]]:
    root = Path(models_dir)
    model = joblib.load(root / "best_model.joblib")
    features: Optional[List[str]] = None
    fl = root / "feature_list.json"
    if fl.exists():
        features = [str(f) for f in json.loads(fl.read_text(encoding="utf-8"))]
    return model, features


def _group_of(feature: str) -> str:
    for g, cols in FEATURE_GROUPS.items():
        if feature in cols:
            return g
    return "base"


def _frame(rows: List[Any], cols: List[str]) -> pd.DataFrame:
    return pd.DataFrame([{c: getattr(r, c) for c in cols} for r in rows], columns=cols)


def _scaling_caption(model: Any, model_name: str, target_mode: str, scale_col: Optional[str], scale: float) -> Tuple[str, Dict[str, Any]]:
    from sklearn.pipeline import Pipeline

    info: Dict[str, Any] = {"target_mode": target_mode, "scale_col": None, "scale_factor": 1.0, "inputs": "none"}
    parts: List[str] = []
    if isinstance(model, Pipeline) and "scaler" in model.named_steps:
        info["inputs"] = "median imputation then standardisation (x - mean) / std, fitted on the training weeks"
        parts.append(
            "Ridge is scale-aware in its inputs: each feature is median-imputed and standardised "
            "((x - mean) / std, fitted on the training weeks) before its coefficient is applied. "
            "Contribution = coefficient x standardised value; base value = the intercept, i.e. the "
            "prediction for an average product-week. The 'value' column shows the raw, unscaled feature."
        )
    else:
        parts.append(
            "Contributions are exact tree attributions (LightGBM pred_contrib); trees need no feature scaling."
        )
    if model_name == "ensemble":
        parts.append(
            "The served output is 0.8 x Ridge + 0.2 x roll_mean_4, so every Ridge contribution and the base "
            "value are multiplied by 0.8 and roll_mean_4 also receives 0.2 x its raw value."
        )
    if target_mode == "scale_aware" and scale_col:
        info.update(scale_col=scale_col, scale_factor=scale)
        parts.append(
            f"The model predicts demand divided by {scale_col}; base value and contributions are multiplied "
            f"back by {scale_col} = {scale:.4f} so they are in units per week."
        )
    return " ".join(parts), info


def build_explanation(
    db: Session,
    config: Dict[str, Any],
    product_id: str,
    target_week: Optional[date] = None,
    models_dir: Path = MODELS_DIR,
) -> Dict[str, Any]:
    from src.explain import explain_row  # lazy: imports shap

    pid = require_product(db, product_id)

    forecast = db.query(NextWeekForecast).filter(NextWeekForecast.product_id == pid).first()
    last_week = db.query(WeeklyDemand.week_start).filter(WeeklyDemand.product_id == pid).order_by(
        WeeklyDemand.week_start.desc()
    ).limit(1).scalar()
    if last_week is None:
        raise ExplainError(f"No weekly_demand rows for product {pid}.")
    next_week = forecast.week_start if forecast is not None else last_week + timedelta(weeks=1)
    target = target_week or next_week
    if target.weekday() != 0:
        raise ExplainError("target_week must be a Monday (week_start).")
    if target > next_week:
        raise ExplainError(f"target_week is beyond the next forecast week {next_week.isoformat()}.")

    weekly = _frame(
        db.query(WeeklyDemand).filter(WeeklyDemand.product_id == pid, WeeklyDemand.week_start < target).all(),
        ["product_id", "week_start", "units", "orders", "customers", "avg_price"],
    )
    weekly["avg_price"] = weekly["avg_price"].astype(float)
    closed = [r[0] for r in db.query(ClosedWeek.week_start).all()]
    active_hist = int((~pd.to_datetime(weekly["week_start"]).isin(pd.to_datetime(closed))).sum()) if len(weekly) else 0
    if active_hist < MIN_ACTIVE_HISTORY + 1:
        raise ExplainError(
            f"Not enough history before {target.isoformat()}: need {MIN_ACTIVE_HISTORY} open weeks, have {active_hist}."
        )

    groups = list(config.get("groups") or ["base"])
    enrichment = {
        "calendar": _frame(
            db.query(CalendarWeekly).all(),
            ["week_start", "uk_bank_holiday_count", "trading_days_expected", "is_christmas_week"],
        ),
        "prior_year": _frame(
            db.query(PriorYearWeekly).filter(PriorYearWeekly.product_id == pid).all(),
            ["product_id", "week_start", "units", "orders"],
        ),
        "enrichment": _frame(
            db.query(EnrichmentWeekly).filter(EnrichmentWeekly.product_id == pid).all(),
            ["product_id", "week_start", "gross_units", "cancelled_units", "export_units", "guest_units"],
        ),
    }
    feats = build_future_features(
        weekly, closed_weeks=closed, target_week=pd.Timestamp(target), enrichment=enrichment, groups=groups
    )
    feats = feats[feats["product_id"].astype(str) == pid]
    if feats.empty:
        raise ExplainError("Could not build a feature row for this product and week.")
    row = feats.iloc[0]

    model, listed = _load_bundle(str(models_dir))
    cols = listed or _resolve_selected_feature_cols(groups)
    missing = [c for c in cols if c not in row.index]
    if missing:
        raise ExplainError(f"Feature columns missing from the built row: {missing}")

    model_name = str(config.get("model_name", "ensemble"))
    target_mode = str(config.get("target_mode", "raw"))
    is_ensemble = model_name == "ensemble"
    scale_col = config.get("scale_col") or ("roll_mean_8" if target_mode == "scale_aware" else None)
    scale = 1.0
    if target_mode == "scale_aware" and scale_col and not is_ensemble:
        scale = float(row[scale_col])

    exp = explain_row(model, row[cols], feature_names=cols, is_ensemble=is_ensemble)
    base_value = exp["base_value"] * scale
    contributions = {k: v * scale for k, v in exp["contributions"].items()}
    model_output = exp["model_output"] * scale
    total = base_value + sum(contributions.values())
    if abs(total - model_output) > IDENTITY_TOL:
        raise AssertionError(f"explain identity failed: {total} vs {model_output}")

    caption, scaling = _scaling_caption(model, model_name, target_mode, scale_col, scale)

    # history_used: the 12 calendar weeks before the target (closed weeks have no units)
    hist_start = target - timedelta(weeks=HISTORY_WEEKS)
    by_week = {
        r.week_start: r.units
        for r in db.query(WeeklyDemand).filter(
            WeeklyDemand.product_id == pid, WeeklyDemand.week_start >= hist_start, WeeklyDemand.week_start < target
        )
    }
    closed_set = set(closed)
    history_used = []
    for i in range(HISTORY_WEEKS):
        w = hist_start + timedelta(weeks=i)
        units = None if w in closed_set else by_week.get(w)
        history_used.append({"week_start": w.isoformat(), "units": None if units is None else float(units)})

    cal = db.query(CalendarWeekly).filter(CalendarWeekly.week_start == target).first()
    calendar = {
        "week_start": target.isoformat(),
        "uk_bank_holiday_count": None if cal is None else int(cal.uk_bank_holiday_count),
        "trading_days_expected": None if cal is None else int(cal.trading_days_expected),
        "is_christmas_week": None if cal is None else bool(cal.is_christmas_week),
    }

    py_lo, py_hi = target - timedelta(weeks=54), target - timedelta(weeks=50)
    prior_year = [
        {"week_start": r.week_start.isoformat(), "units": r.units, "orders": r.orders}
        for r in db.query(PriorYearWeekly)
        .filter(PriorYearWeekly.product_id == pid, PriorYearWeekly.week_start >= py_lo, PriorYearWeekly.week_start <= py_hi)
        .order_by(PriorYearWeekly.week_start)
    ]

    interval: Dict[str, Optional[float]] = {"p10": None, "p90": None}
    interval_table = "none"
    if forecast is not None and forecast.week_start == target:
        interval = {"p10": forecast.forecast_p10, "p90": forecast.forecast_p90}
        interval_table = "next_week_forecast"
    else:
        pred = (
            db.query(Prediction)
            .filter(Prediction.product_id == pid, Prediction.week_start == target, Prediction.model_name == model_name)
            .first()
        )
        if pred is not None:
            interval = {"p10": pred.forecast_p10, "p90": pred.forecast_p90}
            interval_table = "predictions"

    tables = ["products", "weekly_demand", "closed_weeks", "calendar_weekly", "prior_year_weekly", "next_week_forecast"]
    if "g4" in groups:
        tables.append("enrichment_weekly")
    if interval_table == "predictions":
        tables.append("predictions")

    return {
        "product_id": pid,
        "target_week": target.isoformat(),
        "model_name": model_name,
        "history_used": history_used,
        "features": [
            {
                "name": c,
                "group": _group_of(c),
                "value": None if pd.isna(row[c]) else float(row[c]),
                "contribution": contributions[c],
            }
            for c in cols
        ],
        "base_value": base_value,
        "model_output": model_output,
        "prediction": max(model_output, 0.0),
        "calendar": calendar,
        "prior_year": prior_year,
        "interval": interval,
        "scaling": scaling,
        "caption": caption,
        "source": source_caption(*tables),
    }


__all__ = ["ExplainError", "UnknownProduct", "build_explanation"]
