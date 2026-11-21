"""api/main.py - Shelfwatch FastAPI application skeleton and endpoints."""

from __future__ import annotations

import json
import math
import os
import threading
from contextlib import asynccontextmanager
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import inspect, text
from sqlalchemy.orm import Session

from api.crud_inventory import NotFound, ValidationError, delete_stock, list_stock, upsert_stock
from api.crud_orders import (
    ConflictError,
    create_order,
    delete_order,
    get_order,
    list_orders,
    open_order_quantities,
    update_order,
)
from api.db import SessionLocal, get_db, init_db, seed_demo_data
from api.models import InventoryLevel, PredictionLog, PurchaseOrder, ReplenishmentPlan
from api.schemas import (
    AblationEntry,
    DisclaimerResponse,
    ForecastHistoryItem,
    ForecastResponse,
    HealthDatabaseResponse,
    HealthResponse,
    IntervalCoverage,
    InventoryCreateUpdateRequest,
    InventoryResponse,
    KaggleUrls,
    ModelInfoResponse,
    NextWeekForecastItem,
    OrderCreateRequest,
    OrderResponse,
    OrderUpdateRequest,
    PolicyBacktestEntry,
    PredictCustomRequest,
    PredictCustomResponse,
    PredictRequest,
    PredictResponse,
    ProductListItem,
    RiskRequest,
    RiskResponse,
    SensitivityRequest,
    SensitivityRow,
    SimulateRequest,
    SimulateResponse,
    TestPredictionItem,
)
from src.predict import predict_custom
from src.risk_engine import DEFAULTS, compute_risk, sensitivity_table, summarize

# -----------------------------------------------------------------------------
# Required and Optional Data Files Mapping
# -----------------------------------------------------------------------------

REQUIRED_DATA_FILES: Dict[str, str] = {
    "data/processed/weekly_demand.parquet": "python -m src.data_prep",
    "data/processed/next_week_forecast.csv": "python -m src.train",
    "data/processed/metrics.csv": "python -m src.train",
    "data/processed/predictions.csv": "python -m src.train",
    "data/processed/error_analysis.csv": "python -m src.error_analysis",
    "data/processed/product_intelligence.csv": "python -m src.intelligence",
    "data/processed/basket_pairs.csv": "python -m src.intelligence",
    "models/config.json": "python -m src.train",
    "models/ablation.json": "python -m src.ablation",
    "models/interval_coverage.json": "python -m src.quantiles",
}

OPTIONAL_DATA_FILES: Dict[str, str] = {
    "data/processed/policy_backtest.csv": "python -m src.backtest",
}


# -----------------------------------------------------------------------------
# Clean Helper Sanitizer
# -----------------------------------------------------------------------------

def clean(obj: Any) -> Any:
    """
    Recursively sanitize objects for JSON serialization:
    - Replaces NaN, inf, -inf with None (rendered as null in JSON).
    - Converts numpy scalars (int64, float64, bool_) and numpy arrays to native Python types.
    - Converts datetime, date, and pd.Timestamp to ISO 8601 strings.
    - Completely strips any keys containing 'customerid' or 'customer_id' (case-insensitive)
      to guarantee CustomerID is never exposed (SRS FR-A-7).
    """
    if obj is None:
        return None

    # Handle floats / numpy floats / math nan and inf
    if isinstance(obj, (float, np.floating)):
        if math.isnan(obj) or np.isnan(obj) or math.isinf(obj) or np.isinf(obj):
            return None
        return float(obj)

    # Handle numpy integer and boolean scalars
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)

    # Handle dates and timestamps
    if isinstance(obj, (datetime, date, pd.Timestamp)):
        return obj.isoformat()

    # Handle numpy arrays
    if isinstance(obj, np.ndarray):
        return [clean(item) for item in obj.tolist()]

    # Handle pandas DataFrame / Series
    if isinstance(obj, pd.DataFrame):
        return [clean(row) for row in obj.to_dict(orient="records")]
    if isinstance(obj, pd.Series):
        return clean(obj.to_dict())

    # Handle dictionaries (strip CustomerID / customer_id)
    if isinstance(obj, (dict, Mapping)):
        cleaned_dict: Dict[str, Any] = {}
        for k, v in obj.items():
            key_str = str(k)
            normalized_key = key_str.lower().replace("_", "").replace("-", "")
            if normalized_key == "customerid":
                continue  # Never expose CustomerID
            cleaned_dict[key_str] = clean(v)
        return cleaned_dict

    # Handle lists, tuples, sets
    if isinstance(obj, (list, tuple, set)):
        return [clean(item) for item in obj]

    # Handle Pydantic models
    if hasattr(obj, "model_dump"):
        return clean(obj.model_dump())

    return obj


class CleanJSONResponse(JSONResponse):
    """Custom JSONResponse applying clean() to ensure JSON-safety."""

    def render(self, content: Any) -> bytes:
        return super().render(clean(content))


# -----------------------------------------------------------------------------
# Application Data Store Container
# -----------------------------------------------------------------------------

class AppDataStore:
    """Holds data and model artifacts loaded once at start-up."""

    def __init__(self) -> None:
        self.weekly_demand: Optional[pd.DataFrame] = None
        self.next_week_forecast: Optional[pd.DataFrame] = None
        self.metrics: Optional[pd.DataFrame] = None
        self.predictions: Optional[pd.DataFrame] = None
        self.error_analysis: Optional[pd.DataFrame] = None
        self.product_intelligence: Optional[pd.DataFrame] = None
        self.basket_pairs: Optional[pd.DataFrame] = None
        self.config: Dict[str, Any] = {}
        self.ablation: List[Dict[str, Any]] = []
        self.interval_coverage: Dict[str, Any] = {}
        self.policy_backtest: Optional[pd.DataFrame] = None
        self.lifecycle: Dict[str, str] = {}
        self.products: List[Dict[str, Any]] = []
        self.closed_weeks: List[str] = ["2010-12-27"]


data_store = AppDataStore()


# -----------------------------------------------------------------------------
# Risk Cache (FR-A-6, FR-DB-6)
# -----------------------------------------------------------------------------
# In-process dict keyed by (prefix + sorted-JSON of params + version counter).
# Any write to inventory_levels or purchase_orders bumps the counter, which
# makes all stale keys unreachable without iterating the dict.

_risk_cache: Dict[str, Any] = {}
_risk_cache_version: int = 0
_risk_cache_lock = threading.Lock()


def _make_cache_key(prefix: str, **kwargs: Any) -> str:
    """Build a stable string cache key from keyword args plus the current version counter."""
    payload = {"_v": _risk_cache_version, **kwargs}
    return prefix + ":" + json.dumps(payload, sort_keys=True, default=str)


def _invalidate_risk_cache() -> None:
    """Bump the version counter and clear the cache dict (O(n) but n is small)."""
    global _risk_cache_version
    with _risk_cache_lock:
        _risk_cache_version += 1
        _risk_cache.clear()


def _load_risk_inputs(db: Session) -> tuple[pd.DataFrame, Dict[str, Any], Dict[str, int]]:
    """Return (forecast_df, stock_overrides, open_orders) from data_store + DB.

    stock_overrides maps product_id -> {on_hand, lead_time_weeks, service_level, source}
    for every row in inventory_levels.  open_orders maps product_id -> total qty.
    """
    if data_store.next_week_forecast is None:
        raise HTTPException(status_code=500, detail="Forecast data not loaded")
    forecast_df = data_store.next_week_forecast.copy()
    inv_rows = db.query(InventoryLevel).all()
    stock_overrides: Dict[str, Any] = {
        r.product_id: {
            "on_hand": float(r.on_hand),
            "lead_time_weeks": int(r.lead_time_weeks),
            "service_level": float(r.service_level),
            "source": "user",
        }
        for r in inv_rows
    }
    open_orders = open_order_quantities(db)
    return forecast_df, stock_overrides, open_orders


def load_application_data(base_dir: Optional[Path] = None) -> AppDataStore:
    """
    Validate presence of all required files and load them into memory.
    Raises RuntimeError naming the missing file and the command to create it if absent.
    """
    root = base_dir or Path.cwd()

    # 1. Check all required files first
    for rel_path, cmd in REQUIRED_DATA_FILES.items():
        file_path = root / rel_path
        if not file_path.exists():
            raise RuntimeError(
                f"Missing required file: '{rel_path}'. Run '{cmd}' to create it."
            )

    store = AppDataStore()

    # 2. Load required files
    store.weekly_demand = pd.read_parquet(root / "data/processed/weekly_demand.parquet")
    store.next_week_forecast = pd.read_csv(root / "data/processed/next_week_forecast.csv")
    store.metrics = pd.read_csv(root / "data/processed/metrics.csv")
    store.predictions = pd.read_csv(root / "data/processed/predictions.csv")
    store.error_analysis = pd.read_csv(root / "data/processed/error_analysis.csv")
    store.product_intelligence = pd.read_csv(root / "data/processed/product_intelligence.csv")
    store.basket_pairs = pd.read_csv(root / "data/processed/basket_pairs.csv")

    with open(root / "models/config.json", "r", encoding="utf-8") as f:
        store.config = json.load(f)

    with open(root / "models/ablation.json", "r", encoding="utf-8") as f:
        store.ablation = json.load(f)

    with open(root / "models/interval_coverage.json", "r", encoding="utf-8") as f:
        store.interval_coverage = json.load(f)

    # 3. Join next_week_forecast with descriptions from product_intelligence
    if "description" in store.product_intelligence.columns:
        desc_map = (
            store.product_intelligence[["product_id", "description"]]
            .drop_duplicates(subset=["product_id"])
            .copy()
        )
        desc_map["product_id"] = desc_map["product_id"].astype(str)
        store.next_week_forecast["product_id"] = store.next_week_forecast["product_id"].astype(str)
        if "description" not in store.next_week_forecast.columns:
            store.next_week_forecast = store.next_week_forecast.merge(
                desc_map, on="product_id", how="left"
            )

    # 4. Build lifecycle mapping from product_intelligence
    if "lifecycle_stage" in store.product_intelligence.columns:
        store.lifecycle = dict(
            zip(
                store.product_intelligence["product_id"].astype(str),
                store.product_intelligence["lifecycle_stage"].astype(str),
            )
        )

    # 5. Load optional policy_backtest.csv if present
    backtest_path = root / "data/processed/policy_backtest.csv"
    if backtest_path.exists():
        try:
            store.policy_backtest = pd.read_csv(backtest_path)
        except Exception:
            store.policy_backtest = None
    else:
        store.policy_backtest = None

    # 6. Load closed weeks if present
    closed_path = root / "data/processed/closed_weeks.csv"
    if closed_path.exists():
        try:
            cw_df = pd.read_csv(closed_path)
            store.closed_weeks = [str(w)[:10] for w in cw_df["week_start"].dropna().tolist()]
        except Exception:
            store.closed_weeks = ["2010-12-27"]
    else:
        store.closed_weeks = ["2010-12-27"]

    # 7. Precompute Top 100 products list ordered by training units descending
    train_wd = store.weekly_demand[
        store.weekly_demand["week_start"].astype(str).str[:10] < "2011-10-10"
    ]
    totals = (
        train_wd.groupby("product_id")["units"]
        .sum()
        .reset_index()
        .rename(columns={"units": "total_units"})
    )
    totals["product_id"] = totals["product_id"].astype(str)

    pi_subset = store.product_intelligence[
        ["product_id", "description", "pattern_class"]
    ].copy()
    pi_subset["product_id"] = pi_subset["product_id"].astype(str)

    merged_prods = totals.merge(pi_subset, on="product_id", how="left")
    merged_prods = merged_prods.sort_values(
        by="total_units", ascending=False
    ).reset_index(drop=True)

    store.products = [
        {
            "product_id": str(r["product_id"]),
            "description": str(r["description"]) if pd.notna(r["description"]) else None,
            "total_units": int(r["total_units"]),
            "pattern_class": str(r["pattern_class"]) if pd.notna(r["pattern_class"]) else None,
        }
        for _, r in merged_prods.iterrows()
    ]

    return store


# -----------------------------------------------------------------------------
# Lifespan Handler
# -----------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager:
    1. Initialize database tables (init_db).
    2. Seed initial demo data (seed_demo_data).
    3. Load data and model artifacts once, stopping with a clear message if missing.
    """
    # Initialize DB schema and seed demo records
    init_db()
    seed_demo_data()

    # Load project datasets and models
    loaded_store = load_application_data()
    # Populate global container and app.state
    global data_store
    data_store = loaded_store
    app.state.data = loaded_store
    app.state.lifecycle = loaded_store.lifecycle

    yield


# -----------------------------------------------------------------------------
# FastAPI App and Router
# -----------------------------------------------------------------------------

app = FastAPI(
    title="Shelfwatch API",
    description="Retail Product Demand Estimation and Stockout Risk Analysis API",
    version="3.2.0",
    default_response_class=CleanJSONResponse,
    lifespan=lifespan,
)

# Configure CORS
cors_origins_env = os.getenv("CORS_ORIGINS", "*")
origins = [o.strip() for o in cors_origins_env.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins if origins else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

api_router = APIRouter(prefix="/api", default_response_class=CleanJSONResponse)


# -----------------------------------------------------------------------------
# Meta Endpoints
# -----------------------------------------------------------------------------

@api_router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check and database table counts",
)
def get_health(db: Session = Depends(get_db)) -> Dict[str, Any]:
    """Check database connectivity and return key table row counts."""
    tables: Dict[str, int] = {}
    connected = False
    try:
        # Check connection
        db.execute(text("SELECT 1"))
        connected = True
        # Count rows in inventory_levels
        inv_count = db.query(InventoryLevel).count()
        tables["inventory_levels"] = inv_count
    except Exception:
        connected = False

    return {
        "status": "ok" if connected else "degraded",
        "database": {
            "connected": connected,
            "tables": tables,
        },
    }


@api_router.get(
    "/disclaimer",
    response_model=DisclaimerResponse,
    summary="Mandatory inventory simulation disclaimer",
)
def get_disclaimer() -> Dict[str, str]:
    """Return the mandatory simulated-inventory disclaimer text (SRS 6.1, FR-U-3)."""
    return {
        "text": (
            "Inventory levels are simulated. The dataset contains sales only, "
            "not stock-on-hand. Risk flags show how the rule behaves under the "
            "stated assumptions, not real stock positions."
        )
    }


@api_router.get(
    "/model-info",
    response_model=ModelInfoResponse,
    summary="Model metadata, performance metrics, and ablation results",
)
def get_model_info() -> Dict[str, Any]:
    """Return winning model metadata, headline metrics, ablation and backtest results."""
    config = data_store.config or {}
    model_name = config.get("model_name", "ensemble")
    target_mode = config.get("target_mode", "raw")
    feature_groups = config.get("groups", ["base", "g3"])
    trained_on = config.get("train_end", "2011-10-03")
    test_start = config.get("test_start", "2011-10-10")

    # Extract headline metrics from metrics.csv for winning model on test split
    headline_metrics: Dict[str, float] = {}
    if data_store.metrics is not None and not data_store.metrics.empty:
        df_m = data_store.metrics
        matching = df_m[(df_m["model_name"] == model_name) & (df_m["split"] == "test")]
        if not matching.empty:
            row = matching.iloc[0]
            headline_metrics = {
                "MAE": float(row["MAE"]),
                "RMSE": float(row["RMSE"]),
                "WAPE": float(row["WAPE"]),
                "bias": float(row["bias"]),
            }
        else:
            # Fallback to first test row or validation metrics
            test_rows = df_m[df_m["split"] == "test"]
            if not test_rows.empty:
                row = test_rows.iloc[0]
                headline_metrics = {
                    "MAE": float(row["MAE"]),
                    "RMSE": float(row["RMSE"]),
                    "WAPE": float(row["WAPE"]),
                    "bias": float(row["bias"]),
                }

    # Format ablation entries: list of {groups, val_wape, val_bias}
    ablation_list: List[Dict[str, Any]] = []
    if isinstance(data_store.ablation, list):
        for entry in data_store.ablation:
            if isinstance(entry, dict) and "groups" in entry:
                ablation_list.append(
                    {
                        "groups": str(entry.get("groups", "")),
                        "val_wape": float(entry.get("val_wape", 0.0)),
                        "val_bias": float(entry.get("val_bias", 0.0)),
                    }
                )

    # Format interval coverage: {val, test}
    cov = data_store.interval_coverage or {}
    interval_cov = {
        "val": float(cov.get("val", 0.80)),
        "test": float(cov.get("test", 0.80)),
    }

    # Format policy_backtest: list of {policy, lead_time_weeks, fill_rate, stockout_weeks, avg_on_hand, unmet_units}
    policy_backtest_list: List[Dict[str, Any]] = []
    if data_store.policy_backtest is not None and not data_store.policy_backtest.empty:
        pb_df = data_store.policy_backtest
        for _, r in pb_df.iterrows():
            policy_backtest_list.append(
                {
                    "policy": str(r.get("policy", "")),
                    "lead_time_weeks": int(r.get("lead_time_weeks", 1)),
                    "abc_class": str(r.get("abc_class", "ALL")) if pd.notna(r.get("abc_class")) else None,
                    "fill_rate": float(r.get("fill_rate", 0.0)),
                    "stockout_weeks": int(r.get("stockout_weeks", 0)),
                    "avg_on_hand": float(r.get("avg_on_hand", 0.0)),
                    "unmet_units": float(r.get("unmet_units", 0.0)),
                }
            )

    # Kaggle URLs from environment variables
    kaggle_urls = {
        "dataset_url": os.getenv("KAGGLE_DATASET_URL", ""),
        "notebook_url": os.getenv("KAGGLE_NOTEBOOK_URL", ""),
        "model_url": os.getenv("KAGGLE_MODEL_URL", ""),
    }

    library_versions = config.get("library_versions", {})

    return {
        "model_name": model_name,
        "target_mode": target_mode,
        "feature_groups": feature_groups,
        "trained_on": trained_on,
        "test_start": test_start,
        "headline_metrics": headline_metrics,
        "ablation": ablation_list,
        "interval_coverage": interval_cov,
        "policy_backtest": policy_backtest_list,
        "library_versions": library_versions,
        "kaggle": kaggle_urls,
    }


# -----------------------------------------------------------------------------
# Product Catalog and Forecast Endpoints (Task B10)
# -----------------------------------------------------------------------------

@api_router.get(
    "/products",
    response_model=List[ProductListItem],
    summary="Top 100 products catalog ordered by training units",
)
def get_products() -> List[Dict[str, Any]]:
    """Return top 100 products ordered by training units descending."""
    return data_store.products


@api_router.get(
    "/forecast/{product_id}",
    response_model=ForecastResponse,
    summary="Single product 52-week history, test predictions, and next-week forecast",
)
def get_forecast(product_id: str) -> Dict[str, Any]:
    """
    Return 52-week demand history (closed weeks with units null),
    test predictions from best model, and next week point/quantile forecast.
    """
    pid = str(product_id).strip()

    # 1. Verify product exists in top 100
    matching_prod = next(
        (p for p in data_store.products if p["product_id"] == pid), None
    )
    if not matching_prod:
        raise HTTPException(status_code=404, detail="Product not found")

    desc = matching_prod.get("description")
    closed_weeks_set = set(data_store.closed_weeks)

    # 2. Build 52-week history grid (Mondays 2010-12-06 to 2011-11-28)
    full_weeks = (
        pd.date_range("2010-12-06", "2011-11-28", freq="W-MON")
        .strftime("%Y-%m-%d")
        .tolist()
    )

    p_wd = data_store.weekly_demand[
        data_store.weekly_demand["product_id"].astype(str) == pid
    ]
    p_wd_map = dict(
        zip(
            p_wd["week_start"].astype(str).str[:10],
            p_wd["units"],
        )
    )

    history: List[Dict[str, Any]] = []
    for w in full_weeks:
        if w in closed_weeks_set:
            history.append({"week_start": w, "units": None})
        else:
            u = p_wd_map.get(w, 0.0)
            history.append({"week_start": w, "units": float(u)})

    # 3. Test predictions for best model on test split
    best_model_name = data_store.config.get("model_name", "ensemble")
    p_preds = data_store.predictions[
        (data_store.predictions["product_id"].astype(str) == pid)
        & (data_store.predictions["model_name"] == best_model_name)
        & (data_store.predictions["split"] == "test")
    ].sort_values(by="week_start")

    test_predictions: List[Dict[str, Any]] = []
    for _, r in p_preds.iterrows():
        test_predictions.append(
            {
                "week_start": str(r["week_start"])[:10],
                "actual": float(r["actual"]),
                "predicted": float(r["predicted"]),
            }
        )

    # 4. Next week forecast
    p_fc = data_store.next_week_forecast[
        data_store.next_week_forecast["product_id"].astype(str) == pid
    ]
    if p_fc.empty:
        raise HTTPException(
            status_code=404, detail="Forecast not found for product"
        )
    fc_row = p_fc.iloc[0]
    next_week_fc = {
        "week_start": str(fc_row.get("week_start", "2011-12-05"))[:10],
        "forecast_weekly": float(fc_row["forecast_weekly"]),
        "forecast_p10": float(fc_row["forecast_p10"]),
        "forecast_p90": float(fc_row["forecast_p90"]),
    }

    return {
        "product_id": pid,
        "description": desc,
        "closed_weeks": data_store.closed_weeks,
        "history": history,
        "test_predictions": test_predictions,
        "next_week_forecast": next_week_fc,
    }


# -----------------------------------------------------------------------------
# Prediction Endpoints (Task B11)
# -----------------------------------------------------------------------------

@api_router.post(
    "/predict",
    response_model=PredictResponse,
    summary="Generate next-week forecast and stockout risk advice for a product",
)
def predict_product(
    payload: PredictRequest,
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """
    Generate next-week demand forecast, compute inventory stockout risk,
    and return replenishment advice for a single product.
    """
    pid = str(payload.product_id).strip()

    # 1. Lookup forecast row in next_week_forecast
    if data_store.next_week_forecast is None or data_store.next_week_forecast.empty:
        raise HTTPException(status_code=500, detail="Forecast data not loaded")

    p_fc = data_store.next_week_forecast[
        data_store.next_week_forecast["product_id"].astype(str) == pid
    ]
    if p_fc.empty:
        raise HTTPException(status_code=404, detail="Product not found")

    forecast_row_df = p_fc.iloc[0:1].copy()

    # 2. Determine stock override
    stock_overrides: Dict[str, Dict[str, Any]] = {}
    if payload.on_hand is not None:
        stock_overrides[pid] = {
            "on_hand": float(payload.on_hand),
            "lead_time_weeks": payload.lead_time_weeks,
            "service_level": payload.service_level,
            "source": "user",
        }
    else:
        # Check if database has an InventoryLevel record
        inv_record = db.query(InventoryLevel).filter_by(product_id=pid).first()
        if inv_record is not None and inv_record.on_hand is not None:
            stock_overrides[pid] = {
                "on_hand": float(inv_record.on_hand),
                "lead_time_weeks": payload.lead_time_weeks or inv_record.lead_time_weeks,
                "service_level": payload.service_level or inv_record.service_level,
                "source": "db",
            }

    # 3. Fetch open purchase orders (Pending and Approved)
    open_orders = open_order_quantities(db)

    # 4. Parameters for compute_risk
    params = {
        "lead_time_weeks": payload.lead_time_weeks,
        "service_level": payload.service_level,
    }

    try:
        risk_df = compute_risk(
            forecast_df=forecast_row_df,
            params=params,
            stock_overrides=stock_overrides,
            open_orders=open_orders,
            lifecycle=data_store.lifecycle,
        )
    except ValueError as val_err:
        raise HTTPException(status_code=422, detail=str(val_err))

    if risk_df.empty:
        raise HTTPException(status_code=500, detail="Failed to compute risk metrics")

    res = risk_df.iloc[0]
    result_data = {
        "product_id": pid,
        "week_start": str(forecast_row_df.iloc[0].get("week_start", "2011-12-05"))[:10],
        "forecast_weekly": float(res["forecast_weekly"]),
        "forecast_p10": float(res["forecast_p10"]),
        "forecast_p90": float(res["forecast_p90"]),
        "risk_level": str(res["risk_level"]),
        "action": str(res["action"]),
        "on_hand": float(res["on_hand"]),
        "on_hand_source": str(res["on_hand_source"]),
        "reorder_point": float(res["reorder_point"]),
        "days_of_cover": float(res["days_of_cover"]),
        "stockout_probability": float(res["stockout_probability"]),
        "recommended_order_qty": float(res["recommended_order_qty"]),
        "days_to_reorder": float(res["days_to_reorder"]),
        "revenue_at_risk": float(res["revenue_at_risk"]),
    }

    # 5. Log prediction to prediction_log (SRS FR-A-15)
    try:
        log_entry = PredictionLog(
            endpoint="/api/predict",
            product_id=pid,
            inputs=clean(payload.model_dump()),
            outputs=clean(result_data),
            model_name=data_store.config.get("model_name", "ensemble"),
        )
        db.add(log_entry)
        db.commit()
    except Exception:
        db.rollback()

    return result_data


@api_router.post(
    "/predict/custom",
    response_model=PredictCustomResponse,
    summary="Generate demand forecast and intervals for user-entered history",
)
def predict_custom_endpoint(
    payload: PredictCustomRequest,
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """
    Generate point forecast and 80% prediction interval for 8 to 12
    user-entered demand history values.
    """
    if any(u < 0 for u in payload.recent_units):
        raise HTTPException(
            status_code=422,
            detail="Negative demand values are not allowed in recent_units.",
        )

    try:
        res = predict_custom(payload.model_dump())
    except ValueError as val_err:
        raise HTTPException(status_code=422, detail=str(val_err))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    result_data = {
        "forecast_weekly": float(res["forecast_weekly"]),
        "forecast_p10": float(res["forecast_p10"]),
        "forecast_p90": float(res["forecast_p90"]),
        "features_used": [str(f) for f in res.get("features_used", [])],
        "notes": str(res.get("notes", "")),
    }

    # Log custom prediction to prediction_log (SRS FR-A-15)
    try:
        log_entry = PredictionLog(
            endpoint="/api/predict/custom",
            product_id=None,
            inputs=clean(payload.model_dump()),
            outputs=clean(result_data),
            model_name=data_store.config.get("model_name", "ensemble"),
        )
        db.add(log_entry)
        db.commit()
    except Exception:
        db.rollback()

    return result_data


# -----------------------------------------------------------------------------
# Risk, Simulate, Sensitivity Endpoints (Task B12 — FR-A-6, FR-R-12)
# -----------------------------------------------------------------------------

@api_router.post(
    "/risk",
    response_model=RiskResponse,
    summary="Portfolio risk table for all 100 products with given parameters",
)
def post_risk(
    payload: RiskRequest,
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Compute stockout risk for every product using the given parameters.

    Stock overrides and open orders are read from the database; no risk
    formulas live in the API layer.  Results are cached by parameter set
    (including a version counter invalidated on any inventory or order write).
    """
    cache_key = _make_cache_key(
        "risk",
        lead_time_weeks=payload.lead_time_weeks,
        service_level=payload.service_level,
        initial_stock_weeks=payload.initial_stock_weeks,
    )
    with _risk_cache_lock:
        if cache_key in _risk_cache:
            return _risk_cache[cache_key]

    forecast_df, stock_overrides, open_orders = _load_risk_inputs(db)
    params: Dict[str, Any] = {
        "lead_time_weeks": payload.lead_time_weeks,
        "service_level": payload.service_level,
        "initial_stock_weeks": payload.initial_stock_weeks,
    }
    try:
        risk_df = compute_risk(
            forecast_df=forecast_df,
            params=params,
            stock_overrides=stock_overrides,
            open_orders=open_orders,
            lifecycle=data_store.lifecycle,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    summary = summarize(risk_df)
    echoed_params = {**params, "seed": DEFAULTS["seed"]}
    result: Dict[str, Any] = {
        "params": echoed_params,
        "summary": summary,
        "items": clean(risk_df),
    }
    with _risk_cache_lock:
        _risk_cache[cache_key] = result
    return result


@api_router.post(
    "/simulate",
    response_model=SimulateResponse,
    summary="What-if: show impact of a supplier delay on portfolio risk",
)
def post_simulate(
    payload: SimulateRequest,
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Compare risk at effective_lead_time = lead_time_weeks + delay_weeks
    against the baseline (delay_weeks=0).

    Returns the new summary, the baseline summary, and the list of products
    whose risk class changed between the two scenarios.
    """
    effective_lt = payload.lead_time_weeks + payload.delay_weeks
    if effective_lt > 6:
        raise HTTPException(
            status_code=422,
            detail=(
                f"lead_time_weeks ({payload.lead_time_weeks}) + "
                f"delay_weeks ({payload.delay_weeks}) = {effective_lt} exceeds "
                "the maximum allowed lead time of 6 weeks."
            ),
        )

    cache_key = _make_cache_key(
        "simulate",
        lead_time_weeks=payload.lead_time_weeks,
        delay_weeks=payload.delay_weeks,
        service_level=payload.service_level,
        initial_stock_weeks=payload.initial_stock_weeks,
    )
    with _risk_cache_lock:
        if cache_key in _risk_cache:
            return _risk_cache[cache_key]

    forecast_df, stock_overrides, open_orders = _load_risk_inputs(db)

    base_params: Dict[str, Any] = {
        "lead_time_weeks": payload.lead_time_weeks,
        "service_level": payload.service_level,
        "initial_stock_weeks": payload.initial_stock_weeks,
    }
    sim_params: Dict[str, Any] = {
        "lead_time_weeks": effective_lt,
        "service_level": payload.service_level,
        "initial_stock_weeks": payload.initial_stock_weeks,
    }

    try:
        baseline_df = compute_risk(
            forecast_df=forecast_df,
            params=base_params,
            stock_overrides=stock_overrides,
            open_orders=open_orders,
            lifecycle=data_store.lifecycle,
        )
        sim_df = compute_risk(
            forecast_df=forecast_df,
            params=sim_params,
            stock_overrides=stock_overrides,
            open_orders=open_orders,
            lifecycle=data_store.lifecycle,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    baseline_summary = summarize(baseline_df)
    sim_summary = summarize(sim_df)

    # Diff: products whose risk class changed
    merged = baseline_df[["product_id", "risk_level"]].merge(
        sim_df[["product_id", "risk_level"]],
        on="product_id",
        suffixes=("_base", "_sim"),
    )
    changed: List[Dict[str, str]] = [
        {
            "product_id": str(row["product_id"]),
            "from": str(row["risk_level_base"]),
            "to": str(row["risk_level_sim"]),
        }
        for _, row in merged.iterrows()
        if row["risk_level_base"] != row["risk_level_sim"]
    ]

    echoed_params: Dict[str, Any] = {
        **sim_params,
        "delay_weeks": payload.delay_weeks,
        "seed": DEFAULTS["seed"],
    }
    result: Dict[str, Any] = {
        "params": echoed_params,
        "summary": sim_summary,
        "baseline_summary": baseline_summary,
        "changed": changed,
    }
    with _risk_cache_lock:
        _risk_cache[cache_key] = result
    return result


@api_router.post(
    "/sensitivity",
    response_model=List[SensitivityRow],
    summary="Sensitivity table: HIGH/MEDIUM/LOW counts across lead time × service level grid",
)
def post_sensitivity(
    payload: SensitivityRequest,
    db: Session = Depends(get_db),
) -> List[Dict[str, Any]]:
    """Sweep the supplied lead_times (≤ 8) × service_levels (≤ 8) grid and return
    HIGH, MEDIUM, LOW counts for each combination.  Results are cached.
    """
    lead_times = payload.lead_times[:8]
    service_levels = payload.service_levels[:8]

    for lt in lead_times:
        if not (1 <= lt <= 4):
            raise HTTPException(
                status_code=422,
                detail=f"All lead_times values must be between 1 and 4; got {lt}.",
            )
    for sl in service_levels:
        if not (0.80 <= sl <= 0.99):
            raise HTTPException(
                status_code=422,
                detail=f"All service_levels values must be between 0.80 and 0.99; got {sl}.",
            )
    if not lead_times:
        raise HTTPException(status_code=422, detail="lead_times must not be empty.")
    if not service_levels:
        raise HTTPException(status_code=422, detail="service_levels must not be empty.")

    cache_key = _make_cache_key(
        "sensitivity",
        lead_times=sorted(lead_times),
        service_levels=sorted(service_levels),
    )
    with _risk_cache_lock:
        if cache_key in _risk_cache:
            return _risk_cache[cache_key]

    forecast_df, stock_overrides, open_orders = _load_risk_inputs(db)
    try:
        sens_df = sensitivity_table(
            forecast_df=forecast_df,
            lead_times=lead_times,
            service_levels=service_levels,
            stock_overrides=stock_overrides,
            open_orders=open_orders,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    rows: List[Dict[str, Any]] = sens_df.to_dict(orient="records")
    with _risk_cache_lock:
        _risk_cache[cache_key] = rows
    return rows


# -----------------------------------------------------------------------------
# Inventory CRUD Endpoints (FR-DB-4, FR-DB-5, FR-DB-6, contract C7)
# -----------------------------------------------------------------------------

def _get_api_key() -> str:
    """Return the configured API key (empty string if not set)."""
    return os.getenv("API_KEY", "")


def _require_api_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    """Dependency: reject write requests that lack the correct API key (FR-MW-4)."""
    expected = _get_api_key()
    if expected and x_api_key != expected:
        raise HTTPException(status_code=401, detail="Missing or invalid X-API-Key header.")


@api_router.get(
    "/inventory",
    response_model=List[InventoryResponse],
    summary="List all user-entered stock records",
)
def get_inventory(db: Session = Depends(get_db)) -> List[Dict[str, Any]]:
    """Return all inventory_levels rows ordered by product_id."""
    records = list_stock(db)
    return [
        {
            "product_id": r.product_id,
            "on_hand": float(r.on_hand),
            "lead_time_weeks": int(r.lead_time_weeks),
            "service_level": float(r.service_level),
            "updated_at": r.updated_at.isoformat() if r.updated_at else None,
        }
        for r in records
    ]


@api_router.put(
    "/inventory/{product_id}",
    response_model=InventoryResponse,
    status_code=200,
    summary="Create or replace stock record for a product (needs X-API-Key)",
)
def put_inventory(
    product_id: str,
    payload: InventoryCreateUpdateRequest,
    db: Session = Depends(get_db),
    _auth: None = Depends(_require_api_key),
) -> Any:
    """Upsert stock level.  Returns 201 on create, 200 on replace.
    Clears the risk cache after every write (FR-DB-6).
    """
    try:
        record, created = upsert_stock(db, product_id, payload)
    except NotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    _invalidate_risk_cache()

    response_body = {
        "product_id": record.product_id,
        "on_hand": float(record.on_hand),
        "lead_time_weeks": int(record.lead_time_weeks),
        "service_level": float(record.service_level),
        "updated_at": record.updated_at.isoformat() if record.updated_at else None,
    }
    # Return 201 for new records
    if created:
        from fastapi.responses import JSONResponse as _JSONResponse
        return _JSONResponse(content=clean(response_body), status_code=201)
    return response_body


@api_router.delete(
    "/inventory/{product_id}",
    status_code=204,
    response_model=None,
    summary="Delete stock record; product returns to simulated stock (needs X-API-Key)",
)
def delete_inventory(
    product_id: str,
    db: Session = Depends(get_db),
    _auth: None = Depends(_require_api_key),
) -> None:
    """Remove a user-entered stock record.  Clears the risk cache."""
    try:
        delete_stock(db, product_id)
    except NotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    _invalidate_risk_cache()


# -----------------------------------------------------------------------------
# Order CRUD Endpoints (FR-DB-4, FR-DB-5, FR-DB-6, FR-W-*, contract C7)
# -----------------------------------------------------------------------------

@api_router.post(
    "/orders",
    response_model=OrderResponse,
    status_code=201,
    summary="Create a purchase order in Pending status (needs X-API-Key)",
)
def create_purchase_order(
    payload: OrderCreateRequest,
    db: Session = Depends(get_db),
    _auth: None = Depends(_require_api_key),
) -> Dict[str, Any]:
    """Create a Pending purchase order and clear the risk cache."""
    try:
        order = create_order(db, payload)
    except NotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    _invalidate_risk_cache()
    return _order_to_dict(order)


@api_router.get(
    "/orders",
    response_model=List[OrderResponse],
    summary="List purchase orders with optional status and product_id filters",
)
def list_purchase_orders(
    status: Optional[str] = None,
    product_id: Optional[str] = None,
    db: Session = Depends(get_db),
) -> List[Dict[str, Any]]:
    """Return purchase orders, optionally filtered by status and/or product_id."""
    orders = list_orders(db, status=status, product_id=product_id)
    return [_order_to_dict(o) for o in orders]


@api_router.get(
    "/orders/{order_id}",
    response_model=OrderResponse,
    summary="Retrieve a single purchase order by id",
)
def get_purchase_order(order_id: int, db: Session = Depends(get_db)) -> Dict[str, Any]:
    """Return one purchase order."""
    try:
        order = get_order(db, order_id)
    except NotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return _order_to_dict(order)


@api_router.patch(
    "/orders/{order_id}",
    response_model=OrderResponse,
    summary="Change order status or quantity (needs X-API-Key)",
)
def patch_purchase_order(
    order_id: int,
    payload: OrderUpdateRequest,
    db: Session = Depends(get_db),
    _auth: None = Depends(_require_api_key),
) -> Dict[str, Any]:
    """Update status or quantity.  Receiving an order adds its qty to inventory.
    Clears the risk cache on status writes.
    """
    from api.crud_orders import ConflictError, ValidationError as OrderValidationError

    try:
        order = update_order(db, order_id, payload)
    except NotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except (ValidationError, OrderValidationError) as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    _invalidate_risk_cache()
    return _order_to_dict(order)


@api_router.delete(
    "/orders/{order_id}",
    status_code=204,
    response_model=None,
    summary="Delete a Pending or Cancelled order (needs X-API-Key)",
)
def delete_purchase_order(
    order_id: int,
    db: Session = Depends(get_db),
    _auth: None = Depends(_require_api_key),
) -> None:
    """Delete an order.  Only Pending or Cancelled orders may be deleted (409 otherwise)."""
    from api.crud_orders import ConflictError

    try:
        delete_order(db, order_id)
    except NotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc))

    _invalidate_risk_cache()


def _order_to_dict(order: PurchaseOrder) -> Dict[str, Any]:
    """Serialize a PurchaseOrder ORM row to a plain dict."""
    return {
        "id": order.id,
        "product_id": order.product_id,
        "quantity": order.quantity,
        "supplier_note": order.supplier_note,
        "status": order.status,
        "created_at": order.created_at.isoformat() if order.created_at else None,
        "updated_at": order.updated_at.isoformat() if order.updated_at else None,
    }


# Include the API router
app.include_router(api_router)


# -----------------------------------------------------------------------------
# Root Aliases (SRS 7.1, CONTRACTS.md, mentor's Postman tests)
# -----------------------------------------------------------------------------

@app.get(
    "/health",
    response_model=HealthResponse,
    include_in_schema=False,
)
def get_health_root_alias(db: Session = Depends(get_db)) -> Dict[str, Any]:
    """Root alias for /api/health (CONTRACTS_ADDENDUM.md / SRS FR-A-13)."""
    return get_health(db=db)


@app.post(
    "/predict",
    response_model=PredictResponse,
    include_in_schema=False,
)
def predict_product_root_alias(
    payload: PredictRequest,
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Root alias for /api/predict (CONTRACTS_ADDENDUM.md / SRS FR-A-13)."""
    return predict_product(payload=payload, db=db)
