"""api/schemas.py - Pydantic v2 request and response models with documentation examples for OpenAPI /docs."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field


# -----------------------------------------------------------------------------
# Base Schema Configuration
# -----------------------------------------------------------------------------

class BaseSchema(BaseModel):
    """Base schema with standard Pydantic configuration."""
    model_config = ConfigDict(
        populate_by_name=True,
        extra="ignore",
    )


# -----------------------------------------------------------------------------
# Error Schema
# -----------------------------------------------------------------------------

class ErrorDetail(BaseSchema):
    """Standard error response structure."""
    detail: str = Field(..., description="Human-readable error description")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "detail": "Product not found"
            }
        }
    )


# -----------------------------------------------------------------------------
# Meta Endpoints Schemas
# -----------------------------------------------------------------------------

class HealthDatabaseResponse(BaseSchema):
    """Database status details in health check."""
    connected: bool = Field(..., description="Whether database connection is healthy")
    tables: Dict[str, int] = Field(
        default_factory=dict,
        description="Row counts for key application tables",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "connected": True,
                "tables": {
                    "inventory_levels": 100,
                    "purchase_orders": 4,
                }
            }
        }
    )


class HealthResponse(BaseSchema):
    """GET /api/health response."""
    status: str = Field(..., description="API health status (e.g. 'ok')")
    database: HealthDatabaseResponse = Field(..., description="Database connectivity and row counts")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "ok",
                "database": {
                    "connected": True,
                    "tables": {
                        "inventory_levels": 100
                    }
                }
            }
        }
    )


class DisclaimerResponse(BaseSchema):
    """GET /api/disclaimer response."""
    text: str = Field(..., description="Mandatory inventory simulation disclaimer text")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "text": (
                    "Inventory levels are simulated. The dataset contains sales only, "
                    "not stock-on-hand. Risk flags show how the rule behaves under the "
                    "stated assumptions, not real stock positions."
                )
            }
        }
    )


class AblationEntry(BaseSchema):
    """Ablation experiment summary entry."""
    groups: str = Field(..., description="Feature group identifier or combination")
    val_wape: float = Field(..., description="Validation WAPE for the feature set")
    val_bias: float = Field(..., description="Validation bias for the feature set")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "groups": "base+g3",
                "val_wape": 0.5238,
                "val_bias": 2.0242,
            }
        }
    )


class IntervalCoverage(BaseSchema):
    """Coverage rate of 80% prediction intervals on validation and test sets."""
    val: float = Field(..., description="Validation set interval coverage rate (target 0.80)")
    test: float = Field(..., description="Test set interval coverage rate")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "val": 0.80,
                "test": 0.82,
            }
        }
    )


class PolicyBacktestEntry(BaseSchema):
    """Replenishment policy backtest summary row (contract C14)."""
    policy: str = Field(..., description="Inventory policy name")
    lead_time_weeks: int = Field(..., description="Evaluated lead time in weeks")
    abc_class: Optional[str] = Field(None, description="Product revenue tier ('A', 'B', 'C', or 'ALL')")
    fill_rate: float = Field(..., description="Fraction of demand satisfied from stock")
    stockout_weeks: int = Field(..., description="Product-weeks where demand exceeded stock")
    avg_on_hand: float = Field(..., description="Average ending inventory across test period")
    unmet_units: float = Field(..., description="Total unfulfilled demand units")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "policy": "best_model_safety",
                "lead_time_weeks": 1,
                "abc_class": "ALL",
                "fill_rate": 0.9451,
                "stockout_weeks": 47,
                "avg_on_hand": 465.44,
                "unmet_units": 12957.8,
            }
        }
    )


class KaggleUrls(BaseSchema):
    """External links to published Kaggle artifacts."""
    dataset_url: str = Field("", description="Published Kaggle dataset URL")
    notebook_url: str = Field("", description="Published Kaggle notebook URL")
    model_url: str = Field("", description="Published Kaggle model URL")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "dataset_url": "https://www.kaggle.com/datasets/example/shelfwatch",
                "notebook_url": "https://www.kaggle.com/code/example/shelfwatch-training",
                "model_url": "https://www.kaggle.com/models/example/shelfwatch-ensemble",
            }
        }
    )


class ModelInfoResponse(BaseSchema):
    """GET /api/model-info response (contract C7 / CONTRACTS_ADDENDUM.md)."""
    model_name: str = Field(..., description="Winning production model name")
    target_mode: str = Field(..., description="Target modeling approach (e.g. 'raw')")
    feature_groups: List[str] = Field(..., description="Active feature groups used in winning model")
    trained_on: str = Field(..., description="End date of training period (YYYY-MM-DD)")
    test_start: str = Field(..., description="Start date of held-out test period (YYYY-MM-DD)")
    headline_metrics: Dict[str, float] = Field(..., description="MAE, RMSE, WAPE, and bias on test split")
    ablation: List[AblationEntry] = Field(default_factory=list, description="Ablation experiment results")
    interval_coverage: IntervalCoverage = Field(..., description="Interval coverage metrics")
    policy_backtest: List[PolicyBacktestEntry] = Field(default_factory=list, description="Policy backtest metrics")
    library_versions: Dict[str, str] = Field(default_factory=dict, description="Pinned library versions")
    kaggle: KaggleUrls = Field(default_factory=KaggleUrls, description="Kaggle publication URLs")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "model_name": "ensemble",
                "target_mode": "raw",
                "feature_groups": ["base", "g3"],
                "trained_on": "2011-10-03",
                "test_start": "2011-10-10",
                "headline_metrics": {
                    "MAE": 141.3968,
                    "RMSE": 208.3836,
                    "WAPE": 0.4791,
                    "bias": 4.7711,
                },
                "ablation": [
                    {"groups": "base", "val_wape": 0.5357, "val_bias": 2.2483},
                    {"groups": "base+g3", "val_wape": 0.5238, "val_bias": 2.0242},
                ],
                "interval_coverage": {
                    "val": 0.80,
                    "test": 0.82,
                },
                "policy_backtest": [
                    {
                        "policy": "best_model_safety",
                        "lead_time_weeks": 1,
                        "abc_class": "ALL",
                        "fill_rate": 0.9451,
                        "stockout_weeks": 47,
                        "avg_on_hand": 465.44,
                        "unmet_units": 12957.8,
                    }
                ],
                "library_versions": {
                    "python": "3.11.9",
                    "scikit-learn": "1.6.1",
                    "lightgbm": "4.6.0",
                },
                "kaggle": {
                    "dataset_url": "",
                    "notebook_url": "",
                    "model_url": "",
                }
            }
        }
    )


# -----------------------------------------------------------------------------
# Product & Forecast Schemas (Task B10)
# -----------------------------------------------------------------------------

class ProductListItem(BaseSchema):
    """Product summary for catalog listing (GET /api/products)."""
    product_id: str = Field(..., description="Product SKU code")
    description: Optional[str] = Field(None, description="Product description")
    total_units: int = Field(..., description="Total units sold in training period")
    pattern_class: Optional[str] = Field(None, description="Demand pattern classification")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "product_id": "85123A",
                "description": "WHITE HANGING HEART T-LIGHT HOLDER",
                "total_units": 18542,
                "pattern_class": "smooth",
            }
        }
    )


class ForecastHistoryItem(BaseSchema):
    """Historical weekly demand row."""
    week_start: str = Field(..., description="Monday week start date (YYYY-MM-DD)")
    units: Optional[float] = Field(None, description="Units sold (null for closure week 2010-12-27)")


class TestPredictionItem(BaseSchema):
    """Test period actual vs predicted comparison row."""
    week_start: str = Field(..., description="Monday week start date (YYYY-MM-DD)")
    actual: float = Field(..., description="Observed units sold")
    predicted: float = Field(..., description="Model forecasted units")


class NextWeekForecastItem(BaseSchema):
    """Point and interval forecast for the target horizon week."""
    week_start: str = Field(..., description="Target week date (2011-12-05)")
    forecast_weekly: float = Field(..., description="Point forecast units")
    forecast_p10: float = Field(..., description="10th percentile demand forecast")
    forecast_p90: float = Field(..., description="90th percentile demand forecast")


class ForecastResponse(BaseSchema):
    """GET /api/forecast/{product_id} response."""
    product_id: str = Field(..., description="Product SKU code")
    description: Optional[str] = Field(None, description="Product description")
    closed_weeks: List[str] = Field(default_factory=list, description="Shop closure weeks")
    history: List[ForecastHistoryItem] = Field(..., description="Full 52-week demand history")
    test_predictions: List[TestPredictionItem] = Field(..., description="Test week model predictions")
    next_week_forecast: NextWeekForecastItem = Field(..., description="Horizon forecast with intervals")


# -----------------------------------------------------------------------------
# Predict & Risk Schemas (Tasks B11-B13)
# -----------------------------------------------------------------------------

class PredictRequest(BaseSchema):
    """POST /api/predict request."""
    product_id: str = Field(..., description="Product SKU code", json_schema_extra={"example": "85123A"})
    lead_time_weeks: int = Field(1, ge=1, le=4, description="Supplier lead time in weeks (1 to 4)")
    service_level: float = Field(0.95, ge=0.80, le=0.99, description="Target service level (0.80 to 0.99)")
    on_hand: Optional[float] = Field(None, ge=0, description="Override on-hand inventory position")


class PredictResponse(BaseSchema):
    """POST /api/predict response."""
    product_id: str = Field(..., description="Product SKU code")
    week_start: str = Field(..., description="Forecast target week (2011-12-05)")
    forecast_weekly: float = Field(..., description="Point forecast units")
    forecast_p10: float = Field(..., description="10th percentile forecast")
    forecast_p90: float = Field(..., description="90th percentile forecast")
    risk_level: str = Field(..., description="'HIGH', 'MEDIUM', or 'LOW'")
    action: str = Field(..., description="'ORDER_NOW', 'ORDER_SOON', 'REVIEW_STOP', or 'HOLD'")
    on_hand: float = Field(..., description="Evaluated on-hand stock position")
    on_hand_source: str = Field(..., description="'user', 'db', or 'simulated'")
    reorder_point: float = Field(..., description="Reorder point threshold units")
    days_of_cover: float = Field(..., description="Estimated days of inventory cover")
    stockout_probability: float = Field(..., description="Estimated probability of stockout before replenishment")
    recommended_order_qty: float = Field(..., description="Recommended replenishment order units")
    days_to_reorder: float = Field(..., description="Estimated days until reorder point is reached")
    revenue_at_risk: float = Field(..., description="Monetary value of potential stockout demand")


class PredictCustomRequest(BaseSchema):
    """POST /api/predict/custom request."""
    recent_units: List[float] = Field(..., min_length=8, max_length=12, description="8 to 12 weekly demand units, oldest first")
    recent_prices: Optional[List[float]] = Field(None, description="Optional weekly prices corresponding to units")
    recent_orders: Optional[List[int]] = Field(None, description="Optional weekly order counts")
    recent_customers: Optional[List[int]] = Field(None, description="Optional weekly customer counts")
    target_week: str = Field("2011-12-05", description="Forecast target week date (YYYY-MM-DD)")


class PredictCustomResponse(BaseSchema):
    """POST /api/predict/custom response."""
    forecast_weekly: float = Field(..., description="Point forecast units")
    forecast_p10: float = Field(..., description="10th percentile forecast")
    forecast_p90: float = Field(..., description="90th percentile forecast")
    features_used: List[str] = Field(..., description="Feature names utilized by the model")
    notes: str = Field(..., description="Notes regarding unavailable feature groups")


class RiskRequest(BaseSchema):
    """POST /api/risk request."""
    lead_time_weeks: int = Field(1, ge=1, le=4, description="Lead time weeks (1-4)")
    service_level: float = Field(0.95, ge=0.80, le=0.99, description="Target service level (0.80-0.99)")
    initial_stock_weeks: float = Field(3.0, ge=0.5, le=8.0, description="Baseline stock cover weeks (0.5-8.0)")


class RiskItem(BaseSchema):
    """Per-product evaluation item in POST /api/risk response."""
    product_id: str
    description: Optional[str] = None
    forecast_weekly: float
    forecast_p10: float
    forecast_p90: float
    forecast_lead_time: float
    on_hand: float
    on_hand_source: str
    safety_stock: float
    reorder_point: float
    days_of_cover: float
    stockout_probability: float
    on_order: float
    recommended_order_qty: float
    days_to_reorder: float
    revenue_at_risk: float
    risk_level: str
    action: str


class RiskSummary(BaseSchema):
    """Portfolio summary in POST /api/risk response (FR-R-6)."""

    n_high: int
    n_medium: int
    n_low: int
    n_order_now: int
    n_order_soon: int
    n_review_stop: int
    n_hold: int
    share_at_risk: float = Field(..., description="Fraction of products that are HIGH or MEDIUM")
    avg_days_of_cover: float = Field(..., description="Mean days of cover across products (999-capped excluded)")
    total_revenue_at_risk: float

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "n_high": 3,
                "n_medium": 12,
                "n_low": 85,
                "n_order_now": 5,
                "n_order_soon": 10,
                "n_review_stop": 2,
                "n_hold": 83,
                "share_at_risk": 0.15,
                "avg_days_of_cover": 18.4,
                "total_revenue_at_risk": 1234.56,
            }
        }
    )


class RiskResponse(BaseSchema):
    """POST /api/risk response (contract C7, FR-A-6)."""

    params: Dict[str, Any] = Field(..., description="Validated params echoed back (includes seed)")
    summary: RiskSummary
    items: List[RiskItem]

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "params": {"lead_time_weeks": 1, "service_level": 0.95, "initial_stock_weeks": 3.0, "seed": 42},
                "summary": {
                    "n_high": 3, "n_medium": 12, "n_low": 85,
                    "n_order_now": 5, "n_order_soon": 10, "n_review_stop": 2, "n_hold": 83,
                    "share_at_risk": 0.15, "avg_days_of_cover": 18.4, "total_revenue_at_risk": 1234.56,
                },
                "items": [],
            }
        }
    )


class SimulateChangedItem(BaseSchema):
    """One product whose risk class changed between baseline and simulated scenario."""

    product_id: str = Field(..., description="Product SKU")
    from_risk: str = Field(..., alias="from", description="Baseline risk class")
    to_risk: str = Field(..., alias="to", description="Simulated risk class")

    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {"product_id": "85123A", "from": "LOW", "to": "HIGH"}
        },
    )


class SimulateResponse(BaseSchema):
    """POST /api/simulate response (FR-R-12, contract C7)."""

    params: Dict[str, Any] = Field(
        ..., description="Echoed params including effective lead_time_weeks and delay_weeks"
    )
    summary: RiskSummary = Field(..., description="Risk summary under the simulated scenario")
    baseline_summary: RiskSummary = Field(..., description="Risk summary with delay_weeks=0 (baseline)")
    changed: List[SimulateChangedItem] = Field(
        default_factory=list,
        description="Products whose risk class changed from baseline to simulated scenario",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "params": {"lead_time_weeks": 3, "delay_weeks": 1, "service_level": 0.95,
                           "initial_stock_weeks": 3.0, "seed": 42},
                "summary": {"n_high": 10, "n_medium": 20, "n_low": 70,
                            "n_order_now": 12, "n_order_soon": 18, "n_review_stop": 2, "n_hold": 68,
                            "share_at_risk": 0.30, "avg_days_of_cover": 14.2, "total_revenue_at_risk": 5678.9},
                "baseline_summary": {"n_high": 3, "n_medium": 12, "n_low": 85,
                                     "n_order_now": 5, "n_order_soon": 10, "n_review_stop": 2, "n_hold": 83,
                                     "share_at_risk": 0.15, "avg_days_of_cover": 18.4,
                                     "total_revenue_at_risk": 1234.56},
                "changed": [{"product_id": "85123A", "from": "LOW", "to": "HIGH"}],
            }
        }
    )


class SimulateRequest(BaseSchema):
    """POST /api/simulate request (FR-R-12, contract C7)."""

    lead_time_weeks: int = Field(2, ge=1, le=4, description="Supplier lead time in weeks (1-4)")
    delay_weeks: int = Field(0, ge=0, le=2, description="Additional delay shock in weeks (0-2)")
    service_level: float = Field(0.95, ge=0.80, le=0.99, description="Target service level (0.80-0.99)")
    initial_stock_weeks: float = Field(3.0, ge=0.5, le=8.0, description="Starting stock in weeks of avg demand")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "lead_time_weeks": 2,
                "delay_weeks": 1,
                "service_level": 0.95,
                "initial_stock_weeks": 3.0,
            }
        }
    )


class SensitivityRequest(BaseSchema):
    """POST /api/sensitivity request (FR-R-5, contract C7)."""

    lead_times: List[int] = Field(
        default_factory=lambda: [1, 2, 3, 4],
        description="Lead time values to sweep (1-4, at most 8 values)",
    )
    service_levels: List[float] = Field(
        default_factory=lambda: [0.90, 0.95, 0.99],
        description="Service level values to sweep (0.80-0.99, at most 8 values)",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "lead_times": [1, 2, 3, 4],
                "service_levels": [0.90, 0.95, 0.99],
            }
        }
    )


class SensitivityRow(BaseSchema):
    """One row in POST /api/sensitivity response."""

    lead_time_weeks: int
    service_level: float
    n_high: int
    n_medium: int
    n_low: int

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "lead_time_weeks": 2,
                "service_level": 0.95,
                "n_high": 10,
                "n_medium": 25,
                "n_low": 65,
            }
        }
    )


# -----------------------------------------------------------------------------
# Inventory & Order Schemas
# -----------------------------------------------------------------------------

class InventoryCreateUpdateRequest(BaseSchema):
    """PUT /api/inventory/{product_id} body."""
    on_hand: float = Field(..., ge=0)
    lead_time_weeks: int = Field(..., ge=1, le=4)
    service_level: float = Field(..., ge=0.80, le=0.99)


class InventoryResponse(BaseSchema):
    """Inventory record response."""
    product_id: str
    on_hand: float
    lead_time_weeks: int
    service_level: float
    updated_at: Optional[str] = None


class OrderCreateRequest(BaseSchema):
    """POST /api/orders request."""
    product_id: str
    quantity: int = Field(..., gt=0)
    supplier_note: Optional[str] = None


class OrderUpdateRequest(BaseSchema):
    """PATCH /api/orders/{id} request."""
    status: Optional[str] = None
    quantity: Optional[int] = Field(None, gt=0)


class OrderResponse(BaseSchema):
    """Purchase order response."""
    id: int
    product_id: str
    quantity: int
    supplier_note: Optional[str] = None
    status: str
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class PlanGenerateResponse(BaseSchema):
    """POST /api/plans/generate response."""
    id: int
    week_start: str
    status: str
    orders: List[OrderResponse] = Field(default_factory=list)
