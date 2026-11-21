"""api/models.py - SQLAlchemy ORM models matching SRS 4.7, 7.3 and sql/schema.sql."""

from __future__ import annotations

from typing import Any, Optional

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    Column,
    Date,
    DateTime,
    Double,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    SmallInteger,
    String,
    func,
)
from sqlalchemy.orm import DeclarativeBase, relationship
from sqlalchemy.types import JSON


class Base(DeclarativeBase):
    """SQLAlchemy declarative base class for Shelfwatch."""
    pass


# ---------------------------------------------------------------- lineage


class DataSource(Base):
    """Data source lineage metadata (SRS 4.7)."""

    __tablename__ = "data_sources"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(120), nullable=False)
    role = Column(String(30), nullable=False)  # primary, supporting, calendar
    source_url = Column(String(255), nullable=True)
    rows_raw = Column(Integer, nullable=True)
    date_min = Column(Date, nullable=True)
    date_max = Column(Date, nullable=True)
    license_note = Column(String(255), nullable=True)


class PipelineLog(Base):
    """Data cleaning funnel log (SRS 4.7)."""

    __tablename__ = "pipeline_log"

    step_no = Column(Integer, primary_key=True)
    step = Column(String(255), nullable=False)
    rows_remaining = Column(Integer, nullable=True)
    note = Column(String(255), nullable=True)


# ---------------------------------------------------------------- demand data


class Product(Base):
    """Reference table for top-100 products (sql/schema.sql)."""

    __tablename__ = "products"

    product_id = Column(String(20), primary_key=True)
    description = Column(String(255), nullable=True)
    total_units_train = Column(Integer, nullable=True)
    rank_train = Column(Integer, nullable=True)

    # Relationships to system tables
    inventory_level = relationship("InventoryLevel", back_populates="product", uselist=False)
    purchase_orders = relationship("PurchaseOrder", back_populates="product")


class WeeklyDemand(Base):
    """Weekly aggregated demand per product (contract C1)."""

    __tablename__ = "weekly_demand"

    product_id = Column(String(20), ForeignKey("products.product_id"), primary_key=True)
    week_start = Column(Date, primary_key=True)
    units = Column(Integer, nullable=False)
    orders = Column(Integer, nullable=False)
    customers = Column(Integer, nullable=False)
    avg_price = Column(Numeric(10, 4), nullable=False)


class ClosedWeek(Base):
    """Closed weeks with no shop activity (contract C1b)."""

    __tablename__ = "closed_weeks"

    week_start = Column(Date, primary_key=True)


class CalendarWeekly(Base):
    """Weekly calendar features including UK bank holidays (contract C9)."""

    __tablename__ = "calendar_weekly"

    week_start = Column(Date, primary_key=True)
    uk_bank_holiday_count = Column(SmallInteger().with_variant(Integer, "sqlite"), nullable=False)
    trading_days_expected = Column(SmallInteger().with_variant(Integer, "sqlite"), nullable=False)
    is_christmas_week = Column(SmallInteger().with_variant(Integer, "sqlite"), nullable=False)


class PriorYearWeekly(Base):
    """Prior year weekly sales (contract C10)."""

    __tablename__ = "prior_year_weekly"

    product_id = Column(String(20), ForeignKey("products.product_id"), primary_key=True)
    week_start = Column(Date, primary_key=True)
    units = Column(Integer, nullable=False)
    orders = Column(Integer, nullable=False)


class EnrichmentWeekly(Base):
    """Weekly transaction-derived enrichment signals (contract C8)."""

    __tablename__ = "enrichment_weekly"

    product_id = Column(String(20), ForeignKey("products.product_id"), primary_key=True)
    week_start = Column(Date, primary_key=True)
    gross_units = Column(Integer, nullable=False)
    cancelled_units = Column(Integer, nullable=False)
    export_units = Column(Integer, nullable=False)
    guest_units = Column(Integer, nullable=False)


class TransactionsTop100(Base):
    """Invoice lines for the top 100 products without CustomerID (SRS 4.7)."""

    __tablename__ = "transactions_top100"

    line_id = Column(BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True)
    invoice_no = Column(String(20), nullable=False)
    product_id = Column(String(20), ForeignKey("products.product_id"), nullable=False)
    description = Column(String(255), nullable=True)
    quantity = Column(Integer, nullable=False)
    quantity_capped = Column(Integer, nullable=False)
    invoice_date = Column(DateTime, nullable=False)
    week_start = Column(Date, nullable=False)
    unit_price = Column(Numeric(10, 4), nullable=False)
    country = Column(String(60), nullable=True)
    has_customer = Column(SmallInteger().with_variant(Integer, "sqlite"), nullable=False)
    is_cancelled = Column(SmallInteger().with_variant(Integer, "sqlite"), nullable=False)

    __table_args__ = (
        Index("idx_tx_product_week", "product_id", "week_start"),
    )


# ---------------------------------------------------------------- modelling results


class ModelRegistry(Base):
    """Model registry tracking registered models (SRS 4.7)."""

    __tablename__ = "model_registry"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_name = Column(String(60), nullable=False)
    target_mode = Column(String(30), nullable=True)
    feature_groups = Column(String(100), nullable=True)
    trained_on = Column(String(60), nullable=True)
    file_path = Column(String(255), nullable=True)
    library_versions = Column(JSON, nullable=True)
    is_active = Column(SmallInteger().with_variant(Integer, "sqlite"), nullable=False, default=0)
    created_at = Column(DateTime, server_default=func.now())


class Prediction(Base):
    """Model predictions comparison table (contract C2)."""

    __tablename__ = "predictions"

    product_id = Column(String(20), ForeignKey("products.product_id"), primary_key=True)
    week_start = Column(Date, primary_key=True)
    model_name = Column(String(60), primary_key=True)
    split_name = Column(String(10), primary_key=True)
    actual = Column(Float, nullable=True)
    predicted = Column(Float, nullable=True)
    forecast_p10 = Column(Float, nullable=True)
    forecast_p90 = Column(Float, nullable=True)


class Metric(Base):
    """Model evaluation metrics (contract C3)."""

    __tablename__ = "metrics"

    model_name = Column(String(60), primary_key=True)
    split_name = Column(String(10), primary_key=True)
    mae = Column(Float, nullable=True)
    rmse = Column(Float, nullable=True)
    wape = Column(Float, nullable=True)
    bias = Column(Float, nullable=True)


class TuningTrial(Base):
    """Hyperparameter tuning trials (SRS 4.7)."""

    __tablename__ = "tuning_trials"

    id = Column(Integer, primary_key=True, autoincrement=True)
    candidate = Column(String(80), nullable=False)
    trial_no = Column(Integer, nullable=False)
    params = Column(JSON, nullable=True)
    fold_scores = Column(JSON, nullable=True)
    mean_val_wape = Column(Float, nullable=True)
    is_best = Column(SmallInteger().with_variant(Integer, "sqlite"), nullable=False, default=0)

    __table_args__ = (
        Index("idx_trials_candidate", "candidate"),
    )


class Ablation(Base):
    """Feature group ablation results (SRS 4.7)."""

    __tablename__ = "ablation"

    feature_groups = Column(String(100), primary_key=True)
    model_name = Column(String(60), primary_key=True)
    val_wape = Column(Float, nullable=True)
    val_bias = Column(Float, nullable=True)
    note = Column(String(255), nullable=True)


class FeatureImportance(Base):
    """Feature importance and mean absolute contributions (SRS 4.7)."""

    __tablename__ = "feature_importance"

    model_name = Column(String(60), primary_key=True)
    feature = Column(String(80), primary_key=True)
    feature_group = Column(String(20), nullable=True)
    gain_importance = Column(Float, nullable=True)
    mean_abs_contribution = Column(Float, nullable=True)


class NextWeekForecast(Base):
    """Next week forecast for week 2011-12-05 (contract C4)."""

    __tablename__ = "next_week_forecast"

    product_id = Column(String(20), ForeignKey("products.product_id"), primary_key=True)
    week_start = Column(Date, nullable=False)
    forecast_weekly = Column(Float, nullable=False)
    forecast_p10 = Column(Float, nullable=True)
    forecast_p90 = Column(Float, nullable=True)
    sigma_error = Column(Float, nullable=True)
    avg_weekly_units_train = Column(Float, nullable=True)
    last_avg_price = Column(Numeric(10, 4), nullable=True)


class ErrorAnalysis(Base):
    """Error analysis across segments, weeks, and products (SRS 4.7)."""

    __tablename__ = "error_analysis"

    level_name = Column(String(20), primary_key=True)  # segment, week, product
    key_name = Column(String(80), primary_key=True)
    model_name = Column(String(60), primary_key=True)
    n = Column(Integer, nullable=True)
    mae = Column(Float, nullable=True)
    rmse = Column(Float, nullable=True)
    wape = Column(Float, nullable=True)
    bias = Column(Float, nullable=True)
    over_forecast_units = Column(Float, nullable=True)
    under_forecast_units = Column(Float, nullable=True)


# ---------------------------------------------------------------- intelligence and backtest


class ProductIntelligence(Base):
    """Product intelligence metrics and categorization (contract C11)."""

    __tablename__ = "product_intelligence"

    product_id = Column(String(20), ForeignKey("products.product_id"), primary_key=True)
    description = Column(String(255), nullable=True)
    adi = Column(Float, nullable=True)
    cv2 = Column(Float, nullable=True)
    pattern_class = Column(String(20), nullable=True)
    abc_class = Column(String(1), nullable=True)
    xyz_class = Column(String(1), nullable=True)
    lifecycle_stage = Column(String(20), nullable=True)
    weeks_since_last_sale = Column(Integer, nullable=True)
    return_rate = Column(Float, nullable=True)
    export_share = Column(Float, nullable=True)
    top3_customer_share = Column(Float, nullable=True)
    known_customer_coverage = Column(Float, nullable=True)
    revenue_share = Column(Float, nullable=True)


class BasketPair(Base):
    """Market basket analysis companion pairs (contract C12)."""

    __tablename__ = "basket_pairs"

    product_id = Column(String(20), primary_key=True)
    companion_id = Column(String(20), primary_key=True)
    companion_description = Column(String(255), nullable=True)
    n_invoices = Column(Integer, nullable=False)
    confidence = Column(Float, nullable=True)
    lift = Column(Float, nullable=True)


class PolicyBacktest(Base):
    """Inventory policy backtest summary (contract C14)."""

    __tablename__ = "policy_backtest"

    policy = Column(String(30), primary_key=True)
    lead_time_weeks = Column(SmallInteger().with_variant(Integer, "sqlite"), primary_key=True)
    abc_class = Column(String(5), primary_key=True)  # A, B, C, ALL
    fill_rate = Column(Float, nullable=True)
    stockout_weeks = Column(Integer, nullable=True)
    avg_on_hand = Column(Float, nullable=True)
    unmet_units = Column(Float, nullable=True)


class PolicyBacktestWeekly(Base):
    """Week-by-week replay of inventory policy backtest (contract C15)."""

    __tablename__ = "policy_backtest_weekly"

    id = Column(BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True)
    policy = Column(String(30), nullable=False)
    lead_time_weeks = Column(SmallInteger().with_variant(Integer, "sqlite"), nullable=False)
    product_id = Column(String(20), nullable=False)
    week_start = Column(Date, nullable=False)
    on_hand_start = Column(Float, nullable=True)
    order_qty = Column(Float, nullable=True)
    arrived_qty = Column(Float, nullable=True)
    demand = Column(Float, nullable=True)
    sold = Column(Float, nullable=True)
    unmet = Column(Float, nullable=True)
    on_hand_end = Column(Float, nullable=True)

    __table_args__ = (
        Index("idx_pbw_product_policy", "product_id", "policy"),
    )


# ---------------------------------------------------------------- system tables (written by the API)


class InventoryLevel(Base):
    """System table: InventoryLevel (SRS 7.3)."""

    __tablename__ = "inventory_levels"

    product_id = Column(String(20), ForeignKey("products.product_id"), primary_key=True)
    on_hand = Column(Integer, nullable=False)
    lead_time_weeks = Column(
        SmallInteger().with_variant(Integer, "sqlite"),
        nullable=False,
        default=1,
    )
    service_level = Column(Numeric(4, 3), nullable=False, default=0.950)
    updated_at = Column(
        DateTime,
        server_default=func.now(),
        onupdate=func.now(),
    )

    product = relationship("Product", back_populates="inventory_level")

    __table_args__ = (
        CheckConstraint("on_hand >= 0", name="chk_on_hand"),
        CheckConstraint("lead_time_weeks BETWEEN 1 AND 6", name="chk_lead"),
        CheckConstraint("service_level BETWEEN 0.80 AND 0.99", name="chk_service"),
    )


class ReplenishmentPlan(Base):
    """System table: replenishment_plans (SRS 7.3)."""

    __tablename__ = "replenishment_plans"

    id = Column(Integer, primary_key=True, autoincrement=True)
    week_start = Column(Date, nullable=False)
    status = Column(
        Enum("Draft", "Approved", "Cancelled", name="plan_status"),
        nullable=False,
        default="Draft",
    )
    note = Column(String(255), nullable=True)
    created_at = Column(DateTime, server_default=func.now())
    approved_at = Column(DateTime, nullable=True)

    orders = relationship("PurchaseOrder", back_populates="plan")


class PurchaseOrder(Base):
    """System table: purchase_orders (SRS 7.3)."""

    __tablename__ = "purchase_orders"

    id = Column(Integer, primary_key=True, autoincrement=True)
    plan_id = Column(Integer, ForeignKey("replenishment_plans.id"), nullable=True)
    product_id = Column(String(20), ForeignKey("products.product_id"), nullable=False)
    quantity = Column(Integer, nullable=False)
    supplier_note = Column(String(255), nullable=True)
    status = Column(
        Enum("Pending", "Approved", "Received", "Cancelled", name="order_status"),
        nullable=False,
        default="Pending",
    )
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    plan = relationship("ReplenishmentPlan", back_populates="orders")
    product = relationship("Product", back_populates="purchase_orders")

    __table_args__ = (
        CheckConstraint("quantity > 0", name="chk_qty"),
    )


class PredictionLog(Base):
    """System table: prediction_log (SRS 7.3)."""

    __tablename__ = "prediction_log"

    id = Column(BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True)
    request_id = Column(String(40), nullable=True)
    endpoint = Column(String(40), nullable=True)
    product_id = Column(String(20), nullable=True)
    inputs = Column(JSON, nullable=True)
    outputs = Column(JSON, nullable=True)
    model_name = Column(String(60), nullable=True)
    created_at = Column(DateTime, server_default=func.now())


class ApiLog(Base):
    """System table: api_logs (SRS 7.3)."""

    __tablename__ = "api_logs"

    id = Column(BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True)
    request_id = Column(String(40), nullable=True)
    method = Column(String(10), nullable=True)
    path = Column(String(255), nullable=True)
    status = Column(SmallInteger().with_variant(Integer, "sqlite"), nullable=True)
    duration_ms = Column(Numeric(10, 2), nullable=True)
    created_at = Column(DateTime, server_default=func.now())
