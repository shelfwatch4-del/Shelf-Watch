"""api/db.py - Database engine, session management, and CRUD helpers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Generator, List, Optional, Tuple

from dotenv import load_dotenv
from sqlalchemy import Engine, create_engine, inspect, text
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from api.models import (
    ApiLog,
    Base,
    InventoryLevel,
    PredictionLog,
    Product,
    PurchaseOrder,
    ReplenishmentPlan,
)

# Load environment variables from .env
load_dotenv()


def get_database_url(url: Optional[str] = None) -> str:
    """Resolve database URL with fallback to in-memory SQLite if none set."""
    if url:
        return url
    test_url = os.getenv("TEST_DATABASE_URL")
    if test_url:
        return test_url
    db_url = os.getenv("DATABASE_URL")
    if db_url:
        return db_url
    return "sqlite:///:memory:"


def create_db_engine(url: Optional[str] = None) -> Engine:
    """Create a SQLAlchemy engine configured per dialect and requirements."""
    resolved_url = get_database_url(url)

    if resolved_url.startswith("mysql"):
        return create_engine(
            resolved_url,
            pool_pre_ping=True,
            echo=False,
        )
    elif "sqlite" in resolved_url:
        # For in-memory SQLite, use StaticPool so all connections access the same memory DB
        is_memory = ":memory:" in resolved_url or resolved_url == "sqlite://"
        connect_args = {"check_same_thread": False}
        if is_memory:
            return create_engine(
                resolved_url,
                connect_args=connect_args,
                poolclass=StaticPool,
                echo=False,
            )
        return create_engine(
            resolved_url,
            connect_args=connect_args,
            echo=False,
        )
    else:
        return create_engine(resolved_url, pool_pre_ping=True, echo=False)


# Module-level engine and SessionLocal
engine = create_db_engine()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency yielding a database session that always closes."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db(
    engine_instance: Optional[Engine] = None,
    schema_path: str = "sql/schema.sql",
) -> None:
    """Initialize database tables from sql/schema.sql or Base metadata if missing."""
    target_engine = engine_instance or engine
    inspector = inspect(target_engine)
    existing_tables = set(inspector.get_table_names())

    # Check key required system and reference tables
    required_tables = {
        "products",
        "weekly_demand",
        "inventory_levels",
        "purchase_orders",
        "api_logs",
        "replenishment_plans",
        "prediction_log",
    }
    if required_tables.issubset(existing_tables):
        return

    # If dialect is MySQL, parse and execute sql/schema.sql
    if target_engine.dialect.name == "mysql":
        path = Path(schema_path)
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()

            with target_engine.connect() as conn:
                # Remove single-line comments and execute individual statements
                statements = []
                current_stmt: List[str] = []
                for line in content.splitlines():
                    stripped = line.strip()
                    if stripped.startswith("--") or not stripped:
                        continue
                    current_stmt.append(line)
                    if stripped.endswith(";"):
                        full_stmt = "\n".join(current_stmt).strip()
                        current_stmt = []
                        if full_stmt:
                            statements.append(full_stmt)

                for stmt in statements:
                    clean_stmt = stmt.rstrip(";").strip()
                    # Skip CREATE DATABASE / USE shelfwatch if already connected
                    upper = clean_stmt.upper()
                    if upper.startswith("CREATE DATABASE") or upper.startswith("USE "):
                        continue
                    if clean_stmt:
                        conn.execute(text(clean_stmt))
                conn.commit()
            return

    # For SQLite and any other dialect, use SQLAlchemy metadata create_all
    Base.metadata.create_all(bind=target_engine)


DEMO_PRODUCTS: List[Tuple[str, str]] = [
    ("85123A", "WHITE HANGING HEART T-LIGHT HOLDER"),
    ("22423", "REGENCY CAKESTAND 3 TIER"),
    ("85099B", "JUMBO BAG RED RETROSPOT"),
    ("84879", "ASSORTED COLOUR BIRD ORNAMENT"),
    ("47566", "PARTY BUNTING"),
]


def seed_demo_data(session_or_engine: Optional[Session | Engine] = None) -> None:
    """
    Seed 5 stock records and 4 purchase orders (one per status) ONLY when tables are empty.
    Ensures prerequisite product records exist to satisfy foreign keys.
    """
    own_session = False
    if isinstance(session_or_engine, Session):
        session = session_or_engine
    elif isinstance(session_or_engine, Engine):
        session = sessionmaker(bind=session_or_engine)()
        own_session = True
    else:
        session = SessionLocal()
        own_session = True

    try:
        # Check if inventory_levels or purchase_orders already have data
        inv_count = session.query(InventoryLevel).count()
        po_count = session.query(PurchaseOrder).count()

        if inv_count > 0 or po_count > 0:
            return

        # Ensure demo products exist
        for prod_id, desc in DEMO_PRODUCTS:
            existing = session.query(Product).filter_by(product_id=prod_id).first()
            if not existing:
                session.add(Product(product_id=prod_id, description=desc))
        session.flush()

        # Insert 5 stock records
        stock_records = [
            InventoryLevel(product_id="85123A", on_hand=150, lead_time_weeks=1, service_level=0.950),
            InventoryLevel(product_id="22423", on_hand=80, lead_time_weeks=2, service_level=0.950),
            InventoryLevel(product_id="85099B", on_hand=0, lead_time_weeks=1, service_level=0.990),
            InventoryLevel(product_id="84879", on_hand=220, lead_time_weeks=3, service_level=0.900),
            InventoryLevel(product_id="47566", on_hand=45, lead_time_weeks=1, service_level=0.950),
        ]
        session.add_all(stock_records)

        # Insert 4 purchase orders (one per status: Pending, Approved, Received, Cancelled)
        orders = [
            PurchaseOrder(product_id="85123A", quantity=100, supplier_note="Weekly restock", status="Pending"),
            PurchaseOrder(product_id="22423", quantity=50, supplier_note="Approved replenishment", status="Approved"),
            PurchaseOrder(product_id="85099B", quantity=200, supplier_note="Received order", status="Received"),
            PurchaseOrder(product_id="84879", quantity=75, supplier_note="Cancelled duplicate order", status="Cancelled"),
        ]
        session.add_all(orders)

        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        if own_session:
            session.close()


def write_log(
    db: Session,
    request_id: Optional[str],
    method: str,
    path: str,
    status: int,
    duration_ms: float,
) -> ApiLog:
    """Insert an ApiLog row using SQLAlchemy session and bound parameters."""
    log_entry = ApiLog(
        request_id=request_id,
        method=method,
        path=path,
        status=status,
        duration_ms=duration_ms,
    )
    db.add(log_entry)
    db.commit()
    db.refresh(log_entry)
    return log_entry
