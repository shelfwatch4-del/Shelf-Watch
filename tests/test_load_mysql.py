"""tests/test_load_mysql.py - Tests for scripts/load_mysql.py and scripts/reset_demo.py."""

from pathlib import Path
import pytest
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from api.models import (
    ApiLog,
    InventoryLevel,
    PurchaseOrder,
    ReplenishmentPlan,
)
from scripts.load_mysql import REFERENCE_TABLES, load_mysql
from scripts.reset_demo import SYSTEM_TABLES, reset_demo


@pytest.fixture
def sqlite_engine():
    """Create an isolated in-memory SQLite database engine."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    return engine


def test_load_mysql_populates_all_tables(sqlite_engine):
    """Verify load_mysql populates all reference tables from data files."""
    counts = load_mysql(
        engine_instance=sqlite_engine,
        data_dir_path="data/processed",
        mock_dir_path="data/mock",
        raw_path_str="data/raw/Online Retail.parquet",
    )

    # Verify all 21 reference tables are present and non-empty
    for tbl in REFERENCE_TABLES:
        assert tbl in counts, f"Missing table {tbl} in load_mysql counts"
        assert counts[tbl] > 0, f"Table {tbl} has 0 rows"

    # Verify FR-DS-2: transactions_top100 has NO CustomerID column, has has_customer flag
    inspector = inspect(sqlite_engine)
    tx_cols = [c["name"] for c in inspector.get_columns("transactions_top100")]
    assert "CustomerID" not in tx_cols
    assert "customer_id" not in tx_cols
    assert "has_customer" in tx_cols
    assert "is_cancelled" in tx_cols
    assert "quantity_capped" in tx_cols

    # Verify FR-DS-3: data_sources has primary, supporting, calendar
    with sqlite_engine.connect() as conn:
        sources = conn.execute(text("SELECT name, role FROM data_sources ORDER BY id;")).fetchall()
        assert len(sources) == 3
        roles = {s[1] for s in sources}
        assert roles == {"primary", "supporting", "calendar"}

        # Verify pipeline_log has cleaning funnel steps with descriptive notes
        logs = conn.execute(text("SELECT step, rows_remaining, note FROM pipeline_log;")).fetchall()
        assert len(logs) >= 8
        for l in logs:
            assert l[2] is not None and len(str(l[2])) > 0, f"Missing note for step {l[0]}"


def test_foreign_key_integrity(sqlite_engine):
    """Verify all child tables reference valid product_ids in products table."""
    load_mysql(
        engine_instance=sqlite_engine,
        data_dir_path="data/processed",
        mock_dir_path="data/mock",
        raw_path_str="data/raw/Online Retail.parquet",
    )

    child_tables = [
        "weekly_demand",
        "next_week_forecast",
        "product_intelligence",
        "enrichment_weekly",
        "transactions_top100",
    ]

    with sqlite_engine.connect() as conn:
        for tbl in child_tables:
            query = f"""
                SELECT DISTINCT c.product_id
                FROM {tbl} c
                LEFT JOIN products p ON c.product_id = p.product_id
                WHERE p.product_id IS NULL;
            """
            orphans = conn.execute(text(query)).fetchall()
            assert len(orphans) == 0, f"Found orphan product_ids in {tbl}: {orphans}"


def test_load_mysql_idempotent(sqlite_engine):
    """Verify load_mysql is idempotent: second run gives identical row counts."""
    counts_first = load_mysql(
        engine_instance=sqlite_engine,
        data_dir_path="data/processed",
        mock_dir_path="data/mock",
        raw_path_str="data/raw/Online Retail.parquet",
    )

    counts_second = load_mysql(
        engine_instance=sqlite_engine,
        data_dir_path="data/processed",
        mock_dir_path="data/mock",
        raw_path_str="data/raw/Online Retail.parquet",
    )

    assert counts_first == counts_second, "Second load_mysql run produced differing row counts"


def test_reset_demo_clears_and_reseeds(sqlite_engine):
    """Verify reset_demo clears system tables and restores exactly the 5 stock and 4 PO rows."""
    # First load base tables (products needed for foreign keys)
    load_mysql(
        engine_instance=sqlite_engine,
        data_dir_path="data/processed",
        mock_dir_path="data/mock",
        raw_path_str="data/raw/Online Retail.parquet",
    )

    SessionLocal = sessionmaker(bind=sqlite_engine)
    session = SessionLocal()

    # Add extra dummy rows to test that reset clears them
    import datetime
    extra_plan = ReplenishmentPlan(week_start=datetime.date(2021, 1, 1), status="Draft", note="Test plan")
    session.add(extra_plan)
    session.commit()

    extra_log = ApiLog(request_id="test-clear", method="GET", path="/api/health", status=200, duration_ms=10.0)
    session.add(extra_log)
    session.commit()
    session.close()

    # Run reset_demo
    counts = reset_demo(sqlite_engine)

    assert counts["inventory_levels"] == 5
    assert counts["purchase_orders"] == 4
    assert counts["replenishment_plans"] == 0
    assert counts["prediction_log"] == 0
    assert counts["api_logs"] == 0

    # Verify order statuses
    with sqlite_engine.connect() as conn:
        statuses = conn.execute(text("SELECT status FROM purchase_orders;")).fetchall()
        status_set = {s[0] for s in statuses}
        assert status_set == {"Pending", "Approved", "Received", "Cancelled"}
