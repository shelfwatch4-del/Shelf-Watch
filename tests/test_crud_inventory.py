"""tests/test_crud_inventory.py - Unit tests for inventory CRUD service (Task B7)."""

from typing import Generator
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from api.crud_inventory import (
    NotFound,
    NotFoundError,
    ValidationError,
    delete_stock,
    get_stock,
    list_stock,
    upsert_stock,
)
from api.db import init_db
from api.models import InventoryLevel, Product


@pytest.fixture
def test_engine():
    """Create an isolated in-memory SQLite engine with foreign key enforcement."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    init_db(engine)
    return engine


@pytest.fixture
def test_session(test_engine) -> Generator[Session, None, None]:
    """Provide a clean session with known test products seeded."""
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)
    session = TestingSessionLocal()
    try:
        # Seed test catalog products
        session.add_all([
            Product(product_id="85123A", description="WHITE HANGING HEART T-LIGHT HOLDER"),
            Product(product_id="22423", description="REGENCY CAKESTAND 3 TIER"),
            Product(product_id="85099B", description="JUMBO BAG RED RETROSPOT"),
        ])
        session.commit()
        yield session
    finally:
        session.close()


def test_upsert_stock_create_then_read(test_session: Session):
    """Test creating a new stock record then reading it back."""
    data = {
        "on_hand": 120,
        "lead_time_weeks": 2,
        "service_level": 0.95,
    }
    record, created = upsert_stock(test_session, "85123A", data)
    assert created is True
    assert record.product_id == "85123A"
    assert record.on_hand == 120
    assert record.lead_time_weeks == 2
    assert float(record.service_level) == pytest.approx(0.95)

    # Read back
    retrieved = get_stock(test_session, "85123A")
    assert retrieved is not None
    assert retrieved.product_id == "85123A"
    assert retrieved.on_hand == 120
    assert retrieved.lead_time_weeks == 2
    assert float(retrieved.service_level) == pytest.approx(0.95)


def test_upsert_stock_replace_returns_created_false(test_session: Session):
    """Test updating existing stock record returns created=False and updates attributes."""
    initial_data = {
        "on_hand": 50,
        "lead_time_weeks": 1,
        "service_level": 0.90,
    }
    rec1, created1 = upsert_stock(test_session, "22423", initial_data)
    assert created1 is True
    assert rec1.on_hand == 50

    # Replace / update
    updated_data = {
        "on_hand": 75,
        "lead_time_weeks": 3,
        "service_level": 0.98,
    }
    rec2, created2 = upsert_stock(test_session, "22423", updated_data)
    assert created2 is False
    assert rec2.product_id == "22423"
    assert rec2.on_hand == 75
    assert rec2.lead_time_weeks == 3
    assert float(rec2.service_level) == pytest.approx(0.98)

    # Confirm persistence in DB
    refetched = get_stock(test_session, "22423")
    assert refetched.on_hand == 75
    assert refetched.lead_time_weeks == 3
    assert float(refetched.service_level) == pytest.approx(0.98)


def test_delete_stock_then_read_gives_not_found(test_session: Session):
    """Test deleting stock record removes it and subsequent read raises NotFound."""
    data = {"on_hand": 10, "lead_time_weeks": 1, "service_level": 0.95}
    upsert_stock(test_session, "85099B", data)

    # Delete
    delete_stock(test_session, "85099B")

    # Read back should raise NotFound (and NotFoundError)
    with pytest.raises(NotFound):
        get_stock(test_session, "85099B")

    with pytest.raises(NotFoundError):
        get_stock(test_session, "85099B")


def test_unknown_product_raises_not_found(test_session: Session):
    """Unknown product ids (not in our catalog) must raise NotFound for all operations."""
    unknown_sku = "UNKNOWN_SKU_999"

    # get_stock on unknown product
    with pytest.raises(NotFound):
        get_stock(test_session, unknown_sku)

    # upsert_stock on unknown product
    with pytest.raises(NotFound):
        upsert_stock(test_session, unknown_sku, {"on_hand": 10})

    # delete_stock on unknown product
    with pytest.raises(NotFound):
        delete_stock(test_session, unknown_sku)


def test_delete_nonexistent_stock_raises_not_found(test_session: Session):
    """Deleting stock for a known product with no inventory record raises NotFound."""
    # Product 85123A is in products, but has no inventory record yet
    with pytest.raises(NotFound):
        delete_stock(test_session, "85123A")


def test_validation_errors(test_session: Session):
    """Invalid input values must raise a validation error."""
    # on_hand < 0
    with pytest.raises(ValidationError):
        upsert_stock(test_session, "85123A", {"on_hand": -1, "lead_time_weeks": 1, "service_level": 0.95})

    # on_hand not integer
    with pytest.raises(ValidationError):
        upsert_stock(test_session, "85123A", {"on_hand": "10", "lead_time_weeks": 1, "service_level": 0.95})
    with pytest.raises(ValidationError):
        upsert_stock(test_session, "85123A", {"on_hand": 10.5, "lead_time_weeks": 1, "service_level": 0.95})
    with pytest.raises(ValidationError):
        upsert_stock(test_session, "85123A", {"on_hand": True, "lead_time_weeks": 1, "service_level": 0.95})

    # lead_time_weeks < 1
    with pytest.raises(ValidationError):
        upsert_stock(test_session, "85123A", {"on_hand": 10, "lead_time_weeks": 0, "service_level": 0.95})

    # lead_time_weeks > 6
    with pytest.raises(ValidationError):
        upsert_stock(test_session, "85123A", {"on_hand": 10, "lead_time_weeks": 7, "service_level": 0.95})

    # lead_time_weeks not integer
    with pytest.raises(ValidationError):
        upsert_stock(test_session, "85123A", {"on_hand": 10, "lead_time_weeks": 2.5, "service_level": 0.95})

    # service_level < 0.80
    with pytest.raises(ValidationError):
        upsert_stock(test_session, "85123A", {"on_hand": 10, "lead_time_weeks": 1, "service_level": 0.79})

    # service_level > 0.99
    with pytest.raises(ValidationError):
        upsert_stock(test_session, "85123A", {"on_hand": 10, "lead_time_weeks": 1, "service_level": 1.00})

    # service_level non-numeric
    with pytest.raises(ValidationError):
        upsert_stock(test_session, "85123A", {"on_hand": 10, "lead_time_weeks": 1, "service_level": "ninety-five"})


def test_list_stock(test_session: Session):
    """Test list_stock returns empty list initially, then all records ordered by product_id."""
    assert list_stock(test_session) == []

    # Insert two records
    upsert_stock(test_session, "85123A", {"on_hand": 100, "lead_time_weeks": 1, "service_level": 0.95})
    upsert_stock(test_session, "22423", {"on_hand": 50, "lead_time_weeks": 2, "service_level": 0.90})

    stocks = list_stock(test_session)
    assert len(stocks) == 2
    # Verify ordering by product_id
    assert [s.product_id for s in stocks] == ["22423", "85123A"]


def test_upsert_stock_object_input_and_defaults(test_session: Session):
    """Test upsert_stock accepts object attributes and applies default lead time and service level."""
    class StockPayload:
        def __init__(self, on_hand: int):
            self.on_hand = on_hand

    obj = StockPayload(on_hand=30)
    rec, created = upsert_stock(test_session, "85123A", obj)
    assert created is True
    assert rec.on_hand == 30
    assert rec.lead_time_weeks == 1  # Default
    assert float(rec.service_level) == pytest.approx(0.950)  # Default
