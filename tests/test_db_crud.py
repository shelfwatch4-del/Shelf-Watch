"""tests/test_db_crud.py - Unit tests for database layer (Task B6)."""

from typing import Generator
import pytest
from sqlalchemy import create_engine, inspect
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from api.db import create_db_engine, get_db, init_db, seed_demo_data, write_log
from api.models import (
    ApiLog,
    Base,
    InventoryLevel,
    PredictionLog,
    Product,
    PurchaseOrder,
    ReplenishmentPlan,
)


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
    """Provide a clean session connected to the test database."""
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()


def test_init_db_creates_all_tables(test_engine):
    """Verify init_db creates all expected system and reference tables."""
    inspector = inspect(test_engine)
    tables = set(inspector.get_table_names())
    expected_tables = {
        "products",
        "inventory_levels",
        "replenishment_plans",
        "purchase_orders",
        "prediction_log",
        "api_logs",
    }
    assert expected_tables.issubset(tables), f"Missing tables: {expected_tables - tables}"


def test_seed_demo_data_inserts_when_empty(test_session):
    """Verify seed_demo_data inserts exactly 5 stock records and 4 purchase orders."""
    assert test_session.query(InventoryLevel).count() == 0
    assert test_session.query(PurchaseOrder).count() == 0

    seed_demo_data(test_session)

    # Check inventory records
    inventory_records = test_session.query(InventoryLevel).all()
    assert len(inventory_records) == 5

    # Check purchase orders
    orders = test_session.query(PurchaseOrder).all()
    assert len(orders) == 4

    # Verify one order per status: Pending, Approved, Received, Cancelled
    statuses = {order.status for order in orders}
    assert statuses == {"Pending", "Approved", "Received", "Cancelled"}

    # Verify products were seeded
    products = test_session.query(Product).all()
    assert len(products) == 5


def test_seed_demo_data_idempotent_when_not_empty(test_session):
    """Verify seed_demo_data does not re-insert rows when tables are not empty."""
    seed_demo_data(test_session)
    assert test_session.query(InventoryLevel).count() == 5
    assert test_session.query(PurchaseOrder).count() == 4

    # Second call should do nothing
    seed_demo_data(test_session)
    assert test_session.query(InventoryLevel).count() == 5
    assert test_session.query(PurchaseOrder).count() == 4


def test_inventory_level_on_hand_non_negative_constraint(test_session):
    """Verify check constraint on_hand >= 0 is enforced."""
    # Ensure product exists
    test_session.add(Product(product_id="TEST01", description="Test item"))
    test_session.commit()

    # on_hand = -1 should fail
    invalid_stock = InventoryLevel(
        product_id="TEST01",
        on_hand=-1,
        lead_time_weeks=1,
        service_level=0.950,
    )
    test_session.add(invalid_stock)
    with pytest.raises(IntegrityError):
        test_session.commit()
    test_session.rollback()


def test_inventory_level_lead_time_range_constraint(test_session):
    """Verify check constraint lead_time_weeks between 1 and 6 is enforced."""
    test_session.add(Product(product_id="TEST02", description="Test item"))
    test_session.commit()

    # lead_time_weeks = 0 should fail
    invalid_lead_low = InventoryLevel(
        product_id="TEST02",
        on_hand=10,
        lead_time_weeks=0,
        service_level=0.950,
    )
    test_session.add(invalid_lead_low)
    with pytest.raises(IntegrityError):
        test_session.commit()
    test_session.rollback()

    # lead_time_weeks = 7 should fail
    invalid_lead_high = InventoryLevel(
        product_id="TEST02",
        on_hand=10,
        lead_time_weeks=7,
        service_level=0.950,
    )
    test_session.add(invalid_lead_high)
    with pytest.raises(IntegrityError):
        test_session.commit()
    test_session.rollback()


def test_inventory_level_service_level_range_constraint(test_session):
    """Verify check constraint service_level between 0.80 and 0.99 is enforced."""
    test_session.add(Product(product_id="TEST03", description="Test item"))
    test_session.commit()

    # service_level = 0.79 should fail
    invalid_sl_low = InventoryLevel(
        product_id="TEST03",
        on_hand=10,
        lead_time_weeks=1,
        service_level=0.79,
    )
    test_session.add(invalid_sl_low)
    with pytest.raises(IntegrityError):
        test_session.commit()
    test_session.rollback()

    # service_level = 1.00 should fail
    invalid_sl_high = InventoryLevel(
        product_id="TEST03",
        on_hand=10,
        lead_time_weeks=1,
        service_level=1.00,
    )
    test_session.add(invalid_sl_high)
    with pytest.raises(IntegrityError):
        test_session.commit()
    test_session.rollback()


def test_purchase_order_quantity_positive_constraint(test_session):
    """Verify check constraint quantity > 0 is enforced."""
    test_session.add(Product(product_id="TEST04", description="Test item"))
    test_session.commit()

    # quantity = 0 should fail
    invalid_order_zero = PurchaseOrder(
        product_id="TEST04",
        quantity=0,
        supplier_note="Test note",
        status="Pending",
    )
    test_session.add(invalid_order_zero)
    with pytest.raises(IntegrityError):
        test_session.commit()
    test_session.rollback()

    # quantity = -5 should fail
    invalid_order_neg = PurchaseOrder(
        product_id="TEST04",
        quantity=-5,
        supplier_note="Test note",
        status="Pending",
    )
    test_session.add(invalid_order_neg)
    with pytest.raises(IntegrityError):
        test_session.commit()
    test_session.rollback()


def test_write_log_inserts_row(test_session):
    """Verify write_log correctly inserts and returns an ApiLog entry."""
    log = write_log(
        db=test_session,
        request_id="req-test-abc-123",
        method="POST",
        path="/api/predict",
        status=200,
        duration_ms=42.75,
    )

    assert log.id is not None
    assert log.request_id == "req-test-abc-123"
    assert log.method == "POST"
    assert log.path == "/api/predict"
    assert log.status == 200
    assert float(log.duration_ms) == pytest.approx(42.75)
    assert log.created_at is not None

    # Query from database to confirm persistence
    saved_log = test_session.query(ApiLog).filter_by(request_id="req-test-abc-123").first()
    assert saved_log is not None
    assert saved_log.path == "/api/predict"


def test_get_db_session_lifecycle():
    """Verify get_db yields an active session and closes it."""
    db_gen = get_db()
    session = next(db_gen)
    assert isinstance(session, Session)
    assert session.is_active

    # Closing generator should trigger finally block
    with pytest.raises(StopIteration):
        next(db_gen)


# ---------------------------------------------------------------------------
# Task B15: CRUD services against the real schema (transitions table, stock credit)
# ---------------------------------------------------------------------------

from itertools import product as _cross

from api.crud_inventory import NotFound, ValidationError, delete_stock, list_stock, upsert_stock
from api.crud_orders import (
    ConflictError,
    create_order,
    delete_order,
    open_order_quantities,
    update_order,
)

_STATUSES = ["Pending", "Approved", "Received", "Cancelled"]
_ALLOWED = {
    ("Pending", "Approved"),
    ("Pending", "Cancelled"),
    ("Approved", "Received"),
    ("Approved", "Cancelled"),
}
_PATH = {
    "Pending": [],
    "Approved": ["Approved"],
    "Received": ["Approved", "Received"],
    "Cancelled": ["Cancelled"],
}


@pytest.fixture
def crud_session(test_session):
    test_session.add(Product(product_id="P1", description="Widget"))
    test_session.commit()
    return test_session


@pytest.mark.parametrize("src,dst", list(_cross(_STATUSES, _STATUSES)))
def test_order_transitions_table(crud_session, src, dst):
    order = create_order(crud_session, "P1", 10)
    for step in _PATH[src]:
        update_order(crud_session, order.id, status=step)

    if (src, dst) in _ALLOWED:
        assert update_order(crud_session, order.id, status=dst).status == dst
    else:
        with pytest.raises(ConflictError):
            update_order(crud_session, order.id, status=dst)
        crud_session.refresh(order)
        assert order.status == src


def test_only_receiving_changes_stock(crud_session):
    upsert_stock(crud_session, "P1", {"on_hand": 100, "lead_time_weeks": 1, "service_level": 0.95})
    o = create_order(crud_session, "P1", 30)
    update_order(crud_session, o.id, status="Approved")
    assert list_stock(crud_session)[0].on_hand == 100  # approving does not credit
    update_order(crud_session, o.id, status="Received")
    assert list_stock(crud_session)[0].on_hand == 130

    with pytest.raises(ConflictError):  # a second receive must not double-credit
        update_order(crud_session, o.id, status="Received")
    assert list_stock(crud_session)[0].on_hand == 130


def test_rejected_receive_leaves_stock_untouched(crud_session):
    upsert_stock(crud_session, "P1", {"on_hand": 5, "lead_time_weeks": 1, "service_level": 0.95})
    o = create_order(crud_session, "P1", 30)
    with pytest.raises(ConflictError):  # Pending -> Received is not allowed
        update_order(crud_session, o.id, status="Received")
    assert list_stock(crud_session)[0].on_hand == 5


def test_open_orders_count_pending_and_approved_only(crud_session):
    ids = {s: create_order(crud_session, "P1", 10).id for s in _STATUSES}
    for s in ("Approved", "Received"):
        update_order(crud_session, ids[s], status="Approved")
    update_order(crud_session, ids["Received"], status="Received")
    update_order(crud_session, ids["Cancelled"], status="Cancelled")
    assert open_order_quantities(crud_session) == {"P1": 20}  # Pending + Approved


@pytest.mark.parametrize("status,deletable", [("Pending", True), ("Cancelled", True), ("Approved", False), ("Received", False)])
def test_order_delete_rules(crud_session, status, deletable):
    o = create_order(crud_session, "P1", 10)
    for step in _PATH[status]:
        update_order(crud_session, o.id, status=step)
    if deletable:
        delete_order(crud_session, o.id)
        assert crud_session.query(PurchaseOrder).filter_by(id=o.id).count() == 0
    else:
        with pytest.raises(ConflictError):
            delete_order(crud_session, o.id)
        assert crud_session.query(PurchaseOrder).filter_by(id=o.id).count() == 1


def test_stock_upsert_created_flag_and_delete(crud_session):
    row, created = upsert_stock(crud_session, "P1", {"on_hand": 1, "lead_time_weeks": 1, "service_level": 0.9})
    assert created is True
    row, created = upsert_stock(crud_session, "P1", {"on_hand": 2, "lead_time_weeks": 3, "service_level": 0.9})
    assert created is False and row.on_hand == 2 and row.lead_time_weeks == 3
    assert crud_session.query(InventoryLevel).count() == 1  # replaced, not duplicated

    delete_stock(crud_session, "P1")
    with pytest.raises(NotFound):
        delete_stock(crud_session, "P1")


def test_invalid_stock_never_reaches_the_table(crud_session):
    with pytest.raises(ValidationError):
        upsert_stock(crud_session, "P1", {"on_hand": -1, "lead_time_weeks": 1, "service_level": 0.9})
    with pytest.raises(NotFound):
        upsert_stock(crud_session, "ZZ", {"on_hand": 1, "lead_time_weeks": 1, "service_level": 0.9})
    assert crud_session.query(InventoryLevel).count() == 0
