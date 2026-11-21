"""tests/test_crud_orders.py - Unit tests for purchase order CRUD service (Task B8)."""

from typing import Generator
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from api.crud_inventory import NotFound, ValidationError
from api.crud_orders import (
    Conflict,
    ConflictError,
    create_order,
    delete_order,
    get_order,
    list_orders,
    open_order_quantities,
    update_order,
)
from api.db import init_db
from api.models import InventoryLevel, Product, PurchaseOrder


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
        session.add_all([
            Product(product_id="85123A", description="WHITE HANGING HEART T-LIGHT HOLDER"),
            Product(product_id="22423", description="REGENCY CAKESTAND 3 TIER"),
            Product(product_id="85099B", description="JUMBO BAG RED RETROSPOT"),
        ])
        session.commit()
        yield session
    finally:
        session.close()


def test_create_order_pending_and_validations(test_session: Session):
    """Test newly created order defaults to Pending; validate positive integer quantity and catalog check."""
    # Successful creation
    order = create_order(
        test_session,
        product_id="85123A",
        quantity=50,
        supplier_note="Initial batch",
    )
    assert order.id is not None
    assert order.product_id == "85123A"
    assert order.quantity == 50
    assert order.status == "Pending"
    assert order.supplier_note == "Initial batch"

    # Reading it back
    fetched = get_order(test_session, order.id)
    assert fetched.id == order.id
    assert fetched.status == "Pending"

    # Unknown product raises NotFound
    with pytest.raises(NotFound):
        create_order(test_session, product_id="UNKNOWN_999", quantity=20)

    # Quantity <= 0 raises ValidationError
    with pytest.raises(ValidationError):
        create_order(test_session, product_id="85123A", quantity=0)

    with pytest.raises(ValidationError):
        create_order(test_session, product_id="85123A", quantity=-10)

    # Non-integer quantity raises ValidationError
    with pytest.raises(ValidationError):
        create_order(test_session, product_id="85123A", quantity="50")

    with pytest.raises(ValidationError):
        create_order(test_session, product_id="85123A", quantity=12.5)

    with pytest.raises(ValidationError):
        create_order(test_session, product_id="85123A", quantity=True)


def test_allowed_status_transitions(test_session: Session):
    """
    Test allowed transitions:
    1. Pending -> Approved -> Received (and stock crediting)
    2. Pending -> Cancelled
    3. Approved -> Cancelled
    """
    # 1. Pending -> Approved -> Received
    # First seed an inventory record with on_hand = 100
    test_session.add(
        InventoryLevel(product_id="85123A", on_hand=100, lead_time_weeks=1, service_level=0.950)
    )
    test_session.commit()

    order1 = create_order(test_session, product_id="85123A", quantity=40)
    assert order1.status == "Pending"

    # Pending -> Approved
    upd1 = update_order(test_session, order1.id, status="Approved")
    assert upd1.status == "Approved"

    # Approved -> Received
    upd2 = update_order(test_session, order1.id, status="Received")
    assert upd2.status == "Received"

    # Check stock incremented: 100 + 40 = 140
    stock1 = test_session.query(InventoryLevel).filter_by(product_id="85123A").first()
    assert stock1.on_hand == 140

    # 2. Pending -> Cancelled
    order2 = create_order(test_session, product_id="22423", quantity=25)
    upd_canc = update_order(test_session, order2.id, status="Cancelled")
    assert upd_canc.status == "Cancelled"

    # 3. Approved -> Cancelled
    order3 = create_order(test_session, product_id="85099B", quantity=60)
    update_order(test_session, order3.id, status="Approved")
    upd_app_canc = update_order(test_session, order3.id, status="Cancelled")
    assert upd_app_canc.status == "Cancelled"

    # 4. Receiving when no prior inventory record exists initializes stock
    order4 = create_order(test_session, product_id="22423", quantity=35)
    update_order(test_session, order4.id, status="Approved")
    update_order(test_session, order4.id, status="Received")
    stock2 = test_session.query(InventoryLevel).filter_by(product_id="22423").first()
    assert stock2 is not None
    assert stock2.on_hand == 35


def test_forbidden_status_transitions(test_session: Session):
    """Test every forbidden transition raises Conflict (409)."""
    # Pending -> Received (cannot jump directly without approval)
    o1 = create_order(test_session, product_id="85123A", quantity=10)
    with pytest.raises(Conflict):
        update_order(test_session, o1.id, status="Received")
    with pytest.raises(ConflictError):
        update_order(test_session, o1.id, status="Received")

    # Approved -> Pending (cannot rewind)
    o2 = create_order(test_session, product_id="85123A", quantity=10)
    update_order(test_session, o2.id, status="Approved")
    with pytest.raises(Conflict):
        update_order(test_session, o2.id, status="Pending")

    # Received -> Pending, Approved, Cancelled (terminal state)
    o3 = create_order(test_session, product_id="85123A", quantity=10)
    update_order(test_session, o3.id, status="Approved")
    update_order(test_session, o3.id, status="Received")
    for bad_status in ["Pending", "Approved", "Cancelled", "Unknown"]:
        with pytest.raises(Conflict):
            update_order(test_session, o3.id, status=bad_status)

    # Cancelled -> Pending, Approved, Received (terminal state)
    o4 = create_order(test_session, product_id="85123A", quantity=10)
    update_order(test_session, o4.id, status="Cancelled")
    for bad_status in ["Pending", "Approved", "Received", "Unknown"]:
        with pytest.raises(Conflict):
            update_order(test_session, o4.id, status=bad_status)


def test_quantity_update_rules(test_session: Session):
    """Quantity can be changed only while Pending; anything else raises Conflict."""
    order = create_order(test_session, product_id="85123A", quantity=15)

    # Updating while Pending succeeds
    updated = update_order(test_session, order.id, quantity=30)
    assert updated.quantity == 30

    # Invalid quantity raises ValidationError
    with pytest.raises(ValidationError):
        update_order(test_session, order.id, quantity=-5)
    with pytest.raises(ValidationError):
        update_order(test_session, order.id, quantity=0)

    # Move to Approved
    update_order(test_session, order.id, status="Approved")

    # Attempting to change quantity while Approved raises Conflict
    with pytest.raises(Conflict):
        update_order(test_session, order.id, quantity=45)

    # Move to Received
    update_order(test_session, order.id, status="Received")
    with pytest.raises(Conflict):
        update_order(test_session, order.id, quantity=50)

    # Cancelled order quantity change raises Conflict
    order_canc = create_order(test_session, product_id="85123A", quantity=20)
    update_order(test_session, order_canc.id, status="Cancelled")
    with pytest.raises(Conflict):
        update_order(test_session, order_canc.id, quantity=25)


def test_delete_rules(test_session: Session):
    """Delete allowed only when Pending or Cancelled, else Conflict."""
    # Delete Pending succeeds
    o_pend = create_order(test_session, product_id="85123A", quantity=10)
    delete_order(test_session, o_pend.id)
    with pytest.raises(NotFound):
        get_order(test_session, o_pend.id)

    # Delete Cancelled succeeds
    o_canc = create_order(test_session, product_id="85123A", quantity=10)
    update_order(test_session, o_canc.id, status="Cancelled")
    delete_order(test_session, o_canc.id)
    with pytest.raises(NotFound):
        get_order(test_session, o_canc.id)

    # Delete Approved raises Conflict
    o_app = create_order(test_session, product_id="85123A", quantity=10)
    update_order(test_session, o_app.id, status="Approved")
    with pytest.raises(Conflict):
        delete_order(test_session, o_app.id)

    # Delete Received raises Conflict
    o_rec = create_order(test_session, product_id="85123A", quantity=10)
    update_order(test_session, o_rec.id, status="Approved")
    update_order(test_session, o_rec.id, status="Received")
    with pytest.raises(Conflict):
        delete_order(test_session, o_rec.id)

    # Delete non-existent order raises NotFound
    with pytest.raises(NotFound):
        delete_order(test_session, 99999)


def test_list_orders_and_filtering(test_session: Session):
    """Test listing orders with optional status and product_id filtering."""
    assert list_orders(test_session) == []

    o1 = create_order(test_session, product_id="85123A", quantity=10)
    o2 = create_order(test_session, product_id="85123A", quantity=20)
    o3 = create_order(test_session, product_id="22423", quantity=30)

    update_order(test_session, o2.id, status="Approved")
    update_order(test_session, o3.id, status="Cancelled")

    # All orders
    all_orders = list_orders(test_session)
    assert len(all_orders) == 3

    # Filter by status
    pending = list_orders(test_session, status="Pending")
    assert len(pending) == 1
    assert pending[0].id == o1.id

    approved = list_orders(test_session, status="Approved")
    assert len(approved) == 1
    assert approved[0].id == o2.id

    # Filter by product_id
    p85123a = list_orders(test_session, product_id="85123A")
    assert len(p85123a) == 2

    # Filter by both
    filtered = list_orders(test_session, status="Approved", product_id="85123A")
    assert len(filtered) == 1
    assert filtered[0].id == o2.id


def test_open_order_quantities(test_session: Session):
    """
    open_order_quantities returns sum of Pending and Approved quantities per product,
    ignoring Received and Cancelled orders.
    """
    # Empty DB
    assert open_order_quantities(test_session) == {}

    # Seed orders:
    # 85123A: Pending(10), Approved(25), Received(50), Cancelled(100) -> open = 35
    # 22423: Pending(15), Approved(30) -> open = 45
    # 85099B: Received(70), Cancelled(40) -> open = 0 (omitted from dict)
    o1 = create_order(test_session, product_id="85123A", quantity=10)
    o2 = create_order(test_session, product_id="85123A", quantity=25)
    update_order(test_session, o2.id, status="Approved")

    o3 = create_order(test_session, product_id="85123A", quantity=50)
    update_order(test_session, o3.id, status="Approved")
    update_order(test_session, o3.id, status="Received")

    o4 = create_order(test_session, product_id="85123A", quantity=100)
    update_order(test_session, o4.id, status="Cancelled")

    o5 = create_order(test_session, product_id="22423", quantity=15)
    o6 = create_order(test_session, product_id="22423", quantity=30)
    update_order(test_session, o6.id, status="Approved")

    o7 = create_order(test_session, product_id="85099B", quantity=70)
    update_order(test_session, o7.id, status="Approved")
    update_order(test_session, o7.id, status="Received")

    o8 = create_order(test_session, product_id="85099B", quantity=40)
    update_order(test_session, o8.id, status="Cancelled")

    open_orders = open_order_quantities(test_session)
    assert open_orders == {
        "85123A": 35,
        "22423": 45,
    }
    assert "85099B" not in open_orders
