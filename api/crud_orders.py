"""api/crud_orders.py - Purchase-order CRUD service with status rules (Task B8).

SRS: 4.7, 7.3, FR-DB-4, FR-DB-5, FR-W-1, FR-W-2.
CONTRACTS: C7 (/api/orders).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from sqlalchemy import func
from sqlalchemy.orm import Session

from api.crud_inventory import NotFound, NotFoundError, ValidationError
from api.models import InventoryLevel, Product, PurchaseOrder


class ConflictError(Exception):
    """Raised when an operation violates order lifecycle or status transition rules."""
    pass


# Alias for convenience and direct matching with task specifications
Conflict = ConflictError

ALLOWED_STATUS_TRANSITIONS: Dict[str, set[str]] = {
    "Pending": {"Approved", "Cancelled"},
    "Approved": {"Received", "Cancelled"},
}


def _validate_quantity(quantity: Any) -> int:
    """Validate that quantity is a strictly positive integer (> 0)."""
    if quantity is None:
        raise ValidationError("Field 'quantity' is required.")
    if isinstance(quantity, bool) or not isinstance(quantity, int):
        raise ValidationError(
            f"'quantity' must be an integer, got {type(quantity).__name__}."
        )
    if quantity <= 0:
        raise ValidationError(f"'quantity' must be a positive integer (> 0), got {quantity}.")
    return quantity


def create_order(
    db: Session,
    product_id: Union[str, dict, Any],
    quantity: Optional[int] = None,
    supplier_note: Optional[str] = None,
    plan_id: Optional[int] = None,
) -> PurchaseOrder:
    """
    Create a new purchase order in 'Pending' status.

    Args:
        db: Active SQLAlchemy database session.
        product_id: Product SKU identifier or dictionary/object with fields.
        quantity: Order quantity (> 0).
        supplier_note: Optional note for supplier.
        plan_id: Optional replenishment plan id.

    Returns:
        Newly created PurchaseOrder record.

    Raises:
        NotFound: If product_id does not exist in the product catalog.
        ValidationError: If quantity is not a positive integer.
    """
    if isinstance(product_id, dict):
        data = product_id
        target_prod_id = data.get("product_id", "")
        qty_val = data.get("quantity")
        note_val = data.get("supplier_note", supplier_note)
        p_id = data.get("plan_id", plan_id)
    elif not isinstance(product_id, str) and hasattr(product_id, "product_id"):
        target_prod_id = getattr(product_id, "product_id")
        qty_val = getattr(product_id, "quantity", quantity)
        note_val = getattr(product_id, "supplier_note", supplier_note)
        p_id = getattr(product_id, "plan_id", plan_id)
    else:
        target_prod_id = str(product_id)
        qty_val = quantity
        note_val = supplier_note
        p_id = plan_id

    # Verify product exists in catalog
    product = db.query(Product).filter(Product.product_id == target_prod_id).first()
    if product is None:
        raise NotFound(f"Product '{target_prod_id}' not found in catalog.")

    valid_qty = _validate_quantity(qty_val)

    order = PurchaseOrder(
        product_id=target_prod_id,
        quantity=valid_qty,
        supplier_note=note_val,
        plan_id=p_id,
        status="Pending",
    )
    db.add(order)
    db.commit()
    db.refresh(order)
    return order


def list_orders(
    db: Session,
    status: Optional[str] = None,
    product_id: Optional[str] = None,
) -> List[PurchaseOrder]:
    """
    List purchase orders with optional filtering by status and product_id.

    Args:
        db: Active SQLAlchemy database session.
        status: Optional status filter ('Pending', 'Approved', 'Received', 'Cancelled').
        product_id: Optional product SKU filter.

    Returns:
        List of matching PurchaseOrder records ordered by id.
    """
    query = db.query(PurchaseOrder)
    if status is not None:
        query = query.filter(PurchaseOrder.status == status)
    if product_id is not None:
        query = query.filter(PurchaseOrder.product_id == product_id)
    return query.order_by(PurchaseOrder.id.asc()).all()


def get_order(db: Session, order_id: int) -> PurchaseOrder:
    """
    Retrieve a purchase order by its ID.

    Args:
        db: Active SQLAlchemy database session.
        order_id: Purchase order ID.

    Returns:
        PurchaseOrder record.

    Raises:
        NotFound: If order does not exist.
    """
    order = db.query(PurchaseOrder).filter(PurchaseOrder.id == order_id).first()
    if order is None:
        raise NotFound(f"Order #{order_id} not found.")
    return order


def update_order(
    db: Session,
    order_id: int,
    data: Optional[Union[dict, Any]] = None,
    *,
    status: Optional[str] = None,
    quantity: Optional[int] = None,
    supplier_note: Optional[str] = None,
) -> PurchaseOrder:
    """
    Update purchase order status, quantity, or supplier note.

    Rules:
    - Status transitions allowed:
      - Pending -> Approved
      - Pending -> Cancelled
      - Approved -> Received
      - Approved -> Cancelled
      Any other transition raises Conflict (409).
    - Quantity can be changed only while in 'Pending' status (raises Conflict otherwise).
    - Transitioning to 'Received' increments on_hand in inventory_levels.

    Args:
        db: Active SQLAlchemy database session.
        order_id: Purchase order ID.
        data: Optional payload dict or object.
        status: Target status string.
        quantity: New quantity (positive integer).
        supplier_note: Updated supplier note.

    Returns:
        Updated PurchaseOrder record.

    Raises:
        NotFound: If order does not exist.
        Conflict: If an invalid transition or modification is attempted.
        ValidationError: If quantity is invalid.
    """
    order = get_order(db, order_id)

    target_status: Optional[str] = status
    target_quantity: Optional[int] = quantity
    target_note: Optional[str] = supplier_note

    if isinstance(data, dict):
        if "status" in data:
            target_status = data["status"]
        if "quantity" in data:
            target_quantity = data["quantity"]
        if "supplier_note" in data:
            target_note = data["supplier_note"]
    elif isinstance(data, str) and target_status is None:
        target_status = data
    elif isinstance(data, int) and target_quantity is None:
        target_quantity = data
    elif data is not None and not isinstance(data, (str, int)):
        if hasattr(data, "status"):
            target_status = getattr(data, "status")
        if hasattr(data, "quantity"):
            target_quantity = getattr(data, "quantity")
        if hasattr(data, "supplier_note"):
            target_note = getattr(data, "supplier_note")

    # Handle quantity update
    if target_quantity is not None:
        valid_qty = _validate_quantity(target_quantity)
        if order.status != "Pending":
            raise Conflict(
                f"Cannot update quantity for order #{order_id} in '{order.status}' status. "
                "Quantity can be changed only while Pending."
            )
        order.quantity = valid_qty

    # Handle status update
    # A same-status "transition" (e.g. Pending -> Pending) is not in the table, so it is a 409.
    if target_status is not None:
        allowed = ALLOWED_STATUS_TRANSITIONS.get(order.status, set())
        if target_status not in allowed:
            raise Conflict(
                f"Invalid order status transition from '{order.status}' to '{target_status}'. "
                f"Allowed transitions for '{order.status}': {sorted(list(allowed))}."
            )

        # Transitioning to Received credits inventory
        if target_status == "Received":
            stock = (
                db.query(InventoryLevel)
                .filter(InventoryLevel.product_id == order.product_id)
                .first()
            )
            if stock is not None:
                stock.on_hand += order.quantity
            else:
                stock = InventoryLevel(
                    product_id=order.product_id,
                    on_hand=order.quantity,
                    lead_time_weeks=1,
                    service_level=0.950,
                )
                db.add(stock)

        order.status = target_status

    if target_note is not None:
        order.supplier_note = target_note

    db.commit()
    db.refresh(order)
    return order


def delete_order(db: Session, order_id: int) -> None:
    """
    Delete a purchase order.

    Rules:
    - Allowed only when status is 'Pending' or 'Cancelled'.
    - Deleting 'Approved' or 'Received' orders raises Conflict (409).

    Args:
        db: Active SQLAlchemy database session.
        order_id: Purchase order ID.

    Raises:
        NotFound: If order does not exist.
        Conflict: If order is in 'Approved' or 'Received' status.
    """
    order = get_order(db, order_id)
    if order.status not in ("Pending", "Cancelled"):
        raise Conflict(
            f"Cannot delete order #{order_id} with status '{order.status}'. "
            "Orders can be deleted only when Pending or Cancelled."
        )

    db.delete(order)
    db.commit()


def open_order_quantities(db: Session) -> Dict[str, int]:
    """
    Calculate total open order quantities per product (Pending and Approved).

    Ignores Received and Cancelled orders. Used as 'on_order' by the risk engine.

    Args:
        db: Active SQLAlchemy database session.

    Returns:
        Dictionary mapping product_id to sum of open order quantities.
    """
    results = (
        db.query(
            PurchaseOrder.product_id,
            func.sum(PurchaseOrder.quantity).label("total_qty"),
        )
        .filter(PurchaseOrder.status.in_(["Pending", "Approved"]))
        .group_by(PurchaseOrder.product_id)
        .all()
    )
    return {str(prod_id): int(total_qty) for prod_id, total_qty in results}
