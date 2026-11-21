"""api/crud_inventory.py - Inventory CRUD service for Shelfwatch (Task B7).

SRS: 4.7, 7.3, FR-DB-4, FR-DB-5.
CONTRACTS: C7 (/api/inventory).
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any, List, Tuple, Union

from sqlalchemy.orm import Session

from api.models import InventoryLevel, Product


class NotFoundError(Exception):
    """Raised when a product or an inventory record is not found."""
    pass


# Alias for convenience and direct matching with task specifications
NotFound = NotFoundError


class ValidationError(ValueError):
    """Raised when input parameters fail range or type validation."""
    pass


def _validate_inventory_data(
    data: Union[dict, Any],
) -> Tuple[int, int, float]:
    """
    Extract and validate on_hand, lead_time_weeks, and service_level.

    Valid ranges:
    - on_hand: integer >= 0
    - lead_time_weeks: integer between 1 and 6 (inclusive)
    - service_level: numeric between 0.80 and 0.99 (inclusive)

    Raises:
        ValidationError: If any value is missing or outside valid bounds.
    """
    if isinstance(data, dict):
        on_hand_raw = data.get("on_hand")
        lead_time_raw = data.get("lead_time_weeks", 1)
        service_level_raw = data.get("service_level", 0.950)
    else:
        on_hand_raw = getattr(data, "on_hand", None)
        lead_time_raw = getattr(data, "lead_time_weeks", 1)
        service_level_raw = getattr(data, "service_level", 0.950)

    # Validate on_hand
    if on_hand_raw is None:
        raise ValidationError("Field 'on_hand' is required.")
    if isinstance(on_hand_raw, bool) or not isinstance(on_hand_raw, int):
        raise ValidationError(f"'on_hand' must be an integer, got {type(on_hand_raw).__name__}.")
    if on_hand_raw < 0:
        raise ValidationError(f"'on_hand' must be non-negative (>= 0), got {on_hand_raw}.")

    # Validate lead_time_weeks (default 1 if None)
    if lead_time_raw is None:
        lead_time_raw = 1
    if isinstance(lead_time_raw, bool) or not isinstance(lead_time_raw, int):
        raise ValidationError(
            f"'lead_time_weeks' must be an integer, got {type(lead_time_raw).__name__}."
        )
    if lead_time_raw < 1 or lead_time_raw > 6:
        raise ValidationError(
            f"'lead_time_weeks' must be between 1 and 6, got {lead_time_raw}."
        )

    # Validate service_level (default 0.950 if None)
    if service_level_raw is None:
        service_level_raw = 0.950
    if isinstance(service_level_raw, bool) or not isinstance(
        service_level_raw, (int, float, Decimal)
    ):
        raise ValidationError(
            f"'service_level' must be numeric, got {type(service_level_raw).__name__}."
        )
    try:
        sl_float = float(service_level_raw)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"Invalid 'service_level' value: {service_level_raw}") from exc

    # Floating point precision safe check
    sl_rounded = round(sl_float, 4)
    if sl_rounded < 0.80 or sl_rounded > 0.99:
        raise ValidationError(
            f"'service_level' must be between 0.80 and 0.99, got {sl_float}."
        )

    return on_hand_raw, lead_time_raw, sl_float


def list_stock(db: Session) -> List[InventoryLevel]:
    """
    List all recorded stock levels ordered by product_id.

    Args:
        db: Active SQLAlchemy database session.

    Returns:
        List of InventoryLevel records.
    """
    return db.query(InventoryLevel).order_by(InventoryLevel.product_id).all()


def get_stock(db: Session, product_id: str) -> InventoryLevel:
    """
    Retrieve inventory level for a specified product.

    Args:
        db: Active SQLAlchemy database session.
        product_id: Product SKU identifier.

    Returns:
        InventoryLevel record.

    Raises:
        NotFound: If the product does not exist in catalog or has no stock record.
    """
    # Verify product exists in catalog (top 100 products)
    product = db.query(Product).filter(Product.product_id == product_id).first()
    if product is None:
        raise NotFound(f"Product '{product_id}' not found in catalog.")

    stock = db.query(InventoryLevel).filter(InventoryLevel.product_id == product_id).first()
    if stock is None:
        raise NotFound(f"Inventory record for product '{product_id}' not found.")

    return stock


def upsert_stock(
    db: Session,
    product_id: str,
    data: Union[dict, Any],
) -> Tuple[InventoryLevel, bool]:
    """
    Create or update inventory level for a specified product.

    Args:
        db: Active SQLAlchemy database session.
        product_id: Product SKU identifier.
        data: Dictionary or object containing on_hand, lead_time_weeks, service_level.

    Returns:
        Tuple of (InventoryLevel record, created boolean).
        created is True if a new row was inserted, False if an existing row was replaced.

    Raises:
        NotFound: If the product does not exist in catalog.
        ValidationError: If input parameters violate range or type rules.
    """
    # Verify product exists in catalog
    product = db.query(Product).filter(Product.product_id == product_id).first()
    if product is None:
        raise NotFound(f"Product '{product_id}' not found in catalog.")

    # Validate parameters
    on_hand, lead_time_weeks, service_level = _validate_inventory_data(data)

    stock = db.query(InventoryLevel).filter(InventoryLevel.product_id == product_id).first()
    if stock is None:
        stock = InventoryLevel(
            product_id=product_id,
            on_hand=on_hand,
            lead_time_weeks=lead_time_weeks,
            service_level=service_level,
        )
        db.add(stock)
        db.commit()
        db.refresh(stock)
        return stock, True

    # Existing record: update fields
    stock.on_hand = on_hand
    stock.lead_time_weeks = lead_time_weeks
    stock.service_level = service_level
    db.commit()
    db.refresh(stock)
    return stock, False


def delete_stock(db: Session, product_id: str) -> None:
    """
    Delete inventory level record for a specified product.

    Args:
        db: Active SQLAlchemy database session.
        product_id: Product SKU identifier.

    Raises:
        NotFound: If the product does not exist in catalog or has no stock record.
    """
    product = db.query(Product).filter(Product.product_id == product_id).first()
    if product is None:
        raise NotFound(f"Product '{product_id}' not found in catalog.")

    stock = db.query(InventoryLevel).filter(InventoryLevel.product_id == product_id).first()
    if stock is None:
        raise NotFound(f"Inventory record for product '{product_id}' not found.")

    db.delete(stock)
    db.commit()
