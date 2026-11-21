"""api/crud_plans.py - Weekly replenishment plan workflow (Task B19).

SRS: FR-W-1 to FR-W-6, 7.3.  CONTRACTS_ADDENDUM: /api/plans.

A Draft plan holds proposed lines (plan_lines). Approving it creates one Pending
purchase order per line, linked to the plan, in a single transaction. Cancelling
creates none. All functions here commit once at the end, so a failure leaves nothing behind.
"""

from __future__ import annotations

import csv
import io
import math
from datetime import date
from typing import Any, Dict, List, Mapping, Optional

from sqlalchemy.orm import Session

from api.crud_inventory import NotFound
from api.crud_orders import ConflictError
from api.models import PlanLine, PurchaseOrder, ReplenishmentPlan

PLAN_ACTIONS = ("ORDER_NOW", "ORDER_SOON")
EXPORT_COLUMNS = ["product_id", "description", "quantity", "status"]


def build_lines(risk_rows: List[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """One line per ORDER_NOW / ORDER_SOON product with recommended_order_qty > 0.

    Quantities are rounded up to whole units (purchase orders hold integers).
    """
    lines: List[Dict[str, Any]] = []
    for row in risk_rows:
        if row.get("action") not in PLAN_ACTIONS:
            continue
        qty = row.get("recommended_order_qty")
        if qty is None or not math.isfinite(float(qty)) or float(qty) <= 0:
            continue
        lines.append({"product_id": str(row["product_id"]), "quantity": int(math.ceil(float(qty)))})
    return lines


def create_draft_plan(db: Session, week_start: date, lines: List[Dict[str, Any]]) -> ReplenishmentPlan:
    """Create a Draft plan with its lines. Raises ConflictError if a Draft exists for the week."""
    existing = (
        db.query(ReplenishmentPlan)
        .filter(ReplenishmentPlan.week_start == week_start, ReplenishmentPlan.status == "Draft")
        .first()
    )
    if existing is not None:
        raise ConflictError(
            f"A Draft plan (#{existing.id}) already exists for week {week_start.isoformat()}."
        )
    plan = ReplenishmentPlan(week_start=week_start, status="Draft")
    plan.lines = [PlanLine(product_id=l["product_id"], quantity=l["quantity"]) for l in lines]
    db.add(plan)
    db.commit()
    db.refresh(plan)
    return plan


def list_plans(db: Session) -> List[ReplenishmentPlan]:
    return db.query(ReplenishmentPlan).order_by(ReplenishmentPlan.id.desc()).all()


def get_plan(db: Session, plan_id: int) -> ReplenishmentPlan:
    plan = db.query(ReplenishmentPlan).filter(ReplenishmentPlan.id == plan_id).first()
    if plan is None:
        raise NotFound(f"Plan #{plan_id} not found.")
    return plan


def update_plan_status(db: Session, plan_id: int, status: str) -> ReplenishmentPlan:
    """Draft -> Approved (creates Pending orders) or Draft -> Cancelled (creates none)."""
    from datetime import datetime

    plan = get_plan(db, plan_id)
    if plan.status != "Draft" or status not in ("Approved", "Cancelled"):
        raise ConflictError(
            f"Invalid plan status transition from '{plan.status}' to '{status}'. "
            "Only a Draft plan can become Approved or Cancelled."
        )
    try:
        if status == "Approved":
            for line in plan.lines:
                db.add(
                    PurchaseOrder(
                        plan_id=plan.id,
                        product_id=line.product_id,
                        quantity=line.quantity,
                        status="Pending",
                    )
                )
            plan.approved_at = datetime.utcnow()
        plan.status = status
        db.commit()
    except Exception:
        db.rollback()
        raise
    db.refresh(plan)
    return plan


def plan_order_rows(plan: ReplenishmentPlan, descriptions: Mapping[str, Optional[str]]) -> List[Dict[str, Any]]:
    """Order lines of a plan: real purchase orders once Approved, else the proposed lines."""
    if plan.status == "Approved":
        return [
            {
                "id": o.id,
                "product_id": o.product_id,
                "description": descriptions.get(o.product_id),
                "quantity": o.quantity,
                "status": o.status,
            }
            for o in sorted(plan.orders, key=lambda o: o.id)
        ]
    return [
        {
            "id": None,
            "product_id": l.product_id,
            "description": descriptions.get(l.product_id),
            "quantity": l.quantity,
            "status": plan.status,
        }
        for l in plan.lines
    ]


def rows_to_csv(rows: List[Dict[str, Any]]) -> str:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=EXPORT_COLUMNS, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    for r in rows:
        writer.writerow({**r, "description": r.get("description") or ""})
    return buf.getvalue()
