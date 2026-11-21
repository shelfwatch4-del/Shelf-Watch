"""scripts/reset_demo.py - Reset system tables to seeded demo state.

SRS: 7.3, FR-DB-11.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional

# Ensure project root is in sys.path when invoked directly
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from sqlalchemy import Engine, inspect, text

from api.db import create_db_engine, init_db, seed_demo_data

SYSTEM_TABLES = [
    "purchase_orders",
    "replenishment_plans",
    "inventory_levels",
    "prediction_log",
    "api_logs",
]


def reset_demo(engine_instance: Optional[Engine] = None) -> Dict[str, int]:
    """
    Clear all application system tables and reseed demo records.
    Returns row counts for each system table.
    """
    engine = engine_instance or create_db_engine()
    init_db(engine)

    is_mysql = engine.dialect.name == "mysql"
    with engine.begin() as conn:
        if is_mysql:
            conn.execute(text("SET FOREIGN_KEY_CHECKS = 0;"))
            for tbl in SYSTEM_TABLES:
                conn.execute(text(f"TRUNCATE TABLE {tbl};"))
            conn.execute(text("SET FOREIGN_KEY_CHECKS = 1;"))
        else:
            for tbl in SYSTEM_TABLES:
                conn.execute(text(f"DELETE FROM {tbl};"))

    # Reseed demo data (5 stock records, 4 POs covering all 4 statuses)
    seed_demo_data(engine)

    # Inspect resulting row counts
    counts: Dict[str, int] = {}
    with engine.connect() as conn:
        for tbl in reversed(SYSTEM_TABLES):
            res = conn.execute(text(f"SELECT COUNT(*) FROM {tbl};")).scalar()
            counts[tbl] = int(res) if res is not None else 0

    return counts


def print_system_table_counts(counts: Dict[str, int]) -> None:
    """Print formatted summary table of system table row counts."""
    print("\n" + "=" * 45)
    print(f"{'System Table':<28} | {'Row Count':>12}")
    print("=" * 45)
    for tbl, cnt in counts.items():
        print(f"{tbl:<28} | {cnt:>12}")
    print("=" * 45 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Reset Shelfwatch system tables to demo state.")
    parser.add_argument("--url", type=str, default=None, help="Database connection URL override.")

    args = parser.parse_args()
    engine = create_db_engine(args.url)
    print(f"Resetting system tables on {engine.url.drivername} database...")
    counts = reset_demo(engine)
    print_system_table_counts(counts)
    print("System tables successfully reset to seeded demo state.")


if __name__ == "__main__":
    main()
