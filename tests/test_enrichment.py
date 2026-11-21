"""Unit and contract tests for enrichment tables (C8, C9, C10).

Covers:
- Task E1: Calendar table tests (contract C9)
- Task E2: Prior-year table tests with synthetic sheet (contract C10)
- Task E3: Transaction-derived signals table tests (contract C8)
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.enrichment import (
    CALENDAR_OUTPUT_PATH,
    ENRICHMENT_OUTPUT_PATH,
    PRIOR_YEAR_OUTPUT_PATH,
    WEEKLY_DEMAND_PATH,
    build_calendar_weekly,
    build_enrichment_weekly,
    build_prior_year_weekly,
    get_uk_england_holidays,
)


# ---------------------------------------------------------------------------
# Task E1 Tests: Calendar Table (Contract C9)
# ---------------------------------------------------------------------------


def test_calendar_holiday_dates() -> None:
    """Verify that 2011-04-22, 2011-04-25, 2011-04-29, and 2011-05-02 are accounted for."""
    uk_holidays = get_uk_england_holidays(years=[2011])
    required_dates = [
        pd.Timestamp("2011-04-22").date(),  # Good Friday
        pd.Timestamp("2011-04-25").date(),  # Easter Monday
        pd.Timestamp("2011-04-29").date(),  # Royal Wedding
        pd.Timestamp("2011-05-02").date(),  # Early May bank holiday
    ]
    for d in required_dates:
        assert d in uk_holidays, f"Expected holiday date {d} not found in UK England holidays"


def test_calendar_trading_days_and_saturday(tmp_path: Path) -> None:
    """Verify trading_days_expected is lower in holiday weeks, no Saturday counted, and no NaNs."""
    out_csv = tmp_path / "calendar_weekly.csv"
    df = build_calendar_weekly(out_path=out_csv)

    # Verify no missing values
    assert not df.isna().any().any(), "Calendar table contains missing values"

    # Week count verification and mathematical justification:
    # From 2009-12-07 to 2011-12-05 inclusive:
    # 728 days = exactly 104 intervals, meaning 105 Monday-start weeks.
    n_weeks = len(df)
    print(f"\nExact number of calendar weeks: {n_weeks} (justified: 104 full 7-day intervals + 1 start week = 105 weeks)")
    assert n_weeks == 105, f"Expected 105 weeks, got {n_weeks}"

    # Trading days check: normal week has 6 trading days (Mon, Tue, Wed, Thu, Fri, Sun; Saturday is 0)
    # Week 2011-04-25 has Easter Monday (2011-04-25) and Royal Wedding (2011-04-29) -> 6 - 2 = 4 trading days
    # Week 2011-05-02 has May Day (2011-05-02) -> 6 - 1 = 5 trading days
    row_apr25 = df[df["week_start"] == "2011-04-25"].iloc[0]
    row_may02 = df[df["week_start"] == "2011-05-02"].iloc[0]
    row_normal = df[df["week_start"] == "2011-06-06"].iloc[0]

    assert row_normal["trading_days_expected"] == 6, "Normal week should have 6 expected trading days"
    assert row_apr25["trading_days_expected"] < row_normal["trading_days_expected"], (
        "2011-04-25 week trading days should be lower than normal week"
    )
    assert row_may02["trading_days_expected"] < row_normal["trading_days_expected"], (
        "2011-05-02 week trading days should be lower than normal week"
    )
    assert row_apr25["trading_days_expected"] == 4, f"Expected 4 trading days, got {row_apr25['trading_days_expected']}"
    assert row_may02["trading_days_expected"] == 5, f"Expected 5 trading days, got {row_may02['trading_days_expected']}"

    # Verify maximum trading days never exceeds 6 (proving Saturday is never counted)
    assert (df["trading_days_expected"] <= 6).all(), "Trading days expected must never exceed 6"


# ---------------------------------------------------------------------------
# Task E2 Tests: Prior-Year Table (Contract C10)
# ---------------------------------------------------------------------------


def test_prior_year_with_synthetic_sheet(tmp_path: Path) -> None:
    """Test prior-year building with a synthetic second-year sheet."""
    # Create synthetic Online Retail II DataFrame
    dates = pd.date_range("2009-12-01", "2010-12-08", freq="D")
    
    rows = []
    # Product P1: first sale on 2010-01-04
    rows.append({
        "Invoice": "1001",
        "StockCode": "P1",
        "Description": "Product One",
        "Quantity": 500,  # exceeds cap of 480
        "InvoiceDate": pd.Timestamp("2010-01-04 10:00:00"),
        "Price": 2.50,
        "Customer ID": 12345.0,
        "Country": "United Kingdom",
    })
    rows.append({
        "Invoice": "1002",
        "StockCode": "P1",
        "Description": "Product One",
        "Quantity": 10,
        "InvoiceDate": pd.Timestamp("2010-01-11 11:00:00"),
        "Price": 2.50,
        "Customer ID": 12345.0,
        "Country": "United Kingdom",
    })
    # Overlapping sale on or after 2010-12-06 (should be excluded)
    rows.append({
        "Invoice": "1003",
        "StockCode": "P1",
        "Description": "Product One",
        "Quantity": 20,
        "InvoiceDate": pd.Timestamp("2010-12-06 12:00:00"),
        "Price": 2.50,
        "Customer ID": 12345.0,
        "Country": "United Kingdom",
    })
    # Cancellation row (should be dropped by clean_transactions)
    rows.append({
        "Invoice": "C1004",
        "StockCode": "P1",
        "Description": "Product One",
        "Quantity": -5,
        "InvoiceDate": pd.Timestamp("2010-01-04 14:00:00"),
        "Price": 2.50,
        "Customer ID": 12345.0,
        "Country": "United Kingdom",
    })
    # Non-product stockcode (should be dropped)
    rows.append({
        "Invoice": "1005",
        "StockCode": "POST",
        "Description": "Postage",
        "Quantity": 1,
        "InvoiceDate": pd.Timestamp("2010-01-04 15:00:00"),
        "Price": 5.00,
        "Customer ID": 12345.0,
        "Country": "United Kingdom",
    })
    # Product P2: no sales anywhere
    # (P2 is in target products, but not in rows)

    synthetic_df = pd.DataFrame(rows)
    synthetic_parquet = tmp_path / "synthetic_ret2.parquet"
    synthetic_df.to_parquet(synthetic_parquet, index=False)

    out_parquet = tmp_path / "prior_year_weekly.parquet"
    result = build_prior_year_weekly(
        ret2_path=synthetic_parquet,
        cap=480.0,
        products=["P1", "P2"],
        out_path=out_parquet,
    )

    # 1. Product P2 with no sales has NO rows
    assert "P2" not in result["product_id"].values, "Product with no prior-year sales must have no rows"

    # 2. Product P1 should have rows
    p1_rows = result[result["product_id"] == "P1"].sort_values("week_start")
    assert not p1_rows.empty

    # 3. No rows before product's first sale (2010-01-04)
    assert p1_rows["week_start"].min() == pd.Timestamp("2010-01-04"), (
        "No rows should exist before product's first sale"
    )

    # 4. Cap applied: 500 capped at 480
    week_first = p1_rows[p1_rows["week_start"] == pd.Timestamp("2010-01-04")].iloc[0]
    assert week_first["units"] == 480, f"Quantity should be capped at 480, got {week_first['units']}"

    # 5. Weeks on or after 2010-12-06 excluded
    assert (p1_rows["week_start"] < pd.Timestamp("2010-12-06")).all(), (
        "All weeks must be strictly before 2010-12-06"
    )

    # 6. Closure week 2009-12-28 excluded
    assert not (p1_rows["week_start"] == pd.Timestamp("2009-12-28")).any(), (
        "Closure week 2009-12-28 must have no rows"
    )


def test_prior_year_missing_file_handling(tmp_path: Path) -> None:
    """Verify FR-X-4: missing file writes empty C10 table and does not crash."""
    non_existent = tmp_path / "non_existent.parquet"
    out_parquet = tmp_path / "out_empty.parquet"

    df = build_prior_year_weekly(
        ret2_path=non_existent,
        out_path=out_parquet,
    )

    assert df.empty, "DataFrame should be empty when file is missing"
    assert out_parquet.exists(), "Output parquet file should still be written"
    saved = pd.read_parquet(out_parquet)
    assert list(saved.columns) == ["product_id", "week_start", "units", "orders"]


# ---------------------------------------------------------------------------
# Task E3 Tests: Transaction-Derived Signals (Contract C8)
# ---------------------------------------------------------------------------


def test_enrichment_weekly_signals() -> None:
    """Test C8 enrichment_weekly: grid matches C1, gross_units == C1 units, C1 hash unchanged."""
    assert WEEKLY_DEMAND_PATH.exists(), f"Prerequisite {WEEKLY_DEMAND_PATH} not found"

    # Compute hash of C1 before
    demand_bytes_before = open(WEEKLY_DEMAND_PATH, "rb").read()
    hash_before = hashlib.sha256(demand_bytes_before).hexdigest()

    demand_df = pd.read_parquet(WEEKLY_DEMAND_PATH)

    # Build C8 table
    df = build_enrichment_weekly()

    # 1. Grid matches C1 exactly
    assert len(df) == len(demand_df), f"Row count mismatch: {len(df)} vs {len(demand_df)}"
    assert (df["product_id"] == demand_df["product_id"]).all(), "product_id column does not match C1"
    assert (df["week_start"] == demand_df["week_start"]).all(), "week_start column does not match C1"

    # 2. gross_units equals C1 units
    assert (df["gross_units"] == demand_df["units"]).all(), "gross_units must equal C1 units"

    # 3. export_units and guest_units never exceed gross_units
    assert (df["export_units"] <= df["gross_units"]).all(), "export_units exceeds gross_units"
    assert (df["guest_units"] <= df["gross_units"]).all(), "guest_units exceeds gross_units"

    # 4. cancelled_units never negative
    assert (df["cancelled_units"] >= 0).all(), "cancelled_units must be non-negative"

    # 5. C1 itself is unchanged
    demand_bytes_after = open(WEEKLY_DEMAND_PATH, "rb").read()
    hash_after = hashlib.sha256(demand_bytes_after).hexdigest()
    assert hash_before == hash_after, f"C1 hash changed! Before: {hash_before}, After: {hash_after}"
