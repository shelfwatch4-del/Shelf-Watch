"""Enrichment tables for the Shelfwatch pipeline.

Implements:
- Task E1: Calendar table (contract C9)
- Task E2: Prior-year weekly demand table (contract C10)
- Task E3: Transaction-derived weekly signals (contract C8)
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

import holidays
import numpy as np
import pandas as pd

from src.data_prep import (
    _is_non_product_code,
    clean_transactions,
    load_raw,
    week_start_of,
)

logger = logging.getLogger(__name__)

# Default file paths
CALENDAR_OUTPUT_PATH = Path("data/processed/calendar_weekly.csv")
PRIOR_YEAR_OUTPUT_PATH = Path("data/processed/prior_year_weekly.parquet")
ENRICHMENT_OUTPUT_PATH = Path("data/processed/enrichment_weekly.parquet")
WEEKLY_DEMAND_PATH = Path("data/processed/weekly_demand.parquet")
RAW_MAIN_PATH = Path("data/raw/Online Retail.parquet")
RAW_RET2_PATH = Path("data/raw/online_retail_II_Year_2009-2010.parquet")
RAW_RET2_XLSX_PATH = Path("data/raw/online_retail_II.xlsx")

# Fixed dates & parameters
CALENDAR_START = "2009-12-07"
CALENDAR_END = "2011-12-05"
PRIOR_YEAR_END_CUTOFF = "2010-12-06"
PRIOR_YEAR_LAST_FULL_WEEK = "2010-11-29"
PRIOR_YEAR_CLOSED_WEEK = "2009-12-28"
DEFAULT_QUANTITY_CAP = 480.0

# Expected trading weekdays: Mon(0), Tue(1), Wed(2), Thu(3), Fri(4), Sun(6) (no Saturday 5)
TRADING_WEEKDAYS: frozenset[int] = frozenset({0, 1, 2, 3, 4, 6})

# Royal Wedding public holiday date
ROYAL_WEDDING_DATE = "2011-04-29"


# ---------------------------------------------------------------------------
# Task E1: Calendar table (Contract C9)
# ---------------------------------------------------------------------------


def get_uk_england_holidays(years: Sequence[int] | None = None) -> holidays.HolidayBase:
    """Return UK holidays for England subdivision, ensuring Royal Wedding is included.

    Supports holidays.country_holidays('GB', subdiv='England') or
    holidays.UnitedKingdom(subdiv='England').
    """
    if hasattr(holidays, "country_holidays"):
        uk_holidays = holidays.country_holidays("GB", subdiv="England", years=years)
    else:
        uk_holidays = holidays.UnitedKingdom(subdiv="England", years=years)

    # Verify that the Royal Wedding date (2011-04-29) is present; if missing, add explicitly
    wedding_dt = pd.Timestamp(ROYAL_WEDDING_DATE).date()
    if wedding_dt not in uk_holidays:
        logger.info(
            "2011-04-29 (Royal Wedding) missing from holidays package; adding explicitly."
        )
        uk_holidays[wedding_dt] = "Royal Wedding"

    return uk_holidays


def build_calendar_weekly(
    weeks: Sequence[pd.Timestamp | str] | None = None,
    out_path: str | Path = CALENDAR_OUTPUT_PATH,
) -> pd.DataFrame:
    """Build the weekly calendar table (Contract C9) covering 2009-12-07 to 2011-12-05.

    Columns:
        week_start: Monday date string (YYYY-MM-DD)
        uk_bank_holiday_count: int >= 0
        trading_days_expected: int >= 0
        is_christmas_week: int (0 or 1)
    """
    if weeks is None:
        mondays = pd.date_range(CALENDAR_START, CALENDAR_END, freq="W-MON")
    else:
        mondays = pd.DatetimeIndex([pd.Timestamp(w).normalize() for w in weeks])

    years = sorted(list({w.year for w in mondays} | {w.year + 1 for w in mondays}))
    uk_holidays = get_uk_england_holidays(years=years)

    rows: list[dict[str, object]] = []

    for mon in mondays:
        # Full 7-day week from Monday to Sunday
        week_days = pd.date_range(mon, mon + pd.Timedelta(days=6), freq="D")
        
        # Bank holiday count for the week
        holidays_in_week = [d for d in week_days if d.date() in uk_holidays]
        holiday_count = len(holidays_in_week)

        # Trading weekdays in week (Mon-Fri and Sun, excluding Sat)
        # Minus bank holidays that fall on one of these trading weekdays
        holidays_on_trading_days = sum(
            1 for d in holidays_in_week if d.weekday() in TRADING_WEEKDAYS
        )
        trading_days_expected = len(TRADING_WEEKDAYS) - holidays_on_trading_days

        # Christmas week: contains 24, 25, or 26 December
        is_christmas = int(
            any((d.month == 12 and d.day in (24, 25, 26)) for d in week_days)
        )

        rows.append(
            {
                "week_start": mon.strftime("%Y-%m-%d"),
                "uk_bank_holiday_count": int(holiday_count),
                "trading_days_expected": int(trading_days_expected),
                "is_christmas_week": int(is_christmas),
            }
        )

    df = pd.DataFrame(rows)
    df["uk_bank_holiday_count"] = df["uk_bank_holiday_count"].astype("int64")
    df["trading_days_expected"] = df["trading_days_expected"].astype("int64")
    df["is_christmas_week"] = df["is_christmas_week"].astype("int64")

    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_file, index=False)
    logger.info("Wrote %d rows to %s", len(df), out_file)

    return df


# ---------------------------------------------------------------------------
# Task E2: Prior-year table (Contract C10)
# ---------------------------------------------------------------------------


def normalize_ret2_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize Online Retail II column names to match Shelfwatch standards."""
    mapping = {
        "Invoice": "InvoiceNo",
        "Price": "UnitPrice",
        "Customer ID": "CustomerID",
    }
    cols = {k: v for k, v in mapping.items() if k in df.columns}
    return df.rename(columns=cols)


def _load_ret2_data(ret2_path: Path) -> pd.DataFrame:
    """Load Online Retail II 2009-2010 sheet from parquet or excel."""
    if ret2_path.suffix.lower() == ".parquet":
        return pd.read_parquet(ret2_path)
    if ret2_path.suffix.lower() == ".xlsx":
        return pd.read_excel(ret2_path, sheet_name="Year 2009-2010", engine="openpyxl")
    raise ValueError(f"Unsupported extension for Online Retail II: {ret2_path}")


def build_prior_year_weekly(
    ret2_path: str | Path | None = None,
    cap: float | None = None,
    products: Sequence[str] | None = None,
    out_path: str | Path = PRIOR_YEAR_OUTPUT_PATH,
) -> pd.DataFrame:
    """Build the prior-year weekly table (Contract C10).

    Columns:
        product_id: str
        week_start: datetime64[ns]
        units: int64
        orders: int64
    """
    empty_df = pd.DataFrame(
        {
            "product_id": pd.Series(dtype="str"),
            "week_start": pd.Series(dtype="datetime64[ns]"),
            "units": pd.Series(dtype="int64"),
            "orders": pd.Series(dtype="int64"),
        }
    )

    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # Determine input file path
    target_path: Path | None = None
    if ret2_path is not None:
        p = Path(ret2_path)
        if p.exists():
            target_path = p
    else:
        if RAW_RET2_PATH.exists():
            target_path = RAW_RET2_PATH
        elif RAW_RET2_XLSX_PATH.exists():
            target_path = RAW_RET2_XLSX_PATH

    # Missing file handling (FR-X-4)
    if target_path is None or not target_path.exists():
        logger.warning(
            "Online Retail II dataset not found at %s. Writing empty C10 table.",
            ret2_path or f"{RAW_RET2_PATH} / {RAW_RET2_XLSX_PATH}",
        )
        empty_df.to_parquet(out_file, index=False)
        return empty_df

    # Load and normalize columns
    raw_ret2 = _load_ret2_data(target_path)
    raw_ret2 = normalize_ret2_columns(raw_ret2)

    # Apply standard cleaning rules
    cleaned_df, _ = clean_transactions(raw_ret2)

    # Use specified cap or default
    cap_val = float(cap) if cap is not None else DEFAULT_QUANTITY_CAP
    cleaned_df["Quantity"] = cleaned_df["Quantity"].clip(upper=cap_val)

    # Assign week_start
    cleaned_df["week_start"] = week_start_of(cleaned_df["InvoiceDate"])

    # Keep only weeks strictly before 2010-12-06
    cutoff = pd.Timestamp(PRIOR_YEAR_END_CUTOFF)
    cleaned_df = cleaned_df[cleaned_df["week_start"] < cutoff].copy()

    # Determine Top 100 products to filter
    if products is None:
        if WEEKLY_DEMAND_PATH.exists():
            demand_df = pd.read_parquet(WEEKLY_DEMAND_PATH)
            target_products = sorted(demand_df["product_id"].unique().tolist())
        else:
            target_products = sorted(cleaned_df["StockCode"].unique().tolist())
    else:
        target_products = sorted(list(products))

    cleaned_df = cleaned_df[cleaned_df["StockCode"].isin(target_products)].copy()

    # Aggregate by product and week_start
    agg = (
        cleaned_df.groupby(["StockCode", "week_start"])
        .agg(
            units=("Quantity", "sum"),
            orders=("InvoiceNo", "nunique"),
        )
        .reset_index()
        .rename(columns={"StockCode": "product_id"})
    )

    last_full_week = pd.Timestamp(PRIOR_YEAR_LAST_FULL_WEEK)
    closed_week = pd.Timestamp(PRIOR_YEAR_CLOSED_WEEK)

    # Zero-fill each product from its first sale through 2010-11-29;
    # products with no prior-year sales get NO rows.
    product_dfs: list[pd.DataFrame] = []
    products_with_rows = 0
    weeks_per_product: list[int] = []

    for pid in target_products:
        prod_data = agg[agg["product_id"] == pid]
        if prod_data.empty:
            continue

        products_with_rows += 1
        first_week = prod_data["week_start"].min()

        # Build Monday range from first_week to last_full_week, excluding closure week
        all_weeks = pd.date_range(first_week, last_full_week, freq="W-MON")
        valid_weeks = [w for w in all_weeks if w != closed_week]

        grid = pd.DataFrame({"product_id": pid, "week_start": valid_weeks})
        merged = pd.merge(grid, prod_data, on=["product_id", "week_start"], how="left")
        merged["units"] = merged["units"].fillna(0).astype("int64")
        merged["orders"] = merged["orders"].fillna(0).astype("int64")

        weeks_per_product.append(len(merged))
        product_dfs.append(merged)

    if product_dfs:
        result = pd.concat(product_dfs, ignore_index=True)
        result = result.sort_values(["product_id", "week_start"]).reset_index(drop=True)
    else:
        result = empty_df

    result["product_id"] = result["product_id"].astype(str)
    result["week_start"] = pd.to_datetime(result["week_start"])
    result["units"] = result["units"].astype("int64")
    result["orders"] = result["orders"].astype("int64")

    # Coverage report
    total_target = len(target_products)
    if weeks_per_product:
        median_weeks = float(np.median(weeks_per_product))
        at_least_26 = sum(1 for w in weeks_per_product if w >= 26)
        share_26 = (at_least_26 / total_target) * 100.0 if total_target > 0 else 0.0
    else:
        median_weeks = 0.0
        share_26 = 0.0

    print("=== Prior-Year Table (C10) Coverage Report ===")
    print(f"Target products: {total_target}")
    print(f"Products with prior-year sales: {products_with_rows} ({products_with_rows / total_target * 100:.1f}%)")
    print(f"Median weeks per product: {median_weeks:.1f}")
    print(f"Products with >= 26 weeks: {share_26:.1f}%")

    result.to_parquet(out_file, index=False)
    logger.info("Wrote %d rows to %s", len(result), out_file)

    return result


# ---------------------------------------------------------------------------
# Task E3: Transaction-derived weekly signals (Contract C8)
# ---------------------------------------------------------------------------


def build_enrichment_weekly(
    raw_path: str | Path | None = None,
    products: Sequence[str] | None = None,
    weeks: pd.DatetimeIndex | None = None,
    out_path: str | Path = ENRICHMENT_OUTPUT_PATH,
    cap: float | None = None,
) -> pd.DataFrame:
    """Build transaction-derived weekly signals table (Contract C8).

    Columns:
        product_id: str
        week_start: datetime64[ns]
        gross_units: int64
        cancelled_units: int64
        export_units: int64
        guest_units: int64
    """
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # 1. Load C1 demand table to get the exact grid and target units
    if not WEEKLY_DEMAND_PATH.exists():
        raise FileNotFoundError(
            f"Cannot build C8 without C1: {WEEKLY_DEMAND_PATH} does not exist."
        )

    demand_df = pd.read_parquet(WEEKLY_DEMAND_PATH)
    grid = demand_df[["product_id", "week_start", "units"]].copy()

    if products is None:
        target_products = sorted(grid["product_id"].unique().tolist())
    else:
        target_products = sorted(list(products))

    # 2. Load raw data
    target_raw_path = Path(raw_path) if raw_path is not None else RAW_MAIN_PATH
    raw = load_raw(target_raw_path)

    # Normalize StockCode and filter non-products
    raw["StockCode"] = raw["StockCode"].astype(str).str.strip().str.upper()
    raw = raw[~raw["StockCode"].map(_is_non_product_code)].reset_index(drop=True)
    raw = raw[raw["StockCode"].isin(target_products)].reset_index(drop=True)

    # Assign week_start to all rows
    raw["week_start"] = week_start_of(raw["InvoiceDate"])

    # 3. Cancelled units: absolute Quantity on invoices starting with 'C'
    inv_str = raw["InvoiceNo"].astype(str).str.strip().str.upper()
    is_cancel = inv_str.str.startswith("C")

    cancel_df = raw[is_cancel].copy()
    if not cancel_df.empty:
        cancel_agg = (
            cancel_df.groupby(["StockCode", "week_start"])["Quantity"]
            .apply(lambda q: int(abs(q).sum()))
            .reset_index()
            .rename(columns={"StockCode": "product_id", "Quantity": "cancelled_units"})
        )
    else:
        cancel_agg = pd.DataFrame(
            columns=["product_id", "week_start", "cancelled_units"]
        )

    # 4. Valid demand transactions (clean sales)
    valid_mask = (~is_cancel) & (raw["Quantity"] > 0) & (raw["UnitPrice"] > 0)
    sales_df = raw[valid_mask].copy()

    # Cap Quantity with same cap value as training data
    cap_val = float(cap) if cap is not None else DEFAULT_QUANTITY_CAP
    sales_df["Quantity"] = sales_df["Quantity"].clip(upper=cap_val)

    # Export units: capped units where Country != "United Kingdom"
    sales_df["export_qty"] = np.where(
        sales_df["Country"].astype(str).str.strip() != "United Kingdom",
        sales_df["Quantity"],
        0,
    )

    # Guest units: capped units where CustomerID is null/NaN
    sales_df["guest_qty"] = np.where(
        sales_df["CustomerID"].isna(),
        sales_df["Quantity"],
        0,
    )

    sales_agg = (
        sales_df.groupby(["StockCode", "week_start"])
        .agg(
            gross_units=("Quantity", "sum"),
            export_units=("export_qty", "sum"),
            guest_units=("guest_qty", "sum"),
        )
        .reset_index()
        .rename(columns={"StockCode": "product_id"})
    )

    # 5. Merge onto the EXACT C1 product-week grid
    merged = pd.merge(grid, sales_agg, on=["product_id", "week_start"], how="left")
    merged = pd.merge(merged, cancel_agg, on=["product_id", "week_start"], how="left")

    # For gross_units: must equal C1 units exactly
    merged["gross_units"] = merged["gross_units"].fillna(0).astype("int64")
    merged["cancelled_units"] = merged["cancelled_units"].fillna(0).astype("int64")
    merged["export_units"] = merged["export_units"].fillna(0).astype("int64")
    merged["guest_units"] = merged["guest_units"].fillna(0).astype("int64")

    # Assertions required by prompt & contract C8
    assert len(merged) == len(demand_df), f"Row count mismatch: {len(merged)} vs {len(demand_df)}"
    assert (merged["gross_units"] == merged["units"]).all(), (
        "gross_units must equal C1 units exactly"
    )
    assert (merged["export_units"] <= merged["gross_units"]).all(), (
        "export_units must not exceed gross_units"
    )
    assert (merged["guest_units"] <= merged["gross_units"]).all(), (
        "guest_units must not exceed gross_units"
    )
    assert (merged["cancelled_units"] >= 0).all(), (
        "cancelled_units must be non-negative"
    )

    # Final column ordering
    cols = [
        "product_id",
        "week_start",
        "gross_units",
        "cancelled_units",
        "export_units",
        "guest_units",
    ]
    result = merged[cols].copy()
    result["product_id"] = result["product_id"].astype(str)
    result["week_start"] = pd.to_datetime(result["week_start"])

    result.to_parquet(out_file, index=False)
    logger.info("Wrote %d rows to %s", len(result), out_file)

    return result


# ---------------------------------------------------------------------------
# CLI Interface
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entrypoint for building enrichment tables."""
    parser = argparse.ArgumentParser(description="Build Shelfwatch enrichment tables.")
    parser.add_argument("--calendar", action="store_true", help="Build calendar table (C9)")
    parser.add_argument("--prior-year", action="store_true", help="Build prior-year table (C10)")
    parser.add_argument("--enrichment", action="store_true", help="Build transaction signals table (C8)")
    parser.add_argument("--all", action="store_true", help="Build all three tables")

    args = parser.parse_args()

    if not any([args.calendar, args.prior_year, args.enrichment, args.all]):
        parser.print_help()
        return

    if args.calendar or args.all:
        print("\n--- Building Calendar Table (C9) ---")
        build_calendar_weekly()

    if args.prior_year or args.all:
        print("\n--- Building Prior-Year Table (C10) ---")
        build_prior_year_weekly()

    if args.enrichment or args.all:
        print("\n--- Building Enrichment Weekly Table (C8) ---")
        build_enrichment_weekly()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
