"""Unit tests for product intelligence (Task E4, Contract C11).

Validates:
- Hand-made series for each of the four demand pattern quadrants (smooth, erratic, intermittent, lumpy)
- ABC revenue share sum to 1.0 and 80%/95% boundary rules
- XYZ classification thresholds
- Lifecycle stage cases (new, inactive, growing, declining, mature)
- Test-set sales inflation isolation (zero data leakage)
- Product scope isolation (no products outside the top 100)
- Contract C11 schema compliance
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.intelligence import (
    C11_COLUMNS,
    C12_COLUMNS,
    build_basket_pairs,
    build_product_intelligence,
    classify_demand_pattern,
    compute_abc_classes,
    compute_demand_pattern_metrics,
    compute_lifecycle_stage,
    compute_xyz_class,
)

PROCESSED_INTELLIGENCE_PATH = Path("data/processed/product_intelligence.csv")
PROCESSED_BASKET_PAIRS_PATH = Path("data/processed/basket_pairs.csv")
WEEKLY_DEMAND_PATH = Path("data/processed/weekly_demand.parquet")
RAW_MAIN_PATH = Path("data/raw/Online Retail.parquet")


# ---------------------------------------------------------------------------
# 1. Syntetos-Boylan demand pattern quadrants
# ---------------------------------------------------------------------------


def test_demand_pattern_quadrants() -> None:
    """Validate hand-made series that fall into each of the four pattern quadrants.

    Cutoffs:
        ADI cutoff = 1.32
        CV2 cutoff = 0.49

    Quadrant 1: smooth       (ADI < 1.32,  CV2 < 0.49)
    Quadrant 2: erratic      (ADI < 1.32,  CV2 >= 0.49)
    Quadrant 3: intermittent (ADI >= 1.32, CV2 < 0.49)
    Quadrant 4: lumpy        (ADI >= 1.32, CV2 >= 0.49)
    """
    # 1. Smooth: ADI = 1.0 (< 1.32), steady demand -> CV2 = 0.0 (< 0.49)
    smooth_series = [10.0, 10.0, 10.0, 10.0, 10.0]
    adi_s, cv2_s, pclass_s = compute_demand_pattern_metrics(smooth_series)
    assert adi_s == 1.0
    assert cv2_s == 0.0
    assert pclass_s == "smooth"

    # 2. Erratic: ADI = 1.0 (< 1.32), high variance in non-zero sales -> CV2 >= 0.49
    # non-zero = [2, 20, 2, 20]: mean = 11, std = 10.39, CV2 = (10.39/11)^2 = 0.89
    erratic_series = [2.0, 20.0, 2.0, 20.0]
    adi_e, cv2_e, pclass_e = compute_demand_pattern_metrics(erratic_series)
    assert adi_e == 1.0
    assert cv2_e >= 0.49
    assert pclass_e == "erratic"

    # 3. Intermittent: ADI = 2.0 (>= 1.32), non-zero sales are constant -> CV2 = 0.0 (< 0.49)
    # 6 weeks total: non-zero at weeks 0, 2, 4 -> 3 active weeks out of 5 from first sale (or 3 out of 6)
    intermittent_series = [10.0, 0.0, 10.0, 0.0, 10.0, 0.0]
    adi_i, cv2_i, pclass_i = compute_demand_pattern_metrics(intermittent_series)
    assert adi_i == 2.0
    assert cv2_i < 0.49
    assert pclass_i == "intermittent"

    # 4. Lumpy: ADI = 2.0 (>= 1.32), non-zero sales have high variance -> CV2 >= 0.49
    lumpy_series = [2.0, 0.0, 20.0, 0.0, 2.0, 0.0, 20.0, 0.0]
    adi_l, cv2_l, pclass_l = compute_demand_pattern_metrics(lumpy_series)
    assert adi_l == 2.0
    assert cv2_l >= 0.49
    assert pclass_l == "lumpy"


def test_classify_demand_pattern_direct() -> None:
    """Direct verification of boundary cutoffs."""
    assert classify_demand_pattern(1.31, 0.48) == "smooth"
    assert classify_demand_pattern(1.31, 0.49) == "erratic"
    assert classify_demand_pattern(1.32, 0.48) == "intermittent"
    assert classify_demand_pattern(1.32, 0.49) == "lumpy"


# ---------------------------------------------------------------------------
# 2. ABC revenue classification
# ---------------------------------------------------------------------------


def test_abc_revenue_shares_and_classes() -> None:
    """ABC revenue shares sum to 1.0 and follow the 80% and 95% rules."""
    # Hand-made revenues:
    # P1: 50, P2: 25, P3: 15, P4: 7, P5: 3 -> Total = 100
    # Shares: P1=0.50, P2=0.25, P3=0.15, P4=0.07, P5=0.03
    # Cumulative:
    # P1: 0.50 <= 0.80 -> A
    # P2: 0.75 <= 0.80 -> A
    # P3: 0.90 <= 0.95 -> B
    # P4: 0.97 >  0.95 -> C
    # P5: 1.00 >  0.95 -> C
    revenues = pd.Series(
        {"P1": 50.0, "P2": 25.0, "P3": 15.0, "P4": 7.0, "P5": 3.0}
    )
    shares, classes = compute_abc_classes(revenues)

    # Sum of shares must equal 1.0
    assert np.isclose(shares.sum(), 1.0)
    assert classes["P1"] == "A"
    assert classes["P2"] == "A"
    assert classes["P3"] == "B"
    assert classes["P4"] == "C"
    assert classes["P5"] == "C"


def test_abc_single_dominant_product() -> None:
    """A dominant single product (>80% alone) is assigned A, next up to 95% is B, remainder C."""
    revenues = pd.Series({"P1": 82.0, "P2": 10.0, "P3": 8.0})
    shares, classes = compute_abc_classes(revenues)
    assert np.isclose(shares.sum(), 1.0)
    assert classes["P1"] == "A"
    assert classes["P2"] == "B"  # cumulative 92% <= 95%
    assert classes["P3"] == "C"  # cumulative 100% > 95%


# ---------------------------------------------------------------------------
# 3. XYZ classification
# ---------------------------------------------------------------------------


def test_xyz_classification() -> None:
    """Verify XYZ coefficient of variation thresholds: X <= 0.5, Y <= 1.0, Z > 1.0."""
    # X: CV <= 0.5 (low variation)
    cv_x, class_x = compute_xyz_class([10.0, 11.0, 9.0, 10.0, 10.0])
    assert cv_x <= 0.5
    assert class_x == "X"

    # Y: 0.5 < CV <= 1.0 (moderate variation)
    cv_y, class_y = compute_xyz_class([10.0, 20.0, 5.0, 15.0, 2.0])
    assert 0.5 < cv_y <= 1.0
    assert class_y == "Y"

    # Z: CV > 1.0 (high variation / intermittency)
    cv_z, class_z = compute_xyz_class([0.0, 0.0, 0.0, 50.0, 0.0, 0.0, 60.0])
    assert cv_z > 1.0
    assert class_z == "Z"


# ---------------------------------------------------------------------------
# 4. Lifecycle cases
# ---------------------------------------------------------------------------


def test_lifecycle_cases() -> None:
    """Verify all 5 lifecycle cases: new, inactive, growing, declining, mature."""
    # Create 40 weekly timestamps
    weeks = [pd.Timestamp("2011-01-03") + pd.Timedelta(weeks=i) for i in range(40)]

    # 1. Inactive: no sales in the last 8 training weeks
    # Sales in weeks 0..30, 0 in weeks 32..39
    s_inactive = pd.Series(
        [10.0 if i < 30 else 0.0 for i in range(40)], index=weeks
    )
    stage_inact, w_since_inact = compute_lifecycle_stage(s_inactive, weeks)
    assert stage_inact == "inactive"
    assert w_since_inact >= 8

    # 2. New: first sale is within the last 12 training weeks (e.g. week 30)
    s_new = pd.Series(
        [0.0 if i < 32 else 15.0 for i in range(40)], index=weeks
    )
    stage_new, w_since_new = compute_lifecycle_stage(s_new, weeks)
    assert stage_new == "new"
    assert w_since_new == 0

    # 3. Growing: last 8 weeks mean / prior 8 weeks mean >= 1.2
    # Active from start, prev 8 (weeks 24..31) mean = 10, last 8 (weeks 32..39) mean = 20 (ratio = 2.0)
    units_growing = [10.0] * 32 + [20.0] * 8
    s_growing = pd.Series(units_growing, index=weeks)
    stage_grow, w_since_grow = compute_lifecycle_stage(s_growing, weeks)
    assert stage_grow == "growing"
    assert w_since_grow == 0

    # 4. Declining: last 8 weeks mean / prior 8 weeks mean <= 0.8
    # Active from start, prev 8 mean = 20, last 8 mean = 10 (ratio = 0.5)
    units_declining = [20.0] * 32 + [10.0] * 8
    s_declining = pd.Series(units_declining, index=weeks)
    stage_dec, w_since_dec = compute_lifecycle_stage(s_declining, weeks)
    assert stage_dec == "declining"
    assert w_since_dec == 0

    # 5. Mature: ratio between 0.8 and 1.2 (steady)
    units_mature = [10.0] * 40
    s_mature = pd.Series(units_mature, index=weeks)
    stage_mat, w_since_mat = compute_lifecycle_stage(s_mature, weeks)
    assert stage_mat == "mature"
    assert w_since_mat == 0


# ---------------------------------------------------------------------------
# 5. Zero Leakage: Test Period Inflation Isolation
# ---------------------------------------------------------------------------


def test_test_period_inflation_isolation() -> None:
    """Product intelligence must be 100% IDENTICAL when sales in test weeks are inflated."""
    if not WEEKLY_DEMAND_PATH.exists() or not RAW_MAIN_PATH.exists():
        pytest.skip("Production data not available for inflation isolation test")

    weekly_clean = pd.read_parquet(WEEKLY_DEMAND_PATH)

    # Inflate all test-period units by 1,000,000x and price by 100x
    weekly_inflated = weekly_clean.copy()
    test_mask = weekly_inflated["week_start"] >= "2011-10-10"
    weekly_inflated.loc[test_mask, "units"] = weekly_inflated.loc[test_mask, "units"] * 1_000_000 + 999_999
    weekly_inflated.loc[test_mask, "avg_price"] = weekly_inflated.loc[test_mask, "avg_price"] * 100.0

    # Build intelligence for both
    df_clean = build_product_intelligence(
        weekly=weekly_clean,
        raw_path=RAW_MAIN_PATH,
        train_end="2011-10-10",
        out_path=None,
    )
    df_inflated = build_product_intelligence(
        weekly=weekly_inflated,
        raw_path=RAW_MAIN_PATH,
        train_end="2011-10-10",
        out_path=None,
    )

    # Both tables must be strictly identical
    pd.testing.assert_frame_equal(df_clean, df_inflated)


# ---------------------------------------------------------------------------
# 6. Product Scope Isolation (Top 100 only)
# ---------------------------------------------------------------------------


def test_no_row_for_outside_products() -> None:
    """Ensure product_intelligence contains only products from weekly demand (Top 100)."""
    if not WEEKLY_DEMAND_PATH.exists() or not RAW_MAIN_PATH.exists():
        pytest.skip("Production data not available")

    weekly = pd.read_parquet(WEEKLY_DEMAND_PATH)
    top100_ids = set(weekly["product_id"].unique())
    assert len(top100_ids) == 100

    df = build_product_intelligence(
        weekly=weekly,
        raw_path=RAW_MAIN_PATH,
        train_end="2011-10-10",
        out_path=None,
    )

    output_ids = set(df["product_id"].unique())
    assert output_ids == top100_ids
    assert len(df) == 100


# ---------------------------------------------------------------------------
# 7. Production Artifact & Contract C11 validation
# ---------------------------------------------------------------------------


def test_production_file_contract_c11() -> None:
    """Ensure data/processed/product_intelligence.csv satisfies C11 contract."""
    if not PROCESSED_INTELLIGENCE_PATH.exists():
        pytest.skip("data/processed/product_intelligence.csv not yet generated")

    df = pd.read_csv(PROCESSED_INTELLIGENCE_PATH, dtype={"product_id": str})
    assert list(df.columns) == C11_COLUMNS
    assert len(df) == 100
    assert not df.isna().any().any()
    assert not df["product_id"].duplicated().any()

    assert set(df["pattern_class"].unique()).issubset({"smooth", "erratic", "intermittent", "lumpy"})
    assert set(df["abc_class"].unique()).issubset({"A", "B", "C"})
    assert set(df["xyz_class"].unique()).issubset({"X", "Y", "Z"})
    assert set(df["lifecycle_stage"].unique()).issubset({"new", "growing", "mature", "declining", "inactive"})

    assert (df["adi"] > 0).all()
    assert (df["cv2"] >= 0).all()
    assert (df["weeks_since_last_sale"] >= 0).all()
    assert (df["return_rate"] >= 0).all()

    for col in ("export_share", "top3_customer_share", "known_customer_coverage", "revenue_share"):
        assert (df[col] >= 0.0).all() and (df[col] <= 1.0).all()

    # Revenue share sums to ~1.0
    assert np.isclose(df["revenue_share"].sum(), 1.0, atol=0.01)


# ---------------------------------------------------------------------------
# 8. Basket Pairs / Bought-Together Tests (Contract C12)
# ---------------------------------------------------------------------------


def test_basket_pairs_toy_dataset() -> None:
    """Validate hand-computed support, confidence, and lift on a toy dataset.

    Toy dataset:
    - 4 invoices:
      INV1: [P1, P2]
      INV2: [P1, P2, P3]
      INV3: [P1, P3]
      INV4: [P2]  -> only 1 target product, so skipped!
    - Qualifying invoices (>= 2 products): INV1, INV2, INV3 -> N = 3
    - Occurrences in qualifying invoices:
      n_1 = 3 (INV1, INV2, INV3)
      n_2 = 2 (INV1, INV2)
      n_3 = 2 (INV2, INV3)
    - Co-occurrences:
      n_12 = 2 (INV1, INV2)
      n_13 = 2 (INV2, INV3)
      n_23 = 1 (INV2)
    - Metrics:
      conf(1 -> 2) = 2 / 3 = 0.6667
      lift(1 -> 2) = (2 / 3) / (2 / 3) = 1.0
      conf(2 -> 1) = 2 / 2 = 1.0
      lift(2 -> 1) = 1.0 / (3 / 3) = 1.0

      conf(2 -> 3) = 1 / 2 = 0.5
      lift(2 -> 3) = 0.5 / (2 / 3) = 0.75
    """
    toy_rows = [
        {"InvoiceNo": "INV1", "StockCode": "P1", "Description": "Product 1", "Quantity": 1, "UnitPrice": 2.0, "InvoiceDate": "2011-01-05"},
        {"InvoiceNo": "INV1", "StockCode": "P2", "Description": "Product 2", "Quantity": 2, "UnitPrice": 3.0, "InvoiceDate": "2011-01-05"},
        {"InvoiceNo": "INV2", "StockCode": "P1", "Description": "Product 1", "Quantity": 1, "UnitPrice": 2.0, "InvoiceDate": "2011-01-06"},
        {"InvoiceNo": "INV2", "StockCode": "P2", "Description": "Product 2", "Quantity": 1, "UnitPrice": 3.0, "InvoiceDate": "2011-01-06"},
        {"InvoiceNo": "INV2", "StockCode": "P3", "Description": "Product 3", "Quantity": 1, "UnitPrice": 4.0, "InvoiceDate": "2011-01-06"},
        {"InvoiceNo": "INV3", "StockCode": "P1", "Description": "Product 1", "Quantity": 1, "UnitPrice": 2.0, "InvoiceDate": "2011-01-07"},
        {"InvoiceNo": "INV3", "StockCode": "P3", "Description": "Product 3", "Quantity": 1, "UnitPrice": 4.0, "InvoiceDate": "2011-01-07"},
        {"InvoiceNo": "INV4", "StockCode": "P2", "Description": "Product 2", "Quantity": 5, "UnitPrice": 3.0, "InvoiceDate": "2011-01-08"},
    ]
    toy_df = pd.DataFrame(toy_rows)

    # Test with min_support=2 -> (P1, P2), (P2, P1), (P1, P3), (P3, P1) qualify; (P2, P3) fails with n=1
    pairs = build_basket_pairs(
        raw_path=toy_df,
        products=["P1", "P2", "P3"],
        train_end="2011-10-10",
        min_support=2,
        out_path=None,
    )

    assert len(pairs) == 4
    assert set(zip(pairs["product_id"], pairs["companion_id"])) == {
        ("P1", "P2"), ("P2", "P1"), ("P1", "P3"), ("P3", "P1")
    }

    # Verify exact metrics for (P2 -> P1)
    p2_to_p1 = pairs[(pairs["product_id"] == "P2") & (pairs["companion_id"] == "P1")].iloc[0]
    assert p2_to_p1["n_invoices"] == 2
    assert np.isclose(p2_to_p1["confidence"], 1.0)
    assert np.isclose(p2_to_p1["lift"], 1.0)
    assert p2_to_p1["companion_description"] == "Product 1"

    # Verify exact metrics for (P1 -> P2)
    p1_to_p2 = pairs[(pairs["product_id"] == "P1") & (pairs["companion_id"] == "P2")].iloc[0]
    assert p1_to_p2["n_invoices"] == 2
    assert np.isclose(p1_to_p2["confidence"], 2 / 3, atol=1e-4)
    assert np.isclose(p1_to_p2["lift"], 1.0, atol=1e-4)


def test_basket_pairs_minimum_support_enforced() -> None:
    """Verify that pairs below min_support are strictly excluded."""
    toy_rows = [
        {"InvoiceNo": "INV1", "StockCode": "P1", "Description": "Product 1", "Quantity": 1, "UnitPrice": 2.0, "InvoiceDate": "2011-01-05"},
        {"InvoiceNo": "INV1", "StockCode": "P2", "Description": "Product 2", "Quantity": 2, "UnitPrice": 3.0, "InvoiceDate": "2011-01-05"},
    ]
    toy_df = pd.DataFrame(toy_rows)

    # Co-occurrence is 1, but min_support is set to 2 -> should return empty
    pairs = build_basket_pairs(
        raw_path=toy_df,
        products=["P1", "P2"],
        train_end="2011-10-10",
        min_support=2,
        out_path=None,
    )
    assert pairs.empty


def test_basket_pairs_product_no_qualifying_yields_no_rows() -> None:
    """A product with no pairs meeting min_support yields no rows in output."""
    toy_rows = [
        {"InvoiceNo": "INV1", "StockCode": "P1", "Description": "Product 1", "Quantity": 1, "UnitPrice": 2.0, "InvoiceDate": "2011-01-05"},
        {"InvoiceNo": "INV1", "StockCode": "P2", "Description": "Product 2", "Quantity": 2, "UnitPrice": 3.0, "InvoiceDate": "2011-01-05"},
        {"InvoiceNo": "INV2", "StockCode": "P1", "Description": "Product 1", "Quantity": 1, "UnitPrice": 2.0, "InvoiceDate": "2011-01-06"},
        {"InvoiceNo": "INV2", "StockCode": "P2", "Description": "Product 2", "Quantity": 1, "UnitPrice": 3.0, "InvoiceDate": "2011-01-06"},
        {"InvoiceNo": "INV2", "StockCode": "P99", "Description": "Solo Product", "Quantity": 1, "UnitPrice": 4.0, "InvoiceDate": "2011-01-06"},
    ]
    toy_df = pd.DataFrame(toy_rows)

    # min_support=2: P1 and P2 co-occur twice, P99 only once with each
    pairs = build_basket_pairs(
        raw_path=toy_df,
        products=["P1", "P2", "P99"],
        train_end="2011-10-10",
        min_support=2,
        out_path=None,
    )
    assert not pairs.empty
    assert "P99" not in pairs["product_id"].values
    assert "P99" not in pairs["companion_id"].values


def test_basket_pairs_training_period_isolation() -> None:
    """Inflating or adding test-period invoices changes nothing in basket pairs."""
    if not RAW_MAIN_PATH.exists() or not WEEKLY_DEMAND_PATH.exists():
        pytest.skip("Production data not available")

    w_df = pd.read_parquet(WEEKLY_DEMAND_PATH)
    top10 = sorted(w_df["product_id"].unique().tolist())[:10]

    raw_clean = pd.read_parquet(RAW_MAIN_PATH)

    # Baseline on clean data
    pairs_clean = build_basket_pairs(
        raw_path=raw_clean,
        products=top10,
        train_end="2011-10-10",
        min_support=10,
        out_path=None,
    )

    # Add 10,000 synthetic invoices in the test period (>= 2011-10-10)
    fake_invoices = []
    for i in range(500):
        fake_invoices.append({
            "InvoiceNo": f"FAKE_{i}",
            "StockCode": top10[0],
            "Description": "Fake Product 0",
            "Quantity": 50,
            "UnitPrice": 10.0,
            "InvoiceDate": "2011-11-01 10:00:00",
        })
        fake_invoices.append({
            "InvoiceNo": f"FAKE_{i}",
            "StockCode": top10[1],
            "Description": "Fake Product 1",
            "Quantity": 50,
            "UnitPrice": 10.0,
            "InvoiceDate": "2011-11-01 10:00:00",
        })
    raw_inflated = pd.concat([raw_clean, pd.DataFrame(fake_invoices)], ignore_index=True)

    pairs_inflated = build_basket_pairs(
        raw_path=raw_inflated,
        products=top10,
        train_end="2011-10-10",
        min_support=10,
        out_path=None,
    )

    pd.testing.assert_frame_equal(pairs_clean, pairs_inflated)


def test_production_file_contract_c12() -> None:
    """Ensure data/processed/basket_pairs.csv satisfies C12 contract."""
    if not PROCESSED_BASKET_PAIRS_PATH.exists():
        pytest.skip("data/processed/basket_pairs.csv not yet generated")

    df = pd.read_csv(PROCESSED_BASKET_PAIRS_PATH, dtype={"product_id": str, "companion_id": str})
    assert list(df.columns) == C12_COLUMNS
    assert not df.empty
    assert not df.isna().any().any()

    assert (df["product_id"] != df["companion_id"]).all()
    assert (df["n_invoices"] >= 30).all()
    assert (df["confidence"] >= 0.0).all() and (df["confidence"] <= 1.0).all()
    assert (df["lift"] > 0.0).all()
    assert (df.groupby("product_id")["companion_id"].count() <= 3).all()

