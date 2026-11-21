"""Unit and leakage tests for src/features.py."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import pytest

from src.features import (
    FEATURE_COLUMNS,
    FEATURE_GROUPS,
    OPTIONAL_FEATURES,
    build_custom_features,
    build_features,
    build_future_features,
    report_feature_missingness,
)


def _make_weekly_fixture() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create a deterministic 2-product, 14-week fixture including 2010-12-27."""
    weeks = pd.date_range("2010-12-06", periods=14, freq="W-MON")
    rows: list[dict[str, object]] = []
    for pid, base in [("PROD_A", 10), ("PROD_B", 50)]:
        for idx, week in enumerate(weeks):
            units = 0 if week == pd.Timestamp("2010-12-27") else base + idx * 3
            rows.append(
                {
                    "product_id": pid,
                    "description": f"Desc {pid}",
                    "week_start": week,
                    "units": units,
                    "orders": 1 if units > 0 else 0,
                    "customers": 1 if units > 0 else 0,
                    "avg_price": 2.5,
                }
            )
    closed = pd.DataFrame({"week_start": ["2010-12-27"]})
    return pd.DataFrame(rows), closed


def _make_enrichment_fixture(
    weekly: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Create deterministic enrichment tables matching a weekly DataFrame fixture."""
    weeks = pd.date_range("2009-12-07", "2011-12-05", freq="W-MON")
    products = sorted(weekly["product_id"].unique())

    # 1. Calendar fixture
    cal_rows = []
    for w in weeks:
        cal_rows.append(
            {
                "week_start": w.strftime("%Y-%m-%d"),
                "uk_bank_holiday_count": 1 if w.month == 12 else 0,
                "trading_days_expected": 5 if w.month == 12 else 6,
                "is_christmas_week": 1 if (w.month == 12 and w.day in (24, 25, 26)) else 0,
            }
        )
    cal_df = pd.DataFrame(cal_rows)

    # 2. Prior-year fixture
    py_weeks = pd.date_range("2009-12-07", "2010-11-29", freq="W-MON")
    py_rows = []
    for pid in products:
        for idx, w in enumerate(py_weeks):
            py_rows.append(
                {
                    "product_id": pid,
                    "week_start": w,
                    "units": 20 + idx,
                    "orders": 2,
                }
            )
    py_df = pd.DataFrame(py_rows)

    # 3. Enrichment fixture (C8)
    enr_rows = []
    for row in weekly.itertuples(index=False):
        u = int(row.units)
        enr_rows.append(
            {
                "product_id": str(row.product_id),
                "week_start": pd.Timestamp(row.week_start),
                "gross_units": u,
                "cancelled_units": int(u * 0.05),
                "export_units": int(u * 0.10),
                "guest_units": int(u * 0.20),
            }
        )
    enr_df = pd.DataFrame(enr_rows)

    return {
        "calendar": cal_df,
        "prior_year": py_df,
        "enrichment": enr_df,
    }


def _assert_no_future_leakage(
    weekly: pd.DataFrame,
    closed: pd.DataFrame,
    products: list[str],
    n_weeks: int = 3,
) -> None:
    """Assert changing units, orders, customers, and avg_price at >= t leaves features at t identical."""
    baseline = build_features(weekly, closed)
    assert len(products) >= 3

    for target_product in products:
        prod_rows = baseline.loc[baseline["product_id"] == target_product]
        assert len(prod_rows) >= n_weeks
        sample_indices = np.linspace(0, len(prod_rows) - 1, n_weeks, dtype=int)
        target_weeks = [prod_rows["week_start"].iloc[int(i)] for i in sample_indices]

        for target_week in target_weeks:
            perturbed = weekly.copy()
            mask = (perturbed["product_id"] == target_product) & (
                pd.to_datetime(perturbed["week_start"]) >= target_week
            )
            perturbed.loc[mask, "units"] = perturbed.loc[mask, "units"] + 9999
            perturbed.loc[mask, "orders"] = perturbed.loc[mask, "orders"] + 50
            perturbed.loc[mask, "customers"] = perturbed.loc[mask, "customers"] + 40
            perturbed.loc[mask, "avg_price"] = perturbed.loc[mask, "avg_price"] + 99.0

            rebuilt = build_features(perturbed, closed)

            orig_at_t = baseline.loc[
                (baseline["product_id"] == target_product)
                & (baseline["week_start"] == target_week),
                FEATURE_COLUMNS,
            ].iloc[0]
            new_at_t = rebuilt.loc[
                (rebuilt["product_id"] == target_product)
                & (rebuilt["week_start"] == target_week),
                FEATURE_COLUMNS,
            ].iloc[0]

            pd.testing.assert_series_equal(orig_at_t, new_at_t)


def test_no_leakage_from_future() -> None:
    """Changing units, orders, customers, and avg_price at >= t must not alter features at t."""
    # 1. Small 3-product synthetic fixture
    weeks = pd.date_range("2010-12-06", periods=14, freq="W-MON")
    rows: list[dict[str, object]] = []
    for pid, base in [("PROD_A", 10), ("PROD_B", 50), ("PROD_C", 80)]:
        for idx, week in enumerate(weeks):
            units = 0 if week == pd.Timestamp("2010-12-27") else base + idx * 3
            rows.append(
                {
                    "product_id": pid,
                    "description": f"Desc {pid}",
                    "week_start": week,
                    "units": units,
                    "orders": 1 + idx if units > 0 else 0,
                    "customers": 1 + idx if units > 0 else 0,
                    "avg_price": 2.5 + idx * 0.1,
                }
            )
    fixture_weekly = pd.DataFrame(rows)
    fixture_closed = pd.DataFrame({"week_start": ["2010-12-27"]})
    _assert_no_future_leakage(
        fixture_weekly,
        fixture_closed,
        products=["PROD_A", "PROD_B", "PROD_C"],
        n_weeks=3,
    )

    # 2. Processed dataset (data/processed/weekly_demand.parquet + closed_weeks.csv)
    weekly_path = Path("data/processed/weekly_demand.parquet")
    closed_path = Path("data/processed/closed_weeks.csv")
    real_weekly = pd.read_parquet(weekly_path)
    real_closed = pd.read_csv(closed_path)
    real_products = sorted(real_weekly["product_id"].unique())[:3]
    _assert_no_future_leakage(
        real_weekly,
        real_closed,
        products=real_products,
        n_weeks=3,
    )


def test_closed_week_excluded_and_skipped() -> None:
    """Closed week 2010-12-27 is never a target and is skipped by lags and rolling."""
    weekly, closed = _make_weekly_fixture()
    extra_closed = pd.DataFrame({"week_start": ["2010-12-27", "2011-02-21"]})
    features = build_features(weekly, extra_closed)

    assert pd.Timestamp("2010-12-27") not in set(features["week_start"])
    assert pd.Timestamp("2011-02-21") not in set(features["week_start"])

    # Week right after 2011-02-21 is 2011-02-28; its lag_1 must come from 2011-02-14.
    prod_a_feb14 = float(
        weekly.loc[
            (weekly["product_id"] == "PROD_A")
            & (weekly["week_start"] == pd.Timestamp("2011-02-14")),
            "units",
        ].iloc[0]
    )
    row_feb28 = features.loc[
        (features["product_id"] == "PROD_A")
        & (features["week_start"] == pd.Timestamp("2011-02-28"))
    ].iloc[0]
    assert np.isclose(row_feb28["lag_1"], prod_a_feb14)


def test_history_drop_and_output_schema(capsys) -> None:
    """Verify output columns, zero NaNs, and dropped-row reporting."""
    weekly, closed = _make_weekly_fixture()
    features = build_features(weekly, closed)
    captured = capsys.readouterr().out

    expected_cols = ["product_id", "week_start", "target", *FEATURE_COLUMNS]
    assert list(features.columns) == expected_cols
    assert not features.isna().any().any()

    # 14 weeks - 1 closed week = 13 active weeks per product; 8 dropped -> 5 kept each.
    assert len(features) == 10
    assert "Total rows dropped: 16" in captured


def test_no_same_week_leakage_commercial_and_history() -> None:
    """Changing same-week units, orders, customers, or avg_price must not alter features at t."""
    weekly, closed = _make_weekly_fixture()
    baseline = build_features(weekly, closed)

    target_product = "PROD_B"
    target_week = baseline.loc[
        baseline["product_id"] == target_product, "week_start"
    ].iloc[2]

    perturbed = weekly.copy()
    mask = (perturbed["product_id"] == target_product) & (
        pd.to_datetime(perturbed["week_start"]) >= target_week
    )
    perturbed.loc[mask, "units"] = perturbed.loc[mask, "units"] + 500
    perturbed.loc[mask, "orders"] = perturbed.loc[mask, "orders"] + 50
    perturbed.loc[mask, "customers"] = perturbed.loc[mask, "customers"] + 40
    perturbed.loc[mask, "avg_price"] = perturbed.loc[mask, "avg_price"] + 99.0

    rebuilt = build_features(perturbed, closed)

    orig_at_t = baseline.loc[
        (baseline["product_id"] == target_product)
        & (baseline["week_start"] == target_week),
        FEATURE_COLUMNS,
    ].iloc[0]
    new_at_t = rebuilt.loc[
        (rebuilt["product_id"] == target_product)
        & (rebuilt["week_start"] == target_week),
        FEATURE_COLUMNS,
    ].iloc[0]

    pd.testing.assert_series_equal(orig_at_t, new_at_t)


def test_feature_columns_match_output_order() -> None:
    """FEATURE_COLUMNS matches produced feature columns in exact order; OPTIONAL_FEATURES excluded."""
    weekly, closed = _make_weekly_fixture()
    features = build_features(weekly, closed)

    expected_cols = ["product_id", "week_start", "target", *FEATURE_COLUMNS]
    assert list(features.columns) == expected_cols
    assert OPTIONAL_FEATURES == ["weeks_to_christmas"]
    for opt in OPTIONAL_FEATURES:
        assert opt not in FEATURE_COLUMNS
        assert opt not in features.columns


def test_build_future_features_no_leakage_and_schema() -> None:
    """build_future_features for 2011-12-05 ignores >= 2011-12-05 data and matches feature columns."""
    weeks = pd.date_range("2011-09-12", "2011-12-12", freq="W-MON")
    rows: list[dict[str, object]] = []
    for pid, base in [("PROD_A", 20), ("PROD_B", 40)]:
        for idx, week in enumerate(weeks):
            rows.append(
                {
                    "product_id": pid,
                    "description": f"Desc {pid}",
                    "week_start": week,
                    "units": base + idx * 2,
                    "orders": 2 + idx,
                    "customers": 1 + idx,
                    "avg_price": 3.0 + idx * 0.1,
                }
            )
    weekly = pd.DataFrame(rows)
    closed = pd.DataFrame({"week_start": ["2010-12-27"]})

    hist_features = build_features(weekly, closed)
    future_features = build_future_features(weekly, closed, target_week="2011-12-05")

    assert "target" not in future_features.columns
    assert list(future_features.columns) == [c for c in hist_features.columns if c != "target"]
    assert list(future_features.columns) == ["product_id", "week_start", *FEATURE_COLUMNS]
    assert len(future_features) == 2
    assert (future_features["week_start"] == pd.Timestamp("2011-12-05")).all()
    assert not future_features.isna().any().any()

    # Mutate data on and after 2011-12-05: future features must be identical.
    perturbed = weekly.copy()
    future_mask = pd.to_datetime(perturbed["week_start"]) >= pd.Timestamp("2011-12-05")
    perturbed.loc[future_mask, "units"] = 99999
    perturbed.loc[future_mask, "orders"] = 9999
    perturbed.loc[future_mask, "customers"] = 9999
    perturbed.loc[future_mask, "avg_price"] = 999.0

    rebuilt_future = build_future_features(perturbed, closed, target_week="2011-12-05")
    pd.testing.assert_frame_equal(future_features, rebuilt_future)


# ===========================================================================
# Task M6 Tests: Switchable Groups, Extended Leakage, Custom Features
# ===========================================================================


def test_feature_groups_switch_and_column_order() -> None:
    """Each feature group switches on and off by name with stable column order."""
    weekly, closed = _make_weekly_fixture()

    # 1. Base group only
    base_df = build_features(weekly, closed, groups=("base",))
    assert list(base_df.columns) == ["product_id", "week_start", "target", *FEATURE_GROUPS["base"]]
    assert len(base_df.columns) == 3 + 15

    # 2. G1 group only
    g1_df = build_features(weekly, closed, groups=("g1",))
    assert list(g1_df.columns) == ["product_id", "week_start", "target", *FEATURE_GROUPS["g1"]]
    assert len(g1_df.columns) == 3 + 5

    # 3. G2 group only
    cal_fixture = pd.DataFrame(
        {
            "week_start": [w.strftime("%Y-%m-%d") for w in weekly["week_start"].unique()],
            "uk_bank_holiday_count": [0] * len(weekly["week_start"].unique()),
            "trading_days_expected": [6] * len(weekly["week_start"].unique()),
        }
    )
    g2_df = build_features(weekly, closed, enrichment={"calendar": cal_fixture}, groups=("g2",))
    assert list(g2_df.columns) == ["product_id", "week_start", "target", *FEATURE_GROUPS["g2"]]
    assert len(g2_df.columns) == 3 + 2

    # 4. Base + G1
    base_g1_df = build_features(weekly, closed, groups=("base", "g1"))
    expected_base_g1 = ["product_id", "week_start", "target", *FEATURE_GROUPS["base"], *FEATURE_GROUPS["g1"]]
    assert list(base_g1_df.columns) == expected_base_g1
    assert len(base_g1_df.columns) == 3 + 15 + 5

    # 5. All groups
    enr_fixture = _make_enrichment_fixture(weekly)
    all_df = build_features(
        weekly,
        closed,
        enrichment=enr_fixture,
        groups=("base", "g1", "g2", "g3", "g4"),
    )
    expected_all = [
        "product_id",
        "week_start",
        "target",
        *FEATURE_GROUPS["base"],
        *FEATURE_GROUPS["g1"],
        *FEATURE_GROUPS["g2"],
        *FEATURE_GROUPS["g3"],
        *FEATURE_GROUPS["g4"],
    ]
    assert list(all_df.columns) == expected_all
    assert len(all_df.columns) == 3 + 15 + 5 + 2 + 3 + 3


def _assert_extended_no_future_leakage(
    weekly: pd.DataFrame,
    closed: pd.DataFrame,
    enrichment: dict[str, pd.DataFrame] | None,
    products: list[str],
    n_weeks: int = 3,
) -> None:
    """Assert changing demand AND enrichment at >= t leaves ALL feature groups at t identical."""
    all_groups = ("base", "g1", "g2", "g3", "g4")
    baseline = build_features(
        weekly=weekly,
        closed_weeks=closed,
        enrichment=enrichment,
        groups=all_groups,
    )
    assert len(products) >= 3

    feature_cols = [
        *FEATURE_GROUPS["base"],
        *FEATURE_GROUPS["g1"],
        *FEATURE_GROUPS["g2"],
        *FEATURE_GROUPS["g3"],
        *FEATURE_GROUPS["g4"],
    ]

    for target_product in products:
        prod_rows = baseline.loc[baseline["product_id"] == target_product]
        assert len(prod_rows) >= n_weeks
        sample_indices = np.linspace(0, len(prod_rows) - 1, n_weeks, dtype=int)
        target_weeks = [prod_rows["week_start"].iloc[int(i)] for i in sample_indices]

        for target_week in target_weeks:
            # 1. Perturb weekly demand table at >= target_week
            perturbed_weekly = weekly.copy()
            w_mask = (perturbed_weekly["product_id"] == target_product) & (
                pd.to_datetime(perturbed_weekly["week_start"]) >= target_week
            )
            perturbed_weekly.loc[w_mask, "units"] = perturbed_weekly.loc[w_mask, "units"] + 9999
            perturbed_weekly.loc[w_mask, "orders"] = perturbed_weekly.loc[w_mask, "orders"] + 50
            perturbed_weekly.loc[w_mask, "customers"] = perturbed_weekly.loc[w_mask, "customers"] + 40
            perturbed_weekly.loc[w_mask, "avg_price"] = perturbed_weekly.loc[w_mask, "avg_price"] + 99.0

            # 2. Perturb enrichment table at >= target_week
            perturbed_enrichment: dict[str, pd.DataFrame] | None = None
            if enrichment is not None:
                perturbed_enrichment = {k: v.copy() for k, v in enrichment.items()}
                enr_key = "enrichment" if "enrichment" in perturbed_enrichment else "enrichment_weekly"
                if enr_key in perturbed_enrichment:
                    enr_tbl = perturbed_enrichment[enr_key]
                    e_mask = (enr_tbl["product_id"].astype(str) == target_product) & (
                        pd.to_datetime(enr_tbl["week_start"]) >= target_week
                    )
                    enr_tbl.loc[e_mask, "gross_units"] = enr_tbl.loc[e_mask, "gross_units"] + 9999
                    enr_tbl.loc[e_mask, "cancelled_units"] = enr_tbl.loc[e_mask, "cancelled_units"] + 500
                    enr_tbl.loc[e_mask, "export_units"] = enr_tbl.loc[e_mask, "export_units"] + 200
                    enr_tbl.loc[e_mask, "guest_units"] = enr_tbl.loc[e_mask, "guest_units"] + 300

            # 3. Rebuild features
            rebuilt = build_features(
                weekly=perturbed_weekly,
                closed_weeks=closed,
                enrichment=perturbed_enrichment,
                groups=all_groups,
            )

            # 4. Compare every feature column at target_week
            orig_at_t = baseline.loc[
                (baseline["product_id"] == target_product)
                & (baseline["week_start"] == target_week),
                feature_cols,
            ].iloc[0]
            new_at_t = rebuilt.loc[
                (rebuilt["product_id"] == target_product)
                & (rebuilt["week_start"] == target_week),
                feature_cols,
            ].iloc[0]

            pd.testing.assert_series_equal(orig_at_t, new_at_t)


def test_extended_leakage_fixture_and_real() -> None:
    """Extended leakage test passes across all groups for 3 products and 3 weeks on fixture and real tables."""
    # 1. Synthetic 3-product fixture
    weeks = pd.date_range("2010-12-06", periods=16, freq="W-MON")
    rows: list[dict[str, object]] = []
    for pid, base in [("PROD_A", 15), ("PROD_B", 45), ("PROD_C", 75)]:
        for idx, week in enumerate(weeks):
            units = 0 if week == pd.Timestamp("2010-12-27") else base + idx * 2
            rows.append(
                {
                    "product_id": pid,
                    "description": f"Desc {pid}",
                    "week_start": week,
                    "units": units,
                    "orders": 1 + idx if units > 0 else 0,
                    "customers": 1 + idx if units > 0 else 0,
                    "avg_price": 2.5 + idx * 0.1,
                }
            )
    fixture_weekly = pd.DataFrame(rows)
    fixture_closed = pd.DataFrame({"week_start": ["2010-12-27"]})
    fixture_enr = _make_enrichment_fixture(fixture_weekly)

    _assert_extended_no_future_leakage(
        weekly=fixture_weekly,
        closed=fixture_closed,
        enrichment=fixture_enr,
        products=["PROD_A", "PROD_B", "PROD_C"],
        n_weeks=3,
    )

    # 2. Real processed tables
    real_weekly = pd.read_parquet("data/processed/weekly_demand.parquet")
    real_closed = pd.read_csv("data/processed/closed_weeks.csv")
    real_products = sorted(real_weekly["product_id"].unique())[:3]

    real_enrichment = {
        "calendar": pd.read_csv("data/processed/calendar_weekly.csv"),
        "prior_year": pd.read_parquet("data/processed/prior_year_weekly.parquet"),
        "enrichment": pd.read_parquet("data/processed/enrichment_weekly.parquet"),
    }

    _assert_extended_no_future_leakage(
        weekly=real_weekly,
        closed=real_closed,
        enrichment=real_enrichment,
        products=real_products,
        n_weeks=3,
    )


def test_yoy_features_window() -> None:
    """YoY features use strictly weeks t-54 to t-50, requiring at least 3 values."""
    target_week = pd.Timestamp("2011-11-28")
    weeks_window = [target_week - pd.Timedelta(weeks=w) for w in (54, 53, 52, 51, 50)]

    # 1. Product with 4 weeks available in t-54..t-50
    py_rows = [
        {"product_id": "TEST_P", "week_start": weeks_window[0], "units": 100, "orders": 5},
        {"product_id": "TEST_P", "week_start": weeks_window[1], "units": 120, "orders": 6},
        {"product_id": "TEST_P", "week_start": weeks_window[2], "units": 80, "orders": 4},
        {"product_id": "TEST_P", "week_start": weeks_window[3], "units": 100, "orders": 5},
        # Other weeks far away (outside t-54..t-50)
        {"product_id": "TEST_P", "week_start": target_week - pd.Timedelta(weeks=60), "units": 99999, "orders": 1},
        {"product_id": "TEST_P", "week_start": target_week - pd.Timedelta(weeks=45), "units": 88888, "orders": 1},
    ]
    py_df = pd.DataFrame(py_rows)

    hist_weeks = pd.date_range("2010-12-06", "2011-11-28", freq="W-MON")
    demand_rows = [
        {
            "product_id": "TEST_P",
            "week_start": w,
            "units": 50,
            "orders": 5,
            "customers": 5,
            "avg_price": 2.0,
        }
        for w in hist_weeks
    ]
    weekly = pd.DataFrame(demand_rows)

    res = build_features(
        weekly=weekly,
        closed_weeks=None,
        enrichment={"prior_year": py_df},
        groups=("g3",),
    )
    row_t = res.loc[res["week_start"] == target_week].iloc[0]

    # Mean of the 4 window values: (100 + 120 + 80 + 100) / 4 = 100.0
    assert np.isclose(row_t["yoy_mean_5w"], 100.0)
    assert row_t["yoy_available"] == 1.0

    # 2. Product with only 2 weeks available (< 3): must be NaN
    py_sparse = py_df.iloc[:2].copy()  # Only 2 rows in window
    res_sparse = build_features(
        weekly=weekly,
        closed_weeks=None,
        enrichment={"prior_year": py_sparse},
        groups=("g3",),
    )
    row_sparse_t = res_sparse.loc[res_sparse["week_start"] == target_week].iloc[0]
    assert pd.isna(row_sparse_t["yoy_mean_5w"])
    assert pd.isna(row_sparse_t["yoy_seasonal_index"])
    assert row_sparse_t["yoy_available"] == 0.0


def test_custom_features_equal_table_features() -> None:
    """build_custom_features matches build_future_features for base, g1, g2; g3 and g4 are NaN."""
    weekly = pd.read_parquet("data/processed/weekly_demand.parquet")
    product_id = "21212"
    target_week = "2011-12-05"

    prod_hist = (
        weekly.loc[weekly["product_id"] == product_id]
        .sort_values("week_start")
        .copy()
    )

    # 1. Custom features from full product table history
    custom = build_custom_features(
        series=prod_hist,
        target_week=target_week,
    )

    # 2. Table future features
    future = build_future_features(
        weekly=weekly,
        closed_weeks=None,
        target_week=target_week,
        groups=("base", "g1", "g2", "g3", "g4"),
    )
    future_p = future.loc[future["product_id"] == product_id].iloc[0]

    # Compare base, g1, g2 features
    shared_cols = [
        *FEATURE_GROUPS["base"],
        *FEATURE_GROUPS["g1"],
        *FEATURE_GROUPS["g2"],
    ]
    for col in shared_cols:
        val_custom = custom[col].iloc[0]
        val_future = future_p[col]
        assert np.isclose(val_custom, val_future, equal_nan=True), f"Mismatch in feature {col}"

    # Verify g3 and g4 are missing (NaN)
    for g in ("g3", "g4"):
        for col in FEATURE_GROUPS[g]:
            assert pd.isna(custom[col].iloc[0]), f"Column {col} should be NaN in custom features"

    # Also test with a short 8-12 numeric list or dict (user-entered series)
    dict_series = {
        "recent_units": [20.0, 25.0, 30.0, 15.0, 40.0, 35.0, 50.0, 45.0],
        "recent_prices": [2.5] * 8,
        "recent_orders": [2.0] * 8,
        "recent_customers": [2.0] * 8,
    }
    custom_dict = build_custom_features(series=dict_series, target_week=target_week)
    assert len(custom_dict) == 1
    for col in shared_cols:
        assert not pd.isna(custom_dict[col].iloc[0]), f"Expected non-null value for {col}"
    for g in ("g3", "g4"):
        for col in FEATURE_GROUPS[g]:
            assert pd.isna(custom_dict[col].iloc[0])


def test_print_feature_missingness_report(capsys) -> None:
    """Row counts and share of missing values per feature are computed and reported."""
    weekly = pd.read_parquet("data/processed/weekly_demand.parquet")
    closed = pd.read_csv("data/processed/closed_weeks.csv")
    features = build_features(
        weekly=weekly,
        closed_weeks=closed,
        groups=("base", "g1", "g2", "g3", "g4"),
    )
    assert len(features) == 4267

    stats = report_feature_missingness(features)
    captured = capsys.readouterr().out

    assert "Total rows: 4267" in captured
    assert "Group [base]" in captured
    assert "Group [g1]" in captured
    assert "Group [g2]" in captured
    assert "Group [g3]" in captured
    assert "Group [g4]" in captured

    # Verify base, g1, g2, g4 have 0% missingness
    for col in FEATURE_GROUPS["base"]:
        assert stats["base"][col] == 0.0
    for col in FEATURE_GROUPS["g1"]:
        assert stats["g1"][col] == 0.0
    for col in FEATURE_GROUPS["g2"]:
        assert stats["g2"][col] == 0.0
    for col in FEATURE_GROUPS["g4"]:
        assert stats["g4"][col] == 0.0

    # Verify g3 has expected missingness
    assert stats["g3"]["yoy_mean_5w"] > 0.0
    assert stats["g3"]["yoy_available"] == 0.0  # yoy_available is 0 or 1, not NaN
