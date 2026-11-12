"""Unit and leakage tests for src/features.py."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.features import (
    FEATURE_COLUMNS,
    OPTIONAL_FEATURES,
    build_features,
    build_future_features,
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
    # Add enough weeks after 2010-12-27; also test a closure inside the kept range.
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
