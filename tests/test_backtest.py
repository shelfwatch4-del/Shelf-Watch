"""Tests for replenishment-policy backtesting engine (src/backtest.py)."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.backtest import (
    C14_COLUMNS,
    C15_COLUMNS,
    POLICIES,
    run_policy_backtest,
    simulate_product_policy,
)

MOCK_DIR = Path("data/mock")


@pytest.fixture
def toy_predictions_and_meta() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create a toy 1-product 4-week dataset for deterministic tests."""
    weeks = ["2011-10-10", "2011-10-17", "2011-10-24", "2011-10-31"]
    records = []
    for pol in ("naive", "moving_avg_4", "best_model"):
        for w in weeks:
            records.append(
                {
                    "product_id": "PROD_A",
                    "week_start": w,
                    "actual": 50.0,
                    "predicted": 50.0,
                    "model_name": pol,
                    "split": "test",
                }
            )
    preds_df = pd.DataFrame(records)

    weekly_df = pd.DataFrame(
        [
            {
                "product_id": "PROD_A",
                "week_start": "2011-09-12",
                "units": 50,
                "orders": 5,
                "customers": 5,
                "avg_price": 2.0,
            },
            {
                "product_id": "PROD_A",
                "week_start": "2011-09-19",
                "units": 50,
                "orders": 5,
                "customers": 5,
                "avg_price": 2.0,
            },
            {
                "product_id": "PROD_A",
                "week_start": "2011-09-26",
                "units": 50,
                "orders": 5,
                "customers": 5,
                "avg_price": 2.0,
            },
            {
                "product_id": "PROD_A",
                "week_start": "2011-10-03",
                "units": 50,
                "orders": 5,
                "customers": 5,
                "avg_price": 2.0,
            },
        ]
    )

    meta_df = pd.DataFrame(
        [
            {
                "product_id": "PROD_A",
                "week_start": "2011-12-05",
                "forecast_weekly": 50.0,
                "forecast_p10": 30.0,
                "forecast_p90": 70.0,
                "sigma_error": 10.0,
                "avg_weekly_units_train": 50.0,
                "last_avg_price": 2.0,
            }
        ]
    )
    return preds_df, weekly_df, meta_df


def test_perfect_forecast_enough_stock_fill_rate_one(
    toy_predictions_and_meta: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> None:
    """With perfect forecasts and ample stock, fill rate must be 1.0 and stockouts 0."""
    preds_df, weekly_df, meta_df = toy_predictions_and_meta
    summary_df = run_policy_backtest(
        predictions=preds_df,
        weekly=weekly_df,
        forecast_meta=meta_df,
        lead_times=(1,),
    )
    assert not summary_df.empty
    for _, row in summary_df.iterrows():
        assert row["fill_rate"] == 1.0, f"Policy {row['policy']} fill rate is {row['fill_rate']}, expected 1.0"
        assert row["stockout_weeks"] == 0
        assert row["unmet_units"] == 0.0


def test_safety_stock_never_lowers_fill_rate() -> None:
    """Adding safety stock never lowers fill rate or increases unmet units in a toy case."""
    test_rows = pd.DataFrame(
        {
            "product_id": ["P1", "P1", "P1", "P1"],
            "week_start": ["2011-10-10", "2011-10-17", "2011-10-24", "2011-10-31"],
            "demand": [100.0, 150.0, 200.0, 250.0],
            "forecast": [80.0, 80.0, 80.0, 80.0],
        }
    )
    init_stock = 100.0
    lead_time = 1
    sigma = 30.0

    trace_base = simulate_product_policy(
        test_rows=test_rows,
        initial_stock=init_stock,
        lead_time=lead_time,
        policy="best_model",
        sigma=sigma,
    )
    trace_safety = simulate_product_policy(
        test_rows=test_rows,
        initial_stock=init_stock,
        lead_time=lead_time,
        policy="best_model_safety",
        sigma=sigma,
    )

    sold_base = sum(r["sold"] for r in trace_base)
    dem_base = sum(r["demand"] for r in trace_base)
    fill_base = sold_base / dem_base

    sold_safety = sum(r["sold"] for r in trace_safety)
    dem_safety = sum(r["demand"] for r in trace_safety)
    fill_safety = sold_safety / dem_safety

    assert fill_safety >= fill_base, f"Safety fill {fill_safety} < base fill {fill_base}"
    assert sum(r["unmet"] for r in trace_safety) <= sum(r["unmet"] for r in trace_base)


def test_backtest_determinism(
    toy_predictions_and_meta: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> None:
    """Backtest output is strictly deterministic for identical inputs."""
    preds_df, weekly_df, meta_df = toy_predictions_and_meta
    res1 = run_policy_backtest(preds_df, weekly_df, meta_df, lead_times=(1, 2))
    res2 = run_policy_backtest(preds_df, weekly_df, meta_df, lead_times=(1, 2))
    pd.testing.assert_frame_equal(res1, res2)


def test_no_future_leakage_in_forecast(
    toy_predictions_and_meta: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> None:
    """Decisions at week t depend only on week t forecast and prior orders, not future data."""
    test_rows_orig = pd.DataFrame(
        {
            "product_id": ["P1", "P1", "P1"],
            "week_start": ["2011-10-10", "2011-10-17", "2011-10-24"],
            "demand": [50.0, 60.0, 70.0],
            "forecast": [50.0, 60.0, 70.0],
        }
    )
    test_rows_perturbed = test_rows_orig.copy()
    # Drastically perturb future weeks (t=1 and t=2)
    test_rows_perturbed.loc[1:, "demand"] = [1000.0, 5000.0]
    test_rows_perturbed.loc[1:, "forecast"] = [2000.0, 8000.0]

    trace_orig = simulate_product_policy(test_rows_orig, 100.0, 1, "best_model", 10.0)
    trace_pert = simulate_product_policy(test_rows_perturbed, 100.0, 1, "best_model", 10.0)

    # Week 0 metrics MUST be completely unchanged
    assert trace_orig[0]["order_qty"] == trace_pert[0]["order_qty"]
    assert trace_orig[0]["sold"] == trace_pert[0]["sold"]
    assert trace_orig[0]["on_hand_end"] == trace_pert[0]["on_hand_end"]

    # Also verify via run_policy_backtest end-to-end
    preds_df, weekly_df, meta_df = toy_predictions_and_meta
    preds_orig = preds_df.copy()
    preds_pert = preds_df.copy()

    # Modify weeks after 2011-10-10 in predictions
    future_mask = preds_pert["week_start"] > "2011-10-10"
    preds_pert.loc[future_mask, "predicted"] = 9999.0
    preds_pert.loc[future_mask, "actual"] = 8888.0

    _, trace_df_orig = run_policy_backtest(preds_orig, weekly_df, meta_df, lead_times=(1,), return_weekly=True)
    _, trace_df_pert = run_policy_backtest(preds_pert, weekly_df, meta_df, lead_times=(1,), return_weekly=True)

    # For week 2011-10-10, every single metric must be identical across all policies
    w0_orig = trace_df_orig[trace_df_orig["week_start"] == "2011-10-10"].sort_values("policy").reset_index(drop=True)
    w0_pert = trace_df_pert[trace_df_pert["week_start"] == "2011-10-10"].sort_values("policy").reset_index(drop=True)

    pd.testing.assert_frame_equal(
        w0_orig[["policy", "lead_time_weeks", "on_hand_start", "order_qty", "arrived_qty", "sold", "unmet", "on_hand_end"]],
        w0_pert[["policy", "lead_time_weeks", "on_hand_start", "order_qty", "arrived_qty", "sold", "unmet", "on_hand_end"]],
    )


def test_backtest_abc_classes(
    toy_predictions_and_meta: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> None:
    """When intelligence is provided, results include ALL and per ABC class slices."""
    preds_df, weekly_df, meta_df = toy_predictions_and_meta
    intel_df = pd.DataFrame(
        [
            {
                "product_id": "PROD_A",
                "abc_class": "A",
            }
        ]
    )

    summary = run_policy_backtest(
        preds_df,
        weekly_df,
        meta_df,
        intelligence=intel_df,
        lead_times=(1,),
    )
    classes = set(summary["abc_class"].unique())
    assert "ALL" in classes
    assert "A" in classes


def test_backtest_without_intelligence(
    toy_predictions_and_meta: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> None:
    """When intelligence is None, results include only ALL slice."""
    preds_df, weekly_df, meta_df = toy_predictions_and_meta
    summary = run_policy_backtest(
        preds_df,
        weekly_df,
        meta_df,
        intelligence=None,
        lead_times=(1,),
    )
    classes = list(summary["abc_class"].unique())
    assert classes == ["ALL"]


def test_backtest_contract_schemas_and_bounds(
    toy_predictions_and_meta: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> None:
    """Validate Contract C14 and C15 column order, types, and metric bounds."""
    preds_df, weekly_df, meta_df = toy_predictions_and_meta
    summary_df, weekly_df_out = run_policy_backtest(
        preds_df,
        weekly_df,
        meta_df,
        lead_times=(1, 2),
        return_weekly=True,
    )

    assert list(summary_df.columns) == C14_COLUMNS
    assert list(weekly_df_out.columns) == C15_COLUMNS

    assert (summary_df["fill_rate"] >= 0.0).all() and (summary_df["fill_rate"] <= 1.0).all()
    assert (summary_df["stockout_weeks"] >= 0).all()
    assert (summary_df["avg_on_hand"] >= 0.0).all()
    assert (summary_df["unmet_units"] >= 0.0).all()
    assert set(summary_df["policy"].unique()).issubset(set(POLICIES))


def test_backtest_end_to_end_mock_data() -> None:
    """Run backtest on real mock files in data/mock/ if present."""
    preds_path = MOCK_DIR / "predictions.csv"
    weekly_path = MOCK_DIR / "weekly_demand.parquet"
    meta_path = MOCK_DIR / "next_week_forecast.csv"
    intel_path = MOCK_DIR / "product_intelligence.csv"

    if not (preds_path.exists() and weekly_path.exists() and meta_path.exists()):
        pytest.skip("Mock files not present for end-to-end test.")

    preds = pd.read_csv(preds_path)
    weekly = pd.read_parquet(weekly_path)
    meta = pd.read_csv(meta_path)
    intel = pd.read_csv(intel_path) if intel_path.exists() else None

    summary = run_policy_backtest(
        predictions=preds,
        weekly=weekly,
        forecast_meta=meta,
        intelligence=intel,
        lead_times=(1, 2),
    )

    assert not summary.empty
    assert list(summary.columns) == C14_COLUMNS
    assert "ALL" in summary["abc_class"].values
    if intel is not None:
        assert any(c in summary["abc_class"].values for c in ("A", "B", "C"))
