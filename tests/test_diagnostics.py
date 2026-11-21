"""Unit tests for src/diagnostics.py (Task M9)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.diagnostics import (
    analyze_tree_seasonality,
    compute_asymmetric_units,
    compute_intelligence_breakdowns,
    compute_weekly_candidate_metrics,
    select_ensemble_weight,
)


def test_compute_asymmetric_units_hand_computed() -> None:
    """Hand-computed test for over-forecast and under-forecast units.

    y_true = [10, 20, 30, 40]
    y_pred = [15, 18, 30, 50]
    diffs  = [+5, -2,  0, +10]
    over   = 5 + 0 + 0 + 10 = 15
    under  = 0 + 2 + 0 +  0 = 2
    """
    y_true = np.array([10.0, 20.0, 30.0, 40.0])
    y_pred = np.array([15.0, 18.0, 30.0, 50.0])

    res = compute_asymmetric_units(y_true, y_pred)
    assert res["over_forecast_units"] == pytest.approx(15.0)
    assert res["under_forecast_units"] == pytest.approx(2.0)


def test_compute_asymmetric_units_all_over_or_under() -> None:
    """All over or all under predictions."""
    y_true = np.array([10.0, 20.0])
    y_pred_over = np.array([12.0, 25.0])
    res_over = compute_asymmetric_units(y_true, y_pred_over)
    assert res_over["over_forecast_units"] == pytest.approx(7.0)
    assert res_over["under_forecast_units"] == pytest.approx(0.0)

    y_pred_under = np.array([8.0, 15.0])
    res_under = compute_asymmetric_units(y_true, y_pred_under)
    assert res_under["over_forecast_units"] == pytest.approx(0.0)
    assert res_under["under_forecast_units"] == pytest.approx(7.0)


def test_compute_weekly_candidate_metrics() -> None:
    """Test weekly candidate breakdown computes correct per-week WAPE and bias."""
    test_df = pd.DataFrame(
        {
            "week_start": [
                "2011-10-10", "2011-10-10",
                "2011-10-17", "2011-10-17",
            ],
            "target": [100.0, 200.0, 50.0, 150.0],
        }
    )
    test_preds = {
        "model_a": np.array([110.0, 190.0, 60.0, 140.0]),
        "model_b": np.array([120.0, 220.0, 40.0, 130.0]),
    }

    weekly = compute_weekly_candidate_metrics(test_df, test_preds)
    assert len(weekly) == 2
    assert list(weekly["week_start"]) == ["2011-10-10", "2011-10-17"]

    # Week 1 for model_a: actual=[100, 200], pred=[110, 190], sum_act=300, abs_err=10+10=20 -> WAPE = 20/300
    assert weekly.loc[0, "model_a_WAPE"] == pytest.approx(20.0 / 300.0)
    assert weekly.loc[0, "model_a_bias"] == pytest.approx(0.0)

    # Week 2 for model_b: actual=[50, 150], pred=[40, 130], sum_act=200, diffs=[-10, -20] -> bias=-15.0, WAPE=30/200
    assert weekly.loc[1, "model_b_WAPE"] == pytest.approx(30.0 / 200.0)
    assert weekly.loc[1, "model_b_bias"] == pytest.approx(-15.0)


def test_analyze_tree_seasonality() -> None:
    """Test October vs November tree seasonality analysis."""
    test_df = pd.DataFrame(
        {
            "week_start": [
                "2011-10-10", "2011-10-24",
                "2011-11-07", "2011-11-21",
            ],
            "target": [100.0, 100.0, 200.0, 200.0],
        }
    )
    test_preds = {
        "ridge_raw": np.array([100.0, 100.0, 200.0, 200.0]),
        "random_forest_raw": np.array([120.0, 110.0, 180.0, 190.0]),  # Oct over, Nov under
        "lightgbm_raw": np.array([130.0, 120.0, 210.0, 220.0]),       # Oct over, Nov over
    }

    tree_res = analyze_tree_seasonality(test_df, test_preds)
    assert "ridge_raw" not in tree_res
    assert "random_forest_raw" in tree_res
    assert "lightgbm_raw" in tree_res

    rf = tree_res["random_forest_raw"]
    assert rf["oct_bias"] > 0  # over-forecast in October
    assert rf["nov_bias"] < 0  # under-forecast in November

    lgb = tree_res["lightgbm_raw"]
    assert lgb["oct_bias"] > 0
    assert lgb["nov_bias"] > 0


def test_compute_intelligence_breakdowns() -> None:
    """Test error breakdown by demand pattern class and ABC class."""
    test_df = pd.DataFrame(
        {
            "product_id": ["P1", "P2", "P3"],
            "target": [100.0, 50.0, 20.0],
        }
    )
    intel_df = pd.DataFrame(
        {
            "product_id": ["P1", "P2", "P3"],
            "pattern_class": ["smooth", "erratic", "erratic"],
            "abc_class": ["A", "B", "C"],
        }
    )
    test_preds = {
        "cand1": np.array([110.0, 40.0, 30.0]),
    }

    res = compute_intelligence_breakdowns(test_df, test_preds, intel_df)
    assert "by_pattern" in res
    assert "by_abc" in res

    by_pat = res["by_pattern"]
    assert set(by_pat["pattern_class"]) == {"smooth", "erratic"}

    by_abc = res["by_abc"]
    assert set(by_abc["abc_class"]) == {"A", "B", "C"}
    p1_row = by_abc[by_abc["abc_class"] == "A"].iloc[0]
    assert p1_row["cand1_bias"] == pytest.approx(10.0)
    assert p1_row["cand1_over"] == pytest.approx(10.0)
    assert p1_row["cand1_under"] == pytest.approx(0.0)


def test_select_ensemble_weight_validation_only() -> None:
    """Test validation ensemble search selects the optimal w minimizing validation WAPE."""
    oof_val_df = pd.DataFrame(
        {
            "product_id": ["P1", "P1", "P2", "P2"],
            "week_start": ["2011-04-18", "2011-04-25", "2011-04-18", "2011-04-25"],
            "actual": [100.0, 100.0, 200.0, 200.0],
            "predicted": [90.0, 90.0, 210.0, 210.0],  # ML model
            "candidate": ["ridge_raw"] * 4,
        }
    )
    train_df = pd.DataFrame(
        {
            "product_id": ["P1", "P1", "P2", "P2"],
            "week_start": ["2011-04-18", "2011-04-25", "2011-04-18", "2011-04-25"],
            "roll_mean_4": [110.0, 110.0, 190.0, 190.0],  # MA4 model
        }
    )

    # ML error: [ -10, -10, +10, +10 ] -> abs_err = 40
    # MA4 error: [ +10, +10, -10, -10 ] -> abs_err = 40
    # At w=0.5: 0.5 * ML + 0.5 * MA4 = [100, 100, 200, 200] -> PERFECT PREDICTION (WAPE=0.0)
    res = select_ensemble_weight(
        oof_val_df,
        train_df,
        best_ml_candidate="ridge_raw",
        weight_steps=np.array([0.0, 0.2, 0.5, 0.8, 1.0]),
    )

    assert res["chosen_w"] == pytest.approx(0.5)
    assert res["best_val_metrics"]["WAPE"] == pytest.approx(0.0)
    assert res["best_val_metrics"]["bias"] == pytest.approx(0.0)
