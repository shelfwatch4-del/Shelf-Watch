"""Unit tests for feature-group ablation on served model, paired differences, and group selection."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.ablation import (
    FEATURE_SETS,
    check_g3_prior_year_window,
    compute_paired_differences,
    decide_chosen_groups,
    load_tuned_candidate_params,
    run_ablation,
    save_ablation_artifacts,
)
from src.features import FEATURE_COLUMNS, FEATURE_GROUPS, build_custom_features
from src.train import make_model, run_ablation as train_run_ablation


def _make_synthetic_train_frame(n_products: int = 3, n_weeks: int = 15) -> pd.DataFrame:
    """Generate synthetic training data within training date range."""
    weeks = pd.date_range("2011-01-03", periods=n_weeks, freq="W-MON")
    rows: list[dict[str, object]] = []

    for p_idx in range(n_products):
        pid = f"PROD_{p_idx + 1}"
        for w_idx, week in enumerate(weeks):
            base_val = 10.0 + p_idx * 5.0 + w_idx * 1.5
            row: dict[str, object] = {
                "product_id": pid,
                "week_start": week,
                "target": float(base_val),
            }
            # Populate base columns
            for col in FEATURE_COLUMNS:
                row[col] = float(base_val + 0.5)
            # Populate G1
            for col in FEATURE_GROUPS["g1"]:
                row[col] = float(base_val * 0.2)
            # Populate G2
            for col in FEATURE_GROUPS["g2"]:
                row[col] = float(5.0)
            # Populate G3
            for col in FEATURE_GROUPS["g3"]:
                row[col] = float(base_val)
            # Populate G4
            for col in FEATURE_GROUPS["g4"]:
                row[col] = float(0.1)
            # Optional
            row["weeks_to_christmas"] = float(10.0)
            rows.append(row)

    return pd.DataFrame(rows)


def test_feature_sets_definitions() -> None:
    """FEATURE_SETS must contain required target sets without duplicate base_with_calendar."""
    expected_keys = {
        "base",
        "base+g1",
        "base+g2",
        "base+g3",
        "base+g4",
        "base+g1+g3",
        "all_groups",
        "base_no_calendar",
        "base+g3_no_calendar",
        "base+g1+g3_no_calendar",
        "all_groups_no_calendar",
        "base_plus_christmas",
    }
    assert expected_keys.issubset(set(FEATURE_SETS.keys()))
    # Duplicate base_with_calendar must be removed
    assert "base_with_calendar" not in FEATURE_SETS

    assert len(FEATURE_SETS["base"]) == 15
    assert len(FEATURE_SETS["base+g1"]) == 20
    assert len(FEATURE_SETS["base+g2"]) == 17
    assert len(FEATURE_SETS["base+g3"]) == 18
    assert len(FEATURE_SETS["base+g4"]) == 18
    assert len(FEATURE_SETS["base+g1+g3"]) == 23
    assert len(FEATURE_SETS["all_groups"]) == 28
    assert len(FEATURE_SETS["base_no_calendar"]) == 13
    assert len(FEATURE_SETS["base+g3_no_calendar"]) == 16
    assert len(FEATURE_SETS["base+g1+g3_no_calendar"]) == 21
    assert len(FEATURE_SETS["all_groups_no_calendar"]) == 26
    assert len(FEATURE_SETS["base_plus_christmas"]) == 16


def test_compute_paired_differences() -> None:
    """Paired difference calculation must correctly subtract base folds."""
    base_folds = [0.55, 0.50, 0.60, 0.45, 0.40]
    set_folds = [0.52, 0.51, 0.58, 0.44, 0.41]
    # diffs: [-0.03, +0.01, -0.02, -0.01, +0.01]
    diffs, mean_diff, improved = compute_paired_differences(base_folds, set_folds)

    assert len(diffs) == 5
    assert diffs[0] == pytest.approx(-0.03)
    assert diffs[1] == pytest.approx(0.01)
    assert diffs[2] == pytest.approx(-0.02)
    assert diffs[3] == pytest.approx(-0.01)
    assert diffs[4] == pytest.approx(0.01)
    assert mean_diff == pytest.approx(-0.008)
    assert improved == 3


def test_run_ablation_test_set_leakage_guard() -> None:
    """Ablation MUST raise ValueError if test period rows are present."""
    df = _make_synthetic_train_frame(n_products=2, n_weeks=5)
    df.loc[0, "week_start"] = pd.Timestamp("2011-10-10")

    with pytest.raises(ValueError, match="train_df contains rows on or after test_start"):
        run_ablation(train_df=df)


def test_run_ablation_synthetic_with_rf() -> None:
    """Run ablation on synthetic train frame with random_forest_raw."""
    df = _make_synthetic_train_frame(n_products=3, n_weeks=12)

    mock_candidates = {
        "random_forest_raw": {
            "model_name": "random_forest",
            "target_mode": "raw",
            "params": {"n_estimators": 10, "max_depth": 3, "min_samples_leaf": 2},
            "scale_col": None,
        }
    }
    subset_feature_sets = {
        "base": FEATURE_SETS["base"],
        "base+g3": FEATURE_SETS["base+g3"],
        "base+g3_no_calendar": FEATURE_SETS["base+g3_no_calendar"],
    }

    res_df = run_ablation(
        train_df=df,
        candidates=mock_candidates,
        feature_sets=subset_feature_sets,
        n_splits=3,
    )

    assert not res_df.empty
    assert set(res_df["groups"]) == {"base", "base+g3", "base+g3_no_calendar"}
    assert "val_wape" in res_df.columns
    assert "val_bias" in res_df.columns
    assert "paired_diff_per_fold" in res_df.columns
    assert "diff_vs_base" in res_df.columns


def test_decide_chosen_groups_served_model_rule() -> None:
    """decide_chosen_groups must apply 4/5 fold rule on random_forest_raw."""
    mock_records = [
        {
            "candidate": "random_forest_raw",
            "groups": "base",
            "val_wape": 0.5357,
            "val_bias": 2.25,
            "diff_vs_base": 0.0,
            "folds_improved": 0,
            "total_folds": 5,
            "paired_diff_per_fold": [0.0] * 5,
        },
        {
            "candidate": "random_forest_raw",
            "groups": "base+g3",
            "val_wape": 0.5318,
            "val_bias": 2.75,
            "diff_vs_base": -0.0039,
            "folds_improved": 4,
            "total_folds": 5,
            "paired_diff_per_fold": [0.0002, -0.0067, -0.0050, -0.0033, -0.0048],
        },
        {
            "candidate": "random_forest_raw",
            "groups": "base+g1+g3",
            "val_wape": 0.5324,
            "val_bias": 2.79,
            "diff_vs_base": -0.0034,
            "folds_improved": 4,
            "total_folds": 5,
            "paired_diff_per_fold": [0.0021, -0.0072, -0.0051, -0.0034, -0.0033],
        },
        {
            "candidate": "random_forest_raw",
            "groups": "all_groups",
            "val_wape": 0.5328,
            "val_bias": 3.18,
            "diff_vs_base": -0.0030,
            "folds_improved": 4,
            "total_folds": 5,
            "paired_diff_per_fold": [0.0015, -0.0079, -0.0023, -0.0040, -0.0022],
        },
        {
            "candidate": "random_forest_raw",
            "groups": "base_no_calendar",
            "val_wape": 0.5335,
            "val_bias": 1.27,
            "diff_vs_base": -0.0023,
            "folds_improved": 3,
            "total_folds": 5,
            "paired_diff_per_fold": [0.0021, -0.0086, -0.0001, -0.0049, 0.0002],
        },
        {
            "candidate": "random_forest_raw",
            "groups": "base+g3_no_calendar",
            "val_wape": 0.5301,
            "val_bias": 2.29,
            "diff_vs_base": -0.0056,
            "folds_improved": 4,
            "total_folds": 5,
            "paired_diff_per_fold": [0.0012, -0.0134, -0.0040, -0.0078, -0.0042],
        },
        {
            "candidate": "random_forest_raw",
            "groups": "base+g1+g3_no_calendar",
            "val_wape": 0.5307,
            "val_bias": 2.16,
            "diff_vs_base": -0.0051,
            "folds_improved": 4,
            "total_folds": 5,
            "paired_diff_per_fold": [0.0022, -0.0126, -0.0044, -0.0077, -0.0029],
        },
        {
            "candidate": "random_forest_raw",
            "groups": "all_groups_no_calendar",
            "val_wape": 0.5313,
            "val_bias": 2.70,
            "diff_vs_base": -0.0044,
            "folds_improved": 4,
            "total_folds": 5,
            "paired_diff_per_fold": [0.0019, -0.0122, -0.0018, -0.0082, -0.0017],
        },
    ]
    df = pd.DataFrame(mock_records)
    decision = decide_chosen_groups(df, primary_candidate="random_forest_raw")

    # Winner must be base+g3_no_calendar (lowest WAPE among sets improving >= 4/5 folds)
    assert decision["winning_feature_set"] == "base+g3_no_calendar"
    assert decision["winning_val_wape"] == 0.5301
    assert "g3" in decision["chosen_groups"]
    assert "base" in decision["chosen_groups"]
    assert "unproven_groups" in decision
    assert "g2" in decision["unproven_groups"]
    assert "g4" in decision["unproven_groups"]
    assert "guest_share_4w" in decision["hypotheses_and_warnings"]
    assert "overfitting_caution" in decision["hypotheses_and_warnings"]


def test_decide_chosen_groups_fallback_to_base() -> None:
    """If no set improves at least 4 of 5 folds, decision must keep base."""
    mock_records = [
        {
            "candidate": "random_forest_raw",
            "groups": "base",
            "val_wape": 0.5357,
            "val_bias": 2.25,
            "diff_vs_base": 0.0,
            "folds_improved": 0,
            "total_folds": 5,
            "paired_diff_per_fold": [0.0] * 5,
        },
        {
            "candidate": "random_forest_raw",
            "groups": "base+g3",
            "val_wape": 0.5340,
            "val_bias": 2.75,
            "diff_vs_base": -0.0017,
            "folds_improved": 3,  # only 3 folds!
            "total_folds": 5,
            "paired_diff_per_fold": [0.001, -0.005, -0.002, 0.001, -0.003],
        },
    ]
    df = pd.DataFrame(mock_records)
    decision = decide_chosen_groups(df, primary_candidate="random_forest_raw")
    assert decision["winning_feature_set"] == "base"
    assert decision["winning_val_wape"] == 0.5357


def test_custom_prediction_with_missing_g3_g4() -> None:
    """Fitted Random Forest model must successfully predict when g3/g4 are NaN (custom input)."""
    # 1. Generate custom row (g3 and g4 are NaN by contract)
    custom_row = build_custom_features(
        series=[15.0, 20.0, 18.0, 25.0, 30.0, 28.0, 35.0, 40.0],
        target_week="2011-12-05",
        groups=["base", "g1", "g2", "g3", "g4"],
    )
    assert pd.isna(custom_row["yoy_mean_5w"].iloc[0])
    assert pd.isna(custom_row["return_rate_4w"].iloc[0])

    # 2. Fit RF on training dummy data matching chosen columns
    chosen_cols = [c for c in custom_row.columns if c not in ("product_id", "week_start")]
    rng = np.random.RandomState(42)
    X_train = pd.DataFrame(rng.randn(25, len(chosen_cols)), columns=chosen_cols)
    y_train = rng.uniform(10.0, 50.0, size=25)

    model = make_model("random_forest", {"n_estimators": 10, "max_depth": 3})
    model.fit(X_train, y_train)

    # 3. Predict on custom input
    pred = model.predict(custom_row[chosen_cols])
    assert len(pred) == 1
    assert not np.isnan(pred[0])
    assert pred[0] > 0.0


def test_g3_window_audit_sample() -> None:
    """Audit function returns expected structure across target weeks."""
    audit_results = check_g3_prior_year_window()
    assert "2011-11-21" in audit_results
    assert "2011-11-28" in audit_results
    assert "2011-12-05" in audit_results
    # Forecast week must have NaN for yoy_mean_5w
    for rec in audit_results["2011-12-05"]:
        assert rec["yoy_available"] == 0.0
        assert np.isnan(rec["yoy_mean_5w"])


def test_train_module_reexport() -> None:
    """train.py must expose run_ablation per SRS Section 14."""
    assert callable(train_run_ablation)
