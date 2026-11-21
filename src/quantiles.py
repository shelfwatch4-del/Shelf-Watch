"""Quantile models and prediction intervals for Shelfwatch (Task M10, SRS FR-M-13).

Trains two LightGBM quantile models (alpha 0.1 and 0.9) on chosen features,
evaluates out-of-fold validation quantiles, refits on the full training period,
enforces 0 <= p10 <= point forecast <= p90, applies residual-based calibration to achieve
target ~80% empirical coverage, and serializes artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
from lightgbm import LGBMRegressor
import numpy as np
import pandas as pd

from src.features import build_features
from src.train import (
    DEFAULT_TEST_START,
    RANDOM_SEED,
    make_model,
    predict_moving_avg_4,
    split_chronological,
)
from src.tune import WeekTimeSeriesSplit

DATA_PROCESSED_DIR: Path = Path("data/processed")
MODELS_DIR: Path = Path("models")


def pinball_loss(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    alpha: float,
) -> float:
    """Compute pinball loss for quantile forecast at level alpha in (0, 1).

    L_alpha(y, q) = max(alpha * (y - q), (alpha - 1) * (y - q))

    Parameters
    ----------
    y_true:
        Ground truth actual values.
    y_pred:
        Predicted quantiles.
    alpha:
        Target quantile level in (0, 1).

    Returns
    -------
    float
        Mean pinball loss across all observations.
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}.")

    actual = np.asarray(y_true, dtype=np.float64).ravel()
    predicted = np.asarray(y_pred, dtype=np.float64).ravel()

    if actual.shape != predicted.shape:
        raise ValueError(
            f"Shape mismatch: y_true {actual.shape} vs y_pred {predicted.shape}."
        )

    err = actual - predicted
    loss = np.maximum(alpha * err, (alpha - 1.0) * err)
    return float(np.mean(loss))


def empirical_coverage(
    y_true: np.ndarray | pd.Series,
    p10: np.ndarray | pd.Series,
    p90: np.ndarray | pd.Series,
) -> float:
    """Compute empirical coverage: fraction of actuals inside [p10, p90].

    Parameters
    ----------
    y_true:
        Ground truth actual values.
    p10:
        Lower bound (10th percentile forecast).
    p90:
        Upper bound (90th percentile forecast).

    Returns
    -------
    float
        Empirical coverage in [0, 1].
    """
    actual = np.asarray(y_true, dtype=np.float64).ravel()
    lower = np.asarray(p10, dtype=np.float64).ravel()
    upper = np.asarray(p90, dtype=np.float64).ravel()

    if not (actual.shape == lower.shape == upper.shape):
        raise ValueError("Arrays actual, p10, and p90 must have identical shapes.")
    if len(actual) == 0:
        return 0.0

    inside = (actual >= lower) & (actual <= upper)
    return float(np.mean(inside))


def sort_and_clip_intervals(
    p10: np.ndarray | pd.Series,
    point: np.ndarray | pd.Series,
    p90: np.ndarray | pd.Series,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sort and clip prediction intervals row-wise to enforce 0 <= p10 <= point <= p90.

    Parameters
    ----------
    p10:
        Raw 10th percentile predictions.
    point:
        Point forecast predictions.
    p90:
        Raw 90th percentile predictions.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        (p10_sorted, point_clipped, p90_sorted) all clipped at 0 with ordering guaranteed.
    """
    low = np.asarray(p10, dtype=np.float64).ravel()
    pt = np.asarray(point, dtype=np.float64).ravel()
    high = np.asarray(p90, dtype=np.float64).ravel()

    if not (len(low) == len(pt) == len(high)):
        raise ValueError("p10, point, and p90 must have the same length.")

    # Stack along axis 1 and sort each row
    stacked = np.stack([low, pt, high], axis=1)
    sorted_vals = np.sort(stacked, axis=1)

    # Sorted order guarantees col 0 <= col 1 <= col 2
    # Ensure point forecast is preserved if already within [p10, p90], else clamp
    # As per prompt: "Sort p10 <= point forecast <= p90 after prediction and clip at 0."
    p10_res = np.clip(sorted_vals[:, 0], 0.0, None)
    point_res = np.clip(sorted_vals[:, 1], 0.0, None)
    p90_res = np.clip(sorted_vals[:, 2], 0.0, None)

    return p10_res, point_res, p90_res


def make_quantile_estimator(alpha: float, seed: int = RANDOM_SEED) -> LGBMRegressor:
    """Instantiate a reproducible LightGBM quantile regressor."""
    return LGBMRegressor(
        objective="quantile",
        alpha=alpha,
        random_state=seed,
        verbosity=-1,
        deterministic=True,
        n_jobs=1,
    )


def compute_oof_quantiles(
    train_df: pd.DataFrame,
    features: list[str],
    target_col: str = "target",
    n_splits: int = 5,
    seed: int = RANDOM_SEED,
) -> dict[str, np.ndarray]:
    """Generate out-of-fold validation quantile predictions across expanding time splits."""
    splitter = WeekTimeSeriesSplit(n_splits=n_splits)
    splits = splitter.split(train_df)

    oof_p10 = np.full(len(train_df), np.nan, dtype=np.float64)
    oof_p90 = np.full(len(train_df), np.nan, dtype=np.float64)
    val_indices: list[int] = []

    for fold_idx, (tr_idx, val_idx) in enumerate(splits):
        val_indices.extend(val_idx)
        X_tr = train_df.iloc[tr_idx][features]
        y_tr = train_df.iloc[tr_idx][target_col].to_numpy(dtype=np.float64)
        X_val = train_df.iloc[val_idx][features]

        q10 = make_quantile_estimator(alpha=0.1, seed=seed)
        q90 = make_quantile_estimator(alpha=0.9, seed=seed)

        q10.fit(X_tr, y_tr)
        q90.fit(X_tr, y_tr)

        oof_p10[val_idx] = q10.predict(X_val)
        oof_p90[val_idx] = q90.predict(X_val)

    val_idx_arr = np.array(val_indices, dtype=int)
    return {
        "val_indices": val_idx_arr,
        "p10_oof": oof_p10[val_idx_arr],
        "p90_oof": oof_p90[val_idx_arr],
        "actual_oof": train_df.iloc[val_idx_arr][target_col].to_numpy(dtype=np.float64),
    }


def compute_conformal_correction(
    y_val: np.ndarray,
    p10_val: np.ndarray,
    p90_val: np.ndarray,
    target_coverage: float = 0.80,
) -> float:
    """Compute simple residual-based conformal correction offset c.

    Non-conformity score for each validation observation:
        s_i = max(p10_i - y_i, y_i - p90_i)
    The offset c is the target_coverage quantile of s_i.
    Expanding intervals by c: [max(0, p10 - c), p90 + c] achieves target_coverage on validation.

    Parameters
    ----------
    y_val:
        Validation actual values.
    p10_val:
        Validation p10 predictions.
    p90_val:
        Validation p90 predictions.
    target_coverage:
        Desired coverage level (default 0.80).

    Returns
    -------
    float
        Scalar offset c >= 0.0.
    """
    scores = np.maximum(p10_val - y_val, y_val - p90_val)
    c = float(np.quantile(scores, target_coverage))
    return max(0.0, c)


def train_and_evaluate_quantiles(
    data_dir: Path = DATA_PROCESSED_DIR,
    models_dir: Path = MODELS_DIR,
    save_artifacts: bool = True,
) -> dict[str, Any]:
    """Train LightGBM quantile models, evaluate coverage/pinball, calibrate, and save models."""
    print("=" * 85)
    print("TASK M10: QUANTILE MODELS & PREDICTION INTERVALS (SRS FR-M-13)")
    print("=" * 85)

    weekly = pd.read_parquet(data_dir / "weekly_demand.parquet")
    closed = pd.read_csv(data_dir / "closed_weeks.csv")

    chosen_path = models_dir / "chosen_groups.json"
    with open(chosen_path, encoding="utf-8") as f:
        chosen_meta = json.load(f)
    chosen_groups = tuple(chosen_meta.get("chosen_groups", ["base", "g3"]))
    features = list(chosen_meta.get("chosen_feature_list", []))

    features_df = build_features(weekly, closed_weeks=closed, groups=chosen_groups)
    train_df, test_df = split_chronological(features_df, test_start=DEFAULT_TEST_START)

    print(f"Features ({len(features)}): {features}")
    print(f"Training rows: {len(train_df)}, Test rows: {len(test_df)}")

    # 1. Out-of-fold validation quantiles
    print("\n1. Computing 5-fold Out-of-Fold Validation Quantiles...")
    oof_res = compute_oof_quantiles(train_df, features=features)
    y_val = oof_res["actual_oof"]
    p10_val_raw = oof_res["p10_oof"]
    p90_val_raw = oof_res["p90_oof"]

    # Point forecast on validation: use rolling mean 4 or simple point estimate for sorting check
    val_point = train_df.iloc[oof_res["val_indices"]]["roll_mean_4"].fillna(0.0).to_numpy(dtype=np.float64)
    p10_val_sorted, pt_val_sorted, p90_val_sorted = sort_and_clip_intervals(
        p10_val_raw, val_point, p90_val_raw
    )

    val_cov_raw = empirical_coverage(y_val, p10_val_sorted, p90_val_sorted)
    val_pinball_10_raw = pinball_loss(y_val, p10_val_sorted, alpha=0.1)
    val_pinball_90_raw = pinball_loss(y_val, p90_val_sorted, alpha=0.9)
    print(f"  Validation Coverage (Raw Uncalibrated): {val_cov_raw * 100:.2f}% (Target: ~80%)")
    print(f"  Validation Pinball Loss (Raw): p10={val_pinball_10_raw:.2f}, p90={val_pinball_90_raw:.2f}")

    # 2. Residual-based Conformal Correction
    c_offset = 0.0
    if abs(val_cov_raw - 0.80) > 0.05:
        print("\n2. Applying Residual-Based Conformal Correction...")
        c_offset = compute_conformal_correction(
            y_val, p10_val_sorted, p90_val_sorted, target_coverage=0.80
        )
        print(
            f"  Documenting Correction: Validation residuals revealed under-coverage ({val_cov_raw * 100:.2f}% < 80%).\n"
            f"  Computed non-conformity score s_i = max(p10_i - y_i, y_i - p90_i).\n"
            f"  Empirical 80th percentile of residuals yields offset c = {c_offset:.2f} units.\n"
            f"  Calibrated interval: [max(0, p10 - c), p90 + c]."
        )

    p10_val_cal, pt_val_cal, p90_val_cal = sort_and_clip_intervals(
        p10_val_sorted - c_offset, pt_val_sorted, p90_val_sorted + c_offset
    )
    val_cov_cal = empirical_coverage(y_val, p10_val_cal, p90_val_cal)
    val_pinball_10_cal = pinball_loss(y_val, p10_val_cal, alpha=0.1)
    val_pinball_90_cal = pinball_loss(y_val, p90_val_cal, alpha=0.9)
    print(f"  Validation Coverage (Calibrated): {val_cov_cal * 100:.2f}%")
    print(f"  Validation Pinball Loss (Calibrated): p10={val_pinball_10_cal:.2f}, p90={val_pinball_90_cal:.2f}")

    # 3. Refit on full training period and predict on test
    print("\n3. Refitting Quantile Models on Full Training Period...")
    q10_model = make_quantile_estimator(alpha=0.1, seed=RANDOM_SEED)
    q90_model = make_quantile_estimator(alpha=0.9, seed=RANDOM_SEED)

    X_train = train_df[features]
    y_train = train_df["target"].to_numpy(dtype=np.float64)
    q10_model.fit(X_train, y_train)
    q90_model.fit(X_train, y_train)

    X_test = test_df[features]
    y_test = test_df["target"].to_numpy(dtype=np.float64)
    p10_test_raw = q10_model.predict(X_test)
    p90_test_raw = q90_model.predict(X_test)

    # Point forecast on test: moving_avg_4 or best ML model
    test_point = predict_moving_avg_4(test_df)
    p10_test_sorted, pt_test_sorted, p90_test_sorted = sort_and_clip_intervals(
        p10_test_raw, test_point, p90_test_raw
    )

    test_cov_raw = empirical_coverage(y_test, p10_test_sorted, p90_test_sorted)
    test_pinball_10_raw = pinball_loss(y_test, p10_test_sorted, alpha=0.1)
    test_pinball_90_raw = pinball_loss(y_test, p90_test_sorted, alpha=0.9)

    p10_test_cal, pt_test_cal, p90_test_cal = sort_and_clip_intervals(
        p10_test_sorted - c_offset, pt_test_sorted, p90_test_sorted + c_offset
    )
    test_cov_cal = empirical_coverage(y_test, p10_test_cal, p90_test_cal)
    test_pinball_10_cal = pinball_loss(y_test, p10_test_cal, alpha=0.1)
    test_pinball_90_cal = pinball_loss(y_test, p90_test_cal, alpha=0.9)

    print(f"\n4. Test Set Evaluation (8 weeks, 800 rows):")
    print(f"  Test Coverage (Raw Uncalibrated): {test_cov_raw * 100:.2f}%")
    print(f"  Test Coverage (Calibrated):     {test_cov_cal * 100:.2f}% (Target: ~80%)")
    print(f"  Test Pinball Loss (Raw):        p10={test_pinball_10_raw:.2f}, p90={test_pinball_90_raw:.2f}")
    print(f"  Test Pinball Loss (Calibrated): p10={test_pinball_10_cal:.2f}, p90={test_pinball_90_cal:.2f}")

    # Verify ordering holds strictly for every row
    assert np.all(p10_test_cal <= pt_test_cal), "Ordering violation: p10 > point"
    assert np.all(pt_test_cal <= p90_test_cal), "Ordering violation: point > p90"
    assert np.all(p10_test_cal >= 0.0), "Negative values in p10"

    # 4. Save models and coverage figures
    coverage_dict = {
        "val": round(val_cov_cal, 4),
        "test": round(test_cov_cal, 4),
        "val_raw": round(val_cov_raw, 4),
        "val_calibrated": round(val_cov_cal, 4),
        "test_raw": round(test_cov_raw, 4),
        "test_calibrated": round(test_cov_cal, 4),
        "target_coverage": 0.80,
        "residual_correction_offset": round(c_offset, 2),
        "pinball_loss_val": {
            "p10_raw": round(val_pinball_10_raw, 4),
            "p90_raw": round(val_pinball_90_raw, 4),
            "p10_calibrated": round(val_pinball_10_cal, 4),
            "p90_calibrated": round(val_pinball_90_cal, 4),
        },
        "pinball_loss_test": {
            "p10_raw": round(test_pinball_10_raw, 4),
            "p90_raw": round(test_pinball_90_raw, 4),
            "p10_calibrated": round(test_pinball_10_cal, 4),
            "p90_calibrated": round(test_pinball_90_cal, 4),
        },
    }

    if save_artifacts:
        q10_path = models_dir / "q10.joblib"
        q90_path = models_dir / "q90.joblib"
        joblib.dump(q10_model, q10_path)
        joblib.dump(q90_model, q90_path)
        print(f"\nSaved models to {q10_path} and {q90_path}")

        cov_json_path = models_dir / "interval_coverage.json"
        with open(cov_json_path, "w", encoding="utf-8") as f:
            json.dump(coverage_dict, f, indent=2)
        print(f"Saved interval coverage figures to {cov_json_path}")

    print("=" * 85 + "\n")

    return {
        "q10_model": q10_model,
        "q90_model": q90_model,
        "coverage": coverage_dict,
        "test_intervals": {
            "p10": p10_test_cal,
            "point": pt_test_cal,
            "p90": p90_test_cal,
        },
    }


if __name__ == "__main__":
    train_and_evaluate_quantiles()
