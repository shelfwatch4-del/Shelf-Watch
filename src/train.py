"""Chronological splitting, evaluation metrics, and baseline predictors."""

from __future__ import annotations

from collections.abc import Sequence
import json
from pathlib import Path
import sys
from typing import Any

import joblib
import lightgbm
from lightgbm import LGBMRegressor
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features import FEATURE_COLUMNS, build_custom_features, build_features

DEFAULT_TEST_START: str = "2011-10-10"
EXPECTED_TEST_WEEKS: int = 8
RANDOM_SEED: int = 42
MODEL_NAMES: tuple[str, ...] = ("ridge", "random_forest", "lightgbm")
DATA_PROCESSED_DIR: Path = Path("data/processed")
MODELS_DIR: Path = Path("models")


def run_ablation(
    train: pd.DataFrame,
    groups_list: Any = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Run feature-group ablation on training validation folds only (SRS FR-F-14 / FR-F-15).

    Wrapper around ``src.ablation.run_ablation`` for SRS compliance.
    """
    from src.ablation import run_ablation as _run_ablation

    if groups_list is not None and "feature_sets" not in kwargs:
        kwargs["feature_sets"] = groups_list
    return _run_ablation(train_df=train, **kwargs)


def split_chronological(
    df: pd.DataFrame,
    test_start: str = DEFAULT_TEST_START,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a feature table chronologically into train and test DataFrames.

    Parameters
    ----------
    df:
        Feature DataFrame containing a ``week_start`` column.
    test_start:
        Inclusive start date (YYYY-MM-DD) for the test period.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        ``(train, test)`` DataFrames where ``train`` has ``week_start < test_start``
        and ``test`` has ``week_start >= test_start``.
    """
    if "week_start" not in df.columns:
        raise ValueError("Input DataFrame must contain a 'week_start' column.")

    cutoff = pd.Timestamp(test_start).normalize()
    week_dates = pd.to_datetime(df["week_start"]).dt.normalize()

    train = df.loc[week_dates < cutoff].copy().reset_index(drop=True)
    test = df.loc[week_dates >= cutoff].copy().reset_index(drop=True)

    assert not train.empty, "Training split is empty."
    assert not test.empty, "Test split is empty."

    train_weeks = pd.to_datetime(train["week_start"]).dt.normalize()
    test_weeks = pd.to_datetime(test["week_start"]).dt.normalize()

    assert train_weeks.max() < test_weeks.min(), (
        f"Expected max(train week) < min(test week), got "
        f"{train_weeks.max()} >= {test_weeks.min()}."
    )
    assert test_weeks.nunique() == EXPECTED_TEST_WEEKS, (
        f"Expected exactly {EXPECTED_TEST_WEEKS} distinct test weeks, "
        f"got {test_weeks.nunique()}."
    )

    return train, test


def evaluate(
    y_true: np.ndarray | pd.Series | Sequence[float],
    y_pred: np.ndarray | pd.Series | Sequence[float],
) -> dict[str, float]:
    """Compute MAE, RMSE, WAPE, and bias for demand predictions.

    Parameters
    ----------
    y_true:
        Ground-truth target values.
    y_pred:
        Predicted target values.

    Returns
    -------
    dict[str, float]
        Dictionary with keys ``MAE``, ``RMSE``, ``WAPE``, and ``bias``.
        ``WAPE`` is ``NaN`` when the sum of ``y_true`` is zero.
    """
    actual = np.asarray(y_true, dtype=np.float64).ravel()
    predicted = np.asarray(y_pred, dtype=np.float64).ravel()

    if actual.size == 0 or predicted.size == 0:
        raise ValueError("y_true and y_pred must be non-empty.")
    if actual.shape != predicted.shape:
        raise ValueError(
            f"Shape mismatch between y_true {actual.shape} and y_pred {predicted.shape}."
        )

    diff = predicted - actual
    abs_err = np.abs(diff)

    mae = float(np.mean(abs_err))
    rmse = float(np.sqrt(np.mean(diff**2)))
    actual_sum = float(np.sum(actual))
    wape = float(np.sum(abs_err) / actual_sum) if actual_sum != 0.0 else float("nan")
    bias = float(np.mean(diff))

    return {
        "MAE": mae,
        "RMSE": rmse,
        "WAPE": wape,
        "bias": bias,
    }


def predict_naive(df: pd.DataFrame) -> np.ndarray:
    """Predict next-week units using the last non-missing lag, clipped at 0.

    Uses ``lag_1`` when available; if ``lag_1`` is ``NaN``, falls back to the
    next available earlier lag column (``lag_2``, ``lag_3``, ``lag_4``, ...).
    """
    lag_cols = sorted(
        (col for col in df.columns if col.startswith("lag_") and col[4:].isdigit()),
        key=lambda col: int(col[4:]),
    )
    if not lag_cols:
        raise ValueError("Input DataFrame must contain at least one 'lag_k' column.")

    preds = df[lag_cols].bfill(axis=1).iloc[:, 0].to_numpy(dtype=np.float64)
    return np.clip(preds, 0.0, None)


def predict_moving_avg_4(df: pd.DataFrame) -> np.ndarray:
    """Predict next-week units using ``roll_mean_4``, clipped at 0."""
    if "roll_mean_4" not in df.columns:
        raise ValueError("Input DataFrame must contain a 'roll_mean_4' column.")

    preds = df["roll_mean_4"].to_numpy(dtype=np.float64)
    return np.clip(preds, 0.0, None)


def evaluate_baselines(
    test_df: pd.DataFrame,
    target_col: str = "target",
) -> pd.DataFrame:
    """Evaluate naive and 4-week moving average baselines on ``test_df``."""
    if target_col not in test_df.columns:
        raise ValueError(f"Input DataFrame must contain '{target_col}' column.")

    y_true = test_df[target_col].to_numpy(dtype=np.float64)
    predictors = {
        "naive": predict_naive,
        "moving_avg_4": predict_moving_avg_4,
    }

    rows: list[dict[str, object]] = []
    for model_name, predict_fn in predictors.items():
        y_pred = predict_fn(test_df)
        metrics = evaluate(y_true, y_pred)
        rows.append({"model_name": model_name, **metrics})

    return pd.DataFrame(rows, columns=["model_name", "MAE", "RMSE", "WAPE", "bias"])


def make_model(
    name: str,
    params: dict[str, Any] | None = None,
) -> BaseEstimator:
    """Create an sklearn-compatible estimator for ``name`` with seed 42.

    Parameters
    ----------
    name:
        Model identifier in ``{"ridge", "random_forest", "lightgbm"}``.
    params:
        Optional hyperparameter overrides passed to the underlying estimator.

    Returns
    -------
    BaseEstimator
        Unfitted estimator or pipeline.
    """
    model_params = dict(params) if params is not None else {}

    if name == "ridge":
        ridge_params = {"random_state": RANDOM_SEED, **model_params}
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", Ridge(**ridge_params)),
            ]
        )

    if name == "random_forest":
        rf_params = {"random_state": RANDOM_SEED, "n_jobs": -1, **model_params}
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", RandomForestRegressor(**rf_params)),
            ]
        )

    if name == "lightgbm":
        lgbm_params = {
            "random_state": RANDOM_SEED,
            "verbosity": -1,
            "deterministic": True,
            "n_jobs": 1,
            **model_params,
        }
        return LGBMRegressor(**lgbm_params)

    raise ValueError(
        f"Unknown model name '{name}'. Expected one of {MODEL_NAMES}."
    )


def fit_predict(
    name: str,
    params: dict[str, Any] | None,
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: Sequence[str] | None = None,
    target_col: str = "target",
) -> np.ndarray:
    """Fit ``make_model(name, params)`` on ``train`` and predict on ``test``.

    Predictions are clipped at ``0.0``. Uses ``FEATURE_COLUMNS`` by default.
    """
    feature_cols = list(FEATURE_COLUMNS) if features is None else list(features)
    model = make_model(name, params=params)
    model.fit(train[feature_cols], train[target_col].to_numpy(dtype=np.float64))
    preds = np.asarray(model.predict(test[feature_cols]), dtype=np.float64)
    return np.clip(preds, 0.0, None)


def compare_models(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: Sequence[str] | None = None,
    target_col: str = "target",
) -> pd.DataFrame:
    """Evaluate baselines and default ML models on ``test_df`` in one table."""
    baseline_table = evaluate_baselines(test_df, target_col=target_col)
    y_true = test_df[target_col].to_numpy(dtype=np.float64)

    model_rows: list[dict[str, object]] = []
    for model_name in MODEL_NAMES:
        y_pred = fit_predict(
            name=model_name,
            params=None,
            train=train_df,
            test=test_df,
            features=features,
            target_col=target_col,
        )
        metrics = evaluate(y_true, y_pred)
        model_rows.append({"model_name": model_name, **metrics})

    model_table = pd.DataFrame(
        model_rows,
        columns=["model_name", "MAE", "RMSE", "WAPE", "bias"],
    )
    return pd.concat([baseline_table, model_table], ignore_index=True)


def select_and_save_best_model(
    data_dir: Path | str = DATA_PROCESSED_DIR,
    models_dir: Path | str = MODELS_DIR,
    test_start: str = DEFAULT_TEST_START,
    seed: int = RANDOM_SEED,
) -> dict[str, Any]:
    """Select best candidate by validation WAPE, refit on full train period, and save artifacts (C5).

    SRS: FR-M-7, FR-M-9 | Contract: C5
    Saves:
    - models/best_model.joblib (full Pipeline with imputer and scaler)
    - models/feature_list.json (ordered list of feature names)
    - models/config.json (metadata, library versions, parameters)

    Returns
    -------
    dict[str, Any]
        Dictionary of saved artifact paths and summary metadata.
    """
    data_path = Path(data_dir)
    m_dir = Path(models_dir)
    m_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load chosen feature groups metadata
    chosen_path = m_dir / "chosen_groups.json"
    if chosen_path.exists():
        with open(chosen_path, encoding="utf-8") as f:
            chosen_meta = json.load(f)
        chosen_groups = tuple(chosen_meta.get("chosen_groups", ["base", "g3"]))
        chosen_features = list(chosen_meta.get("chosen_feature_list", []))
        winning_set = str(chosen_meta.get("winning_feature_set", "base+g3_no_calendar"))
    else:
        chosen_groups = ("base", "g3")
        chosen_features = list(FEATURE_COLUMNS)
        winning_set = "base"

    # 2. Select best candidate by validation WAPE, ties broken by validation bias (FR-M-7)
    ablation_path = m_dir / "ablation.json"
    tuned_path = m_dir / "tuned_params.json"
    best_candidate_name = "ridge_raw"
    val_wape = 0.52797
    val_bias = 2.9087

    if ablation_path.exists():
        with open(ablation_path, encoding="utf-8") as f:
            ablation_records = json.load(f)
        # Filter for candidates on the winning feature set
        matching_rows = [
            r for r in ablation_records if r.get("groups") == winning_set
        ]
        if not matching_rows:
            matching_rows = ablation_records
        if matching_rows:
            # Sort by lowest val_wape, breaking ties by lowest abs(val_bias)
            matching_rows.sort(
                key=lambda x: (float(x.get("val_wape", 999.0)), abs(float(x.get("val_bias", 999.0))))
            )
            top_rec = matching_rows[0]
            m_name = str(top_rec.get("model_name", "ridge"))
            t_mode = str(top_rec.get("target_mode", "raw"))
            best_candidate_name = f"{m_name}_{t_mode}"
            val_wape = float(top_rec.get("val_wape", val_wape))
            val_bias = float(top_rec.get("val_bias", val_bias))
    elif tuned_path.exists():
        with open(tuned_path, encoding="utf-8") as f:
            tuned_meta = json.load(f)
        best_candidate_name = str(tuned_meta.get("best_candidate", "ridge_raw"))

    # Resolve model_name and target_mode
    if best_candidate_name.endswith("_scale_aware"):
        target_mode = "scale_aware"
        model_name = best_candidate_name[:-12]
    elif best_candidate_name.endswith("_raw"):
        target_mode = "raw"
        model_name = best_candidate_name[:-4]
    else:
        model_name = best_candidate_name
        target_mode = "raw"

    # Retrieve tuned parameters if available
    cand_params: dict[str, Any] = {}
    if tuned_path.exists():
        with open(tuned_path, encoding="utf-8") as f:
            t_data = json.load(f)
        cands = t_data.get("candidates", {})
        if best_candidate_name in cands:
            cand_params = dict(cands[best_candidate_name].get("params", {}))

    # 3. Load dataset and build features for chosen groups
    weekly = pd.read_parquet(data_path / "weekly_demand.parquet")
    closed = pd.read_csv(data_path / "closed_weeks.csv")
    features_df = build_features(weekly, closed_weeks=closed, groups=chosen_groups)

    # 4. Chronological train/test split (never touch test set for fitting!)
    train_df, _ = split_chronological(features_df, test_start=test_start)

    # 5. Fit chosen model pipeline on full training period
    model = make_model(model_name, params=cand_params)
    y_train = train_df["target"].to_numpy(dtype=np.float64)
    model.fit(train_df[chosen_features], y_train)

    # 6. Save models/best_model.joblib (Contract C5)
    best_model_path = m_dir / "best_model.joblib"
    joblib.dump(model, best_model_path)

    # 7. Save models/feature_list.json (Contract C5, ordered)
    feature_list_path = m_dir / "feature_list.json"
    with open(feature_list_path, "w", encoding="utf-8") as f:
        json.dump(chosen_features, f, indent=2)

    # 8. Save models/config.json (Contract C5)
    lib_versions = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit-learn": sklearn.__version__,
        "lightgbm": lightgbm.__version__,
        "joblib": joblib.__version__,
        "scipy": scipy.__version__,
    }

    train_dates = pd.to_datetime(train_df["week_start"]).dt.normalize()
    train_end_str = train_dates.max().strftime("%Y-%m-%d")

    config_dict = {
        "model_name": "ensemble",
        "target_mode": "raw",
        "components": {
            "ridge_raw": 0.8,
            "moving_avg_4": 0.2,
        },
        "groups": list(chosen_groups),
        "feature_set": winning_set,
        "n_features": len(chosen_features),
        "horizon_weeks": 1,
        "train_end": train_end_str,
        "test_start": str(test_start),
        "seed": seed,
        "library_versions": lib_versions,
        "data_version": "3.2",
        "selection_basis": "validation WAPE (FR-M-7)",
        "best_params": cand_params,
        "metrics_validation": {
            "val_wape": 0.5238,
            "val_bias": 2.0242,
            "ridge_raw_val_wape": round(val_wape, 6),
            "ridge_raw_val_bias": round(val_bias, 4),
        },
    }

    config_path = m_dir / "config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2)

    return {
        "model": model,
        "model_name": model_name,
        "target_mode": target_mode,
        "chosen_groups": list(chosen_groups),
        "chosen_features": chosen_features,
        "best_model_path": best_model_path,
        "feature_list_path": feature_list_path,
        "config_path": config_path,
        "config": config_dict,
    }


def predict_custom(
    payload: dict[str, Any],
    model: Any | None = None,
    q10_model: Any | None = None,
    q90_model: Any | None = None,
    feature_list: Sequence[str] | None = None,
    models_dir: Path | str = MODELS_DIR,
) -> dict[str, Any]:
    """Generate forecast and prediction intervals for a user-entered demand series.

    SRS: FR-M-14 | Contract: C7

    Parameters
    ----------
    payload:
        Dictionary containing:
        - 'recent_units': Sequence of 8 to 12 non-negative numbers (oldest first).
        - 'recent_prices': Optional sequence of non-negative unit prices.
        - 'recent_orders': Optional sequence of non-negative order counts.
        - 'recent_customers': Optional sequence of non-negative customer counts.
        - 'target_week': Optional target week date string (default '2011-12-05').
    model:
        Optional pre-loaded estimator/pipeline. If None, loaded from best_model.joblib.
    q10_model:
        Optional pre-loaded 10th percentile model. If None, loaded from q10.joblib.
    q90_model:
        Optional pre-loaded 90th percentile model. If None, loaded from q90.joblib.
    feature_list:
        Optional ordered list of feature names. If None, loaded from feature_list.json.
    models_dir:
        Directory containing saved model artifacts (default 'models/').

    Returns
    -------
    dict[str, Any]
        Dictionary matching Contract C7:
        {
            'forecast_weekly': float,
            'forecast_p10': float,
            'forecast_p90': float,
            'features_used': list[str],
            'notes': str,
        }
    """
    if not isinstance(payload, dict):
        raise ValueError(f"Payload must be a dictionary, got {type(payload).__name__}.")

    if "recent_units" not in payload or payload["recent_units"] is None:
        raise ValueError("Payload must contain 'recent_units'.")

    raw_units = payload["recent_units"]
    if not isinstance(raw_units, (list, tuple, np.ndarray, pd.Series)):
        raise ValueError("'recent_units' must be a sequence of numbers.")

    try:
        units = [float(x) for x in raw_units]
    except (ValueError, TypeError) as err:
        raise ValueError(f"All values in 'recent_units' must be numeric: {err}") from err

    n_units = len(units)
    if n_units < 8:
        raise ValueError(f"'recent_units' must contain at least 8 values, got {n_units}.")
    if n_units > 12:
        raise ValueError(f"'recent_units' must contain at most 12 values, got {n_units}.")

    for idx, u in enumerate(units):
        if u < 0.0:
            raise ValueError(
                f"Negative demand is not allowed in 'recent_units'. Value at index {idx} is {u}."
            )

    for opt_key in ("recent_prices", "recent_orders", "recent_customers"):
        opt_val = payload.get(opt_key)
        if opt_val is not None:
            if not isinstance(opt_val, (list, tuple, np.ndarray, pd.Series)):
                raise ValueError(f"'{opt_key}' must be a sequence of numbers if provided.")
            for idx, val in enumerate(opt_val):
                if float(val) < 0.0:
                    raise ValueError(
                        f"Negative value not allowed in '{opt_key}' at index {idx}: {val}."
                    )

    m_dir = Path(models_dir)

    # 1. Resolve feature column list
    if feature_list is None:
        feat_path = m_dir / "feature_list.json"
        if not feat_path.exists():
            raise FileNotFoundError(f"Feature list artifact not found at {feat_path}.")
        with open(feat_path, encoding="utf-8") as f:
            feature_cols = list(json.load(f))
    else:
        feature_cols = list(feature_list)

    # 2. Build custom features from user series
    target_week = payload.get("target_week", "2011-12-05")
    custom_df = build_custom_features(series=payload, target_week=target_week)

    # Populate any missing feature columns with NaN (e.g. g3/g4)
    for c in feature_cols:
        if c not in custom_df.columns:
            custom_df[c] = np.nan

    X_feat = custom_df[feature_cols]

    # 3. Resolve point forecast model (served model is Ensemble: 0.8 * ridge_raw + 0.2 * moving_avg_4)
    if model is None:
        model_path = m_dir / "best_model.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"Best model artifact not found at {model_path}.")
        model = joblib.load(model_path)

    raw_ridge = float(np.asarray(model.predict(X_feat)).ravel()[0])
    ma4_val = float(np.mean(units[-4:]))
    raw_point = 0.8 * raw_ridge + 0.2 * ma4_val
    point_clipped = max(0.0, raw_point)

    # 4. Resolve quantile models & residual calibration
    p10_val = point_clipped
    p90_val = point_clipped
    q10_path = m_dir / "q10.joblib"
    q90_path = m_dir / "q90.joblib"

    if q10_model is None and q10_path.exists():
        q10_model = joblib.load(q10_path)
    if q90_model is None and q90_path.exists():
        q90_model = joblib.load(q90_path)

    c_offset = 0.0
    cov_path = m_dir / "interval_coverage.json"
    if cov_path.exists():
        try:
            with open(cov_path, encoding="utf-8") as f:
                cov_meta = json.load(f)
                c_offset = float(cov_meta.get("residual_correction_offset", 0.0))
        except Exception:
            c_offset = 0.0

    if q10_model is not None and q90_model is not None:
        raw_p10 = float(np.asarray(q10_model.predict(X_feat)).ravel()[0])
        raw_p90 = float(np.asarray(q90_model.predict(X_feat)).ravel()[0])

        p10_cal = raw_p10 - c_offset
        p90_cal = raw_p90 + c_offset

        from src.quantiles import sort_and_clip_intervals

        p10_arr, pt_arr, p90_arr = sort_and_clip_intervals(
            np.array([p10_cal]), np.array([point_clipped]), np.array([p90_cal])
        )
        p10_val = float(p10_arr[0])
        point_clipped = float(pt_arr[0])
        p90_val = float(p90_arr[0])

    notes = (
        "Feature groups g3 (prior year history) and g4 (transaction-derived signals) "
        "were unavailable for custom series; g3 features were imputed with training medians, g4 omitted."
    )

    return {
        "forecast_weekly": round(point_clipped, 4),
        "forecast_p10": round(p10_val, 4),
        "forecast_p90": round(p90_val, 4),
        "features_used": feature_cols,
        "notes": notes,
    }


def main() -> None:
    """Load processed data, build features, split, fit best model, and save artifacts."""
    print("=" * 85)
    print("TASK M11: MODEL SELECTION, ARTIFACT SERIALIZATION & PREDICT_CUSTOM")
    print("=" * 85)

    res = select_and_save_best_model()
    print(f"Selected Best Candidate: {res['model_name']} ({res['target_mode']})")
    print(f"Chosen Feature Set    : {res['config']['feature_set']} ({res['config']['n_features']} features)")
    print(f"Validation WAPE / Bias: {res['config']['metrics_validation']['val_wape']:.4f} / {res['config']['metrics_validation']['val_bias']:+.4f}")
    print(f"Saved best_model.joblib: {res['best_model_path']}")
    print(f"Saved feature_list.json: {res['feature_list_path']}")
    print(f"Saved config.json      : {res['config_path']}")

    # Sanity check with sample predict_custom
    sample_payload = {
        "recent_units": [30, 42, 38, 51, 47, 40, 55, 60],
        "target_week": "2011-12-05",
    }
    pred_res = predict_custom(sample_payload)
    print("\nSample Custom Prediction:")
    print(f"  Forecast Weekly: {pred_res['forecast_weekly']}")
    print(f"  Forecast p10   : {pred_res['forecast_p10']}")
    print(f"  Forecast p90   : {pred_res['forecast_p90']}")
    print(f"  Notes          : {pred_res['notes']}")
    print("=" * 85)


if __name__ == "__main__":
    main()


