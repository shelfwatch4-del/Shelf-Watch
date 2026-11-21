"""Feature attribution, explainability helper, and global feature importance."""

from __future__ import annotations

from collections.abc import Sequence
import json
from pathlib import Path
from typing import Any

from lightgbm import LGBMRegressor
import numpy as np
import pandas as pd
import shap
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from src.features import (
    FEATURE_COLUMNS,
    FEATURE_GROUPS,
    build_features,
)
from src.train import (
    DEFAULT_TEST_START,
    RANDOM_SEED,
    make_model,
    split_chronological,
)

DATA_PROCESSED_DIR: Path = Path("data/processed")
MODELS_DIR: Path = Path("models")


def explain_row(
    model: Any,
    feature_row: pd.Series | pd.DataFrame | np.ndarray,
    feature_names: Sequence[str] | None = None,
    scale_factor: float = 1.0,
    is_ensemble: bool = False,
    ensemble_weights: tuple[float, float] = (0.8, 0.2),
) -> dict[str, Any]:
    """Explain a single observation with exact feature contributions and base value.

    Parameters
    ----------
    model:
        Trained estimator (LightGBM, Ridge Pipeline, or linear model).
    feature_row:
        Single row of feature values (Series, 1-row DataFrame, or 1D array).
    feature_names:
        Names of the features in order. Inferred if feature_row is Series or DataFrame.
    scale_factor:
        Scaling factor for scale-aware targets (defaults to 1.0 for raw targets).
    is_ensemble:
        Whether to produce an ensemble explanation (0.8 * Ridge + 0.2 * roll_mean_4).
    ensemble_weights:
        Tuple (w_ridge, w_ma4) for ensemble blending, default (0.8, 0.2).

    Returns
    -------
    dict[str, Any]
        Dictionary containing:
        - 'feature_values': dict[str, float]
        - 'contributions': dict[str, float]
        - 'base_value': float
        - 'model_output': float
        - 'scaling_factor': float
        - 'unscaled_output': float
    """
    if is_ensemble:
        # Compute underlying Ridge explanation first
        base_exp = explain_row(
            model=model,
            feature_row=feature_row,
            feature_names=feature_names,
            scale_factor=scale_factor,
            is_ensemble=False,
        )
        w_ridge, w_ma4 = ensemble_weights
        base_val = w_ridge * base_exp["base_value"]
        feat_vals = base_exp["feature_values"]
        ma4_val = float(feat_vals.get("roll_mean_4", 0.0))

        ens_contribs: dict[str, float] = {}
        for feat_k, c_val in base_exp["contributions"].items():
            if feat_k == "roll_mean_4":
                ens_contribs[feat_k] = w_ridge * c_val + w_ma4 * ma4_val
            else:
                ens_contribs[feat_k] = w_ridge * c_val

        ens_raw_output = w_ridge * base_exp["model_output"] + w_ma4 * ma4_val
        ens_contrib_sum = base_val + sum(ens_contribs.values())
        diff = abs(ens_contrib_sum - ens_raw_output)
        if diff > 1e-6:
            raise AssertionError(
                f"Ensemble decomposition identity failed: base_val + sum(contrib) = {ens_contrib_sum}, "
                f"ens_raw_output = {ens_raw_output}, diff = {diff} > 1e-6"
            )

        scale = float(scale_factor) if scale_factor is not None else 1.0
        ens_unscaled = float(np.clip(ens_raw_output * scale, 0.0, None))
        return {
            "feature_values": feat_vals,
            "contributions": ens_contribs,
            "base_value": base_val,
            "model_output": ens_raw_output,
            "scaling_factor": scale,
            "unscaled_output": ens_unscaled,
        }
    # 1. Normalize feature_row and extract names
    if isinstance(feature_row, pd.DataFrame):
        if len(feature_row) != 1:
            raise ValueError(f"feature_row DataFrame must have exactly 1 row, got {len(feature_row)}.")
        cols = list(feature_row.columns) if feature_names is None else list(feature_names)
        vals_1d = feature_row[cols].iloc[0].to_numpy(dtype=np.float64)
        df_1row = feature_row[cols].copy()
    elif isinstance(feature_row, pd.Series):
        cols = list(feature_row.index) if feature_names is None else list(feature_names)
        vals_1d = feature_row[cols].to_numpy(dtype=np.float64)
        df_1row = pd.DataFrame([feature_row[cols]], columns=cols)
    else:
        vals_1d = np.asarray(feature_row, dtype=np.float64).ravel()
        if feature_names is None:
            cols = [f"f{i}" for i in range(len(vals_1d))]
        else:
            cols = list(feature_names)
            if len(cols) != len(vals_1d):
                raise ValueError(f"Length mismatch: {len(cols)} feature names vs {len(vals_1d)} values.")
        df_1row = pd.DataFrame([vals_1d], columns=cols)

    feature_values: dict[str, float] = {col: float(vals_1d[j]) for j, col in enumerate(cols)}

    # 2. Extract contributions and base value by model family
    if isinstance(model, LGBMRegressor) or hasattr(model, "booster_"):
        # LightGBM TreeSHAP via pred_contrib=True
        # Returns [contrib_0, contrib_1, ..., contrib_{P-1}, base_value]
        pred_contrib = model.predict(df_1row, pred_contrib=True)
        pred_contrib_1d = np.asarray(pred_contrib, dtype=np.float64).ravel()
        base_value = float(pred_contrib_1d[-1])
        contributions = {col: float(pred_contrib_1d[j]) for j, col in enumerate(cols)}
        raw_output = float(model.predict(df_1row)[0])

    elif isinstance(model, Pipeline):
        imputer = model.named_steps.get("imputer")
        scaler = model.named_steps.get("scaler")
        estimator = model.named_steps.get("model")

        x_proc = vals_1d.copy().reshape(1, -1)
        if imputer is not None:
            x_proc = imputer.transform(x_proc)
        if scaler is not None:
            x_proc = scaler.transform(x_proc)

        x_proc_1d = x_proc.ravel()

        if isinstance(estimator, RandomForestRegressor) or hasattr(estimator, "estimators_"):
            explainer = shap.TreeExplainer(estimator)
            shap_vals = explainer.shap_values(x_proc)
            exp_val = explainer.expected_value
            base_value = float(exp_val[0] if isinstance(exp_val, (list, np.ndarray)) else exp_val)
            contributions = {col: float(shap_vals[0, j]) for j, col in enumerate(cols)}
            raw_output = float(model.predict(df_1row)[0])

        elif hasattr(estimator, "coef_") and hasattr(estimator, "intercept_"):
            coef = np.asarray(estimator.coef_, dtype=np.float64).ravel()
            intercept = float(estimator.intercept_)
            base_value = intercept
            contributions = {col: float(coef[j] * x_proc_1d[j]) for j, col in enumerate(cols)}
            raw_output = float(model.predict(df_1row)[0])
        else:
            raise TypeError(f"Pipeline estimator {type(estimator)} does not expose linear coefficients or tree estimators.")

    elif isinstance(model, RandomForestRegressor) or hasattr(model, "estimators_"):
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(vals_1d.reshape(1, -1))
        exp_val = explainer.expected_value
        base_value = float(exp_val[0] if isinstance(exp_val, (list, np.ndarray)) else exp_val)
        contributions = {col: float(shap_vals[0, j]) for j, col in enumerate(cols)}
        raw_output = float(model.predict(df_1row)[0])

    elif hasattr(model, "coef_") and hasattr(model, "intercept_"):
        coef = np.asarray(model.coef_, dtype=np.float64).ravel()
        base_value = float(model.intercept_)
        contributions = {col: float(coef[j] * vals_1d[j]) for j, col in enumerate(cols)}
        raw_output = float(model.predict(df_1row)[0])

    else:
        raise TypeError(f"Unsupported model type '{type(model)}' for exact explain_row.")

    # 3. Verify exact sum identity within 1e-6
    contrib_sum = base_value + sum(contributions.values())
    diff = abs(contrib_sum - raw_output)
    if diff > 1e-6:
        raise AssertionError(
            f"Decomposition identity failed: base_value + sum(contributions) = {contrib_sum}, "
            f"model_output = {raw_output}, diff = {diff} > 1e-6"
        )

    # 4. Handle scale-aware targets
    scale = float(scale_factor) if scale_factor is not None else 1.0
    unscaled_output = float(np.clip(raw_output * scale, 0.0, None))

    return {
        "feature_values": feature_values,
        "contributions": contributions,
        "base_value": base_value,
        "model_output": raw_output,
        "scaling_factor": scale,
        "unscaled_output": unscaled_output,
    }


def _get_feature_group(feature_name: str, groups: dict[str, list[str]] | None = None) -> str:
    """Map a feature name to its registered feature group."""
    group_map = FEATURE_GROUPS if groups is None else groups
    for g_name, f_list in group_map.items():
        if feature_name in f_list:
            return g_name
    return "base"


def compute_feature_importance(
    model: Any,
    test_df: pd.DataFrame,
    features: Sequence[str],
    feature_groups: dict[str, list[str]] | None = None,
    model_name: str = "lightgbm",
) -> pd.DataFrame:
    """Compute gain importance and mean absolute contribution on test rows.

    Parameters
    ----------
    model:
        Trained model (LightGBM or Ridge Pipeline).
    test_df:
        Held-out test set DataFrame.
    features:
        List of feature column names.
    feature_groups:
        Optional feature group dictionary mapping group name -> list of features.
    model_name:
        Model identifier label (e.g. 'lightgbm' or 'ridge').

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        ['model_name', 'feature', 'feature_group', 'group', 'gain_importance', 'mean_abs_contribution']
    """
    feature_cols = list(features)
    X_test = test_df[feature_cols]

    # 1. Gain importance or absolute coefficient
    if isinstance(model, LGBMRegressor) or hasattr(model, "booster_"):
        # For LightGBM: gain importance
        booster = getattr(model, "booster_", model)
        gain_imp = booster.feature_importance(importance_type="gain")
        gain_values = [float(v) for v in gain_imp]

        # Mean absolute contribution on test rows using batch pred_contrib
        contrib_matrix = booster.predict(X_test, pred_contrib=True)[:, :-1]
        mean_abs_contrib = np.mean(np.abs(contrib_matrix), axis=0)

    elif isinstance(model, Pipeline):
        imputer = model.named_steps.get("imputer")
        scaler = model.named_steps.get("scaler")
        estimator = model.named_steps.get("model")
        X_proc = X_test.to_numpy(dtype=np.float64)
        if imputer is not None:
            X_proc = imputer.transform(X_proc)

        if isinstance(estimator, RandomForestRegressor) or hasattr(estimator, "estimators_"):
            gain_values = [float(v) for v in estimator.feature_importances_]
            explainer = shap.TreeExplainer(estimator)
            shap_matrix = explainer.shap_values(X_proc)
            mean_abs_contrib = np.mean(np.abs(shap_matrix), axis=0)
        elif hasattr(estimator, "coef_"):
            if scaler is not None:
                X_proc = scaler.transform(X_proc)
            coef = np.asarray(estimator.coef_, dtype=np.float64).ravel()
            contrib_matrix = X_proc * coef.reshape(1, -1)
            if model_name == "ensemble":
                contrib_matrix = 0.8 * contrib_matrix
                if "roll_mean_4" in feature_cols:
                    ma4_idx = feature_cols.index("roll_mean_4")
                    contrib_matrix[:, ma4_idx] += 0.2 * test_df["roll_mean_4"].to_numpy(dtype=np.float64)
                mean_abs_contrib = np.mean(np.abs(contrib_matrix), axis=0)
                sum_mac = float(np.sum(mean_abs_contrib))
                gain_values = [float(v / sum_mac) if sum_mac > 0 else 0.0 for v in mean_abs_contrib]
            else:
                gain_values = [float(v) for v in np.abs(coef)]
                mean_abs_contrib = np.mean(np.abs(contrib_matrix), axis=0)
        else:
            raise TypeError(f"Pipeline estimator {type(estimator)} not supported for feature importance.")

    elif isinstance(model, RandomForestRegressor) or hasattr(model, "estimators_"):
        gain_values = [float(v) for v in model.feature_importances_]
        explainer = shap.TreeExplainer(model)
        shap_matrix = explainer.shap_values(X_test.to_numpy(dtype=np.float64))
        mean_abs_contrib = np.mean(np.abs(shap_matrix), axis=0)

    elif hasattr(model, "coef_"):
        X_proc = X_test.to_numpy(dtype=np.float64)
        coef = np.asarray(model.coef_, dtype=np.float64).ravel()
        gain_values = [float(v) for v in np.abs(coef)]
        contrib_matrix = X_proc * coef.reshape(1, -1)
        mean_abs_contrib = np.mean(np.abs(contrib_matrix), axis=0)

    else:
        raise TypeError(f"Unsupported model type '{type(model)}' for feature importance.")

    rows: list[dict[str, Any]] = []
    for j, feat in enumerate(feature_cols):
        grp = _get_feature_group(feat, feature_groups)
        rows.append(
            {
                "model_name": model_name,
                "feature": feat,
                "feature_group": grp,
                "group": grp,
                "gain_importance": round(float(gain_values[j]), 4),
                "mean_abs_contribution": round(float(mean_abs_contrib[j]), 4),
            }
        )

    # Sort descending by gain_importance
    df_fi = pd.DataFrame(rows).sort_values(by="gain_importance", ascending=False).reset_index(drop=True)
    return df_fi


def generate_feature_importance_csv(
    features_df: pd.DataFrame | None = None,
    output_path: Path | None = None,
    model_name: str | None = None,
    features: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Fit final model on training period, evaluate on test, and save feature_importance.csv."""
    if features_df is None:
        weekly = pd.read_parquet(DATA_PROCESSED_DIR / "weekly_demand.parquet")
        closed_weeks = pd.read_csv(DATA_PROCESSED_DIR / "closed_weeks.csv")
        features_df = build_features(weekly, closed_weeks=closed_weeks)

    train_df, test_df = split_chronological(features_df, test_start=DEFAULT_TEST_START)
    feature_cols = list(FEATURE_COLUMNS) if features is None else list(features)

    # Determine final model and load tuned parameters
    tuned_params_file = MODELS_DIR / "tuned_params.json"
    config_file = MODELS_DIR / "config.json"
    resolved_model_name = model_name
    model_params = None

    if resolved_model_name is None and config_file.exists():
        with open(config_file, "r", encoding="utf-8") as f:
            c_data = json.load(f)
            resolved_model_name = c_data.get("model_name", "ensemble")

    if tuned_params_file.exists():
        with open(tuned_params_file, "r", encoding="utf-8") as f:
            t_data = json.load(f)
            if resolved_model_name is None:
                best_cand = str(t_data.get("best_candidate", "ensemble"))
                resolved_model_name = "ensemble" if "ensemble" in best_cand else "ridge"

            if resolved_model_name == "ensemble" or resolved_model_name.startswith("ridge"):
                cand_key = "ridge_raw"
            else:
                cand_key = f"{resolved_model_name}_raw"

            if cand_key in t_data.get("candidates", {}):
                model_params = t_data["candidates"][cand_key]["params"]

    if resolved_model_name is None:
        resolved_model_name = "ensemble"

    # Train underlying linear model (Ridge) on full training period
    fit_name = "ridge" if resolved_model_name == "ensemble" else resolved_model_name
    model = make_model(fit_name, params=model_params)
    y_train = train_df["target"].to_numpy(dtype=np.float64)
    model.fit(train_df[feature_cols], y_train)

    # Compute importance on test set
    df_fi = compute_feature_importance(
        model=model,
        test_df=test_df,
        features=feature_cols,
        model_name=resolved_model_name,
    )

    out_file = DATA_PROCESSED_DIR / "feature_importance.csv" if output_path is None else output_path
    out_file.parent.mkdir(parents=True, exist_ok=True)
    # Save standard contract columns (including feature_group and group)
    df_fi.to_csv(out_file, index=False)
    # Also save to models dir for reference
    df_fi.to_csv(MODELS_DIR / "feature_importance.csv", index=False)

    return df_fi


def explain_prediction(
    product_id: str,
    target_week: str,
    model: Any | None = None,
    features_df: pd.DataFrame | None = None,
    feature_cols: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Provide detailed explanation payload for a specific product and target week."""
    cols = list(FEATURE_COLUMNS) if feature_cols is None else list(feature_cols)

    if features_df is None:
        weekly = pd.read_parquet(DATA_PROCESSED_DIR / "weekly_demand.parquet")
        closed_weeks = pd.read_csv(DATA_PROCESSED_DIR / "closed_weeks.csv")
        features_df = build_features(weekly, closed_weeks=closed_weeks)

    t_week = pd.Timestamp(target_week).normalize()
    sub = features_df.loc[
        (features_df["product_id"] == str(product_id)) & (pd.to_datetime(features_df["week_start"]).dt.normalize() == t_week)
    ]
    if sub.empty:
        raise ValueError(f"No observation found for product '{product_id}' at week '{target_week}'.")

    model_name_used = "ensemble"
    is_ens = True
    if model is None:
        best_model_file = MODELS_DIR / "best_model.joblib"
        if best_model_file.exists():
            model = joblib.load(best_model_file)
        else:
            # Fit default Ridge
            train_df, _ = split_chronological(features_df, test_start=DEFAULT_TEST_START)
            model = make_model("ridge", params={"alpha": 157.4189})
            model.fit(train_df[cols], train_df["target"].to_numpy(dtype=np.float64))
        model_name_used = "ensemble"
        is_ens = True
    elif isinstance(model, LGBMRegressor) or hasattr(model, "booster_"):
        model_name_used = "lightgbm"
        is_ens = False
    elif isinstance(model, Pipeline) or hasattr(model, "coef_"):
        estimator = model.named_steps.get("model") if isinstance(model, Pipeline) else model
        if isinstance(estimator, RandomForestRegressor):
            model_name_used = "random_forest"
            is_ens = False
        else:
            model_name_used = "ensemble"
            is_ens = True

    row_data = sub.iloc[0]
    exp = explain_row(model, row_data[cols], feature_names=cols, is_ensemble=is_ens)

    features_list = [
        {
            "name": feat,
            "group": _get_feature_group(feat),
            "value": exp["feature_values"][feat],
            "contribution": exp["contributions"][feat],
        }
        for feat in cols
    ]

    return {
        "product_id": str(product_id),
        "target_week": t_week.strftime("%Y-%m-%d"),
        "model_name": model_name_used,
        "base_value": exp["base_value"],
        "model_output": exp["model_output"],
        "features": features_list,
    }


def main() -> None:
    """Generate and display feature_importance.csv on disk."""
    print("=" * 80)
    print("TASK M15: GENERATING FEATURE IMPORTANCE (Final Model on Test Rows)")
    print("=" * 80)

    df_fi = generate_feature_importance_csv(model_name=None)
    csv_path = DATA_PROCESSED_DIR / "feature_importance.csv"

    print(f"\nSaved feature importance to {csv_path} ({len(df_fi)} features).")
    print(f"Final Model: {df_fi['model_name'].iloc[0]}")
    print("\nTop 10 features by importance and mean absolute contribution on test rows:")
    print(df_fi.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
