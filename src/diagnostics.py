"""Diagnostics, asymmetric error evaluation, and validation-chosen ensembling (Task M9).

SRS: FR-E-1..6, FR-M-12
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.features import build_features
from src.train import (
    DEFAULT_TEST_START,
    evaluate,
    make_model,
    predict_moving_avg_4,
    predict_naive,
    split_chronological,
)

DATA_PROCESSED_DIR: Path = Path("data/processed")
MODELS_DIR: Path = Path("models")


def compute_asymmetric_units(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
) -> dict[str, float]:
    """Compute asymmetric forecast metrics: over-forecast and under-forecast units.

    Over-forecast: model predicted more than actual sales (ties up cash in inventory).
    Under-forecast: model predicted less than actual sales (leads to stockouts / lost revenue).

    Parameters
    ----------
    y_true:
        Ground-truth actual units.
    y_pred:
        Predicted units.

    Returns
    -------
    dict[str, float]
        Dictionary with keys ``over_forecast_units`` and ``under_forecast_units``.
    """
    actual = np.asarray(y_true, dtype=np.float64).ravel()
    predicted = np.asarray(y_pred, dtype=np.float64).ravel()

    diff = predicted - actual
    over_units = float(np.sum(np.clip(diff, 0.0, None)))
    under_units = float(np.sum(np.clip(-diff, 0.0, None)))

    return {
        "over_forecast_units": over_units,
        "under_forecast_units": under_units,
    }


def compute_weekly_candidate_metrics(
    test_df: pd.DataFrame,
    test_preds: dict[str, np.ndarray],
    week_col: str = "week_start",
    target_col: str = "target",
) -> pd.DataFrame:
    """Compute per-week bias and WAPE across test weeks for every candidate.

    Parameters
    ----------
    test_df:
        Test set DataFrame containing target and week identifiers.
    test_preds:
        Mapping candidate name -> array of test predictions (length == len(test_df)).
    week_col:
        Column name identifying the week start date.
    target_col:
        Column name for ground-truth actual units.

    Returns
    -------
    pd.DataFrame
        Table with one row per test week and per-candidate WAPE and bias columns.
    """
    y_true = test_df[target_col].to_numpy(dtype=np.float64)
    weeks = sorted(test_df[week_col].unique())

    records: list[dict[str, Any]] = []
    for w in weeks:
        mask = (test_df[week_col] == w).to_numpy()
        w_str = pd.Timestamp(w).strftime("%Y-%m-%d")
        w_actual = y_true[mask]
        row: dict[str, Any] = {
            "week_start": w_str,
            "actual_units": float(np.sum(w_actual)),
        }
        for cand_name, preds in test_preds.items():
            cand_pred = preds[mask]
            metrics = evaluate(w_actual, cand_pred)
            row[f"{cand_name}_WAPE"] = metrics["WAPE"]
            row[f"{cand_name}_bias"] = metrics["bias"]
        records.append(row)

    return pd.DataFrame(records)


def analyze_tree_seasonality(
    test_df: pd.DataFrame,
    test_preds: dict[str, np.ndarray],
    target_col: str = "target",
    week_col: str = "week_start",
) -> dict[str, dict[str, float]]:
    """State whether tree models under-forecast or over-forecast in October and November and by how much.

    Parameters
    ----------
    test_df:
        Test set DataFrame.
    test_preds:
        Predictions dictionary.
    target_col:
        Ground truth column.
    week_col:
        Week column.

    Returns
    -------
    dict[str, dict[str, float]]
        Breakdown of October, November, and total test bias, WAPE, and asymmetric units
        for tree model candidates.
    """
    tree_cands = [
        c
        for c in test_preds
        if ("random_forest" in c or "lightgbm" in c)
    ]

    dates = pd.to_datetime(test_df[week_col])
    oct_mask = (dates.dt.month == 10).to_numpy()
    nov_mask = (dates.dt.month == 11).to_numpy()
    y_true = test_df[target_col].to_numpy(dtype=np.float64)

    results: dict[str, dict[str, float]] = {}

    for cand in tree_cands:
        preds = test_preds[cand]
        oct_eval = evaluate(y_true[oct_mask], preds[oct_mask])
        nov_eval = evaluate(y_true[nov_mask], preds[nov_mask])
        tot_eval = evaluate(y_true, preds)

        oct_asym = compute_asymmetric_units(y_true[oct_mask], preds[oct_mask])
        nov_asym = compute_asymmetric_units(y_true[nov_mask], preds[nov_mask])

        results[cand] = {
            "oct_bias": oct_eval["bias"],
            "oct_wape": oct_eval["WAPE"],
            "oct_over": oct_asym["over_forecast_units"],
            "oct_under": oct_asym["under_forecast_units"],
            "nov_bias": nov_eval["bias"],
            "nov_wape": nov_eval["WAPE"],
            "nov_over": nov_asym["over_forecast_units"],
            "nov_under": nov_asym["under_forecast_units"],
            "total_bias": tot_eval["bias"],
            "total_wape": tot_eval["WAPE"],
        }

    return results


def compute_intelligence_breakdowns(
    test_df: pd.DataFrame,
    test_preds: dict[str, np.ndarray],
    intelligence_df: pd.DataFrame,
    target_col: str = "target",
) -> dict[str, pd.DataFrame]:
    """Compute forecast error by demand pattern class and ABC class from product_intelligence.csv.

    Parameters
    ----------
    test_df:
        Test set DataFrame containing product_id and target.
    test_preds:
        Dictionary of candidate predictions.
    intelligence_df:
        DataFrame from product_intelligence.csv with columns product_id, pattern_class, abc_class.
    target_col:
        Target column name.

    Returns
    -------
    dict[str, pd.DataFrame]
        Two DataFrames: ``"by_pattern"`` and ``"by_abc"``.
    """
    merged_test = test_df[["product_id", target_col]].copy()
    merged_test["product_id"] = merged_test["product_id"].astype(str)
    intel = intelligence_df[["product_id", "pattern_class", "abc_class"]].copy()
    intel["product_id"] = intel["product_id"].astype(str)
    merged_test = merged_test.merge(intel, on="product_id", how="left")

    for cand_name, preds in test_preds.items():
        merged_test[f"pred_{cand_name}"] = preds

    def _aggregate(group_col: str) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for group_val, grp in merged_test.groupby(group_col, sort=True):
            y_act = grp[target_col].to_numpy(dtype=np.float64)
            n_rows = len(grp)
            total_act = float(np.sum(y_act))
            base_row: dict[str, Any] = {
                group_col: group_val,
                "n_rows": n_rows,
                "actual_units": total_act,
            }
            for cand_name in test_preds:
                y_p = grp[f"pred_{cand_name}"].to_numpy(dtype=np.float64)
                m = evaluate(y_act, y_p)
                asym = compute_asymmetric_units(y_act, y_p)
                base_row[f"{cand_name}_WAPE"] = m["WAPE"]
                base_row[f"{cand_name}_bias"] = m["bias"]
                base_row[f"{cand_name}_MAE"] = m["MAE"]
                base_row[f"{cand_name}_over"] = asym["over_forecast_units"]
                base_row[f"{cand_name}_under"] = asym["under_forecast_units"]
            rows.append(base_row)
        return pd.DataFrame(rows)

    return {
        "by_pattern": _aggregate("pattern_class"),
        "by_abc": _aggregate("abc_class"),
    }


def select_ensemble_weight(
    oof_val_df: pd.DataFrame,
    train_df: pd.DataFrame,
    best_ml_candidate: str = "ridge_raw",
    weight_steps: np.ndarray | None = None,
) -> dict[str, Any]:
    """Find the best ensemble weight w in [0.0, 1.0] in steps of 0.1 on VALIDATION only.

    The ensemble is defined as:
        y_ensemble = clip(w * best_ML + (1 - w) * moving_avg_4, 0, None)

    Parameters
    ----------
    oof_val_df:
        Out-of-fold validation predictions containing columns:
        product_id, week_start, actual, predicted, candidate.
    train_df:
        Training feature frame containing product_id, week_start, roll_mean_4.
    best_ml_candidate:
        Candidate key for best ML model (default "ridge_raw").
    weight_steps:
        Array of weights to evaluate (default 0.0 to 1.0 in steps of 0.1).

    Returns
    -------
    dict[str, Any]
        Dictionary with keys:
        - "chosen_w": best weight w
        - "grid_search": list of metrics for each tested w
        - "best_val_metrics": dict of validation metrics for chosen w
        - "val_predictions": Series/array of OOF ensemble predictions
    """
    if weight_steps is None:
        weight_steps = np.round(np.linspace(0.0, 1.0, 11), 1)

    ml_oof = oof_val_df[oof_val_df["candidate"] == best_ml_candidate].copy()
    ml_oof["product_id"] = ml_oof["product_id"].astype(str)
    ml_oof["week_start"] = pd.to_datetime(ml_oof["week_start"]).dt.normalize()

    train_ref = train_df[["product_id", "week_start", "roll_mean_4"]].copy()
    train_ref["product_id"] = train_ref["product_id"].astype(str)
    train_ref["week_start"] = pd.to_datetime(train_ref["week_start"]).dt.normalize()

    merged = ml_oof.merge(train_ref, on=["product_id", "week_start"], how="left")
    y_actual = merged["actual"].to_numpy(dtype=np.float64)
    ml_pred = merged["predicted"].to_numpy(dtype=np.float64)
    ma4_pred = np.clip(merged["roll_mean_4"].fillna(0.0).to_numpy(dtype=np.float64), 0.0, None)

    grid_results: list[dict[str, Any]] = []
    best_w = float(weight_steps[0])
    best_wape = float("inf")
    best_bias = float("inf")

    for w_val in weight_steps:
        w = float(round(w_val, 2))
        ens_pred = np.clip(w * ml_pred + (1.0 - w) * ma4_pred, 0.0, None)
        eval_metrics = evaluate(y_actual, ens_pred)
        asym = compute_asymmetric_units(y_actual, ens_pred)

        record = {
            "w": w,
            "WAPE": eval_metrics["WAPE"],
            "bias": eval_metrics["bias"],
            "MAE": eval_metrics["MAE"],
            "RMSE": eval_metrics["RMSE"],
            "over_units": asym["over_forecast_units"],
            "under_units": asym["under_forecast_units"],
        }
        grid_results.append(record)

        # Minimize validation WAPE; break ties by absolute bias
        if eval_metrics["WAPE"] < best_wape or (
            np.isclose(eval_metrics["WAPE"], best_wape) and abs(eval_metrics["bias"]) < abs(best_bias)
        ):
            best_wape = eval_metrics["WAPE"]
            best_bias = eval_metrics["bias"]
            best_w = w

    best_pred = np.clip(best_w * ml_pred + (1.0 - best_w) * ma4_pred, 0.0, None)
    best_val_metrics = evaluate(y_actual, best_pred)
    best_val_asym = compute_asymmetric_units(y_actual, best_pred)
    best_val_metrics.update(best_val_asym)

    return {
        "best_ml_candidate": best_ml_candidate,
        "chosen_w": best_w,
        "grid_search": grid_results,
        "best_val_metrics": best_val_metrics,
        "merged_val_df": merged,
        "val_ensemble_pred": best_pred,
    }


def run_diagnostics(
    data_dir: Path = DATA_PROCESSED_DIR,
    models_dir: Path = MODELS_DIR,
) -> dict[str, Any]:
    """Execute complete Task M9 diagnostics suite and print comparison reports."""
    print("=" * 85)
    print("TASK M9: MODEL DIAGNOSTICS & ENSEMBLE EVALUATION")
    print("=" * 85)

    weekly = pd.read_parquet(data_dir / "weekly_demand.parquet")
    closed = pd.read_csv(data_dir / "closed_weeks.csv")
    intel_path = data_dir / "product_intelligence.csv"
    intel_df = pd.read_csv(intel_path) if intel_path.exists() else pd.DataFrame()

    chosen_path = models_dir / "chosen_groups.json"
    with open(chosen_path, encoding="utf-8") as f:
        chosen_meta = json.load(f)
    chosen_groups = tuple(chosen_meta.get("chosen_groups", ["base", "g3"]))
    chosen_features = chosen_meta.get("chosen_feature_list")

    features_df = build_features(weekly, closed_weeks=closed, groups=chosen_groups)
    train_df, test_df = split_chronological(features_df, test_start=DEFAULT_TEST_START)

    params_path = models_dir / "tuned_params.json"
    with open(params_path, encoding="utf-8") as f:
        tuned_meta = json.load(f)

    y_train = train_df["target"].to_numpy(dtype=np.float64)
    y_test = test_df["target"].to_numpy(dtype=np.float64)

    test_preds: dict[str, np.ndarray] = {
        "naive": predict_naive(test_df),
        "moving_avg_4": predict_moving_avg_4(test_df),
    }

    # Fit each tuned candidate on full train_df and predict on test_df
    for cand_name, info in tuned_meta.get("candidates", {}).items():
        m_name = info["model_name"]
        t_mode = info["target_mode"]
        sc_col = info["scale_col"]
        p_dict = info["params"]

        if t_mode == "raw":
            model = make_model(m_name, params=p_dict)
            model.fit(train_df[chosen_features], y_train)
            raw_p = np.asarray(model.predict(test_df[chosen_features]), dtype=np.float64)
            test_preds[cand_name] = np.clip(raw_p, 0.0, None)
        else:
            tr_scale = np.clip(train_df[sc_col].to_numpy(dtype=np.float64), 1.0, None)
            te_scale = np.clip(test_df[sc_col].to_numpy(dtype=np.float64), 1.0, None)
            scaled_y = y_train / tr_scale
            model = make_model(m_name, params=p_dict)
            model.fit(train_df[chosen_features], scaled_y)
            raw_p = np.asarray(model.predict(test_df[chosen_features]), dtype=np.float64)
            test_preds[cand_name] = np.clip(raw_p * te_scale, 0.0, None)

    # 1. Ensemble selection on VALIDATION only
    oof_path = models_dir / "oof_val_predictions.csv"
    oof_df = pd.read_csv(oof_path)
    best_ml = tuned_meta.get("best_candidate", "ridge_raw")
    ensemble_res = select_ensemble_weight(oof_df, train_df, best_ml_candidate=best_ml)
    chosen_w = ensemble_res["chosen_w"]

    test_preds["ensemble"] = np.clip(
        chosen_w * test_preds[best_ml] + (1.0 - chosen_w) * test_preds["moving_avg_4"],
        0.0,
        None,
    )

    # 2. Overall Candidate Comparison Table
    print("\n1. OVERALL CANDIDATE COMPARISON (Test Set: 8 Weeks, 800 Rows)")
    print("-" * 85)
    comp_rows: list[dict[str, Any]] = []
    for cand_name, preds in test_preds.items():
        m = evaluate(y_test, preds)
        asym = compute_asymmetric_units(y_test, preds)
        comp_rows.append(
            {
                "candidate": cand_name,
                "WAPE": m["WAPE"],
                "bias": m["bias"],
                "MAE": m["MAE"],
                "RMSE": m["RMSE"],
                "over_units": asym["over_forecast_units"],
                "under_units": asym["under_forecast_units"],
            }
        )
    comp_df = pd.DataFrame(comp_rows)
    print(
        comp_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}" if abs(x) < 1 else f"{x:.2f}",
        )
    )

    # 3. Weekly breakdown (8 weeks x candidates table)
    print("\n2. PER-WEEK WAPE AND BIAS TABLE (8 Test Weeks x Candidates)")
    print("-" * 85)
    weekly_df = compute_weekly_candidate_metrics(test_df, test_preds)
    # Pivot for clean display: week_start as index, candidate WAPE / bias columns
    display_weekly = weekly_df.copy()
    print(
        display_weekly.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}" if abs(x) < 1 else f"{x:.2f}",
        )
    )

    # 4. Tree models October vs. November analysis
    print("\n3. TREE MODELS OCTOBER VS NOVEMBER ANALYSIS")
    print("-" * 85)
    tree_analysis = analyze_tree_seasonality(test_df, test_preds)
    for tree_model, vals in tree_analysis.items():
        print(f"Candidate: {tree_model}")
        print(
            f"  October : bias = {vals['oct_bias']:+7.2f} units/wk | WAPE = {vals['oct_wape']:.4f} | "
            f"over = {vals['oct_over']:7.0f} | under = {vals['oct_under']:7.0f}"
        )
        print(
            f"  November: bias = {vals['nov_bias']:+7.2f} units/wk | WAPE = {vals['nov_wape']:.4f} | "
            f"over = {vals['nov_over']:7.0f} | under = {vals['nov_under']:7.0f}"
        )
        print(
            f"  Total   : bias = {vals['total_bias']:+7.2f} units/wk | WAPE = {vals['total_wape']:.4f}"
        )

    # 5. Breakdown by demand pattern class and ABC class
    intel_breakdowns: dict[str, pd.DataFrame] = {}
    if not intel_df.empty:
        print("\n4. FORECAST ERROR BY DEMAND PATTERN CLASS & ABC CLASS")
        print("-" * 85)
        intel_breakdowns = compute_intelligence_breakdowns(
            test_df, test_preds, intel_df
        )
        print("\n--- By Demand Pattern Class ---")
        cols_pat = [
            c
            for c in intel_breakdowns["by_pattern"].columns
            if any(k in c for k in ["pattern", "actual", "ridge", "ensemble", "lightgbm_raw", "moving_avg_4"])
        ]
        print(
            intel_breakdowns["by_pattern"][cols_pat].to_string(
                index=False, float_format=lambda x: f"{x:.4f}" if abs(x) < 1 else f"{x:.2f}"
            )
        )

        print("\n--- By ABC Class ---")
        cols_abc = [
            c
            for c in intel_breakdowns["by_abc"].columns
            if any(k in c for k in ["abc", "actual", "ridge", "ensemble", "lightgbm_raw", "moving_avg_4"])
        ]
        print(
            intel_breakdowns["by_abc"][cols_abc].to_string(
                index=False, float_format=lambda x: f"{x:.4f}" if abs(x) < 1 else f"{x:.2f}"
            )
        )

    # 6. Ensemble details
    print("\n5. ENSEMBLE CANDIDATE SUMMARY")
    print("-" * 85)
    print(f"Best ML model: {best_ml}")
    print(f"Validation grid search over w in [0.0, 1.0] in steps of 0.1:")
    for row in ensemble_res["grid_search"]:
        marker = " <--- CHOSEN (MIN VAL WAPE)" if row["w"] == chosen_w else ""
        print(
            f"  w={row['w']:.1f}: Val WAPE={row['WAPE']:.4f}, bias={row['bias']:+6.2f}, "
            f"MAE={row['MAE']:6.2f}, RMSE={row['RMSE']:6.2f}{marker}"
        )

    ens_test = evaluate(y_test, test_preds["ensemble"])
    ens_test_asym = compute_asymmetric_units(y_test, test_preds["ensemble"])
    print(f"\nChosen w = {chosen_w}")
    print(
        f"Validation Metrics: WAPE = {ensemble_res['best_val_metrics']['WAPE']:.4f}, "
        f"bias = {ensemble_res['best_val_metrics']['bias']:+.2f}, "
        f"MAE = {ensemble_res['best_val_metrics']['MAE']:.2f}, "
        f"over = {ensemble_res['best_val_metrics']['over_forecast_units']:.0f}, "
        f"under = {ensemble_res['best_val_metrics']['under_forecast_units']:.0f}"
    )
    print(
        f"Test Metrics      : WAPE = {ens_test['WAPE']:.4f}, bias = {ens_test['bias']:+.2f}, "
        f"MAE = {ens_test['MAE']:.2f}, "
        f"over = {ens_test_asym['over_forecast_units']:.0f}, "
        f"under = {ens_test_asym['under_forecast_units']:.0f}"
    )
    print("=" * 85 + "\n")

    return {
        "comparison_table": comp_df,
        "weekly_table": weekly_df,
        "tree_analysis": tree_analysis,
        "intelligence_breakdowns": intel_breakdowns,
        "ensemble_results": ensemble_res,
        "test_predictions": test_preds,
    }


if __name__ == "__main__":
    run_diagnostics()
