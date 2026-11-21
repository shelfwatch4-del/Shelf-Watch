"""Diagnostic script for M5 baseline and ML model results on the test set."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.features import FEATURE_COLUMNS, build_features
from src.train import (
    DEFAULT_TEST_START,
    MODEL_NAMES,
    evaluate,
    make_model,
    predict_moving_avg_4,
    predict_naive,
    split_chronological,
)

DATA_PROCESSED_DIR = Path("data/processed")


def main() -> None:
    print("=" * 80)
    print("M5 DIAGNOSIS: BASELINES AND DEFAULT MODELS ON TEST SET")
    print("=" * 80)

    # 1. Load data and build features
    weekly = pd.read_parquet(DATA_PROCESSED_DIR / "weekly_demand.parquet")
    closed_weeks = pd.read_csv(DATA_PROCESSED_DIR / "closed_weeks.csv")
    segments = pd.read_csv(DATA_PROCESSED_DIR / "product_segments.csv")

    features_df = build_features(weekly, closed_weeks=closed_weeks)
    train_df, test_df = split_chronological(features_df, test_start=DEFAULT_TEST_START)

    print(f"\nTrain rows: {len(train_df)}, Test rows: {len(test_df)}")
    print(f"Products: {train_df['product_id'].nunique()}, Test weeks: {test_df['week_start'].nunique()}")

    # 2. Compute predictions for all 5 models
    models = ["naive", "moving_avg_4", "ridge", "random_forest", "lightgbm"]
    test_preds: dict[str, np.ndarray] = {}
    train_preds: dict[str, np.ndarray] = {}

    y_train = train_df["target"].to_numpy(dtype=np.float64)
    y_test = test_df["target"].to_numpy(dtype=np.float64)

    # Baselines
    test_preds["naive"] = predict_naive(test_df)
    test_preds["moving_avg_4"] = predict_moving_avg_4(test_df)

    # ML models
    trained_estimators = {}
    for name in MODEL_NAMES:
        model = make_model(name, params=None)
        model.fit(train_df[FEATURE_COLUMNS], y_train)
        trained_estimators[name] = model

        preds_test = np.asarray(model.predict(test_df[FEATURE_COLUMNS]), dtype=np.float64)
        test_preds[name] = np.clip(preds_test, 0.0, None)

        preds_train = np.asarray(model.predict(train_df[FEATURE_COLUMNS]), dtype=np.float64)
        train_preds[name] = np.clip(preds_train, 0.0, None)

    # Overall Test Metrics
    print("\n" + "-" * 80)
    print("OVERALL TEST METRICS (8 weeks, 800 rows)")
    print("-" * 80)
    overall_rows = []
    for m in models:
        metrics = evaluate(y_test, test_preds[m])
        overall_rows.append({"model": m, **metrics})
    overall_df = pd.DataFrame(overall_rows)
    print(overall_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # 1. Per test week: WAPE and bias for each model
    print("\n" + "-" * 80)
    print("1. PER TEST WEEK BREAKDOWN (WAPE and bias)")
    print("-" * 80)
    test_weeks = sorted(test_df["week_start"].unique())

    weekly_records = []
    for w in test_weeks:
        mask = (test_df["week_start"] == w).to_numpy()
        w_str = pd.Timestamp(w).strftime("%Y-%m-%d")
        w_actual = y_test[mask]
        row_entry = {"week_start": w_str, "actual_units": float(w_actual.sum())}
        for m in models:
            m_metrics = evaluate(w_actual, test_preds[m][mask])
            row_entry[f"{m}_WAPE"] = m_metrics["WAPE"]
            row_entry[f"{m}_bias"] = m_metrics["bias"]
        weekly_records.append(row_entry)

    weekly_df = pd.DataFrame(weekly_records)
    print(weekly_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Also display per-model focused view for weekly
    print("\nWeekly WAPE comparison:")
    wape_cols = ["week_start", "actual_units"] + [f"{m}_WAPE" for m in models]
    print(weekly_df[wape_cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nWeekly bias comparison:")
    bias_cols = ["week_start", "actual_units"] + [f"{m}_bias" for m in models]
    print(weekly_df[bias_cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # 2. By volume segment: WAPE and bias for each model
    print("\n" + "-" * 80)
    print("2. BY VOLUME SEGMENT BREAKDOWN (WAPE and bias)")
    print("-" * 80)
    test_df_merged = test_df.merge(segments, on="product_id", how="left")
    segment_order = ["low", "medium", "high"]

    seg_records = []
    for seg in segment_order:
        mask = (test_df_merged["segment"] == seg).to_numpy()
        seg_actual = y_test[mask]
        row_entry = {
            "segment": seg,
            "n_products": test_df_merged.loc[mask, "product_id"].nunique(),
            "n_rows": int(mask.sum()),
            "actual_units": float(seg_actual.sum()),
        }
        for m in models:
            m_metrics = evaluate(seg_actual, test_preds[m][mask])
            row_entry[f"{m}_WAPE"] = m_metrics["WAPE"]
            row_entry[f"{m}_bias"] = m_metrics["bias"]
        seg_records.append(row_entry)

    seg_df = pd.DataFrame(seg_records)
    print(seg_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nSegment WAPE comparison:")
    wape_seg_cols = ["segment", "actual_units"] + [f"{m}_WAPE" for m in models]
    print(seg_df[wape_seg_cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nSegment bias comparison:")
    bias_seg_cols = ["segment", "actual_units"] + [f"{m}_bias" for m in models]
    print(seg_df[bias_seg_cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # 3. Share of test rows where actual exceeds that product's maximum training target
    print("\n" + "-" * 80)
    print("3. EXTREME ROWS: TEST ACTUAL > PRODUCT TRAINING MAX")
    print("-" * 80)
    prod_train_max = train_df.groupby("product_id")["target"].max()
    test_train_max = test_df["product_id"].map(prod_train_max).to_numpy()

    exceeded_mask = y_test > test_train_max
    n_exceeded = int(exceeded_mask.sum())
    n_total_test = len(test_df)
    share_exceeded = n_exceeded / n_total_test

    print(f"Total test rows: {n_total_test}")
    print(f"Test rows where actual > product training max: {n_exceeded} ({share_exceeded:.4%})")

    under_pred_stats = []
    for m in models:
        m_preds_exc = test_preds[m][exceeded_mask]
        m_actual_exc = y_test[exceeded_mask]
        n_under = int((m_preds_exc < m_actual_exc).sum())
        share_under = n_under / n_exceeded if n_exceeded > 0 else 0.0
        under_pred_stats.append({
            "model": m,
            "n_under_predicted": n_under,
            "total_exceeded_rows": n_exceeded,
            "share_under_predicted": share_under,
            "mean_under_gap": float(np.mean(m_actual_exc - m_preds_exc)),
        })
    under_df = pd.DataFrame(under_pred_stats)
    print("\nUnder-prediction on exceeded rows:")
    print(under_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # 4. For LightGBM: 10 products and 10 rows contributing most absolute error
    print("\n" + "-" * 80)
    print("4. LIGHTGBM ERROR CONCENTRATION ANALYSIS")
    print("-" * 80)
    lgbm_test_pred = test_preds["lightgbm"]
    lgbm_abs_err = np.abs(lgbm_test_pred - y_test)
    total_lgbm_abs_err = float(lgbm_abs_err.sum())
    print(f"Total LightGBM Absolute Error on Test Set: {total_lgbm_abs_err:.2f}")

    test_analysis = test_df.copy()
    test_analysis["actual"] = y_test
    test_analysis["predicted"] = lgbm_test_pred
    test_analysis["abs_error"] = lgbm_abs_err
    test_analysis["error"] = lgbm_test_pred - y_test

    # Top 10 products
    prod_err = (
        test_analysis.groupby("product_id")
        .agg(
            total_abs_error=("abs_error", "sum"),
            mean_abs_error=("abs_error", "mean"),
            net_bias=("error", "mean"),
            total_actual=("actual", "sum"),
            total_predicted=("predicted", "sum"),
        )
        .sort_values("total_abs_error", ascending=False)
        .reset_index()
    )
    prod_err["share_of_total_abs_error"] = prod_err["total_abs_error"] / total_lgbm_abs_err
    prod_err["cumulative_share"] = prod_err["share_of_total_abs_error"].cumsum()

    print("\nTop 10 Products by LightGBM Absolute Error:")
    top_10_prods = prod_err.head(10)
    print(top_10_prods.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"Top 10 products share of total error: {top_10_prods['share_of_total_abs_error'].sum():.4%}")

    # Top 10 (product, week) rows
    row_err = test_analysis.sort_values("abs_error", ascending=False).reset_index(drop=True)
    row_err["share_of_total_abs_error"] = row_err["abs_error"] / total_lgbm_abs_err
    row_err["cumulative_share"] = row_err["share_of_total_abs_error"].cumsum()

    top_10_rows = row_err.head(10)[
        [
            "product_id",
            "week_start",
            "actual",
            "predicted",
            "abs_error",
            "error",
            "share_of_total_abs_error",
            "cumulative_share",
        ]
    ]
    print("\nTop 10 (product, week) Rows by LightGBM Absolute Error:")
    print(top_10_rows.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"Top 10 rows share of total error: {top_10_rows['share_of_total_abs_error'].sum():.4%}")

    # 5. LightGBM training WAPE versus test WAPE (overfitting check)
    print("\n" + "-" * 80)
    print("5. LIGHTGBM OVERFITTING CHECK: TRAIN VS TEST")
    print("-" * 80)
    lgbm_train_pred = train_preds["lightgbm"]
    train_metrics = evaluate(y_train, lgbm_train_pred)
    test_metrics = evaluate(y_test, lgbm_test_pred)

    comp_rows = [
        {"split": "train", "n_rows": len(train_df), **train_metrics},
        {"split": "test", "n_rows": len(test_df), **test_metrics},
    ]
    comp_df = pd.DataFrame(comp_rows)
    print(comp_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(
        f"\nLightGBM Train WAPE: {train_metrics['WAPE']:.4f} vs Test WAPE: {test_metrics['WAPE']:.4f} "
        f"(Ratio Test/Train: {test_metrics['WAPE'] / train_metrics['WAPE']:.2f}x)"
    )
    print(
        f"LightGBM Train Bias: {train_metrics['bias']:.4f} vs Test Bias: {test_metrics['bias']:.4f}"
    )

    # All models train vs test comparison for additional context
    print("\nAll Models Train vs Test WAPE and Bias:")
    all_train_test = []
    for m in models:
        if m in train_preds:
            tr_m = evaluate(y_train, train_preds[m])
            te_m = evaluate(y_test, test_preds[m])
            all_train_test.append({
                "model": m,
                "train_WAPE": tr_m["WAPE"],
                "test_WAPE": te_m["WAPE"],
                "wape_gap": te_m["WAPE"] - tr_m["WAPE"],
                "train_bias": tr_m["bias"],
                "test_bias": te_m["bias"],
            })
    print(pd.DataFrame(all_train_test).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("=" * 80)


if __name__ == "__main__":
    main()
