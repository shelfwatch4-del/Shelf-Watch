"""Result files generation for Task M12 (Contracts C2, C3, C4).

SRS: FR-M-5, FR-M-10 | Contracts: C2, C3, C4
Produces:
- data/processed/predictions.csv (C2)
- data/processed/metrics.csv (C3)
- data/processed/next_week_forecast.csv (C4)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.features import build_features, build_future_features
from src.quantiles import sort_and_clip_intervals
from src.train import (
    DEFAULT_TEST_START,
    RANDOM_SEED,
    evaluate,
    make_model,
    predict_moving_avg_4,
    predict_naive,
    split_chronological,
)

DATA_PROCESSED_DIR: Path = Path("data/processed")
MODELS_DIR: Path = Path("models")
FORECAST_TARGET_WEEK: str = "2011-12-05"


def compute_product_sigma_error(
    oof_val_df: pd.DataFrame,
    segments_df: pd.DataFrame,
    best_candidate: str = "ensemble",
    min_observations: int = 6,
) -> dict[str, float]:
    """Compute per-product forecast error std on out-of-fold validation errors with fallback.

    SRS FR-M-10:
    sigma_error = per-product standard deviation of the best model's OUT-OF-FOLD
    validation errors (predicted minus actual); with fewer than 6 validation observations
    use the pooled value of the product's volume segment.

    Parameters
    ----------
    oof_val_df:
        Out-of-fold predictions DataFrame containing product_id, predicted, actual, candidate.
    segments_df:
        DataFrame from product_segments.csv with product_id, segment.
    best_candidate:
        Key identifying best model in oof_val_df (default 'ensemble').
    min_observations:
        Minimum observations required before falling back to pooled segment sigma (default 6).

    Returns
    -------
    dict[str, float]
        Mapping product_id -> sigma_error (>= 0.0).
    """
    oof_sub = oof_val_df[oof_val_df["candidate"] == best_candidate].copy()
    if oof_sub.empty and best_candidate == "ensemble":
        # Fallback to ridge_raw if ensemble not explicitly in oof_val_df
        oof_sub = oof_val_df[oof_val_df["candidate"] == "ridge_raw"].copy()
    oof_sub["product_id"] = oof_sub["product_id"].astype(str)
    oof_sub["err"] = oof_sub["predicted"] - oof_sub["actual"]

    seg_map = segments_df.copy()
    seg_map["product_id"] = seg_map["product_id"].astype(str)
    merged = oof_sub.merge(seg_map[["product_id", "segment"]], on="product_id", how="left")

    # Pooled standard deviation per volume segment
    seg_pooled = merged.groupby("segment")["err"].std().to_dict()
    global_pooled = float(merged["err"].std()) if len(merged) > 1 else 50.0

    # Per-product stats
    prod_groups = merged.groupby("product_id")
    sigma_dict: dict[str, float] = {}

    for pid, grp in prod_groups:
        n_obs = len(grp)
        seg = str(grp["segment"].iloc[0]) if "segment" in grp else "medium"
        pooled_val = float(seg_pooled.get(seg, global_pooled))

        if n_obs < min_observations:
            sigma_val = pooled_val
        else:
            s = float(grp["err"].std(ddof=1))
            sigma_val = pooled_val if (pd.isna(s) or s <= 0.0) else s

        sigma_dict[pid] = max(0.0, float(sigma_val))

    # Guard any missing products in segments_df
    for pid in seg_map["product_id"].unique():
        if pid not in sigma_dict:
            seg = str(seg_map.loc[seg_map["product_id"] == pid, "segment"].iloc[0])
            sigma_dict[pid] = max(0.0, float(seg_pooled.get(seg, global_pooled)))

    return sigma_dict


def generate_next_week_forecast(
    data_dir: Path | str = DATA_PROCESSED_DIR,
    models_dir: Path | str = MODELS_DIR,
    target_week: str = FORECAST_TARGET_WEEK,
    output_path: Path | str | None = None,
) -> pd.DataFrame:
    """Generate and write next_week_forecast.csv (Contract C4 & Addendum C4).

    Columns:
    product_id, week_start, forecast_weekly, forecast_p10, forecast_p90,
    sigma_error, avg_weekly_units_train, last_avg_price

    Parameters
    ----------
    data_dir:
        Directory containing processed tables.
    models_dir:
        Directory holding saved model artifacts.
    target_week:
        Target forecast Monday date (default '2011-12-05').
    output_path:
        Optional custom save destination.

    Returns
    -------
    pd.DataFrame
        Contract C4 DataFrame (exactly 100 products).
    """
    d_dir = Path(data_dir)
    m_dir = Path(models_dir)
    out_file = d_dir / "next_week_forecast.csv" if output_path is None else Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    weekly = pd.read_parquet(d_dir / "weekly_demand.parquet")
    closed = pd.read_csv(d_dir / "closed_weeks.csv")
    segments = pd.read_csv(d_dir / "product_segments.csv")
    oof_val = pd.read_csv(m_dir / "oof_val_predictions.csv")

    with open(m_dir / "chosen_groups.json", encoding="utf-8") as f:
        chosen_meta = json.load(f)
    chosen_groups = tuple(chosen_meta.get("chosen_groups", ["base", "g3"]))

    with open(m_dir / "feature_list.json", encoding="utf-8") as f:
        features = list(json.load(f))

    # 1. Build future features for target week
    future_df = build_future_features(
        weekly=weekly,
        closed_weeks=closed,
        target_week=target_week,
        groups=chosen_groups,
    )
    future_df = future_df.sort_values("product_id").reset_index(drop=True)

    # 2. Point forecast & quantiles
    best_model = joblib.load(m_dir / "best_model.joblib")
    q10_model = joblib.load(m_dir / "q10.joblib")
    q90_model = joblib.load(m_dir / "q90.joblib")

    X_fut = future_df[features]
    raw_ridge = np.asarray(best_model.predict(X_fut), dtype=np.float64)
    raw_p10 = np.asarray(q10_model.predict(X_fut), dtype=np.float64)
    raw_p90 = np.asarray(q90_model.predict(X_fut), dtype=np.float64)

    # Point forecast is Ensemble: 0.8 * ridge_raw + 0.2 * moving_avg_4
    ma4_fut = future_df["roll_mean_4"].to_numpy(dtype=np.float64)
    raw_pt = 0.8 * raw_ridge + 0.2 * ma4_fut

    # Residual conformal offset
    c_offset = 0.0
    cov_path = m_dir / "interval_coverage.json"
    if cov_path.exists():
        with open(cov_path, encoding="utf-8") as f:
            cov_meta = json.load(f)
            c_offset = float(cov_meta.get("residual_correction_offset", 0.0))

    p10_cal = raw_p10 - c_offset
    p90_cal = raw_p90 + c_offset

    p10_sorted, pt_sorted, p90_sorted = sort_and_clip_intervals(
        p10_cal, np.clip(raw_pt, 0.0, None), p90_cal
    )

    # 3. sigma_error from Ensemble out-of-fold validation errors
    sigma_map = compute_product_sigma_error(oof_val, segments, best_candidate="ensemble")

    # 4. avg_weekly_units_train (excluding closure week, which is not in weekly_demand)
    train_cutoff = pd.Timestamp(DEFAULT_TEST_START).normalize()
    train_mask = pd.to_datetime(weekly["week_start"]).dt.normalize() < train_cutoff
    avg_train_map = (
        weekly.loc[train_mask]
        .groupby("product_id")["units"]
        .mean()
        .to_dict()
    )

    # 5. last_avg_price (product's average price in last open week before forecast week)
    last_price_map = (
        weekly.sort_values("week_start")
        .groupby("product_id")["avg_price"]
        .last()
        .to_dict()
    )

    # Construct Contract C4 rows
    rows: list[dict[str, Any]] = []
    target_str = pd.Timestamp(target_week).strftime("%Y-%m-%d")

    for idx, row in future_df.iterrows():
        pid = str(row["product_id"])
        pt = round(float(pt_sorted[idx]), 2)
        p10 = round(float(p10_sorted[idx]), 2)
        p90 = round(float(p90_sorted[idx]), 2)
        sig = round(float(sigma_map.get(pid, 50.0)), 2)
        avg_u = round(float(avg_train_map.get(pid, 100.0)), 2)
        lp = round(float(last_price_map.get(pid, 1.0)), 2)

        rows.append(
            {
                "product_id": pid,
                "week_start": target_str,
                "forecast_weekly": pt,
                "forecast_p10": p10,
                "forecast_p90": p90,
                "sigma_error": sig,
                "avg_weekly_units_train": avg_u,
                "last_avg_price": lp,
            }
        )

    c4_df = pd.DataFrame(
        rows,
        columns=[
            "product_id",
            "week_start",
            "forecast_weekly",
            "forecast_p10",
            "forecast_p90",
            "sigma_error",
            "avg_weekly_units_train",
            "last_avg_price",
        ],
    )
    c4_df.to_csv(out_file, index=False)
    return c4_df


def generate_predictions_and_metrics(
    data_dir: Path | str = DATA_PROCESSED_DIR,
    models_dir: Path | str = MODELS_DIR,
    test_start: str = DEFAULT_TEST_START,
    ensemble_w: float = 0.8,
    predictions_path: Path | str | None = None,
    metrics_path: Path | str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate predictions.csv (C2) and metrics.csv (C3) across all candidates.

    SRS: FR-M-5 | Contracts: C2, C3

    Candidates evaluated:
    - naive
    - moving_avg_4
    - ridge
    - ridge_scale_aware
    - random_forest
    - random_forest_scale_aware
    - lightgbm
    - lightgbm_scale_aware
    - ensemble

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (predictions_df, metrics_df) matching C2 and C3 contracts.
    """
    d_dir = Path(data_dir)
    m_dir = Path(models_dir)
    pred_file = d_dir / "predictions.csv" if predictions_path is None else Path(predictions_path)
    met_file = d_dir / "metrics.csv" if metrics_path is None else Path(metrics_path)

    pred_file.parent.mkdir(parents=True, exist_ok=True)
    met_file.parent.mkdir(parents=True, exist_ok=True)

    weekly = pd.read_parquet(d_dir / "weekly_demand.parquet")
    closed = pd.read_csv(d_dir / "closed_weeks.csv")
    oof_val_df = pd.read_csv(m_dir / "oof_val_predictions.csv")

    with open(m_dir / "chosen_groups.json", encoding="utf-8") as f:
        chosen_meta = json.load(f)
    chosen_groups = tuple(chosen_meta.get("chosen_groups", ["base", "g3"]))
    features = list(chosen_meta.get("chosen_feature_list", []))

    with open(m_dir / "tuned_params.json", encoding="utf-8") as f:
        tuned_meta = json.load(f)
    candidates_meta = tuned_meta.get("candidates", {})

    # Build features & split
    features_df = build_features(weekly, closed_weeks=closed, groups=chosen_groups)
    train_df, test_df = split_chronological(features_df, test_start=test_start)

    y_train = train_df["target"].to_numpy(dtype=np.float64)
    y_test = test_df["target"].to_numpy(dtype=np.float64)

    # 1. Align validation frame
    # Extract reference validation metadata from oof_val_df (candidate ridge_raw)
    ref_oof = oof_val_df[oof_val_df["candidate"] == "ridge_raw"].copy().reset_index(drop=True)
    val_pids = ref_oof["product_id"].astype(str).tolist()
    val_weeks = ref_oof["week_start"].tolist()
    val_actuals = ref_oof["actual"].to_numpy(dtype=np.float64)

    # Merge with train_df to get feature columns for validation folds
    ref_lookup = ref_oof[["product_id", "week_start"]].copy()
    ref_lookup["product_id"] = ref_lookup["product_id"].astype(str)
    ref_lookup["week_start"] = pd.to_datetime(ref_lookup["week_start"]).dt.normalize()

    train_lookup = train_df.copy()
    train_lookup["product_id"] = train_lookup["product_id"].astype(str)
    train_lookup["week_start"] = pd.to_datetime(train_lookup["week_start"]).dt.normalize()
    merged_val = ref_lookup.merge(train_lookup, on=["product_id", "week_start"], how="inner")

    # 2. Compute validation predictions for each candidate
    val_preds_map: dict[str, np.ndarray] = {
        "naive": predict_naive(merged_val),
        "moving_avg_4": predict_moving_avg_4(merged_val),
    }

    # Populate ML validation predictions from oof_val_df
    cand_name_aliases = {
        "ridge_raw": "ridge_raw",
        "ridge_scale_aware": "ridge_scale_aware",
        "random_forest_raw": "random_forest_raw",
        "random_forest_scale_aware": "random_forest_scale_aware",
        "lightgbm_raw": "lightgbm_raw",
        "lightgbm_scale_aware": "lightgbm_scale_aware",
    }

    for target_name, oof_key in cand_name_aliases.items():
        sub_oof = oof_val_df[oof_val_df["candidate"] == oof_key].copy().reset_index(drop=True)
        val_preds_map[target_name] = np.clip(
            sub_oof["predicted"].to_numpy(dtype=np.float64), 0.0, None
        )

    # Ensemble on validation
    best_ml_val = val_preds_map["ridge_raw"]
    val_preds_map["ensemble"] = np.clip(
        ensemble_w * best_ml_val + (1.0 - ensemble_w) * val_preds_map["moving_avg_4"],
        0.0,
        None,
    )

    # Ensure ensemble is recorded in oof_val_predictions.csv
    ens_oof = ref_oof.copy()
    ens_oof["predicted"] = val_preds_map["ensemble"]
    ens_oof["candidate"] = "ensemble"
    updated_oof = pd.concat(
        [oof_val_df[oof_val_df["candidate"] != "ensemble"], ens_oof],
        ignore_index=True,
    )
    updated_oof.to_csv(m_dir / "oof_val_predictions.csv", index=False)

    # 3. Compute test predictions for each candidate (fit on train_df, predict on test_df)
    test_preds_map: dict[str, np.ndarray] = {
        "naive": predict_naive(test_df),
        "moving_avg_4": predict_moving_avg_4(test_df),
    }

    for target_name, cand_key in cand_name_aliases.items():
        c_info = candidates_meta[cand_key]
        m_name = c_info["model_name"]
        t_mode = c_info["target_mode"]
        sc_col = c_info["scale_col"]
        p_dict = c_info["params"]

        if t_mode == "raw":
            mdl = make_model(m_name, params=p_dict)
            mdl.fit(train_df[features], y_train)
            raw_p = np.asarray(mdl.predict(test_df[features]), dtype=np.float64)
            test_preds_map[target_name] = np.clip(raw_p, 0.0, None)
        else:
            tr_scale = np.clip(train_df[sc_col].to_numpy(dtype=np.float64), 1.0, None)
            te_scale = np.clip(test_df[sc_col].to_numpy(dtype=np.float64), 1.0, None)
            scaled_y = y_train / tr_scale
            mdl = make_model(m_name, params=p_dict)
            mdl.fit(train_df[features], scaled_y)
            raw_p = np.asarray(mdl.predict(test_df[features]), dtype=np.float64)
            test_preds_map[target_name] = np.clip(raw_p * te_scale, 0.0, None)

    # Ensemble on test
    test_preds_map["ensemble"] = np.clip(
        ensemble_w * test_preds_map["ridge_raw"] + (1.0 - ensemble_w) * test_preds_map["moving_avg_4"],
        0.0,
        None,
    )

    # 4. Assemble predictions.csv (Contract C2)
    pred_rows: list[dict[str, Any]] = []

    # Validation predictions (2,500 rows per candidate)
    for model_name, preds in val_preds_map.items():
        for i in range(len(preds)):
            pred_rows.append(
                {
                    "product_id": val_pids[i],
                    "week_start": val_weeks[i],
                    "actual": round(float(val_actuals[i]), 2),
                    "predicted": round(float(preds[i]), 2),
                    "model_name": model_name,
                    "split": "val",
                }
            )

    # Test predictions (800 rows per candidate)
    test_pids = test_df["product_id"].astype(str).tolist()
    test_weeks = pd.to_datetime(test_df["week_start"]).dt.strftime("%Y-%m-%d").tolist()
    test_actuals = y_test

    for model_name, preds in test_preds_map.items():
        for i in range(len(preds)):
            pred_rows.append(
                {
                    "product_id": test_pids[i],
                    "week_start": test_weeks[i],
                    "actual": round(float(test_actuals[i]), 2),
                    "predicted": round(float(preds[i]), 2),
                    "model_name": model_name,
                    "split": "test",
                }
            )

    c2_df = pd.DataFrame(
        pred_rows,
        columns=["product_id", "week_start", "actual", "predicted", "model_name", "split"],
    )
    c2_df.to_csv(pred_file, index=False)

    # 5. Assemble metrics.csv (Contract C3)
    metric_rows: list[dict[str, Any]] = []
    candidate_order = [
        "naive",
        "moving_avg_4",
        "ridge_raw",
        "random_forest_raw",
        "lightgbm_raw",
        "ridge_scale_aware",
        "random_forest_scale_aware",
        "lightgbm_scale_aware",
        "ensemble",
    ]

    for c_name in candidate_order:
        # Validation metrics
        v_eval = evaluate(val_actuals, val_preds_map[c_name])
        metric_rows.append(
            {
                "model_name": c_name,
                "split": "val",
                "MAE": round(float(v_eval["MAE"]), 4),
                "RMSE": round(float(v_eval["RMSE"]), 4),
                "WAPE": round(float(v_eval["WAPE"]), 4),
                "bias": round(float(v_eval["bias"]), 4),
            }
        )

        # Test metrics
        t_eval = evaluate(test_actuals, test_preds_map[c_name])
        metric_rows.append(
            {
                "model_name": c_name,
                "split": "test",
                "MAE": round(float(t_eval["MAE"]), 4),
                "RMSE": round(float(t_eval["RMSE"]), 4),
                "WAPE": round(float(t_eval["WAPE"]), 4),
                "bias": round(float(t_eval["bias"]), 4),
            }
        )

    c3_df = pd.DataFrame(
        metric_rows,
        columns=["model_name", "split", "MAE", "RMSE", "WAPE", "bias"],
    )
    c3_df.to_csv(met_file, index=False)

    return c2_df, c3_df


def generate_all_m12_results(
    data_dir: Path | str = DATA_PROCESSED_DIR,
    models_dir: Path | str = MODELS_DIR,
) -> dict[str, Any]:
    """Execute complete Task M12 pipeline and produce C2, C3, and C4 files."""
    print("=" * 85)
    print("TASK M12: RESULT FILES GENERATION (CONTRACTS C2, C3, C4)")
    print("=" * 85)

    c2_df, c3_df = generate_predictions_and_metrics(data_dir=data_dir, models_dir=models_dir)
    c4_df = generate_next_week_forecast(data_dir=data_dir, models_dir=models_dir)

    print(f"Generated predictions.csv: {len(c2_df)} rows across {c2_df['model_name'].nunique()} candidates")
    print(f"Generated metrics.csv    : {len(c3_df)} rows across {c3_df['model_name'].nunique()} candidates and 2 splits")
    print(f"Generated next_week_forecast.csv: {len(c4_df)} products for week {c4_df['week_start'].iloc[0]}")
    print("\nMetrics Summary (Test Split):")
    print(c3_df[c3_df["split"] == "test"].to_string(index=False))
    print("=" * 85)

    return {
        "predictions": c2_df,
        "metrics": c3_df,
        "next_week_forecast": c4_df,
    }


if __name__ == "__main__":
    generate_all_m12_results()
