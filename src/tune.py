"""Hyperparameter tuning with week-based cross-validation and scale-aware modeling."""

from __future__ import annotations

from collections.abc import Sequence
import json
from pathlib import Path
import time
from typing import Any

from lightgbm import LGBMRegressor
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from src.features import FEATURE_COLUMNS, build_features
from src.train import (
    DEFAULT_TEST_START,
    RANDOM_SEED,
    evaluate,
    make_model,
    predict_moving_avg_4,
    predict_naive,
    split_chronological,
)

CANDIDATE_NAMES: tuple[str, ...] = (
    "ridge_raw",
    "ridge_scale_aware",
    "random_forest_raw",
    "random_forest_scale_aware",
    "lightgbm_raw",
    "lightgbm_scale_aware",
)

DATA_PROCESSED_DIR: Path = Path("data/processed")
MODELS_DIR: Path = Path("models")


class WeekTimeSeriesSplit:
    """Expanding-window TimeSeriesSplit over unique weeks in a training frame.

    Ensures that for every fold, all train weeks strictly precede all validation weeks,
    and no week appears in both sets.
    """

    def __init__(self, n_splits: int = 5) -> None:
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2.")
        self.n_splits = n_splits

    def split(
        self,
        df: pd.DataFrame,
        week_col: str = "week_start",
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Yield (train_indices, val_indices) partitioned by unique weeks."""
        if week_col not in df.columns:
            raise ValueError(f"DataFrame must contain '{week_col}'.")

        week_series = pd.to_datetime(df[week_col]).dt.normalize()
        unique_weeks = np.sort(week_series.unique())

        if len(unique_weeks) < self.n_splits + 1:
            raise ValueError(
                f"Need at least {self.n_splits + 1} distinct weeks, got {len(unique_weeks)}."
            )

        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        splits: list[tuple[np.ndarray, np.ndarray]] = []

        for fold_idx, (tr_week_idx, va_week_idx) in enumerate(tscv.split(unique_weeks)):
            tr_weeks = set(unique_weeks[tr_week_idx])
            va_weeks = set(unique_weeks[va_week_idx])

            assert tr_weeks.isdisjoint(va_weeks), (
                f"Fold {fold_idx}: Overlapping weeks found between train and validation."
            )
            assert max(tr_weeks) < min(va_weeks), (
                f"Fold {fold_idx}: Train weeks must strictly precede validation weeks."
            )

            tr_mask = week_series.isin(tr_weeks).to_numpy()
            va_mask = week_series.isin(va_weeks).to_numpy()

            train_indices = np.where(tr_mask)[0]
            val_indices = np.where(va_mask)[0]

            splits.append((train_indices, val_indices))

        return splits


def sample_params(
    name: str,
    target_mode: str,
    rng: np.random.RandomState,
) -> tuple[dict[str, Any], str | None]:
    """Sample hyperparameters for a given model family and target mode.

    Parameters
    ----------
    name:
        Model family name: 'ridge', 'random_forest', or 'lightgbm'.
    target_mode:
        'raw' or 'scale_aware'.
    rng:
        Random state instance for reproducibility.

    Returns
    -------
    tuple[dict[str, Any], str | None]
        (sampled_params, scale_col) where scale_col is None for raw target mode.
    """
    scale_col: str | None = None
    if target_mode == "scale_aware":
        scale_col = str(rng.choice(["roll_mean_4", "roll_mean_8"]))

    if name == "ridge":
        # log 1e-3 to 1e3
        log_alpha = rng.uniform(-3.0, 3.0)
        params: dict[str, Any] = {"alpha": float(10.0**log_alpha)}
        return params, scale_col

    if name == "random_forest":
        # n_estimators 100 to 500
        n_estimators = int(rng.randint(100, 501))
        # max_depth 3 to 20 or None
        depth_choices: list[int | None] = list(range(3, 21)) + [None]
        max_depth = depth_choices[rng.randint(0, len(depth_choices))]
        # min_samples_leaf 1 to 20
        min_samples_leaf = int(rng.randint(1, 21))

        params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_leaf": min_samples_leaf,
        }
        return params, scale_col

    if name == "lightgbm":
        # num_leaves 8 to 64
        num_leaves = int(rng.randint(8, 65))
        # learning_rate log 0.01 to 0.2
        log_lr = rng.uniform(np.log10(0.01), np.log10(0.2))
        learning_rate = float(10.0**log_lr)
        # n_estimators 100 to 600
        n_estimators = int(rng.randint(100, 601))
        # min_child_samples 5 to 80
        min_child_samples = int(rng.randint(5, 81))
        # subsample 0.6 to 1.0
        subsample = float(rng.uniform(0.6, 1.0))
        # colsample_bytree 0.6 to 1.0
        colsample_bytree = float(rng.uniform(0.6, 1.0))
        # reg_lambda log 1e-3 to 10
        log_reg = rng.uniform(-3.0, 1.0)
        reg_lambda = float(10.0**log_reg)

        params = {
            "num_leaves": num_leaves,
            "learning_rate": learning_rate,
            "n_estimators": n_estimators,
            "min_child_samples": min_child_samples,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "reg_lambda": reg_lambda,
        }
        if subsample < 1.0:
            params["subsample_freq"] = 1

        return params, scale_col

    raise ValueError(f"Unknown model name '{name}'. Expected ridge, random_forest, or lightgbm.")


def evaluate_trial_cv(
    name: str,
    train_df: pd.DataFrame,
    features: Sequence[str],
    params: dict[str, Any],
    target_mode: str,
    scale_col: str | None,
    splits: list[tuple[np.ndarray, np.ndarray]],
    target_col: str = "target",
) -> tuple[list[float], float, float, pd.DataFrame]:
    """Evaluate one parameter configuration across precomputed expanding-window splits.

    Returns
    -------
    tuple[list[float], float, float, pd.DataFrame]
        (fold_wapes, mean_val_wape, val_bias, oof_predictions_df)
    """
    feature_cols = list(features)
    fold_wapes: list[float] = []
    oof_rows: list[pd.DataFrame] = []

    for fold_train_idx, fold_val_idx in splits:
        tr_slice = train_df.iloc[fold_train_idx]
        va_slice = train_df.iloc[fold_val_idx]

        X_tr = tr_slice[feature_cols]
        y_tr_actual = tr_slice[target_col].to_numpy(dtype=np.float64)

        X_va = va_slice[feature_cols]
        y_va_actual = va_slice[target_col].to_numpy(dtype=np.float64)

        model = make_model(name, params=params)

        if target_mode == "scale_aware":
            if not scale_col or scale_col not in tr_slice.columns:
                raise ValueError(f"Scale column '{scale_col}' missing from data.")

            s_tr = np.maximum(1.0, tr_slice[scale_col].to_numpy(dtype=np.float64))
            y_tr_scaled = y_tr_actual / s_tr
            model.fit(X_tr, y_tr_scaled)

            s_va = np.maximum(1.0, va_slice[scale_col].to_numpy(dtype=np.float64))
            y_pred_scaled = np.asarray(model.predict(X_va), dtype=np.float64)
            y_pred = np.clip(y_pred_scaled * s_va, 0.0, None)
        else:
            model.fit(X_tr, y_tr_actual)
            y_pred = np.clip(np.asarray(model.predict(X_va), dtype=np.float64), 0.0, None)

        fold_metrics = evaluate(y_va_actual, y_pred)
        fold_wapes.append(float(fold_metrics["WAPE"]))

        oof_df = pd.DataFrame(
            {
                "product_id": va_slice["product_id"].to_numpy(),
                "week_start": va_slice["week_start"].to_numpy(),
                "actual": y_va_actual,
                "predicted": y_pred,
            }
        )
        oof_rows.append(oof_df)

    mean_val_wape = float(np.mean(fold_wapes))
    all_oof = pd.concat(oof_rows, ignore_index=True)
    overall_metrics = evaluate(all_oof["actual"], all_oof["predicted"])
    val_bias = float(overall_metrics["bias"])

    return fold_wapes, mean_val_wape, val_bias, all_oof


def tune(
    name: str,
    train: pd.DataFrame,
    features: Sequence[str] | None = None,
    target_mode: str = "raw",
    n_trials: int = 30,
    seed: int = RANDOM_SEED,
    n_splits: int = 5,
) -> dict[str, Any]:
    """Perform random search cross-validation on the training set.

    Parameters
    ----------
    name:
        Model family name in {'ridge', 'random_forest', 'lightgbm'}.
    train:
        Training DataFrame (must NOT contain any test rows).
    features:
        Feature column names. Defaults to FEATURE_COLUMNS (base 15 features).
    target_mode:
        'raw' or 'scale_aware'.
    n_trials:
        Number of random search trials (default 30).
    seed:
        Random seed for parameter generation.
    n_splits:
        Number of expanding-window folds.

    Returns
    -------
    dict[str, Any]
        Dictionary with keys:
        - 'candidate': str
        - 'model_name': str
        - 'target_mode': str
        - 'best_params': dict[str, Any]
        - 'best_scale_col': str | None
        - 'best_fold_scores': list[float]
        - 'best_mean_val_wape': float
        - 'best_val_bias': float
        - 'best_oof_preds': pd.DataFrame
        - 'trials_records': list[dict[str, Any]]
        - 'time_taken_s': float
    """
    feature_cols = list(FEATURE_COLUMNS) if features is None else list(features)
    splitter = WeekTimeSeriesSplit(n_splits=n_splits)
    splits = splitter.split(train, week_col="week_start")

    rng = np.random.RandomState(seed)
    candidate_label = f"{name}_{target_mode}"

    trials_records: list[dict[str, Any]] = []
    best_mean_wape = float("inf")
    best_trial_idx = -1
    best_trial_data: tuple[dict[str, Any], str | None, list[float], float, float, pd.DataFrame] | None = None

    start_time = time.time()

    for trial_no in range(1, n_trials + 1):
        params, scale_col = sample_params(name=name, target_mode=target_mode, rng=rng)
        fold_scores, mean_wape, bias, oof_preds = evaluate_trial_cv(
            name=name,
            train_df=train,
            features=feature_cols,
            params=params,
            target_mode=target_mode,
            scale_col=scale_col,
            splits=splits,
        )

        full_params = dict(params)
        if scale_col is not None:
            full_params["scale_col"] = scale_col

        trial_record = {
            "candidate": candidate_label,
            "trial_no": trial_no,
            "params": json.dumps(full_params, sort_keys=True),
            "mean_val_wape": round(mean_wape, 6),
            "is_best": 0,
        }
        trials_records.append(trial_record)

        if mean_wape < best_mean_wape:
            best_mean_wape = mean_wape
            best_trial_idx = trial_no - 1
            best_trial_data = (params, scale_col, fold_scores, mean_wape, bias, oof_preds)

    time_taken = time.time() - start_time

    assert best_trial_data is not None
    trials_records[best_trial_idx]["is_best"] = 1

    best_params, best_scale_col, best_folds, best_wape, best_bias, best_oof = best_trial_data

    trials_cols = ["candidate", "trial_no", "params", "mean_val_wape", "is_best"]
    trials_df = pd.DataFrame(trials_records)[trials_cols]

    return {
        "candidate": candidate_label,
        "model_name": name,
        "target_mode": target_mode,
        "best_params": best_params,
        "best_scale_col": best_scale_col,
        "best_fold_scores": best_folds,
        "best_mean_val_wape": best_wape,
        "best_val_bias": best_bias,
        "best_oof_preds": best_oof,
        "trials_records": trials_records,
        "trials_df": trials_df,
        "time_taken_s": time_taken,
    }


def evaluate_baselines_cv(
    train_df: pd.DataFrame,
    n_splits: int = 5,
    target_col: str = "target",
) -> dict[str, dict[str, float]]:
    """Evaluate baseline predictors on the exact same expanding-window validation folds."""
    splitter = WeekTimeSeriesSplit(n_splits=n_splits)
    splits = splitter.split(train_df, week_col="week_start")

    predictors = {
        "naive": predict_naive,
        "moving_avg_4": predict_moving_avg_4,
    }

    baseline_results: dict[str, dict[str, float]] = {}

    for b_name, b_fn in predictors.items():
        fold_wapes: list[float] = []
        oof_rows: list[pd.DataFrame] = []

        for _, va_idx in splits:
            va_slice = train_df.iloc[va_idx]
            y_va_actual = va_slice[target_col].to_numpy(dtype=np.float64)
            y_pred = b_fn(va_slice)

            fold_m = evaluate(y_va_actual, y_pred)
            fold_wapes.append(float(fold_m["WAPE"]))

            oof_rows.append(
                pd.DataFrame(
                    {
                        "product_id": va_slice["product_id"].to_numpy(),
                        "week_start": va_slice["week_start"].to_numpy(),
                        "actual": y_va_actual,
                        "predicted": y_pred,
                    }
                )
            )

        all_oof = pd.concat(oof_rows, ignore_index=True)
        overall = evaluate(all_oof["actual"], all_oof["predicted"])
        baseline_results[b_name] = {
            "mean_val_wape": float(np.mean(fold_wapes)),
            "pooled_val_wape": float(overall["WAPE"]),
            "val_bias": float(overall["bias"]),
            "MAE": float(overall["MAE"]),
            "RMSE": float(overall["RMSE"]),
        }

    return baseline_results


def run_tuning_suite(
    train_df: pd.DataFrame,
    features: Sequence[str] | None = None,
    n_trials: int = 30,
    seed: int = RANDOM_SEED,
    models_dir: Path = MODELS_DIR,
    data_dir: Path = DATA_PROCESSED_DIR,
) -> dict[str, Any]:
    """Execute complete M7 tuning suite across 6 candidates and baselines."""
    models_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    candidates_spec = [
        ("ridge", "raw"),
        ("ridge", "scale_aware"),
        ("random_forest", "raw"),
        ("random_forest", "scale_aware"),
        ("lightgbm", "raw"),
        ("lightgbm", "scale_aware"),
    ]

    all_trials_records: list[dict[str, Any]] = []
    candidate_summaries: list[dict[str, Any]] = []
    tuned_params_dict: dict[str, Any] = {}
    oof_frames: list[pd.DataFrame] = []

    print("\n" + "=" * 90)
    print(f"TASK M7: HYPERPARAMETER TUNING & SCALE-AWARE CANDIDATES (Trials per candidate={n_trials})")
    print("=" * 90)

    # 1. Evaluate baselines on validation folds
    print("\nEvaluating baselines on expanding-window validation folds...")
    baseline_metrics = evaluate_baselines_cv(train_df, n_splits=5)
    ma4_wape = baseline_metrics["moving_avg_4"]["mean_val_wape"]
    ma4_bias = baseline_metrics["moving_avg_4"]["val_bias"]
    print(f"Moving Average 4 Validation: Mean WAPE={ma4_wape:.4f}, Bias={ma4_bias:.4f}")
    naive_wape = baseline_metrics["naive"]["mean_val_wape"]
    print(f"Naive Baseline Validation:   Mean WAPE={naive_wape:.4f}, Bias={baseline_metrics['naive']['val_bias']:.4f}")

    # 2. Tune each candidate
    for model_name, target_mode in candidates_spec:
        cand_key = f"{model_name}_{target_mode}"
        print(f"\nTuning candidate '{cand_key}' ({n_trials} trials, 5 folds)...")
        res = tune(
            name=model_name,
            train=train_df,
            features=features,
            target_mode=target_mode,
            n_trials=n_trials,
            seed=seed,
            n_splits=5,
        )

        all_trials_records.extend(res["trials_records"])

        summary = {
            "candidate": cand_key,
            "model_name": model_name,
            "target_mode": target_mode,
            "mean_val_wape": res["best_mean_val_wape"],
            "val_bias": res["best_val_bias"],
            "time_taken_s": res["time_taken_s"],
            "best_params": res["best_params"],
            "best_scale_col": res["best_scale_col"],
        }
        candidate_summaries.append(summary)

        tuned_params_dict[cand_key] = {
            "model_name": model_name,
            "target_mode": target_mode,
            "scale_col": res["best_scale_col"],
            "params": res["best_params"],
            "fold_scores": res["best_fold_scores"],
            "mean_val_wape": res["best_mean_val_wape"],
            "val_bias": res["best_val_bias"],
            "time_taken_s": round(res["time_taken_s"], 2),
        }

        cand_oof = res["best_oof_preds"].copy()
        cand_oof["candidate"] = cand_key
        oof_frames.append(cand_oof)

        print(
            f"  -> Best Mean Val WAPE: {res['best_mean_val_wape']:.4f} | "
            f"Val Bias: {res['best_val_bias']:.4f} | Time: {res['time_taken_s']:.1f}s"
        )
        if res["best_scale_col"]:
            print(f"     Scale column chosen: {res['best_scale_col']}")

    # 3. Overall best candidate
    sorted_candidates = sorted(candidate_summaries, key=lambda x: (x["mean_val_wape"], abs(x["val_bias"])))
    best_candidate = sorted_candidates[0]

    tuned_params_payload = {
        "best_candidate": best_candidate["candidate"],
        "moving_avg_4_baseline": {
            "mean_val_wape": ma4_wape,
            "val_bias": ma4_bias,
        },
        "naive_baseline": {
            "mean_val_wape": naive_wape,
            "val_bias": baseline_metrics["naive"]["val_bias"],
        },
        "candidates": tuned_params_dict,
    }

    # 4. Save artifacts
    params_path = models_dir / "tuned_params.json"
    with open(params_path, "w", encoding="utf-8") as f:
        json.dump(tuned_params_payload, f, indent=2)
    print(f"\nSaved best parameters to {params_path}")

    trials_df = pd.DataFrame(all_trials_records)
    trials_csv_path = models_dir / "tuning_trials.csv"
    trials_df.to_csv(trials_csv_path, index=False)
    # Also save to data_dir for load_mysql compatibility
    trials_df.to_csv(data_dir / "tuning_trials.csv", index=False)
    print(f"Saved {len(trials_df)} tuning trials to {trials_csv_path} and {data_dir / 'tuning_trials.csv'}")

    all_oof_df = pd.concat(oof_frames, ignore_index=True)
    oof_path = models_dir / "oof_val_predictions.csv"
    all_oof_df.to_csv(oof_path, index=False)
    print(f"Saved {len(all_oof_df)} out-of-fold validation predictions to {oof_path}")

    # 5. Print candidate comparison table
    print("\n" + "=" * 90)
    print("M7 VALIDATION SUMMARY TABLE (Expanding-Window 5-Fold Cross-Validation)")
    print("=" * 90)

    display_rows: list[dict[str, Any]] = [
        {
            "candidate": "moving_avg_4 (baseline)",
            "mean_val_wape": ma4_wape,
            "val_bias": ma4_bias,
            "best_parameters": "roll_mean_4",
            "time_taken": "-",
        },
        {
            "candidate": "naive (baseline)",
            "mean_val_wape": naive_wape,
            "val_bias": baseline_metrics["naive"]["val_bias"],
            "best_parameters": "lag_1",
            "time_taken": "-",
        },
    ]

    for c in sorted_candidates:
        p_str = json.dumps(c["best_params"], separators=(",", ":"))
        if c["best_scale_col"]:
            p_str = f"scale={c['best_scale_col']}," + p_str
        if len(p_str) > 42:
            p_str = p_str[:39] + "..."
        display_rows.append(
            {
                "candidate": c["candidate"],
                "mean_val_wape": c["mean_val_wape"],
                "val_bias": c["val_bias"],
                "best_parameters": p_str,
                "time_taken": f"{c['time_taken_s']:.1f}s",
            }
        )

    summary_df = pd.DataFrame(display_rows)
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # 6. Plain language verdict
    print("\n" + "=" * 90)
    print("VERDICT ON VALIDATION:")
    beats_ma = [c for c in candidate_summaries if c["mean_val_wape"] < ma4_wape]
    if beats_ma:
        best_beat = min(beats_ma, key=lambda x: x["mean_val_wape"])
        diff = ma4_wape - best_beat["mean_val_wape"]
        print(
            f"Candidate '{best_beat['candidate']}' is slightly better on validation, within noise of the baseline:\n"
            f"  - Candidate Mean Val WAPE: {best_beat['mean_val_wape']:.4f} (Bias: {best_beat['val_bias']:+.2f})\n"
            f"  - Moving Avg Mean Val WAPE: {ma4_wape:.4f} (Bias: {ma4_bias:+.2f})\n"
            f"  - WAPE Margin:             {diff:.4f} ({diff / ma4_wape * 100:.2f}% relative reduction)\n"
            f"  - Note: Validation weeks span quieter months; the baseline may still win on test."
        )
    else:
        best_overall = min(candidate_summaries, key=lambda x: x["mean_val_wape"])
        print(
            f"NO: None of the tuned candidates beat the 4-week moving average on VALIDATION.\n"
            f"  - Best Candidate ('{best_overall['candidate']}'): {best_overall['mean_val_wape']:.4f}\n"
            f"  - Moving Avg Baseline:                         {ma4_wape:.4f}\n"
            f"  - The moving average baseline remains ahead on validation."
        )
    print("=" * 90 + "\n")

    return {
        "candidate_summaries": candidate_summaries,
        "baseline_metrics": baseline_metrics,
        "best_candidate": best_candidate,
        "tuned_params": tuned_params_payload,
    }


def main() -> None:
    """Load training data, extract features for chosen groups, and run tuning suite."""
    weekly = pd.read_parquet(DATA_PROCESSED_DIR / "weekly_demand.parquet")
    closed_weeks = pd.read_csv(DATA_PROCESSED_DIR / "closed_weeks.csv")

    chosen_path = MODELS_DIR / "chosen_groups.json"
    groups: tuple[str, ...] = ("base",)
    features_to_tune: list[str] = list(FEATURE_COLUMNS)

    if chosen_path.exists():
        with open(chosen_path, encoding="utf-8") as f:
            chosen_data = json.load(f)
        raw_groups = chosen_data.get("chosen_groups", ["base"])
        groups = tuple(raw_groups)
        chosen_features = chosen_data.get("chosen_feature_list")
        if chosen_features:
            features_to_tune = list(chosen_features)
        print(f"Loaded chosen groups from {chosen_path}: {groups}")
        print(f"Features for tuning ({len(features_to_tune)}): {features_to_tune}")

    features_df = build_features(weekly, closed_weeks=closed_weeks, groups=groups)

    # Strictly training split - test set is NOT touched!
    train_df, _ = split_chronological(features_df, test_start=DEFAULT_TEST_START)

    run_tuning_suite(
        train_df=train_df,
        features=features_to_tune,
        n_trials=30,
        seed=RANDOM_SEED,
    )


if __name__ == "__main__":
    main()
