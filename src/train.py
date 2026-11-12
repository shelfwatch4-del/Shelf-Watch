"""Chronological splitting, evaluation metrics, and baseline predictors."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from src.features import build_features

DEFAULT_TEST_START: str = "2011-10-10"
EXPECTED_TEST_WEEKS: int = 8
DATA_PROCESSED_DIR: Path = Path("data/processed")


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


def main() -> None:
    """Load processed data, build features, split, and print baseline test metrics."""
    weekly = pd.read_parquet(DATA_PROCESSED_DIR / "weekly_demand.parquet")
    closed_weeks = pd.read_csv(DATA_PROCESSED_DIR / "closed_weeks.csv")

    features_df = build_features(weekly, closed_weeks=closed_weeks)
    _, test_df = split_chronological(features_df, test_start=DEFAULT_TEST_START)

    baseline_table = evaluate_baselines(test_df)
    print("\nBaseline Metrics (Test Set):")
    print(baseline_table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()

