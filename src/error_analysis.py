"""Forecast-error analysis utilities for Shelfwatch (SRS section 5.3)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ERROR_ANALYSIS_COLUMNS: list[str] = [
    "level",
    "key",
    "model_name",
    "n",
    "MAE",
    "RMSE",
    "WAPE",
    "bias",
    "over_forecast_units",
    "under_forecast_units",
]

_LEVEL_ORDER: dict[str, int] = {"segment": 0, "week": 1, "product": 2}
_SEGMENT_ORDER: dict[str, int] = {"low": 0, "medium": 1, "high": 2}


def _compute_group_metrics(group: pd.DataFrame) -> dict[str, float | int]:
    """Compute error and over/under-forecast metrics for a single prediction slice."""
    actual = group["actual"].to_numpy(dtype=float)
    predicted = group["predicted"].to_numpy(dtype=float)
    err = predicted - actual
    abs_err = np.abs(err)

    total_actual = float(actual.sum())
    total_abs_err = float(abs_err.sum())
    wape = total_abs_err / total_actual if total_actual > 0.0 else (0.0 if total_abs_err == 0.0 else float("nan"))

    return {
        "n": int(len(group)),
        "MAE": float(abs_err.mean()),
        "RMSE": float(np.sqrt(np.mean(err**2))),
        "WAPE": float(wape),
        "bias": float(err.mean()),
        "over_forecast_units": float(np.clip(err, 0.0, None).sum()),
        "under_forecast_units": float(np.clip(-err, 0.0, None).sum()),
    }


def _aggregate_level(
    df: pd.DataFrame,
    level_name: str,
    key_col: str,
) -> list[dict[str, object]]:
    """Aggregate test-set error metrics by (key_col, model_name) for one level."""
    rows: list[dict[str, object]] = []
    for (key_val, model_name), grp in df.groupby([key_col, "model_name"], sort=True):
        metrics = _compute_group_metrics(grp)
        rows.append(
            {
                "level": level_name,
                "key": str(key_val),
                "model_name": str(model_name),
                **metrics,
            }
        )
    return rows


def build_error_analysis(
    predictions: pd.DataFrame,
    segments: pd.DataFrame,
    out_path: str | Path | None = "data/processed/error_analysis.csv",
) -> pd.DataFrame:
    """Build tidy test-period error analysis across segment, week, and product levels.

    Parameters
    ----------
    predictions:
        Contract C2 predictions table (``product_id``, ``week_start``,
        ``actual``, ``predicted``, ``model_name``, ``split``).
    segments:
        DataFrame with columns ``product_id`` and ``segment`` (``low``,
        ``medium``, ``high``).
    out_path:
        Optional CSV destination. Defaults to
        ``data/processed/error_analysis.csv``; pass ``None`` to skip writing.

    Returns
    -------
    pd.DataFrame with columns :data:`ERROR_ANALYSIS_COLUMNS` computed strictly
    on ``split == "test"`` rows for all models.
    """
    if "split" not in predictions.columns:
        raise ValueError("predictions DataFrame must include a 'split' column")

    test_df = predictions[predictions["split"] == "test"].reset_index(drop=True).copy()
    if test_df.empty:
        raise ValueError("No rows with split == 'test' found in predictions")

    test_df["product_id"] = test_df["product_id"].astype(str)
    test_df["week_start"] = pd.to_datetime(test_df["week_start"]).dt.strftime("%Y-%m-%d")
    test_df["model_name"] = test_df["model_name"].astype(str)

    seg_map = segments[["product_id", "segment"]].reset_index(drop=True).copy()
    seg_map["product_id"] = seg_map["product_id"].astype(str)
    seg_map["segment"] = seg_map["segment"].astype(str)

    merged = test_df.merge(seg_map, on="product_id", how="left")
    if merged["segment"].isna().any():
        missing = sorted(merged.loc[merged["segment"].isna(), "product_id"].unique().tolist())
        raise ValueError(f"Missing segment mapping for product_id(s): {missing}")

    all_rows: list[dict[str, object]] = []
    all_rows.extend(_aggregate_level(merged, "segment", "segment"))
    all_rows.extend(_aggregate_level(merged, "week", "week_start"))
    all_rows.extend(_aggregate_level(merged, "product", "product_id"))

    result = pd.DataFrame(all_rows, columns=ERROR_ANALYSIS_COLUMNS)
    result["_level_rank"] = result["level"].map(_LEVEL_ORDER)
    result["_seg_rank"] = np.where(
        result["level"] == "segment",
        result["key"].map(_SEGMENT_ORDER).fillna(99),
        0,
    )
    result = (
        result.sort_values(["_level_rank", "model_name", "_seg_rank", "key"], kind="mergesort")
        .drop(columns=["_level_rank", "_seg_rank"])
        .reset_index(drop=True)
    )

    if out_path is not None:
        dest = Path(out_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(dest, index=False)

    return result
