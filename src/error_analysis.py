"""Forecast-error analysis utilities for Shelfwatch (SRS section 5.3)."""

from __future__ import annotations

import argparse
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

_LEVEL_ORDER: dict[str, int] = {
    "segment": 0,
    "pattern_class": 1,
    "abc_class": 2,
    "week": 3,
    "product": 4,
}
_SEGMENT_ORDER: dict[str, int] = {"low": 0, "medium": 1, "high": 2}
_PATTERN_ORDER: dict[str, int] = {"smooth": 0, "erratic": 1, "intermittent": 2, "lumpy": 3}
_ABC_ORDER: dict[str, int] = {"A": 0, "B": 1, "C": 2}


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
    segments: pd.DataFrame | None = None,
    intelligence: pd.DataFrame | None = None,
    out_path: str | Path | None = "data/processed/error_analysis.csv",
) -> pd.DataFrame:
    """Build tidy test-period error analysis across segment, pattern, ABC, week, and product levels.

    Parameters
    ----------
    predictions:
        Contract C2 predictions table (``product_id``, ``week_start``,
        ``actual``, ``predicted``, ``model_name``, ``split``).
    segments:
        DataFrame with columns ``product_id`` and ``segment`` (``low``,
        ``medium``, ``high``). If None and ``data/processed/product_segments.csv``
        exists, it will be loaded automatically.
    intelligence:
        Optional DataFrame with columns ``product_id``, ``pattern_class``,
        and ``abc_class`` (from contract C11 ``product_intelligence.csv``).
        If None and ``data/processed/product_intelligence.csv`` exists, it will
        be loaded automatically.
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

    # Auto-load defaults only if neither segments nor intelligence was provided
    if segments is None and intelligence is None:
        default_seg = Path("data/processed/product_segments.csv")
        if default_seg.exists():
            segments = pd.read_csv(default_seg, dtype={"product_id": str, "segment": str})
        default_intel = Path("data/processed/product_intelligence.csv")
        if default_intel.exists():
            intelligence = pd.read_csv(default_intel, dtype={"product_id": str})

    merged = test_df.copy()
    all_rows: list[dict[str, object]] = []

    if segments is not None:
        seg_map = segments[["product_id", "segment"]].reset_index(drop=True).copy()
        seg_map["product_id"] = seg_map["product_id"].astype(str)
        seg_map["segment"] = seg_map["segment"].astype(str)
        merged = merged.merge(seg_map, on="product_id", how="left")
        if merged["segment"].isna().any():
            missing = sorted(merged.loc[merged["segment"].isna(), "product_id"].unique().tolist())
            raise ValueError(f"Missing segment mapping for product_id(s): {missing}")
        all_rows.extend(_aggregate_level(merged, "segment", "segment"))

    if intelligence is not None:
        intel_cols = ["product_id"]
        for col in ["pattern_class", "abc_class"]:
            if col in intelligence.columns:
                intel_cols.append(col)
        intel_map = intelligence[intel_cols].reset_index(drop=True).copy()
        intel_map["product_id"] = intel_map["product_id"].astype(str)
        merged = merged.merge(intel_map, on="product_id", how="left")

        if "pattern_class" in intel_cols:
            if merged["pattern_class"].isna().any():
                missing = sorted(merged.loc[merged["pattern_class"].isna(), "product_id"].unique().tolist())
                raise ValueError(f"Missing pattern_class mapping for product_id(s): {missing}")
            all_rows.extend(_aggregate_level(merged, "pattern_class", "pattern_class"))

        if "abc_class" in intel_cols:
            if merged["abc_class"].isna().any():
                missing = sorted(merged.loc[merged["abc_class"].isna(), "product_id"].unique().tolist())
                raise ValueError(f"Missing abc_class mapping for product_id(s): {missing}")
            all_rows.extend(_aggregate_level(merged, "abc_class", "abc_class"))

    all_rows.extend(_aggregate_level(merged, "week", "week_start"))
    all_rows.extend(_aggregate_level(merged, "product", "product_id"))

    result = pd.DataFrame(all_rows, columns=ERROR_ANALYSIS_COLUMNS)
    result["_level_rank"] = result["level"].map(_LEVEL_ORDER).fillna(99)

    seg_rank = np.where(result["level"] == "segment", result["key"].map(_SEGMENT_ORDER).fillna(99), 99)
    pat_rank = np.where(result["level"] == "pattern_class", result["key"].map(_PATTERN_ORDER).fillna(99), 99)
    abc_rank = np.where(result["level"] == "abc_class", result["key"].map(_ABC_ORDER).fillna(99), 99)
    result["_sub_rank"] = np.minimum(np.minimum(seg_rank, pat_rank), abc_rank)

    result = (
        result.sort_values(["_level_rank", "model_name", "_sub_rank", "key"], kind="mergesort")
        .drop(columns=["_level_rank", "_sub_rank"])
        .reset_index(drop=True)
    )

    if out_path is not None:
        dest = Path(out_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(dest, index=False)

    return result


def main() -> None:
    """CLI runner to build error_analysis.csv from processed files."""
    parser = argparse.ArgumentParser(description="Build forecast error analysis table (C18 / error_analysis.csv)")
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("data/processed/predictions.csv"),
        help="Path to predictions.csv (C2)",
    )
    parser.add_argument(
        "--segments",
        type=Path,
        default=Path("data/processed/product_segments.csv"),
        help="Path to product_segments.csv",
    )
    parser.add_argument(
        "--intelligence",
        type=Path,
        default=Path("data/processed/product_intelligence.csv"),
        help="Path to product_intelligence.csv (C11)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/error_analysis.csv"),
        help="Destination path for error_analysis.csv",
    )
    args = parser.parse_args()

    preds = pd.read_csv(args.predictions, dtype={"product_id": str, "model_name": str, "split": str})
    segs = pd.read_csv(args.segments, dtype={"product_id": str, "segment": str}) if args.segments.exists() else None
    intel = pd.read_csv(args.intelligence, dtype={"product_id": str}) if args.intelligence.exists() else None

    df_out = build_error_analysis(preds, segments=segs, intelligence=intel, out_path=args.output)
    print(f"Generated {args.output} with {len(df_out)} rows across levels: {sorted(df_out['level'].unique().tolist())}")


if __name__ == "__main__":
    main()
