"""Unit tests for src/error_analysis.py."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.error_analysis import ERROR_ANALYSIS_COLUMNS, build_error_analysis


def _make_sample_predictions_and_segments() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create a small C2-shaped predictions DataFrame (with val and test) and segments."""
    preds = pd.DataFrame(
        {
            "product_id": [
                "P1", "P1", "P2", "P2", "P3", "P3",  # model A test
                "P1", "P1", "P2", "P2", "P3", "P3",  # model B test
                "P1", "P2",                           # val rows (must be excluded)
            ],
            "week_start": [
                "2011-10-10", "2011-10-17", "2011-10-10", "2011-10-17", "2011-10-10", "2011-10-17",
                "2011-10-10", "2011-10-17", "2011-10-10", "2011-10-17", "2011-10-10", "2011-10-17",
                "2011-10-03", "2011-10-03",
            ],
            "actual": [
                100.0, 120.0, 50.0, 40.0, 200.0, 180.0,
                100.0, 120.0, 50.0, 40.0, 200.0, 180.0,
                90.0, 45.0,
            ],
            "predicted": [
                110.0, 100.0, 60.0, 30.0, 190.0, 200.0,
                95.0, 115.0, 55.0, 45.0, 210.0, 170.0,
                500.0, 500.0,  # extreme val errors that must NOT affect test metrics
            ],
            "model_name": [
                "naive", "naive", "naive", "naive", "naive", "naive",
                "lightgbm", "lightgbm", "lightgbm", "lightgbm", "lightgbm", "lightgbm",
                "naive", "lightgbm",
            ],
            "split": ["test"] * 12 + ["val", "val"],
        }
    )
    segs = pd.DataFrame(
        {
            "product_id": ["P1", "P2", "P3"],
            "segment": ["medium", "low", "high"],
        }
    )
    return preds, segs


def test_error_analysis_schema_and_csv_export(tmp_path: Path) -> None:
    """Output matches required columns, levels, models, and excludes validation rows."""
    preds, segs = _make_sample_predictions_and_segments()
    out_csv = tmp_path / "error_analysis.csv"

    ea = build_error_analysis(preds, segs, out_path=out_csv)
    assert out_csv.exists()
    reloaded = pd.read_csv(out_csv)

    assert list(ea.columns) == ERROR_ANALYSIS_COLUMNS
    assert list(reloaded.columns) == ERROR_ANALYSIS_COLUMNS
    assert not ea.isna().any().any()

    assert set(ea["level"]) == {"segment", "week", "product"}
    assert set(ea["model_name"]) == {"naive", "lightgbm"}
    # 3 segments + 2 weeks + 3 products = 8 keys per model * 2 models = 16 rows
    assert len(ea) == 16
    assert set(ea.loc[ea["level"] == "week", "key"]) == {"2011-10-10", "2011-10-17"}
    assert "2011-10-03" not in set(ea["key"])


def test_error_analysis_reproduces_segment_metrics_from_predictions() -> None:
    """Segment-level metrics reproduce directly from split=='test' predictions."""
    preds, segs = _make_sample_predictions_and_segments()
    ea = build_error_analysis(preds, segs, out_path=None)

    test_merged = preds[preds["split"] == "test"].merge(segs, on="product_id", how="left")
    seg_rows = ea[ea["level"] == "segment"]

    for _, row in seg_rows.iterrows():
        slice_df = test_merged[
            (test_merged["segment"] == row["key"])
            & (test_merged["model_name"] == row["model_name"])
        ]
        err = slice_df["predicted"] - slice_df["actual"]
        assert row["n"] == len(slice_df)
        assert row["MAE"] == pytest.approx(float(err.abs().mean()))
        assert row["RMSE"] == pytest.approx(float(np.sqrt((err**2).mean())))
        assert row["WAPE"] == pytest.approx(float(err.abs().sum() / slice_df["actual"].sum()))
        assert row["bias"] == pytest.approx(float(err.mean()))


def test_error_analysis_reproduces_mock_predictions_segments() -> None:
    """Works end-to-end on data/mock/predictions.csv with training-derived mock tertiles."""
    mock_preds = pd.read_csv(
        "data/mock/predictions.csv",
        dtype={"product_id": str, "model_name": str, "split": str},
    )
    mock_weekly = pd.read_parquet("data/mock/weekly_demand.parquet")
    train_means = (
        mock_weekly[pd.to_datetime(mock_weekly["week_start"]) < pd.Timestamp("2011-10-10")]
        .groupby("product_id")["units"]
        .mean()
    )
    mock_segs = pd.DataFrame(
        {
            "product_id": train_means.index.astype(str),
            "segment": pd.qcut(train_means, q=3, labels=["low", "medium", "high"]).astype(str),
        }
    )

    ea = build_error_analysis(mock_preds, mock_segs, out_path=None)
    assert set(ea.loc[ea["level"] == "segment", "key"]) == {"low", "medium", "high"}
    assert len(ea.loc[ea["level"] == "week", "key"].unique()) == 8
    assert len(ea.loc[ea["level"] == "product", "key"].unique()) == 10


def test_error_analysis_over_plus_under_equals_total_abs_error() -> None:
    """Over-forecast + under-forecast units equal total absolute error for every row."""
    preds, segs = _make_sample_predictions_and_segments()
    ea = build_error_analysis(preds, segs, out_path=None)

    assert (ea["over_forecast_units"] >= 0).all()
    assert (ea["under_forecast_units"] >= 0).all()
    total_abs = ea["MAE"] * ea["n"]
    np.testing.assert_allclose(
        ea["over_forecast_units"] + ea["under_forecast_units"],
        total_abs,
        rtol=1e-9,
        atol=1e-9,
    )


def test_error_analysis_validation_errors() -> None:
    """Missing test split or missing segment mapping raises ValueError."""
    preds, segs = _make_sample_predictions_and_segments()
    with pytest.raises(ValueError, match="split == 'test'"):
        build_error_analysis(preds[preds["split"] == "val"], segs, out_path=None)

    with pytest.raises(ValueError, match="Missing segment mapping"):
        build_error_analysis(preds, segs[segs["product_id"] != "P1"], out_path=None)

