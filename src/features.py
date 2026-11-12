"""Feature engineering for weekly product demand forecasting."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd

_LAG_HORIZONS: tuple[int, ...] = (1, 2, 3, 4)
_ROLL_WINDOWS: tuple[int, ...] = (4, 8)
_ROLL_MIN_PERIODS: int = 2
_LONGEST_WINDOW: int = max(*_LAG_HORIZONS, *_ROLL_WINDOWS)
_CHRISTMAS_DATE: pd.Timestamp = pd.Timestamp("2011-12-25")

FEATURE_COLUMNS: list[str] = [
    *(f"lag_{k}" for k in _LAG_HORIZONS),
    "roll_mean_4",
    "roll_std_4",
    "roll_mean_8",
    "roll_std_8",
    "week_of_year",
    "month",
    "prev_avg_price",
    "prev_orders",
    "prev_customers",
    "prod_past_mean",
    "prod_past_std",
]

OPTIONAL_FEATURES: list[str] = ["weeks_to_christmas"]


def _normalize_closed_weeks(
    closed_weeks: pd.DataFrame | Iterable[Any] | None,
) -> set[pd.Timestamp]:
    """Normalize *closed_weeks* into a set of midnight ``pd.Timestamp`` values."""
    if closed_weeks is None:
        return set()

    if isinstance(closed_weeks, pd.DataFrame):
        if closed_weeks.empty:
            return set()
        col = "week_start" if "week_start" in closed_weeks.columns else closed_weeks.columns[0]
        raw_values = closed_weeks[col]
    else:
        raw_values = list(closed_weeks)
        if not raw_values:
            return set()

    parsed = pd.to_datetime(raw_values).dropna()
    return {pd.Timestamp(ts).normalize() for ts in parsed}


def _compute_feature_frame(
    weekly: pd.DataFrame,
    closed_set: set[pd.Timestamp],
    include_optional: bool = False,
) -> tuple[pd.DataFrame, list[str]]:
    """Compute all demand features on non-closed rows without dropping history."""
    df = weekly.copy()
    df["week_start"] = pd.to_datetime(df["week_start"]).dt.normalize()
    df = df.sort_values(["product_id", "week_start"], kind="mergesort").reset_index(drop=True)
    df["units"] = df["units"].astype("float64")

    if closed_set:
        is_closed = df["week_start"].isin(closed_set)
        df.loc[is_closed, "units"] = np.nan

    # Drop closed/missing-unit rows before shifting/rolling so closed weeks
    # never appear as target rows and lags/rolling skip over them.
    active = df.loc[df["units"].notna()].copy()

    grouped_units = active.groupby("product_id", sort=False)["units"]
    for k in _LAG_HORIZONS:
        active[f"lag_{k}"] = grouped_units.shift(k)

    shifted = grouped_units.shift(1)
    for window in _ROLL_WINDOWS:
        roller = shifted.groupby(active["product_id"], sort=False).rolling(
            window=window,
            min_periods=_ROLL_MIN_PERIODS,
        )
        active[f"roll_mean_{window}"] = roller.mean().reset_index(level=0, drop=True)
        active[f"roll_std_{window}"] = roller.std().reset_index(level=0, drop=True)

    # Calendar features (known in advance from target week_start).
    active["week_of_year"] = active["week_start"].dt.isocalendar().week.astype("int64")
    active["month"] = active["week_start"].dt.month.astype("int64")

    # Previous-week commercial features (shifted on active rows, skipping closed weeks).
    active["prev_avg_price"] = active.groupby("product_id", sort=False)["avg_price"].shift(1)
    active["prev_orders"] = active.groupby("product_id", sort=False)["orders"].shift(1)
    active["prev_customers"] = active.groupby("product_id", sort=False)["customers"].shift(1)

    # Product-level expanding history strictly before target week t.
    expanding_units = shifted.groupby(active["product_id"], sort=False).expanding(
        min_periods=_ROLL_MIN_PERIODS,
    )
    active["prod_past_mean"] = expanding_units.mean().reset_index(level=0, drop=True)
    active["prod_past_std"] = expanding_units.std().reset_index(level=0, drop=True)

    feature_cols = list(FEATURE_COLUMNS)
    if include_optional:
        active["weeks_to_christmas"] = (_CHRISTMAS_DATE - active["week_start"]).dt.days / 7.0
        feature_cols.extend(OPTIONAL_FEATURES)

    return active, feature_cols


def build_features(
    weekly: pd.DataFrame,
    closed_weeks: pd.DataFrame | Iterable[Any] | None = None,
    include_optional: bool = False,
) -> pd.DataFrame:
    """Build demand features per product and target week.

    Closed weeks have ``units`` masked to ``NaN`` before feature calculation,
    are excluded as target rows, and are skipped over by lags, rolling
    windows, and previous-week commercial features so prior non-missing weeks
    are used instead of treating closures as zero demand. Rows lacking
    ``_LONGEST_WINDOW`` (8) prior non-closed weeks of history are dropped, and
    dropped-row counts are printed.

    Parameters
    ----------
    weekly:
        Weekly demand table (contract C1) with ``product_id``, ``week_start``,
        ``units``, ``orders``, ``customers``, and ``avg_price``.
    closed_weeks:
        Shop-closed weeks to treat as missing (DataFrame with ``week_start``
        column, iterable of date-like values, or ``None``).
    include_optional:
        Whether to append ``OPTIONAL_FEATURES`` (e.g. ``weeks_to_christmas``)
        for ablation experiments.

    Returns
    -------
    pd.DataFrame
        One row per ``(product_id, week_start)`` target week with columns:
        ``product_id``, ``week_start``, ``target``, and the feature columns.
    """
    closed_set = _normalize_closed_weeks(closed_weeks)
    active, feature_cols = _compute_feature_frame(
        weekly=weekly,
        closed_set=closed_set,
        include_optional=include_optional,
    )
    active = active.rename(columns={"units": "target"})

    history_len = active.groupby("product_id", sort=False).cumcount()
    has_full_history = (history_len >= _LONGEST_WINDOW) & active[feature_cols].notna().all(axis=1)

    dropped_per_product = (
        active.loc[~has_full_history]
        .groupby("product_id", sort=True)
        .size()
    )
    total_dropped = int((~has_full_history).sum())
    print(f"Dropped rows per product (insufficient history for window={_LONGEST_WINDOW}):")
    if dropped_per_product.empty:
        print("  (none)")
    else:
        for pid, count in dropped_per_product.items():
            print(f"  {pid}: {count}")
    print(f"Total rows dropped: {total_dropped}")

    output_cols = ["product_id", "week_start", "target", *feature_cols]
    return active.loc[has_full_history, output_cols].reset_index(drop=True)


def build_future_features(
    weekly: pd.DataFrame,
    closed_weeks: pd.DataFrame | Iterable[Any] | None = None,
    target_week: str | pd.Timestamp = "2011-12-05",
    include_optional: bool = False,
) -> pd.DataFrame:
    """Build feature rows for a future *target_week* with no target column.

    Only observations strictly before *target_week* are used, guaranteeing
    no leakage from *target_week* or later. Returns one row per product
    present in the historical window.

    Parameters
    ----------
    weekly:
        Weekly demand table (contract C1).
    closed_weeks:
        Shop-closed weeks to treat as missing.
    target_week:
        Future Monday week_start to build features for (default ``2011-12-05``).
    include_optional:
        Whether to append ``OPTIONAL_FEATURES``.

    Returns
    -------
    pd.DataFrame
        One row per ``product_id`` with columns ``product_id``, ``week_start``,
        and the feature columns (no ``target`` column).
    """
    target_ts = pd.Timestamp(target_week).normalize()
    closed_set = _normalize_closed_weeks(closed_weeks) - {target_ts}

    hist = weekly.copy()
    hist["week_start"] = pd.to_datetime(hist["week_start"]).dt.normalize()
    hist = hist.loc[hist["week_start"] < target_ts].copy()

    product_ids = sorted(hist["product_id"].unique())
    future_rows = pd.DataFrame(
        {
            "product_id": product_ids,
            "week_start": target_ts,
            "units": 0.0,
            "orders": np.nan,
            "customers": np.nan,
            "avg_price": np.nan,
        }
    )
    combined = pd.concat([hist, future_rows], ignore_index=True, sort=False)
    active, feature_cols = _compute_feature_frame(
        weekly=combined,
        closed_set=closed_set,
        include_optional=include_optional,
    )

    output_cols = ["product_id", "week_start", *feature_cols]
    return (
        active.loc[active["week_start"] == target_ts, output_cols]
        .sort_values("product_id", kind="mergesort")
        .reset_index(drop=True)
    )
