"""Feature engineering for weekly product demand forecasting."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_LAG_HORIZONS: tuple[int, ...] = (1, 2, 3, 4)
_ROLL_WINDOWS: tuple[int, ...] = (4, 8)
_ROLL_MIN_PERIODS: int = 2
_LONGEST_WINDOW: int = max(*_LAG_HORIZONS, *_ROLL_WINDOWS)
_CHRISTMAS_DATE: pd.Timestamp = pd.Timestamp("2011-12-25")

# Default file paths for enrichment sources
DEFAULT_CALENDAR_PATH = Path("data/processed/calendar_weekly.csv")
DEFAULT_PRIOR_YEAR_PATH = Path("data/processed/prior_year_weekly.parquet")
DEFAULT_ENRICHMENT_PATH = Path("data/processed/enrichment_weekly.parquet")

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

FEATURE_GROUPS: dict[str, list[str]] = {
    "base": list(FEATURE_COLUMNS),
    "g1": [
        "order_size_lag1",
        "trend_ratio",
        "price_index",
        "cv_4",
        "zero_streak",
    ],
    "g2": [
        "uk_bank_holiday_count",
        "trading_days_expected",
    ],
    "g3": [
        "yoy_mean_5w",
        "yoy_seasonal_index",
        "yoy_available",
    ],
    "g4": [
        "return_rate_4w",
        "export_share_4w",
        "guest_share_4w",
    ],
}


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


def _get_table_from_enrichment(
    enrichment: dict[str, pd.DataFrame] | None,
    candidate_keys: Sequence[str],
    default_path: Path,
) -> pd.DataFrame | None:
    """Retrieve an enrichment DataFrame from dictionary or fallback to disk."""
    if enrichment is not None:
        for key in candidate_keys:
            if key in enrichment and isinstance(enrichment[key], pd.DataFrame):
                return enrichment[key]
        return None

    if default_path.exists():
        if default_path.suffix.lower() == ".parquet":
            return pd.read_parquet(default_path)
        if default_path.suffix.lower() == ".csv":
            return pd.read_csv(default_path)
    return None


def _calc_zero_streaks(shifted_series: pd.Series) -> list[float]:
    """Calculate consecutive zero weeks strictly before target week."""
    streaks: list[float] = []
    current = 0.0
    for val in shifted_series:
        if pd.isna(val):
            current = 0.0
        elif val == 0:
            current += 1.0
        else:
            current = 0.0
        streaks.append(current)
    return streaks


def _resolve_selected_feature_cols(
    groups: Sequence[str],
    include_optional: bool = False,
) -> list[str]:
    """Return ordered list of feature columns for the selected groups."""
    selected_cols: list[str] = []
    for g in groups:
        if g in FEATURE_GROUPS:
            for col in FEATURE_GROUPS[g]:
                if col not in selected_cols:
                    selected_cols.append(col)
        elif g in FEATURE_COLUMNS:
            if g not in selected_cols:
                selected_cols.append(g)

    if include_optional:
        for opt in OPTIONAL_FEATURES:
            if opt not in selected_cols:
                selected_cols.append(opt)

    return selected_cols


def _compute_feature_frame(
    weekly: pd.DataFrame,
    closed_set: set[pd.Timestamp],
    enrichment: dict[str, pd.DataFrame] | None = None,
    groups: Sequence[str] = ("base",),
    include_optional: bool = False,
) -> tuple[pd.DataFrame, list[str]]:
    """Compute demand features across requested groups without future leakage."""
    df = weekly.copy()
    df["week_start"] = pd.to_datetime(df["week_start"]).dt.normalize()
    df["product_id"] = df["product_id"].astype(str)
    df = df.sort_values(["product_id", "week_start"], kind="mergesort").reset_index(drop=True)
    df["units"] = df["units"].astype("float64")

    if closed_set:
        is_closed = df["week_start"].isin(closed_set)
        df.loc[is_closed, "units"] = np.nan

    # Drop closed/missing-unit rows before shifting/rolling so closed weeks
    # never appear as target rows and lags/rolling skip over them.
    active = df.loc[df["units"].notna()].copy().reset_index(drop=True)

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
    if "avg_price" in active.columns:
        active["prev_avg_price"] = active.groupby("product_id", sort=False)["avg_price"].shift(1)
    else:
        active["prev_avg_price"] = np.nan

    if "orders" in active.columns:
        active["prev_orders"] = active.groupby("product_id", sort=False)["orders"].shift(1)
    else:
        active["prev_orders"] = np.nan

    if "customers" in active.columns:
        active["prev_customers"] = active.groupby("product_id", sort=False)["customers"].shift(1)
    else:
        active["prev_customers"] = np.nan

    # Product-level expanding history strictly before target week t.
    expanding_units = shifted.groupby(active["product_id"], sort=False).expanding(
        min_periods=_ROLL_MIN_PERIODS,
    )
    active["prod_past_mean"] = expanding_units.mean().reset_index(level=0, drop=True)
    active["prod_past_std"] = expanding_units.std().reset_index(level=0, drop=True)

    # Group G1: Derived behaviour features
    needed_groups = set(groups)
    if "g1" in needed_groups:
        # order_size_lag1: units / orders at t-1 (guard zero orders)
        order_size = np.where(
            active["prev_orders"] > 0,
            active["lag_1"] / active["prev_orders"],
            0.0,
        )
        active["order_size_lag1"] = np.where(active["lag_1"].isna(), np.nan, order_size)

        # trend_ratio: roll_mean_4 / (roll_mean_8 + 1e-9)
        active["trend_ratio"] = active["roll_mean_4"] / (active["roll_mean_8"] + 1e-9)

        # price_index: prev_avg_price / mean of the previous 8 weeks' prices
        if "avg_price" in active.columns:
            shifted_price = active.groupby("product_id", sort=False)["avg_price"].shift(1)
            roll_price_8 = (
                shifted_price.groupby(active["product_id"], sort=False)
                .rolling(window=8, min_periods=_ROLL_MIN_PERIODS)
                .mean()
                .reset_index(level=0, drop=True)
            )
            active["price_index"] = active["prev_avg_price"] / (roll_price_8 + 1e-9)
        else:
            active["price_index"] = np.nan

        # cv_4: roll_std_4 / (roll_mean_4 + 1.0)
        active["cv_4"] = active["roll_std_4"] / (active["roll_mean_4"] + 1.0)

        # zero_streak: consecutive zero weeks before t
        zero_streaks = (
            shifted.groupby(active["product_id"], sort=False)
            .transform(_calc_zero_streaks)
        )
        active["zero_streak"] = zero_streaks.astype("float64")

    # Group G2: Calendar features (target week)
    if "g2" in needed_groups:
        cal_df = _get_table_from_enrichment(
            enrichment=enrichment,
            candidate_keys=("calendar", "calendar_weekly"),
            default_path=DEFAULT_CALENDAR_PATH,
        )
        if cal_df is not None and not cal_df.empty:
            cal = cal_df.copy()
            cal["week_start"] = pd.to_datetime(cal["week_start"]).dt.normalize()
            keep_cal_cols = ["week_start", "uk_bank_holiday_count", "trading_days_expected"]
            cal = cal[[c for c in keep_cal_cols if c in cal.columns]].drop_duplicates("week_start")
            active = active.merge(cal, on="week_start", how="left")
            active["uk_bank_holiday_count"] = active["uk_bank_holiday_count"].fillna(0).astype("int64")
            active["trading_days_expected"] = active["trading_days_expected"].fillna(6).astype("int64")
        else:
            active["uk_bank_holiday_count"] = 0
            active["trading_days_expected"] = 6

    # Group G3: Prior-year features (weeks t-54 to t-50)
    if "g3" in needed_groups:
        py_df = _get_table_from_enrichment(
            enrichment=enrichment,
            candidate_keys=("prior_year", "prior_year_weekly"),
            default_path=DEFAULT_PRIOR_YEAR_PATH,
        )
        if py_df is not None and not py_df.empty:
            py = py_df.copy()
            py["week_start"] = pd.to_datetime(py["week_start"]).dt.normalize()
            py["product_id"] = py["product_id"].astype(str)
            py_map = py.set_index(["product_id", "week_start"])["units"].to_dict()
            py_prod_mean = py.groupby("product_id")["units"].mean().to_dict()

            yoy_means: list[float] = []
            yoy_indices: list[float] = []
            yoy_avail: list[float] = []

            for pid, ws in zip(active["product_id"], active["week_start"]):
                # Look up weeks t-54, t-53, t-52, t-51, t-50
                vals = [py_map.get((pid, ws - pd.Timedelta(weeks=w))) for w in (54, 53, 52, 51, 50)]
                valid = [v for v in vals if v is not None and not np.isnan(v)]
                if len(valid) >= 3:
                    m = float(np.mean(valid))
                    pm = py_prod_mean.get(pid, 0.0)
                    idx = (m / pm) if (pm is not None and pm > 0) else np.nan
                    yoy_means.append(m)
                    yoy_indices.append(idx)
                    yoy_avail.append(1.0)
                else:
                    yoy_means.append(np.nan)
                    yoy_indices.append(np.nan)
                    yoy_avail.append(0.0)

            active["yoy_mean_5w"] = yoy_means
            active["yoy_seasonal_index"] = yoy_indices
            active["yoy_available"] = yoy_avail
        else:
            active["yoy_mean_5w"] = np.nan
            active["yoy_seasonal_index"] = np.nan
            active["yoy_available"] = 0.0

    # Group G4: Transaction-derived rolling features (shifted, skipping closures)
    if "g4" in needed_groups:
        enr_df = _get_table_from_enrichment(
            enrichment=enrichment,
            candidate_keys=("enrichment", "enrichment_weekly"),
            default_path=DEFAULT_ENRICHMENT_PATH,
        )
        if enr_df is not None and not enr_df.empty:
            enr = enr_df.copy()
            enr["week_start"] = pd.to_datetime(enr["week_start"]).dt.normalize()
            enr["product_id"] = enr["product_id"].astype(str)
            enr_cols = ["product_id", "week_start", "gross_units", "cancelled_units", "export_units", "guest_units"]
            enr = enr[[c for c in enr_cols if c in enr.columns]].drop_duplicates(["product_id", "week_start"])
            active = active.merge(enr, on=["product_id", "week_start"], how="left")
        else:
            active["gross_units"] = active["units"]
            active["cancelled_units"] = 0.0
            active["export_units"] = 0.0
            active["guest_units"] = 0.0

        active["gross_units"] = active["gross_units"].fillna(active["units"])
        active["cancelled_units"] = active["cancelled_units"].fillna(0.0)
        active["export_units"] = active["export_units"].fillna(0.0)
        active["guest_units"] = active["guest_units"].fillna(0.0)

        grp_enr = active.groupby("product_id", sort=False)
        shifted_gross = grp_enr["gross_units"].shift(1)
        shifted_cancel = grp_enr["cancelled_units"].shift(1)
        shifted_export = grp_enr["export_units"].shift(1)
        shifted_guest = grp_enr["guest_units"].shift(1)

        roll_gross_4 = (
            shifted_gross.groupby(active["product_id"], sort=False)
            .rolling(window=4, min_periods=_ROLL_MIN_PERIODS)
            .sum()
            .reset_index(level=0, drop=True)
        )
        roll_cancel_4 = (
            shifted_cancel.groupby(active["product_id"], sort=False)
            .rolling(window=4, min_periods=_ROLL_MIN_PERIODS)
            .sum()
            .reset_index(level=0, drop=True)
        )
        roll_export_4 = (
            shifted_export.groupby(active["product_id"], sort=False)
            .rolling(window=4, min_periods=_ROLL_MIN_PERIODS)
            .sum()
            .reset_index(level=0, drop=True)
        )
        roll_guest_4 = (
            shifted_guest.groupby(active["product_id"], sort=False)
            .rolling(window=4, min_periods=_ROLL_MIN_PERIODS)
            .sum()
            .reset_index(level=0, drop=True)
        )

        denom = np.maximum(roll_gross_4, 1.0)
        active["return_rate_4w"] = roll_cancel_4 / denom
        active["export_share_4w"] = roll_export_4 / denom
        active["guest_share_4w"] = roll_guest_4 / denom

    if include_optional:
        active["weeks_to_christmas"] = (_CHRISTMAS_DATE - active["week_start"]).dt.days / 7.0

    selected_feature_cols = _resolve_selected_feature_cols(
        groups=groups,
        include_optional=include_optional,
    )
    return active, selected_feature_cols


def build_features(
    weekly: pd.DataFrame,
    closed_weeks: pd.DataFrame | Iterable[Any] | None = None,
    enrichment: dict[str, pd.DataFrame] | None = None,
    groups: tuple[str, ...] | list[str] = ("base",),
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
    enrichment:
        Dictionary holding enrichment DataFrames (keys: ``calendar``,
        ``prior_year``, ``enrichment``). If None, loads defaults from
        ``data/processed/`` when required groups are selected.
    groups:
        Feature groups to build (subset of ``base``, ``g1``, ``g2``, ``g3``, ``g4``).
        Default is ``("base",)``.
    include_optional:
        Whether to append ``OPTIONAL_FEATURES`` (e.g. ``weeks_to_christmas``)
        for ablation experiments.

    Returns
    -------
    pd.DataFrame
        One row per ``(product_id, week_start)`` target week with columns:
        ``product_id``, ``week_start``, ``target``, and the requested feature columns.
    """
    closed_set = _normalize_closed_weeks(closed_weeks)
    active, feature_cols = _compute_feature_frame(
        weekly=weekly,
        closed_set=closed_set,
        enrichment=enrichment,
        groups=groups,
        include_optional=include_optional,
    )
    active = active.rename(columns={"units": "target"})

    # Check history sufficiency based on _LONGEST_WINDOW and base feature availability
    history_len = active.groupby("product_id", sort=False).cumcount()
    base_cols_in_active = [c for c in FEATURE_COLUMNS if c in active.columns]
    has_full_history = history_len >= _LONGEST_WINDOW
    if base_cols_in_active:
        has_full_history = has_full_history & active[base_cols_in_active].notna().all(axis=1)

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
    enrichment: dict[str, pd.DataFrame] | None = None,
    groups: tuple[str, ...] | list[str] = ("base",),
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
    enrichment:
        Optional dictionary holding enrichment DataFrames.
    groups:
        Feature groups to build (default ``("base",)``).
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
        enrichment=enrichment,
        groups=groups,
        include_optional=include_optional,
    )

    output_cols = ["product_id", "week_start", *feature_cols]
    return (
        active.loc[active["week_start"] == target_ts, output_cols]
        .sort_values("product_id", kind="mergesort")
        .reset_index(drop=True)
    )


def build_custom_features(
    series: pd.DataFrame | dict[str, Any] | Sequence[float],
    target_week: str | pd.Timestamp = "2011-12-05",
    enrichment: dict[str, pd.DataFrame] | None = None,
    groups: tuple[str, ...] | list[str] | None = None,
) -> pd.DataFrame:
    """Build features for a user-entered series.

    Builds base, g1, and g2 features exactly like ``build_features``/``build_future_features``.
    Columns for groups g3 and g4 are populated with ``NaN`` because prior-year history
    and transaction logs are not available for custom series.

    Parameters
    ----------
    series:
        User-entered input: DataFrame with columns ``units``, optional ``avg_price``,
        ``orders``, ``customers``; or dictionary with keys ``recent_units``,
        optional ``recent_prices``, etc.; or a sequence of numbers (units).
    target_week:
        Target week Monday date (default ``2011-12-05``).
    enrichment:
        Optional enrichment dictionary for calendar data (g2).
    groups:
        Feature groups to include. If None, includes all groups
        (``base``, ``g1``, ``g2``, ``g3``, ``g4``).

    Returns
    -------
    pd.DataFrame
        Single-row DataFrame for the custom product with target week features.
    """
    target_ts = pd.Timestamp(target_week).normalize()

    # Normalize input series into a standard DataFrame
    if isinstance(series, pd.DataFrame):
        df = series.copy()
        if "units" not in df.columns:
            raise ValueError("Input DataFrame must contain 'units' column.")
        n_obs = len(df)
        if "product_id" not in df.columns:
            df["product_id"] = "CUSTOM"
        if "week_start" not in df.columns:
            df["week_start"] = [target_ts - pd.Timedelta(weeks=n_obs - i) for i in range(n_obs)]
        else:
            df["week_start"] = pd.to_datetime(df["week_start"]).dt.normalize()
        if "orders" not in df.columns:
            df["orders"] = np.where(df["units"] > 0, 1.0, 0.0)
        if "customers" not in df.columns:
            df["customers"] = np.where(df["units"] > 0, 1.0, 0.0)
        if "avg_price" not in df.columns:
            if "price" in df.columns:
                df["avg_price"] = df["price"]
            else:
                df["avg_price"] = 1.0
    elif isinstance(series, dict):
        raw_units = series.get("recent_units", series.get("units"))
        if raw_units is None:
            raise ValueError("Dictionary series must contain 'recent_units' or 'units'.")
        n_obs = len(raw_units)
        raw_prices = series.get("recent_prices", series.get("prices", series.get("avg_price")))
        raw_orders = series.get("recent_orders", series.get("orders"))
        raw_cust = series.get("recent_customers", series.get("customers"))

        df = pd.DataFrame(
            {
                "product_id": "CUSTOM",
                "week_start": [target_ts - pd.Timedelta(weeks=n_obs - i) for i in range(n_obs)],
                "units": [float(x) for x in raw_units],
                "avg_price": [float(p) for p in raw_prices] if raw_prices is not None else [1.0] * n_obs,
                "orders": [float(o) for o in raw_orders] if raw_orders is not None else [1.0 if u > 0 else 0.0 for u in raw_units],
                "customers": [float(c) for c in raw_cust] if raw_cust is not None else [1.0 if u > 0 else 0.0 for u in raw_units],
            }
        )
    elif isinstance(series, (list, tuple, np.ndarray, pd.Series)):
        raw_units = list(series)
        n_obs = len(raw_units)
        df = pd.DataFrame(
            {
                "product_id": "CUSTOM",
                "week_start": [target_ts - pd.Timedelta(weeks=n_obs - i) for i in range(n_obs)],
                "units": [float(x) for x in raw_units],
                "avg_price": [1.0] * n_obs,
                "orders": [1.0 if float(x) > 0 else 0.0 for x in raw_units],
                "customers": [1.0 if float(x) > 0 else 0.0 for x in raw_units],
            }
        )
    else:
        raise TypeError(f"Unsupported series type: {type(series)}")

    # Requested groups
    if groups is None:
        target_groups = ("base", "g1", "g2", "g3", "g4")
    else:
        target_groups = tuple(groups)

    # Compute base, g1, g2 features using build_future_features
    available_groups = [g for g in target_groups if g in ("base", "g1", "g2")]
    if not available_groups:
        available_groups = ["base"]

    custom_future = build_future_features(
        weekly=df,
        closed_weeks=None,
        target_week=target_ts,
        enrichment=enrichment,
        groups=available_groups,
    )

    # Populate g3 and g4 columns with NaN
    all_desired_cols = _resolve_selected_feature_cols(target_groups)
    for g in ("g3", "g4"):
        if g in target_groups:
            for col in FEATURE_GROUPS[g]:
                custom_future[col] = np.nan

    final_cols = ["product_id", "week_start", *all_desired_cols]
    return custom_future[[c for c in final_cols if c in custom_future.columns]].reset_index(drop=True)


def report_feature_missingness(
    features_df: pd.DataFrame,
    groups: Sequence[str] = ("base", "g1", "g2", "g3", "g4"),
) -> dict[str, dict[str, float]]:
    """Print the row counts and share of missing values per feature for each group."""
    total_rows = len(features_df)
    print(f"=== Feature Groups Missingness Report (Total rows: {total_rows}) ===")
    missing_report: dict[str, dict[str, float]] = {}

    for g in groups:
        cols = FEATURE_GROUPS.get(g, [])
        present_cols = [c for c in cols if c in features_df.columns]
        if not present_cols:
            continue
        print(f"\nGroup [{g}] ({len(present_cols)} features):")
        group_stats: dict[str, float] = {}
        for c in present_cols:
            n_missing = int(features_df[c].isna().sum())
            pct_missing = (n_missing / total_rows * 100.0) if total_rows > 0 else 0.0
            group_stats[c] = pct_missing
            print(f"  {c:<25}: {n_missing:>5} missing ({pct_missing:>6.2f}%)")
        missing_report[g] = group_stats

    return missing_report


if __name__ == "__main__":
    weekly_path = Path("data/processed/weekly_demand.parquet")
    closed_path = Path("data/processed/closed_weeks.csv")
    if weekly_path.exists():
        weekly_data = pd.read_parquet(weekly_path)
        closed_data = pd.read_csv(closed_path) if closed_path.exists() else None
        print("Building features with all groups ('base', 'g1', 'g2', 'g3', 'g4')...")
        feat_df = build_features(
            weekly=weekly_data,
            closed_weeks=closed_data,
            groups=("base", "g1", "g2", "g3", "g4"),
        )
        report_feature_missingness(feat_df)
    else:
        print(f"Dataset {weekly_path} not found.")
