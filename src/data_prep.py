"""Data loading and preparation utilities for the Shelfwatch pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Non-product StockCode filtering (FR-D-2)
# ---------------------------------------------------------------------------

_NON_PRODUCT_CODES: frozenset[str] = frozenset({
    "POST", "D", "M", "DOT", "BANK CHARGES", "AMAZONFEE",
    "S", "B", "CRUK", "PADS", "C2",
})


def _is_non_product_code(code: str) -> bool:
    """Return True if *code* should be dropped as a non-product StockCode.

    Rules (applied after strip + upper):
    - Exact match against the fixed blocklist.
    - No digit anywhere in the code.
    - Starts with 'GIFT' or 'TEST'.
    """
    s = code.strip().upper()
    if s in _NON_PRODUCT_CODES:
        return True
    if not any(ch.isdigit() for ch in s):
        return True
    if s.startswith("GIFT") or s.startswith("TEST"):
        return True
    return False


def _cast_raw_string_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Cast mixed-type raw text columns so parquet serialization succeeds."""
    result = df.copy()
    for col in ("InvoiceNo", "StockCode"):
        if col in result.columns:
            result[col] = result[col].astype(str)
    if "Description" in result.columns:
        result["Description"] = result["Description"].astype("string")
    return result


def load_raw(raw_path: str | Path) -> pd.DataFrame:
    """Load raw Online Retail transactions from .xlsx, .csv, or .parquet.

    For .xlsx files, reads via openpyxl once, normalizes mixed-type string
    columns, writes a sibling .parquet cache, and reads that cache on later
    calls. CSV files are read with latin-1 encoding and parsed InvoiceDate.
    """
    path = Path(raw_path)
    suffix = path.suffix.lower()

    if suffix == ".xlsx":
        cache_path = path.with_suffix(".parquet")
        if cache_path.exists():
            return pd.read_parquet(cache_path)
        if not path.exists():
            raise FileNotFoundError(f"Raw data file not found: {path}")
        df = pd.read_excel(path, engine="openpyxl")
        df = _cast_raw_string_columns(df)
        df.to_parquet(cache_path, index=False)
        return df

    if suffix == ".csv":
        if not path.exists():
            raise FileNotFoundError(f"Raw data file not found: {path}")
        return pd.read_csv(path, encoding="latin-1", parse_dates=["InvoiceDate"])

    if suffix == ".parquet":
        if not path.exists():
            raise FileNotFoundError(f"Raw data file not found: {path}")
        return pd.read_parquet(path)

    raise ValueError(f"Unsupported raw file extension: {path.suffix!r}")


# ---------------------------------------------------------------------------
# Row-level cleaning (FR-D-2, FR-D-3, FR-D-9)
# ---------------------------------------------------------------------------


def clean_transactions(
    raw: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply row-level cleaning rules and return ``(cleaned_df, log_df)``.

    Parameters
    ----------
    raw:
        Raw transaction DataFrame as returned by :func:`load_raw`.

    Returns
    -------
    cleaned_df:
        Cleaned transactions with ``StockCode`` normalised to strip + upper.
    log_df:
        DataFrame with columns ``step`` (str) and ``rows_remaining`` (int),
        one row per cleaning step in order.

    Cleaning steps (in order):
        1. raw rows (starting count)
        2. drop cancellations — ``InvoiceNo`` after strip + upper starts with ``"C"``
        3. drop ``Quantity <= 0``
        4. drop ``UnitPrice <= 0``
        5. drop non-product ``StockCode`` values

    Rules:
        - ``StockCode`` is stripped and upper-cased on the working copy before
          step 5; the normalised value is kept in the returned DataFrame.
        - Rows with missing ``CustomerID`` are **kept** (FR-D-3).
        - The input DataFrame is never mutated.
    """
    df = raw.copy()
    log: list[dict[str, object]] = []

    def _record(step: str) -> None:
        log.append({"step": step, "rows_remaining": len(df)})

    _record("raw rows")

    # Step 2: drop cancellations
    mask_cancel = df["InvoiceNo"].astype(str).str.strip().str.upper().str.startswith("C")
    df = df[~mask_cancel].reset_index(drop=True)
    _record("drop cancellations")

    # Step 3: drop Quantity <= 0
    df = df[df["Quantity"] > 0].reset_index(drop=True)
    _record("drop Quantity <= 0")

    # Step 4: drop UnitPrice <= 0
    df = df[df["UnitPrice"] > 0].reset_index(drop=True)
    _record("drop UnitPrice <= 0")

    # Normalise StockCode before non-product filtering
    df["StockCode"] = df["StockCode"].astype(str).str.strip().str.upper()

    # Step 5: drop non-product StockCodes
    df = df[~df["StockCode"].map(_is_non_product_code)].reset_index(drop=True)
    _record("drop non-product StockCodes")

    log_df = pd.DataFrame(log)
    return df, log_df


# ---------------------------------------------------------------------------
# Week helpers (FR-D-4, FR-D-8)
# ---------------------------------------------------------------------------


def week_start_of(dates: pd.Series) -> pd.Series:
    """Return the Monday 00:00 of each timestamp's week.

    Parameters
    ----------
    dates:
        A Series of datetime-like values.

    Returns
    -------
    pd.Series of dtype datetime64[ns], same index as *dates*.
    """
    dt = pd.to_datetime(dates)
    return (dt - pd.to_timedelta(dt.dt.weekday, unit="D")).dt.normalize()


def full_week_range(dates: pd.Series) -> pd.DatetimeIndex:
    """Return Mondays of all full Mon-Sun weeks covered by *dates*.

    A week is 'full' if its Monday is on or after the earliest date in *dates*
    and its Sunday (Monday + 6 days) is on or before the latest date.
    Computed from the data; no dates are hard-coded.

    Parameters
    ----------
    dates:
        Series of datetime-like values (e.g. ``InvoiceDate`` after cleaning).

    Returns
    -------
    pd.DatetimeIndex of Monday timestamps at midnight, freq ``'W-MON'``.
    """
    dt = pd.to_datetime(dates)
    data_start = dt.min()
    data_end = dt.max()

    # First full week: first Monday on or after data_start
    days_ahead = (7 - data_start.weekday()) % 7
    first_mon = data_start.normalize() + pd.Timedelta(days=days_ahead)

    # Last full week: largest Monday M such that M + 6 days <= data_end
    latest_possible = data_end.normalize() - pd.Timedelta(days=6)
    days_back = latest_possible.weekday()
    last_mon = latest_possible - pd.Timedelta(days=days_back)

    if last_mon < first_mon:
        return pd.DatetimeIndex([])

    return pd.date_range(first_mon, last_mon, freq="W-MON")


def find_closed_weeks(
    df: pd.DataFrame,
    weeks: pd.DatetimeIndex,
) -> pd.DatetimeIndex:
    """Return the full weeks in *weeks* with zero transactions shop-wide.

    These are closures (e.g. the Christmas shutdown 2010-12-27), not
    zero-demand weeks for individual products (FR-D-8).

    Parameters
    ----------
    df:
        Cleaned transaction DataFrame.  If a ``week_start`` column is present
        it is used directly; otherwise ``InvoiceDate`` is mapped via
        :func:`week_start_of`.
    weeks:
        DatetimeIndex of candidate full-week Mondays (from
        :func:`full_week_range`).

    Returns
    -------
    pd.DatetimeIndex of Monday timestamps that had zero transactions.
    """
    if "week_start" in df.columns:
        active = pd.to_datetime(df["week_start"]).unique()
    else:
        active = week_start_of(df["InvoiceDate"]).unique()

    active_set = set(pd.Timestamp(w) for w in active)
    closed = [w for w in weeks if pd.Timestamp(w) not in active_set]
    return pd.DatetimeIndex(closed)


# ---------------------------------------------------------------------------
# Outlier cap (FR-D-5)
# ---------------------------------------------------------------------------


def cap_quantity(
    df: pd.DataFrame,
    train_mask: pd.Series,
    quantile: float = 0.999,
) -> tuple[pd.DataFrame, float, int]:
    """Cap line-level Quantity at a quantile derived from training rows only.

    Parameters
    ----------
    df:
        Cleaned transaction DataFrame with a ``Quantity`` column.
    train_mask:
        Boolean Series (same index as *df*) marking training-period rows.
        The cap is computed **only** on rows where ``train_mask`` is True.
    quantile:
        Quantile to use as the upper cap.  Default 0.999 (FR-D-5).

    Returns
    -------
    capped_df:
        Copy of *df* with ``Quantity`` clipped at *cap_value*.
    cap_value:
        The computed quantile (float).  On the real file, expected ~480.
    n_capped:
        Number of rows where ``Quantity`` exceeded *cap_value* before clipping.
    """
    cap_value = float(df.loc[train_mask, "Quantity"].quantile(quantile))
    result = df.copy()
    n_capped = int((result["Quantity"] > cap_value).sum())
    result["Quantity"] = result["Quantity"].clip(upper=cap_value)
    return result, cap_value, n_capped


# ---------------------------------------------------------------------------
# Contract C1 validator (FR-D-9)
# ---------------------------------------------------------------------------

_C1_COLUMNS: list[str] = [
    "product_id",
    "description",
    "week_start",
    "units",
    "orders",
    "customers",
    "avg_price",
]


def validate_c1(
    df: pd.DataFrame,
    last_full_week: str | pd.Timestamp | None = None,
    closed_weeks: pd.DatetimeIndex | pd.Series | list[Any] | None = None,
) -> None:
    """Raise ``AssertionError`` if *df* violates any contract C1 rule.

    Checks column order, types, value constraints, Monday-only ``week_start``,
    no duplicate product-week rows, correct sort order, exclusion of closed
    weeks, and gapless per-product week ranges ending at *last_full_week*.
    """
    assert list(df.columns) == _C1_COLUMNS, (
        f"Column mismatch: {list(df.columns)} != {_C1_COLUMNS}"
    )
    assert not df.empty, "weekly_demand must not be empty"
    assert not df.isna().any().any(), "weekly_demand contains NaN values"

    assert pd.api.types.is_string_dtype(df["product_id"]), "product_id must be string dtype"
    assert pd.api.types.is_string_dtype(df["description"]), "description must be string dtype"
    assert pd.api.types.is_datetime64_any_dtype(df["week_start"]), "week_start must be datetime"
    assert pd.api.types.is_integer_dtype(df["units"]), "units must be integer dtype"
    assert pd.api.types.is_integer_dtype(df["orders"]), "orders must be integer dtype"
    assert pd.api.types.is_integer_dtype(df["customers"]), "customers must be integer dtype"
    assert pd.api.types.is_float_dtype(df["avg_price"]), "avg_price must be float dtype"

    assert (df["units"] >= 0).all(), "units must be >= 0"
    assert (df["orders"] >= 0).all(), "orders must be >= 0"
    assert (df["customers"] >= 0).all(), "customers must be >= 0"
    assert (
        np.isfinite(df["avg_price"]).all() and (df["avg_price"] > 0).all()
    ), "avg_price must be finite and > 0"

    weeks = pd.to_datetime(df["week_start"])
    assert (weeks.dt.weekday == 0).all(), "All week_start values must be Mondays"
    assert not df.duplicated(subset=["product_id", "week_start"]).any(), (
        "Duplicate product-week rows found"
    )

    sorted_df = df.sort_values(["product_id", "week_start"]).reset_index(drop=True)
    assert df.reset_index(drop=True).equals(sorted_df), (
        "Rows must be sorted by product_id, week_start"
    )

    end_week = (
        pd.Timestamp(last_full_week).normalize()
        if last_full_week is not None
        else pd.Timestamp(weeks.max()).normalize()
    )
    if closed_weeks is not None:
        closed_set = {pd.Timestamp(w).normalize() for w in pd.to_datetime(list(closed_weeks))}
    else:
        all_grid = pd.date_range(weeks.min(), weeks.max(), freq="W-MON")
        present_weeks = set(weeks.dt.normalize())
        closed_set = {w for w in all_grid if w not in present_weeks}

    for pid, group in df.groupby("product_id", sort=False):
        prod_weeks = pd.to_datetime(group["week_start"]).dt.normalize().sort_values()
        if closed_set:
            assert not prod_weeks.isin(closed_set).any(), (
                f"Closed week must not be filled for product '{pid}'"
            )
        expected = pd.DatetimeIndex(
            [w for w in pd.date_range(prod_weeks.min(), end_week, freq="W-MON") if w not in closed_set]
        )
        assert list(prod_weeks) == list(expected), (
            f"Gap or early end in week range for product '{pid}' (expected end {end_week.date()})"
        )


# ---------------------------------------------------------------------------
# Weekly demand pipeline (FR-D-6, FR-D-7, FR-D-9, FR-D-10)
# ---------------------------------------------------------------------------


def _fill_product_weeks(
    group: pd.DataFrame,
    end_week: pd.Timestamp | None = None,
    closed_weeks: pd.DatetimeIndex | set[pd.Timestamp] | None = None,
) -> pd.DataFrame:
    """Reindex a single-product aggregated group from its first active week to *end_week*.

    Excludes *closed_weeks* (kept missing, never filled as 0), fills open gaps
    with units/orders/customers = 0, and forward-fills avg_price.
    """
    target_end = (
        pd.Timestamp(end_week).normalize()
        if end_week is not None
        else pd.Timestamp(group["week_start"].max()).normalize()
    )
    closed_set = (
        {pd.Timestamp(w).normalize() for w in closed_weeks}
        if closed_weeks is not None
        else set()
    )
    full_range = pd.date_range(group["week_start"].min(), target_end, freq="W-MON")
    if closed_set:
        full_range = pd.DatetimeIndex([w for w in full_range if w not in closed_set])
    g = group.set_index("week_start").reindex(full_range)
    g.index.name = "week_start"

    # Carry forward the constant columns
    for col in ("product_id", "description"):
        g[col] = g[col].ffill().bfill()

    # Fill zeros for count columns
    for col in ("units", "orders", "customers"):
        g[col] = g[col].fillna(0).astype("int64")

    # Forward-fill price (first row is always active so no leading NaN)
    g["avg_price"] = g["avg_price"].ffill()

    return g.reset_index()


def build_weekly_demand(
    raw_path: str | Path,
    out_path: str | Path = "data/processed/weekly_demand.parquet",
    log_path: str | Path = "data/processed/cleaning_log.csv",
    closed_weeks_path: str | Path = "data/processed/closed_weeks.csv",
    top_n: int = 100,
    test_weeks: int = 8,
    min_active_train_weeks: int = 30,
) -> pd.DataFrame:
    """Run the full weekly-demand pipeline and write contract C1 artifacts.

    Parameters
    ----------
    raw_path:
        Path to the raw UCI Online Retail file (.xlsx, .csv, or .parquet).
    out_path:
        Destination for ``weekly_demand.parquet`` (created if missing).
    log_path:
        Destination for ``cleaning_log.csv``.
    closed_weeks_path:
        Destination for ``closed_weeks.csv`` (C1b).
    top_n:
        Number of top products to retain (by training-period units).
    test_weeks:
        Number of trailing full weeks reserved as the test set.
    min_active_train_weeks:
        Minimum number of training weeks with sales for a product to qualify.

    Returns
    -------
    pd.DataFrame conforming to contract C1.
    """
    out_path = Path(out_path)
    log_path = Path(log_path)
    closed_weeks_path = Path(closed_weeks_path)
    for p in (out_path, log_path, closed_weeks_path):
        p.parent.mkdir(parents=True, exist_ok=True)

    # --- Step 1–5: load & clean (D1–D2 helpers) ---
    raw = load_raw(raw_path)
    df, log = clean_transactions(raw)
    # log already contains 5 steps from clean_transactions

    # --- Assign week_start ---
    df["week_start"] = week_start_of(df["InvoiceDate"])

    # --- Step 6: keep only full weeks ---
    weeks = full_week_range(df["InvoiceDate"])
    weeks_set = set(weeks)
    df = df[df["week_start"].isin(weeks_set)].reset_index(drop=True)
    log = pd.concat(
        [log, pd.DataFrame([{"step": "drop partial weeks", "rows_remaining": len(df)}])],
        ignore_index=True,
    )

    # --- Step 7: detect & save closed weeks (C1b) ---
    closed = find_closed_weeks(df, weeks)
    closed_df = pd.DataFrame({"week_start": closed.strftime("%Y-%m-%d")})
    closed_df.to_csv(closed_weeks_path, index=False)

    # --- Step 8: define train/test boundary ---
    test_start = weeks[-test_weeks]
    train_mask = df["week_start"] < test_start

    # --- Step 9: cap Quantity (training rows only) ---
    df, cap_value, n_capped = cap_quantity(df, train_mask)
    log = pd.concat(
        [log, pd.DataFrame([{"step": "cap Quantity (train)", "rows_remaining": len(df)}])],
        ignore_index=True,
    )

    # --- Step 10: select Top-N products (training period only) ---
    train_df = df[train_mask]
    product_stats = (
        train_df.groupby("StockCode")
        .agg(
            total_units=("Quantity", "sum"),
            active_weeks=("week_start", "nunique"),
        )
        .reset_index()
    )
    eligible = product_stats[product_stats["active_weeks"] >= min_active_train_weeks]
    top_products: list[str] = (
        eligible.nlargest(top_n, "total_units")["StockCode"].tolist()
    )
    df = df[df["StockCode"].isin(top_products)].reset_index(drop=True)
    log = pd.concat(
        [log, pd.DataFrame([{"step": f"keep top-{top_n} products", "rows_remaining": len(df)}])],
        ignore_index=True,
    )

    # --- Step 11: compute product descriptions (mode over training rows) ---
    # Recompute train mask against the now-filtered, re-indexed df
    train_mask_2 = df["week_start"] < test_start
    desc_df = (
        df[train_mask_2]
        .assign(Description=lambda x: x["Description"].astype(str))
        .groupby("StockCode")["Description"]
        .agg(lambda s: s.mode().iloc[0])
        .reset_index()
        .rename(columns={"StockCode": "product_id", "Description": "description"})
    )

    # --- Step 12: aggregate to product × week ---
    agg = (
        df.groupby(["StockCode", "week_start"], sort=True)
        .agg(
            units=("Quantity", "sum"),
            orders=("InvoiceNo", "nunique"),
            customers=("CustomerID", "nunique"),
            avg_price=("UnitPrice", "mean"),
        )
        .reset_index()
        .rename(columns={"StockCode": "product_id"})
    )

    # Merge description
    agg = agg.merge(desc_df, on="product_id", how="left")

    # --- Step 13: fill missing weeks per product through last full week ---
    end_week = pd.Timestamp(weeks.max()).normalize()
    closed_set = {pd.Timestamp(w).normalize() for w in closed}
    filled_parts: list[pd.DataFrame] = []
    for pid, group in agg.groupby("product_id", sort=False):
        filled_parts.append(
            _fill_product_weeks(group, end_week=end_week, closed_weeks=closed_set)
        )

    result = pd.concat(filled_parts, ignore_index=True)

    # --- Step 14: enforce column order, types, sort ---
    result = result[_C1_COLUMNS].copy()
    result["product_id"] = result["product_id"].astype(str)
    result["description"] = result["description"].astype(str)
    result["week_start"] = pd.to_datetime(result["week_start"])
    for col in ("units", "orders", "customers"):
        result[col] = result[col].astype("int64")
    result["avg_price"] = result["avg_price"].astype(float)
    result = result.sort_values(["product_id", "week_start"]).reset_index(drop=True)

    # --- Step 15: validate & save ---
    validate_c1(result, last_full_week=end_week, closed_weeks=closed)
    result.to_parquet(out_path, index=False)
    log.to_csv(log_path, index=False)

    return result



# ---------------------------------------------------------------------------
# CLI: python -m src.data_prep --raw "data/raw/Online Retail.xlsx"
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    _parser = argparse.ArgumentParser(
        description="Build weekly_demand.parquet from the raw UCI Online Retail file."
    )
    _parser.add_argument("--raw", required=True, help="Path to raw Online Retail file")
    _parser.add_argument(
        "--out",
        default="data/processed/weekly_demand.parquet",
        help="Output parquet path",
    )
    _parser.add_argument(
        "--log",
        default="data/processed/cleaning_log.csv",
        help="Cleaning log CSV path",
    )
    _parser.add_argument(
        "--closed",
        default="data/processed/closed_weeks.csv",
        help="Closed-weeks CSV path",
    )
    _args = _parser.parse_args()

    print("Running weekly demand pipeline…")
    _result = build_weekly_demand(
        raw_path=_args.raw,
        out_path=_args.out,
        log_path=_args.log,
        closed_weeks_path=_args.closed,
    )

    _log = pd.read_csv(_args.log)
    print("\nCleaning log:")
    print(_log.to_string(index=False))

    _n_rows = len(_result)
    _n_products = _result["product_id"].nunique()
    _weeks = pd.to_datetime(_result["week_start"])
    _first_week = _weeks.min().date()
    _last_week = _weeks.max().date()
    _n_weeks = _result["week_start"].nunique()

    _closed = pd.read_csv(_args.closed)
    _closed_list = _closed["week_start"].tolist()

    print(
        f"\nSummary: {_n_rows:,} rows | {_n_products} products | {_n_weeks} weeks"
        f" | {_first_week} to {_last_week}"
    )
    print(f"Closed weeks: {_closed_list}")
    print(f"\nSaved: {_args.out}")
