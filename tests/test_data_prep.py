"""Unit tests for src/data_prep.py."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.data_prep import (
    build_weekly_demand,
    cap_quantity,
    clean_transactions,
    find_closed_weeks,
    full_week_range,
    load_raw,
    validate_c1,
    week_start_of,
)
from tests.synthetic import make_synthetic_raw


def _sample_raw_df() -> pd.DataFrame:
    """Build a tiny raw-schema DataFrame including numeric Description cells."""
    return pd.DataFrame(
        {
            "InvoiceNo": [536365, "C536366", 536367],
            "StockCode": [85123, "85123A", "84879"],
            "Description": [12345, None, "ASSORTED COLOUR BIRD ORNAMENT"],
            "Quantity": [6, -1, 32],
            "InvoiceDate": pd.to_datetime(
                ["2010-12-01 08:26:00", "2010-12-01 08:28:00", "2010-12-01 08:34:00"]
            ),
            "UnitPrice": [2.55, 2.55, 1.69],
            "CustomerID": [17850.0, None, 13047.0],
            "Country": ["United Kingdom", "United Kingdom", "United Kingdom"],
        }
    )


def test_load_raw_excel_caches_parquet_and_handles_numeric_description(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """First Excel load writes parquet cache; second load reads cache without read_excel."""
    xlsx_path = tmp_path / "sample.xlsx"
    cache_path = tmp_path / "sample.parquet"
    _sample_raw_df().to_excel(xlsx_path, index=False, engine="openpyxl")

    first = load_raw(xlsx_path)
    assert cache_path.exists()
    assert len(first) == 3
    assert first["Description"].dtype == pd.StringDtype()
    assert first.loc[0, "Description"] == "12345"
    assert pd.isna(first.loc[1, "Description"])
    assert first.loc[2, "Description"] == "ASSORTED COLOUR BIRD ORNAMENT"
    assert first["InvoiceNo"].tolist() == ["536365", "C536366", "536367"]
    assert first["StockCode"].tolist() == ["85123", "85123A", "84879"]

    def _fail_read_excel(*args: object, **kwargs: object) -> pd.DataFrame:
        raise AssertionError("read_excel must not be called when parquet cache exists")

    monkeypatch.setattr(pd, "read_excel", _fail_read_excel)

    second = load_raw(xlsx_path)
    pd.testing.assert_frame_equal(first, second)


def test_load_raw_csv_latin1_and_invoice_date(tmp_path: Path) -> None:
    """CSV loads with latin-1 encoding and parses InvoiceDate as datetime."""
    csv_path = tmp_path / "sample.csv"
    df = _sample_raw_df()
    df["Description"] = ["CAFÉ MUG £5", "T-LIGHT HOLDER", "BIRD ORNAMENT"]
    df.to_csv(csv_path, index=False, encoding="latin-1")

    loaded = load_raw(csv_path)
    assert len(loaded) == 3
    assert pd.api.types.is_datetime64_any_dtype(loaded["InvoiceDate"])
    assert loaded.loc[0, "Description"] == "CAFÉ MUG £5"


def test_load_raw_parquet_direct(tmp_path: Path) -> None:
    """Parquet files load directly without modification."""
    parquet_path = tmp_path / "sample.parquet"
    df = _sample_raw_df()
    df["InvoiceNo"] = df["InvoiceNo"].astype(str)
    df["StockCode"] = df["StockCode"].astype(str)
    df["Description"] = df["Description"].astype("string")
    df.to_parquet(parquet_path, index=False)

    loaded = load_raw(parquet_path)
    pd.testing.assert_frame_equal(df, loaded)


def test_load_raw_invalid_extension_and_missing_file(tmp_path: Path) -> None:
    """Unsupported extensions raise ValueError; missing files raise FileNotFoundError."""
    bad_ext = tmp_path / "sample.txt"
    bad_ext.write_text("hello", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported raw file extension"):
        load_raw(bad_ext)

    with pytest.raises(FileNotFoundError):
        load_raw(tmp_path / "does_not_exist.xlsx")


# ---------------------------------------------------------------------------
# clean_transactions tests
# ---------------------------------------------------------------------------


def _make_junk_raw() -> pd.DataFrame:
    """Return a synthetic raw DataFrame with one row per junk category plus normal rows."""
    return pd.DataFrame(
        {
            "InvoiceNo": [
                "C999001",   # cancellation — starts with C
                "536001",    # Quantity <= 0
                "536002",    # UnitPrice <= 0
                "536003",    # blocklist code: POST
                "536004",    # no-digit code: DOTABC
                "536005",    # GIFT-prefix code
                "536006",    # TEST-prefix code
                "536007",    # normal, with CustomerID
                "536008",    # normal, CustomerID missing (must be kept per FR-D-3)
            ],
            "StockCode": [
                "85123A",    # junk row — cancellation
                "85123A",    # junk row — Quantity <= 0
                "85123A",    # junk row — UnitPrice <= 0
                "POST",      # non-product: blocklist
                "dotabc",    # non-product: no digit (lowercase, to test normalisation)
                "gift_0001", # non-product: GIFT prefix
                "TEST001",   # non-product: TEST prefix
                "85123A",    # normal
                " 84879 ",   # normal with whitespace (to test strip)
            ],
            "Description": [
                "WHITE HANGING HEART",
                "WHITE HANGING HEART",
                "WHITE HANGING HEART",
                "POSTAGE",
                "DOTTED ABC",
                "GIFT VOUCHER",
                "TEST ITEM",
                "WHITE HANGING HEART",
                "ASSORTED BIRD ORNAMENT",
            ],
            "Quantity": [6, -1, 6, 6, 6, 6, 6, 6, 32],
            "InvoiceDate": pd.to_datetime(["2010-12-01"] * 9),
            "UnitPrice": [2.55, 2.55, -1.0, 2.55, 2.55, 2.55, 2.55, 2.55, 1.69],
            "CustomerID": [17850.0, 17850.0, 17850.0, None, 17850.0, 17850.0, 17850.0, 17850.0, None],
            "Country": ["United Kingdom"] * 9,
        }
    )


def test_clean_transactions_only_normal_rows_survive() -> None:
    """Only the two normal rows survive; the CustomerID-null normal row is kept (FR-D-3)."""
    raw = _make_junk_raw()
    cleaned, _ = clean_transactions(raw)

    assert len(cleaned) == 2, f"Expected 2 normal rows, got {len(cleaned)}"
    # Both surviving StockCodes should be the product codes
    assert set(cleaned["StockCode"]) == {"85123A", "84879"}
    # The CustomerID-null normal row (84879) must be present
    assert cleaned["CustomerID"].isna().any(), "CustomerID-null normal row was wrongly dropped"


def test_clean_transactions_log_shape_and_monotonic() -> None:
    """Log has the right columns, 5 rows, and rows_remaining is non-increasing."""
    raw = _make_junk_raw()
    _, log = clean_transactions(raw)

    assert list(log.columns) == ["step", "rows_remaining"]
    assert len(log) == 5

    counts = log["rows_remaining"].tolist()
    assert counts == sorted(counts, reverse=True), "rows_remaining must be non-increasing"
    # First count equals total raw rows
    assert counts[0] == len(raw)


def test_clean_transactions_does_not_mutate_input() -> None:
    """The input DataFrame must be unchanged after clean_transactions."""
    raw = _make_junk_raw()
    original = raw.copy()
    clean_transactions(raw)
    pd.testing.assert_frame_equal(raw, original)


def test_clean_transactions_stock_code_normalised() -> None:
    """All StockCode values in the cleaned output are stripped and upper-cased."""
    raw = _make_junk_raw()
    cleaned, _ = clean_transactions(raw)

    for code in cleaned["StockCode"]:
        assert code == code.strip().upper(), f"StockCode not normalised: {code!r}"


# ---------------------------------------------------------------------------
# week_start_of tests
# ---------------------------------------------------------------------------


def test_week_start_of_known_dates() -> None:
    """Wed 2011-03-02 and Sun 2011-03-06 → Mon 2011-02-28; Mon 2011-03-07 → itself."""
    dates = pd.Series(
        pd.to_datetime(["2011-03-02", "2011-03-06", "2011-03-07"])
    )
    result = week_start_of(dates)

    expected = pd.Series(
        pd.to_datetime(["2011-02-28", "2011-02-28", "2011-03-07"])
    )
    pd.testing.assert_series_equal(result.reset_index(drop=True), expected)
    assert (result.dt.weekday == 0).all(), "All results must be Mondays"
    assert (result.dt.time == pd.Timestamp("00:00:00").time()).all()


# ---------------------------------------------------------------------------
# full_week_range tests
# ---------------------------------------------------------------------------


def test_full_week_range_uci_shape() -> None:
    """UCI date range 2010-12-01 to 2011-12-09 → 52 weeks, 2010-12-06 to 2011-11-28."""
    dates = pd.Series(pd.to_datetime(["2010-12-01", "2011-12-09"]))
    result = full_week_range(dates)

    assert len(result) == 52, f"Expected 52 weeks, got {len(result)}"
    assert result[0] == pd.Timestamp("2010-12-06")
    assert result[-1] == pd.Timestamp("2011-11-28")


def test_full_week_range_all_monday() -> None:
    """Every date in the result is a Monday."""
    dates = pd.Series(pd.to_datetime(["2010-12-01", "2011-12-09"]))
    result = full_week_range(dates)
    assert (result.weekday == 0).all(), "Not all dates are Mondays"


# ---------------------------------------------------------------------------
# find_closed_weeks tests
# ---------------------------------------------------------------------------


def _make_transactions_with_gaps() -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    """Return a tiny transaction DataFrame with activity on weeks 1 and 3 only."""
    # weeks: 2011-01-03, 2011-01-10, 2011-01-17, 2011-01-24
    week1 = pd.Timestamp("2011-01-03")
    week3 = pd.Timestamp("2011-01-17")
    df = pd.DataFrame(
        {
            "week_start": [week1, week1, week3],
            "StockCode": ["A1", "B2", "A1"],
            "Quantity": [5, 3, 7],
        }
    )
    all_weeks = pd.date_range("2011-01-03", periods=4, freq="W-MON")
    return df, all_weeks


def test_find_closed_weeks_small_example() -> None:
    """Weeks 2 and 4 (no transactions) are detected as closed."""
    df, all_weeks = _make_transactions_with_gaps()
    closed = find_closed_weeks(df, all_weeks)

    assert len(closed) == 2
    assert pd.Timestamp("2011-01-10") in closed   # week 2
    assert pd.Timestamp("2011-01-24") in closed   # week 4
    assert pd.Timestamp("2011-01-03") not in closed
    assert pd.Timestamp("2011-01-17") not in closed


# ---------------------------------------------------------------------------
# cap_quantity tests
# ---------------------------------------------------------------------------


def _make_cap_df() -> tuple[pd.DataFrame, pd.Series]:
    """Return a DataFrame and train_mask where test has a huge outlier."""
    n_train = 200
    rng = __import__("numpy").random.default_rng(42)
    train_qty = rng.integers(1, 101, size=n_train).tolist()   # max ~100
    # One test row with a giant quantity that must NOT affect the cap
    test_qty = [50_000]
    quantities = train_qty + test_qty
    df = pd.DataFrame({"Quantity": quantities, "Other": range(len(quantities))})
    train_mask = pd.Series([True] * n_train + [False])
    return df, train_mask


def test_cap_quantity_uses_train_only() -> None:
    """Cap value is derived from training rows; a 50,000-unit test row doesn't move it."""
    df, train_mask = _make_cap_df()
    cap_before = float(df.loc[train_mask, "Quantity"].quantile(0.999))

    capped, cap_value, n_capped = cap_quantity(df, train_mask, quantile=0.999)

    assert cap_value == cap_before, "Cap must not be influenced by test-period rows"
    assert cap_value < 200, "Cap should be near the training max (~100), not 50,000"
    # The test-period outlier row (50,000) must have been capped
    assert capped.loc[~train_mask, "Quantity"].iloc[0] == cap_value


def test_cap_quantity_returns_copy_not_mutate() -> None:
    """The original DataFrame is unchanged after cap_quantity."""
    df, train_mask = _make_cap_df()
    original_qty = df["Quantity"].copy()
    cap_quantity(df, train_mask)
    pd.testing.assert_series_equal(df["Quantity"], original_qty)


# ---------------------------------------------------------------------------
# End-to-end synthetic pipeline tests (Task D5 / SRS T-D-1 to T-D-4)
# ---------------------------------------------------------------------------


def _run_synthetic_pipeline(tmp_path: Path, raw_df: pd.DataFrame) -> tuple[pd.DataFrame, Path]:
    """Save *raw_df* to a temp parquet and run build_weekly_demand into *tmp_path*."""
    raw_parquet = tmp_path / "synthetic_raw.parquet"
    out_parquet = tmp_path / "weekly_demand.parquet"
    log_csv = tmp_path / "cleaning_log.csv"
    closed_csv = tmp_path / "closed_weeks.csv"

    raw_df.to_parquet(raw_parquet, index=False)
    result = build_weekly_demand(
        raw_path=raw_parquet,
        out_path=out_parquet,
        log_path=log_csv,
        closed_weeks_path=closed_csv,
    )
    return result, closed_csv


def test_pipeline_synthetic_c1_schema_and_invariants(tmp_path: Path) -> None:
    """Synthetic pipeline produces valid C1 output with 100 products and gapless weeks."""
    raw = make_synthetic_raw(n_products=150, seed=42)
    df, closed_csv = _run_synthetic_pipeline(tmp_path, raw)

    # Schema equals C1
    validate_c1(df)
    assert list(df.columns) == [
        "product_id",
        "description",
        "week_start",
        "units",
        "orders",
        "customers",
        "avg_price",
    ]

    # Exactly 100 products
    assert df["product_id"].nunique() == 100

    # Mondays only and no duplicates
    weeks = pd.to_datetime(df["week_start"])
    assert (weeks.dt.weekday == 0).all()
    assert not df.duplicated(subset=["product_id", "week_start"]).any()

    # No partial weeks
    assert weeks.min() >= pd.Timestamp("2010-12-06")
    assert weeks.max() <= pd.Timestamp("2011-11-28")

    # No negative units, units are ints, zero-demand weeks exist
    assert pd.api.types.is_integer_dtype(df["units"])
    assert (df["units"] >= 0).all()
    assert (df["units"] == 0).any(), "Expected filled zero-demand weeks"

    # Closed week 2010-12-27 detected in synthetic data and excluded from weekly_demand
    closed = pd.read_csv(closed_csv)
    assert "2010-12-27" in closed["week_start"].tolist()
    closed_set = set(pd.to_datetime(closed["week_start"]).dt.normalize())
    assert not weeks.dt.normalize().isin(closed_set).any()

    # No gaps inside each product's range, ending at last full week, excluding closed weeks
    end_week = weeks.max().normalize()
    for _, group in df.groupby("product_id", sort=False):
        prod_weeks = pd.to_datetime(group["week_start"]).dt.normalize().sort_values()
        expected = [
            w for w in pd.date_range(prod_weeks.min(), end_week, freq="W-MON")
            if w not in closed_set
        ]
        assert list(prod_weeks) == expected


def test_pipeline_synthetic_junk_removed_and_outlier_capped(tmp_path: Path) -> None:
    """Injected junk rows are removed and the 80,995-unit outlier is capped below 20,000."""
    raw = make_synthetic_raw(n_products=150, seed=42)

    cleaned, _ = clean_transactions(raw)
    assert "C999001" not in set(cleaned["InvoiceNo"])
    assert "POST" not in set(cleaned["StockCode"])
    assert "DOT" not in set(cleaned["StockCode"])
    assert "GIFT_0001_20" not in set(cleaned["StockCode"])
    assert (cleaned["UnitPrice"] > 0).all()

    df, _ = _run_synthetic_pipeline(tmp_path, raw)
    assert not {"POST", "DOT", "GIFT_0001_20", "99999Z"}.intersection(set(df["product_id"]))
    assert df["units"].max() < 20_000


def test_pipeline_synthetic_top100_unchanged_by_test_inflation(tmp_path: Path) -> None:
    """Inflating a runner-up product in the test period does not change the Top 100."""
    raw_base = make_synthetic_raw(n_products=150, seed=42)
    base_dir = tmp_path / "base"
    base_dir.mkdir()
    base_df, _ = _run_synthetic_pipeline(base_dir, raw_base)
    base_products = set(base_df["product_id"].unique())
    assert "PROD100" not in base_products

    # Massively inflate PROD100 strictly inside the test period (>= 2011-10-10)
    raw_inflated = raw_base.copy()
    test_prod100_mask = (
        (raw_inflated["StockCode"] == "PROD100")
        & (raw_inflated["InvoiceDate"] >= pd.Timestamp("2011-10-10"))
    )
    assert test_prod100_mask.any(), "PROD100 must have test-period rows to inflate"
    raw_inflated.loc[test_prod100_mask, "Quantity"] = 100_000

    inflated_dir = tmp_path / "inflated"
    inflated_dir.mkdir()
    inflated_df, _ = _run_synthetic_pipeline(inflated_dir, raw_inflated)
    assert set(inflated_df["product_id"].unique()) == base_products


def test_pipeline_closed_weeks_small_example(tmp_path: Path) -> None:
    """Closed-week detection via build_weekly_demand flags an empty week in a small dataset."""
    # 4 full Mon-Sun weeks: 2011-01-03, 2011-01-10 (closed), 2011-01-17, 2011-01-24
    raw_small = pd.DataFrame(
        {
            "InvoiceNo": ["1001", "1002", "1003", "1004", "1005", "1006"],
            "StockCode": ["A1", "B2", "A1", "B2", "A1", "B2"],
            "Description": ["ITEM A", "ITEM B", "ITEM A", "ITEM B", "ITEM A", "ITEM B"],
            "Quantity": [10, 8, 12, 9, 11, 7],
            "InvoiceDate": pd.to_datetime(
                [
                    "2011-01-03 09:00:00",
                    "2011-01-09 16:00:00",
                    "2011-01-17 10:00:00",
                    "2011-01-18 11:00:00",
                    "2011-01-24 10:00:00",
                    "2011-01-30 17:00:00",
                ]
            ),
            "UnitPrice": [2.0, 3.0, 2.0, 3.0, 2.0, 3.0],
            "CustomerID": [101.0, 102.0, 101.0, 102.0, 101.0, 102.0],
            "Country": ["United Kingdom"] * 6,
        }
    )
    raw_parquet = tmp_path / "small.parquet"
    raw_small.to_parquet(raw_parquet, index=False)

    closed_csv = tmp_path / "small_closed.csv"
    build_weekly_demand(
        raw_path=raw_parquet,
        out_path=tmp_path / "small_out.parquet",
        log_path=tmp_path / "small_log.csv",
        closed_weeks_path=closed_csv,
        top_n=2,
        test_weeks=1,
        min_active_train_weeks=1,
    )
    closed = pd.read_csv(closed_csv)
    assert closed["week_start"].tolist() == ["2011-01-10"]


def test_validate_c1_raises_on_contract_violations() -> None:
    """validate_c1 raises AssertionError for every C1 contract violation."""
    valid = pd.DataFrame(
        {
            "product_id": ["P1", "P1", "P2", "P2"],
            "description": ["ITEM 1", "ITEM 1", "ITEM 2", "ITEM 2"],
            "week_start": pd.to_datetime(
                ["2011-01-03", "2011-01-10", "2011-01-03", "2011-01-10"]
            ),
            "units": [10, 0, 5, 0],
            "orders": [2, 0, 1, 0],
            "customers": [2, 0, 1, 0],
            "avg_price": [2.5, 2.5, 3.0, 3.0],
        }
    )
    validate_c1(valid)

    # 1. Wrong column order
    with pytest.raises(AssertionError):
        validate_c1(valid[["description", "product_id", "week_start", "units", "orders", "customers", "avg_price"]])

    # 2. Non-Monday week_start
    bad_day = valid.copy()
    bad_day.loc[0, "week_start"] = pd.Timestamp("2011-01-04")
    with pytest.raises(AssertionError):
        validate_c1(bad_day)

    # 3. Negative units
    neg_units = valid.copy()
    neg_units.loc[0, "units"] = -1
    with pytest.raises(AssertionError):
        validate_c1(neg_units)

    # 4. Duplicate product-week rows
    dup = pd.concat([valid.iloc[[0]], valid], ignore_index=True)
    with pytest.raises(AssertionError):
        validate_c1(dup)

    # 5. Non-integer count column
    bad_dtype = valid.copy()
    bad_dtype["units"] = bad_dtype["units"].astype(float)
    with pytest.raises(AssertionError):
        validate_c1(bad_dtype)

    # 6. Non-positive avg_price
    bad_price = valid.copy()
    bad_price.loc[0, "avg_price"] = 0.0
    with pytest.raises(AssertionError):
        validate_c1(bad_price)

    # 7. Gap in product week range
    gapped = valid.copy()
    gapped.loc[1, "week_start"] = pd.Timestamp("2011-01-17")
    with pytest.raises(AssertionError):
        validate_c1(gapped)

    # 8. Unsorted rows
    unsorted = valid.iloc[[2, 0, 1, 3]].reset_index(drop=True)
    with pytest.raises(AssertionError):
        validate_c1(unsorted)

    # 9. Product ends early (before last_full_week)
    ended_early = valid.iloc[[0, 1, 2]].reset_index(drop=True)
    with pytest.raises(AssertionError):
        validate_c1(ended_early)
    with pytest.raises(AssertionError):
        validate_c1(valid, last_full_week="2011-01-17")

    # 10. Closure week filled
    with pytest.raises(AssertionError):
        validate_c1(valid, closed_weeks=["2011-01-10"])


def test_fill_extends_through_last_full_week_and_skips_closure(tmp_path: Path) -> None:
    """Verify tail zero-fill through last full week, no pre-first-sale rows, and closure excluded."""
    # 14 full Mon-Sun weeks: 2011-01-03 .. 2011-04-04 (test_weeks=8 -> test starts 2011-02-14).
    # 2011-01-17 is closed (no sales for any product).
    # P100 starts 2011-01-03, stops selling 2011-01-31 (before test period).
    # P200 starts 2011-01-24 (after closure) and sells through 2011-04-10 (end of week 2011-04-04).
    rows = [
        ("1001", "P100", "EARLY", 10, "2011-01-03 10:00:00", 2.0, 101.0),
        ("1002", "P100", "EARLY", 12, "2011-01-10 10:00:00", 2.0, 101.0),
        ("1003", "P100", "EARLY", 15, "2011-01-31 10:00:00", 2.5, 101.0),
        ("1004", "P200", "LATE", 20, "2011-01-24 10:00:00", 4.0, 102.0),
        ("1005", "P200", "LATE", 25, "2011-02-07 10:00:00", 4.0, 102.0),
    ]
    for i, w_str in enumerate(pd.date_range("2011-02-14", "2011-04-04", freq="W-MON")):
        rows.append((str(2000 + i), "P200", "LATE", 30, f"{w_str.date()} 10:00:00", 4.5, 102.0))
    rows.append(("2099", "P200", "LATE", 5, "2011-04-10 17:00:00", 4.5, 102.0))
    raw = pd.DataFrame(
        rows,
        columns=[
            "InvoiceNo",
            "StockCode",
            "Description",
            "Quantity",
            "InvoiceDate",
            "UnitPrice",
            "CustomerID",
        ],
    )
    raw["InvoiceDate"] = pd.to_datetime(raw["InvoiceDate"])
    raw["Country"] = "United Kingdom"

    raw_parquet = tmp_path / "fill_test.parquet"
    raw.to_parquet(raw_parquet, index=False)

    out_df = build_weekly_demand(
        raw_path=raw_parquet,
        out_path=tmp_path / "out.parquet",
        log_path=tmp_path / "log.csv",
        closed_weeks_path=tmp_path / "closed.csv",
        top_n=2,
        test_weeks=8,
        min_active_train_weeks=1,
    )

    # (c) Closure week 2011-01-17 is never filled
    assert pd.Timestamp("2011-01-17") not in set(out_df["week_start"])

    # (a) P100 has all 8 test-period rows with units == 0 and forward-filled avg_price == 2.5
    early_test = out_df[
        (out_df["product_id"] == "P100")
        & (out_df["week_start"] >= pd.Timestamp("2011-02-14"))
    ]
    assert len(early_test) == 8
    assert (early_test["units"] == 0).all()
    assert (early_test["orders"] == 0).all()
    assert (early_test["customers"] == 0).all()
    assert (early_test["avg_price"] == 2.5).all()

    # (b) P200 has no rows before its first sale (2011-01-24)
    late_rows = out_df[out_df["product_id"] == "P200"]
    assert late_rows["week_start"].min() == pd.Timestamp("2011-01-24")
    assert late_rows["week_start"].max() == pd.Timestamp("2011-04-04")
