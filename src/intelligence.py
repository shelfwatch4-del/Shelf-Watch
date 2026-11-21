"""Product intelligence module for the Shelfwatch pipeline (Contract C11).

Implements:
- Syntetos-Boylan demand pattern classification (smooth, erratic, intermittent, lumpy)
- ABC revenue Pareto analysis
- XYZ demand predictability classification
- Product lifecycle stage estimation
- Operational signals: return rate, export share, customer concentration, and coverage
- Catalog-wide demand pattern benchmarking
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.data_prep import (
    _is_non_product_code,
    load_raw,
    week_start_of,
)

logger = logging.getLogger(__name__)

# Default file paths
DEFAULT_WEEKLY_PATH = Path("data/processed/weekly_demand.parquet")
DEFAULT_RAW_PATH = Path("data/raw/Online Retail.parquet")
DEFAULT_ENRICHMENT_PATH = Path("data/processed/enrichment_weekly.parquet")
DEFAULT_INTELLIGENCE_OUTPUT = Path("data/processed/product_intelligence.csv")
DEFAULT_PATTERN_ALL_OUTPUT = Path("data/processed/pattern_all_products.csv")
DEFAULT_BASKET_PAIRS_OUTPUT = Path("data/processed/basket_pairs.csv")

# Standard training cutoff
DEFAULT_TRAIN_END = "2011-10-10"

# Contract C11 columns
C11_COLUMNS = [
    "product_id",
    "description",
    "adi",
    "cv2",
    "pattern_class",
    "abc_class",
    "xyz_class",
    "lifecycle_stage",
    "weeks_since_last_sale",
    "return_rate",
    "export_share",
    "top3_customer_share",
    "known_customer_coverage",
    "revenue_share",
]

# Contract C12 columns
C12_COLUMNS = [
    "product_id",
    "companion_id",
    "companion_description",
    "n_invoices",
    "confidence",
    "lift",
]


# ---------------------------------------------------------------------------
# 1. Demand pattern classification (Syntetos-Boylan)
# ---------------------------------------------------------------------------


def classify_demand_pattern(adi: float, cv2: float) -> str:
    """Classify demand pattern based on Syntetos-Boylan cutoffs.

    Cutoffs:
        ADI cutoff = 1.32
        CV2 cutoff = 0.49

    Returns:
        'smooth', 'erratic', 'intermittent', or 'lumpy'
    """
    if adi < 1.32 and cv2 < 0.49:
        return "smooth"
    if adi < 1.32 and cv2 >= 0.49:
        return "erratic"
    if adi >= 1.32 and cv2 < 0.49:
        return "intermittent"
    return "lumpy"


def compute_demand_pattern_metrics(
    units_series: Sequence[float] | pd.Series,
    total_open_weeks: int | None = None,
) -> tuple[float, float, str]:
    """Compute ADI, CV2, and demand pattern class for a product.

    Parameters
    ----------
    units_series:
        Weekly unit sales in open training weeks, ordered chronologically.
    total_open_weeks:
        Optional total number of open training weeks in the period. If provided,
        n_weeks is measured from the index of the product's first sale (units > 0)
        to the end of the training period. If None, len(units_series) after the
        first sale is used.

    Returns
    -------
    tuple[float, float, str]:
        (adi, cv2, pattern_class)
    """
    s = pd.Series(units_series, dtype=float).reset_index(drop=True)
    nonzero_mask = s > 0

    if not nonzero_mask.any():
        # No sales at all
        n_weeks = total_open_weeks if total_open_weeks is not None else len(s)
        return float(n_weeks) if n_weeks > 0 else 1.0, 0.0, "intermittent"

    first_sale_idx = int(nonzero_mask.idxmax())
    if total_open_weeks is not None:
        # Number of open weeks from first sale to the end
        n_weeks = total_open_weeks - first_sale_idx
    else:
        n_weeks = len(s) - first_sale_idx

    active_weeks = int(nonzero_mask.sum())
    adi = float(n_weeks / active_weeks) if active_weeks > 0 else float(n_weeks)

    nonzero_units = s[nonzero_mask].values
    if len(nonzero_units) <= 1:
        cv2 = 0.0
    else:
        mean_val = float(np.mean(nonzero_units))
        std_val = float(np.std(nonzero_units, ddof=1))
        cv2 = float((std_val / mean_val) ** 2) if mean_val > 0 else 0.0

    pattern_class = classify_demand_pattern(adi, cv2)
    return adi, cv2, pattern_class


# ---------------------------------------------------------------------------
# 2. ABC & XYZ Classification
# ---------------------------------------------------------------------------


def compute_abc_classes(
    revenue_by_product: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    """Compute revenue shares and ABC classes from total training revenue.

    Pareto rules:
    - Sort products descending by revenue.
    - Cumulative revenue share up to 80% is 'A'.
    - Cumulative revenue share up to 95% is 'B'.
    - Remainder is 'C'.
    - The top revenue product is always assigned 'A'.

    Returns
    -------
    tuple[pd.Series, pd.Series]:
        (revenue_share, abc_class) indexed by product_id matching revenue_by_product.
    """
    rev = revenue_by_product.astype(float).copy()
    total_rev = rev.sum()

    if total_rev <= 0:
        shares = pd.Series(1.0 / len(rev), index=rev.index)
        classes = pd.Series("C", index=rev.index)
        return shares, classes

    sorted_rev = rev.sort_values(ascending=False)
    sorted_shares = sorted_rev / total_rev
    cum_shares = sorted_shares.cumsum()

    classes_list: list[str] = []
    for idx, cum_val in enumerate(cum_shares):
        if idx == 0 or cum_val <= 0.80:
            classes_list.append("A")
        elif cum_val <= 0.95:
            classes_list.append("B")
        else:
            classes_list.append("C")

    abc_classes = pd.Series(classes_list, index=sorted_rev.index).reindex(rev.index)
    shares_series = sorted_shares.reindex(rev.index)
    return shares_series, abc_classes


def compute_xyz_class(units_series: Sequence[float] | pd.Series) -> tuple[float, str]:
    """Compute coefficient of variation and XYZ class for weekly units.

    X: CV <= 0.5 (steady, high predictability)
    Y: 0.5 < CV <= 1.0 (fluctuating, moderate predictability)
    Z: CV > 1.0 (volatile, low predictability)
    """
    s = pd.Series(units_series, dtype=float)
    mean_val = float(s.mean())
    if mean_val <= 0:
        return 0.0, "Z"

    std_val = float(s.std(ddof=1)) if len(s) > 1 else 0.0
    cv = float(std_val / mean_val)

    if cv <= 0.5:
        xyz = "X"
    elif cv <= 1.0:
        xyz = "Y"
    else:
        xyz = "Z"

    return cv, xyz


# ---------------------------------------------------------------------------
# 3. Product Lifecycle Stage
# ---------------------------------------------------------------------------


def compute_lifecycle_stage(
    weekly_units: pd.Series,
    all_train_weeks: list[pd.Timestamp],
) -> tuple[str, int]:
    """Compute lifecycle stage and weeks_since_last_sale.

    Rules:
    - new: first sale is within the last 12 training weeks.
    - inactive: no sales in the last 8 training weeks.
    - otherwise compare mean of last 8 weeks with mean of 8 weeks before:
        ratio = mean_last_8 / mean_prev_8
        growing if ratio >= 1.2
        declining if ratio <= 0.8
        else mature.
    - weeks_since_last_sale: open training weeks elapsed since last non-zero demand.
    """
    # Reindex over all open training weeks
    s = weekly_units.reindex(all_train_weeks, fill_value=0.0).astype(float)
    total_weeks = len(all_train_weeks)

    nonzero_mask = s > 0
    if not nonzero_mask.any():
        return "inactive", total_weeks

    last_sale_week = s[nonzero_mask].index[-1]
    last_sale_idx = all_train_weeks.index(last_sale_week)
    weeks_since_last_sale = total_weeks - 1 - last_sale_idx

    first_sale_week = s[nonzero_mask].index[0]
    first_sale_idx = all_train_weeks.index(first_sale_week)

    # 1. New: first sale within the last 12 training weeks
    if (total_weeks - first_sale_idx) <= 12:
        return "new", weeks_since_last_sale

    # 2. Inactive: no sales in the last 8 training weeks
    if weeks_since_last_sale >= 8:
        return "inactive", weeks_since_last_sale

    # 3. Trend ratio: last 8 weeks vs 8 weeks prior
    recent_8 = s.iloc[-8:]
    prev_8 = s.iloc[-16:-8]

    mean_recent = float(recent_8.mean())
    mean_prev = float(prev_8.mean())

    if mean_prev <= 0:
        stage = "growing" if mean_recent > 0 else "mature"
    else:
        ratio = mean_recent / mean_prev
        if ratio >= 1.2:
            stage = "growing"
        elif ratio <= 0.8:
            stage = "declining"
        else:
            stage = "mature"

    return stage, weeks_since_last_sale


# ---------------------------------------------------------------------------
# 4. Customer concentration & coverage from raw data
# ---------------------------------------------------------------------------


def compute_raw_customer_metrics(
    raw_df: pd.DataFrame,
    target_products: set[str],
    train_end: str = DEFAULT_TRAIN_END,
) -> pd.DataFrame:
    """Compute top3_customer_share and known_customer_coverage from raw file."""
    raw = raw_df.copy()
    raw["StockCode"] = raw["StockCode"].astype(str).str.strip().str.upper()
    raw = raw[~raw["StockCode"].map(_is_non_product_code)]
    raw = raw[raw["StockCode"].isin(target_products)]

    if "week_start" not in raw.columns:
        raw["week_start"] = week_start_of(raw["InvoiceDate"])

    # Filter to training period, non-cancellations, positive sales
    train_end_ts = pd.Timestamp(train_end)
    train_mask = (
        (raw["week_start"] >= "2010-12-06")
        & (raw["week_start"] < train_end_ts)
        & (raw["week_start"] != "2010-12-27")
        & (~raw["InvoiceNo"].astype(str).str.strip().str.upper().str.startswith("C"))
        & (raw["Quantity"] > 0)
        & (raw["UnitPrice"] > 0)
    )
    sales = raw[train_mask].copy()

    records: list[dict[str, object]] = []
    for pid in sorted(target_products):
        sub = sales[sales["StockCode"] == pid]
        if sub.empty:
            records.append({
                "product_id": pid,
                "top3_customer_share": 0.0,
                "known_customer_coverage": 0.0,
            })
            continue

        total_lines = len(sub)
        known_sub = sub[sub["CustomerID"].notna()]
        known_lines = len(known_sub)
        coverage = float(known_lines / total_lines) if total_lines > 0 else 0.0

        if known_sub.empty:
            top3_share = 0.0
        else:
            cust_vol = known_sub.groupby("CustomerID")["Quantity"].sum()
            top3_vol = float(cust_vol.nlargest(3).sum())
            total_known_vol = float(known_sub["Quantity"].sum())
            top3_share = float(top3_vol / total_known_vol) if total_known_vol > 0 else 0.0

        records.append({
            "product_id": pid,
            "top3_customer_share": round(top3_share, 4),
            "known_customer_coverage": round(coverage, 4),
        })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# 5. Catalog-wide demand pattern benchmarking
# ---------------------------------------------------------------------------


def compute_pattern_all_products(
    raw_path: str | Path = DEFAULT_RAW_PATH,
    train_end: str = DEFAULT_TRAIN_END,
    min_active_weeks: int = 10,
    out_path: str | Path | None = DEFAULT_PATTERN_ALL_OUTPUT,
) -> pd.DataFrame:
    """Compute demand pattern class for ALL products with >= min_active_weeks."""
    logger.info("Computing demand patterns for all products with >= %d active weeks...", min_active_weeks)
    raw = load_raw(raw_path) if not isinstance(raw_path, pd.DataFrame) else raw_path.copy()

    raw["StockCode"] = raw["StockCode"].astype(str).str.strip().str.upper()
    raw = raw[~raw["StockCode"].map(_is_non_product_code)]

    if "week_start" not in raw.columns:
        raw["week_start"] = week_start_of(raw["InvoiceDate"])

    train_end_ts = pd.Timestamp(train_end)
    train_mask = (
        (raw["week_start"] >= "2010-12-06")
        & (raw["week_start"] < train_end_ts)
        & (raw["week_start"] != "2010-12-27")
        & (~raw["InvoiceNo"].astype(str).str.strip().str.upper().str.startswith("C"))
        & (raw["Quantity"] > 0)
        & (raw["UnitPrice"] > 0)
    )
    sales = raw[train_mask]

    weekly = sales.groupby(["StockCode", "week_start"])["Quantity"].sum().reset_index()
    all_train_weeks = sorted(weekly["week_start"].unique().tolist())
    total_weeks = len(all_train_weeks)

    active_counts = weekly.groupby("StockCode")["week_start"].nunique()
    eligible = sorted(active_counts[active_counts >= min_active_weeks].index.tolist())

    records: list[dict[str, object]] = []
    for pid in eligible:
        sub = weekly[weekly["StockCode"] == pid]
        first_week = sub["week_start"].min()
        first_idx = all_train_weeks.index(first_week)
        n_weeks = total_weeks - first_idx
        active = len(sub)
        adi = float(n_weeks / active)

        units_vals = sub["Quantity"].values
        if len(units_vals) <= 1:
            cv2 = 0.0
        else:
            m = float(np.mean(units_vals))
            s = float(np.std(units_vals, ddof=1))
            cv2 = float((s / m) ** 2) if m > 0 else 0.0

        pclass = classify_demand_pattern(adi, cv2)
        records.append({
            "product_id": pid,
            "adi": round(adi, 4),
            "cv2": round(cv2, 4),
            "pattern_class": pclass,
        })

    df = pd.DataFrame(records)
    if out_path is not None:
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(p, index=False)
        logger.info("Saved %d products to %s", len(df), p)

    return df


# ---------------------------------------------------------------------------
# 6. Main Contract C11 builder
# ---------------------------------------------------------------------------


def build_product_intelligence(
    weekly: pd.DataFrame | str | Path = DEFAULT_WEEKLY_PATH,
    raw_path: str | Path | pd.DataFrame = DEFAULT_RAW_PATH,
    train_end: str = DEFAULT_TRAIN_END,
    enrichment: pd.DataFrame | str | Path | None = None,
    out_path: str | Path | None = DEFAULT_INTELLIGENCE_OUTPUT,
) -> pd.DataFrame:
    """Build product intelligence table (Contract C11).

    Parameters
    ----------
    weekly:
        C1 weekly demand DataFrame or file path.
    raw_path:
        Raw transactions DataFrame or file path.
    train_end:
        Training period cutoff date (exclusive, default '2011-10-10').
    enrichment:
        Optional C8 enrichment DataFrame or path. If None, loads DEFAULT_ENRICHMENT_PATH
        if present, or derives from raw/weekly.
    out_path:
        Optional path to save product_intelligence.csv.

    Returns
    -------
    pd.DataFrame:
        Contract C11 DataFrame with 14 columns for the Top 100 products.
    """
    if isinstance(weekly, (str, Path)):
        p = Path(weekly)
        w_df = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
    else:
        w_df = weekly.copy()

    w_df["week_start"] = pd.to_datetime(w_df["week_start"])
    w_df["product_id"] = w_df["product_id"].astype(str)

    # Top 100 products defined strictly by weekly DataFrame
    target_products = sorted(w_df["product_id"].unique().tolist())

    # Filter to training period only
    train_end_ts = pd.Timestamp(train_end)
    train_w = w_df[w_df["week_start"] < train_end_ts].copy()

    # Identify all open training weeks
    all_train_weeks = sorted(train_w["week_start"].unique().tolist())
    total_open_weeks = len(all_train_weeks)

    # 1. Description: most frequent description in training period
    desc_map: dict[str, str] = {}
    for pid, grp in train_w.groupby("product_id"):
        descs = grp["description"].dropna()
        if not descs.empty:
            mode_desc = str(descs.mode().iloc[0])
        else:
            mode_desc = str(pid)
        desc_map[pid] = mode_desc

    # 2. Revenue for ABC classification
    train_w["revenue"] = train_w["units"].astype(float) * train_w["avg_price"].astype(float)
    rev_by_prod = train_w.groupby("product_id")["revenue"].sum().reindex(target_products, fill_value=0.0)
    rev_shares, abc_classes = compute_abc_classes(rev_by_prod)

    # 3. Demand pattern, XYZ, lifecycle, and weeks_since_last_sale
    adi_map: dict[str, float] = {}
    cv2_map: dict[str, float] = {}
    pattern_map: dict[str, str] = {}
    xyz_map: dict[str, str] = {}
    lifecycle_map: dict[str, str] = {}
    weeks_since_map: dict[str, int] = {}

    for pid in target_products:
        p_rows = train_w[train_w["product_id"] == pid].sort_values("week_start")
        p_series = p_rows.set_index("week_start")["units"]

        # Grid reindexed to product's active series starting from first sale
        nonzero_mask = p_series > 0
        if nonzero_mask.any():
            first_sale_dt = p_series[nonzero_mask].index[0]
            first_idx = all_train_weeks.index(first_sale_dt)
            product_open_weeks = all_train_weeks[first_idx:]
            active_series = p_series.reindex(product_open_weeks, fill_value=0.0)
        else:
            active_series = p_series.reindex(all_train_weeks, fill_value=0.0)

        adi, cv2, pclass = compute_demand_pattern_metrics(active_series, total_open_weeks=len(active_series))
        adi_map[pid] = round(adi, 4)
        cv2_map[pid] = round(cv2, 4)
        pattern_map[pid] = pclass

        # XYZ
        _, xyz_class = compute_xyz_class(active_series)
        xyz_map[pid] = xyz_class

        # Lifecycle
        stage, w_since = compute_lifecycle_stage(p_series, all_train_weeks)
        lifecycle_map[pid] = stage
        weeks_since_map[pid] = int(w_since)

    # 4. Return rate and export share from enrichment_weekly
    enr_df: pd.DataFrame | None = None
    if enrichment is not None:
        if isinstance(enrichment, (str, Path)):
            ep = Path(enrichment)
            enr_df = pd.read_parquet(ep) if ep.suffix == ".parquet" else pd.read_csv(ep)
        else:
            enr_df = enrichment.copy()
    elif DEFAULT_ENRICHMENT_PATH.exists():
        enr_df = pd.read_parquet(DEFAULT_ENRICHMENT_PATH)

    return_rate_map: dict[str, float] = {}
    export_share_map: dict[str, float] = {}

    if enr_df is not None:
        enr_df["week_start"] = pd.to_datetime(enr_df["week_start"])
        enr_df["product_id"] = enr_df["product_id"].astype(str)
        enr_train = enr_df[enr_df["week_start"] < train_end_ts]

        for pid in target_products:
            p_enr = enr_train[enr_train["product_id"] == pid]
            gross = float(p_enr["gross_units"].sum()) if not p_enr.empty else 0.0
            cancelled = float(p_enr["cancelled_units"].sum()) if not p_enr.empty else 0.0
            exports = float(p_enr["export_units"].sum()) if not p_enr.empty else 0.0

            r_rate = (cancelled / gross) if gross > 0 else 0.0
            e_share = (exports / gross) if gross > 0 else 0.0

            return_rate_map[pid] = round(r_rate, 4)
            export_share_map[pid] = round(e_share, 4)
    else:
        # Fallback if no enrichment table is provided
        for pid in target_products:
            return_rate_map[pid] = 0.0
            export_share_map[pid] = 0.0

    # 5. Customer concentration and coverage from raw file
    if isinstance(raw_path, (str, Path)):
        raw_df = load_raw(raw_path)
    else:
        raw_df = raw_path.copy()

    cust_metrics = compute_raw_customer_metrics(raw_df, set(target_products), train_end=train_end)
    cust_metrics_indexed = cust_metrics.set_index("product_id")

    # Assemble final C11 DataFrame
    rows: list[dict[str, object]] = []
    for pid in target_products:
        row = {
            "product_id": pid,
            "description": desc_map.get(pid, pid),
            "adi": adi_map[pid],
            "cv2": cv2_map[pid],
            "pattern_class": pattern_map[pid],
            "abc_class": abc_classes.loc[pid],
            "xyz_class": xyz_map[pid],
            "lifecycle_stage": lifecycle_map[pid],
            "weeks_since_last_sale": weeks_since_map[pid],
            "return_rate": return_rate_map.get(pid, 0.0),
            "export_share": export_share_map.get(pid, 0.0),
            "top3_customer_share": float(cust_metrics_indexed.loc[pid, "top3_customer_share"]),
            "known_customer_coverage": float(cust_metrics_indexed.loc[pid, "known_customer_coverage"]),
            "revenue_share": round(float(rev_shares.loc[pid]), 4),
        }
        rows.append(row)

    out_df = pd.DataFrame(rows)[C11_COLUMNS]

    if out_path is not None:
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(p, index=False)
        logger.info("Saved product intelligence table to %s (%d rows)", p, len(out_df))

    return out_df


# ---------------------------------------------------------------------------
# 7. Basket Pairs / Bought-Together Analysis (Contract C12)
# ---------------------------------------------------------------------------


def build_basket_pairs(
    raw_path: str | Path | pd.DataFrame = DEFAULT_RAW_PATH,
    products: Sequence[str] | None = None,
    train_end: str = DEFAULT_TRAIN_END,
    min_support: int = 30,
    max_distinct_products: int = 100,
    out_path: str | Path | None = DEFAULT_BASKET_PAIRS_OUTPUT,
) -> pd.DataFrame:
    """Build bought-together companion pairs table (Contract C12).

    Parameters
    ----------
    raw_path:
        Path or DataFrame with raw transactions.
    products:
        Optional sequence of target product IDs. Defaults to Top 100 products
        from DEFAULT_WEEKLY_PATH.
    train_end:
        Cutoff timestamp (exclusive); only invoices before train_end are used.
    min_support:
        Minimum co-occurring invoice count (>= 30).
    max_distinct_products:
        Invoices with more than this number of distinct products are ignored
        as wholesale bulk orders (default 100).
    out_path:
        Optional output CSV path.

    Returns
    -------
    pd.DataFrame:
        Contract C12 DataFrame with columns:
        product_id, companion_id, companion_description, n_invoices, confidence, lift
    """
    logger.info("Building basket pairs (Contract C12)...")
    if isinstance(raw_path, (str, Path)):
        raw = load_raw(raw_path)
    else:
        raw = raw_path.copy()

    # 1. Clean product codes and format fields
    raw["StockCode"] = raw["StockCode"].astype(str).str.strip().str.upper()
    raw = raw[~raw["StockCode"].map(_is_non_product_code)]

    raw["InvoiceDate"] = pd.to_datetime(raw["InvoiceDate"])
    train_end_ts = pd.Timestamp(train_end)

    # 2. Filter valid sales before train_end
    is_cancel = raw["InvoiceNo"].astype(str).str.strip().str.upper().str.startswith("C")
    valid_mask = (
        (raw["InvoiceDate"] < train_end_ts)
        & (~is_cancel)
        & (raw["Quantity"] > 0)
        & (raw["UnitPrice"] > 0)
    )
    sales = raw[valid_mask].copy()

    # 3. Ignore wholesale mega-invoices with > max_distinct_products
    counts_all = sales.groupby("InvoiceNo")["StockCode"].nunique()
    ignored_invoices = set(counts_all[counts_all > max_distinct_products].index)
    n_ignored = len(ignored_invoices)
    print(f"Ignored {n_ignored} invoices with more than {max_distinct_products} distinct products.")
    logger.info("Ignored %d invoices with more than %d distinct products.", n_ignored, max_distinct_products)

    valid_sales = sales[~sales["InvoiceNo"].isin(ignored_invoices)]

    # 4. Resolve target products
    if products is None:
        if DEFAULT_WEEKLY_PATH.exists():
            w_df = pd.read_parquet(DEFAULT_WEEKLY_PATH)
            target_products = sorted(w_df["product_id"].unique().tolist())
        else:
            target_products = sorted(valid_sales["StockCode"].unique().tolist())
    else:
        target_products = sorted(list(products))

    target_set = set(target_products)
    top_sales = valid_sales[valid_sales["StockCode"].isin(target_set)]

    # 5. Skip invoices with fewer than 2 target products
    counts_target = top_sales.groupby("InvoiceNo")["StockCode"].nunique()
    qualifying_invoices = sorted(counts_target[counts_target >= 2].index.tolist())
    N = len(qualifying_invoices)
    print(f"Qualifying invoices (N): {N}")
    logger.info("Qualifying invoices (N): %d", N)

    if N == 0:
        empty_df = pd.DataFrame(columns=C12_COLUMNS)
        if out_path is not None:
            p = Path(out_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            empty_df.to_csv(p, index=False)
        return empty_df

    # 6. Build binary occurrence matrix X (sparse)
    baskets = top_sales[top_sales["InvoiceNo"].isin(set(qualifying_invoices))][
        ["InvoiceNo", "StockCode"]
    ].drop_duplicates()

    inv_map = {inv: i for i, inv in enumerate(qualifying_invoices)}
    prod_map = {p: i for i, p in enumerate(target_products)}

    row_idx = baskets["InvoiceNo"].map(inv_map).values
    col_idx = baskets["StockCode"].map(prod_map).values
    data = np.ones(len(row_idx), dtype=int)

    X = sp.csr_matrix((data, (row_idx, col_idx)), shape=(N, len(target_products)))
    cooc = (X.T @ X).toarray()
    diag = np.diag(cooc)

    # 7. Companion descriptions (mode description in training period)
    desc_map: dict[str, str] = {}
    if "Description" in valid_sales.columns:
        desc_grp = valid_sales[valid_sales["StockCode"].isin(target_set)].groupby("StockCode")["Description"]
        for p, d_series in desc_grp:
            d_clean = d_series.dropna()
            if not d_clean.empty:
                desc_map[p] = str(d_clean.mode().iloc[0])
            else:
                desc_map[p] = str(p)

    # 8. Compute confidence and lift; keep up to 3 companions ranked by lift
    pairs: list[dict[str, object]] = []
    for i, p_a in enumerate(target_products):
        n_a = int(diag[i])
        if n_a == 0:
            continue

        candidates: list[dict[str, object]] = []
        for j, p_b in enumerate(target_products):
            if i == j:
                continue
            n_ab = int(cooc[i, j])
            if n_ab >= min_support:
                n_b = int(diag[j])
                conf = float(n_ab / n_a)
                lift = float(conf / (n_b / N))
                candidates.append({
                    "product_id": p_a,
                    "companion_id": p_b,
                    "companion_description": desc_map.get(p_b, p_b),
                    "n_invoices": n_ab,
                    "confidence": round(conf, 4),
                    "lift": round(lift, 4),
                })

        # Rank candidates by lift desc, then n_invoices desc
        candidates.sort(key=lambda x: (x["lift"], x["n_invoices"]), reverse=True)
        pairs.extend(candidates[:3])

    out_df = pd.DataFrame(pairs, columns=C12_COLUMNS)
    if out_path is not None:
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(p, index=False)
        logger.info("Saved %d basket pairs to %s", len(out_df), p)

    return out_df


# ---------------------------------------------------------------------------
# 8. CLI entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI runner: python -m src.intelligence."""
    parser = argparse.ArgumentParser(description="Build product intelligence tables (Contracts C11 & C12).")
    parser.add_argument("--weekly", type=str, default=str(DEFAULT_WEEKLY_PATH))
    parser.add_argument("--raw", type=str, default=str(DEFAULT_RAW_PATH))
    parser.add_argument("--enrichment", type=str, default=str(DEFAULT_ENRICHMENT_PATH))
    parser.add_argument("--train-end", type=str, default=DEFAULT_TRAIN_END)
    parser.add_argument("--out", type=str, default=str(DEFAULT_INTELLIGENCE_OUTPUT))
    parser.add_argument("--pattern-all-out", type=str, default=str(DEFAULT_PATTERN_ALL_OUTPUT))
    parser.add_argument("--basket-pairs-out", type=str, default=str(DEFAULT_BASKET_PAIRS_OUTPUT))

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print(" SHELFWATCH: BUILDING PRODUCT INTELLIGENCE (Contract C11)")
    print("=" * 60)

    # 1. Build C11 table for Top 100 products
    df_top100 = build_product_intelligence(
        weekly=args.weekly,
        raw_path=args.raw,
        train_end=args.train_end,
        enrichment=args.enrichment,
        out_path=args.out,
    )
    print(f"-> Generated {len(df_top100)} rows in {args.out}")

    # 2. Build benchmark demand pattern table for all products (>= 10 active training weeks)
    df_all = compute_pattern_all_products(
        raw_path=args.raw,
        train_end=args.train_end,
        min_active_weeks=10,
        out_path=args.pattern_all_out,
    )
    print(f"-> Generated {len(df_all)} rows in {args.pattern_all_out}")

    # 3. Print distributions
    print("\n" + "-" * 60)
    print(" DEMAND PATTERN DISTRIBUTION: TOP 100 vs ALL PRODUCTS")
    print("-" * 60)
    top_counts = df_top100["pattern_class"].value_counts()
    top_pcts = df_top100["pattern_class"].value_counts(normalize=True) * 100

    all_counts = df_all["pattern_class"].value_counts()
    all_pcts = df_all["pattern_class"].value_counts(normalize=True) * 100

    patterns = ["smooth", "erratic", "intermittent", "lumpy"]
    print(f"{'Pattern Class':<15} | {'Top 100 Count':>13} | {'Top 100 %':>10} | {'All Count':>11} | {'All %':>8}")
    print("-" * 68)
    for pat in patterns:
        tc = top_counts.get(pat, 0)
        tp = top_pcts.get(pat, 0.0)
        ac = all_counts.get(pat, 0)
        ap = all_pcts.get(pat, 0.0)
        print(f"{pat:<15} | {tc:>13} | {tp:>9.1f}% | {ac:>11} | {ap:>7.1f}%")

    print("\n" + "-" * 60)
    print(" TOP 100 ABC & XYZ DISTRIBUTION")
    print("-" * 60)
    print("ABC Classes:")
    for c, cnt in df_top100["abc_class"].value_counts().items():
        print(f"  Class {c}: {cnt} products ({cnt / len(df_top100):.1%})")

    print("\nXYZ Classes:")
    for c, cnt in df_top100["xyz_class"].value_counts().items():
        print(f"  Class {c}: {cnt} products ({cnt / len(df_top100):.1%})")

    print("\nLifecycle Stages:")
    for s, cnt in df_top100["lifecycle_stage"].value_counts().items():
        print(f"  Stage {s}: {cnt} products ({cnt / len(df_top100):.1%})")

    # 4. Build C12 basket pairs
    print("\n" + "=" * 60)
    print(" SHELFWATCH: BUILDING BOUGHT-TOGETHER PAIRS (Contract C12)")
    print("=" * 60)
    df_baskets = build_basket_pairs(
        raw_path=args.raw,
        products=df_top100["product_id"].tolist(),
        train_end=args.train_end,
        min_support=30,
        max_distinct_products=100,
        out_path=args.basket_pairs_out,
    )
    print(f"-> Generated {len(df_baskets)} rows in {args.basket_pairs_out}")

    # Read directly from file to verify and print 10 strongest pairs
    pairs_on_disk = pd.read_csv(args.basket_pairs_out)
    top10_pairs = pairs_on_disk.sort_values(by=["lift", "n_invoices"], ascending=[False, False]).head(10)

    print("\n" + "-" * 60)
    print(" TOP 10 STRONGEST BOUGHT-TOGETHER PAIRS (READ FROM FILE)")
    print("-" * 60)
    print(f"{'Product':<10} | {'Companion':<10} | {'Companion Description':<32} | {'Invoices':>8} | {'Conf':>7} | {'Lift':>7}")
    print("-" * 88)
    for _, r in top10_pairs.iterrows():
        desc = str(r['companion_description'])[:32]
        print(f"{r['product_id']:<10} | {r['companion_id']:<10} | {desc:<32} | {int(r['n_invoices']):>8} | {r['confidence']:>7.4f} | {r['lift']:>7.4f}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
