"""Generate deterministic mock data and dashboard JSON files in data/mock/.

All outputs conform to CONTRACTS.md (C1, C1b, C2, C3, C4, C7) and docs/SRS.md
sections 4.4 and 7. Uses fixed random seed 42 for reproducibility.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm

SEED = 42
MOCK_DIR = Path("data/mock")

PRODUCTS: list[tuple[str, str, float, float]] = [
    ("85123A", "WHITE HANGING HEART T-LIGHT HOLDER", 420.0, 2.55),
    ("85099B", "JUMBO BAG RED RETROSPOT", 520.0, 1.95),
    ("84077", "WORLD WAR 2 GLIDERS ASSTD DESIGNS", 380.0, 0.29),
    ("21212", "PACK OF 72 RETROSPOT CAKE CASES", 340.0, 0.55),
    ("22197", "POPCORN HOLDER", 290.0, 0.85),
    ("84879", "ASSORTED COLOUR BIRD ORNAMENT", 260.0, 1.69),
    ("22423", "REGENCY CAKESTAND 3 TIER", 180.0, 12.75),
    ("47566", "PARTY BUNTING", 145.0, 4.95),
    ("20725", "LUNCH BAG RED RETROSPOT", 110.0, 1.65),
    ("22629", "SPACEBOY LUNCH BOX", 75.0, 1.95),
]

MODELS: list[tuple[str, float, float]] = [
    ("naive", 0.18, 0.02),
    ("moving_avg_4", 0.15, -0.01),
    ("ridge", 0.11, -0.02),
    ("random_forest", 0.12, -0.05),
    ("lightgbm", 0.09, -0.01),
]

BEST_MODEL = "lightgbm"
CLOSED_WEEK = "2010-12-27"
FORECAST_TARGET_WEEK = "2011-12-05"
N_WEEKS = 20
TEST_WEEKS = 8
VAL_WEEKS = 4

DISCLAIMER_TEXT = (
    "Inventory levels are simulated. The dataset contains sales only, "
    "not stock-on-hand. Risk flags show how the rule behaves under the "
    "stated assumptions, not real stock positions."
)


def build_mock_weekly_demand(rng: np.random.Generator) -> pd.DataFrame:
    """Create a 10-product x 20-Monday-week DataFrame conforming to contract C1."""
    weeks = pd.date_range(end="2011-11-28", periods=N_WEEKS, freq="W-MON")
    rows: list[dict[str, Any]] = []

    for p_idx, (product_id, description, base_units, base_price) in enumerate(PRODUCTS):
        # Realistic Q4 retail upward drift; a few seasonal products ramp steeper
        end_mult = 1.85 if p_idx in (1, 9) else 1.25
        start_mult = 0.65 if p_idx in (1, 9) else 0.85
        trend = np.linspace(start_mult, end_mult, N_WEEKS)
        for idx, week_start in enumerate(weeks):
            raw_units = base_units * trend[idx] + rng.normal(0.0, base_units * 0.15)
            units = int(max(0, round(raw_units)))
            orders = int(max(1, round(units / rng.uniform(8.0, 15.0)))) if units > 0 else 0
            customers = int(max(1, round(orders * rng.uniform(0.65, 0.95)))) if orders > 0 else 0
            avg_price = round(float(base_price * rng.uniform(0.95, 1.05)), 2)
            rows.append(
                {
                    "product_id": str(product_id),
                    "description": str(description),
                    "week_start": week_start,
                    "units": units,
                    "orders": orders,
                    "customers": customers,
                    "avg_price": avg_price,
                }
            )

    df = pd.DataFrame(rows)
    df = df.sort_values(["product_id", "week_start"]).reset_index(drop=True)
    df["units"] = df["units"].astype("int64")
    df["orders"] = df["orders"].astype("int64")
    df["customers"] = df["customers"].astype("int64")
    df["avg_price"] = df["avg_price"].astype("float64")
    return df


def build_mock_closed_weeks() -> pd.DataFrame:
    """Create closed_weeks DataFrame conforming to contract C1b."""
    return pd.DataFrame({"week_start": [CLOSED_WEEK]})


def build_mock_predictions(
    weekly_df: pd.DataFrame, rng: np.random.Generator
) -> pd.DataFrame:
    """Create predictions DataFrame conforming to contract C2."""
    all_weeks = sorted(weekly_df["week_start"].unique())
    train_weeks = all_weeks[:-TEST_WEEKS]
    val_weeks = set(train_weeks[-VAL_WEEKS:])
    test_weeks = set(all_weeks[-TEST_WEEKS:])

    rows: list[dict[str, Any]] = []
    for model_name, rel_noise, rel_bias in MODELS:
        subset = weekly_df[weekly_df["week_start"].isin(val_weeks | test_weeks)]
        for row in subset.itertuples(index=False):
            actual = float(row.units)
            split = "val" if row.week_start in val_weeks else "test"
            noise_scale = max(5.0, actual * rel_noise)
            raw_pred = actual * (1.0 + rel_bias) + rng.normal(0.0, noise_scale)
            predicted = round(float(max(0.0, raw_pred)), 2)
            rows.append(
                {
                    "product_id": str(row.product_id),
                    "week_start": pd.Timestamp(row.week_start).strftime("%Y-%m-%d"),
                    "actual": actual,
                    "predicted": predicted,
                    "model_name": model_name,
                    "split": split,
                }
            )

    return pd.DataFrame(
        rows,
        columns=["product_id", "week_start", "actual", "predicted", "model_name", "split"],
    )


def _compute_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """Compute MAE, RMSE, WAPE, and bias without MAPE."""
    errors = predicted - actual
    abs_errors = np.abs(errors)
    total_actual = float(np.sum(actual))
    mae = float(np.mean(abs_errors))
    rmse = float(np.sqrt(np.mean(errors**2)))
    wape = float(np.sum(abs_errors) / total_actual) if total_actual > 0 else 0.0
    bias = float(np.mean(errors))
    return {
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "WAPE": round(wape, 4),
        "bias": round(bias, 4),
    }


def build_mock_metrics(preds_df: pd.DataFrame) -> pd.DataFrame:
    """Create metrics DataFrame conforming to contract C3."""
    rows: list[dict[str, Any]] = []
    for (model_name, split), group in preds_df.groupby(
        ["model_name", "split"], sort=False
    ):
        stats = _compute_metrics(
            group["actual"].to_numpy(dtype=float),
            group["predicted"].to_numpy(dtype=float),
        )
        rows.append({"model_name": model_name, "split": split, **stats})

    return pd.DataFrame(
        rows, columns=["model_name", "split", "MAE", "RMSE", "WAPE", "bias"]
    )


def build_mock_next_week_forecast(
    weekly_df: pd.DataFrame, preds_df: pd.DataFrame, rng: np.random.Generator
) -> pd.DataFrame:
    """Create next_week_forecast DataFrame conforming to contract C4."""
    all_weeks = sorted(weekly_df["week_start"].unique())
    train_weeks = set(all_weeks[:-TEST_WEEKS])

    best_val = preds_df[
        (preds_df["model_name"] == BEST_MODEL) & (preds_df["split"] == "val")
    ]
    rows: list[dict[str, Any]] = []

    for product_id, group in weekly_df.groupby("product_id", sort=True):
        train_group = group[group["week_start"].isin(train_weeks)]
        avg_train = float(train_group["units"].mean())

        recent_mean = float(group.sort_values("week_start")["units"].tail(4).mean())
        forecast_weekly = round(
            float(max(0.0, recent_mean * rng.uniform(0.95, 1.12))), 2
        )

        val_group = best_val[best_val["product_id"] == product_id]
        val_errors = (val_group["predicted"] - val_group["actual"]).to_numpy(dtype=float)
        sigma_error = float(np.std(val_errors, ddof=1)) if len(val_errors) > 1 else 10.0

        p10 = round(float(max(0.0, forecast_weekly - 1.28 * sigma_error)), 2)
        p90 = round(float(forecast_weekly + 1.28 * sigma_error), 2)
        p10 = min(p10, forecast_weekly)
        p90 = max(p90, forecast_weekly)

        last_price = float(group.sort_values("week_start")["avg_price"].iloc[-1])

        rows.append(
            {
                "product_id": str(product_id),
                "week_start": FORECAST_TARGET_WEEK,
                "forecast_weekly": forecast_weekly,
                "forecast_p10": p10,
                "forecast_p90": p90,
                "sigma_error": round(sigma_error, 2),
                "avg_weekly_units_train": round(avg_train, 2),
                "last_avg_price": round(last_price, 2),
            }
        )

    return pd.DataFrame(
        rows,
        columns=[
            "product_id",
            "week_start",
            "forecast_weekly",
            "forecast_p10",
            "forecast_p90",
            "sigma_error",
            "avg_weekly_units_train",
            "last_avg_price",
        ],
    )


def build_mock_enrichment_weekly(
    weekly_df: pd.DataFrame, rng: np.random.Generator
) -> pd.DataFrame:
    """Create enrichment_weekly DataFrame conforming to contract C8."""
    rows: list[dict[str, Any]] = []
    for row in weekly_df.itertuples(index=False):
        gross = int(row.units)
        cancelled = int(max(0, round(gross * float(rng.uniform(0.01, 0.08)))))
        export_units = int(max(0, round(gross * float(rng.uniform(0.05, 0.20)))))
        guest_units = int(max(0, round(gross * float(rng.uniform(0.10, 0.30)))))
        export_units = min(export_units, gross)
        guest_units = min(guest_units, gross)
        rows.append(
            {
                "product_id": str(row.product_id),
                "week_start": pd.Timestamp(row.week_start),
                "gross_units": gross,
                "cancelled_units": cancelled,
                "export_units": export_units,
                "guest_units": guest_units,
            }
        )

    df = pd.DataFrame(rows)
    df["gross_units"] = df["gross_units"].astype("int64")
    df["cancelled_units"] = df["cancelled_units"].astype("int64")
    df["export_units"] = df["export_units"].astype("int64")
    df["guest_units"] = df["guest_units"].astype("int64")
    return df


def build_mock_calendar_weekly() -> pd.DataFrame:
    """Create calendar_weekly DataFrame conforming to contract C9 (105 weeks)."""
    weeks = pd.date_range(start="2009-12-07", end="2011-12-05", freq="W-MON")
    # Verified UK bank holiday Mondays or weeks with bank holidays in 2009-12-07 to 2011-12-05
    # Key dates: 2011-04-22 (Good Fri), 2011-04-25 (Easter Mon), 2011-04-29 (Royal Wedding), 2011-05-02 (Early May)
    holiday_map: dict[str, int] = {
        "2009-12-28": 2,  # Christmas/Boxing Day substitute
        "2010-01-04": 1,  # New Year substitute
        "2010-04-05": 1,  # Easter Monday
        "2010-05-03": 1,  # Early May
        "2010-05-31": 1,  # Spring
        "2010-08-30": 1,  # Summer
        "2010-12-27": 2,  # Christmas/Boxing Day substitute
        "2011-01-03": 1,  # New Year substitute
        "2011-04-18": 1,  # Good Friday 2011-04-22
        "2011-04-25": 2,  # Easter Monday 2011-04-25 + Royal Wedding 2011-04-29
        "2011-05-02": 1,  # Early May bank holiday 2011-05-02
        "2011-05-30": 1,  # Spring bank holiday
        "2011-08-29": 1,  # Summer bank holiday
    }

    rows: list[dict[str, Any]] = []
    for w in weeks:
        w_str = w.strftime("%Y-%m-%d")
        h_count = holiday_map.get(w_str, 0)
        # 6 trading days (Mon-Fri + Sun, no Saturday sales) minus bank holidays
        trading_days = max(1, 6 - h_count)

        # Christmas week contains Dec 24, 25, or 26
        week_days = pd.date_range(start=w, periods=7)
        is_xmas = 1 if any((d.month == 12 and d.day in (24, 25, 26)) for d in week_days) else 0

        rows.append(
            {
                "week_start": w_str,
                "uk_bank_holiday_count": int(h_count),
                "trading_days_expected": int(trading_days),
                "is_christmas_week": int(is_xmas),
            }
        )

    df = pd.DataFrame(rows)
    df["uk_bank_holiday_count"] = df["uk_bank_holiday_count"].astype("int64")
    df["trading_days_expected"] = df["trading_days_expected"].astype("int64")
    df["is_christmas_week"] = df["is_christmas_week"].astype("int64")
    return df


def build_mock_prior_year_weekly(
    weekly_df: pd.DataFrame, rng: np.random.Generator
) -> pd.DataFrame:
    """Create prior_year_weekly DataFrame conforming to contract C10."""
    # Prior year weeks before 2010-12-06, excluding closure week 2009-12-28
    prior_weeks = [
        w for w in pd.date_range(start="2010-01-04", end="2010-05-17", freq="W-MON")
        if w.strftime("%Y-%m-%d") != "2009-12-28"
    ]
    # 9 out of 10 products have prior-year rows (SKU 22629 is new with 0 prior rows)
    eligible_products = [p for p in PRODUCTS if p[0] != "22629"]

    rows: list[dict[str, Any]] = []
    for product_id, _, base_units, _ in eligible_products:
        for w in prior_weeks:
            raw_units = base_units * 0.9 + rng.normal(0.0, base_units * 0.15)
            units = int(max(0, round(raw_units)))
            orders = int(max(1, round(units / 10.0))) if units > 0 else 0
            rows.append(
                {
                    "product_id": str(product_id),
                    "week_start": pd.Timestamp(w),
                    "units": units,
                    "orders": orders,
                }
            )

    df = pd.DataFrame(rows).sort_values(["product_id", "week_start"]).reset_index(drop=True)
    df["units"] = df["units"].astype("int64")
    df["orders"] = df["orders"].astype("int64")
    return df


def build_mock_product_intelligence(
    weekly_df: pd.DataFrame, rng: np.random.Generator
) -> pd.DataFrame:
    """Create product_intelligence DataFrame conforming to contract C11."""
    # Syntetos-Boylan quadrant combinations across the 10 products
    # smooth: ADI < 1.32 & CV2 < 0.49
    # erratic: ADI < 1.32 & CV2 >= 0.49
    # intermittent: ADI >= 1.32 & CV2 < 0.49
    # lumpy: ADI >= 1.32 & CV2 >= 0.49
    adi_vals = [1.05, 1.10, 1.15, 1.20, 1.25, 1.35, 1.40, 1.05, 1.12, 1.38]
    cv2_vals = [0.25, 0.35, 0.45, 0.55, 0.30, 0.60, 0.20, 0.40, 0.50, 0.65]
    abc_vals = ["A", "A", "A", "A", "B", "B", "B", "C", "C", "C"]
    xyz_vals = ["X", "X", "Y", "Y", "X", "Z", "X", "Y", "Z", "Z"]
    stages = ["mature", "growing", "mature", "mature", "growing", "mature", "mature", "declining", "mature", "new"]
    weeks_inactive = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]

    desc_map = dict(weekly_df.drop_duplicates("product_id").set_index("product_id")["description"])

    # Compute mock revenue shares
    rev_weights = [p[2] * p[3] for p in PRODUCTS]
    tot_rev = sum(rev_weights)
    rev_shares = [round(w / tot_rev, 4) for w in rev_weights]
    rev_shares[-1] = round(1.0 - sum(rev_shares[:-1]), 4)

    rows: list[dict[str, Any]] = []
    for idx, (product_id, _, _, _) in enumerate(PRODUCTS):
        adi = float(adi_vals[idx])
        cv2 = float(cv2_vals[idx])
        if adi < 1.32 and cv2 < 0.49:
            pattern = "smooth"
        elif adi < 1.32 and cv2 >= 0.49:
            pattern = "erratic"
        elif adi >= 1.32 and cv2 < 0.49:
            pattern = "intermittent"
        else:
            pattern = "lumpy"

        ret_rate = round(float(rng.uniform(0.015, 0.055)), 4)
        exp_share = round(float(rng.uniform(0.08, 0.22)), 4)
        top3_share = round(float(rng.uniform(0.20, 0.45)), 4)
        known_cov = round(float(rng.uniform(0.70, 0.85)), 4)

        rows.append(
            {
                "product_id": str(product_id),
                "description": str(desc_map.get(product_id, "")),
                "adi": adi,
                "cv2": cv2,
                "pattern_class": pattern,
                "abc_class": abc_vals[idx],
                "xyz_class": xyz_vals[idx],
                "lifecycle_stage": stages[idx],
                "weeks_since_last_sale": int(weeks_inactive[idx]),
                "return_rate": ret_rate,
                "export_share": exp_share,
                "top3_customer_share": top3_share,
                "known_customer_coverage": known_cov,
                "revenue_share": float(rev_shares[idx]),
            }
        )

    df = pd.DataFrame(rows)
    df["weeks_since_last_sale"] = df["weeks_since_last_sale"].astype("int64")
    return df


def build_mock_basket_pairs(
    weekly_df: pd.DataFrame, rng: np.random.Generator
) -> pd.DataFrame:
    """Create basket_pairs DataFrame conforming to contract C12."""
    desc_map = dict(weekly_df.drop_duplicates("product_id").set_index("product_id")["description"])
    # Realistic pairs among mock products
    pair_defs = [
        ("85123A", "85099B", 65, 0.38, 2.15),
        ("85123A", "22423", 42, 0.28, 1.85),
        ("85099B", "20725", 78, 0.45, 3.10),
        ("85099B", "85123A", 65, 0.35, 2.15),
        ("21212", "22197", 52, 0.42, 2.60),
        ("22197", "21212", 52, 0.40, 2.60),
        ("22423", "47566", 38, 0.32, 2.20),
        ("47566", "22423", 38, 0.30, 2.20),
        ("84879", "85123A", 35, 0.26, 1.75),
        ("20725", "85099B", 78, 0.52, 3.10),
    ]
    rows: list[dict[str, Any]] = []
    for pid, cid, n_inv, conf, lift in pair_defs:
        rows.append(
            {
                "product_id": str(pid),
                "companion_id": str(cid),
                "companion_description": str(desc_map.get(cid, "")),
                "n_invoices": int(n_inv),
                "confidence": float(conf),
                "lift": float(lift),
            }
        )

    df = pd.DataFrame(rows)
    df["n_invoices"] = df["n_invoices"].astype("int64")
    df["confidence"] = df["confidence"].astype("float64")
    df["lift"] = df["lift"].astype("float64")
    return df


def _compute_mock_risk(
    forecast_df: pd.DataFrame,
    lead_time_weeks: int = 1,
    service_level: float = 0.95,
    initial_stock_weeks: float = 3.0,
    seed: int = SEED,
) -> pd.DataFrame:
    """Compute simulated inventory risk using SRS section 6.2 formulas."""
    ordered = forecast_df.sort_values("product_id").reset_index(drop=True)
    rng = np.random.default_rng(seed)
    noise = rng.uniform(0.5, 1.5, size=len(ordered))
    z = float(norm.ppf(service_level))

    items: list[dict[str, Any]] = []
    for idx, row in enumerate(ordered.itertuples(index=False)):
        on_hand = float(initial_stock_weeks * row.avg_weekly_units_train * noise[idx])
        forecast_lt = float(row.forecast_weekly * lead_time_weeks)
        safety_stock = float(z * row.sigma_error * math.sqrt(lead_time_weeks))
        reorder_point = float(forecast_lt + safety_stock)
        days_of_cover = (
            min(999.0, on_hand / (row.forecast_weekly / 7.0))
            if row.forecast_weekly > 0
            else 999.0
        )

        if on_hand < forecast_lt:
            risk_level = "HIGH"
        elif on_hand < reorder_point:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        items.append(
            {
                "product_id": str(row.product_id),
                "description": str(row.description),
                "forecast_weekly": round(float(row.forecast_weekly), 1),
                "forecast_lead_time": round(forecast_lt, 1),
                "on_hand": round(on_hand, 1),
                "safety_stock": round(safety_stock, 1),
                "reorder_point": round(reorder_point, 1),
                "days_of_cover": round(days_of_cover, 1),
                "risk_level": risk_level,
            }
        )

    return pd.DataFrame(items)


def _build_error_analysis(
    weekly_df: pd.DataFrame, preds_df: pd.DataFrame
) -> dict[str, Any]:
    """Build error_analysis.json payload matching SRS 5.3 and 7.1."""
    test_df = preds_df[
        (preds_df["model_name"] == BEST_MODEL) & (preds_df["split"] == "test")
    ].copy()
    desc_map = (
        weekly_df.drop_duplicates("product_id")
        .set_index("product_id")["description"]
        .to_dict()
    )

    all_weeks = sorted(weekly_df["week_start"].unique())
    train_weeks = set(all_weeks[:-TEST_WEEKS])
    train_means = (
        weekly_df[weekly_df["week_start"].isin(train_weeks)]
        .groupby("product_id")["units"]
        .mean()
    )
    segment_labels = pd.qcut(train_means, q=3, labels=["low", "medium", "high"]).to_dict()
    test_df["segment"] = test_df["product_id"].map(segment_labels)

    by_segment: list[dict[str, Any]] = []
    for seg in ["low", "medium", "high"]:
        grp = test_df[test_df["segment"] == seg]
        stats = _compute_metrics(
            grp["actual"].to_numpy(dtype=float),
            grp["predicted"].to_numpy(dtype=float),
        )
        by_segment.append(
            {
                "segment": seg,
                "n_products": int(grp["product_id"].nunique()),
                **stats,
            }
        )

    by_week: list[dict[str, Any]] = []
    for week_start, grp in test_df.groupby("week_start", sort=True):
        stats = _compute_metrics(
            grp["actual"].to_numpy(dtype=float),
            grp["predicted"].to_numpy(dtype=float),
        )
        by_week.append({"week_start": str(week_start), **stats})

    by_product: list[dict[str, Any]] = []
    for product_id, grp in test_df.groupby("product_id", sort=True):
        stats = _compute_metrics(
            grp["actual"].to_numpy(dtype=float),
            grp["predicted"].to_numpy(dtype=float),
        )
        by_product.append(
            {
                "product_id": str(product_id),
                "description": str(desc_map.get(product_id, "")),
                "segment": str(segment_labels.get(product_id, "medium")),
                "mean_actual": round(float(grp["actual"].mean()), 2),
                "mean_predicted": round(float(grp["predicted"].mean()), 2),
                **stats,
            }
        )
    by_product.sort(key=lambda item: item["WAPE"])

    diff = test_df["predicted"] - test_df["actual"]
    over_mask = diff > 0
    under_mask = diff < 0
    exact_mask = diff == 0
    total = max(1, len(test_df))
    over_under = {
        "n_over": int(over_mask.sum()),
        "n_under": int(under_mask.sum()),
        "n_exact": int(exact_mask.sum()),
        "over_pct": round(float(over_mask.sum() / total * 100.0), 1),
        "under_pct": round(float(under_mask.sum() / total * 100.0), 1),
        "mean_over_error": round(float(diff[over_mask].mean()), 2)
        if over_mask.any()
        else 0.0,
        "mean_under_error": round(float(diff[under_mask].mean()), 2)
        if under_mask.any()
        else 0.0,
    }

    return {
        "model_name": BEST_MODEL,
        "by_segment": by_segment,
        "by_week": by_week,
        "by_product": by_product,
        "over_under": over_under,
    }


def _write_json(path: Path, payload: Any) -> None:
    """Write deterministic formatted JSON to disk."""
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def generate_mocks(output_dir: Path = MOCK_DIR) -> None:
    """Generate all mock parquet, CSV, and dashboard JSON files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    weekly_df = build_mock_weekly_demand(rng)
    closed_df = build_mock_closed_weeks()
    preds_df = build_mock_predictions(weekly_df, rng)
    metrics_df = build_mock_metrics(preds_df)
    forecast_df = build_mock_next_week_forecast(weekly_df, preds_df, rng)
    enrichment_df = build_mock_enrichment_weekly(weekly_df, rng)
    calendar_df = build_mock_calendar_weekly()
    prior_year_df = build_mock_prior_year_weekly(weekly_df, rng)
    intelligence_df = build_mock_product_intelligence(weekly_df, rng)
    basket_df = build_mock_basket_pairs(weekly_df, rng)

    weekly_df.to_parquet(output_dir / "weekly_demand.parquet", index=False)
    closed_df.to_csv(output_dir / "closed_weeks.csv", index=False)
    preds_df.to_csv(output_dir / "predictions.csv", index=False)
    metrics_df.to_csv(output_dir / "metrics.csv", index=False)
    forecast_df.to_csv(output_dir / "next_week_forecast.csv", index=False)
    enrichment_df.to_parquet(output_dir / "enrichment_weekly.parquet", index=False)
    calendar_df.to_csv(output_dir / "calendar_weekly.csv", index=False)
    prior_year_df.to_parquet(output_dir / "prior_year_weekly.parquet", index=False)
    intelligence_df.to_csv(output_dir / "product_intelligence.csv", index=False)
    basket_df.to_csv(output_dir / "basket_pairs.csv", index=False)

    # Dashboard mock JSON files (matching CONTRACTS.md C7 and SRS section 7)
    all_weeks = sorted(weekly_df["week_start"].unique())
    train_weeks = set(all_weeks[:-TEST_WEEKS])
    desc_df = weekly_df.drop_duplicates("product_id")[["product_id", "description"]]
    train_totals = (
        weekly_df[weekly_df["week_start"].isin(train_weeks)]
        .groupby("product_id")["units"]
        .sum()
        .reset_index(name="total_units")
    )
    products_df = desc_df.merge(train_totals, on="product_id").sort_values(
        ["total_units", "product_id"], ascending=[False, True]
    )
    products_payload = [
        {
            "product_id": str(r.product_id),
            "description": str(r.description),
            "total_units": int(r.total_units),
        }
        for r in products_df.itertuples(index=False)
    ]
    _write_json(output_dir / "products.json", products_payload)

    best_test = preds_df[
        (preds_df["model_name"] == BEST_MODEL) & (preds_df["split"] == "test")
    ]
    forecast_map = forecast_df.set_index("product_id").to_dict("index")
    for product_id, group in weekly_df.groupby("product_id", sort=True):
        group_sorted = group.sort_values("week_start")
        pred_group = best_test[best_test["product_id"] == product_id].sort_values(
            "week_start"
        )
        fc_row = forecast_map[product_id]
        forecast_payload = {
            "product_id": str(product_id),
            "description": str(group_sorted["description"].iloc[0]),
            "history": [
                {
                    "week_start": pd.Timestamp(r.week_start).strftime("%Y-%m-%d"),
                    "units": int(r.units),
                }
                for r in group_sorted.itertuples(index=False)
            ],
            "test_predictions": [
                {
                    "week_start": str(r.week_start),
                    "actual": float(r.actual),
                    "predicted": float(r.predicted),
                }
                for r in pred_group.itertuples(index=False)
            ],
            "next_week_forecast": {
                "week_start": str(fc_row["week_start"]),
                "forecast_weekly": float(fc_row["forecast_weekly"]),
            },
        }
        _write_json(output_dir / f"forecast_{product_id}.json", forecast_payload)

    forecast_with_desc = forecast_df.merge(desc_df, on="product_id")
    risk_df = _compute_mock_risk(forecast_with_desc)
    counts = risk_df["risk_level"].value_counts()
    risk_payload = {
        "params": {
            "lead_time_weeks": 1,
            "service_level": 0.95,
            "initial_stock_weeks": 3.0,
            "seed": SEED,
        },
        "summary": {
            "n_high": int(counts.get("HIGH", 0)),
            "n_medium": int(counts.get("MEDIUM", 0)),
            "n_low": int(counts.get("LOW", 0)),
            "avg_days_of_cover": round(float(risk_df["days_of_cover"].mean()), 1),
        },
        "items": risk_df.to_dict("records"),
    }
    _write_json(output_dir / "risk.json", risk_payload)

    sensitivity_payload: list[dict[str, Any]] = []
    for lead_time in (1, 2, 3, 4):
        for service_level in (0.90, 0.95, 0.99):
            df_s = _compute_mock_risk(
                forecast_with_desc,
                lead_time_weeks=lead_time,
                service_level=service_level,
            )
            c_s = df_s["risk_level"].value_counts()
            sensitivity_payload.append(
                {
                    "lead_time_weeks": lead_time,
                    "service_level": service_level,
                    "n_high": int(c_s.get("HIGH", 0)),
                    "n_medium": int(c_s.get("MEDIUM", 0)),
                    "n_low": int(c_s.get("LOW", 0)),
                }
            )
    _write_json(output_dir / "sensitivity.json", sensitivity_payload)

    _write_json(output_dir / "metrics.json", metrics_df.to_dict("records"))
    _write_json(
        output_dir / "error_analysis.json",
        _build_error_analysis(weekly_df, preds_df),
    )
    _write_json(output_dir / "disclaimer.json", {"text": DISCLAIMER_TEXT})


if __name__ == "__main__":
    generate_mocks()
    print(f"Mock files generated in {MOCK_DIR}")

