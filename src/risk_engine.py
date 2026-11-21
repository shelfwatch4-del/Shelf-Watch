"""Risk and replenishment engine for inventory stockout analysis.

Implements SRS sections 6.1 to 6.3, FR-R-1 to FR-R-4, FR-R-10, FR-R-11,
and FR-R-13.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm

DEFAULTS: dict[str, Any] = {
    "lead_time_weeks": 1,
    "service_level": 0.95,
    "initial_stock_weeks": 3.0,
    "seed": 42,
}

OUTPUT_COLUMNS: list[str] = [
    "product_id",
    "description",
    "forecast_weekly",
    "forecast_p10",
    "forecast_p90",
    "forecast_lead_time",
    "on_hand",
    "on_hand_source",
    "safety_stock",
    "reorder_point",
    "days_of_cover",
    "stockout_probability",
    "on_order",
    "recommended_order_qty",
    "days_to_reorder",
    "revenue_at_risk",
    "risk_level",
    "action",
]


def validate_params(params: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Validate and fill defaults for risk engine simulation parameters.

    Parameters
    ----------
    params : Mapping[str, Any] | None
        User-supplied dictionary of parameters. Allowed keys are:
        - lead_time_weeks: int in [1, 6]
        - service_level: float/int in [0.80, 0.99]
        - initial_stock_weeks: float/int in [0.5, 8.0]
        - seed: int

    Returns
    -------
    dict[str, Any]
        Dictionary of validated parameters with missing values filled from DEFAULTS.

    Raises
    ------
    ValueError
        If an unknown key is encountered, a type is invalid (e.g. bool passed as int),
        or a value is outside its permissible range.
    """
    merged = dict(DEFAULTS)
    if params is not None:
        for key, val in params.items():
            if key not in DEFAULTS:
                raise ValueError(f"Unknown parameter: {key}")
            merged[key] = val

    # 1. lead_time_weeks: integer 1 to 6 (bool is not int)
    lt = merged["lead_time_weeks"]
    if isinstance(lt, bool) or not isinstance(lt, (int, np.integer)):
        raise ValueError(
            f"Parameter 'lead_time_weeks' must be an integer, got {type(lt).__name__}"
        )
    if not (1 <= int(lt) <= 6):
        raise ValueError(
            f"Parameter 'lead_time_weeks' must be between 1 and 6, got {lt}"
        )
    merged["lead_time_weeks"] = int(lt)

    # 2. service_level: numeric 0.80 to 0.99 (bool is not float/int)
    sl = merged["service_level"]
    if isinstance(sl, bool) or not isinstance(sl, (int, float, np.floating, np.integer)):
        raise ValueError(
            f"Parameter 'service_level' must be numeric, got {type(sl).__name__}"
        )
    if not (0.80 <= float(sl) <= 0.99):
        raise ValueError(
            f"Parameter 'service_level' must be between 0.80 and 0.99, got {sl}"
        )
    merged["service_level"] = float(sl)

    # 3. initial_stock_weeks: numeric 0.5 to 8.0 (bool is not float/int)
    isw = merged["initial_stock_weeks"]
    if isinstance(isw, bool) or not isinstance(isw, (int, float, np.floating, np.integer)):
        raise ValueError(
            f"Parameter 'initial_stock_weeks' must be numeric, got {type(isw).__name__}"
        )
    if not (0.5 <= float(isw) <= 8.0):
        raise ValueError(
            f"Parameter 'initial_stock_weeks' must be between 0.5 and 8.0, got {isw}"
        )
    merged["initial_stock_weeks"] = float(isw)

    # 4. seed: integer
    seed = merged["seed"]
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError(
            f"Parameter 'seed' must be an integer, got {type(seed).__name__}"
        )
    merged["seed"] = int(seed)

    return merged


def stock_noise(
    product_ids: Sequence[Any] | pd.Series | pd.Index,
    seed: int = 42,
) -> pd.Series:
    """Generate deterministic Uniform(0.5, 1.5) noise for each unique product_id.

    The noise is drawn once per product from Uniform(0.5, 1.5) using a generator
    seeded with `seed` over unique products sorted alphabetically by product_id,
    so noise depends only on seed and the sorted product IDs.

    Parameters
    ----------
    product_ids : Sequence[Any] | pd.Series | pd.Index
        Collection of product IDs.
    seed : int
        Random seed for the generator (default 42).

    Returns
    -------
    pd.Series
        Noise multipliers aligned with input product_ids.
    """
    if isinstance(product_ids, pd.Series):
        prod_series = product_ids.astype(str)
    else:
        prod_series = pd.Series(product_ids).astype(str)
    unique_sorted = sorted(prod_series.unique())
    rng = np.random.default_rng(seed)
    noise_vals = rng.uniform(0.5, 1.5, size=len(unique_sorted))
    mapping = dict(zip(unique_sorted, noise_vals))
    return prod_series.map(mapping)


def compute_risk(
    forecast_df: pd.DataFrame,
    params: Mapping[str, Any] | None = None,
    stock_overrides: Mapping[str, Mapping[str, Any]] | None = None,
    open_orders: Mapping[str, float | int] | None = None,
    lifecycle: Mapping[str, str] | None = None,
    _noise: Sequence[float] | pd.Series | np.ndarray | Mapping[str, float] | None = None,
) -> pd.DataFrame:
    """Compute inventory stockout risk and replenishment advice for each product.

    Implements SRS 6.2 formulas:
    1. on_hand = override value, else initial_stock_weeks x avg_weekly_units_train x noise.
       Source is "user" if on_hand is overridden, else "simulated".
    2. forecast_lead_time = forecast_weekly x L
    3. z = scipy.stats.norm.ppf(service_level)
    4. safety_stock = z x sigma_error x sqrt(L)
    5. reorder_point = forecast_lead_time + safety_stock
    6. days_of_cover = on_hand / (forecast_weekly / 7); 999 if forecast is 0.
    7. risk_level: HIGH if on_hand < forecast_lead_time;
       MEDIUM if on_hand < reorder_point and not HIGH; else LOW.
    8. replenishment advice: on_order from open_orders;
       inventory_position = on_hand + on_order;
       order_up_to = forecast_weekly x (L + 1) + z x sigma_error x sqrt(L + 1);
       recommended_order_qty = max(0, ceil(order_up_to - inventory_position));
       days_to_reorder = max(0, (on_hand - reorder_point) / (forecast_weekly / 7));
       revenue_at_risk = max(0, forecast_lead_time - on_hand) x last_avg_price.
    9. stockout_probability = 1 - Phi((on_hand - forecast_lead_time) / (sigma_error x sqrt(L)));
       if sigma_error is 0 then 1 when on_hand < forecast_lead_time else 0.
    10. action: if lifecycle is inactive then REVIEW_STOP and recommended_order_qty = 0;
        else if risk_level is HIGH, or days_to_reorder is 0 with recommended_order_qty > 0,
        then ORDER_NOW;
        else if days_to_reorder <= 7 with recommended_order_qty > 0 then ORDER_SOON;
        else HOLD.

    Parameters
    ----------
    forecast_df : pd.DataFrame
        DataFrame with columns: product_id, description (optional, filled if missing),
        forecast_weekly, forecast_p10, forecast_p90, sigma_error, avg_weekly_units_train,
        last_avg_price.
    params : Mapping[str, Any] | None
        Simulation parameters (lead_time_weeks, service_level, initial_stock_weeks, seed).
    stock_overrides : Mapping[str, Mapping[str, Any]] | None
        Dict mapping product_id -> {on_hand, lead_time_weeks (optional), service_level (optional)}.
    open_orders : Mapping[str, float | int] | None
        Dict mapping product_id -> total quantity of Pending and Approved orders.
    lifecycle : Mapping[str, str] | None
        Dict mapping product_id -> lifecycle stage ('new', 'growing', 'mature', 'declining', 'inactive').
    _noise : Sequence[float] | pd.Series | np.ndarray | Mapping[str, float] | None
        Optional explicit noise multiplier(s) for tests.

    Returns
    -------
    pd.DataFrame
        DataFrame containing all 18 output columns in specified order, unrounded.
    """
    valid_params = validate_params(params)
    default_l = valid_params["lead_time_weeks"]
    default_sl = valid_params["service_level"]
    init_stock_weeks = valid_params["initial_stock_weeks"]
    seed = valid_params["seed"]

    if forecast_df.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    df = forecast_df.copy()

    # Ensure required columns are present
    required_cols = [
        "product_id",
        "forecast_weekly",
        "forecast_p10",
        "forecast_p90",
        "sigma_error",
        "avg_weekly_units_train",
        "last_avg_price",
    ]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"forecast_df is missing required column: '{col}'")

    if "description" not in df.columns:
        df["description"] = ""

    # Compute noise multiplier
    if _noise is not None:
        if isinstance(_noise, Mapping):
            noise_arr = df["product_id"].astype(str).map(_noise).to_numpy(dtype=float)
        elif isinstance(_noise, pd.Series):
            if set(df["product_id"].astype(str)).issubset(set(_noise.index.astype(str))):
                noise_arr = df["product_id"].astype(str).map(_noise).to_numpy(dtype=float)
            else:
                noise_arr = _noise.to_numpy(dtype=float)
        else:
            noise_arr = np.asarray(_noise, dtype=float)
    else:
        noise_arr = stock_noise(df["product_id"], seed=seed).to_numpy(dtype=float)

    n_rows = len(df)
    overrides_map = stock_overrides or {}
    orders_map = open_orders or {}
    lifecycle_map = lifecycle or {}

    # Extract vectors
    prod_ids = df["product_id"].astype(str).tolist()
    descriptions = df["description"].tolist()
    forecast_weekly = df["forecast_weekly"].to_numpy(dtype=float)
    forecast_p10 = df["forecast_p10"].to_numpy(dtype=float)
    forecast_p90 = df["forecast_p90"].to_numpy(dtype=float)
    sigma_error = df["sigma_error"].to_numpy(dtype=float)
    avg_train = df["avg_weekly_units_train"].to_numpy(dtype=float)
    last_price = df["last_avg_price"].to_numpy(dtype=float)

    # Output arrays
    forecast_lt = np.zeros(n_rows, dtype=float)
    on_hand = np.zeros(n_rows, dtype=float)
    on_hand_source: list[str] = []
    safety_stock = np.zeros(n_rows, dtype=float)
    reorder_point = np.zeros(n_rows, dtype=float)
    days_of_cover = np.zeros(n_rows, dtype=float)
    stockout_prob = np.zeros(n_rows, dtype=float)
    on_order = np.zeros(n_rows, dtype=float)
    rec_order_qty = np.zeros(n_rows, dtype=float)
    days_to_reorder = np.zeros(n_rows, dtype=float)
    rev_at_risk = np.zeros(n_rows, dtype=float)
    risk_level: list[str] = []
    action: list[str] = []

    for i in range(n_rows):
        pid = prod_ids[i]
        fw = forecast_weekly[i]
        sigma = sigma_error[i]
        price = last_price[i]
        train_avg = avg_train[i]
        prod_noise = noise_arr[i]

        prod_override = overrides_map.get(pid, {})

        # Per-product L and service level
        l_val = prod_override.get("lead_time_weeks", default_l)
        if l_val is None:
            l_val = default_l
        l_val = float(l_val)

        sl_val = prod_override.get("service_level", default_sl)
        if sl_val is None:
            sl_val = default_sl
        sl_val = float(sl_val)

        # 1. On-hand and source
        if "on_hand" in prod_override and prod_override["on_hand"] is not None:
            oh = float(prod_override["on_hand"])
            source = prod_override.get("source", "user")
        else:
            oh = float(init_stock_weeks * train_avg * prod_noise)
            source = "simulated"

        on_hand[i] = oh
        on_hand_source.append(source)

        # 2. Forecast lead time
        flt = fw * l_val
        forecast_lt[i] = flt

        # 3 & 4. z and safety stock
        z = norm.ppf(sl_val)
        ss = z * sigma * math.sqrt(l_val)
        safety_stock[i] = ss

        # 5. Reorder point
        rop = flt + ss
        reorder_point[i] = rop

        # 6. Days of cover
        if fw <= 0.0:
            doc = 999.0
        else:
            daily_demand = fw / 7.0
            doc = oh / daily_demand
        days_of_cover[i] = doc

        # 7. Risk level
        if oh < flt:
            r_lvl = "HIGH"
        elif oh < rop:
            r_lvl = "MEDIUM"
        else:
            r_lvl = "LOW"
        risk_level.append(r_lvl)

        # 8. Replenishment advice
        ord_qty = float(orders_map.get(pid, 0.0))
        on_order[i] = ord_qty
        inv_pos = oh + ord_qty

        order_up_to = (fw * (l_val + 1.0)) + (z * sigma * math.sqrt(l_val + 1.0))
        recom_qty = max(0.0, float(math.ceil(order_up_to - inv_pos)))

        if fw <= 0.0:
            dtr = 999.0 if oh >= rop else 0.0
        else:
            daily_demand = fw / 7.0
            dtr = max(0.0, (oh - rop) / daily_demand)
        days_to_reorder[i] = dtr

        rar = max(0.0, flt - oh) * price
        rev_at_risk[i] = rar

        # 9. Stockout probability
        denom = sigma * math.sqrt(l_val)
        if denom == 0.0:
            sp = 1.0 if oh < flt else 0.0
        else:
            sp = float(1.0 - norm.cdf((oh - flt) / denom))
        stockout_prob[i] = sp

        # 10. Action label
        lc_stage = lifecycle_map.get(pid, None)
        if lc_stage == "inactive":
            act = "REVIEW_STOP"
            recom_qty = 0.0
        elif r_lvl == "HIGH" or (dtr == 0.0 and recom_qty > 0.0):
            act = "ORDER_NOW"
        elif dtr <= 7.0 and recom_qty > 0.0:
            act = "ORDER_SOON"
        else:
            act = "HOLD"

        rec_order_qty[i] = recom_qty
        action.append(act)

    result_df = pd.DataFrame(
        {
            "product_id": prod_ids,
            "description": descriptions,
            "forecast_weekly": forecast_weekly,
            "forecast_p10": forecast_p10,
            "forecast_p90": forecast_p90,
            "forecast_lead_time": forecast_lt,
            "on_hand": on_hand,
            "on_hand_source": on_hand_source,
            "safety_stock": safety_stock,
            "reorder_point": reorder_point,
            "days_of_cover": days_of_cover,
            "stockout_probability": stockout_prob,
            "on_order": on_order,
            "recommended_order_qty": rec_order_qty,
            "days_to_reorder": days_to_reorder,
            "revenue_at_risk": rev_at_risk,
            "risk_level": risk_level,
            "action": action,
        },
        columns=OUTPUT_COLUMNS,
    )

    return result_df


def summarize(risk_df: pd.DataFrame) -> dict[str, Any]:
    """Compute summary statistics across products from a risk DataFrame.

    Parameters
    ----------
    risk_df : pd.DataFrame
        DataFrame produced by `compute_risk` containing columns:
        risk_level, action, days_of_cover, revenue_at_risk.

    Returns
    -------
    dict[str, Any]
        Dictionary with keys:
        - n_high: count of HIGH risk products
        - n_medium: count of MEDIUM risk products
        - n_low: count of LOW risk products
        - n_order_now: count of ORDER_NOW action products
        - n_order_soon: count of ORDER_SOON action products
        - n_review_stop: count of REVIEW_STOP action products
        - n_hold: count of HOLD action products
        - share_at_risk: fraction of products that are HIGH or MEDIUM
        - avg_days_of_cover: average days of cover ignoring 999 values
        - total_revenue_at_risk: sum of revenue at risk across all products
        Empty input returns zeros for all fields.
    """
    if risk_df.empty:
        return {
            "n_high": 0,
            "n_medium": 0,
            "n_low": 0,
            "n_order_now": 0,
            "n_order_soon": 0,
            "n_review_stop": 0,
            "n_hold": 0,
            "share_at_risk": 0.0,
            "avg_days_of_cover": 0.0,
            "total_revenue_at_risk": 0.0,
        }

    n_total = len(risk_df)
    n_high = int((risk_df["risk_level"] == "HIGH").sum())
    n_medium = int((risk_df["risk_level"] == "MEDIUM").sum())
    n_low = int((risk_df["risk_level"] == "LOW").sum())

    n_order_now = int((risk_df["action"] == "ORDER_NOW").sum())
    n_order_soon = int((risk_df["action"] == "ORDER_SOON").sum())
    n_review_stop = int((risk_df["action"] == "REVIEW_STOP").sum())
    n_hold = int((risk_df["action"] == "HOLD").sum())

    share_at_risk = float((n_high + n_medium) / n_total) if n_total > 0 else 0.0

    valid_doc = risk_df.loc[risk_df["days_of_cover"] < 999.0, "days_of_cover"]
    avg_doc = float(valid_doc.mean()) if not valid_doc.empty else 0.0

    tot_rar = float(risk_df["revenue_at_risk"].sum())

    return {
        "n_high": n_high,
        "n_medium": n_medium,
        "n_low": n_low,
        "n_order_now": n_order_now,
        "n_order_soon": n_order_soon,
        "n_review_stop": n_review_stop,
        "n_hold": n_hold,
        "share_at_risk": share_at_risk,
        "avg_days_of_cover": avg_doc,
        "total_revenue_at_risk": tot_rar,
    }


def sensitivity_table(
    forecast_df: pd.DataFrame,
    lead_times: Sequence[int] = (1, 2, 3, 4),
    service_levels: Sequence[float] = (0.90, 0.95, 0.99),
    initial_stock_weeks: float = 3.0,
    seed: int = 42,
    stock_overrides: Mapping[str, Mapping[str, Any]] | None = None,
    open_orders: Mapping[str, float | int] | None = None,
) -> pd.DataFrame:
    """Compute sensitivity analysis across lead times and service levels.

    Evaluates the combinations of supplied `lead_times` and `service_levels`
    by calling `compute_risk` (reusing risk formulas without duplication).

    Parameters
    ----------
    forecast_df : pd.DataFrame
        Forecast DataFrame (contract C4).
    lead_times : Sequence[int]
        Collection of lead time weeks to evaluate (default: 1, 2, 3, 4).
    service_levels : Sequence[float]
        Collection of service level targets to evaluate (default: 0.90, 0.95, 0.99).
    initial_stock_weeks : float
        Starting stock assumption (default 3.0).
    seed : int
        Random seed for stock noise (default 42).
    stock_overrides : Mapping[str, Mapping[str, Any]] | None
        Optional per-product overrides.
    open_orders : Mapping[str, float | int] | None
        Optional open purchase order quantities.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - lead_time_weeks: int
        - service_level: float
        - n_high: int
        - n_medium: int
        - n_low: int
    """
    rows: list[dict[str, Any]] = []

    for lt in lead_times:
        for sl in service_levels:
            params = {
                "lead_time_weeks": lt,
                "service_level": sl,
                "initial_stock_weeks": initial_stock_weeks,
                "seed": seed,
            }
            risk_df = compute_risk(
                forecast_df=forecast_df,
                params=params,
                stock_overrides=stock_overrides,
                open_orders=open_orders,
            )
            s = summarize(risk_df)
            rows.append(
                {
                    "lead_time_weeks": int(lt),
                    "service_level": float(sl),
                    "n_high": s["n_high"],
                    "n_medium": s["n_medium"],
                    "n_low": s["n_low"],
                }
            )

    return pd.DataFrame(
        rows,
        columns=["lead_time_weeks", "service_level", "n_high", "n_medium", "n_low"],
    )


