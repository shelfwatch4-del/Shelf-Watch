# CONTRACTS.md

> The hand-off agreements between members. Changing anything here needs all four members to agree.
> On Day 1, each producer commits a **mock** version of their file to `data/mock/` with these exact columns.

---

## C1. `data/processed/weekly_demand.parquet`
**Producer:** M1 | **Consumers:** M2, M3 (for average demand), M4

| Column | Type | Notes |
|---|---|---|
| `product_id` | str | Original `StockCode` |
| `description` | str | Most frequent description for that product |
| `week_start` | date | Monday of the week |
| `units` | int | Total units sold that week (>= 0) |
| `orders` | int | Number of distinct invoices that week |
| `customers` | int | Number of distinct non-null CustomerIDs that week |
| `avg_price` | float | Mean UnitPrice that week. Forward-filled in zero-demand weeks |

Rules:
- One row per product per week; only the Top 100 products; only full weeks.
- The shop-closure week `2010-12-27` has **no row** (it is missing data, not zero demand) and is listed in `closed_weeks.csv` (C1b).
- Each product is zero-filled from its first active week **through 2011-11-28**; weeks before a product's first sale are never filled.
- Current dataset figures: 5,067 rows, 100 products, 51 weeks with rows.
- Sorted by `product_id, week_start`.


### C1b. `data/processed/closed_weeks.csv` (companion to C1)
**Producer:** M1 | **Consumer:** M2. One column, `week_start` (date). Lists full weeks in which the whole shop made no sales. These are closures, not zero demand. Currently: `2010-12-27`.

---

## C2. `data/processed/predictions.csv`
**Producer:** M2 | **Consumers:** M1 (error analysis), M3 (sigma_error), M4

| Column | Type | Notes |
|---|---|---|
| `product_id` | str | |
| `week_start` | date | Target week being predicted |
| `actual` | float | True units |
| `predicted` | float | Model forecast, clipped at 0 |
| `model_name` | str | `naive`, `moving_avg_4`, `ridge`, `random_forest`, `lightgbm`, ... |
| `split` | str | `test` (final evaluation) or `val` (validation inside training) |

Contains predictions for **all** models so they can be compared.

---

## C3. `data/processed/metrics.csv`
**Producer:** M2 | **Consumers:** M1, M4

| Column | Type | Notes |
|---|---|---|
| `model_name` | str | Same names as C2 |
| `split` | str | `test` or `val` |
| `MAE` | float | |
| `RMSE` | float | |
| `WAPE` | float | sum(abs error) / sum(actual) |
| `bias` | float | mean(predicted - actual) |

---

## C4. `data/processed/next_week_forecast.csv`
**Producer:** M2 | **Consumers:** M3, M4

Forecast for the week **after the last full week** (week_start `2011-12-05`) using the best model.

| Column | Type | Notes |
|---|---|---|
| `product_id` | str | Top-100 product SKU |
| `week_start` | date | `2011-12-05` |
| `forecast_weekly` | float | Point forecast (predicted units, clipped at 0) |
| `forecast_p10` | float | 10th percentile demand forecast, clipped at 0 |
| `forecast_p90` | float | 90th percentile demand forecast, clipped at 0 |
| `sigma_error` | float | Std of that product's forecast error (best model, val/test) |
| `avg_weekly_units_train` | float | Mean weekly units in training period |
| `last_avg_price` | float | Average price in product's last open week before forecast week |

Rules:
- `forecast_p10 <= forecast_weekly <= forecast_p90` (sorted after prediction), all clipped at 0.
- `last_avg_price` = the product's average price in its last open week before the forecast week.

---

## C5. Model files in `models/`
**Producer:** M2 | **Consumers:** M3

- `best_model.joblib`: the trained model (a full sklearn Pipeline if scaling or encoding is used)
- `feature_list.json`: ordered list of feature names exactly as used in training
- `config.json`: `{"model_name": "...", "horizon_weeks": 1, "train_end": "...", "seed": 42, "library_versions": {...}}`

---

## C6. Risk engine function (`src/risk_engine.py`)
**Producer:** M3 | **Consumers:** M3 (API), M4 (tests and sensitivity)

```python
def compute_risk(forecast_df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    forecast_df: contract C4 (product_id, forecast_weekly, sigma_error,
                 avg_weekly_units_train, ...)
    params (all optional, defaults shown):
        lead_time_weeks: int = 1          # allowed 1-4
        service_level: float = 0.95       # allowed 0.80-0.99
        initial_stock_weeks: float = 3.0  # allowed 0.5-8
        seed: int = 42
    returns DataFrame with columns:
        product_id, description, forecast_weekly, forecast_lead_time,
        on_hand, safety_stock, reorder_point, days_of_cover, risk_level
    risk_level in {"HIGH", "MEDIUM", "LOW"}
    """
```

Formulas are defined in PROJECT_CONTEXT.md section 8. Invalid params raise `ValueError`.

```python
def sensitivity_table(forecast_df, lead_times=(1,2,3,4),
                      service_levels=(0.90,0.95,0.99)) -> pd.DataFrame
# columns: lead_time_weeks, service_level, n_high, n_medium, n_low
```

---

## C7. API endpoints (`api/main.py`, FastAPI)
**Producer:** M3 | **Consumer:** M4. Base URL locally: `http://localhost:8000`

Every endpoint moves under `/api`. Existing endpoints keep their shapes and gain the `/api` prefix.

### `GET /api/health`
```json
{"status": "ok", "database": {"connected": true, "tables": {"inventory_levels": 100}}}
```

### `GET /api/products`
Returns `[{product_id, description, total_units, pattern_class}]`, ordered by `total_units` descending (training-period total).

### `GET /api/forecast/{product_id}`
```json
{
  "product_id": "85123A",
  "description": "...",
  "closed_weeks": ["2010-12-27"],
  "history": [{"week_start": "2010-12-27", "units": null}],
  "test_predictions": [{"week_start": "2011-10-10", "actual": 120, "predicted": 105.3}],
  "next_week_forecast": {"week_start": "2011-12-05", "forecast_weekly": 98.4, "forecast_p10": 60.2, "forecast_p90": 141.0}
}
```
History has 52 entries; the closure week has `units: null`. Unknown product returns HTTP 404 with `{"detail": "Product not found"}`.

### `POST /api/predict`
Request:
```json
{"product_id": "85123A", "lead_time_weeks": 1, "service_level": 0.95, "on_hand": null}
```
Returns:
`{product_id, week_start, forecast_weekly, forecast_p10, forecast_p90, risk_level, action, on_hand, on_hand_source, reorder_point, days_of_cover, stockout_probability, recommended_order_qty, days_to_reorder, revenue_at_risk}`. `on_hand: null` means use the stored record or simulated stock.

### `POST /api/predict/custom`
Request:
```json
{"recent_units": [30, 42, 38, 51, 47, 40, 55, 60], "recent_prices": null, "recent_orders": null, "recent_customers": null, "target_week": "2011-12-05"}
```
8 to 12 non-negative numbers, oldest first. Returns `{forecast_weekly, forecast_p10, forecast_p90, features_used, notes}`; `notes` states which feature groups were unavailable.

### `POST /api/simulate`
Request:
```json
{"lead_time_weeks": 2, "delay_weeks": 1, "service_level": 0.95, "initial_stock_weeks": 3}
```
`lead_time_weeks` 1 to 4, `delay_weeks` 0 to 2. Returns `{params, summary, baseline_summary, changed: [{product_id, from, to}]}`.

### `POST /api/risk`
Request:
```json
{"lead_time_weeks": 1, "service_level": 0.95, "initial_stock_weeks": 3}
```
Response:
`items` gain: `forecast_p10, forecast_p90, on_hand_source, stockout_probability, on_order, recommended_order_qty, days_to_reorder, revenue_at_risk, action`. `action` is one of `ORDER_NOW`, `ORDER_SOON`, `REVIEW_STOP`, `HOLD`. Summary gains `total_revenue_at_risk` and `n_order_now, n_order_soon, n_review_stop, n_hold`. Out-of-range params return HTTP 422.

### `POST /api/sensitivity`
Request: `{"lead_times": [1,2,3,4], "service_levels": [0.90,0.95,0.99]}`
Response: list of `{lead_time_weeks, service_level, n_high, n_medium, n_low}`

### `GET /api/metrics`
```json
[{"model_name": "lightgbm", "split": "test", "MAE": 0.0, "RMSE": 0.0, "WAPE": 0.0, "bias": 0.0}]
```

### `GET /api/disclaimer`
```json
{"text": "Inventory levels are simulated. The dataset contains sales only, not stock-on-hand. Risk flags show how the rule behaves under the stated assumptions, not real stock positions."}
```

### `GET /api/model-info`
Returns `{model_name, target_mode, feature_groups, trained_on, test_start, headline_metrics, ablation: [{groups, val_wape, val_bias}], interval_coverage: {val, test}, policy_backtest: [{policy, lead_time_weeks, fill_rate, stockout_weeks, avg_on_hand, unmet_units}], library_versions, kaggle: {dataset_url, notebook_url, model_url}}`. `policy_backtest` is an empty list if `policy_backtest.csv` does not exist yet.

### `GET /api/intelligence` & `GET /api/intelligence/{product_id}`
- `GET /api/intelligence` -> list of C11 rows.
- `GET /api/intelligence/{product_id}` -> `{profile: {...C11 row...}, companions: [{companion_id, companion_description, n_invoices, confidence, lift}]}`.

### Inventory (writes need header `X-API-Key`)
- `GET /api/inventory` -> `[{product_id, on_hand, lead_time_weeks, service_level, updated_at}]`
- `PUT /api/inventory/{product_id}` body `{on_hand, lead_time_weeks, service_level}` -> 201 created or 200 replaced
- `DELETE /api/inventory/{product_id}` -> 204

### Orders (writes need `X-API-Key`)
- `POST /api/orders` body `{product_id, quantity, supplier_note}` -> 201 `{id, status: "Pending", ...}`
- `GET /api/orders?status=&product_id=`; `GET /api/orders/{id}`
- `PATCH /api/orders/{id}` body `{status}` or `{quantity}`; transitions Pending to Approved or Cancelled, Approved to Received or Cancelled; anything else 409
- `DELETE /api/orders/{id}` only when Pending or Cancelled, else 409
- Receiving an order (`PATCH /api/orders/{id}` status Received) adds its quantity to `inventory_levels`.

### Weekly Plans (writes need `X-API-Key`)
- `POST /api/plans/generate` -> 201 `{id, week_start, status: "Draft", orders: [...]}`; 409 if a Draft plan exists for the week
- `GET /api/plans`, `GET /api/plans/{id}` (with orders); `PATCH /api/plans/{id}` body `{status: "Approved" | "Cancelled"}`; `GET /api/plans/{id}/export` -> CSV `product_id,description,quantity,status`

### Analysis Endpoints (all read from MySQL; each block carries a `source` caption string)
- `GET /api/analysis/lineage` -> `{datasets: [{name, role, rows_raw, date_min, date_max}], funnel: [{step, rows_remaining}], split: {...}, overlap_check: {...}}`
- `GET /api/analysis/features` -> `{groups, importance: [{feature, group, gain_importance, mean_abs_contribution}], ablation}`
- `GET /api/analysis/tuning` -> `{candidates, trials: [{candidate, trial_no, params, mean_val_wape, is_best}], best_params}`
- `GET /api/analysis/evaluation` -> `{metrics, by_week, by_segment, by_pattern, over_under, coverage}`
- `GET /api/analysis/backtest?product_id=` -> `{summary: [...C14 rows], replay: [...C15 rows for the product]}`
- `GET /api/explain/{product_id}?target_week=` -> `{product_id, target_week, model_name, history_used: [12 x {week_start, units}], features: [{name, group, value, contribution}], base_value, model_output, calendar: {...}, prior_year: [...] , interval: {p10, p90}}`
- `GET /api/data/transactions?product_id=&week_start=` -> `{lines: [{invoice_no, invoice_date, quantity, unit_price, country, is_cancelled}]}` (at most 200; no customer fields)
- `GET /api/predictions/history?limit=` -> saved `prediction_log` rows.
- `GET /api/logs?limit=50&offset=0` -> `[{id, request_id, method, path, status, duration_ms, created_at}]`, newest first.

### Root Aliases (mentor's Postman steps)
`GET /health` and `POST /predict` behave exactly like `/api/health` and `/api/predict`.

Error shape everywhere: `{"detail": "..."}`. 401 missing or wrong API key, 404 unknown id, 409 conflict, 422 validation, 429 rate limit.

---

## New Data Contracts

All values are for the named week (no shifting). `features.py` is the only place that shifts.

### C8. `data/processed/enrichment_weekly.parquet`
**Producer:** M1 | **Consumer:** M2

| Column | Type | Notes |
|---|---|---|
| `product_id` | str | Original `StockCode` |
| `week_start` | datetime | Monday of the week |
| `gross_units` | int | Total demand units (equals C1 `units`) |
| `cancelled_units` | int | Absolute Quantity on cancelled invoices (InvoiceNo starting with C) |
| `export_units` | int | Capped units where Country != "United Kingdom" |
| `guest_units` | int | Capped units where CustomerID is missing |

Rules:
- Same product-week grid as C1 (`weekly_demand.parquet`).
- `gross_units` equals C1 `units` (assert this).
- `export_units <= gross_units`; `guest_units <= gross_units`.
- `cancelled_units` never reduces demand units.

### C9. `data/processed/calendar_weekly.csv`
**Producer:** M1 | **Consumer:** M2

| Column | Type | Notes |
|---|---|---|
| `week_start` | date | Monday of the week (YYYY-MM-DD) |
| `uk_bank_holiday_count` | int | Number of UK bank holidays in the week (>= 0) |
| `trading_days_expected` | int | Trading weekdays minus bank holidays on trading weekdays |
| `is_christmas_week` | int | 1 if week contains 24 to 26 December, else 0 |

Rules:
- Covers Mondays from `2009-12-07` to `2011-12-05` (105 weeks).
- Bank holidays from Python `holidays` package (UK, England); includes 2011-04-22, 2011-04-25, 2011-04-29 (royal wedding), and 2011-05-02.

### C10. `data/processed/prior_year_weekly.parquet`
**Producer:** M1 | **Consumer:** M2

| Column | Type | Notes |
|---|---|---|
| `product_id` | str | Original `StockCode` |
| `week_start` | datetime | Monday date of prior-year week |
| `units` | int | Weekly units sold (capped with main training cap) |
| `orders` | int | Distinct invoices |

Rules:
- From Online Retail II `Year 2009-2010` sheet, cleaned like C1, quantity cap taken from main training data.
- Only weeks before `2010-12-06`.
- Zero-filled from each product's first prior-year sale; products with no prior-year sales have no rows (missing, not zero).
- Closure week `2009-12-28` has no rows.

### C11. `data/processed/product_intelligence.csv`
**Producer:** M1 | **Consumers:** M3 (API), M4 (pages)

| Column | Type | Notes |
|---|---|---|
| `product_id` | str | Original `StockCode` |
| `description` | str | Most frequent description in training period |
| `adi` | float | Average Demand Interval (open weeks / non-zero weeks) |
| `cv2` | float | Squared coefficient of variation of non-zero weekly units |
| `pattern_class` | str | `smooth`, `erratic`, `intermittent`, or `lumpy` |
| `abc_class` | str | `A` (top 80% revenue), `B` (next 15%), `C` (tail 5%) |
| `xyz_class` | str | Demand variability: `X` (CV <= 0.5), `Y` (0.5 < CV <= 1.0), `Z` (CV > 1.0) |
| `lifecycle_stage` | str | `new`, `growing`, `mature`, `declining`, or `inactive` |
| `weeks_since_last_sale` | int | Training weeks elapsed since last non-zero demand |
| `return_rate` | float | `cancelled_units / gross_units` over training period |
| `export_share` | float | Fraction of training units shipped outside UK |
| `top3_customer_share` | float | Share of volume from top 3 identified customers |
| `known_customer_coverage` | float | Fraction of lines with non-null CustomerID |
| `revenue_share` | float | Share of total training revenue (sums to 1.0) |

Rules:
- Strictly training weeks only (`week_start < 2011-10-10`).
- 100 rows for Top-100 products.

### C12. `data/processed/basket_pairs.csv`
**Producer:** M1 | **Consumers:** M3, M4

| Column | Type | Notes |
|---|---|---|
| `product_id` | str | Focal product SKU |
| `companion_id` | str | Associated product SKU (`!= product_id`) |
| `companion_description` | str | Companion product description |
| `n_invoices` | int | Number of training invoices containing both products (>= 30) |
| `confidence` | float | P(companion \| product) = n_AB / n_A (0.0 to 1.0) |
| `lift` | float | confidence / (n_B / N) (> 0) |

Rules:
- Non-cancelled training invoices dated before `2011-10-10`.
- Top-100 products only; at least 30 co-occurring invoices; up to 3 companions per product ranked by lift.

### C13. Database Schema (`sql/schema.sql`)
**Producer:** M3 | **Consumers:** Backend, API

MySQL 8 database `shelfwatch`; all tables defined in `sql/schema.sql` (reference tables listed in SRS 4.7, system tables in 7.3: `inventory_levels`, `replenishment_plans`, `purchase_orders`, `prediction_log`, `api_logs`). `DATABASE_URL=mysql+pymysql://user:password@localhost:3306/shelfwatch` comes from `.env` (never committed). No table stores `CustomerID`. Unit tests may use in-memory SQLite.

### C14. `data/processed/policy_backtest.csv`
**Producer:** M3 | **Consumers:** API, Analysis board

| Column | Type | Notes |
|---|---|---|
| `policy` | str | `naive`, `moving_avg_4`, `best_model`, `best_model_safety` |
| `lead_time_weeks` | int | 1 to 4 |
| `abc_class` | str | `A`, `B`, `C`, or `ALL` |
| `fill_rate` | float | Fraction of demand satisfied from stock |
| `stockout_weeks` | int | Total product-weeks where demand exceeded available stock |
| `avg_on_hand` | float | Average ending on-hand inventory across test period |
| `unmet_units` | float | Total units requested but unsatisfied |

Rules:
- Test weeks only (`2011-10-10` to `2011-11-28`). Same starting stock and demand across all policies.

### C15. `data/processed/policy_backtest_weekly.csv`
**Producer:** M3 | **Consumers:** API, Analysis board

| Column | Type | Notes |
|---|---|---|
| `policy` | str | Inventory policy evaluated |
| `lead_time_weeks` | int | Lead time in weeks |
| `product_id` | str | Product SKU |
| `week_start` | date | Monday test week date |
| `on_hand_start` | float | Inventory position at start of week |
| `order_qty` | float | Replenishment quantity ordered |
| `arrived_qty` | float | Orders received this week |
| `demand` | float | Customer demand this week |
| `sold` | float | Units fulfilled |
| `unmet` | float | Stockout units |
| `on_hand_end` | float | Ending inventory position |

---

## Definition of Done for Each Hand-off
- File or function matches the schema exactly (column names, types, order).
- Passes a schema-check test in `tests/test_contracts.py`.
- Documented in the README / docs with one example.
- Committed on the member's branch and merged to `main` after review.

---

## Change Log
| Date | Change | Agreed by |
|---|---|---|
| 11 Nov 2026 | Initial version (C1, C1b, C2, C3, C4, C5, C6, C7) | All |
| 17 Nov 2026 | C1 amended (closure week excluded, zero-fill through 2011-11-28); C4 +3 columns (intervals, last_avg_price); C7 under `/api` with new endpoints; C8 to C13 added | Team (accepted recommendations) |
| 20 Nov 2026 | MySQL replaces SQLite; plans, analysis, explain, transactions, root aliases, C15 added | Team |
| 20 Nov 2026 | Single stakeholder decided: risk items gain `action`; model-info gains `policy_backtest`; C14 added | Team |
| 21 Nov 2026 | Task E7: Formally merged CONTRACTS_ADDENDUM.md into CONTRACTS.md; added C8–C12 mock generators and schema verification tests in test_contracts.py | Team (Task E7) |
