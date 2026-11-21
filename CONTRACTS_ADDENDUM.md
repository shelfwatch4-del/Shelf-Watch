# CONTRACTS_ADDENDUM.md (version 3)

Additive changes to `CONTRACTS.md`. The team agreed to follow these recommendations. Merge this file into `CONTRACTS.md` (task E7 in `PROMPTS_DATA.md`) and record the merge in the change log. Nothing here removes an existing column.

---

## Amended contracts

### C1 (already applied in the repo)
- The shop-closure week 2010-12-27 has **no row**. It is listed in `closed_weeks.csv` (C1b).
- Each product is zero-filled from its first active week **through 2011-11-28**. Weeks before a product's first sale are never filled.
- Current figures: 5,067 rows, 100 products, 51 weeks with rows.

### C4 `next_week_forecast.csv` (new columns appended)
`product_id, week_start, forecast_weekly, forecast_p10, forecast_p90, sigma_error, avg_weekly_units_train, last_avg_price`
- `forecast_p10 <= forecast_weekly <= forecast_p90` (sorted after prediction), all clipped at 0.
- `last_avg_price` = the product's average price in its last open week before the forecast week.

### C7 API: every endpoint moves under `/api`
Existing endpoints keep their shapes and gain the `/api` prefix. New or changed endpoints:

`GET /api/forecast/{product_id}`
```json
{"product_id": "85123A", "description": "...", "closed_weeks": ["2010-12-27"],
 "history": [{"week_start": "2010-12-27", "units": null}],
 "test_predictions": [{"week_start": "2011-10-10", "actual": 120, "predicted": 105.3}],
 "next_week_forecast": {"week_start": "2011-12-05", "forecast_weekly": 98.4, "forecast_p10": 60.2, "forecast_p90": 141.0}}
```
History has 52 entries; the closure week has `units: null`.

`POST /api/predict`
```json
{"product_id": "85123A", "lead_time_weeks": 1, "service_level": 0.95, "on_hand": null}
```
returns `{product_id, week_start, forecast_weekly, forecast_p10, forecast_p90, risk_level, action, on_hand, on_hand_source, reorder_point, days_of_cover, stockout_probability, recommended_order_qty, days_to_reorder, revenue_at_risk}`. `on_hand: null` means use the stored record or simulated stock.

`POST /api/predict/custom`
```json
{"recent_units": [30, 42, 38, 51, 47, 40, 55, 60], "recent_prices": null,
 "recent_orders": null, "recent_customers": null, "target_week": "2011-12-05"}
```
8 to 12 non-negative numbers, oldest first. Returns `{forecast_weekly, forecast_p10, forecast_p90, features_used, notes}`; `notes` states which feature groups were unavailable.

`POST /api/simulate`
```json
{"lead_time_weeks": 2, "delay_weeks": 1, "service_level": 0.95, "initial_stock_weeks": 3}
```
`lead_time_weeks` 1 to 4, `delay_weeks` 0 to 2. Returns `{params, summary, baseline_summary, changed: [{product_id, from, to}]}`.

`POST /api/risk` items gain: `forecast_p10, forecast_p90, on_hand_source, stockout_probability, on_order, recommended_order_qty, days_to_reorder, revenue_at_risk, action`. `action` is one of `ORDER_NOW`, `ORDER_SOON`, `REVIEW_STOP`, `HOLD` (SRS 6.2 item 10; uses lifecycle stage from C11 when available). Summary gains `total_revenue_at_risk` and `n_order_now, n_order_soon, n_review_stop, n_hold`.

`GET /api/products` -> `[{product_id, description, total_units, pattern_class}]`, ordered by `total_units` descending (training-period total).

`GET /api/model-info` -> `{model_name, target_mode, feature_groups, trained_on, test_start, headline_metrics, ablation: [{groups, val_wape, val_bias}], interval_coverage: {val, test}, policy_backtest: [{policy, lead_time_weeks, fill_rate, stockout_weeks, avg_on_hand, unmet_units}], library_versions, kaggle: {dataset_url, notebook_url, model_url}}`. `policy_backtest` is an empty list if `policy_backtest.csv` does not exist yet.

`GET /api/intelligence` -> list of C11 rows. `GET /api/intelligence/{product_id}` -> `{profile: {...C11 row...}, companions: [{companion_id, companion_description, n_invoices, confidence, lift}]}`.

Inventory (writes need header `X-API-Key`):
- `GET /api/inventory` -> `[{product_id, on_hand, lead_time_weeks, service_level, updated_at}]`
- `PUT /api/inventory/{product_id}` body `{on_hand, lead_time_weeks, service_level}` -> 201 created or 200 replaced
- `DELETE /api/inventory/{product_id}` -> 204

Orders (writes need `X-API-Key`):
- `POST /api/orders` body `{product_id, quantity, supplier_note}` -> 201 `{id, status: "Pending", ...}`
- `GET /api/orders?status=&product_id=`; `GET /api/orders/{id}`
- `PATCH /api/orders/{id}` body `{status}` or `{quantity}`; transitions Pending to Approved or Cancelled, Approved to Received or Cancelled; anything else 409
- `DELETE /api/orders/{id}` only when Pending or Cancelled, else 409

`GET /api/logs?limit=50&offset=0` -> `[{id, request_id, method, path, status, duration_ms, created_at}]`, newest first.

Root aliases (mentor's Postman steps): `GET /health` and `POST /predict` behave exactly like `/api/health` and `/api/predict`. `GET /api/health` returns `{status, database: {connected, tables: {name: rows}}}`.

Weekly plans (writes need `X-API-Key`):
- `POST /api/plans/generate` -> 201 `{id, week_start, status: "Draft", orders: [...]}`; 409 if a Draft plan exists for the week
- `GET /api/plans`, `GET /api/plans/{id}` (with orders); `PATCH /api/plans/{id}` body `{status: "Approved" | "Cancelled"}`; `GET /api/plans/{id}/export` -> CSV `product_id,description,quantity,status`
- Receiving an order (`PATCH /api/orders/{id}` status Received) adds its quantity to `inventory_levels`.

Analysis (all read from MySQL; each block carries a `source` caption string):
- `GET /api/analysis/lineage` -> `{datasets: [{name, role, rows_raw, date_min, date_max}], funnel: [{step, rows_remaining}], split: {...}, overlap_check: {...}}`
- `GET /api/analysis/features` -> `{groups, importance: [{feature, group, gain_importance, mean_abs_contribution}], ablation}`
- `GET /api/analysis/tuning` -> `{candidates, trials: [{candidate, trial_no, params, mean_val_wape, is_best}], best_params}`
- `GET /api/analysis/evaluation` -> `{metrics, by_week, by_segment, by_pattern, over_under, coverage}`
- `GET /api/analysis/backtest?product_id=` -> `{summary: [...C14 rows], replay: [...C15 rows for the product]}`
- `GET /api/explain/{product_id}?target_week=` -> `{product_id, target_week, model_name, history_used: [12 x {week_start, units}], features: [{name, group, value, contribution}], base_value, model_output, calendar: {...}, prior_year: [...] , interval: {p10, p90}}`; contributions plus base value equal `model_output` (for scale-aware models, the caption states how scaling is applied).
- `GET /api/data/transactions?product_id=&week_start=` -> `{lines: [{invoice_no, invoice_date, quantity, unit_price, country, is_cancelled}]}` (at most 200; no customer fields)
- `GET /api/predictions/history?limit=` -> saved `prediction_log` rows.

Error shape everywhere: `{"detail": "..."}`. 401 missing or wrong API key, 404 unknown id, 409 conflict, 422 validation, 429 rate limit.

---

## New data contracts

All values are for the named week (no shifting). `features.py` is the only place that shifts.

### C8 `data/processed/enrichment_weekly.parquet` (producer M1, consumer M2)
`product_id (str), week_start (datetime, Monday), gross_units (int), cancelled_units (int), export_units (int), guest_units (int)`
- Same product-week grid as C1.
- `gross_units` equals C1 `units` (capped quantity).
- `export_units` = units where Country is not United Kingdom. `guest_units` = units where CustomerID is missing. `cancelled_units` = absolute quantity on cancelled invoices (InvoiceNo starting with C), product codes cleaned the same way.

### C9 `data/processed/calendar_weekly.csv` (producer M1, consumer M2)
`week_start, uk_bank_holiday_count (int), trading_days_expected (int), is_christmas_week (0 or 1)`
- Covers Mondays from 2009-12-07 to 2011-12-05.
- `trading_days_expected` = trading weekdays (derived from the data; no Saturday sales) minus bank holidays that fall on a trading weekday.

### C10 `data/processed/prior_year_weekly.parquet` (producer M1, consumer M2)
`product_id (str), week_start (datetime), units (int), orders (int)`
- From the Online Retail II 2009-2010 sheet, cleaned like C1, quantity cap taken from the main training data, only weeks before 2010-12-06.
- Zero-filled from each product's first prior-year sale; products with no prior-year sales have no rows (missing, not zero). Closure weeks have no row.

### C11 `data/processed/product_intelligence.csv` (producer M1, consumers M3 API, M4 pages)
`product_id, description, adi, cv2, pattern_class, abc_class, xyz_class, lifecycle_stage, weeks_since_last_sale, return_rate, export_share, top3_customer_share, known_customer_coverage, revenue_share`
- `pattern_class` in smooth, erratic, intermittent, lumpy; `abc_class` A, B, C; `xyz_class` X, Y, Z; `lifecycle_stage` in new, growing, mature, declining, inactive.
- Training weeks only (week_start before 2011-10-10).

### C12 `data/processed/basket_pairs.csv` (producer M1, consumers M3, M4)
`product_id, companion_id, companion_description, n_invoices, confidence, lift`
- Training-period invoices; top-100 products only; at least 30 invoices; up to 3 companions per product ranked by lift.

### C15 `data/processed/policy_backtest_weekly.csv` (producer M3)
`policy, lead_time_weeks, product_id, week_start, on_hand_start, order_qty, arrived_qty, demand, sold, unmet, on_hand_end` (the week-by-week replay shown on the Analysis board).

### C14 `data/processed/policy_backtest.csv` (producer M3, consumers API and Analysis board)
`policy, lead_time_weeks, abc_class (or ALL), fill_rate, stockout_weeks, avg_on_hand, unmet_units`
- Policies: `naive`, `moving_avg_4`, `best_model`, `best_model_safety`. Test weeks only; same starting stock and demand for every policy.

### C13 database schema (producer M3)
MySQL 8 database `shelfwatch`; all tables are defined in `sql/schema.sql` (reference tables listed in SRS 4.7, system tables in 7.3: `inventory_levels`, `replenishment_plans`, `purchase_orders`, `prediction_log`, `api_logs`). `DATABASE_URL=mysql+pymysql://user:password@localhost:3306/shelfwatch` comes from `.env` (never committed). No table stores `CustomerID`. Unit tests may use in-memory SQLite.

### C16 `data/processed/tuning_trials.csv` (producer M7/M15, consumers M3 loader, Analysis board)
`candidate, trial_no, params, mean_val_wape, is_best`
- `candidate`: candidate identifier (e.g. `ridge_raw`, `random_forest_raw`, `lightgbm_raw`, `random_forest_scale_aware`, etc.).
- `trial_no`: integer 1 to 30.
- `params`: JSON string of sampled hyperparameters.
- `mean_val_wape`: float, mean validation WAPE across expanding-window validation folds.
- `is_best`: 1 for the candidate's winning trial, 0 otherwise.

### C17 `data/processed/feature_importance.csv` (producer M15, consumers M3 loader, Analysis board)
`model_name, feature, feature_group, gain_importance, mean_abs_contribution`
- `model_name`: identifier of the evaluated model (e.g. `lightgbm` or `ridge`).
- `feature`: feature column name.
- `feature_group`: feature group (`base`, `g1`, `g2`, `g3`, `g4`). Also aliased as `group`.
- `gain_importance`: total gain importance for tree models (LightGBM) or absolute coefficient for linear models (Ridge).
- `mean_abs_contribution`: mean of the absolute values of per-feature contributions evaluated on held-out test rows.

---

## Change log
| Date | Change | Agreed by |
|---|---|---|
| 17 Nov 2026 | C1 amended; C4 +3 columns; C7 under `/api` with new endpoints; C8 to C13 added | Team (accepted recommendations) |
| 20 Nov 2026 | MySQL replaces SQLite; plans, analysis, explain, transactions, root aliases, C15 added | Team |
| 17 Nov 2026 | Single stakeholder decided: risk items gain `action`; model-info gains `policy_backtest`; C14 added | Team |
| 21 Nov 2026 | C16 (tuning_trials.csv) and C17 (feature_importance.csv) added for M15 explainability | Team |
