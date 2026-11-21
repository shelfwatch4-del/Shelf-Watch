# PROMPTS_BACKEND.md: Risk Engine, Database and API (Member 3), version 3

**You own:** `src/risk_engine.py`, `api/` (`main.py`, `schemas.py`, `db.py`, `models.py`, `middleware.py`, `routers/`), `scripts/api_client.py`, `scripts/bench_api.py`, `tests/test_risk_engine.py`, `test_db_crud.py`, `test_middleware.py`, `test_api.py`, `CHANGE_REQUESTS.md`.
**Inputs:** `next_week_forecast.csv` (C4 with p10, p90, last price), `weekly_demand.parquet`, `predictions.csv`, `metrics.csv`, `error_analysis.csv`, `product_intelligence.csv`, `basket_pairs.csv`, `models/*`. Use `data/mock/` until the real files exist.
**Setup:** paste the Windows preface and closing check from `PROMPTS_00_INDEX.md`. Planning mode; strongest model for B1, B4, B8, B11.
**Read first:** SRS sections 6, 7 and `CONTRACTS_ADDENDUM.md` (all endpoints are under `/api`).

> Never edit `CONTRACTS.md` or `docs/SRS.md` yourself. Contract changes go in `CHANGE_REQUESTS.md`.

---

# Part 1: Risk engine v3

## Task B1: `compute_risk` with advice and overrides
**SRS:** 6.1 to 6.3, FR-R-1 to FR-R-4, FR-R-10, FR-R-11, FR-R-13

```
Read docs/SRS.md sections 6.1 to 6.3 and CONTRACTS_ADDENDUM.md C4 and C7.

Create src/risk_engine.py (replace any earlier draft) with:

DEFAULTS = {"lead_time_weeks": 1, "service_level": 0.95, "initial_stock_weeks": 3.0, "seed": 42}
Allowed: lead_time_weeks integer 1 to 6 (sliders and /api/risk use 1 to 4; /api/simulate adds delay_weeks 0 to 2), service_level 0.80 to 0.99, initial_stock_weeks 0.5 to 8.

validate_params(params) -> dict: fill defaults, reject unknown keys, check types and ranges (a bool is not an int), raise ValueError naming the parameter.

compute_risk(forecast_df, params=None, stock_overrides=None, open_orders=None, lifecycle=None, _noise=None) -> DataFrame.
- forecast_df columns: product_id, description, forecast_weekly, forecast_p10, forecast_p90, sigma_error, avg_weekly_units_train, last_avg_price.
- stock_overrides: dict product_id -> {on_hand, lead_time_weeks (optional), service_level (optional)}. A per-product lead time or service level replaces the global one for that product.
- open_orders: dict product_id -> total quantity of Pending and Approved orders.
- lifecycle: dict product_id -> lifecycle stage (new, growing, mature, declining, inactive) from product_intelligence.csv; optional.
- Output columns in order: product_id, description, forecast_weekly, forecast_p10, forecast_p90, forecast_lead_time, on_hand, on_hand_source ("user" or "simulated"), safety_stock, reorder_point, days_of_cover, stockout_probability, on_order, recommended_order_qty, days_to_reorder, revenue_at_risk, risk_level, action.

Formulas, exactly as SRS 6.2:
1. on_hand = override value, else initial_stock_weeks x avg_weekly_units_train x noise
2. forecast_lead_time = forecast_weekly x L
3. z = scipy.stats.norm.ppf(service_level)
4. safety_stock = z x sigma_error x sqrt(L)
5. reorder_point = forecast_lead_time + safety_stock
6. days_of_cover = on_hand / (forecast_weekly / 7); 999 if the forecast is 0
7. HIGH if on_hand < forecast_lead_time; MEDIUM if on_hand < reorder_point and not HIGH; else LOW. Equality is NOT below the threshold. Compare unrounded values.
8. on_order from open_orders; inventory_position = on_hand + on_order; order_up_to = forecast_weekly x (L + 1) + z x sigma_error x sqrt(L + 1); recommended_order_qty = max(0, ceil(order_up_to - inventory_position)); days_to_reorder = max(0, (on_hand - reorder_point) / (forecast_weekly / 7)) (0 means order now); revenue_at_risk = max(0, forecast_lead_time - on_hand) x last_avg_price.
9. stockout_probability = 1 - Phi((on_hand - forecast_lead_time) / (sigma_error x sqrt(L))); if sigma_error is 0 then 1 when on_hand < forecast_lead_time else 0.
10. action (what the manager should do next): if lifecycle is inactive then REVIEW_STOP and recommended_order_qty = 0; else if risk_level is HIGH, or days_to_reorder is 0 with recommended_order_qty > 0, then ORDER_NOW; else if days_to_reorder <= 7 with recommended_order_qty > 0 then ORDER_SOON; else HOLD. Without lifecycle data only the risk and reorder rules apply.

The optional _noise argument (aligned to product_id) exists only for tests. Do not round inside the function.
```

## Task B2: Deterministic noise and summary
```
Add stock_noise(product_ids, seed) -> Series: for products sorted by product_id draw one Uniform(0.5, 1.5) per product with numpy default_rng(seed); the noise depends only on the seed and the sorted list. Use it inside compute_risk when _noise is None.

Add summarize(risk_df) -> dict: n_high, n_medium, n_low, n_order_now, n_order_soon, n_review_stop, n_hold, share_at_risk, avg_days_of_cover (ignoring 999), total_revenue_at_risk. Empty input returns zeros.

Tests: same seed gives identical output; different seeds differ; changing lead time or service level leaves on_hand unchanged; summarize matches a hand count.
```

## Task B3: `sensitivity_table`
```
Add sensitivity_table(forecast_df, lead_times=(1,2,3,4), service_levels=(0.90,0.95,0.99), initial_stock_weeks=3.0, seed=42, stock_overrides=None, open_orders=None) -> DataFrame (lead_time_weeks, service_level, n_high, n_medium, n_low) by calling compute_risk (no duplicated formulas). Test: 12 rows by default; the row for lead 1 and service 0.95 equals compute_risk with defaults; counts add up to the number of products in every row; HIGH counts do not decrease as lead time grows (explain if this can legitimately fail).
```

## Task B4: Risk engine tests
**SRS:** Appendix A, T-R-1 to T-R-6

```
Create tests/test_risk_engine.py:
1. Appendix A: forecast_weekly 100, sigma_error 20, avg_weekly_units_train 90, initial_stock_weeks 3, L 2, service_level 0.95; inject noise 1.1, 0.8, 0.6. Expected forecast_lead_time 200, safety_stock about 46.5, reorder_point about 246.5, on_hand 297, 216, 162, classes LOW, MEDIUM, HIGH, days of cover about 20.8, 15.1, 11.3.
2. Advice on the same example: order_up_to = 100 x 3 + 1.645 x 20 x sqrt(3) (about 356.99); with on_order 0 and noise 0.6 (on_hand 162) recommended_order_qty = ceil(order_up_to - 162); with on_order 100 it drops by 100; revenue_at_risk = max(0, 200 - 162) x last_avg_price.
3. stockout_probability: equals 0.5 when on_hand equals forecast_lead_time; increases as on_hand decreases; sigma 0 gives 1 or 0.
4. A stock override replaces simulated stock, sets on_hand_source "user" and can change the class; a per-product lead time override is applied only to that product.
5. Edge cases: forecast 0 (days of cover 999, no crash), sigma 0, lead time 6, service 0.99, on_hand exactly equal to a threshold (not HIGH when equal to forecast_lead_time; LOW when equal to reorder_point), empty input.
6. Determinism, and on_hand independent of other parameters.
7. Invalid params raise ValueError (lead 0, 7 and 1.5; service 0.5 and 1.0; initial stock 0 and 9; unknown key; bool).
8. Output column order equals the SRS list (ending with risk_level, action).
9. Action: an inactive product gives REVIEW_STOP with quantity 0; a HIGH product gives ORDER_NOW; a MEDIUM product with days_to_reorder 3 and quantity > 0 gives ORDER_SOON; a product with ample stock gives HOLD; with lifecycle=None the rules still work.
```
**Done when:** `pytest tests/test_risk_engine.py -q` is green.

## Task B5: Sensitivity notebook
```
Create notebooks/04_sensitivity.py (cell format) with the real forecast: the sensitivity table; a heatmap of HIGH and MEDIUM counts; a chart of at-risk share versus initial_stock_weeks (0.5 to 8) for lead times 1 to 4; total revenue at risk at the defaults. Print how many products are HIGH and MEDIUM at the defaults. Add a 4-sentence interpretation. Do NOT tune the defaults to produce a desired count.
```

## Task B5b: Replenishment-policy backtest (the manager's success measure)
**SRS:** FR-R-9 | **Contract:** C14 | **Needs:** M12 (`predictions.csv`)

```
Read docs/SRS.md FR-R-9 and docs/STAKEHOLDER.md section 5. Create src/backtest.py with run_policy_backtest(predictions, weekly, forecast_meta, intelligence=None, lead_times=(1, 2)) -> DataFrame and a CLI python -m src.backtest that writes data/processed/policy_backtest.csv (C14) and docs/BACKTEST.md.

Simulate the 8 test weeks (2011-10-10 to 2011-11-28) for every product with a weekly-review order-up-to policy. Use the same starting stock, actual demand and lead time for every policy; unmet demand is lost (not back-ordered). Policies: naive (last week's units), moving_avg_4, best_model (point forecast from predictions.csv, split test), best_model_safety (best model plus z x sigma_error x sqrt(L + 1), z for 0.95). Order-up-to level = forecast x (L + 1) (+ safety stock for the last policy), using the one-step forecast as the proxy for the lead-time weeks, exactly as the risk rule does; state this simplification in docs/BACKTEST.md. Orders arrive after L weeks. Starting stock = 3 weeks of the product's average training demand (no random noise).

Metrics per policy and lead time, overall and per ABC class when intelligence is given: fill rate (units sold / units demanded), stockout weeks (product-weeks with unmet demand), average on-hand units at week end, total unmet units. Do NOT tune any policy parameter on these test weeks.

Tests: with perfect forecasts and enough stock the fill rate is 1 in a toy case; adding safety stock never lowers the fill rate in a toy case; results are deterministic; no information from week t or later enters the forecast used at week t (use only the predictions row for that week).

In docs/BACKTEST.md report the table with real numbers, say plainly which policy wins on fill rate and at what cost in on-hand stock, and say so if the differences are small. Do not claim an improvement that the table does not show.
```

---

# Part 2: Database and CRUD

## Task B6: Database layer
**SRS:** 7.3, FR-DB-1, FR-DB-2, FR-DB-6

```
Add sqlalchemy (2.x), pymysql, cryptography and python-dotenv to requirements.txt, pinned. MySQL 8 is installed locally; database `shelfwatch` and user are created from sql/schema.sql and .env (copy .env.example). Create api/db.py and api/models.py:
- engine and session factory from DATABASE_URL in .env (mysql+pymysql://...), pool_pre_ping=True; no default password in code. Unit tests use an in-memory SQLite or TEST_DATABASE_URL.
- ORM models exactly as SRS 7.3: InventoryLevel(product_id PK, on_hand >= 0, lead_time_weeks 1..6, service_level 0.80..0.99, updated_at), PurchaseOrder(id, product_id, quantity > 0, supplier_note, status, created_at, updated_at), ApiLog(id, request_id, method, path, status, duration_ms, created_at).
- init_db() runs sql/schema.sql if tables are missing; extra ORM models for replenishment_plans and prediction_log as in SRS 7.3; seed_demo_data() inserts 5 stock records and 4 purchase orders (one per status) ONLY when the tables are empty; write_log(...) inserts an ApiLog row.
- get_db() dependency yielding a session that always closes.
- All access through SQLAlchemy sessions and ORM or bound parameters; never build SQL from strings.
Tests (tests/test_db_crud.py) with in-memory SQLite (or TEST_DATABASE_URL): tables created, seed inserts once and only when empty, constraints enforced, write_log works.
```

## Task B6b: Load project data into MySQL, reset script
**SRS:** 4.7, 7.3, FR-DB-3
```
Create scripts/load_mysql.py: read every file in data/processed (C1 to C15, tuning_trials.csv, feature_importance.csv, ablation, metrics, predictions, next_week_forecast, error_analysis, product_intelligence, basket_pairs, policy_backtest and policy_backtest_weekly) plus the pipeline row counts, and insert them into the matching tables of sql/schema.sql in batches with SQLAlchemy. Idempotent (truncate reference tables, then insert); write data_sources and pipeline_log rows; load transactions_top100 from the cleaned transactions WITHOUT CustomerID (has_customer flag only). Print a row-count table.
Create scripts/reset_demo.py: clears inventory_levels, replenishment_plans, purchase_orders, prediction_log and api_logs, then reseeds the demo rows. Tests for both on SQLite. Do not run anything against my MySQL; give me the commands.
```
## Task B7: Inventory CRUD service
```
Create api/crud_inventory.py with functions: list_stock(db), get_stock(db, product_id), upsert_stock(db, product_id, data) -> (record, created: bool), delete_stock(db, product_id). Validate ranges (on_hand >= 0; lead_time_weeks 1..6; service_level 0.80..0.99). Unknown product ids (not in our 100 products) must raise a NotFound error. Tests: create then read, replace returns created False, delete then read gives NotFound, invalid values raise a validation error.
```

## Task B8: Purchase-order CRUD with status rules
```
Create api/crud_orders.py: create_order, list_orders(status, product_id), get_order, update_order (status or quantity), delete_order. Rules: new orders are Pending; allowed transitions Pending to Approved, Pending to Cancelled, Approved to Received, Approved to Cancelled; anything else raises Conflict (409); quantity can be changed only while Pending; delete only when Pending or Cancelled, else Conflict; unknown product raises NotFound; quantity must be a positive integer. Add open_order_quantities(db) -> dict product_id -> sum of Pending and Approved quantities (used as on_order by the risk engine).
Tests: every allowed and every forbidden transition; delete rules; open_order_quantities ignores Received and Cancelled orders.
```

---

# Part 3: API

## Task B9: App skeleton, schemas and meta endpoints
**SRS:** 7.1, 7.2 (FR-A-1, FR-A-4, FR-A-5, FR-A-7, FR-A-10)

```
Create api/schemas.py (Pydantic v2 request and response models with example values for /docs) and api/main.py with:
- a lifespan handler that calls init_db() and seed_demo_data(), and loads once from data/processed/ and models/: weekly_demand.parquet, next_week_forecast.csv (joined with descriptions), metrics.csv, predictions.csv, error_analysis.csv, product_intelligence.csv, basket_pairs.csv, models/config.json, models/ablation.json, models/interval_coverage.json, and (optional, may not exist yet) data/processed/policy_backtest.csv, product_intelligence.csv (also used to build the lifecycle mapping passed to compute_risk). A missing required file stops start-up with a message that names the file and the command that creates it.
- all JSON endpoints in an APIRouter with prefix /api; a clean(obj) helper converting NaN, inf and numpy types to JSON-safe values (NaN to null) used for every response; never expose CustomerID.
- GET /api/health; GET /api/disclaimer ({"text": "Inventory levels are simulated. The dataset contains sales only, not stock-on-hand. Risk flags show how the rule behaves under the stated assumptions, not real stock positions."}); GET /api/model-info as in CONTRACTS_ADDENDUM.md (Kaggle URLs from environment variables KAGGLE_DATASET_URL, KAGGLE_NOTEBOOK_URL, KAGGLE_MODEL_URL, empty string if unset).
Tests: /api/health, /api/disclaimer and /api/model-info with the documented fields.
```

## Task B10: `/api/products` and `/api/forecast/{product_id}`
```
GET /api/products -> [{product_id, description, total_units, pattern_class}] ordered by total_units descending (training-period total; pattern_class from product_intelligence.csv or null).
GET /api/forecast/{product_id} -> exactly the shape in CONTRACTS_ADDENDUM.md: history with ALL 52 weeks (the closure week 2010-12-27 has units null), closed_weeks, test_predictions from the best model, next_week_forecast with p10 and p90. Unknown product gives 404 {"detail": "Product not found"}.
Record total_units and pattern_class in CHANGE_REQUESTS.md if the contract file does not yet list them.
Tests: 100 products and ordering; 52 history entries with exactly one null; ISO dates; 404; no NaN token in the JSON text.
```

## Task B11: `/api/predict` and `/api/predict/custom`
**SRS:** FR-A-11, FR-M-14

```
POST /api/predict body {product_id, lead_time_weeks, service_level, on_hand (optional)}: use the product's forecast row, apply the stored inventory record or the supplied on_hand, call compute_risk for that one product (with open orders from the database) and return the fields listed in CONTRACTS_ADDENDUM.md. 404 for an unknown product, 422 for invalid values.
POST /api/predict/custom: call src.train.predict_custom (or src.predict) with the payload; 8 to 12 non-negative numbers, oldest first; 422 otherwise. Return forecast_weekly, forecast_p10, forecast_p90, features_used and notes.
Tests: p10 <= forecast <= p90; override changes the result; custom rejects 7 values, 13 values and negatives; the same custom payload twice gives the same output.
```

## Task B12: `/api/risk`, `/api/simulate`, `/api/sensitivity`
**SRS:** FR-A-6, FR-R-12

```
POST /api/risk: body RiskRequest (lead_time_weeks 1..4, service_level 0.80..0.99, initial_stock_weeks 0.5..8) -> {params (with seed), summary, items} using compute_risk with stock overrides and open orders read from the database. No formulas in the API layer.
POST /api/simulate: body {lead_time_weeks 1..4, delay_weeks 0..2, service_level, initial_stock_weeks}; effective lead time = lead_time_weeks + delay_weeks; return {params, summary, baseline_summary (the same call with delay 0), changed: [{product_id, from, to}]} listing products whose risk class changed.
POST /api/sensitivity: body {lead_times, service_levels} (each list at most 8 values) -> rows.
Cache results by parameter set (including a version counter of the stock and order tables). Any write to inventory or orders MUST clear the cache.
Tests: default risk equals compute_risk called directly; a stock write changes the next /api/risk response (cache cleared); simulate with delay 0 equals risk; out-of-range values give 422; the same request twice gives identical JSON.
```

## Task B13: Metrics, errors, intelligence, baskets, logs
```
GET /api/metrics (all candidates, both splits), GET /api/error-analysis (segments, weeks, products from error_analysis.csv; return empty lists with a note if the file does not exist yet), GET /api/intelligence and GET /api/intelligence/{product_id} (profile plus up to 3 companions from basket_pairs.csv; 404 unknown), GET /api/logs?limit=50&offset=0 (newest first; limit at most 200).
Tests: schemas, counts, 404, paging, JSON safe.
```

## Task B14: Middleware
**SRS:** 7.4, FR-MW-1 to FR-MW-6

```
Create api/middleware.py and register it in main.py:
1. CORS from ALLOWED_ORIGINS (comma-separated; empty means none), methods GET POST PUT PATCH DELETE.
2. Request ID and timing: every response gets X-Request-ID and X-Process-Time-Ms.
3. Audit log: write method, path, status and duration to api_logs for /api paths except /api/logs. Never let a logging failure break a request.
4. API key: write requests (PUT, DELETE on inventory and orders, POST /api/orders, PATCH) need header X-API-Key equal to the API_KEY environment variable (default demo-key for local use), else 401 {"detail": "Invalid or missing API key"}. Reads and the prediction endpoints stay open.
5. Rate limit: in-memory sliding window per client address, default 120 requests per minute (RATE_LIMIT env), else 429. Disable it in tests through the setting.
6. Global exception handler: unhandled errors return {"detail": "Internal server error"} with status 500 and log the details server-side only.
Tests (tests/test_middleware.py): headers present on every response; write without key gives 401, with key succeeds, GET needs no key; requests appear in /api/logs; rate limit returns 429 after the limit; a forced error returns a generic 500 with no stack trace; CORS headers appear only for allowed origins.
After the tests, write docs/MIDDLEWARE_EXPLAINED.md: each middleware in 3 sentences with the exact code location, so every team member can explain it.
```

## Task B15: CRUD routes
```
Wire the inventory and order services to the routes in CONTRACTS_ADDENDUM.md: GET/PUT/DELETE /api/inventory and /api/inventory/{product_id}; POST/GET /api/orders, GET/PATCH/DELETE /api/orders/{id}. Status codes: 201 on create (PUT returns 201 when new and 200 when replaced), 204 on delete, 404, 409, 422, 401 without a key. Every write clears the risk cache.
Tests (tests/test_api.py and tests/test_db_crud.py): the full happy path; every status code; the transitions table; a stock write changes on_hand_source to "user" in /api/risk; an approved order increases on_order and lowers recommended_order_qty.
```

## Task B16: Serve the dashboard from the same app
```
At the VERY END of api/main.py mount the dashboard folder as static files at "/" with html=True, after every API route. If dashboard/ does not exist, log a warning and skip the mount. Tests: /api/health, /docs, /openapi.json and a POST to /api/risk still work; GET / returns dashboard/index.html when it exists (create a temporary one in the test); the five page files are served at /index.html, /predict.html and so on.
```

## Task B17: Test suite and contract check
```
Finish tests/test_api.py for every endpoint and every case in SRS 10.2 including T-A-1 to T-A-3, T-P-1, T-P-2, T-DB-1 to T-DB-3, T-MW-1 to T-MW-3. Run /contract-check against CONTRACTS.md plus the addendum and fix mismatches (except documented additions). Run pytest -q for the whole project and print the totals.
```

## Task B18: Python API client, docs and benchmark
**SRS:** 7.5, FR-PC-1 to FR-PC-3, FR-A-12, NFR-1

```
1. scripts/api_client.py with httpx: functions for health, products, predict, predict_custom, risk, simulate, inventory CRUD and order CRUD; base URL and API key from arguments or environment; timeout 10 seconds; up to 2 retries on 5xx or connection errors; clear exceptions with the status code and the detail message. Command line: python scripts/api_client.py demo walks through health, predict, create an order, approve it, change stock, delete the order, printing each response.
2. A test that starts the app locally (uvicorn in a thread or TestClient transport) and runs the demo sequence.
3. OpenAPI tags, summaries and example bodies on every endpoint; a README section "API" with a curl example per endpoint family; check that /docs renders them.
4. scripts/bench_api.py: median and 95th percentile over 50 calls each for POST /api/risk (warm and cold), GET /api/forecast/{id}, GET /api/products. Targets: /api/risk under 500 ms warm, /api/forecast under 300 ms. If missed, profile, fix and show before and after.
5. docs/API_CLIENT_EXPLAINED.md: how a Python program calls a REST API (request, headers, JSON, status codes, timeouts, retries) with the project's own code excerpts.
```
**Done when:** the demo runs end to end and the whole suite is green. Suggested commit: `feat(api): database, CRUD, middleware, client`.

---

# Part 4: Plan workflow, analysis board and evaluation readiness (added in version 3.2)

## Task B19: Weekly replenishment plan workflow
**SRS:** FR-W-1 to FR-W-6, 7.3
```
Implement the plan endpoints in CONTRACTS_ADDENDUM.md: POST /api/plans/generate (builds a Draft plan from the current risk results: one order line for every ORDER_NOW and ORDER_SOON product with recommended_order_qty > 0; 409 if a Draft plan already exists for the week), GET /api/plans, GET /api/plans/{id}, PATCH /api/plans/{id} (Draft to Approved or Cancelled; approving creates Pending purchase_orders rows linked to the plan, in one transaction), GET /api/plans/{id}/export as CSV. Receiving an order (PATCH /api/orders/{id} to Received) adds the quantity to inventory_levels. Writes need the API key. Tests: generate, duplicate 409, approve creates orders, cancel creates none, export columns, receive updates stock and on_hand_source becomes "user".
```

## Task B20: Analysis, explain and transaction endpoints
**SRS:** FR-U-20, FR-M-17, 7.2
```
Implement /api/analysis/lineage, /features, /tuning, /evaluation, /backtest, /api/explain/{product_id}, /api/data/transactions and /api/predictions/history exactly as in CONTRACTS_ADDENDUM.md. Everything is read from MySQL tables, never from CSV at request time. Every analysis block returns a `source` string naming its tables. /api/explain builds the same feature row as src/features.py for the target week, returns feature values and per-feature contributions (LightGBM pred_contrib, or coefficient x scaled value for Ridge) whose sum plus base_value equals model_output within 1e-6; for a scale-aware model say how scaling is applied. /api/data/transactions returns at most 200 lines and no customer fields. POST /api/predict also inserts a prediction_log row. Tests for each endpoint, the explain sum check, the 200-row cap and the absence of customer columns.
```

## Task B21: Root aliases and health with database status
**SRS:** 11.2, 11.4
```
Add GET /health and POST /predict at the root that call the same handlers as /api/health and /api/predict (same schemas, same middleware, same logs). /api/health returns {status, database: {connected, tables: {name: row_count}}} and status "degraded" (HTTP 200) if MySQL is unreachable. Both root routes must be registered BEFORE the static mount of B16. Tests: both pairs return identical bodies; the Postman request "00 mentor flow" assertions pass against a local server.
```
