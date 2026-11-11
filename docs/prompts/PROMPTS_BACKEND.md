# PROMPTS_BACKEND.md: Risk Engine and FastAPI Backend (Member 3)

**You own:** `src/risk_engine.py`, `notebooks/04_sensitivity.py`, `api/main.py`, `api/schemas.py`, `tests/test_risk_engine.py`, `tests/test_api.py`, `CHANGE_REQUESTS.md`.
**Inputs:** `next_week_forecast.csv` (C4), `weekly_demand.parquet` (C1), `predictions.csv` (C2), `metrics.csv` (C3), `error_analysis.csv`, `models/*` (C5). Use `data/mock/` until real files exist.
**Hands off to:** Member 4 (dashboard calls your endpoints) and Member 2 (reads your feedback on file formats).
**Setup first:** Phase 0 in `PROMPTS.md`. **Mode:** Planning. Use the strongest model for B1 to B4 and B7.

**Closing check (add to any prompt that lacks it):**
```
Before you finish: run the tests for this task, show exact commands and output, list every file you created or changed, and list anything in the SRS sections I named that you did NOT implement. Do not claim success unless tests ran and passed.
```

> Never edit `CONTRACTS.md` yourself. If an endpoint or field needs to change, write it into `CHANGE_REQUESTS.md` and tell the team.

---

# Part 1: Risk engine

## Task B1: Core `compute_risk`

**SRS:** section 6.1 to 6.3, FR-R-1 to FR-R-4 | **Output:** `src/risk_engine.py`

```
Read docs/SRS.md sections 6.1 to 6.3 and CONTRACTS.md C4 and C6.

Create src/risk_engine.py with:

DEFAULTS = {"lead_time_weeks": 1, "service_level": 0.95, "initial_stock_weeks": 3.0, "seed": 42}
RANGES: lead_time_weeks integer 1 to 4; service_level 0.80 to 0.99; initial_stock_weeks 0.5 to 8.

validate_params(params: dict | None) -> dict: fill defaults, reject unknown keys, check types and ranges (bool is not an int), raise ValueError with a clear message naming the parameter.

compute_risk(forecast_df, params=None, _noise=None) -> pd.DataFrame. Input columns: product_id, description, forecast_weekly, sigma_error, avg_weekly_units_train. Output columns, in order: product_id, description, forecast_weekly, forecast_lead_time, on_hand, safety_stock, reorder_point, days_of_cover, risk_level.

Formulas (exactly as SRS 6.2):
- on_hand = initial_stock_weeks * avg_weekly_units_train * noise
- forecast_lead_time = forecast_weekly * lead_time_weeks
- z = scipy.stats.norm.ppf(service_level)
- safety_stock = z * sigma_error * sqrt(lead_time_weeks)
- reorder_point = forecast_lead_time + safety_stock
- days_of_cover = on_hand / (forecast_weekly / 7); if forecast_weekly == 0 then 999
- risk_level: "HIGH" if on_hand < forecast_lead_time; "MEDIUM" if on_hand < reorder_point; else "LOW". Equality is NOT below the threshold. Compare unrounded values.

The optional _noise argument (a Series or array aligned to product_id) is only for tests so a worked example can force exact noise values.

Do not round inside the function; rounding is a display concern.
```
**Done when:** it imports and runs on the mock forecast.

## Task B2: Deterministic stock noise and summary

**SRS:** 6.2 formula 1, FR-R-2, FR-R-6

```
In src/risk_engine.py add stock_noise(product_ids, seed) -> pd.Series: for products sorted by product_id draw one value per product from Uniform(0.5, 1.5) using numpy.random.default_rng(seed), and return it aligned to the given product_ids. The noise must depend only on the seed and the sorted product list, never on lead time, service level or initial stock. Use it inside compute_risk when _noise is None.

Add summarize(risk_df) -> dict with n_high, n_medium, n_low, share_at_risk ((high+medium)/total), avg_days_of_cover (ignore values equal to 999). Handle empty input by returning zeros without error.

Tests: same seed gives identical output; different seeds differ; changing lead_time_weeks or service_level leaves on_hand unchanged; summarize on a small frame matches hand-counted values.
```

## Task B3: `sensitivity_table`

**SRS:** FR-R-5

```
Add sensitivity_table(forecast_df, lead_times=(1,2,3,4), service_levels=(0.90,0.95,0.99), initial_stock_weeks=3.0, seed=42) -> DataFrame with columns lead_time_weeks, service_level, n_high, n_medium, n_low, one row per combination. It must reuse compute_risk (no duplicated formulas). Validate every lead time and service level with validate_params. Test: shape is 12 rows by default; the row for lead 1 and service 0.95 equals the counts from compute_risk with defaults; n_high + n_medium + n_low equals the number of products in every row; counts of HIGH never decrease as lead time increases (assert this monotonic property on the mock data and explain if it can legitimately fail).
```

## Task B4: Risk engine tests (SRS Appendix A)

**SRS:** Appendix A, T-R-1 to T-R-4

```
Create tests/test_risk_engine.py:

1. Worked example from SRS Appendix A: forecast_weekly 100, sigma_error 20, avg_weekly_units_train 90, initial_stock_weeks 3, lead_time_weeks 2, service_level 0.95. Inject noise (via _noise) 1.1, 0.8 and 0.6 for three copies of the product. Expected: forecast_lead_time 200; safety_stock about 46.5; reorder_point about 246.5; on_hand 297, 216, 162; classes LOW, MEDIUM, HIGH; days_of_cover about 20.8, 15.1, 11.3 (use pytest.approx with 0.1 tolerance).
2. Edge cases: forecast 0 gives days_of_cover 999 and does not crash; sigma 0 gives safety_stock 0; lead_time 4; service_level 0.99; on_hand exactly equal to forecast_lead_time is MEDIUM or LOW (not HIGH) and exactly equal to reorder_point is LOW; empty input returns an empty frame with the correct columns.
3. Determinism and independence of on_hand from other parameters.
4. Invalid params raise ValueError: lead_time 0, 5 and 1.5; service_level 0.5 and 1.0; initial_stock_weeks 0 and 9; unknown key; bool values.
5. Output column order equals contract C6.
```
**Done when:** `pytest tests/test_risk_engine.py -q` is fully green. Commit: `feat(risk): risk engine, summary, sensitivity`.

## Task B5: Sensitivity notebook

```
Create notebooks/04_sensitivity.py (cell format). Using the real next_week_forecast.csv, produce: the sensitivity table; a heatmap of HIGH and MEDIUM counts across lead time and service level (saved to reports/figures/); a line chart of at-risk share versus initial_stock_weeks (0.5 to 8) for lead times 1 to 4. Print how many products are HIGH and MEDIUM at the defaults. Add a markdown interpretation of 4 sentences. Do NOT adjust the defaults to produce a desired count; state honestly if the defaults flag few products and why (3 weeks of starting stock with a 1-week lead time is generous).
```

---

# Part 2: FastAPI backend

## Task B6: App skeleton, loading and schemas

**SRS:** 7.2 (FR-A-1, FR-A-4, FR-A-8), 14.3

```
Read docs/SRS.md sections 7 and 14.3 and CONTRACTS.md C7.

Create api/schemas.py with Pydantic v2 models: RiskRequest (lead_time_weeks int 1..4, service_level float 0.80..0.99, initial_stock_weeks float 0.5..8, defaults 1, 0.95, 3.0), SensitivityRequest (lead_times list[int], service_levels list[float], each validated with the same ranges), and response models for each endpoint.

Create api/main.py with a lifespan handler loading ONCE at startup, from data/processed/ and models/: weekly_demand.parquet, next_week_forecast.csv (joined with description from weekly_demand), metrics.csv, predictions.csv, error_analysis.csv, models/config.json. If a file is missing, raise at startup with a message naming the file and the command that creates it.

Add CORS middleware reading ALLOWED_ORIGINS (comma-separated; empty means no origins), allow_methods GET and POST. Add GET /health -> {"status": "ok"} and GET /disclaimer -> {"text": "Inventory levels are simulated. The dataset contains sales only, not stock-on-hand. Risk flags show how the rule behaves under the stated assumptions, not real stock positions."}.

A helper clean(obj) must convert NaN, inf and numpy types to JSON-safe values (NaN -> null); use it for every response. Never expose CustomerID. Add a first test in tests/test_api.py for /health and /disclaimer using FastAPI TestClient.
```
**Done when:** `uvicorn api.main:app --port 8000` starts and `/health` returns ok.

## Task B7: `/products` and `/forecast/{product_id}`

**SRS:** 7.1

```
Add to api/main.py:

GET /products -> list of {product_id, description, total_units} ordered by total_units descending, where total_units is the product's total over the TRAINING period only (weeks before 2011-10-10). This adds total_units to CONTRACTS.md C7: record it in CHANGE_REQUESTS.md with the reason (the dashboard orders bins by volume) and do not edit CONTRACTS.md.

GET /forecast/{product_id} -> {product_id, description, history: [{week_start, units}] (all 52 weeks, ISO dates), test_predictions: [{week_start, actual, predicted}] for the best model (model_name from config.json) on split "test", next_week_forecast: {week_start: "2011-12-05", forecast_weekly}}. Unknown product -> HTTP 404 with {"detail": "Product not found"}. Include a "closed_weeks" list so the chart can mark the shutdown.

Tests: schema of both responses; 100 products; ordering by total_units; 404 for an unknown id; the history has 52 entries; ISO date strings; no NaN in the JSON text.
```

## Task B8: `/risk` and `/sensitivity`

**SRS:** 7.1, FR-A-2, FR-A-6

```
Add POST /risk: body RiskRequest -> {"params": {lead_time_weeks, service_level, initial_stock_weeks, seed}, "summary": {n_high, n_medium, n_low, avg_days_of_cover}, "items": [...]} where items follow contract C6 columns, rounded to 1 decimal for display fields. Call src.risk_engine.compute_risk and summarize; no formulas in the API layer. Out-of-range input must give HTTP 422 (Pydantic) with a readable message.

Add POST /sensitivity: body SensitivityRequest -> list of {lead_time_weeks, service_level, n_high, n_medium, n_low}, using sensitivity_table with the requested lists. Limit each list to at most 8 values (422 otherwise).

Cache both by their parameter tuple (functools.lru_cache or a dict) so repeated slider values return instantly. The cache must not leak state between different parameters.

Tests: default request matches compute_risk called directly; summary counts equal the counts in items; lead_time_weeks 0 and 5 give 422; service_level 0.5 gives 422; the same request twice returns identical JSON; /sensitivity returns 12 rows by default; items contain exactly the C6 columns.
```

## Task B9: `/metrics` and `/error-analysis`

```
Add GET /metrics -> list of {model_name, split, MAE, RMSE, WAPE, bias} from metrics.csv (all models, both splits, NaN -> null).

Add GET /error-analysis -> {"segments": [...], "weeks": [...], "products": [...]} built from error_analysis.csv by its level column (segment, week, product). Record /error-analysis in CHANGE_REQUESTS.md as a proposed addition to C7 (needed by the dashboard's Model performance page); do not edit CONTRACTS.md.

Tests: /metrics includes the baselines and every model in metrics.csv; /error-analysis has the three blocks with 3 segments and 8 weeks and 100 products; values are JSON-safe.
```

## Task B10: Error handling, security and robustness

**SRS:** FR-A-3, FR-A-5, FR-A-7, NFR-8

```
Harden the API:
1. Add exception handlers so every error returns {"detail": "..."} with the right status (404, 422, 500) and NEVER a stack trace. A deliberately triggered internal error in a test must return 500 with a generic message and log the details server-side.
2. Confirm CORS: with ALLOWED_ORIGINS unset no CORS headers are sent; with ALLOWED_ORIGINS=http://localhost:5173 a request from that origin gets the header and another origin does not.
3. Confirm no response contains the strings CustomerID or customer_id.
4. Add basic request logging (method, path, status, duration ms) and an startup log of artifact file names and modification times.
5. Add a test that scans every JSON endpoint for NaN or Infinity tokens.
```

## Task B11: Static dashboard mount (single URL)

**SRS:** FR-A-9, 11.3 Option A

```
At the VERY END of api/main.py, mount the dashboard folder as static files at "/" with html=True: app.mount("/", StaticFiles(directory="dashboard", html=True), name="dashboard"). It must be registered AFTER every API route so routes keep priority. If dashboard/ does not exist, skip the mount with a log warning instead of crashing.

Tests: GET /health and GET /docs and GET /openapi.json still work; GET / returns the dashboard index.html when dashboard/index.html exists (create a temporary one in the test); a POST to /risk is not shadowed by the static mount.
```

## Task B12: Complete the API test suite and run the contract check

```
Finish tests/test_api.py so it covers every endpoint and every case in SRS section 10.2 (T-A-1 to T-A-3): schema and status codes, 404 for an unknown product, 422 for out-of-range risk parameters, no CustomerID. Run /contract-check and fix mismatches between API responses and CONTRACTS.md C7 (except the two documented additions). Run pytest -q for the whole project and report the totals.
```
**Done when:** the whole suite is green. Commit: `feat(api): FastAPI service with static dashboard mount`.

## Task B13: API documentation

```
Add OpenAPI tags and summaries to every endpoint, example request and response bodies in the Pydantic schemas (use json_schema_extra), and a README section "API" listing each endpoint with a curl example and the expected response shape. Check that /docs renders every endpoint with examples.
```

## Task B14: Performance check

**SRS:** NFR-1

```
Write scripts/bench_api.py that starts the app with TestClient (or a local server) and times, over 50 calls each: POST /risk with default parameters (warm), POST /risk with random valid parameters, GET /forecast/{id}, GET /products. Print median and 95th percentile in ms. Targets: /risk under 500 ms warm, /forecast under 300 ms. If a target is missed, profile and fix (for example precompute per-product forecast payloads at startup), then re-measure and show before and after.
```
**Done when:** targets are met or the miss is explained. Commit: `perf(api): benchmark and fixes`.
