# PROMPTS.md: Building Shelfwatch with Google Antigravity

Phase-by-phase prompts to implement the SRS (version 2.0) for **Retail Product Demand Estimation and Stockout Risk Analysis**. Copy each prompt into a **new Antigravity conversation**, one phase at a time.

---

## 0. How to work with Antigravity (read once)

**Before you start**
1. Export the SRS from the doc (click the doc name, then Export, then Markdown) and save it in your repo as `docs/SRS.md`. Also keep `PROJECT_CONTEXT.md` and `CONTRACTS.md` in the repo root. Every prompt below tells the agent to read the relevant SRS sections, so it never needs the whole file at once.
2. Put the prototype `shelfwatch_dashboard.html` into `dashboard/prototype/` (Phase 9 uses it as the visual reference).
3. Open the **project folder itself** (not its parent) in Antigravity.

**Settings that work well**
- **Planning mode** for every build phase. The agent first produces a Task List and an Implementation Plan artifact. Read the plan, comment on anything that deviates from the SRS, and only then approve. Use **Fast mode** only for small fixes (typos, renaming, formatting).
- **Terminal policy:** use a reviewed or auto policy, not the fully automatic one. Allow `python`, `pytest`, `pip`, `uvicorn`, `git status/diff/add/commit`. Deny anything that deletes folders, force-pushes, or touches files outside the project.
- **Model:** pick the strongest model offered for Phases 4, 5, 7 and 8 (correctness-critical), and a faster one for Phases 1, 9 and 12.
- **One phase, one conversation.** Long conversations drift. Start fresh and let the prompt carry the context.
- After each phase: run the **acceptance check**, read the **Walkthrough** artifact, then commit on your member branch.

**Always end a phase with this check (paste it if the agent forgets)**

```
Before you finish: run the tests for this phase, show me the exact commands and their output, list every file you created or changed, and list anything in the SRS section I gave you that you did NOT implement. Do not claim success unless the tests ran and passed.
```

**Folder names:** Antigravity guides use `.agents/rules/` and `.agents/workflows/` (some older guides write `.agent/`). The safest route is Customizations, then Rules or Workflows, then "+ Workspace", so the IDE places the files correctly for your version. The prompts below are plain text and work either way.

---

## Phase 0: Workspace setup (everyone, once)

**Owner:** all four members, on the shared repo. **Mode:** Fast.

### 0.1 Prompt: create project identity and rules

```
Set up this workspace for a team ML + web project. Do not write any application code yet.

1. Create GEMINI.md in the project root (short project identity, under 40 lines) containing: project name "Shelfwatch: Retail Product Demand Estimation and Stockout Risk Analysis"; the stack (Python 3.10/3.11, pandas, pyarrow, scikit-learn, LightGBM, scipy, FastAPI, pydantic, uvicorn, pytest, and a plain HTML/CSS/JavaScript dashboard with no framework and no build step); a pointer to docs/SRS.md, PROJECT_CONTEXT.md and CONTRACTS.md as the sources of truth; and the instruction "If code and SRS disagree, stop and ask; never silently change a contract."

2. Create these workspace rules (Always On unless stated), each as its own short markdown file:
   - 01-frozen-decisions.md: weekly granularity, Monday-start weeks; target = units per product per week; horizon = next week; Top 100 products chosen on training period only; full weeks only (2010-12-06 to 2011-11-28, 52 weeks); test = last 8 full weeks (week_start 2011-10-10 to 2011-11-28); train = the 44 weeks before; forecast target week 2011-12-05; random seed 42 everywhere; shop-closed week 2010-12-27 is MISSING data, never zero demand.
   - 02-ml-rules.md: chronological split only, never random; every lag and rolling feature uses shift(1) before rolling; tuning only with TimeSeriesSplit inside the training period; never touch the test set for tuning, imputation fitting, capping or product selection; metrics are MAE, RMSE, WAPE and bias; never MAPE; clip predictions at 0.
   - 03-python-style.md: type hints and docstrings on public functions, PEP 8, pandas not polars, pure functions where possible, no hidden global state, every module has pytest tests in tests/.
   - 04-frontend-rules.md (Model Decision, relevant when editing dashboard/): plain HTML, CSS, JS only; no framework, no build step; all API access through dashboard/js/api.js only; escape all API text (textContent or an escape function), never raw innerHTML; light and dark themes through CSS variables; keyboard accessible; respect prefers-reduced-motion; total size under 150 KB excluding fonts.
   - 05-contract-rules.md: files and endpoints must match CONTRACTS.md exactly (column names, types, order). Any proposed contract change goes into a CHANGE_REQUESTS.md file and waits for the team.

3. Create a .antigravityignore that excludes data/raw/, *.xlsx, *.parquet, models/*.joblib, __pycache__/, .venv/, node_modules/.

4. Create a .gitignore with the same exclusions plus .env and .pytest_cache/.

Show me the list of files you created.
```

### 0.2 Prompt: create reusable workflows (invoke with `/` in chat)

```
Create three workspace workflows (markdown files in the workflows folder):

1. verify: Run `pytest -q`, then print the output. If anything fails, diagnose the root cause, fix it, rerun, and stop only when green or when you need a decision from me. Then list files changed.

2. leakage-review: Review the code I name for data leakage. Check: lag and rolling features use shift(1) before rolling; no statistic (mean, std, imputation value, cap, product selection) is computed using test-period rows; train weeks strictly precede test weeks; the closed week 2010-12-27 is never a target and lags skip it. Output a table: finding, file and line, severity, suggested fix. Do not edit code unless I say so.

3. contract-check: Compare the files in data/processed/ and the API responses against CONTRACTS.md and docs/SRS.md section 7. Report every mismatch in column names, types, order, status codes or field names. Do not edit code.

Show me the three files.
```

**Acceptance check:** `GEMINI.md`, five rule files, two ignore files and three workflows exist. Commit: `chore: antigravity workspace setup`.

---

## Phase 1: Repository scaffold, environment, contracts and mocks (all members)

**SRS sections:** 2.5, 3.5, 4.4, 14.1 step 1. **Mode:** Planning.

```
Read docs/SRS.md sections 3.5 (repository layout), 4.4 (processed data dictionary) and 14.1 step 1, and read CONTRACTS.md.

Task: scaffold the repository and mock data.

1. Create the folder structure from SRS 3.5 (src/, api/, dashboard/ with css/ and js/ subfolders, tests/, notebooks/, models/, data/raw, data/processed, data/mock, docs/). Add __init__.py to src/ and tests/.
2. Create requirements.txt with pinned versions: pandas, numpy, pyarrow, openpyxl, scikit-learn, lightgbm, scipy, joblib, fastapi, pydantic, uvicorn, matplotlib, pytest, httpx. Use versions compatible with Python 3.10 and 3.11. Tell me which versions you pinned and why.
3. Write scripts/make_mocks.py that generates deterministic (seed 42) mock files in data/mock/: weekly_demand.parquet (contract C1: 10 products, 20 Monday weeks, no gaps), closed_weeks.csv (C1b), predictions.csv (C2), metrics.csv (C3), next_week_forecast.csv (C4) for those 10 products, and the same data as JSON files the dashboard can load in mock mode (products.json, forecast_<id>.json, risk.json, sensitivity.json, metrics.json, error_analysis.json, disclaimer.json) matching CONTRACTS.md C7 and SRS section 7.
4. Write tests/test_contracts.py with schema checks for C1 to C4 that run against both data/mock/ and data/processed/ when the latter exists. The C1 check must also verify Mondays only, no duplicate product-week rows, and no gaps inside each product's range.
5. Write a README.md with setup (venv, pip install), how to run the mock generator and tests, and the repository layout.

Do not implement any model, API or dashboard logic yet.
```

**Acceptance check:** `python scripts/make_mocks.py && pytest tests/test_contracts.py -q` passes. Commit: `feat: scaffold, pinned requirements, mocks, contract tests`.

---

## Phase 2: Data pipeline and EDA (Member 1)

**SRS sections:** 4.1 to 4.5 (FR-D-1 to FR-D-10). **Mode:** Planning.

> If you already have `phase1_member1.zip` (the pipeline built earlier), skip the build prompt and use **2B: verify**.

### 2A. Build prompt

```
Read docs/SRS.md sections 4.1 to 4.5 and PROJECT_CONTEXT.md sections 4 and 5.

Implement src/data_prep.py with build_weekly_demand(raw_path, out_path, log_path, top_n=100, test_weeks=8, min_active_train_weeks=30) producing data/processed/weekly_demand.parquet (contract C1) from data/raw/Online Retail.xlsx.

Requirements (all MUST):
- Load the Excel once and cache a parquet copy next to it. Force InvoiceNo, StockCode and Description to string (some Description cells are numeric in the real file; this crashed an earlier version).
- Remove cancellations (InvoiceNo starting with C), Quantity <= 0, UnitPrice <= 0, and non-product StockCodes (POST, D, M, DOT, BANK CHARGES, AMAZONFEE, S, B, CRUK, PADS, C2, codes with no digit, prefixes GIFT and TEST). Keep rows with missing CustomerID.
- Define weeks as Monday-start and keep only FULL weeks, computed from the data range (expected 2010-12-06 to 2011-11-28, 52 weeks).
- Detect shop-wide closed weeks (full weeks with no transactions at all) and write data/processed/closed_weeks.csv.
- Cap line-level Quantity at the 99.9th percentile computed on TRAINING rows only (expected cap about 480).
- Choose the Top 100 products by total units in the TRAINING period only, requiring sales in at least 30 training weeks.
- Aggregate to product x week: units, orders (distinct InvoiceNo), customers (distinct non-null CustomerID), avg_price. Fill missing weeks between each product's first and last active week with zeros and forward-filled price. description = most frequent description.
- Write a cleaning log (row counts after each step) to data/processed/cleaning_log.csv and print it.
- Include validate_c1(df) that raises AssertionError if the contract is violated, and call it before saving.
- CLI: python -m src.data_prep --raw "data/raw/Online Retail.xlsx"

Tests (tests/test_data_prep.py, using a synthetic raw DataFrame with planted junk: a cancellation, POST, a zero price, an 80,995-unit order, a gift voucher): exactly 100 products, Mondays only, no duplicates, no gaps, no partial weeks, no negative units, junk removed, outlier capped, closed-week detection, and top-100 selection unchanged when test-period sales are inflated.

Expected on the real file: 5,147 rows, 100 products, closed week 2010-12-27. If your numbers differ, report the difference and why instead of forcing a match.
```

### 2B. Verify prompt (if the pipeline already exists)

```
src/data_prep.py and tests/test_data_prep.py already exist. Read docs/SRS.md sections 4.3 to 4.5. Audit the code against every requirement FR-D-1 to FR-D-10 and list any gap. Then run the pipeline on data/raw/Online Retail.xlsx and confirm: 5,147 rows, 100 products, 52 weeks from 2010-12-06 to 2011-11-28, closed week 2010-12-27 in closed_weeks.csv, cap about 480. Run pytest and report. Fix only real gaps and explain each fix.
```

### 2C. EDA prompt

```
Create notebooks/01_eda.py (cell format with # %% markers, runnable top to bottom) using data/processed/weekly_demand.parquet, closed_weeks.csv and the raw cleaned transactions. Produce, saving figures to reports/figures/: cleaning steps bar chart; total weekly units with the test start and the closed week marked; top 15 products; share of zero-demand weeks per product; weekly units distribution (raw and log); volume segments (low, medium, high tertiles of average weekly units, saved to data/processed/product_segments.csv); coefficient of variation per product; country share; effect of the quantity cap. Under each section add a markdown note "What to write in the report". After running, tell me what the data shows about: the level shift in October to November, the closure week, and sparsity.
```

**Acceptance check:** `pytest tests/test_data_prep.py -q`; open the figures. Commit: `feat(data): weekly demand pipeline and EDA`.

---

## Phase 3: Feature engineering (Member 2)

**SRS sections:** 5.1 (FR-F-1 to FR-F-9). **Mode:** Planning. **Strongest model.**

```
Read docs/SRS.md section 5.1 and PROJECT_CONTEXT.md sections 4 to 6. Input: data/processed/weekly_demand.parquet and closed_weeks.csv (use data/mock/ only if the real files do not exist).

Implement src/features.py with:
- build_features(weekly, closed_weeks) -> DataFrame with one row per product per target week t and a column "target" = units at t.
- FEATURE_COLUMNS: the ordered list of feature names.

Features (all computed ONLY from weeks before t):
- lag_1, lag_2, lag_3, lag_4 of units
- roll_mean_4, roll_std_4, roll_mean_8, roll_std_8: ALWAYS apply shift(1) BEFORE rolling
- week_of_year, month
- prev_avg_price, prev_orders, prev_customers (previous week values)
- prod_past_mean, prod_past_std: expanding mean and std of the product's units strictly before t
- weeks_to_christmas as a SEPARATE optional column list (OPTIONAL_FEATURES), not in FEATURE_COLUMNS by default, because with one year of data it behaves like a time index; it will be tested by ablation in the next phase.

Closed week 2010-12-27: treat as MISSING (NaN) before computing anything. It must never be a target row, and lags and rolling statistics must skip it (rolling uses min_periods=2). Do not impute inside build_features; imputation belongs in the model pipeline and must be fitted on training data only.

Drop rows lacking enough history for the longest window and print how many rows were dropped per product and in total.

Tests in tests/test_features.py (all required):
1. LEAKAGE TEST: take a weekly table, build features, then change units at week t and all later weeks for a product, rebuild, and assert the features at week t are identical.
2. The closed week is never a target, and lag_1 for the week after the closure is NaN or skips correctly (state which rule you chose and why).
3. Feature columns match the saved FEATURE_COLUMNS order.
4. No feature uses the same-week value of units, orders, customers or price.
5. Row counts: report training rows (expected roughly 3,000 to 3,500) and test rows (expected 800).

Finish by running /leakage-review on src/features.py and show the table.
```

**Acceptance check:** `pytest tests/test_features.py -q`; the leakage test passes. Commit: `feat(features): leakage-safe features with closed-week handling`.

---

## Phase 4: Models, tuning and evaluation (Member 2)

**SRS sections:** 5.2 (FR-M-1 to FR-M-10). **Mode:** Planning. **Strongest model.**

```
Read docs/SRS.md section 5.2 and CONTRACTS.md contracts C2, C3, C4, C5. Input: output of src/features.py.

Implement src/train.py. Split: training = target weeks before 2011-10-10; test = the 8 weeks from 2011-10-10 to 2011-11-28.

1. Baselines: naive (last non-closed week) and 4-week moving average. These need no fitting.
2. Models on identical features and splits: Ridge (with scaling and imputation inside a sklearn Pipeline, fitted on training only), RandomForestRegressor, LGBMRegressor (native NaN handling). Optional Poisson regression is a stretch goal; skip unless everything else is done.
3. Tuning: TimeSeriesSplit inside the training period only, random search with a fixed budget of 30 trials per model, seed 42. Search spaces: Ridge alpha (log range); Random Forest n_estimators, max_depth, min_samples_leaf; LightGBM num_leaves, learning_rate, n_estimators, min_child_samples, subsample. Score by WAPE.
4. Produce out-of-fold 'val' predictions from the time-series folds for every model, and 'test' predictions from models refit on the full training period. Clip predictions at 0.
5. evaluate(y_true, y_pred) returns MAE, RMSE, WAPE (sum abs error / sum actual) and bias (mean predicted minus actual). Never MAPE.
6. Run an ablation: with and without OPTIONAL_FEATURES (weeks_to_christmas) and with and without calendar features; report validation WAPE for each and decide whether to keep them.
7. Select the best model by validation WAPE, ties broken by validation bias. Refit on the full training period, save with joblib to models/best_model.joblib, write models/feature_list.json and models/config.json (model name, horizon 1, split dates, seed 42, library versions).
8. Write contract files: data/processed/predictions.csv (C2, all models, with split column), metrics.csv (C3, val and test), next_week_forecast.csv (C4) for week_start 2011-12-05 where sigma_error = per-product std of the best model's out-of-fold validation errors (if a product has fewer than 6 validation observations use the pooled std of its volume segment from data/processed/product_segments.csv), and avg_weekly_units_train.
9. IMPORTANT known risk: October to November demand sits above most of the training range and tree models cannot predict above what they saw. Report the weekly bias of every model across the 8 test weeks and say explicitly whether tree models under-forecast. If they do, try a ratio-target variant (target / roll_mean_4) and report it as a separate candidate.

Rules: never tune on test; report honestly if no model beats the baselines. Add tests in tests/test_models.py: split dates strictly chronological; metric functions match hand-computed values on a tiny example; saved model reloads and reproduces stored predictions; predictions finite and >= 0; C2 to C4 schema checks.

At the end print: a metrics table (validation and test) for all models, the chosen model and why, and a plain-language paragraph I can paste into the report about what beat what.
```

**Acceptance check:** `pytest tests/test_models.py -q`; files C2 to C5 exist; the contract tests pass. Commit: `feat(model): baselines, ML models, tuning, artifacts`.

---

## Phase 5: Forecast-error analysis (Member 1)

**SRS sections:** 5.3 (FR-E-1 to FR-E-6). **Mode:** Planning.

```
Read docs/SRS.md section 5.3. Inputs: data/processed/predictions.csv (C2), metrics.csv, product_segments.csv, weekly_demand.parquet.

Create notebooks/03_error_analysis.py (cell format) and src/error_analysis.py with a function build_error_analysis(predictions, segments) that writes data/processed/error_analysis.csv with three tidy blocks distinguished by a "level" column: segment (low, medium, high), week (the 8 test weeks), product (all products). Columns: level, key, model_name, n, MAE, RMSE, WAPE, bias, over_forecast_units, under_forecast_units.

The notebook must cover, for the best model and the best baseline: error by volume segment; error by test week with a comment on spike weeks; predicted vs actual scatter and residual histogram; the 10 best and 10 worst products with a one-line reason each (intermittent demand, bulk orders, level shift); over-forecast versus under-forecast totals separately, with a sentence on why under-forecasting matters for stockouts. Save figures to reports/figures/. Finish with a paragraph labelled "For the report: findings" listing 5 concrete findings with numbers. Tests: error_analysis.csv schema and that segment-level numbers reproduce from predictions.csv.
```

**Acceptance check:** `error_analysis.csv` written; the notebook runs top to bottom. Commit: `feat(analysis): forecast-error analysis`.

---

## Phase 6: Risk engine (Member 3)

**SRS sections:** 6.1 to 6.4, Appendix A (FR-R-1 to FR-R-9). **Mode:** Planning. **Strongest model.**

```
Read docs/SRS.md section 6 and Appendix A, and CONTRACTS.md C6. Input: next_week_forecast.csv (C4) plus a description column joined from weekly_demand.parquet (use data/mock/ until real files exist).

Implement src/risk_engine.py:

compute_risk(forecast_df, params=None) -> DataFrame with columns product_id, description, forecast_weekly, forecast_lead_time, on_hand, safety_stock, reorder_point, days_of_cover, risk_level.
params defaults: lead_time_weeks=1 (integer 1 to 4), service_level=0.95 (0.80 to 0.99), initial_stock_weeks=3.0 (0.5 to 8), seed=42. Invalid or out-of-range params raise ValueError.

Formulas (authoritative, exactly as in SRS 6.2):
1. on_hand = initial_stock_weeks x avg_weekly_units_train x noise, where noise is drawn once per product from Uniform(0.5, 1.5) using numpy default_rng(seed) over products sorted by product_id. Noise must NOT depend on any other parameter.
2. forecast_lead_time = forecast_weekly x lead_time_weeks
3. z = scipy.stats.norm.ppf(service_level)
4. safety_stock = z x sigma_error x sqrt(lead_time_weeks)
5. reorder_point = forecast_lead_time + safety_stock
6. days_of_cover = on_hand / (forecast_weekly / 7); if forecast is 0 then 999
7. HIGH if on_hand < forecast_lead_time; MEDIUM if on_hand < reorder_point and not HIGH; LOW otherwise. Equality is NOT below the threshold. Compare unrounded values.

Also implement sensitivity_table(forecast_df, lead_times=(1,2,3,4), service_levels=(0.90,0.95,0.99), initial_stock_weeks=3.0) -> columns lead_time_weeks, service_level, n_high, n_medium, n_low, and summarize(df) -> n_high, n_medium, n_low, share_at_risk, avg_days_of_cover (cap 999 values out of the average).

Tests in tests/test_risk_engine.py:
- The worked example in SRS Appendix A: forecast_weekly 100, sigma_error 20, avg_weekly_units_train 90, initial_stock_weeks 3, L 2, service 0.95. With noise forced to 1.1, 0.8, 0.6 the classes must be LOW, MEDIUM, HIGH with days of cover 20.8, 15.1, 11.3, safety stock 46.5 and reorder point 246.5 (allow a hook to inject noise for the test).
- Edge cases: forecast 0, sigma 0, L 4, service 0.99, on_hand exactly equal to a threshold, empty input.
- Determinism with the same seed; changing lead time or service level does NOT change on_hand.
- Invalid params raise ValueError.

Then create notebooks/04_sensitivity.py that produces the sensitivity table and a heatmap on the real forecast, and a short written interpretation. State honestly how many products are HIGH and MEDIUM at the defaults; do not tune the defaults to produce a desired count.
```

**Acceptance check:** `pytest tests/test_risk_engine.py -q`. Commit: `feat(risk): configurable stockout risk engine and sensitivity`.

---

## Phase 7: FastAPI backend (Member 3)

**SRS sections:** 7.1, 7.2, 14.3 (FR-A-1 to FR-A-9). **Mode:** Planning. **Strongest model.**

```
Read docs/SRS.md sections 7 and 14.3, and CONTRACTS.md C7. Inputs: files in data/processed/ and models/, and src/risk_engine.py.

Implement api/main.py (and api/schemas.py for the Pydantic models) with a lifespan handler that loads, once at startup: weekly_demand.parquet, next_week_forecast.csv (joined with descriptions), metrics.csv, predictions.csv, error_analysis.csv. If any file is missing, fail at startup with a clear message naming the file.

Endpoints, exactly per SRS 7.1:
- GET /health, GET /disclaimer ({"text": "Inventory levels are simulated. The dataset contains sales only, not stock-on-hand. Risk flags show how the rule behaves under the stated assumptions, not real stock positions."})
- GET /products -> [{product_id, description, total_units}] ordered by total_units descending (training-period total)
- GET /forecast/{product_id} -> history, test_predictions (best model), next_week_forecast; 404 {"detail": "Product not found"} when unknown
- POST /risk with RiskRequest (lead_time_weeks int 1 to 4, service_level 0.80 to 0.99, initial_stock_weeks 0.5 to 8) -> {params (with seed), summary {n_high, n_medium, n_low, avg_days_of_cover}, items}; 422 on invalid input
- POST /sensitivity with {lead_times, service_levels} -> list of rows
- GET /metrics, GET /error-analysis

Behaviour requirements: JSON must contain no NaN (convert to null or fill); dates are ISO YYYY-MM-DD; never expose CustomerID; errors use {"detail": "..."} with no stack traces; CORS allow-list from the ALLOWED_ORIGINS environment variable (empty means none); results of /risk and /sensitivity cached by parameter set; stateless.

LAST line of the file: mount the dashboard folder as static files at "/" with html=True, registered AFTER all API routes so API routes keep priority.

Tests in tests/test_api.py with FastAPI TestClient: every endpoint returns the documented schema and status code; unknown product gives 404; out-of-range lead_time_weeks gives 422; /risk is deterministic; the response has no CustomerID; the static mount does not shadow /health or /docs.

Note: /products total_units and /error-analysis are additions to CONTRACTS.md. Add both to CHANGE_REQUESTS.md and do not edit CONTRACTS.md; tell me they are pending team agreement.
```

**Acceptance check:** `pytest tests/test_api.py -q`; `uvicorn api.main:app --port 8000`, then open `http://localhost:8000/docs`. Commit: `feat(api): FastAPI service with static dashboard mount`.

---

## Phase 8: Dashboard (Member 4)

**SRS sections:** 8.1 to 8.4 (FR-U-1 to FR-U-14). **Mode:** Planning. Antigravity can drive a browser to check the page if you installed its browser extension.

### 8A. Split the prototype into files

```
Read docs/SRS.md sections 8.1 to 8.4 and open dashboard/prototype/shelfwatch_dashboard.html (the visual reference).

Create the production dashboard in dashboard/: index.html, config.js, css/style.css, js/api.js, js/charts.js, js/app.js. Keep the prototype's look exactly: the shelf wall of bins, hazard-tape disclaimer strip, flat panels, Big Shoulders Display for headline and numbers, Public Sans for text (Google Fonts with Arial Narrow, Impact and system-ui fallbacks), light and dark themes through CSS variables with the tokens from SRS 8.4, a manual theme toggle, hash routes #overview, #explorer, #risk, #models, #sensitivity.

Rules: plain HTML, CSS and JS only; no framework, no build step; total under 150 KB excluding fonts; keyboard operable with visible focus; honor prefers-reduced-motion; every piece of API text inserted through textContent or an escape function.

For this step keep the data in mock mode only: config.js sets window.SHELFWATCH = { apiBase: "mock" } and js/api.js reads data/mock JSON files copied or served from dashboard/mock/. Do not call a real API yet. Remove all risk calculation code from the UI: the UI only renders what api.js returns.
```

### 8B. API client and behaviour

```
Implement js/api.js as the ONLY module that talks to the API, exposing getProducts, getForecast(id), getRisk(params), getSensitivity(params), getMetrics, getErrorAnalysis, getDisclaimer. Support three apiBase values: "mock" (read JSON from dashboard/mock/), "" (same origin) and a full URL.

Behaviour: debounce slider-driven getRisk calls by 150 ms; abort a stale request with AbortController when a newer one starts; cache results per parameter set; map HTTP 422 and network errors to a friendly message; show loading, empty and error states with a Retry button (FR-U-5).

Shelf page (FR-U-11, FR-U-12): one bin per product, ordered by total_units; fill height = on_hand / forecast_weekly in weeks, clamped 0 to 6; red dashed line at forecast_lead_time / forecast_weekly; yellow tick at reorder_point / forecast_weekly; colour AND icon AND text for risk level; bins are updated in place by changing CSS variables so the refill transition runs on slider changes; the headline states the live count of HIGH-risk products from summary.n_high; selecting a bin opens that product's forecast page.

Other pages: Product forecast (SVG line chart with shaded test weeks, closed-week marker, hover crosshair showing date, actual and prediction, next-week forecast marker, six stat tiles), Risk table (sort any column, filter by risk level with counts, CSV download, row click opens the product), Model performance (metrics table, per-week error bars from /error-analysis, volume-segment error), Sensitivity (heatmap with current setting outlined, assumptions list). The disclaimer from /disclaimer is visible on every page without scrolling.
```

### 8C. Verify in the browser

```
Start the app with "uvicorn api.main:app --port 8000" (or serve dashboard/ with mock mode if the API is not ready). Using the browser, visit each of the five pages, move each of the three sliders, toggle the theme, and use only the keyboard to reach every control. Take screenshots of each page in light and dark mode and attach them. Check: the bin count equals the product count; every bin colour matches its risk_level; the disclaimer is visible on every page; there are no console errors; nothing overflows horizontally at phone width (about 390 px). List every problem you find, fix them, and re-verify. Do not change the visual design.
```

**Acceptance check:** manual checklist T-U-1 to T-U-4 (SRS 10.2). Commit: `feat(dashboard): Shelfwatch HTML/CSS/JS dashboard`.

---

## Phase 9: Switch the dashboard to the live API and integrate (Member 3 + 4)

**SRS sections:** 8.3, 10, 14.1 step 10. **Mode:** Planning.

```
Switch dashboard/config.js to apiBase "" (same origin). Run the full stack with: python -m src.data_prep (skip if already done), python -m src.train (skip if artifacts exist), then uvicorn api.main:app --port 8000. Open http://localhost:8000/.

Tasks:
1. Run /contract-check and fix mismatches between api.js expectations and the real API responses (field names, nulls, ordering). Do not change API contracts; if one must change, add it to CHANGE_REQUESTS.md and tell me.
2. Confirm the dashboard shows real numbers: headline counts equal the /risk summary; the Product forecast page chart matches /forecast/{id}; Model performance matches metrics.csv; Sensitivity matches /sensitivity.
3. Add tests: a jsdom or Playwright smoke test (optional if not installable) that loads the page against the TestClient or a running server and asserts bin count equals product count and each bin class matches risk_level; keep it skipped with a clear message when the tool is unavailable.
4. Run the entire test suite with pytest -q and print the summary. Report any warnings.
5. Write docs/INTEGRATION_REPORT.md: what was tested, what passed, what failed and how it was fixed.
```

**Acceptance check:** `pytest -q` fully green; the dashboard works at `http://localhost:8000/`. Commit: `test: end-to-end integration`.

---

## Phase 10: Deployment (Member 4)

**SRS sections:** 11 (Option A: one service). **Mode:** Planning.

```
Read docs/SRS.md section 11. Prepare Option A: one service where FastAPI serves both the API and the dashboard.

1. Create a Dockerfile (python:3.11-slim, install requirements.txt, copy only what is needed: api/, src/, dashboard/, models/, data/processed/; expose the PORT environment variable; start with uvicorn api.main:app --host 0.0.0.0 --port $PORT). Add .dockerignore.
2. Confirm total artifact size is under 50 MB (SRS NFR-10) and report the size of each folder that is copied.
3. Make the app start cleanly on a machine with no data/raw folder (it must only need data/processed and models). Test this by temporarily running from a clean clone.
4. Add a render.yaml (or document the equivalent settings) and a README section "Deploy": steps for Render or a Hugging Face Docker Space, environment variables (ALLOWED_ORIGINS optional, PORT provided by the host), and how to check /health, /docs and the dashboard on the public URL.
5. Add a "How to run locally" section with the three commands from SRS 14.4.
6. Write docs/RELEASE_CHECKLIST.md from SRS section 11.4 and tick what is already verified.

Do not deploy anything yourself. Tell me what I must click or paste in the hosting service.
```

**Acceptance check:** `docker build` succeeds (if Docker is installed); health check works locally. Deploy a `/health`-only hello-world by Day 8, then the full app. Commit: `chore: deployment setup`.

---

## Phase 11: Final review, report and viva (all)

**Mode:** Planning for the reviews; Fast for formatting.

### 11A. Whole-project review

```
Act as a strict examiner and senior engineer. Read docs/SRS.md and review the whole repository against it. Produce docs/REVIEW.md with: (1) a traceability table mapping every FR and NFR to the file that implements it and the test that verifies it, marking any requirement with no implementation or no test; (2) a list of deviations from the SRS with severity; (3) data-leakage risks (run /leakage-review on src/features.py and src/train.py); (4) places where the simulated-inventory disclaimer is missing or weak; (5) anything in the code that overstates what the results show. Do not edit code; only report.
```

### 11B. Report draft

```
Using docs/SRS.md, the cleaning log, metrics.csv, error_analysis.csv, the sensitivity results and the figures in reports/figures/, draft docs/REPORT.md for a college submission with sections: Problem and objectives, Dataset and cleaning (use the real row counts), EDA findings, Feature engineering (explain shift-before-rolling and the closed-week rule), Models and evaluation (baselines versus ML, with the real numbers), Forecast-error analysis, Inventory simulation (state every assumption and the disclaimer), Dashboard and API, Testing, Deployment, Limitations (one year of data, level shift, intermittent demand, simulated stock), and Future work. Use only numbers that exist in the project files; mark any number you could not find as TODO instead of inventing it.
```

### 11C. Slides outline

```
From docs/REPORT.md create docs/SLIDES.md: 12 slides maximum, each with a title, 3 to 5 short bullets, and a note on which figure to show. Include one slide on the shelf-wall dashboard and one on honest limitations.
```

### 11D. Viva rehearsal

```
Act as a college project examiner. Ask me 10 hard questions about this project one at a time (data leakage, why weekly granularity, why WAPE and not MAPE, the closed week, tree models under-forecasting, simulated inventory, safety stock and reorder point, why a static dashboard served by FastAPI). After each of my answers, grade it out of 10 and tell me what a stronger answer would include. Wait for my answer before asking the next question.
```

---

## Appendix A: Prompts for when things go wrong

**The agent drifted from the SRS**

```
Stop. Compare what you just did with docs/SRS.md section [X]. List each place you deviated, undo the deviations, and re-do the work to match the SRS. If you believe the SRS is wrong, say so and wait; do not change it.
```

**A test fails**

```
Run pytest -q for [file]. Show the failing output. Explain the root cause in two sentences before changing anything. Fix the code, not the test, unless the test itself contradicts the SRS. Re-run and show that it passes.
```

**Suspiciously good model results**

```
The test error looks too good. Run /leakage-review on src/features.py and src/train.py. Then check that no statistic used in training (imputation values, scaler, cap, product selection, hyperparameters) was computed using test-period rows, and that the test set was used once, only for final reporting. Report findings with file and line.
```

**Merge conflict or broken environment**

```
Show git status and the last 5 commits. Do not run destructive git commands (reset --hard, clean -f, force push). Explain the conflict and propose a resolution; apply it only after I confirm.
```

**Add a feature safely**

```
Before changing code, write the change as a short plan: files to touch, contract impact, tests to add. Wait for my approval. Then implement, run /verify, and run /contract-check.
```

---

## Appendix B: Commit and branch convention

- Branches: `m1-data`, `m2-model`, `m3-risk-api`, `m4-dashboard`; merge into `main` through a pull request reviewed by a teammate.
- Commit after every phase using the message shown in that phase.
- Never commit `data/raw/`, `*.xlsx`, or secrets.
- Contract changes only through `CHANGE_REQUESTS.md`, agreed by all four members, then recorded in the change log of `CONTRACTS.md`.
