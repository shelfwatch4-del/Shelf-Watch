# Software Requirements Specification

## Retail Product Demand Estimation and Stockout Risk Analysis

Version 2.0 | 9 November 2026 | Prepared by the project team (4 members) | Status: Final for implementation

---

## 1. Introduction

### 1.1 Purpose

This document specifies the requirements, architecture, data, algorithms, interfaces, quality attributes and test strategy for a web-based system that (a) forecasts weekly retail product demand from historical sales and (b) flags products that may need inventory attention using an explicit, configurable stockout-risk rule. It is the single reference for the four team members and for every Claude chat used while building the project.

### 1.2 Scope

**In scope**

- Cleaning the UCI Online Retail transactions and aggregating them to product x week demand
- Time-based feature engineering with strict no-leakage rules
- Comparison of baselines and regression models, evaluated on later (held-out) weeks
- Product-level demand estimates for a defined horizon (next week)
- Forecast-error analysis (by volume segment, by week, by product, over- versus under-forecast)
- A simulated-inventory model and a configurable stockout-risk indicator with sensitivity analysis
- A FastAPI backend and an interactive web dashboard built with HTML, CSS and JavaScript, deployed with a local or public URL

**Out of scope**

- Real inventory, supplier, or purchasing integrations (the dataset has no stock-on-hand records)
- Automatic purchase-order creation
- Multi-step (beyond one week) production-grade forecasting; a 4-week horizon is an optional extension only
- User accounts, authentication and multi-tenant features
- Real-time data ingestion

### 1.3 Definitions and abbreviations

| Term | Meaning |
| --- | --- |
| SKU / product | A product identified by `StockCode` (called `product_id` after cleaning) |
| Week | Monday-to-Sunday week, identified by its Monday date `week_start` |
| Demand | Total units sold of a product in a week (`units`) |
| Horizon | How far ahead the model predicts. Here: one week (t+1) |
| Lag feature | A past value of demand, for example demand one week earlier |
| Data leakage | Using information from the target week or later to build a feature or tune a model |
| WAPE | Weighted Absolute Percentage Error = sum(abs error) / sum(actual) |
| Bias | Mean of (predicted minus actual). Positive means over-forecasting |
| On-hand | Units of a product in stock at the start of the forecast week (simulated) |
| Lead time (L) | Weeks between placing and receiving a replenishment order |
| Safety stock | Extra stock held to absorb forecast error during the lead time |
| Reorder point (ROP) | Stock level at which a replenishment order should be placed |
| Days of cover | How many days current stock lasts at the forecast sales rate |
| Service level | Target probability of not running out during the lead time |
| API | The FastAPI service. UI means the dashboard |

### 1.4 References

- UCI Machine Learning Repository, Online Retail dataset (id 352): https://archive.ics.uci.edu/dataset/352/online+retail
- `PROJECT_CONTEXT.md` (frozen design decisions) and `CONTRACTS.md` (data and interface contracts C1 to C7) in the project repository
- Problem statement supplied by the course

### 1.5 Conventions

Requirement IDs follow `FR-<area>-<n>` (functional) and `NFR-<n>` (non-functional). Priority words: **MUST** (mandatory for acceptance), **SHOULD** (expected unless there is a documented reason), **MAY** (optional extension).

---

## 2. Overall Description

### 2.1 Product perspective

The system is a self-contained, offline-trained analytics application. Training happens once in Python notebooks and scripts and produces versioned artifacts (a parquet table, CSV result files, a serialized model). A FastAPI service loads those artifacts at start-up and serves forecasts and risk results. A web dashboard built with plain HTML, CSS and JavaScript (no framework, no build step) consumes only the API. There are no external runtime dependencies other than the Python libraries and web fonts loaded from a CDN.

### 2.2 Product functions (summary)

1. **Prepare data:** clean raw transactions and build the weekly demand table.
2. **Engineer features:** lags, rolling statistics, calendar and commercial signals, without leakage.
3. **Train and compare models:** two baselines plus Ridge, Random Forest and gradient boosting; tune with time-series cross-validation.
4. **Estimate demand:** next-week forecast for each of the 100 selected products.
5. **Analyse errors:** quantify where and why forecasts fail.
6. **Assess stockout risk:** apply an explicit rule to simulated inventory with user-adjustable assumptions.
7. **Explain and explore:** present everything in a dashboard with a visible simulated-inventory disclaimer.

### 2.3 User classes

| User | Goal | Technical level |
| --- | --- | --- |
| Inventory planner (primary persona) | See which products need attention, change assumptions and see risk change | Non-technical |
| Examiner / evaluator | Verify methodology, metrics, honesty of assumptions | Technical |
| Development team | Run, test, extend and deploy the system | Technical |

### 2.4 Operating environment

- Python 3.10 or 3.11 on Windows, macOS or Linux
- Modern desktop browser (Chrome, Edge, Firefox, Safari, latest two versions); the dashboard SHOULD remain usable at tablet width
- Local execution is mandatory; cloud hosting (Render or a Hugging Face Docker Space, with Netlify or GitHub Pages as an option for a separate dashboard host) is the target for the public URL

### 2.5 Design and implementation constraints

- **C-1:** Chronological evaluation only. Random train/test splits are prohibited.
- **C-2:** All features at week t use information from weeks before t only.
- **C-3:** Seed 42 for every random process.
- **C-4:** Free-tier hosting limits (limited memory, cold starts). The model and data files MUST stay small (target under 50 MB in total).
- **C-5:** Delivery within 1.5 to 2 weeks by four people, so scope is deliberately focused (100 products, weekly granularity, one-week horizon).
- **C-6:** Library versions are pinned so the saved model loads identically on every machine.

### 2.6 Assumptions and dependencies

- **A-1:** The UCI file `Online_Retail.xlsx` is available to every member and is not committed to Git.
- **A-2:** Inventory-on-hand is simulated (section 7). Risk results are illustrative, not real stock positions.
- **A-3:** One year of history cannot reveal yearly seasonality. Calendar features are treated as hypotheses to be tested, not facts.
- **A-4:** The forecast target week is `2011-12-05`, the first week after the last full week of data.
- **A-5:** The shop-wide closure week `2010-12-27` is a closure, not zero demand.
- Dependencies: pandas, numpy, scikit-learn, LightGBM (or XGBoost), scipy, FastAPI, uvicorn, pydantic, pytest, joblib, pyarrow, openpyxl.

---

## 3. System Architecture

### 3.1 Logical architecture

```
  OFFLINE (run once, reproducible)                 ONLINE (always running)

  Online_Retail.xlsx                                 +----------------------+
        |                                            |  Web dashboard (JS)  |
        v  src/data_prep.py                          |  js/api.js only      |
  weekly_demand.parquet  closed_weeks.csv            +----------+-----------+
        |                                                       | HTTP/JSON
        v  src/features.py + src/train.py                       v
  models/best_model.joblib, feature_list.json,       +----------------------+
  config.json                                        |  FastAPI service      |
  predictions.csv, metrics.csv,                      |  api/main.py          |
  next_week_forecast.csv, error_analysis.csv  -----> |  loads artifacts at   |
        |                                            |  start-up             |
        v  src/risk_engine.py  <---------------------+  calls risk_engine    |
  sensitivity tables                                 +----------------------+
```

### 3.2 Components and responsibilities

| Component | Owner | Responsibility |
| --- | --- | --- |
| Data pipeline (`src/data_prep.py`) | M1 | Clean, aggregate, validate against contract C1 |
| EDA and error analysis notebooks | M1 | Insight, report figures |
| Feature module (`src/features.py`) | M2 | Leakage-safe features |
| Training module (`src/train.py`) | M2 | Baselines, models, tuning, evaluation, artifact export |
| Risk engine (`src/risk_engine.py`) | M3 | Pure, deterministic risk and sensitivity functions |
| API (`api/main.py`) | M3 | Endpoints, validation, serving artifacts |
| Dashboard (`dashboard/`) | M4 | Presentation and interaction; no business logic |
| Tests, deployment, documents | M4 (all review) | Quality gate and delivery |

### 3.3 Design principles

1. **Contracts first:** components communicate only through the files and endpoints in `CONTRACTS.md`.
2. **Separation of concerns:** the dashboard displays, the API serves, the risk engine computes, the training code learns. The dashboard never recomputes risk.
3. **Pure functions:** `compute_risk` and `sensitivity_table` have no hidden state, so they are trivial to test.
4. **Reproducibility:** fixed seeds, pinned versions, saved configuration (including library versions and split dates).
5. **Fail loud:** schema validation at every hand-off (`validate_c1` and equivalents).

### 3.4 Technology stack

| Layer | Choice | Reason |
| --- | --- | --- |
| Language | Python 3.10/3.11 | One language across data, ML and API |
| Data | pandas, pyarrow (parquet) | Fast reloads, typed columns |
| ML | scikit-learn, LightGBM | Standard, fast on small tabular data |
| Stats | scipy (`norm.ppf`) | z-score from service level |
| API | FastAPI, pydantic, uvicorn | Automatic validation and OpenAPI docs at `/docs` |
| UI | HTML, CSS, JS | Beautiful UI and interactive Dashboard to attract the users and make their work easy |
| Testing | pytest, FastAPI TestClient | Unit, API and schema tests |
| Hosting | Render for the API, which also serves the dashboard files; Netlify or GitHub Pages if the dashboard is hosted separately; ngrok as a fallback | Free tiers with public URLs |

### 3.5 Repository layout

```
project/
  PROJECT_CONTEXT.md  CONTRACTS.md  SRS (this document)  README.md
  data/raw/ (not committed)  data/processed/  data/mock/
  notebooks/  01_eda  02_modeling  03_error_analysis  04_sensitivity
  src/        data_prep.py  features.py  train.py  risk_engine.py
  models/     best_model.joblib  feature_list.json  config.json
  api/        main.py  schemas.py
  dashboard/  index.html  config.js  css/style.css  js/app.js  js/api.js  js/charts.js
  tests/      test_data_prep.py  test_features.py  test_models.py
              test_risk_engine.py  test_api.py  test_contracts.py
  requirements.txt  Dockerfile (optional)
```

---

## 4. Data Requirements

### 4.1 Source dataset

UCI Online Retail: a UK-based online gift retailer, transactions from 1 December 2010 to 9 December 2011.

| Raw column | Type | Notes |
| --- | --- | --- |
| `InvoiceNo` | text | Prefix `C` marks a cancellation |
| `StockCode` | text | Product code. Contains non-product codes (`POST`, `DOT`, `M`, ...) |
| `Description` | text | Some cells are numeric or missing in the real file |
| `Quantity` | int | Negative for cancellations and adjustments |
| `InvoiceDate` | datetime | Transaction timestamp |
| `UnitPrice` | float | Some zero or negative values |
| `CustomerID` | float | About 25% missing |
| `Country` | text | About 83.5% of units are UK |

### 4.2 Measured data characteristics (from the real file)

| Pipeline step | Rows remaining |
| --- | --- |
| Raw | 541,909 |
| After removing cancellations (9,288) | 532,621 |
| After removing Quantity <= 0 (1,336) | 531,285 |
| After removing UnitPrice <= 0 (1,181) | 530,104 |
| After removing non-product codes (2,370) | 527,734 |
| After dropping partial weeks (27,348) | 500,386 |
| After keeping Top 100 products | 76,648 |

Other measured facts: the line-level quantity cap is 480 units (99.9th percentile of training rows; 415 lines capped); the largest raw single lines are 74,215 and 80,995 units; 1,224 products have at least 30 active training weeks; the final table has 5,147 product-week rows, 100 products and 52 full weeks (2010-12-06 to 2011-11-28); 4% of product-weeks are zero; the week of 2010-12-27 has zero sales for all products (shop closure).

### 4.3 Data processing requirements

| ID | Requirement | Priority |
| --- | --- | --- |
| FR-D-1 | The system MUST load the Excel file once and cache a parquet copy for later runs. | MUST |
| FR-D-2 | The system MUST remove cancelled invoices (`InvoiceNo` starting with `C`), rows with `Quantity <= 0`, rows with `UnitPrice <= 0`, and non-product `StockCode` values (fixed list, codes with no digits, and prefixes `GIFT` and `TEST`). | MUST |
| FR-D-3 | The system MUST keep rows with missing `CustomerID`. | MUST |
| FR-D-4 | The system MUST define weeks as Monday-start and retain only full weeks, computed from the data range. | MUST |
| FR-D-5 | The system MUST cap line-level `Quantity` at the 99.9th percentile, with the cap learned on training-period rows only. | MUST |
| FR-D-6 | The system MUST select the Top 100 products by total units in the training period, requiring sales in at least 30 training weeks. | MUST |
| FR-D-7 | The system MUST aggregate to product x week (`units`, `orders`, `customers`, `avg_price`), fill missing weeks between a product's first and last active week with zero units and forward-filled price. | MUST |
| FR-D-8 | The system MUST detect shop-wide closed weeks and export them to `closed_weeks.csv`. | MUST |
| FR-D-9 | The system MUST validate output against contract C1 before saving and MUST write a cleaning log with row counts per step. | MUST |
| FR-D-10 | The pipeline MUST be runnable with one command and be deterministic. | MUST |

### 4.4 Processed data dictionary (contract C1)

| Column | Type | Description |
| --- | --- | --- |
| `product_id` | str | Cleaned `StockCode` |
| `description` | str | Most frequent description |
| `week_start` | datetime | Monday of the week |
| `units` | int >= 0 | Target variable |
| `orders` | int | Distinct invoices that week |
| `customers` | int | Distinct non-null customers that week |
| `avg_price` | float > 0 | Mean unit price that week |

### 4.5 Train and test periods

| Set | Weeks | Period |
| --- | --- | --- |
| Training | 44 | 2010-12-06 to 2011-10-03 |
| Test | 8 | 2011-10-10 to 2011-11-28 |
| Forecast target | 1 | 2011-12-05 |

The test period MUST NOT be used for feature selection, tuning, imputation fitting, outlier capping, or product selection.

---

## 5. Functional Requirements: Features, Models, Evaluation

### 5.1 Feature engineering

| ID | Requirement | Priority |
| --- | --- | --- |
| FR-F-1 | For target week t the system MUST build lag features of units at t-1, t-2, t-3, t-4. | MUST |
| FR-F-2 | The system MUST build rolling mean and rolling standard deviation over 4 and 8 weeks, always applying `shift(1)` before `rolling()`. | MUST |
| FR-F-3 | The system MUST build calendar features (week of year, month) and SHOULD test weeks-to-Christmas by ablation, since one year of data makes it close to a time index. | MUST / SHOULD |
| FR-F-4 | The system MUST build previous-week commercial features: average price, orders, customers. | MUST |
| FR-F-5 | The system MUST build product-level history features (past mean and volatility) computed only from data before t. | MUST |
| FR-F-6 | Closed weeks MUST be treated as missing: never a training target, and lags and rolling statistics MUST ignore them (NaN). Missing values MUST be imputed inside the model pipeline using statistics fitted on training data only (LightGBM may use native NaN handling). | MUST |
| FR-F-7 | Rows without enough history for the longest window MUST be dropped from training, and the count MUST be reported. | MUST |
| FR-F-8 | An automated leakage test MUST prove that changing data at week t or later does not change any feature at week t. | MUST |
| FR-F-9 | The ordered feature list MUST be saved to `feature_list.json` and reused unchanged by the API. | MUST |

Expected size: about 100 products x (44 training weeks minus up to 8 history weeks and the closure) gives roughly 3,000 to 3,500 training rows and 800 test rows (100 products x 8 weeks).

### 5.2 Modelling

| ID | Requirement | Priority |
| --- | --- | --- |
| FR-M-1 | The system MUST implement two baselines: naive (last week) and 4-week moving average. | MUST |
| FR-M-2 | The system MUST train Ridge regression, Random Forest, and LightGBM (or XGBoost) on identical features and splits. | MUST |
| FR-M-3 | The system MAY add Poisson regression for count data. | MAY |
| FR-M-4 | Hyperparameters MUST be tuned using `TimeSeriesSplit` inside the training period only, with a documented search space and budget. | MUST |
| FR-M-5 | The system MUST report MAE, RMSE, WAPE and bias for every model on validation and test sets. MAPE MUST NOT be used. | MUST |
| FR-M-6 | Predictions MUST be clipped at zero. | MUST |
| FR-M-7 | The best model MUST be selected by validation WAPE, with ties broken by validation bias; the test set is used only for the final report. | MUST |
| FR-M-8 | A model MUST be reported honestly even if it does not beat the baselines; the report MUST say so. | MUST |
| FR-M-9 | The best model MUST be refit on the full training period, saved with `joblib`, and accompanied by `feature_list.json` and `config.json` (model name, horizon, split dates, seed, library versions). | MUST |
| FR-M-10 | The system MUST produce `next_week_forecast.csv` (contract C4) for week 2011-12-05. `sigma_error` MUST come from out-of-fold validation errors only; where a product has fewer than 6 validation observations, the pooled standard deviation of its volume segment MUST be used. | MUST |

**Candidate hyperparameter search (guideline):** Ridge `alpha` in a log range; Random Forest `n_estimators`, `max_depth`, `min_samples_leaf`; LightGBM `num_leaves`, `learning_rate`, `n_estimators`, `min_child_samples`, `subsample`. Use random search with a fixed budget (for example 30 trials) to stay within the timeline.

**Known modelling risk:** demand drifts upward in October and November, and tree models cannot predict above the range seen in training. The evaluation MUST check for systematic under-forecasting at the end of the test period, and Ridge or a hybrid (for example a tree model on a ratio target) SHOULD be considered if it appears.

### 5.3 Forecast-error analysis

| ID | Requirement | Priority |
| --- | --- | --- |
| FR-E-1 | Error MUST be reported by volume segment (low, medium, high sellers, tertiles of average weekly units). | MUST |
| FR-E-2 | Error MUST be reported by test week, with comments on weeks of spikes. | MUST |
| FR-E-3 | The analysis MUST include predicted-versus-actual plots and the residual distribution. | MUST |
| FR-E-4 | The analysis MUST list best and worst products with a reason (for example intermittent demand or bulk orders). | MUST |
| FR-E-5 | Over-forecast versus under-forecast MUST be quantified separately, because under-forecasting leads to stockouts. | MUST |
| FR-E-6 | Results MUST be exported to `error_analysis.csv` (segment, week and product aggregates) for the dashboard. | SHOULD |

---

## 6. Functional Requirements: Simulated Inventory and Stockout Risk

### 6.1 Simulation parameters

The dataset has no inventory-on-hand data, so stock is simulated. Every parameter below is configurable through the API and the dashboard.

| Parameter | Default | Allowed range | Meaning |
| --- | --- | --- | --- |
| `initial_stock_weeks` | 3.0 | 0.5 to 8 | Starting stock expressed in weeks of average training-period demand |
| `lead_time_weeks` (L) | 1 | integer 1 to 4 | Replenishment lead time |
| `service_level` | 0.95 | 0.80 to 0.99 | Target probability of no stockout during lead time |
| `seed` | 42 | integer | Seed for the per-product stock noise |
| stock noise | Uniform(0.5, 1.5) | fixed in v1 | Per-product multiplier so products do not all start with identical cover |

### 6.2 Formulas (authoritative)

For each product i, using the forecast file (contract C4):

1. `on_hand_i = initial_stock_weeks x avg_weekly_units_train_i x noise_i`. The noise is drawn once per product from a generator seeded with `seed`, over products sorted by `product_id`, so it does not depend on any other parameter. Moving a slider therefore changes thresholds only, never the stock itself.
2. `forecast_lead_time_i = forecast_weekly_i x L`
3. `z = norm.ppf(service_level)`
4. `safety_stock_i = z x sigma_error_i x sqrt(L)`
5. `reorder_point_i = forecast_lead_time_i + safety_stock_i`
6. `days_of_cover_i = on_hand_i / (forecast_weekly_i / 7)`; if the forecast is 0, days of cover is set to 999.
7. Risk rule:
   - **HIGH:** `on_hand < forecast_lead_time`
   - **MEDIUM:** `on_hand < reorder_point` and not HIGH
   - **LOW:** otherwise

Comparisons use unrounded values; displayed values are rounded to one decimal place. Equality (on-hand exactly equal to a threshold) is NOT below the threshold, so it falls in the safer class.

### 6.3 Requirements

| ID | Requirement | Priority |
| --- | --- | --- |
| FR-R-1 | `compute_risk(forecast_df, params)` MUST return `product_id, description, forecast_weekly, forecast_lead_time, on_hand, safety_stock, reorder_point, days_of_cover, risk_level` for every product in the input. | MUST |
| FR-R-2 | Results MUST be deterministic for identical inputs and seed. | MUST |
| FR-R-3 | Missing parameters MUST take defaults; out-of-range or wrongly typed parameters MUST raise `ValueError` (the API maps this to HTTP 422). | MUST |
| FR-R-4 | The engine MUST handle edge cases without errors: forecast 0, sigma 0, L = 4, service level 0.99, empty input. | MUST |
| FR-R-5 | `sensitivity_table` MUST return the number of HIGH, MEDIUM and LOW products for every combination of the supplied lead times and service levels (default L in 1..4, service level in 0.90, 0.95, 0.99). | MUST |
| FR-R-6 | Risk results MUST include a summary: counts per class, share at risk, average days of cover. | MUST |
| FR-R-7 | The simulated-inventory disclaimer MUST accompany every risk output (API and UI). | MUST |
| FR-R-8 | The report MUST state how the defaults were chosen. They MUST NOT be tuned to produce a desired number of flagged products; the sensitivity analysis shows how results change. | MUST |
| FR-R-9 | A rule backtest MAY be provided: simulate stock depletion across the 8 test weeks using actual demand and compare flags against simulated stockouts (precision and recall of the HIGH and MEDIUM flags). | MAY |

### 6.4 Modelling limitations to document

- The one-step forecast is reused for every week of the lead time (`forecast_weekly x L`), a simplification.
- Stock is a snapshot at the start of the forecast week; no ordering policy or in-transit stock is simulated.
- With the default 3 weeks of starting stock and L = 1, few products may be HIGH. This is a property of the assumptions, not a model failure, and must be reported as such together with the sensitivity table.
- Because stock is synthetic, risk counts are illustrative and demonstrate rule behaviour, not real stock positions.

---

## 7. API Specification

Base URL locally: `http://localhost:8000`. Interactive documentation is generated at `/docs`.

### 7.1 Endpoints

| Method and path | Purpose | Success | Errors |
| --- | --- | --- | --- |
| GET `/health` | Liveness check | 200 `{"status":"ok"}` | n/a |
| GET `/products` | List the 100 products | 200 list of `{product_id, description, total_units}` | 500 |
| GET `/forecast/{product_id}` | History, test predictions, next-week forecast | 200 object | 404 unknown product |
| POST `/risk` | Risk table for given parameters | 200 `{params, summary, items}` | 422 invalid parameters |
| POST `/sensitivity` | Counts per lead time and service level | 200 list of rows | 422 |
| GET `/metrics` | Model comparison (validation and test) | 200 list | 500 |
| GET `/disclaimer` | Simulated-inventory disclaimer text | 200 `{"text":"..."}` | n/a |
| GET `/error-analysis` | Error aggregates by segment, week, product (**proposed addition to C7, needs team approval**) | 200 object | 500 |

Field-level schemas are those in `CONTRACTS.md` section C7.

### 7.2 Requirements

| ID | Requirement | Priority |
| --- | --- | --- |
| FR-A-1 | The API MUST load the model, feature list, config and result files once at start-up and fail with a clear message if any is missing. | MUST |
| FR-A-2 | All request bodies MUST be validated with Pydantic models and the ranges in section 6.1. | MUST |
| FR-A-3 | Errors MUST use a JSON body `{"detail": "..."}` with correct status codes (404, 422, 500); stack traces MUST NOT be exposed. | MUST |
| FR-A-4 | The API MUST enable CORS for the dashboard origin(s), configurable by an environment variable. | MUST |
| FR-A-5 | Dates MUST be ISO 8601 strings (`YYYY-MM-DD`). Numbers MUST be finite (no NaN in JSON). | MUST |
| FR-A-6 | `/risk` and `/sensitivity` SHOULD cache results by parameter set. | SHOULD |
| FR-A-7 | The API MUST NOT expose `CustomerID` or any other personal data. | MUST |
| FR-A-8 | Request handling MUST be stateless. | MUST |
| FR-A-9 | The API MUST serve the files in dashboard/ at the root path, registered after all API routes, so one deployment gives one URL. API routes keep priority over static files. | MUST |

---

## 8. User Interface Specification

### 8.1 Global layout

- **Control rail (all pages, above the content):** sliders for lead time (1 to 4), service level (0.80 to 0.99) and initial stock weeks (0.5 to 8), a reset-to-defaults button, and the API status.
- **Banner (all pages):** the simulated-inventory disclaimer text from `/disclaimer`.
- **Navigation:** Shelf, Product forecast, Risk table, Model performance, Sensitivity (the assumptions are listed on the Sensitivity page).

### 8.2 Pages

| Page (route) | Content |
| --- | --- |
| Shelf (`#overview`) | Headline sentence stating how many products run out in the simulation; the shelf wall with one bin per product (details in 8.4); colour and reorder key |
| Product forecast (`#explorer`) | Product selector; line chart of weekly history, test-week actuals versus predictions, shop-closure marker and the next-week forecast; hover crosshair; tiles for average weekly units, forecast, typical forecast miss, simulated stock, reorder point and days of cover |
| Risk table (`#risk`) | Sortable table with filter buttons per risk level (each showing its count), text plus icon plus colour for every class, row click opens the product, CSV download |
| Model performance (`#models`) | Metrics table for all models on validation and test; per-week error bars; error by volume segment; over versus under-forecast summary |
| Sensitivity (`#sensitivity`) | Heatmap of HIGH and MEDIUM counts across lead time and service level with the current setting outlined; written list of every assumption and formula |

### 8.3 Requirements

| ID | Requirement | Priority |
| --- | --- | --- |
| FR-U-1 | The UI MUST obtain all data through one module, `dashboard/js/api.js`. No model or risk logic may remain in the UI in the submitted version (the prototype's in-browser risk calculation is replaced by calls to the API). | MUST |
| FR-U-2 | Changing a slider MUST refresh the risk outputs within the performance target in NFR-1. | MUST |
| FR-U-3 | The disclaimer MUST be visible on every page without scrolling. | MUST |
| FR-U-4 | Risk class MUST be conveyed by text and icon as well as colour. | MUST |
| FR-U-5 | Loading, empty and error states MUST be handled, for example an "API unreachable" message with a retry button instead of a blank page. | MUST |
| FR-U-6 | The Risk table MUST support sorting by any column and filtering by risk level. | MUST |
| FR-U-7 | The Risk table SHOULD export to CSV. | SHOULD |
| FR-U-8 | Charts MUST have a title or caption, axis labels with units, and the forecast MUST be visually distinct from actuals. | MUST |
| FR-U-9 | The UI SHOULD be readable at tablet width and usable at phone width. | SHOULD |
| FR-U-10 | The API address MUST come from `dashboard/config.js` (`apiBase`; an empty string means same origin). | MUST |
| FR-U-11 | The Shelf page MUST draw one bin per product. Fill height = simulated stock in weeks of forecast demand (0 to 6 weeks, clamped); red dashed line = demand during the delivery (`forecast_lead_time`); yellow tick = `reorder_point`; colour and icon = `risk_level`. Bins MUST refill with an animated transition when a slider changes. | MUST |
| FR-U-12 | The headline MUST state the live number of HIGH-risk products, and selecting a bin MUST open that product's forecast page. | MUST |
| FR-U-13 | The UI MUST provide light and dark themes that follow the system setting, with a manual toggle; both MUST keep text readable. | MUST |
| FR-U-14 | The UI MUST be keyboard operable with visible focus, and MUST respect the reduced-motion setting. | MUST |

### 8.4 Front-end design and implementation

The dashboard is a static web app (HTML, CSS, JavaScript) whose look is fixed by a working prototype, [Shelfwatch](https://claude.ai/artifact/WEBAobVRSHian8Hdj5y3tf), built on the real UCI sales of 30 products with a moving-average baseline. The submitted version keeps its look and replaces its in-browser risk calculation with calls to the API.

**Design concept.** The shelf wall is the signature element: each product is a bin filled to its simulated stock, so risk reads as level and space instead of a number. The rest of the interface stays flat and quiet, with one accent, so the wall carries the page.

| Token | Light | Dark |
| --- | --- | --- |
| Background | #E4E9ED | #0F1A24 |
| Panel | #F5F7F8 | #16232F |
| Text | #16212B | #E8EEF3 |
| Accent (hazard yellow) | #FFC72C | #FFCB3D |
| High, Medium, Low risk | #D93A2F, #E8791A, #1F8F5F | #FF6154, #FF9A3C, #3DD598 |

Typography: Big Shoulders Display for the headline and numbers, Public Sans for text, with Arial Narrow, Impact and system-ui as fallbacks so the page works if fonts are blocked.

**Pages and their data**

| Page | API calls | Rendering rules |
| --- | --- | --- |
| Shelf | `POST /risk`, `GET /products` | Weeks of stock = `on_hand / forecast_weekly`, fill clamped to 0 to 6 weeks; red line at `forecast_lead_time / forecast_weekly`; yellow tick at `reorder_point / forecast_weekly`; bins ordered by `total_units`; headline from `summary.n_high` |
| Product forecast | `GET /forecast/{id}`, `POST /risk` | Hand-built SVG line chart; shaded test weeks; marker for the closed week; hover shows date, actual and prediction |
| Risk table | `POST /risk` | Sorting and filtering in the browser; CSV built from the loaded rows |
| Model performance | `GET /metrics`, `GET /error-analysis` | Metrics table and per-week error bars |
| Sensitivity | `POST /sensitivity`, `GET /disclaimer` | Heatmap with the current setting outlined; assumptions list |

**Files**

- `index.html`: page shell, control rail, disclaimer strip, navigation
- `config.js`: `window.SHELFWATCH = { apiBase: "" }` (use `"mock"` for offline work against `data/mock/*.json`)
- `css/style.css`: tokens as CSS variables, light and dark themes
- `js/api.js`: the only module that talks to the API (`getProducts`, `getForecast`, `getRisk`, `getSensitivity`, `getMetrics`, `getErrorAnalysis`, `getDisclaimer`)
- `js/charts.js`: SVG line chart, bars and heatmap helpers
- `js/app.js`: state, hash routing, page rendering

**Behaviour rules**

- Routes are the hashes `#overview`, `#explorer`, `#risk`, `#models` and `#sensitivity`.
- Slider changes are debounced by 150 ms; a newer request aborts the older one (`AbortController`); results are cached per parameter set.
- Shelf bins are updated in place by changing CSS variables, so the refill transition runs; other pages re-render.
- Text from the API is inserted with `textContent` or an escape function, never raw into `innerHTML`; no `eval`.
- Supported browsers: the latest two versions of Chrome, Edge, Firefox and Safari.

---

## 9. Non-Functional Requirements

| ID | Category | Requirement |
| --- | --- | --- |
| NFR-1 | Performance | `/risk` for 100 products responds in under 500 ms (warm); `/forecast/{id}` under 300 ms; dashboard page loads in under 3 s on a warm server. |
| NFR-2 | Reproducibility | Running the data and training scripts on the same input with the same library versions yields identical outputs (seed 42). `config.json` records split dates and versions. |
| NFR-3 | Usability | A first-time user can find the at-risk products and change an assumption within 2 minutes. Terminology is explained in tooltips or a glossary. |
| NFR-4 | Accessibility | Color is never the only carrier of meaning; sufficient contrast. |
| NFR-5 | Reliability | The API starts cleanly, exposes `/health`, and returns structured errors rather than crashing on bad input. |
| NFR-6 | Maintainability | PEP 8, type hints and docstrings on public functions; module boundaries as in section 3.2; at least 80% line coverage of `src/`. |
| NFR-7 | Portability | Runs on Windows, macOS and Linux with Python 3.10 or 3.11 using `requirements.txt` only. |
| NFR-8 | Security and privacy | No secrets in the repository; input validation on every endpoint; CORS allow-list; no personal data exposed; no use of `eval` or unsafe deserialization of user input. Only trusted, team-created `joblib` files are loaded. |
| NFR-9 | Observability | The API logs start-up, request errors and artifact versions. |
| NFR-10 | Size | Artifacts total under 50 MB so free-tier hosting works. |
| NFR-11 | Documentation | README with setup, run, test and deploy steps; every module documented. |
| NFR-12 | Front-end payload | Dashboard HTML, CSS and JavaScript total under 150 KB uncompressed (web fonts excluded), with no build step and a mock-data mode for offline development. |

---

## 10. Testing and Quality Assurance

### 10.1 Test levels

| Level | Scope | Tools |
| --- | --- | --- |
| Unit | Cleaning functions, week logic, features, metrics, risk formulas | pytest |
| Contract | Every hand-off file and endpoint matches `CONTRACTS.md` | pytest schema checks, `validate_c1` |
| Leakage | Features at week t unchanged by data at t or later | pytest |
| Model sanity | Predictions are finite and non-negative; models beat or are compared against baselines | pytest and notebook checks |
| API | Status codes, schemas, 404 and 422 cases, determinism | FastAPI TestClient |
| UI smoke | Dashboard launches, pages render with API reachable and with API down | manual checklist; optional jsdom or Playwright script |
| System / acceptance | End-to-end run on the deployed URL | checklist in Appendix C |

### 10.2 Key test cases

| ID | Test | Requirement |
| --- | --- | --- |
| T-D-1 | Output has exactly 100 products, only Mondays, no duplicate product-week rows, no gaps | FR-D-4, 6, 7, 9 |
| T-D-2 | Planted junk (cancellation, `POST`, zero price, 80,995-unit order) does not survive | FR-D-2, 5 |
| T-D-3 | Top-100 selection is unchanged when test-period sales are inflated | FR-D-6 |
| T-D-4 | Closed week is detected and exported | FR-D-8 |
| T-F-1 | Altering any value at week t or later leaves features at week t unchanged | FR-F-2, 8 |
| T-F-2 | Closed week is never a target and lags skip it | FR-F-6 |
| T-M-1 | Train weeks strictly precede test weeks; no tuning call touches test data | C-1, FR-M-4 |
| T-M-2 | Metric functions match hand-computed values on a small example; MAPE absent | FR-M-5 |
| T-M-3 | Saved model reloads and reproduces stored predictions | FR-M-9 |
| T-R-1 | Worked example in Appendix A gives HIGH, MEDIUM and LOW as specified | FR-R-1, 6.2 |
| T-R-2 | Edge cases: forecast 0, sigma 0, L = 4, service 0.99, on-hand equals threshold | FR-R-4 |
| T-R-3 | Same seed gives same output; changing other parameters does not change `on_hand` | FR-R-2, 6.2 |
| T-R-4 | Invalid parameters raise `ValueError` | FR-R-3 |
| T-A-1 | Every endpoint returns the documented schema and status code | FR-A-2, 3, 5 |
| T-A-2 | Unknown product gives 404; out-of-range risk parameter gives 422 | FR-A-3 |
| T-A-3 | Response contains no `CustomerID` | FR-A-7 |
| T-U-1 | Disclaimer visible on every page; slider change updates KPIs | FR-U-2, 3 |
| T-U-2 | With the API stopped, the UI shows a friendly error | FR-U-5 |
| T-U-3 | Number of bins equals the number of products from the API; each bin's risk class matches risk\_level in the API response; bin order follows total\_units | FR-U-11, FR-U-12 |
| T-U-4 | Keyboard-only navigation reaches every control; both themes keep text readable; no animation with reduced motion enabled | FR-U-13, FR-U-14 |

### 10.3 Quality gates

A pull request merges only when: all tests pass, a teammate has reviewed it, the contract is respected, and (for modelling code) the leakage test passes. A fresh-chat review with Claude (leakage, contracts, formulas, edge cases) is recommended before merging.

---

## 11. Deployment and URL Generation

### 11.1 Environments

| Environment | Use | How |
| --- | --- | --- |
| Local | Development and the fallback demo | Run `uvicorn api.main:app --port 8000`, then open `http://localhost:8000/`. The API serves the dashboard files from the same address; API documentation stays at `/docs`. |
| Cloud | Public URL for submission | See options below |

### 11.2 Configuration

API environment variables: `ALLOWED_ORIGINS` (CORS allow-list, needed only when the dashboard is hosted on a different origin) and `PORT` (set by the host). The dashboard reads the API address from `dashboard/config.js` (`apiBase`; empty means same origin). No secrets are required.

### 11.3 Options

| Option | Setup | Pros | Cons |
| --- | --- | --- | --- |
| A. One service (recommended) | FastAPI serves the API and the `dashboard/` files from one app on Render or a Hugging Face Docker Space (`uvicorn api.main:app --host 0.0.0.0 --port $PORT`) | One public URL, same origin so no CORS, simplest to test | Free tier sleeps when idle (first request slow) |
| B. Two hosts | API on Render; dashboard on Netlify or GitHub Pages with `apiBase` set to the API URL and the dashboard origin in `ALLOWED_ORIGINS` | Dashboard loads instantly | Two deployments, CORS must be configured |
| C. Temporary public tunnel | Run locally and expose port 8000 with ngrok | 5-minute setup | URL changes; laptop must stay on |

**Recommendation:** deploy a "hello world" `/health` API by Day 8 (Option A), keep Option C as a backup, and record a 3-minute screen capture of the local system as a final fallback for the demonstration.

### 11.4 Release checklist

1. `requirements.txt` pinned; clean virtual environment installs and runs.
2. All artifacts present in `models/` and `data/processed/`.
3. `/health`, `/docs` and the dashboard open from the public URL.
4. Disclaimer visible; defaults match the report.
5. Smoke test of every page and every endpoint on the deployed URL.
6. The URL, the repository link and the README are in the submission.

---

## 12. Project Organisation and Plan

### 12.1 Roles

| Member | Role | Primary deliverables |
| --- | --- | --- |
| M1 | Data and Analysis | C1, C1b, EDA, error-analysis notebook |
| M2 | ML Modelling | C2, C3, C4, C5, model comparison |
| M3 | Risk Engine and Backend | C6, C7, sensitivity analysis |
| M4 | Dashboard, QA and Deployment | UI, tests, deployed URL, report and slides assembly |

Every member must be able to explain the whole pipeline, because evaluators may question any member about any stage.

M4 owns the HTML, CSS and JavaScript dashboard and adapts the Shelfwatch prototype; M3 reviews `js/api.js` against the API schemas. If nobody on the team is comfortable with JavaScript, M3 and M4 pair on `js/api.js` and reuse the prototype's rendering code unchanged.

### 12.2 Timeline and checkpoints (14 days)

| Days | Milestone |
| --- | --- |
| 1 | Contracts, mock files, repository, environment (all members) |
| 2 to 3 | Data pipeline (done in v1), features on real data. **Checkpoint 1:** real `weekly_demand.parquet` |
| 4 to 6 | Baselines, models, tuning, risk engine on mocks, dashboard skeleton. **Checkpoint 2:** real predictions and model |
| 7 to 9 | Final model, API on real artifacts, hello-world deploy, error analysis |
| 10 to 11 | Integration, tests, report sections. **Checkpoint 3:** full local integration |
| 12 to 13 | Deployment, final testing, slides |
| 14 | Buffer, rehearsal, submission |

### 12.3 Workflow rules

One Git branch per member; pull requests reviewed by a teammate; small frequent commits; contract changes only by team agreement and recorded in the change log of `CONTRACTS.md`.

---

## 13. Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
| --- | --- | --- | --- |
| Models do not beat the baselines | Medium | Medium | Report honestly; analyse why (intermittent demand, level shift); baselines are a valid result |
| Tree models under-forecast the high October and November level | High | Medium | Check bias per week; consider Ridge or ratio-target hybrid |
| Schema mismatch at integration | Medium | High | Mock files on Day 1, contract tests |
| Data leakage inflates accuracy | Medium | High | Mandatory leakage test, chronological split only |
| Few HIGH-risk products at default assumptions | Medium | Low | Document honestly; sensitivity analysis; do not tune to a target |
| Free-tier cold start during demo | Medium | Medium | Warm the service before the demo; local fallback; recording |
| Library version mismatch breaks model loading | Medium | High | Pinned requirements, versions saved in `config.json` |
| Time overrun | Medium | High | Fixed scope; early deployment; daily 15-minute sync |
| Custom HTML, CSS and JavaScript dashboard takes longer than a ready-made framework | Medium | Medium | A working prototype already exists (section 8.4); build against mock data first; keep to five pages; no framework |

---

## 14. Implementation Guide

### 14.1 Build order

| Step | Owner | Task | Done when |
| --- | --- | --- | --- |
| 1 | All | Repository, branches, pinned `requirements.txt`, mock files for C1 to C4 in `data/mock/` | Everyone can run `pytest` against the mocks |
| 2 | M1 | Data pipeline (built: 5,147 rows, 100 products, `closed_weeks.csv`) | `validate_c1` passes on the real file; cleaning log saved |
| 3 | M2 | `src/features.py` with the leakage test | T-F-1 and T-F-2 pass |
| 4 | M2 | Baselines, Ridge, Random Forest, LightGBM, tuning, metrics | `metrics.csv` and `predictions.csv` written |
| 5 | M2 | Best model, `next_week_forecast.csv`, model files | Saved model reloads and reproduces predictions (T-M-3) |
| 6 | M3 | `src/risk_engine.py` tested against Appendix A | T-R-1 to T-R-4 pass |
| 7 | M3 | `api/main.py` with all endpoints and the static mount | T-A-1 to T-A-3 pass; `/docs` lists every endpoint |
| 8 | M4 | Adapt the prototype to `js/api.js`, first in mock mode, then live | All five pages work against the real API |
| 9 | M1 | Error-analysis notebook and report figures | FR-E-1 to FR-E-6 satisfied |
| 10 | M4 | Hello-world deploy by Day 8, then full deploy and integration tests | Appendix C fully ticked |
| 11 | All | Report, slides, viva rehearsal | Submission made |

### 14.2 Module interfaces

```python
# src/data_prep.py (built)
build_weekly_demand(raw_path, out_path, ...) -> pd.DataFrame
validate_c1(df) -> None

# src/features.py
build_features(weekly: pd.DataFrame, closed_weeks: list) -> pd.DataFrame
FEATURE_COLUMNS: list[str]          # saved to models/feature_list.json

# src/train.py
split_chronological(df, test_weeks=8) -> tuple[pd.DataFrame, pd.DataFrame]
evaluate(y_true, y_pred) -> dict    # MAE, RMSE, WAPE, bias
run_all() -> None                   # writes C2, C3, C4, C5

# src/risk_engine.py
compute_risk(forecast_df, params=None) -> pd.DataFrame
sensitivity_table(forecast_df, lead_times=(1, 2, 3, 4),
                  service_levels=(0.90, 0.95, 0.99)) -> pd.DataFrame
```

### 14.3 API skeleton

```python
import os
from contextlib import asynccontextmanager
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from src.risk_engine import compute_risk

STATE: dict = {}
DISCLAIMER = ("Inventory levels are simulated. The dataset contains sales only, "
              "not stock-on-hand. Risk flags show how the rule behaves under the "
              "stated assumptions, not real stock positions.")

@asynccontextmanager
async def lifespan(app: FastAPI):
    STATE["weekly"] = pd.read_parquet("data/processed/weekly_demand.parquet")
    fc = pd.read_csv("data/processed/next_week_forecast.csv")
    desc = STATE["weekly"].drop_duplicates("product_id")[["product_id", "description"]]
    STATE["forecast"] = fc.merge(desc, on="product_id")   # compute_risk needs description
    STATE["metrics"] = pd.read_csv("data/processed/metrics.csv")
    STATE["preds"] = pd.read_csv("data/processed/predictions.csv")
    yield

app = FastAPI(title="Shelfwatch API", lifespan=lifespan)
origins = [o for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o]
app.add_middleware(CORSMiddleware, allow_origins=origins,
                   allow_methods=["GET", "POST"], allow_headers=["*"])

class RiskRequest(BaseModel):
    lead_time_weeks: int = Field(1, ge=1, le=4)
    service_level: float = Field(0.95, ge=0.80, le=0.99)
    initial_stock_weeks: float = Field(3.0, ge=0.5, le=8)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/disclaimer")
def disclaimer():
    return {"text": DISCLAIMER}

@app.post("/risk")
def risk(req: RiskRequest):
    params = req.model_dump()
    df = compute_risk(STATE["forecast"], params)
    counts = df["risk_level"].value_counts()
    summary = {"n_high": int(counts.get("HIGH", 0)),
               "n_medium": int(counts.get("MEDIUM", 0)),
               "n_low": int(counts.get("LOW", 0)),
               "avg_days_of_cover": float(df["days_of_cover"].mean())}
    return {"params": {**params, "seed": 42}, "summary": summary,
            "items": df.round(1).to_dict("records")}

# /products, /forecast/{product_id}, /sensitivity, /metrics and /error-analysis
# follow the same pattern (contract C7); convert NaN to None before returning.

# Must be the LAST registration: API routes keep priority over static files.
app.mount("/", StaticFiles(directory="dashboard", html=True), name="dashboard")
```

### 14.4 Run and test

```
pip install -r requirements.txt
python -m src.data_prep --raw "data/raw/Online Retail.xlsx"
python -m src.train
pytest tests/ -v
uvicorn api.main:app --reload --port 8000
# dashboard: http://localhost:8000/    API docs: http://localhost:8000/docs
```

---

## Appendix A. Worked Example of the Risk Rule

Given: `forecast_weekly = 100`, `sigma_error = 20`, `avg_weekly_units_train = 90`, `initial_stock_weeks = 3`, `L = 2`, `service_level = 0.95` (z = 1.645).

- `forecast_lead_time = 100 x 2 = 200`
- `safety_stock = 1.645 x 20 x sqrt(2) = 46.5`
- `reorder_point = 200 + 46.5 = 246.5`

| Stock noise | `on_hand` | Check | Class | Days of cover |
| --- | --- | --- | --- | --- |
| 1.1 | 297 | 297 >= 246.5 | LOW | 20.8 |
| 0.8 | 216 | 200 <= 216 < 246.5 | MEDIUM | 15.1 |
| 0.6 | 162 | 162 < 200 | HIGH | 11.3 |

## Appendix B. Requirement Traceability (summary)

| Problem-statement deliverable | Requirements | Verified by |
| --- | --- | --- |
| Demand model for a defined product and time horizon | FR-D-6, FR-F-1 to 9, FR-M-1 to 10 | T-F-1, T-M-1 to 3 |
| Product-level estimates | FR-M-10, FR-A, FR-U (Product forecast page) | T-A-1, T-U-1 |
| Forecast-error analysis | FR-E-1 to 6 | Notebook review, T-M-2 |
| Configurable stockout indicator | FR-R-1 to 9, section 6.2 | T-R-1 to 4 |
| Inventory dashboard | FR-U-1 to 14 | T-U-1 to T-U-4, Appendix C |
| Stated simulated-inventory assumptions | Section 6.1, FR-R-7, FR-R-8, FR-U-3 | T-U-1 |
| Compare regression approaches, evaluate on later periods | FR-M-1 to 7, C-1 | T-M-1 |

## Appendix C. Final Acceptance Checklist

- [ ] Pipeline reproduces `weekly_demand.parquet` from the raw file with one command
- [ ] Leakage test passes; split is chronological
- [ ] Metrics table shows baselines and all models on validation and test
- [ ] Best model, feature list and config saved and reloadable
- [ ] Error analysis covers segments, weeks, best and worst products, over versus under-forecast
- [ ] Risk engine matches Appendix A and all edge-case tests pass
- [ ] Sensitivity table and chart produced
- [ ] API implements every endpoint; tests pass
- [ ] Dashboard has the shelf wall, product forecast chart, risk table, model panel, sliders, visible disclaimer, light and dark themes
- [ ] Public or local URL works; README complete; report and slides ready

## Appendix D. Open Issues for Team Decision

| # | Issue | Proposal |
| --- | --- | --- |
| 1 | Contract C4 currently says `sigma_error` is computed from validation or test errors | Use validation (out-of-fold) errors only, so the simulation never uses test-period error. Update C4 |
| 2 | `/error-analysis` endpoint is not in C7 | Add it so the dashboard can show error analysis without reading files directly |
| 3 | Default `initial_stock_weeks = 3` may flag very few HIGH-risk products at L = 1 | Keep 3 as the stated default unless the team agrees on a documented alternative; show the sensitivity table. Prototype check on 30 products with the baseline forecast: 1 HIGH, 7 MEDIUM, 22 LOW at the defaults |
| 4 | Model family under the October and November level shift | Decide after Phase 2 results; keep Ridge and a ratio-target variant as candidates |
| 5 | Hosting choice (Option A versus B) | Confirm by Day 8 after the hello-world deploy |
| 6 | The shelf wall orders bins by total units, but /products does not return it | Add total\_units (training-period total) to the /products response and update C7 |
| 7 | Dashboard delivery: served by FastAPI at the root (one URL) or hosted separately | Serve it from FastAPI (Option A); use Option B only if a separate host is required |
| 8 | Showing all 100 bins on one shelf wall | Wrapping grid with a minimum bin width of 44 px; optionally group by volume segment (high, medium, low) |
| 9 | The prototype computes risk in the browser with a JavaScript random generator, so its stock noise differs from the Python engine | The submitted version uses values from POST /risk only (FR-U-1) |
