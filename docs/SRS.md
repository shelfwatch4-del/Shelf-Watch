# Software Requirements Specification

## Retail Product Demand Estimation and Stockout Risk Analysis

Version 3.2 | 17 November 2026 | Prepared by the project team (4 members) | Status: Final for implementation (revised after mentor review; single primary stakeholder)

---

## 1. Introduction

### 1.1 Purpose

This document specifies the requirements, architecture, data, algorithms, interfaces, quality attributes and test strategy for a web-based system that (a) forecasts weekly retail product demand from historical sales and (b) flags products that may need inventory attention using an explicit, configurable stockout-risk rule. It is the single reference for the four team members and for every Claude chat used while building the project.

### 1.2 Scope

**In scope**

- Cleaning the UCI Online Retail transactions (primary dataset) and aggregating them to product x week demand, and using UCI Online Retail II for prior-year history
- Time-based, derived behavioural, calendar and prior-year features, all with strict no-leakage rules
- Comparison of baselines and regression models with time-series tuning, evaluated on later (held-out) weeks
- Next-week demand estimates with prediction intervals, and interactive prediction for a chosen product or for user-entered recent sales
- Forecast-error analysis (by volume segment, by week, by product, over- versus under-forecast)
- Simulated and user-entered inventory, a configurable stockout-risk indicator, replenishment advice, an Action list and sensitivity analysis
- A working replenishment workflow: weekly plan, purchase orders, receiving stock
- Product intelligence: demand pattern, ABC and XYZ class, lifecycle stage, return rate, export share, customer concentration and bought-together links
- An analysis board showing the data used, how it was split, the features, tuning, evaluation, a business backtest and an explanation of each forecast
- A FastAPI backend with a MySQL database, CRUD endpoints, middleware and a Python API client
- A multi-page web application in HTML, CSS and JavaScript with no login page: it opens directly on the Home page
- Local deployment with a public ngrok URL, Postman testing evidence, and publishing of the dataset, notebook and model on Kaggle

**Out of scope**

- Real ERP or supplier integrations and automatic ordering from real suppliers
- Separate screens, roles or reports for other stakeholders (finance, merchandising, warehouse)
- User accounts and roles (a single API key protects the write endpoints)
- Real-time data ingestion
- Forecasting beyond one week (a 4-week horizon is an optional extension only)
- Horizontally scaled or high-availability deployment

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
- UCI Machine Learning Repository, Online Retail II (id 502, CC BY 4.0, 1,067,371 rows, 01/12/2009 to 09/12/2011): https://archive.ics.uci.edu/dataset/502/online+retail+ii
- Python holidays package, United Kingdom calendar (includes the 2011-04-29 royal wedding holiday): https://holidays.readthedocs.io/
- `PROJECT_CONTEXT.md`, `CONTRACTS.md` and `CONTRACTS_ADDENDUM.md` (frozen decisions and contracts C1 to C13) in the project repository
- Problem statement supplied by the course, and the mentor's review requirements summarised in Appendix E

### 1.5 Conventions

Requirement IDs follow `FR-<area>-<n>` (functional) and `NFR-<n>` (non-functional). Priority words: **MUST** (mandatory for acceptance), **SHOULD** (expected unless there is a documented reason), **MAY** (optional extension).

### 1.6 Primary stakeholder and real-life use

**Decision record.** The system is built around one primary stakeholder: the **Retail Inventory and Replenishment Manager**. Every page, endpoint and report section serves that person's work. Reasons: the problem statement asks which products may need inventory attention, which is exactly this person's weekly question; every model output (forecast, range, risk, order quantity, reorder date) maps to a decision they own; and a single user keeps the interface and the report coherent.

| Attribute | Description |
| --- | --- |
| Role | Retail Inventory and Replenishment Manager at a UK online giftware wholesaler and retailer (the business behind the dataset) |
| Owns | The weekly replenishment plan for about 100 fast-moving products, purchase-order approval and stock-cover targets |
| Decisions | What to order, how much and when; which products to stop reordering; how much safety stock to hold; how bad a supplier delay would be |
| Rhythm | A weekly review before supplier orders are placed, plus daily exception checks |
| Pain points | Stockouts of best sellers (lost sales); overstock of slow or discontinued items (tied-up cash); bulk wholesale orders that distort averages; supplier delays; no single view of forecast, stock and open orders |
| Success measures | Fewer stockout weeks, fill rate, days of cover within target, less excess stock, and a forecast error they can trust |
| Constraints | Supplier lead time of 1 to 4 weeks; no reliable on-hand feed in the dataset, so stock is entered or simulated |
| Status of this persona | Assumed from the problem statement and standard replenishment practice, not from interviews. It is reviewed with the mentor and, if possible, one retail practitioner |

**Real-life use.** Retailers, e-commerce sellers and wholesalers use demand forecasts with safety stock and reorder points to balance two costs: running out (lost sales) and holding too much (tied-up cash, waste). The same weekly routine applies to grocery, pharmacy, spare parts and warehouse operations. Here it is applied to a UK online giftware business, where bulk wholesale orders and the Christmas build-up are the main patterns to handle.

**Reporting rule.** The report must describe the manager's weekly routine, show how each page supports a step in it, and quantify the benefit with the project's own results (forecast accuracy, risk counts, revenue at risk and the policy backtest in FR-R-9). Named companies may be mentioned only as general examples of the business type, never as claims that they use this exact system, unless a source is cited.

**How the system serves the manager's weekly routine**

| Step | The manager asks | Where the system answers |
| --- | --- | --- |
| 1. See what needs attention | Which products will run short before the next delivery? | Home: headline, shelf wall and Action list |
| 2. Understand one product | How much will it sell, and how uncertain is that? | Predict: forecast with a low-to-high range |
| 3. Decide the order | How much, and by when? | Home and Predict: recommended quantity, days to reorder |
| 4. Act | Record stock and raise purchase orders | Inventory manager (create, read, update, delete) |
| 5. Test assumptions | What if the supplier is a week late? | Predict: what-if simulator |
| 6. Set policy | Which items need closer control, and which should stop being reordered? | Product intelligence: ABC and XYZ, lifecycle, demand pattern |
| 7. Trust the numbers | How accurate has the forecast been? | Analysis board: data used, split, features, tuning, accuracy, backtest and explanations |

**Other stakeholders.** Finance, merchandising and warehouse staff benefit indirectly. They are not designed for: there are no separate screens, roles or reports for them.

**Features reframed for the manager.** Return rate affects net demand and order size; bought-together links name companion products to check when one is at risk; revenue at risk ranks which shortfall to fix first; ABC class sets how closely an item is reviewed.

---

## 2. Overall Description

### 2.1 Product perspective

The system is a self-contained, offline-trained analytics application. Training happens once in Python notebooks and scripts and produces versioned artifacts (a parquet table, CSV result files, a serialized model). A FastAPI service loads those artifacts at start-up and serves forecasts and risk results. A web dashboard built with plain HTML, CSS and JavaScript (no framework, no build step) consumes only the API. There are no external runtime dependencies other than the Python libraries and web fonts loaded from a CDN.

### 2.2 Product functions (summary)

1. **Prepare data:** clean raw transactions and build the weekly demand table.
2. **Enrich:** add UK calendar features, prior-year demand and transaction-derived signals such as returns and exports.
3. **Engineer features:** lags, rolling statistics, derived behaviour and calendar features, without leakage.
4. **Train, tune and compare models:** baselines, Ridge, Random Forest and gradient boosting, plus a scale-aware variant, an ensemble and quantile models for prediction intervals.
5. **Estimate and predict:** next-week forecast with a range for every product, and an interactive predict feature for a chosen product or user-entered sales.
6. **Analyse errors:** quantify where and why forecasts fail.
7. **Assess risk and advise:** apply an explicit risk rule to user-entered or simulated stock, with order quantity, reorder date and revenue at risk.
8. **Profile products:** demand pattern, ABC and XYZ class, lifecycle, return rate, export share, customer concentration and bought-together links.
9. **Manage inventory and orders:** CRUD on stock levels and purchase orders in a database, with every request logged by middleware.
10. **Present:** a multi-page web application with a visible simulated-inventory disclaimer.

### 2.3 User classes

| User | Goal | Technical level |
| --- | --- | --- |
| Retail Inventory and Replenishment Manager (primary and only designed-for user) | Decide what to order, how much and when, and keep stock cover on target | Non-technical |
| Examiner or evaluator | Verify methodology and the honesty of assumptions | Technical |
| Development team | Run, test, extend and deploy the system | Technical |

### 2.4 Operating environment

- Python 3.10 or 3.11 on Windows, macOS or Linux
- Modern desktop browser (Chrome, Edge, Firefox, Safari, latest two versions); the dashboard SHOULD remain usable at tablet width
- Local execution is mandatory; cloud hosting (Render or a Hugging Face Docker Space, with Netlify or GitHub Pages as an option for a separate dashboard host) is the target for the public URL

### 2.5 Design and implementation constraints

- **C-1:** Chronological evaluation only. Random train/test splits are prohibited.
- **C-2:** All features at week t use information from weeks before t only. Calendar facts known in advance are allowed.
- **C-3:** Seed 42 for every random process.
- **C-4:** The model, data and database dump MUST stay small enough to copy to a second laptop (target under 500 MB in total) so a backup machine can be prepared.
- **C-5:** Delivery within the course deadline by four people, so scope is deliberately focused (100 products, weekly granularity, one-week horizon, one stakeholder).
- **C-6:** Library versions are pinned so the saved model loads identically on every machine.
- **C-7:** All data is stored in MySQL on the demo machine. The public URL is a tunnel to that machine, so the laptop, MySQL, FastAPI and the tunnel must all be running during evaluation.
- **C-8:** External data must come from a documented, citable source or a library. No external value may be typed from memory.

### 2.6 Assumptions and dependencies

- **A-1:** The UCI files `Online Retail.xlsx` and `online_retail_II.xlsx` are available to every member and are not committed to Git.
- **A-2:** Inventory-on-hand is simulated unless a user enters real stock for a product. Results for simulated stock are illustrative, not real stock positions.
- **A-3:** The main table covers one year, so yearly seasonality is learned only through the prior-year features built from Online Retail II. Calendar features such as week-of-year behave like a time index and are kept only if validation supports them.
- **A-4:** The forecast target week is `2011-12-05`, the first week after the last full week of data.
- **A-5:** The shop-wide closure week `2010-12-27` is a closure, not zero demand, and has no row in the weekly table.
- **A-6:** Online Retail II covers the same retailer from 2009-12-01 and overlaps the main file from 2010-12-01, so it is used for prior-year history only and is not an independent test set. Its sheet structure and product overlap are checked before use.
- **A-7:** The public URL exists only while the demo laptop, MySQL, FastAPI and the ngrok tunnel are running. A backup (a second laptop or local screenshots, plus a database dump) is prepared.
- Dependencies: pandas, numpy, scikit-learn, LightGBM (or XGBoost), scipy, FastAPI, uvicorn, pydantic, SQLAlchemy, PyMySQL, python-dotenv, holidays, httpx, pytest, joblib, pyarrow, openpyxl. External tools: MySQL 8, ngrok, Postman.

---

## 3. System Architecture

### 3.1 Logical architecture

```
  OFFLINE (run once, reproducible)                     ONLINE (on the demo laptop)

  Online Retail.xlsx (primary) -> src/data_prep.py            Browser / Postman / faculty
  Online Retail II (2009-10) --> src/enrichment.py                      |
  UK holidays (holidays pkg) ---/                              ngrok public HTTPS URL
        |                                                               |
        v                                                     +---------v----------+
  processed tables, models, results                           | FastAPI  /api/*    |
        |                                                     | + five web pages   |
        v  scripts/load_mysql.py                              | middleware: CORS,  |
  +-------------+                                             | request id, audit, |
  |  MySQL 8    | <-------------------------------------------| API key on writes  |
  | all project |                                             +--------------------+
  | + user data |        models/*.joblib stay on disk,
  +-------------+        registered in MySQL (model_registry)

  Kaggle: dataset + notebook + model (publishing only)    scripts/api_client.py, Postman collection
```

### 3.2 Components and responsibilities

| Component | Owner | Responsibility |
| --- | --- | --- |
| Data pipeline (`src/data_prep.py`) | M1 | Clean, aggregate, validate against contract C1 (built) |
| Enrichment (`src/enrichment.py`) | M1 | Calendar, prior-year and transaction-derived weekly tables (C8 to C10) |
| Intelligence (`src/intelligence.py`) | M1 | Demand pattern, ABC and XYZ, lifecycle, returns, exports, basket pairs (C11, C12) |
| Features (`src/features.py`) | M2 | Leakage-safe, switchable feature groups; custom-series features |
| Training (`src/train.py`) | M2 | Baselines, models, tuning, ablation, quantile models, artifacts, result files |
| Risk engine (`src/risk_engine.py`) | M3 | Pure, deterministic risk, advice and sensitivity functions |
| API, database, middleware (`api/`) | M3 | Endpoints, CRUD, validation, audit log, API key |
| Python client (`scripts/api_client.py`) | M3 | Example client for every endpoint family |
| Web app (`dashboard/`) | M4 | Five pages; presentation and interaction only |
| Tests, deployment, Kaggle publishing, documents | M4 (all review) | Quality gate and delivery |

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
| Hosting | Local FastAPI server with an ngrok public URL (the flow required by the mentor); Render only as an optional backup | Free tiers with public URLs |
| Database | SQLAlchemy 2 with MySQL 8 (PyMySQL driver) | Stores all project data and user data (required by the mentor); unit tests may use in-memory SQLite |
| Calendar data | holidays package (UK, England) | Reproducible bank holidays, including the 2011-04-29 royal wedding holiday |
| Python client | httpx | Shows how Python calls the REST API with timeouts and error handling |
| Model publishing | Kaggle Datasets, Notebooks, Models | Public, reproducible record; not a live host (see 11.5) |
| Public URL | ngrok | Tunnel from a public HTTPS address to the local server on port 8000, as in the mentor's guide |
| API testing | Postman collection | Required by the mentor; also runs the same checks against the local and the public address |

### 3.5 Repository layout

```
project/
  PROJECT_CONTEXT.md  CONTRACTS.md  CONTRACTS_ADDENDUM.md  CHANGE_REQUESTS.md  README.md  .env.example
  data/raw/ (not committed)  data/processed/  data/mock/
  sql/        schema.sql  seed_demo.sql
  notebooks/  01_eda  02_modeling  03_error_analysis  04_sensitivity  05_insights
  src/        data_prep.py  enrichment.py  intelligence.py  features.py  train.py
              explain.py  backtest.py  risk_engine.py  error_analysis.py
  models/     best_model.joblib  q10.joblib  q90.joblib  feature_list.json  config.json
              tuning_trials.csv  ablation.json  feature_importance.csv
  api/        main.py  schemas.py  db.py  models.py  middleware.py  routers/
  dashboard/  index.html  predict.html  intelligence.html  inventory.html  analysis.html
              config.js  css/style.css  js/(common api charts home predict intelligence inventory analysis).js
  scripts/    setup.ps1  setup.sh  doctor.py  load_mysql.py  reset_demo.py  api_client.py
              smoke.py  bench_api.py  make_mocks.py  start_server.bat  start_ngrok.bat
  postman/    Shelfwatch.postman_collection.json  Shelfwatch.local.postman_environment.json
  kaggle/     dataset/  notebook/  model/
  docs/       SRS.md  STAKEHOLDER.md  DEPLOY_LOCAL_NGROK.md  EVALUATION_PLAN.md  JS_GUIDE.md
  tests/      test_data_prep  test_enrichment  test_intelligence  test_features  test_models
              test_risk_engine  test_db_crud  test_workflow  test_middleware  test_api  test_contracts
  requirements.txt
```

---

## 4. Data Requirements

### 4.1 Source dataset

UCI Online Retail: a UK-based online gift retailer, transactions from 1 December 2010 to 9 December 2011.

**Datasets used.** Primary: UCI Online Retail (id 352) supplies the weekly demand table, the train and test periods and every forecast. Supporting: UCI Online Retail II (id 502), whose 2009-2010 sheet supplies prior-year demand for the same products, and UK bank holidays from the `holidays` package. Online Retail II covers the same retailer and overlaps the primary file from 2010-12-01, so it extends the history; it is not an independent test set. A consistency check on the overlap is shown on the analysis board.

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

Other measured facts: the line-level quantity cap is 480 units (99.9th percentile of training rows; 415 lines capped); the largest raw single lines are 74,215 and 80,995 units; 1,224 products have at least 30 active training weeks; the final table has 5,067 product-week rows, 100 products and 52 full weeks (2010-12-06 to 2011-11-28); about 2.5% of product-weeks are zero; the shop-closure week 2010-12-27 (no sales for any product) is left out of the table, so 51 of the 52 weeks carry rows, and it is listed in closed\_weeks.csv; each product is zero-filled from its first sale through 2011-11-28, so products that stopped selling (21829 and 22740) show real zero weeks in the test period.

### 4.3 Data processing requirements

| ID | Requirement | Priority |
| --- | --- | --- |
| FR-D-1 | The system MUST load the Excel file once and cache a parquet copy for later runs. | MUST |
| FR-D-2 | The system MUST remove cancelled invoices (`InvoiceNo` starting with `C`), rows with `Quantity <= 0`, rows with `UnitPrice <= 0`, and non-product `StockCode` values (fixed list, codes with no digits, and prefixes `GIFT` and `TEST`). | MUST |
| FR-D-3 | The system MUST keep rows with missing `CustomerID`. | MUST |
| FR-D-4 | The system MUST define weeks as Monday-start and retain only full weeks, computed from the data range. | MUST |
| FR-D-5 | The system MUST cap line-level `Quantity` at the 99.9th percentile, with the cap learned on training-period rows only. | MUST |
| FR-D-6 | The system MUST select the Top 100 products by total units in the training period, requiring sales in at least 30 training weeks. | MUST |
| FR-D-7 | The system MUST aggregate to product x week (`units`, `orders`, `customers`, `avg_price`), zero-fill missing weeks from each product's first active week through the last full week (2011-11-28) with forward-filled price; weeks before a product's first sale and the shop-closure week are never filled. | MUST |
| FR-D-8 | The system MUST detect shop-wide closed weeks, leave them out of the weekly table (no row, not filled with zero) and export them to `closed_weeks.csv`. | MUST |
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
| Training | 44 | 2010-12-06 to 2011-10-03 (35 target weeks after the closure week and 8 history weeks) |
| Test | 8 | 2011-10-10 to 2011-11-28 |
| Forecast target | 1 | 2011-12-05 |

The test period MUST NOT be used for feature selection, tuning, imputation fitting, outlier capping, or product selection.

### 4.6 Enrichment and external data

Three enrichment sources add information the weekly table lacks. Enrichment tables hold the value **for** week w; only `src/features.py` applies the one-week shift, so the leakage rule lives in one place.

| Source | Gives | Known in advance? | Leakage rule |
| --- | --- | --- | --- |
| UK bank holidays (Python `holidays` package, UK, England) | `uk_bank_holiday_count`, `trading_days_expected`, Christmas-week flag | Yes, calendar | Target-week value may be used directly |
| Online Retail II, 2009-2010 sheet (UCI id 502, same retailer one year earlier) | Prior-year units for the same product and week | Yes, it is the past | Use only weeks t-54 to t-50; clean exactly like the main file; reuse the training cap |
| Raw transactions (existing file, including cancelled invoices) | `cancelled_units`, `gross_units`, `export_units`, `guest_units` per product-week | No, same-week | `shift(1)` in `features.py` |

**Not used (decision).** Weather and national retail-index series are excluded from the model: a one-year panel supplies only 12 monthly or 52 weekly values that move with time, so their effect cannot be separated from a time trend. They may appear in exploratory analysis only.

| ID | Requirement | Priority |
| --- | --- | --- |
| FR-X-1 | The system MUST build `calendar_weekly` from the `holidays` package and verify that 2011-04-22, 2011-04-25, 2011-04-29 and 2011-05-02 are bank holidays. `trading_days_expected` = the shop's trading weekdays (derived from the training data; there are no Saturday sales) minus bank holidays. | MUST |
| FR-X-2 | The system MUST build `prior_year_weekly` from the Online Retail II 2009-2010 sheet with the same cleaning rules and week definition, and report coverage (share of the 100 products with at least 26 prior-year weeks). | MUST |
| FR-X-3 | The system MUST build `enrichment_weekly` from the raw transactions, using cancelled invoices only for `cancelled_units`; cancellations never enter the demand target. | MUST |
| FR-X-4 | If the Online Retail II file is absent the pipeline MUST still run, set prior-year features to missing with `yoy_available = 0`, and log a warning. | MUST |
| FR-X-5 | All sources MUST be cited in `docs/DATA_NOTES.md` (dataset DOI and library version). | MUST |
| FR-X-6 | Static product attributes computed over the whole training period (for example customer concentration) MUST NOT be model features; they are for intelligence and display only. | MUST |

### 4.7 Data stored in MySQL

All project data lives in one MySQL 8 database named `shelfwatch`. Reference tables are loaded from the pipeline outputs by `scripts/load_mysql.py`; system tables are written by the application. The API reads from MySQL. Model files stay on disk and are registered in the database.

| Group | Tables | Written by |
| --- | --- | --- |
| Sources and lineage | `data_sources`, `pipeline_log` | Loader |
| Demand data | `products`, `weekly_demand`, `closed_weeks`, `calendar_weekly`, `prior_year_weekly`, `enrichment_weekly`, `transactions_top100` | Loader |
| Modelling results | `model_registry`, `predictions`, `metrics`, `tuning_trials`, `ablation`, `feature_importance`, `next_week_forecast`, `error_analysis` | Loader (from training) |
| Intelligence and backtest | `product_intelligence`, `basket_pairs`, `policy_backtest`, `policy_backtest_weekly` | Loader |
| System | `inventory_levels`, `replenishment_plans`, `purchase_orders`, `api_logs`, `prediction_log` | API |

| ID | Requirement | Priority |
| --- | --- | --- |
| FR-DS-1 | The reference tables MUST be loaded by one idempotent script that reports the row count of every table. | MUST |
| FR-DS-2 | `transactions_top100` MUST hold the invoice lines (including cancelled lines, flagged) of the 100 products so any product-week can be traced to its source lines. It MUST NOT store `CustomerID`; only a `has_customer` flag. | MUST |
| FR-DS-3 | `data_sources` and `pipeline_log` MUST hold the lineage shown on the analysis board: both datasets with row counts and date ranges, and the cleaning funnel. | MUST |
| FR-DS-4 | The schema MUST be in `sql/schema.sql` with primary keys, foreign keys on `product_id` and indexes on (`product_id`, `week_start`). | MUST |

---

## 5. Functional Requirements: Features, Models, Evaluation

### 5.1 Feature engineering

| ID | Requirement | Priority |
| --- | --- | --- |
| FR-F-1 | For target week t the system MUST build lags of units at t-1, t-2, t-3, t-4. | MUST |
| FR-F-2 | The system MUST build rolling mean and standard deviation over 4 and 8 weeks, always applying `shift(1)` before `rolling()`. | MUST |
| FR-F-3 | The system MUST build `week_of_year` and `month`. `weeks_to_christmas` MUST be tested by ablation before use. | MUST |
| FR-F-4 | The system MUST build previous-week price, orders and customers. | MUST |
| FR-F-5 | The system MUST build past mean and standard deviation of each product's units from data before t only. | MUST |
| FR-F-6 | Closed weeks MUST be missing: never a target, skipped by lags and rolling windows. Missing values MUST be imputed inside the model pipeline with statistics fitted on training data only (LightGBM may use native missing handling). | MUST |
| FR-F-7 | Rows without enough history MUST be dropped and counted. | MUST |
| FR-F-8 | An automated leakage test MUST change units, orders, customers and price at week t and later, for several products and weeks, on a fixture and on the real table, and prove the features at week t do not change. | MUST |
| FR-F-9 | The ordered feature list MUST be saved to `feature_list.json` and reused unchanged by the API. | MUST |
| FR-F-10 | Group G1 (derived behaviour): `order_size_lag1` (units / orders at t-1), `trend_ratio` (roll\_mean\_4 / roll\_mean\_8), `price_index` (previous price / mean price of the previous 8 weeks), `cv_4` (roll\_std\_4 / (roll\_mean\_4 + 1)), `zero_streak` (consecutive zero weeks before t). | MUST |
| FR-F-11 | Group G2 (calendar): `uk_bank_holiday_count` and `trading_days_expected` for the target week, from `calendar_weekly`. | MUST |
| FR-F-12 | Group G3 (prior year): `yoy_mean_5w` (mean units over weeks t-54 to t-50), `yoy_seasonal_index` (that mean divided by the product's mean over its prior-year weeks) and `yoy_available`. | SHOULD |
| FR-F-13 | Group G4 (transaction-derived, shifted): `return_rate_4w`, `export_share_4w`, `guest_share_4w`. `guest_share_4w` is a hypothesis (a missing CustomerID is not documented as a guest checkout) and MUST be reported as such. | SHOULD |
| FR-F-14 | Feature groups MUST be switchable by name so ablation can compare base, base+G1, +G2, +G3, +G4 and all. | MUST |
| FR-F-15 | Each group MUST be kept or dropped using validation WAPE only, never the test set, and its effect MUST be reported even when negative. | MUST |
| FR-F-16 | `build_custom_features(series, target_week)` MUST build the same features for a user-entered series (used by `/api/predict/custom`); groups G3 and G4 are missing there. | MUST |

Expected size: measured on the real data, 3,467 training rows over 35 target weeks (2011-02-07 to 2011-10-03) and 800 test rows (100 products x 8 weeks). The 44 training weeks include the closure week and 8 history weeks per product that serve only as lag history, which leaves 35 target weeks.

### 5.2 Modelling

| ID | Requirement | Priority |
| --- | --- | --- |
| FR-M-1 | The system MUST implement two baselines: naive (last week) and 4-week moving average. | MUST |
| FR-M-2 | The system MUST train Ridge, Random Forest and LightGBM (or XGBoost) on identical features and splits. | MUST |
| FR-M-3 | The system MAY add Poisson or Tweedie objectives for count data. | MAY |
| FR-M-4 | Hyperparameters MUST be tuned with a week-based `TimeSeriesSplit` inside the training period only, with a documented search space and budget. | MUST |
| FR-M-5 | The system MUST report MAE, RMSE, WAPE and bias for every model on validation and test. MAPE MUST NOT be used. | MUST |
| FR-M-6 | Predictions MUST be clipped at zero. | MUST |
| FR-M-7 | The best model MUST be selected by validation WAPE, ties broken by validation bias; the test set is used only for the final report. | MUST |
| FR-M-8 | A model MUST be reported honestly even if it does not beat the baselines. | MUST |
| FR-M-9 | The best model MUST be refit on the full training period and saved with `joblib`, with `feature_list.json` and `config.json` (model name, horizon, split dates, seed, library versions). | MUST |
| FR-M-10 | The system MUST produce `next_week_forecast.csv` (C4) for week 2011-12-05. `sigma_error` MUST come from out-of-fold validation errors only; with fewer than 6 observations the pooled value of the product's volume segment is used. | MUST |
| FR-M-11 | A scale-aware candidate MUST be trained on target divided by a rolling mean (4 or 8 weeks, chosen on validation) and multiplied back at prediction time, because default LightGBM over-forecast in M5. | MUST |
| FR-M-12 | An ensemble candidate (average of the best ML model and the moving average, weights chosen on validation only) SHOULD be evaluated. | SHOULD |
| FR-M-13 | Quantile models (LightGBM, alpha 0.1 and 0.9) MUST produce `forecast_p10` and `forecast_p90`; empirical coverage on validation and test MUST be reported (target about 80%). | MUST |
| FR-M-14 | `predict_custom(payload)` MUST load the saved model and return forecast, p10 and p90 for a user-entered series. | MUST |
| FR-M-15 | A pretrained time-series model (zero-shot or fine-tuned on a Kaggle GPU) MAY be added as one more candidate. No improvement may be claimed before it is measured. | MAY |
| FR-M-16 | All tuning trials (candidate, parameters, fold scores, mean validation WAPE) MUST be saved to models/tuning\_trials.csv and loaded into MySQL so the analysis board can show the tuning process. | MUST |
| FR-M-17 | src/explain.py MUST provide explain\_prediction(...) returning feature values and contributions for a product and week from the saved model, and global feature importance saved to models/feature\_importance.csv. | MUST |

**Candidate hyperparameter search (guideline):** Ridge `alpha` in a log range; Random Forest `n_estimators`, `max_depth`, `min_samples_leaf`; LightGBM `num_leaves`, `learning_rate`, `n_estimators`, `min_child_samples`, `subsample`. Use random search with a fixed budget (for example 30 trials) to stay within the timeline.

**Observed in M5 (default settings, test set).** Moving average WAPE 0.4801 (bias -5.94), Ridge WAPE 0.4913 (bias -0.18), LightGBM WAPE 0.5315 (bias +26.23). The baseline won before tuning, and LightGBM over-forecast. The earlier worry that trees would under-forecast the higher October and November level is therefore not what happened; the likelier cause is overfitting to wholesale spikes when the target is in raw units across products of very different size. Planned responses, in order: stronger regularisation, the scale-aware target (FR-M-11), the ensemble (FR-M-12), and the extra feature groups. The per-week bias check across the 8 test weeks remains required. An honest result may be that the best model beats the moving average only slightly; that is acceptable if reported as measured.

### 5.3 Forecast-error analysis

| ID | Requirement | Priority |
| --- | --- | --- |
| FR-E-1 | Error MUST be reported by volume segment (low, medium, high sellers, tertiles of average weekly units). | MUST |
| FR-E-2 | Error MUST be reported by test week, with comments on weeks of spikes. | MUST |
| FR-E-3 | The analysis MUST include predicted-versus-actual plots and the residual distribution. | MUST |
| FR-E-4 | The analysis MUST list best and worst products with a reason (for example intermittent demand or bulk orders). | MUST |
| FR-E-5 | Over-forecast versus under-forecast MUST be quantified separately, because under-forecasting leads to stockouts. | MUST |
| FR-E-6 | Results MUST be exported to `error_analysis.csv` (segment, week and product aggregates) for the dashboard. | SHOULD |

### 5.4 Product intelligence (derived information)

These outputs turn the transactions into information a planner can act on. All are computed from the **training period only** and exported to `product_intelligence.csv` and `basket_pairs.csv`.

| ID | Requirement | Priority |
| --- | --- | --- |
| FR-I-1 | Demand pattern class per product (smooth, erratic, intermittent, lumpy) from ADI (weeks open divided by weeks with demand) and CV squared of non-zero demand, with the standard cutoffs 1.32 and 0.49. It MUST also be computed for all products with enough history to show why the top 100 were chosen. | MUST |
| FR-I-2 | ABC class by cumulative revenue share (A up to 80%, B up to 95%, C the rest, revenue = units x price) and XYZ class by coefficient of variation (X up to 0.5, Y up to 1.0, Z above; common convention, thresholds documented). | MUST |
| FR-I-3 | Lifecycle stage per product (new, growing, mature, declining, inactive) from training-period trend and recency, with documented thresholds; products with no sales in the last 8 training weeks are inactive. | MUST |
| FR-I-4 | Return rate = cancelled units divided by gross units sold, from the cancelled invoices that are excluded from demand. | MUST |
| FR-I-5 | Export share (units outside the United Kingdom) and top-3 customer share (known customers only, with the known-customer coverage stated). | SHOULD |
| FR-I-6 | Bought-together pairs: up to 3 companions per product by lift among the top 100, with a minimum number of invoices (at least 30), exported with confidence, lift and invoice count. The interface MUST show only real computed values. | SHOULD |
| FR-I-7 | Intelligence MUST be computed on training weeks only, validated by tests, and served by the API. | MUST |
| FR-I-8 | The report MUST state the class distributions honestly. At weekly level most top-100 products are expected to be smooth or erratic because only about 2.5% of product-weeks are zero. | MUST |

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

1. **Stock.** If the product has a user-entered stock record, `on_hand` is that value and `on_hand_source = "user"`. Otherwise `on_hand_i = initial_stock_weeks x avg_weekly_units_train_i x noise_i` and the source is `"simulated"`. The noise is drawn once per product from Uniform(0.5, 1.5) using a generator seeded with `seed` over products sorted by `product_id`, so it never depends on another parameter.
2. `forecast_lead_time_i = forecast_weekly_i x L`, where L is the lead time in weeks (lead time plus any delay shock in the what-if simulator).
3. `z = norm.ppf(service_level)`
4. `safety_stock_i = z x sigma_error_i x sqrt(L)`
5. `reorder_point_i = forecast_lead_time_i + safety_stock_i`
6. `days_of_cover_i = on_hand_i / (forecast_weekly_i / 7)`; if the forecast is 0, days of cover is set to 999.
7. **Risk rule.** HIGH if `on_hand < forecast_lead_time`; MEDIUM if `on_hand < reorder_point` and not HIGH; LOW otherwise.
8. **Replenishment advice** (never changes the risk class). `on_order` = quantities of Pending and Approved purchase orders for the product; `inventory_position = on_hand + on_order`; `order_up_to = forecast_weekly x (L + 1) + z x sigma_error x sqrt(L + 1)` (weekly review); `recommended_order_qty = max(0, ceil(order_up_to - inventory_position))`; `days_to_reorder = max(0, (on_hand - reorder_point) / (forecast_weekly / 7))`, where 0 means order now; `revenue_at_risk = max(0, forecast_lead_time - on_hand) x last_avg_price`.
9. **Stockout probability.** `stockout_probability = 1 - Phi((on_hand - forecast_lead_time) / (sigma_error x sqrt(L)))` under a normal approximation. If `sigma_error` is 0 it is 1 when `on_hand < forecast_lead_time` and 0 otherwise.

10) **Action label** (what the manager should do next). If the product's lifecycle stage is inactive, the action is `REVIEW_STOP` and `recommended_order_qty` is set to 0. Otherwise, if `risk_level` is HIGH, or `days_to_reorder` is 0 with `recommended_order_qty` above 0, the action is `ORDER_NOW`. Otherwise, if `days_to_reorder` is at most 7 with `recommended_order_qty` above 0, it is `ORDER_SOON`. In every other case it is `HOLD`. If lifecycle data is not available, only the risk and reorder rules apply.

Comparisons use unrounded values; displayed values are rounded to one decimal place. Equality (on-hand exactly equal to a threshold) is NOT below the threshold, so it falls in the safer class.

### 6.3 Requirements

| ID | Requirement | Priority |
| --- | --- | --- |
| FR-R-1 | `compute_risk(forecast_df, params, stock_overrides, open_orders, lifecycle)` MUST return `product_id, description, forecast_weekly, forecast_p10, forecast_p90, forecast_lead_time, on_hand, on_hand_source, safety_stock, reorder_point, days_of_cover, stockout_probability, on_order, recommended_order_qty, days_to_reorder, revenue_at_risk, risk_level, action` for every product. | MUST |
| FR-R-2 | Results MUST be deterministic for identical inputs and seed. | MUST |
| FR-R-3 | Missing parameters take defaults. Out-of-range or wrongly typed parameters MUST raise `ValueError` (HTTP 422). Allowed: `lead_time_weeks` integer 1 to 6 (the sliders and `/api/risk` use 1 to 4; `/api/simulate` adds `delay_weeks` 0 to 2), `service_level` 0.80 to 0.99, `initial_stock_weeks` 0.5 to 8. | MUST |
| FR-R-4 | The engine MUST handle edge cases: forecast 0, sigma 0, longest lead time, service level 0.99, empty input, empty overrides, orders and lifecycle. | MUST |
| FR-R-5 | `sensitivity_table` MUST return counts of HIGH, MEDIUM and LOW for every combination of the supplied lead times and service levels (default 1 to 4 and 0.90, 0.95, 0.99). | MUST |
| FR-R-6 | A summary MUST report counts per class, counts per action, share at risk, average days of cover and total revenue at risk. | MUST |
| FR-R-7 | The simulated-inventory disclaimer MUST accompany every risk output and every product whose stock source is simulated MUST be labelled so. | MUST |
| FR-R-8 | The report MUST state how the defaults were chosen; they MUST NOT be tuned to produce a desired number of flagged products. | MUST |
| FR-R-9 | The system SHOULD include a replenishment-policy backtest over the 8 test weeks using actual demand: with a one-week lead time and weekly review, compare policies (naive forecast, moving average, best model, best model with safety stock) on fill rate, number of stockout weeks and average on-hand units. It uses the same one-step forecast simplification as the risk rule, and results are reported as measured. This is the manager's own success measure. | SHOULD |
| FR-R-10 | A user-entered stock record MUST replace the simulated value for that product and MUST be shown as `"user"`. | MUST |
| FR-R-11 | Replenishment advice MUST follow 6.2 item 8, using open purchase orders as `on_order`. | MUST |
| FR-R-12 | `/api/simulate` MUST accept lead time, delay shock, service level and starting stock, and return the new summary with the change against the current settings. | MUST |
| FR-R-13 | `stockout_probability` MUST follow 6.2 item 9, and the report MUST state the normal-error assumption. | MUST |
| FR-R-14 | `action` MUST follow 6.2 item 10 and is the field the Home Action list is built from. | MUST |

### 6.4 Modelling limitations to document

- The one-step forecast is reused for every week of the lead time (`forecast_weekly x L`), a simplification.
- Stock is a snapshot at the start of the forecast week; in-transit stock is represented only by open purchase orders.
- With the default 3 weeks of starting stock and L = 1, few products may be HIGH. In the browser prototype (30 products, baseline forecast) the defaults gave 1 HIGH, 7 MEDIUM and 22 LOW. This reflects the assumptions, not a model failure, and must be reported together with the sensitivity table.
- Simulated stock is illustrative. User-entered stock is only as good as the entry, and can become out of date.
- `stockout_probability` assumes normally distributed forecast errors; `revenue_at_risk` is an approximation using the latest average price.

---

## 7. API Specification

Base URL locally: `http://localhost:8000`. Interactive documentation is generated at `/docs`.

### 7.1 Endpoints

| Method and path | Purpose | Success | Errors |
| --- | --- | --- | --- |
| GET `/` | The web application (Home page, no login) | 200 | n/a |
| GET `/health` | Same as `/api/health` (mirrors the mentor's example) | 200 | n/a |
| POST `/predict` | Alias of `POST /api/predict`, same body and response (mirrors the mentor's example) | 200 | 404, 422 |
| GET `/docs` | Interactive Swagger documentation | 200 | n/a |
| GET `/api/health` | Liveness, database status and row counts of key tables | 200 | n/a |
| GET `/api/disclaimer` | Simulated-inventory text | 200 | n/a |
| GET `/api/model-info` | Model name, versions, headline metrics, data version, Kaggle links | 200 | 500 |
| GET `/api/products` | Products with `total_units` and demand pattern class | 200 | 500 |
| GET `/api/forecast/{product_id}` | History (52 weeks, closure week null), test predictions, next-week forecast with p10 and p90 | 200 | 404 |
| POST `/api/predict` | Forecast, range, risk, action and advice for a product, with optional stock, lead time and service level overrides | 200 | 404, 422 |
| POST `/api/predict/custom` | Forecast and range from 8 to 12 user-entered weekly unit values | 200 | 422 |
| GET `/api/predictions/history` | Recent predictions saved in `prediction_log` | 200 | 422 |
| POST `/api/risk` | Risk table, actions and summary for given parameters | 200 | 422 |
| POST `/api/simulate` | What-if: lead time, delay shock, service level, starting stock | 200 | 422 |
| POST `/api/sensitivity` | HIGH, MEDIUM, LOW counts per lead time and service level | 200 | 422 |
| GET `/api/metrics` | Model comparison, validation and test | 200 | 500 |
| GET `/api/error-analysis` | Error by segment, week, product | 200 | 500 |
| GET `/api/intelligence` | Product intelligence for all products | 200 | 500 |
| GET `/api/intelligence/{product_id}` | One product plus bought-together companions | 200 | 404 |
| GET `/api/inventory` | User-entered stock records | 200 | n/a |
| PUT `/api/inventory/{product_id}` | Create or replace stock, lead time, service level (needs API key) | 200, 201 | 404, 422 |
| DELETE `/api/inventory/{product_id}` | Remove the record; the product returns to simulated stock (needs API key) | 204 | 404 |
| POST `/api/orders` | Create a purchase order (needs API key) | 201 | 404, 422 |
| GET `/api/orders` | List orders, filter by status, product or plan | 200 | 422 |
| GET `/api/orders/{id}` | One order | 200 | 404 |
| PATCH `/api/orders/{id}` | Change status or quantity (needs API key) | 200 | 404, 409, 422 |
| DELETE `/api/orders/{id}` | Delete a Pending or Cancelled order (needs API key) | 204 | 404, 409 |
| POST `/api/plans/generate` | Create a Draft weekly plan from the Action list (needs API key) | 201 | 409 |
| GET `/api/plans`, GET `/api/plans/{id}` | List plans, one plan with its orders | 200 | 404 |
| PATCH `/api/plans/{id}` | Approve or cancel a plan (needs API key) | 200 | 404, 409, 422 |
| GET `/api/plans/{id}/export` | The plan as CSV for the supplier | 200 | 404 |
| GET `/api/analysis/lineage` | Both datasets, cleaning funnel, split description, overlap check | 200 | 500 |
| GET `/api/analysis/features` | Feature groups, importance, ablation | 200 | 500 |
| GET `/api/analysis/tuning` | Candidates, best parameters, tuning trials | 200 | 500 |
| GET `/api/analysis/evaluation` | Metrics, error by week, segment and pattern class, interval coverage | 200 | 500 |
| GET `/api/analysis/backtest` | Policy backtest summary and, with `product_id`, the week-by-week replay | 200 | 404 |
| GET `/api/explain/{product_id}` | How a forecast was produced: data used, features, contributions | 200 | 404 |
| GET `/api/data/transactions` | Invoice lines behind a product-week (no customer id) | 200 | 404, 422 |
| GET `/api/logs` | Recent audit-log entries, paged | 200 | 422 |

All endpoints live under `/api`, so they never collide with the dashboard pages. Field-level schemas are in `CONTRACTS.md` section C7 as amended by `CONTRACTS_ADDENDUM.md`.

### 7.2 Requirements

| ID | Requirement | Priority |
| --- | --- | --- |
| FR-A-1 | The API MUST load the model, feature list, config and result files once at start-up, create the database tables, and fail with a clear message if a required file is missing. | MUST |
| FR-A-2 | All request bodies MUST be validated with Pydantic models and the ranges in FR-R-3. | MUST |
| FR-A-3 | Errors MUST be JSON `{"detail": "..."}` with correct status codes; stack traces MUST NOT be exposed. | MUST |
| FR-A-4 | CORS MUST be enabled for the dashboard origin(s), set by an environment variable. | MUST |
| FR-A-5 | Dates MUST be ISO 8601 strings and numbers MUST be finite (no NaN). | MUST |
| FR-A-6 | `/api/risk` and `/api/sensitivity` SHOULD cache results by parameter set; the cache MUST be cleared on any inventory or order write. | SHOULD |
| FR-A-7 | The API MUST NOT expose `CustomerID` or any personal data. | MUST |
| FR-A-8 | Request handling MUST be stateless apart from the database. | MUST |
| FR-A-9 | The API MUST serve the files in `dashboard/` at the root path, registered after all API routes. | MUST |
| FR-A-10 | Every JSON endpoint MUST live under the `/api` prefix. | MUST |
| FR-A-11 | `/api/predict` and `/api/predict/custom` MUST use the saved model and the same feature code as training. | MUST |
| FR-A-12 | `/docs` MUST list every endpoint with request and response examples. | MUST |
| FR-A-13 | The API MUST also answer GET /health and POST /predict at the root path (same behaviour as the /api versions), so the mentor's Postman steps work unchanged. | MUST |
| FR-A-14 | Reference and analysis data MUST be read from MySQL at request time; only the saved model files are read from disk (at start-up). | MUST |
| FR-A-15 | Every /predict call MUST be saved to prediction\_log with its inputs, outputs and model name. | MUST |

### 7.3 Database and CRUD

The database is MySQL 8 (schema `shelfwatch`), accessed through SQLAlchemy 2 with the PyMySQL driver. `DATABASE_URL`, for example `mysql+pymysql://user:password@localhost:3306/shelfwatch`, is read from a `.env` file that is not committed. All tables are listed in 4.7 and defined in `sql/schema.sql`. The system tables written by the application are:

| Table | Key columns | Purpose |
| --- | --- | --- |
| `inventory_levels` | `product_id` (primary key), `on_hand` (>= 0), `lead_time_weeks` (1 to 6), `service_level` (0.80 to 0.99), `updated_at` | Real stock replacing simulated stock |
| `replenishment_plans` | `id`, `week_start`, `status` (Draft, Approved, Cancelled), `note`, `created_at`, `approved_at` | A weekly plan that groups orders |
| `purchase_orders` | `id`, `plan_id` (optional foreign key), `product_id`, `quantity` (> 0), `supplier_note`, `status`, `created_at`, `updated_at` | Restock orders |
| `prediction_log` | `id`, `request_id`, `endpoint`, `product_id`, `inputs` (JSON), `outputs` (JSON), `model_name`, `created_at` | Every prediction made |
| `api_logs` | `id`, `request_id`, `method`, `path`, `status`, `duration_ms`, `created_at` | Audit log written by middleware |

Order status moves only Pending to Approved, Pending to Cancelled, Approved to Received, Approved to Cancelled. Only Pending or Cancelled orders can be deleted. Receiving an order adds its quantity to the product's stock record (created from the current simulated stock if none exists). Status codes: 201 create, 200 read or update, 204 delete, 404 unknown product or id, 409 invalid transition or delete, 422 invalid input.

| ID | Requirement | Priority |
| --- | --- | --- |
| FR-DB-1 | `scripts/load_mysql.py` MUST create the schema from `sql/schema.sql` and load every pipeline output idempotently (truncate and reload), reporting row counts. | MUST |
| FR-DB-2 | The API MUST read reference data from MySQL; model files are loaded from disk at start-up and registered in `model_registry`. | MUST |
| FR-DB-3 | All database access MUST use SQLAlchemy with parameterised queries; SQL MUST NOT be built from strings. | MUST |
| FR-DB-4 | CRUD behaviour and status transitions MUST be enforced as above. | MUST |
| FR-DB-5 | When a stock record exists, `compute_risk` MUST use it (source `user`), open orders MUST count as `on_order`, and receiving an order MUST add its quantity to stock. | MUST |
| FR-DB-6 | Any write MUST clear cached risk results. | MUST |
| FR-DB-7 | `DATABASE_URL` MUST come from `.env`; `.env` MUST NOT be committed and `.env.example` MUST be provided. | MUST |
| FR-DB-8 | Each prediction request MUST be saved in `prediction_log`. | MUST |
| FR-DB-9 | No personal data: `CustomerID` MUST NOT be stored in the database (only a `has_customer` flag). | MUST |
| FR-DB-10 | `/api/health` MUST report database connectivity and the row counts of key tables. | MUST |
| FR-DB-11 | `scripts/reset_demo.py` MUST reset the system tables to a seeded demo state for evaluation. | MUST |
| FR-DB-12 | The README MUST document how to back up and restore the database with `mysqldump`. | SHOULD |
| FR-DB-13 | Tests SHOULD run on in-memory SQLite for speed and on MySQL when `TEST_DATABASE_URL` is set. | SHOULD |

### 7.4 Middleware

| ID | Middleware | Behaviour | Priority |
| --- | --- | --- | --- |
| FR-MW-1 | CORS | Allow-list from `ALLOWED_ORIGINS` | MUST |
| FR-MW-2 | Request ID and timing | Adds `X-Request-ID` and `X-Process-Time-Ms` headers to every response | MUST |
| FR-MW-3 | Audit log | Writes method, path, status and duration to `api_logs`; skips `/api/logs` and static files | MUST |
| FR-MW-4 | API key | Write requests (PUT, POST on orders, PATCH, DELETE) need header `X-API-Key` equal to the `API_KEY` environment variable, otherwise 401; reads are open | MUST |
| FR-MW-5 | Rate limit | In-memory limit per client address (default 120 requests per minute), otherwise 429 | SHOULD |
| FR-MW-6 | Error handler | Converts unhandled errors to a JSON 500 without a stack trace and logs details server-side | MUST |

The API key is visible to anyone who inspects the page, so it protects against casual misuse only. Real systems would use user login, which is out of scope.

### 7.5 Python API client

| ID | Requirement | Priority |
| --- | --- | --- |
| FR-PC-1 | `scripts/api_client.py` MUST provide functions for health, products, predict, custom predict, risk, simulate, inventory CRUD and order CRUD, using `httpx` with timeouts, up to 2 retries on 5xx responses, and clear error messages. | MUST |
| FR-PC-2 | `python scripts/api_client.py demo` MUST walk through a full sequence: health, predict, create an order, approve it, update stock, delete the order. | MUST |
| FR-PC-3 | The client SHOULD be covered by a test that runs against a local server. | SHOULD |

### 7.6 Replenishment workflow

The manager's weekly routine is a workflow, not a form. **1 Generate:** the system turns the Action list (ORDER\_NOW and ORDER\_SOON) into a Draft plan with one Pending order per product at the recommended quantity. **2 Review:** the manager edits quantities or removes lines. **3 Approve:** approving the plan sets its Pending orders to Approved. **4 Export:** a CSV for the supplier. **5 Receive:** marking an order Received adds stock, so the next risk calculation reflects it and the loop closes.

| ID | Requirement | Priority |
| --- | --- | --- |
| FR-W-1 | `POST /api/plans/generate` MUST create a Draft plan for the current week with one Pending order per product whose action is ORDER\_NOW or ORDER\_SOON and whose recommended quantity is above 0; products already covered by open orders are skipped. A second call while a Draft plan exists for that week returns 409. | MUST |
| FR-W-2 | Plan lines are purchase orders carrying `plan_id`; quantities and statuses are edited with the order endpoints. | MUST |
| FR-W-3 | `PATCH /api/plans/{id}` with Approved sets all its Pending orders to Approved; with Cancelled it cancels them; other changes return 409. | MUST |
| FR-W-4 | `GET /api/plans/{id}/export` MUST return CSV (`product_id, description, quantity, status`) with the right content type. | MUST |
| FR-W-5 | Receiving an order MUST update stock and clear the risk cache. | MUST |
| FR-W-6 | Every plan action MUST appear in `api_logs` and in the analysis board's system log. | MUST |

### 7.7 Analysis and explanation

The analysis board and the Predict page explain how outputs are produced. Every endpoint below reads from MySQL tables.

| ID | Requirement | Priority |
| --- | --- | --- |
| FR-AN-1 | Every `/api/analysis/*` endpoint MUST read from MySQL tables, not from files. | MUST |
| FR-AN-2 | `/api/explain/{product_id}` MUST return, for the chosen target week: the last 12 weeks of units used; each feature value with its group; each feature's contribution to the forecast (LightGBM contributions, or coefficient times scaled value for Ridge); the prior-year weeks used; the bank-holiday facts of the target week; the model name and the interval. | MUST |
| FR-AN-3 | The explanation MUST use the same saved model and feature code as the forecast, and a test MUST check that the contributions plus the base value reproduce the model output. | MUST |
| FR-AN-4 | `/api/data/transactions` MUST return at most 200 lines per call and MUST NOT include any customer identifier. | MUST |
| FR-AN-5 | `/api/analysis/lineage` MUST show the two datasets separately with row counts and date ranges, the cleaning funnel from raw rows to the weekly table, the train and test split, and the result of the overlap check. | MUST |
| FR-AN-6 | `/api/analysis/backtest` MUST return the policy summary and, for a chosen product, the week-by-week replay (stock, order, arrival, demand, unmet demand). | MUST |

---

## 8. User Interface Specification

### 8.1 Global layout

- **Disclaimer strip (all pages):** the simulated-inventory text from `/api/disclaimer`.
- **Control rail (Home and Predict):** sliders for lead time (1 to 4 weeks), service level (0.80 to 0.99) and starting stock (0.5 to 8 weeks), plus a reset button.
- **Navigation (all pages):** Home, Predict, Product intelligence, Inventory manager, Analysis board, as ordinary links between five HTML pages.
- **Footer (all pages):** links to the Kaggle model, notebook and dataset, and to the API documentation at `/docs`.
- **No login:** there is no login, sign-up or role-selection page. The site opens directly on Home. Write actions are protected by a server-side API key that the pages send from `config.js`; this is not a user login.

### 8.2 Pages

| Page (file) | Manager's step | Content |
| --- | --- | --- |
| Home (`index.html`) | 1. See what needs attention; 3. Decide the order | Headline stating how many products run out in the simulation; the Action list (Order now, Order soon, Review to stop, Hold) with recommended quantity, days to reorder and revenue at risk; the shelf wall; control rail; a short panel on who the system is for and where it is used; links to the API docs and the Kaggle pages |
| Predict (`predict.html`) | 2. Understand one product; 5. Test assumptions | Choose a product or enter 8 to 12 recent weekly sales, then Predict: forecast, low-to-high range, risk badge, action and advice; a panel **How this forecast was made** (data used, feature values and contributions, prior-year comparison); a what-if simulator; the history of recent predictions |
| Product intelligence (`intelligence.html`) | 6. Set policy | Searchable and filterable table: demand pattern, ABC and XYZ class, lifecycle stage, return rate, export share, bought-together companions |
| Inventory manager (`inventory.html`) | 4. Act | Stock records, purchase orders and the weekly plan workflow (Generate plan, review, Approve, Export CSV, Receive); risk badges and actions update after every change without reloading the page |
| Analysis board (`analysis.html`) | 7. Trust the numbers | Eight sections (8.6): data sources and lineage, training and testing, features, models and tuning, evaluation, business backtest and replay, explain a forecast, system log |

### 8.3 Requirements

| ID | Requirement | Priority |
| --- | --- | --- |
| FR-U-1 | The UI MUST obtain all data through one module, `dashboard/js/api.js`, and MUST contain no model or risk logic in the submitted version. | MUST |
| FR-U-2 | The application MUST be five separate HTML pages with shared navigation links. It MUST NOT be a single-page app that switches views by hash. | MUST |
| FR-U-3 | The disclaimer MUST be visible on every page without scrolling. | MUST |
| FR-U-4 | Risk class MUST be conveyed by text and icon as well as colour. | MUST |
| FR-U-5 | Loading, empty and error states MUST be handled, for example an API-unreachable message with a retry button. | MUST |
| FR-U-6 | Tables MUST support sorting and filtering where they hold more than a few rows. | MUST |
| FR-U-7 | The risk table and the order list SHOULD export to CSV. | SHOULD |
| FR-U-8 | Charts MUST have a title or caption, axis labels with units, and the forecast MUST be visually distinct from actuals. | MUST |
| FR-U-9 | Pages SHOULD be readable at tablet width and usable at phone width. | SHOULD |
| FR-U-10 | The API address and write key MUST come from `dashboard/config.js` (`apiBase` empty for same origin, `apiKey`). | MUST |
| FR-U-11 | The Shelf MUST draw one bin per product: fill = stock in weeks of forecast demand (0 to 6, clamped); red dashed line = demand during delivery; yellow tick = reorder point; colour and icon = risk class; animated refill when a slider changes. | MUST |
| FR-U-12 | The headline MUST state the live count of HIGH-risk products, and selecting a bin MUST open `predict.html?product=<id>`. | MUST |
| FR-U-13 | Light and dark themes MUST follow the system setting with a manual toggle. | MUST |
| FR-U-14 | The UI MUST be keyboard operable with visible focus and MUST respect reduced motion. | MUST |
| FR-U-15 | The Predict page MUST validate inputs in the browser before sending (8 to 12 non-negative numbers, valid date) and handle `submit`, `change` and `input` events. | MUST |
| FR-U-16 | The Inventory page MUST create, edit, approve and delete through `fetch` (PUT, POST, PATCH, DELETE), update the affected row and badge without a page reload, and ask for confirmation before delete. | MUST |
| FR-U-17 | Every page MUST have its own script file plus the shared `common.js` and `api.js`; event handlers MUST be registered with `addEventListener`, not inline attributes. | MUST |
| FR-U-18 | The Home page MUST open with the Action list and a short panel on who the system is for (the Retail Inventory and Replenishment Manager) and where it is used in real life, and MUST link to the Kaggle model, notebook and dataset and to `/docs`. | MUST |
| FR-U-19 | The Intelligence page MUST show only real computed values and an empty state when companion data is missing. | MUST |
| FR-U-20 | The Analysis board MUST contain the eight sections in 8.6 and read only from the analysis endpoints; its system log is read from `/api/logs`. | MUST |
| FR-U-21 | The Home page MUST show an Action list built from the action field: groups Order now, Order soon, Review to stop and Hold, each row with product, recommended quantity, days to reorder and revenue at risk, ordered by urgency; selecting a row opens that product's Predict page. | MUST |
| FR-U-22 | Each page SHOULD state in one line which step of the manager's weekly routine it supports (section 1.6). | SHOULD |
| FR-U-23 | The Analysis board MUST have eight sections: data sources and lineage, training and testing, features, models and tuning, evaluation, business backtest and replay, explain a forecast, system log. | MUST |
| FR-U-24 | The Predict page MUST include a How this forecast was made panel from /api/explain: data used, feature values and contributions as a bar chart, prior-year comparison and the interval. | MUST |
| FR-U-25 | A weekly history point SHOULD offer a drill-down to the invoice lines behind it through /api/data/transactions (no customer identifier). | SHOULD |
| FR-U-26 | There MUST be no login, sign-up or role-selection page; the site opens directly on Home. | MUST |
| FR-U-27 | The Inventory manager MUST provide the plan workflow (Generate plan, review lines, Approve plan, Export CSV, Receive order) with stock, badges and actions updating without a page reload. | MUST |
| FR-U-28 | Every chart and table on the Analysis board MUST carry a caption naming its data source (MySQL table or endpoint) and how it was computed. | MUST |

### 8.4 Front-end design and implementation

The dashboard is a static web app (HTML, CSS, JavaScript) whose look is fixed by a working prototype, [Shelfwatch](https://claude.ai/artifact/WEBAobVRSHian8Hdj5y3tf), built on the real UCI sales of 30 products with a moving-average baseline. The submitted version keeps its look and replaces its in-browser risk calculation with calls to the API.

The prototype is a single page. The submitted version splits it into five pages (8.2) and keeps the shelf wall as the centrepiece of the home page.

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
| Home | `POST /api/risk`, `GET /api/products`, `GET /api/model-info` | Weeks of stock = `on_hand / forecast_weekly`, fill clamped 0 to 6; red line at `forecast_lead_time / forecast_weekly`; yellow tick at `reorder_point / forecast_weekly`; Action list from the `action` field; headline from `summary.n_high`; Kaggle links from `model-info` |
| Predict | `GET /api/products`, `POST /api/predict`, `POST /api/predict/custom`, `POST /api/simulate`, `GET /api/explain/{id}`, `GET /api/predictions/history`, `GET /api/data/transactions` | Result cards, SVG range chart, contribution bar chart, prior-year comparison, simulator on `input` events, drill-down table of invoice lines |
| Product intelligence | `GET /api/intelligence`, `GET /api/intelligence/{id}` | Filter by class, search by code or name, companions list, empty state |
| Inventory manager | `/api/inventory`, `/api/orders`, `/api/plans`, `POST /api/risk` | Edit-stock form, create-order form, plan buttons (Generate, Approve, Export), order buttons (Approve, Receive, Cancel, Delete); after each write re-fetch risk and update the rows |
| Analysis board | `/api/analysis/lineage`, `/features`, `/tuning`, `/evaluation`, `/backtest`, `GET /api/explain/{id}`, `GET /api/logs`, `GET /api/predictions/history` | Funnel chart, split timeline, importance bars, tuning-trial chart, error charts, replay chart, system log tables; each with a source caption |

**Files**

- `index.html`, `predict.html`, `intelligence.html`, `inventory.html`, `analysis.html`: one file per page
- `config.js`: `window.SHELFWATCH = { apiBase: "", apiKey: "demo-key" }` (use `"mock"` for offline work against `dashboard/mock/*.json`)
- `css/style.css`: tokens as CSS variables, light and dark themes
- `js/common.js`: navigation, theme toggle, disclaimer strip, footer links, shared helpers
- `js/api.js`: the only module that talks to the API
- `js/charts.js`: SVG line chart, range chart, bars and heatmap helpers
- `js/home.js`, `js/predict.js`, `js/intelligence.js`, `js/inventory.js`, `js/analysis.js`: one script per page

**Behaviour rules**

- Each page loads `common.js`, `api.js` and its own script; navigation is ordinary links.
- Selecting a bin opens `predict.html?product=<id>`; the page reads the id with `URLSearchParams`.
- Slider changes are debounced by 150 ms; a newer request aborts the older one (`AbortController`); results are cached per parameter set and the cache is cleared after any write.
- Shelf bins are updated in place by changing CSS variables so the refill transition runs.
- Write requests send `X-API-Key`; after a successful write the page re-fetches the affected data instead of reloading.
- Text from the API is inserted with `textContent` or an escape function, never raw into `innerHTML`; no `eval`; no inline event attributes.
- Supported browsers: the latest two versions of Chrome, Edge, Firefox and Safari.

### 8.5 JavaScript concepts the team must be able to explain

The mentor expects every member to understand the JavaScript, not only run it. `docs/JS_GUIDE.md` explains each item with a code excerpt from the project.

| Concept | Where it is used | Question to be ready for |
| --- | --- | --- |
| Event listeners (`click`, `input`, `change`, `submit`, `keydown`) | Sliders, forms, buttons, bins, table headers | What exactly happens, step by step, when a slider moves? |
| `fetch`, `async` and `await`, JSON | `api.js` | What does `await` wait for, and what does `response.json()` return? |
| Error handling (`try` and `catch`, `response.ok`) | `api.js`, every page | What does the user see if the server is down? |
| DOM updates (`createElement`, `textContent`, `classList`, `dataset`) | Shelf wall, tables, result cards | Why use `textContent` and not `innerHTML`? |
| Event delegation | One listener on the shelf container | Why one listener instead of one hundred? |
| Debouncing and `AbortController` | Slider-driven requests | What problem does cancelling the older request solve? |
| `URLSearchParams` | Passing a product from the shelf to Predict | How does the Predict page know which product to show? |
| CSS variables set from JavaScript | Shelf bins | Why does the bin animate without being redrawn? |

### 8.6 Analysis board

The mentor asked for a board that shows the analysis and graphs of what the model computed, which data it used, and how the user gets the output from the trained data. Design rule: every number is traceable. Each chart carries a caption naming its MySQL table or endpoint and how it was computed (FR-U-28).

| Section | Question it answers | What it shows |
| --- | --- | --- |
| 1. Data sources and lineage | Which data was used, and how much was removed? | The primary dataset and Online Retail II with row counts and date ranges; the cleaning funnel from raw rows to the weekly table; the overlap check between the two datasets; the weekly timeline with the closure week marked |
| 2. Training and testing | How was the data split, and why? | A timeline of history-only weeks, training target weeks, the validation folds, the 8 test weeks and the forecast week; row counts; a sentence on why the split is chronological |
| 3. Features | What does the model look at, and does each group help? | Feature groups with their features; importance bars; the ablation table with validation WAPE and bias per group |
| 4. Models and tuning | How were models compared and tuned? | Candidate table; tuning trials per model (trial against validation WAPE); best parameters; search spaces |
| 5. Evaluation | How accurate is it, and where does it fail? | Metrics on validation and test; error by week, volume segment, demand pattern and ABC class; over- versus under-forecast; interval coverage; residual distribution |
| 6. Business backtest and replay | Would following the advice have helped? | Policy table (fill rate, stockout weeks, average on-hand); a week-by-week replay for a chosen product |
| 7. Explain a forecast | How did the system get this number? | For a chosen product and week: the weekly units used, feature values, contributions, prior-year comparison, and a drill-down to the invoice lines |
| 8. System log | What did the system do? | Recent requests, saved predictions and plan actions |

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
| NFR-13 | Persistence | All data is stored in MySQL and survives restarts of the server and the laptop. The system also starts cleanly on a freshly loaded database, and scripts/reset\_demo.py restores the demo state. |
| NFR-14 | Write security | Write endpoints need an API key; SQL is parameterised; every piece of API text is escaped before display. |
| NFR-15 | Evaluation readiness | The system runs for an evaluation of more than 2 hours without crashes or slowdown (3-hour soak test), starts in under 60 seconds, and has a documented fallback (11.7). |
| NFR-16 | Traceability | Every number on the Analysis board can be traced to a MySQL table and a computation described in its caption. |
| NFR-17 | Open access | No login: the base URL opens Home, and every page is reachable within two clicks. |

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
| T-D-1 | Output has exactly 100 products, only Mondays, no duplicate product-week rows, no gaps except the closure week, every product ending at 2011-11-28 | FR-D-4, 6, 7, 9 |
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

**Additional test cases (version 3)**

| ID | Test | Requirement |
| --- | --- | --- |
| T-X-1 | `calendar_weekly` marks 2011-04-22, 04-25, 04-29 and 05-02 as bank holidays and has fewer trading days in those weeks | FR-X-1 |
| T-X-2 | `prior_year_weekly` uses the same cleaning; coverage is printed; the pipeline runs without the Online Retail II file | FR-X-2, FR-X-4 |
| T-X-3 | Enrichment values are for week w; cancellations never reduce demand units | FR-X-3 |
| T-F-3 | Feature groups switch on and off by name and the leakage test passes for every group | FR-F-8, FR-F-14 |
| T-F-4 | Prior-year features use only weeks t-54 to t-50 | FR-F-12 |
| T-M-4 | Quantile outputs are ordered p10 <= forecast <= p90 and coverage is computed | FR-M-13 |
| T-M-5 | `predict_custom` reproduces `build_features` for a series taken from the table | FR-M-14, FR-F-16 |
| T-I-1 | Pattern cutoffs classify hand-made series correctly; ABC shares sum to 100% | FR-I-1, FR-I-2 |
| T-I-2 | Intelligence is unchanged when test-period sales are inflated | FR-I-7 |
| T-R-5 | Stockout probability, advice and revenue at risk match a worked example | FR-R-11, FR-R-13 |
| T-R-6 | A user stock record overrides simulated stock and changes the class and the source label | FR-R-10 |
| T-DB-1 | CRUD happy paths and status codes for inventory and orders | FR-DB-3 |
| T-DB-2 | Invalid status transitions and deleting an Approved order give 409 | FR-DB-3 |
| T-DB-3 | Any write clears the cached risk results | FR-A-6, FR-DB-5 |
| T-MW-1 | Every response carries `X-Request-ID` and `X-Process-Time-Ms` | FR-MW-2 |
| T-MW-2 | A write without the API key gives 401, with the key succeeds, and reads need no key | FR-MW-4 |
| T-MW-3 | Requests appear in `api_logs` and in `/api/logs` | FR-MW-3 |
| T-P-1 | `/api/predict` returns a forecast with p10 <= forecast <= p90 | FR-A-11 |
| T-P-2 | `/api/predict/custom` rejects fewer than 8 values and negative values | FR-A-2 |
| T-U-5 | The five pages load, links work, and the CRUD flow works in a browser without a page reload | FR-U-2, FR-U-16 |
| T-PC-1 | The client demo runs end to end against a local server | FR-PC-2 |

**Additional test cases (version 3.2)**

| ID | Test | Requirement |
| --- | --- | --- |
| T-MY-1 | The schema loads on an empty MySQL database and `load_mysql.py` is idempotent (a second run gives identical row counts) | FR-DB-1 |
| T-MY-2 | The API serves reference data from MySQL; tests pass with the processed CSV files removed | FR-DB-2, FR-A-14 |
| T-MY-3 | No table contains a customer identifier and `/api/data/transactions` returns no customer field | FR-DB-9, FR-AN-4 |
| T-DB-4 | `/api/health` reports database status and row counts; `reset_demo.py` restores the seed state | FR-DB-10, FR-DB-11 |
| T-W-1 | Generate plan creates Pending orders only for ORDER\_NOW and ORDER\_SOON products with quantity above 0; a second call gives 409 | FR-W-1 |
| T-W-2 | Approving a plan approves its Pending orders; cancelling cancels them; an invalid change gives 409 | FR-W-3 |
| T-W-3 | Receiving an order raises stock and changes the next risk result and source label | FR-W-5, FR-DB-5 |
| T-W-4 | Plan export returns CSV with the right header and rows | FR-W-4 |
| T-EX-1 | Contributions plus base value equal the model output for 5 products and 3 weeks | FR-AN-3 |
| T-EX-2 | Explain returns exactly 12 history weeks, the holiday facts and the prior-year weeks (or an explicit none) | FR-AN-2 |
| T-AN-1 | Each analysis endpoint returns data from MySQL tables with a source caption | FR-AN-1, FR-U-28 |
| T-RT-1 | `GET /health` and `POST /predict` at the root behave like the `/api` versions | FR-A-13 |
| T-PM-1 | The Postman collection runs green against the local server and again against the public URL | FR-PM-1 to FR-PM-4 |
| T-U-6 | There is no login page; opening the base URL shows Home | FR-U-26 |
| T-U-7 | The plan workflow works in the browser without a page reload | FR-U-27 |

### 10.3 Quality gates

A pull request merges only when: all tests pass, a teammate has reviewed it, the contract is respected, and (for modelling code) the leakage test passes. A fresh-chat review with Claude (leakage, contracts, formulas, edge cases) is recommended before merging.

---

## 11. Deployment and URL Generation

### 11.1 Environments

| Environment | Use | How |
| --- | --- | --- |
| Local | Development and the live evaluation | Start MySQL, then run `uvicorn api.main:app --port 8000` (or `scripts/start_server.bat`) and open `http://localhost:8000/` and `/docs`. The API serves the pages from the same address. |
| Public | The address given to faculty | Run `ngrok http 8000` (or `scripts/start_ngrok.bat`) in a second window and share the HTTPS address and its `/docs`. See 11.3. |

### 11.2 Configuration

Environment variables are read from a `.env` file that is never committed (`.env.example` lists the names): `DATABASE_URL` (MySQL connection), `API_KEY`, `ALLOWED_ORIGINS` (only needed if the pages are hosted elsewhere), `RATE_LIMIT`, `KAGGLE_DATASET_URL`, `KAGGLE_NOTEBOOK_URL`, `KAGGLE_MODEL_URL`. The pages read the API address from `dashboard/config.js` (`apiBase`, empty for same origin) and send the header `ngrok-skip-browser-warning` on every request.

### 11.3 Options

| Option | Setup | Pros | Cons |
| --- | --- | --- | --- |
| A. Local server and ngrok (required flow, recommended) | MySQL and FastAPI run on the demo laptop (`uvicorn api.main:app --port 8000`); `ngrok http 8000` gives a public HTTPS address; test with Postman | Matches the mentor's instructions; MySQL data stays on the laptop; no cloud account beyond ngrok | The laptop must stay on and connected; if FastAPI or ngrok stops, the public URL stops |
| B. Alternative tunnel | Another tunnel tool, for example a Cloudflare quick tunnel | No account needed for quick tunnels | The address changes on every run |
| C. Cloud backup (optional) | Render or a Hugging Face Space plus a hosted MySQL service | Always on | Needs an external MySQL; the dataset and database must be uploaded; more setup |
| D. Kaggle notebook as a live host (not recommended) | A notebook with a tunnel | No cost | No incoming connections; sessions end after hours; the address changes each session |

**ngrok free-plan facts** (from ngrok's documentation): the free plan includes one assigned dev domain that does not change when the agent restarts, and free endpoints have no timeout. Browser visitors see a one-time interstitial page, which is bypassed by sending the header `ngrok-skip-browser-warning` (any value) or a non-standard user agent; Postman and the pages' `fetch` calls therefore send that header. Usage quotas apply, so check the Usage page of the ngrok dashboard before the evaluation.

**Recommendation:** follow Option A, the flow the mentor prescribed. Test the whole chain (MySQL, FastAPI, ngrok, Postman) by Day 8 with a simple build, prepare a second laptop or screenshots as a fallback, and rehearse the evaluation (11.7).

### 11.4 Release checklist

1. `requirements.txt` pinned; clean virtual environment installs and runs.
2. All artifacts present in `models/` and `data/processed/`.
3. `/health`, `/docs` and the dashboard open from the public URL.
4. Disclaimer visible; defaults match the report.
5. Smoke test of every page and every endpoint on the deployed URL.
6. The URL, the repository link and the README are in the submission.

### 11.5 Publishing on Kaggle

Kaggle notebooks run on cloud machines that cannot accept incoming connections, so Kaggle is used to **publish and reproduce** the work, while the live URL runs on Render or a Hugging Face Space.

| Item | Content |
| --- | --- |
| Kaggle Dataset | Processed files only (weekly demand, closed weeks, calendar, prior year, enrichment, product intelligence, basket pairs) with a data card crediting the UCI source |
| Kaggle Notebook | Public notebook that reproduces features, tuning, ablation, evaluation and quantile models from the dataset, top to bottom |
| Kaggle Model | The best model, the quantile models, `feature_list.json`, `config.json` and a model card |

| ID | Requirement | Priority |
| --- | --- | --- |
| FR-K-1 | The dataset, notebook and model MUST be published before the demonstration. | MUST |
| FR-K-2 | The notebook MUST run top to bottom on Kaggle without manual edits (paths set from one variable). | MUST |
| FR-K-3 | The three Kaggle links MUST be stored in configuration, returned by `/api/model-info`, and shown on the Home page footer. | MUST |
| FR-K-4 | The data card MUST credit UCI; the raw Excel file is linked, not republished. | MUST |
| FR-K-5 | An optional GPU experiment (FR-M-15) MUST report measured results only. | MAY |

### 11.6 Data persistence and backup

MySQL keeps the data on the demo laptop and survives restarts of FastAPI and of the laptop. It is not shared with other machines, so: (1) back it up with `mysqldump` before the evaluation and after any data change; (2) keep the dump and `scripts/load_mysql.py` so a second laptop can be ready in minutes; (3) `scripts/reset_demo.py` restores the seeded demo state of the system tables between faculty sessions. If the laptop sleeps or loses its network connection, the public URL stops working.

### 11.7 Evaluation readiness

The project will be evaluated by several faculty for more than two hours, so preparation is part of the deliverable.

| Item | Requirement |
| --- | --- |
| Run sheet | `docs/EVALUATION_PLAN.md`: a timed plan for a 2-hour-plus evaluation with a segment, a presenter and a fallback for each part |
| Pre-flight (30 minutes before) | Power connected, sleep disabled, MySQL running, FastAPI running, ngrok online, public URL opened from a phone, Postman collection green, tabs prepared, backup dump restored on the second laptop |
| Fallback | Second laptop with the same setup, screenshots, a screen recording and a PDF of the report |
| Data state | `scripts/reset_demo.py` before each session; the seeded stock and orders are documented |
| Roles | Each member demonstrates and answers for their own module; one member navigates the browser |
| Unexpected input | Every form validates input and shows a friendly message; no stack trace is ever shown |
| ngrok | Free plan; open the public address once and accept the interstitial; check the usage quota in the ngrok dashboard |
| Evidence | The Analysis board, the request log and the Postman results are used to answer how-does-it-work questions with real data |

### 11.8 Postman testing

| ID | Requirement | Priority |
| --- | --- | --- |
| FR-PM-1 | `postman/Shelfwatch.postman_collection.json` MUST cover every endpoint group (root aliases, health, products, forecast, predict, custom predict, risk, simulate, sensitivity, intelligence, inventory, orders, plans, analysis, explain, logs) with test scripts. | MUST |
| FR-PM-2 | Each request MUST assert the status code, key response fields and a response time under 3 seconds; write requests send `X-API-Key`. | MUST |
| FR-PM-3 | An environment file MUST define `baseUrl` (local and public) and `apiKey`; every request MUST send `ngrok-skip-browser-warning: true`. | MUST |
| FR-PM-4 | The first requests MUST mirror the mentor's example: `GET /health` and `POST /predict` with a sample JSON body. | MUST |
| FR-PM-5 | Screenshots of the Collection Runner results for the local and the public address MUST be included in the report. | MUST |

---

## 12. Project Organisation and Plan

### 12.1 Roles

| Member | Role | Primary deliverables |
| --- | --- | --- |
| M1 | Data, enrichment and intelligence | C1, C1b, C8 to C12, EDA, insights notebook, error analysis |
| M2 | ML modelling | Feature groups, tuning, ablation, quantile models, `predict_custom`, C2 to C5, model card |
| M3 | Risk engine, database and API | Risk engine, CRUD, middleware, endpoints, Python client, sensitivity analysis |
| M4 | Web app, deployment and Kaggle | Five pages, JS guide, tests, deployed URL, Kaggle publishing, report and slides assembly |

Every member must be able to explain the whole pipeline, because evaluators may question any member about any stage.

M4 owns the HTML, CSS and JavaScript dashboard and adapts the Shelfwatch prototype; M3 reviews `js/api.js` against the API schemas. If nobody on the team is comfortable with JavaScript, M3 and M4 pair on `js/api.js` and reuse the prototype's rendering code unchanged.

### 12.2 Timeline and checkpoints (14 days)

| Phase | Work | Cut line if late |
| --- | --- | --- |
| 1. Done | Data pipeline (D1 to D9), features and first models (M1 to M5) | n/a |
| 2. Enrichment | Calendar, prior-year and enrichment tables; feature groups G1 to G4 | Drop G4 first |
| 3. Modelling and backend in parallel | Tuning, ablation, ensemble, quantile models, final model; risk engine, database, CRUD, middleware | Drop the ensemble, then the pretrained-model experiment |
| 4. API and pages | Endpoints, Python client, five pages, intelligence tables | Drop bought-together links |
| 5. Delivery | Integration tests, Kaggle publishing, deployment | Keep: public URL and Kaggle links are mandatory |
| 6. Presentation | Report, slides, JS guide, viva and mentor demo rehearsal | Keep |

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
| Leakage through enrichment features | Medium | High | Enrichment tables hold week-w values and only features.py shifts them; the leakage test covers every group |
| Low prior-year coverage (products launched after 2010) | Medium | Medium | Report coverage; missing values plus a yoy\_available flag; the model must work without them |
| MySQL not running, misconfigured, or the laptop sleeping during the evaluation | High | Medium | Start script checks MySQL; /api/health shows database status; sleep disabled; backup dump; second laptop ready |
| Scope creep from many new features | High | High | Priority tiers in 14.1; cut stretch items first; every feature must be evaluated before it is kept |
| New features do not improve accuracy | Medium | Medium | Report the ablation honestly; the same features still serve as product intelligence |
| Unverified claims about uniqueness or about named companies | Medium | Medium | State only what the project measured; cite sources for anything else |
| ngrok interstitial page or request quota disrupts the demonstration | Medium | Medium | Send the skip header from the pages and Postman; open the URL once beforehand; check the usage page; keep the local address as a fallback |
| Faculty try unexpected inputs during a 2-hour-plus evaluation | High | Medium | Input validation everywhere, friendly errors, reset script, run sheet, full rehearsals |
| MySQL, plans and the analysis board add scope | High | High | Priority tiers in 14.1; cut the transaction drill-down and prediction history first |

---

## 14. Implementation Guide

### 14.1 Build order

| Step | Owner | Task | Tier | Done when |
| --- | --- | --- | --- | --- |
| 1 | All | Scaffold, mocks, workspace rules | Core | Done |
| 2 | M1 | Data pipeline and EDA (D1 to D9) | Core | Done: 5,067 rows, 100 products |
| 3 | M2 | Features and first models (M1 to M5) | Core | Done: baselines and defaults compared |
| 4 | M1 | E0 check of Online Retail II; calendar, prior-year and enrichment tables (E1 to E3) | Differentiator | C8 to C10 validated |
| 5 | M2 | Feature groups (M6), tuning with trials saved (M7), ablation (M8) | Core | `tuning_trials.csv` and `ablation.json` exist |
| 6 | M1 | Product intelligence and bought-together pairs (E4, E5) | Differentiator | C11 and C12 delivered |
| 7 | M2 | Diagnostics, quantile models, final model, explain module, `predict_custom`, result files (M9 to M13) | Core | C2 to C5 and `feature_importance.csv` |
| 8 | M3 | MySQL schema and loader (B6, B6b) | Core | All tables loaded; row counts match the files |
| 9 | M3 | Risk engine v3 with action label and policy backtest (B1 to B5b) | Core | T-R tests and backtest files |
| 10 | M3 | CRUD, plan workflow, middleware (B7, B8, B14, B15) | Core | T-DB, T-W, T-MW pass |
| 11 | M3 | Endpoints, analysis and explain, root aliases, client (B9 to B13, B16 to B18) | Core | T-A, T-AN, T-EX, T-RT pass |
| 12 | M4 | Shell, `api.js`, five pages including the Analysis board (F1 to F12) | Core | T-U pass |
| 13 | M4 | Start scripts, ngrok, Postman collection, Kaggle publishing (Q4 to Q8) | Core | Public URL and Postman run green |
| 14 | M1 | Error analysis (D8) | Core | FR-E satisfied |
| 15 | All | Evaluation plan, report, slides, two full rehearsals (Q9 to Q13) | Core | Rehearsals done |
| 16 | M2 | Pretrained time-series model on a Kaggle GPU (M14) | Stretch | Measured result reported |

### 14.2 Module interfaces

```python
# src/data_prep.py (built)
build_weekly_demand(raw_path, out_path, ...) -> pd.DataFrame
validate_c1(df) -> None

# src/enrichment.py
build_calendar_weekly(weeks) -> pd.DataFrame                              # C9
build_prior_year_weekly(ret2_path, cap, products) -> pd.DataFrame         # C10
build_enrichment_weekly(raw_path, products, weeks) -> pd.DataFrame        # C8

# src/intelligence.py
build_product_intelligence(weekly, raw_path, train_end) -> pd.DataFrame   # C11
build_basket_pairs(raw_path, products, train_end) -> pd.DataFrame         # C12

# src/features.py
FEATURE_GROUPS: dict[str, list[str]]        # base, g1, g2, g3, g4
build_features(weekly, closed_weeks, enrichment=None, groups=("base",)) -> pd.DataFrame
build_future_features(weekly, closed_weeks, target_week, enrichment=None, groups=("base",)) -> pd.DataFrame
build_custom_features(series, target_week) -> pd.DataFrame

# src/train.py
split_chronological(df, test_start="2011-10-10") -> tuple[pd.DataFrame, pd.DataFrame]
evaluate(y_true, y_pred) -> dict            # MAE, RMSE, WAPE, bias
tune(name, train, features, n_trials=30, seed=42) -> dict   # also writes tuning_trials.csv
run_ablation(train, groups_list) -> pd.DataFrame
fit_quantile_models(train, features) -> dict
predict_custom(payload) -> dict
run_all() -> None                           # writes C2 to C5

# src/explain.py
explain_prediction(bundle, weekly, enrichment, product_id, target_week) -> dict
global_importance(bundle, X_train) -> pd.DataFrame          # feature_importance.csv

# src/risk_engine.py
compute_risk(forecast_df, params=None, stock_overrides=None, open_orders=None, lifecycle=None) -> pd.DataFrame
sensitivity_table(forecast_df, lead_times=(1, 2, 3, 4), service_levels=(0.90, 0.95, 0.99)) -> pd.DataFrame
summarize(risk_df) -> dict

# src/backtest.py
run_policy_backtest(predictions, weekly, forecast_meta, intelligence=None, lead_times=(1, 2)) -> tuple[pd.DataFrame, pd.DataFrame]

# scripts/load_mysql.py
load_all(engine, processed_dir, models_dir) -> dict[str, int]   # row count per table
```

### 14.3 API skeleton

```python
import os, time, uuid
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy import text
from api.db import engine, get_db, write_log      # engine = create_engine(os.environ["DATABASE_URL"])
from api.schemas import RiskRequest, PredictRequest, OrderCreate, OrderPatch, StockPut
from src.risk_engine import compute_risk

load_dotenv()
STATE: dict = {}
API_KEY = os.getenv("API_KEY", "demo-key")

@asynccontextmanager
async def lifespan(app: FastAPI):
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))             # fail fast if MySQL is not running
    STATE["models"] = ...                          # joblib files from disk, paths from model_registry
    yield

app = FastAPI(title="Shelfwatch API", lifespan=lifespan)
origins = [o for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o]
app.add_middleware(CORSMiddleware, allow_origins=origins,
                   allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE"], allow_headers=["*"])

@app.middleware("http")
async def request_id_timing_audit(request: Request, call_next):
    rid, t0 = str(uuid.uuid4()), time.perf_counter()
    response = await call_next(request)
    ms = (time.perf_counter() - t0) * 1000
    response.headers["X-Request-ID"] = rid
    response.headers["X-Process-Time-Ms"] = f"{ms:.1f}"
    if request.url.path.startswith("/api") and request.url.path != "/api/logs":
        write_log(rid, request.method, request.url.path, response.status_code, ms)
    return response

def require_key(x_api_key: str = Header(default="")):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

api = APIRouter(prefix="/api")

@api.get("/health")
def health(db=Depends(get_db)):
    ...   # database status and row counts of key tables

@api.post("/predict")
def predict(req: PredictRequest, db=Depends(get_db)):
    ...   # features -> saved model -> compute_risk; save the call to prediction_log

@api.put("/inventory/{product_id}", dependencies=[Depends(require_key)])
def put_inventory(product_id: str, body: StockPut, db=Depends(get_db)):
    ...   # upsert, clear the risk cache, return 200 or 201

# /risk, /simulate, /orders, /plans, /analysis/*, /explain/*, /intelligence, /logs ... follow the same pattern.

app.include_router(api)

# Root aliases that mirror the mentor's example; same behaviour as the /api versions.
@app.get("/health")
def health_root(db=Depends(get_db)):
    return health(db)

@app.post("/predict")
def predict_root(req: PredictRequest, db=Depends(get_db)):
    return predict(req, db)

# Must be the LAST registration: API routes keep priority over static files.
app.mount("/", StaticFiles(directory="dashboard", html=True), name="dashboard")
```

### 14.4 Run and test

```
pip install -r requirements.txt
copy .env.example .env            (then set DATABASE_URL and API_KEY)
python -m src.data_prep --raw "data/raw/Online Retail.xlsx"
python -m src.enrichment
python -m src.intelligence
python -m src.train
python -m src.backtest
python scripts/load_mysql.py
pytest tests/ -v
python -m uvicorn api.main:app --reload --port 8000
ngrok http 8000                   (second window)
python scripts/api_client.py demo
# site: http://localhost:8000/   docs: http://localhost:8000/docs   health: http://localhost:8000/health
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

| Requirement source | SRS requirements | Verified by |
| --- | --- | --- |
| Demand model for a defined product and horizon | FR-D-1 to 10, FR-F-1 to 16, FR-M-1 to 15 | T-F-1 to T-F-4, T-M-1 to T-M-5 |
| Product-level estimates with ranges | FR-M-10, FR-M-13, FR-A-11 | T-A-1, T-P-1 |
| Forecast-error analysis | FR-E-1 to 6 | Notebook review, T-M-2 |
| Configurable stockout indicator | FR-R-1 to 13, section 6.2 | T-R-1 to T-R-6 |
| Inventory dashboard | FR-U-1 to 20 | T-U-1 to T-U-5 |
| Stated simulated-inventory assumptions | Section 6.1, FR-R-7, FR-R-8, FR-U-3 | T-U-1 |
| Compare regression approaches, evaluate on later periods | FR-M-1 to 7, C-1 | T-M-1 |

## Appendix C. Final Acceptance Checklist

- [ ] Pipeline reproduces `weekly_demand` from the raw file with one command
- [ ] Enrichment tables built; prior-year coverage reported; the pipeline runs without Online Retail II
- [ ] Leakage test passes for every feature group; the split is chronological
- [ ] Metrics table shows baselines and all models on validation and test; ablation and tuning trials saved
- [ ] Quantile coverage reported; best model, feature list, config and explanation importance saved and reloadable
- [ ] Error analysis covers segments, weeks, best and worst products, over versus under-forecast
- [ ] Product intelligence and bought-together files produced from training data only
- [ ] MySQL schema created and every reference table loaded; row counts match the files; no customer identifiers stored
- [ ] Risk engine matches Appendix A, the advice formulas and the action rule; policy backtest produced
- [ ] CRUD, status transitions, plan workflow, middleware headers, API key and audit log tested
- [ ] API implements every endpoint under `/api` plus the root `/health` and `/predict`; Python client demo runs
- [ ] Five pages work with `fetch` and event handling; no login page; Analysis board has all eight sections with source captions
- [ ] Local server and ngrok public URL work; Postman collection is green on both addresses with screenshots saved
- [ ] Kaggle dataset, notebook and model published and linked
- [ ] README, report (persona, routine, measured benefit), slides, JS guide, evaluation plan and two full rehearsals done

## Appendix D. Open Issues for Team Decision

| # | Issue | Proposal |
| --- | --- | --- |
| 1 | Online Retail II sheet names and product overlap are unverified | Run the read-only check E0 first; if overlap is low, report it and keep group G3 optional |
| 2 | Kaggle: public pages or a live host? | Public dataset, notebook and model on Kaggle; live URL elsewhere; confirm with the mentor |
| 3 | Fine-tuning: hyperparameter tuning or a pretrained model? | Do tuning plus the optional pretrained-model experiment; confirm with the mentor |
| 4 | Contract changes: C4 adds p10, p90 and last average price; C7 moves under `/api` with new endpoints; C8 to C13 are new | Apply `CONTRACTS_ADDENDUM.md`; the team has agreed to follow these recommendations |
| 5 | Default starting stock of 3 weeks flags few HIGH items (prototype: 1 HIGH, 7 MEDIUM, 22 LOW of 30) | Keep as the stated default; show the sensitivity table; do not tune to a target |
| 6 | One shelf wall with 100 bins | Wrapping grid with a minimum bin width of 44 px |
| 7 | The API key in the page configuration is visible | Accept for the demo, rate-limit, and state the limitation |
| 8 | `guest_share_4w` is a hypothesis | Keep only if validation supports it and label it as a hypothesis |
| 9 | The manager persona is assumed, not interviewed | State it as an assumption; review it with the mentor and, if possible, one retail practitioner |
| 10 | MySQL must be installed and configured on every member's machine | Follow docs/DEPLOY\_LOCAL\_NGROK.md; tests may fall back to in-memory SQLite so work is not blocked |
| 11 | Store all 527,734 cleaned lines or only the top-100 products' lines | Only the top-100 lines (about 77,000) to keep the database small and the board fast |
| 12 | Explanations when the final model uses a scale-aware target or an ensemble | Show contributions on the underlying LightGBM model and say in the caption how the scaling or blending is applied |
| 13 | ngrok free-plan quota and interstitial during a long evaluation | Check the usage page beforehand, send the skip header, keep the local address and a second laptop as fallbacks |

## Appendix E. Mentor Requirements Traceability

| Mentor input | Where addressed | Evidence in the project |
| --- | --- | --- |
| Give the URL of the hosted model | 11.3 Option A, FR-A-9 | Public URL, `/api/health`, `/docs` |
| Predict output | 8.2 Predict page, FR-U-15, `/api/predict` | Forecast, range, risk and advice shown to the user |
| Stakeholders | 1.6 | Persona, decision record and workflow table in 1.6; docs/STAKEHOLDER.md; report section |
| Where it is used in real life | 1.6 | Report section on the manager's weekly routine with measured results |
| Fine-tuning | FR-M-4, FR-M-11 to FR-M-15, ablation | Tuned parameters, ablation table |
| Host on Kaggle | 11.5 | Three Kaggle links |
| JavaScript UI, fetch API, event handling, know JS | 8.4, 8.5, FR-U-15 to 17 | Five pages, `docs/JS_GUIDE.md` |
| Not a single-page app, add features | FR-U-2, 8.2 | Five HTML pages |
| CRUD and database | 7.3 | Inventory and order tables |
| Python API call | 7.5 | `scripts/api_client.py` |
| Middleware | 7.4 | Headers, audit log, API key, rate limit |
| New derived features and other data | 4.6, 5.1 (G1 to G4), 5.4 | Enrichment tables, product intelligence |
| How training and testing are split | 4.5, FR-M-4, C-1 | Chronological split with week-based validation folds |
| No login page | 8.1, FR-U-26, NFR-17 | The base URL opens Home directly |
| Everything built around one stakeholder | 1.6, FR-U-21, FR-U-22 | Persona, weekly routine, Action list; docs/STAKEHOLDER.md |
| Distinct and new features | 4.6, 5.1, 5.4, 6.2 items 8 to 10, 7.6, 8.6, Appendix F | Feature groups, product intelligence, Action list, plan workflow, policy backtest, Analysis board |
| A working system, not just input and output | 7.3, 7.6, FR-U-27 | MySQL, weekly plans, purchase orders, stock that updates on receiving |
| Analysis board: computation, data used, how the output is reached | 7.7, 8.6, FR-U-23, FR-U-24, FR-U-28 | Analysis board with eight sections and the Explain panel |
| Data stored in MySQL | 4.7, 7.3 | sql/schema.sql, scripts/load\_mysql.py |
| Evaluation by several faculty for more than 2 hours | 11.7, NFR-15 | docs/EVALUATION\_PLAN.md, rehearsals, fallback laptop |
| Local FastAPI, public URL, Postman test | 11.1 to 11.3, 11.8, FR-A-13 | ngrok address, Postman collection and screenshots |
| Use two datasets | 4.1, 4.6, A-6, FR-AN-5 | Online Retail (primary) and Online Retail II (prior year), shown separately on the Analysis board |

## Appendix F. ML Workflow Coverage and Differentiators

### F.1 The 16-step workflow and the mentor's four steps

| Step | Where covered | Tasks and notes |
| --- | --- | --- |
| 1. Problem definition | Sections 1 and 1.6 | One stakeholder, measurable objectives |
| 2. Data collection | 4.1, 4.6 | Two UCI datasets and the holidays package (D1, E0) |
| 3. Preprocessing | 4.3 | D2 to D4, E1 to E3 |
| 4. Exploratory data analysis | Notebooks 01 and 05 | D6, E6 |
| 5. Feature engineering | 5.1 | M1, M2, M6 |
| 6. Model selection | 5.2 | M4, M5, M7 (baselines and four model families compared) |
| 7. Model training | 5.2 | M5, M7 |
| 8. Model evaluation | 5.2, 5.3, FR-R-9 | M5b, M9, D8, B5b |
| 9. Hyperparameter tuning | FR-M-4, FR-M-16 | M7: random search with week-based time-series cross-validation; every trial saved and shown on the Analysis board |
| 10. Model saving | FR-M-9 | M11: joblib files, feature list, config, registry row in MySQL |
| 11. Backend (FastAPI) | Section 7 | B9 to B18 |
| 12. Frontend | Section 8 | F1 to F12 |
| 13. Integration | 8, 14.1 | F11, B17 |
| 14. Testing | Sections 10 and 11.8 | B17, Q4, Postman collection |
| 15. Deployment (local server) | 11.1 to 11.3 | Q5, Q6 |
| 16. URL generation (ngrok) | 11.3 | Q6, with the skip-warning header |

The mentor's four steps (train, serve locally with FastAPI, create a public URL, test with Postman) map to steps 7 and 10, 11 and 15, 16, and 14. Items beyond the original flow that are now required: MySQL storage, the Analysis board, no login page, and a single stakeholder.

### F.2 What goes beyond the minimum problem statement

These are features beyond the problem statement. The report must not claim that other groups lack them; it states what this project does and measures.

| Feature | Where | Evidence |
| --- | --- | --- |
| Two datasets, with prior-year seasonality | 4.6, FR-F-12 | Coverage figure, ablation, Christmas 2009 versus 2010 chart |
| Derived behaviour and transaction-based features, each tested by ablation | 5.1 | Ablation table, including groups that did not help |
| Prediction intervals with measured coverage | FR-M-13 | Coverage on validation and test |
| Product intelligence (pattern, ABC and XYZ, lifecycle, returns, bought-together) | 5.4 | Product intelligence page |
| Action list, replenishment advice, stockout probability, revenue at risk | 6.2 | Home page and Predict page |
| Business backtest of ordering policies | FR-R-9 | Fill rate, stockout weeks, average stock, weekly replay |
| Working replenishment workflow stored in MySQL | 7.3, 7.6 | Plans, orders, stock updates, audit log |
| Analysis board and per-forecast explanation | 7.7, 8.6 | Eight sections with source captions |
| Public URL and a Postman collection that runs on both addresses | 11.3, 11.8 | Collection results and screenshots |
