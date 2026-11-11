# PROJECT_CONTEXT.md

> Paste this file (plus CONTRACTS.md) at the start of every Claude chat.
> If a decision here changes, the whole team agrees first, then this file is updated.

## 1. Project title
Retail Product Demand Estimation and Stockout Risk Analysis

## 2. Problem statement
Estimate product demand from historical sales records and identify products that may need inventory attention. Use dates, product identifiers and quantities to build time-based demand features. Compare regression approaches, evaluate on later periods, and create a stockout-risk indicator based on an explicit inventory rule.

**Expected deliverables**
1. A demand model for a defined product set and time horizon
2. Product-level demand estimates
3. Forecast-error analysis
4. A configurable stockout-risk indicator
5. An inventory dashboard
6. Clearly stated simulated-inventory assumptions (the dataset has no inventory-on-hand data)

## 3. Dataset
- UCI Online Retail: https://archive.ics.uci.edu/dataset/352/online+retail
- About 541,909 rows, 1 Dec 2010 to 9 Dec 2011, UK online gift retailer
- Columns: `InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country`

## 4. FROZEN design decisions (do not change without team agreement)

| Item | Decision |
|---|---|
| Granularity | Product x week |
| Week definition | Monday-start. `week_start` is the Monday date |
| Target | `units` = total units sold of a product in that week |
| Forecast horizon | **Next week (t+1)** only. 4-week horizon is an optional extra |
| Product scope | Top 100 products by total units in the **training period**, with sales in at least 30 of the training weeks |
| Valid weeks | Only full weeks. Drop partial first week (starting 2010-11-29) and partial last weeks (starting 2011-12-05). Full weeks are 2010-12-06 to 2011-11-28 (52 weeks) |
| Train / test split | Chronological. **Test = last 8 full weeks** (week_start 2011-10-10 to 2011-11-28). Train = everything before |
| Tuning | `TimeSeriesSplit` inside the training period only. Never touch the test set for tuning |
| Random seed | `42` everywhere |
| Python | 3.10 or 3.11. Libraries pinned in `requirements.txt` |

## 5. Data cleaning rules (Member 1 implements, everyone must know them)
1. Remove cancellations: `InvoiceNo` starting with `C`
2. Remove rows with `Quantity <= 0` or `UnitPrice <= 0`
3. Remove non-product `StockCode` values: `POST, D, M, DOT, BANK CHARGES, AMAZONFEE, S, B, CRUK, PADS` and any code with no digits
4. Keep rows with missing `CustomerID` (not needed for product demand)
5. Outliers: cap line-level `Quantity` at the 99.9th percentile. Document the cap value in the report
6. Fill missing weeks with `units = 0` between each product's first and last active week
7. Drop partial weeks (see section 4)
8. **Shop-closed weeks:** the week of 2010-12-27 has zero sales for every product (Christmas/New Year shutdown). It stays in the table as `units = 0` to keep the weekly grid regular, but it is listed in `data/processed/closed_weeks.csv`. Member 2 must treat it as **missing, not zero demand**: no training row with it as the target, and lags/rolling stats must skip it (or mask it)

## 6. Feature rules (Member 2 implements)
- Target at week t uses only information from weeks **before t**
- All lag and rolling features use `shift(1)` **before** `rolling()`
- Planned features: lags 1, 2, 3, 4; rolling mean and std over 4 and 8 weeks; week of year, month, weeks to Christmas; previous-week average price, orders, customers; product average demand and volatility computed on past data only
- **Leakage check is mandatory**: a test that confirms no feature uses data from week t or later

## 7. Models to compare
1. Baseline A: last week's value (naive)
2. Baseline B: 4-week moving average
3. Ridge regression
4. Random Forest
5. LightGBM or XGBoost
6. Optional: Poisson regression

**Metrics:** MAE, RMSE, WAPE, bias (mean of predicted minus actual). Do not use MAPE (zero-demand weeks).
A model is only "good" if it beats both baselines on the test period.

## 8. Simulated inventory assumptions (state these in the dashboard and the report)
The dataset has **no inventory-on-hand records**, so stock is simulated:

- `on_hand = initial_stock_weeks x (average weekly units in training period) x noise`
  - `noise` is drawn once per product from Uniform(0.5, 1.5) with a fixed seed, so results are reproducible
  - Default `initial_stock_weeks = 3`
- `lead_time_weeks`: default 1, adjustable 1 to 4
- `service_level`: default 0.95, adjustable 0.80 to 0.99. `z = norm.ppf(service_level)`
- `sigma_error`: standard deviation of the model's weekly forecast error per product, measured on validation or test predictions
- `forecast_lead_time = forecast_weekly x lead_time_weeks`
- `safety_stock = z x sigma_error x sqrt(lead_time_weeks)`
- `reorder_point = forecast_lead_time + safety_stock`
- `days_of_cover = on_hand / (forecast_weekly / 7)` (if forecast is 0, cover is capped at 999)

**Risk rule**
- HIGH: `on_hand < forecast_lead_time`
- MEDIUM: `on_hand < reorder_point` (and not HIGH)
- LOW: otherwise

**Disclaimer text (must be visible on the dashboard):**
> Inventory levels are simulated. The dataset contains sales only, not stock-on-hand. Risk flags show how the rule behaves under the stated assumptions, not real stock positions.

## 9. Known limitations (state honestly in the report)
- Only about one year of history, so yearly seasonality cannot be learned. Calendar features such as week-of-year or weeks-to-Christmas act almost like a time index with a single year, so they must be tested by ablation (with and without) before being trusted
- Demand drifts upward in Oct/Nov, so the test period sits at a higher level than most of training. Tree-based models cannot predict above the range they saw in training, so check for under-forecasting at the test peak
- One shop-wide closure week (2010-12-27) in the history
- Intermittent demand for many products
- Simulated inventory, so risk outputs are illustrative
- Wholesale customers cause bulk orders that are hard to predict

## 10. Repository structure
```
project/
├── PROJECT_CONTEXT.md
├── CONTRACTS.md
├── MEMBER_PROMPTS.md
├── data/raw/            (not committed)
├── data/processed/
├── data/mock/           (fake files for parallel work)
├── notebooks/           01_eda, 02_modeling, 03_error_analysis
├── src/                 data_prep.py, features.py, train.py, risk_engine.py
├── models/              best_model.joblib, feature_list.json, config.json
├── api/main.py
├── dashboard/           index.html, config.js, css/, js/ (app, api, charts)
├── tests/
├── requirements.txt
└── README.md
```

## 11. Ownership
| Member | Role | Owns |
|---|---|---|
| M1 | Data and Analysis | `src/data_prep.py`, EDA notebook, error-analysis notebook, `weekly_demand.parquet` |
| M2 | ML Modeling | `src/features.py`, `src/train.py`, model files, `predictions.csv`, `metrics.csv` |
| M3 | Risk Engine and Backend | `src/risk_engine.py`, `api/main.py`, sensitivity analysis |
| M4 | Dashboard, QA, Deployment | `dashboard/`, `tests/`, deployment, report and slides assembly |

## 12. Working rules
- One Git branch per member. Merge to `main` through a pull request reviewed by a teammate
- Never edit another member's owned file without telling them
- Never change a schema in CONTRACTS.md alone
- Every module has docstrings and at least a few pytest tests
- Every member must be able to explain the whole pipeline in the viva
