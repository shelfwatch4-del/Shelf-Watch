# PROJECT_CONTEXT.md (version 3.2)

> Paste this file (plus CONTRACTS.md and CONTRACTS_ADDENDUM.md) at the start of every AI chat.
> The source of truth is the SRS (version 3.2, the Claude document; export it to docs/SRS.md). If a decision here changes, the team agrees first.

## 1. Project
**Shelfwatch: Retail Product Demand Estimation and Stockout Risk Analysis.**
Estimate weekly product demand from historical sales, identify products that need inventory attention, and give an explicit, configurable stockout-risk indicator with replenishment advice. Dataset: UCI Online Retail (541,909 rows, 2010-12-01 to 2011-12-09).

## 2. Primary stakeholder (decided)
**Retail Inventory and Replenishment Manager.** One user. Every page, endpoint and report section serves their weekly routine: see what needs attention, understand one product, decide the order, act, test assumptions, set policy, trust the numbers. Other roles are not designed for. The persona is assumed, not interviewed. Details: `docs/STAKEHOLDER.md`, SRS 1.6.

## 3. Frozen design decisions
| Item | Decision |
|---|---|
| Granularity | Product x Monday-start week; target = units sold per product per week |
| Horizon | Next week (t+1); 4-week horizon is an optional extra |
| Products | Top 100 by training-period units, sales in at least 30 training weeks |
| Valid weeks | Full weeks 2010-12-06 to 2011-11-28 (52). The shop-closure week 2010-12-27 has **no row** (it is missing data, not zero demand) and is listed in `closed_weeks.csv` |
| Zero-fill | Each product is zero-filled from its first sale **through 2011-11-28**; never before its first sale. C1 = 5,067 rows, 100 products, 51 weeks with rows |
| Split | Chronological. Test = last 8 full weeks (2011-10-10 to 2011-11-28, 800 rows). Training target weeks 2011-02-07 to 2011-10-03 (35 weeks, 3,467 rows) after 8 history-only weeks per product |
| Validation | Week-based expanding-window TimeSeriesSplit inside the training period |
| Forecast target week | 2011-12-05 |
| Seed | 42 everywhere |
| Python | 3.10 or 3.11, pinned versions |

## 4. Data rules
1. Remove cancellations (InvoiceNo starts with C) from demand; keep them only to compute returns.
2. Remove Quantity <= 0, UnitPrice <= 0, non-product StockCodes (POST, D, M, DOT, BANK CHARGES, AMAZONFEE, S, B, CRUK, PADS, C2, no-digit codes, GIFT and TEST prefixes).
3. Keep rows with missing CustomerID.
4. Cap line quantity at the 99.9th percentile of TRAINING rows (480 units; 415 lines capped).
5. Top-100 selection and every learned statistic use training weeks only.
6. Enrichment tables hold the value FOR week w; only `src/features.py` applies the one-week shift.

## 5. Enrichment and features
- **Base (15):** lag_1..4, roll_mean_4/8, roll_std_4/8, week_of_year, month, prev_avg_price, prev_orders, prev_customers, prod_past_mean, prod_past_std.
- **G1 derived behaviour:** order size, trend ratio, price index, volatility (cv_4), zero streak.
- **G2 calendar (holidays package, UK):** bank-holiday count and expected trading days of the target week.
- **G3 prior year (Online Retail II, 2009-2010 sheet):** yoy_mean_5w, yoy_seasonal_index, yoy_available. Coverage is checked first (task E0).
- **G4 transaction-derived (shifted):** return rate, export share, guest share (a hypothesis).
- Groups are switchable and kept or dropped on VALIDATION WAPE only. Weather and national retail-index data are deliberately not used (one year cannot separate them from a time trend).
- Static product attributes over the whole training period (for example customer concentration) are NOT model features.

## 6. Models and metrics
Baselines: naive, 4-week moving average. Models: Ridge, Random Forest, LightGBM, each with raw and scale-aware targets; ensemble; quantile models (p10, p90). Metrics: MAE, RMSE, WAPE, bias. Never MAPE.
M5 result (test, defaults): moving average WAPE 0.4801 (bias -5.94), Ridge 0.4913 (bias -0.18), LightGBM 0.5315 (bias +26.23). The baseline wins; LightGBM over-forecasts. Report honestly even if the best model only slightly beats the baseline.

## 7. Risk and replenishment rules (summary; formulas in SRS 6.2)
- Stock = user-entered record if present, else simulated: `initial_stock_weeks x avg weekly units x noise` (noise Uniform 0.5 to 1.5, seed 42). Defaults: lead time 1 week, service level 0.95, starting stock 3 weeks.
- `forecast_lead_time = forecast_weekly x L`; `safety_stock = z x sigma_error x sqrt(L)`; `reorder_point = forecast_lead_time + safety_stock`.
- HIGH if on_hand < forecast_lead_time; MEDIUM if on_hand < reorder_point; else LOW.
- Advice: recommended order quantity, days to reorder, revenue at risk, stockout probability, and an **action** label: ORDER_NOW, ORDER_SOON, REVIEW_STOP (inactive products), HOLD.
- Success measure for the manager: the replenishment-policy backtest (fill rate, stockout weeks, average on-hand), FR-R-9.

**Disclaimer (visible on every page):**
> Inventory levels are simulated. The dataset contains sales only, not stock-on-hand. Risk flags show how the rule behaves under the stated assumptions, not real stock positions.

## 8. System
- **API:** FastAPI, all JSON endpoints under `/api`; **MySQL 8** via SQLAlchemy and PyMySQL (all project data plus inventory levels, plans, purchase orders, prediction log, request log; `sql/schema.sql`); CRUD with status codes 201, 204, 404, 409, 422; middleware: CORS, request id and timing, audit log, API key on writes, rate limit, error handler; `scripts/api_client.py` (httpx).
- **Web:** five separate HTML pages (not a single-page app): Home, Predict, Product intelligence, Inventory manager, Analysis board. Plain HTML, CSS, JavaScript with `fetch` and event handling; served by FastAPI from the same app.
- **Deployment (mentor's flow):** train, serve locally with FastAPI, public URL with **ngrok**, test with **Postman** (collection in `postman/`). The public URL works only while the laptop, MySQL, FastAPI and ngrok run; keep a second laptop and a `mysqldump` backup. Guide: `docs/DEPLOY_LOCAL_NGROK.md`. Kaggle is for publishing the dataset, notebook and model, not the live URL.
- **No login page.** The site opens on Home. Writes use a server-side API key.
- **Analysis board** (`analysis.html`): data sources and lineage, split, features, tuning, evaluation, business backtest, explain a forecast, system log; every chart names its MySQL source.
- **Working system:** weekly plan workflow (generate, review, approve, export, receive) stored in MySQL.
- **Datasets:** Online Retail (primary) and Online Retail II (prior-year supporting) plus UK holidays.
- **Evaluation:** several faculty for 2+ hours; see `docs/EVALUATION_PLAN.md`.

## 9. Repository layout and ownership
See SRS 3.5. M1 data, enrichment, intelligence. M2 features and models. M3 risk engine, database, API, client. M4 web pages, deployment, Kaggle, report assembly.

## 10. Known limitations (state honestly)
One year of main data; intermittent and discontinued products (21829, 22740); bulk wholesale orders; simulated stock unless entered; persona assumed; the public URL depends on the presenter's laptop staying on; prior-year coverage may be partial; weekly one-step forecast reused across the lead time.

## 11. Working rules
One branch per member; pull requests reviewed by a teammate; one task per AI conversation in Planning mode; contract changes only through `CHANGE_REQUESTS.md` and the contracts files; never edit `docs/SRS.md` by hand (export it); every feature is evaluated before it is kept; print numbers from files, never from memory.
