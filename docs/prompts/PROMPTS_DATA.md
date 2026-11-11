# PROMPTS_DATA.md: Data and Analysis (Member 1)

**You own:** `src/data_prep.py`, `src/error_analysis.py`, `notebooks/01_eda.py`, `notebooks/03_error_analysis.py`, `data/processed/weekly_demand.parquet`, `closed_weeks.csv`, `cleaning_log.csv`, `product_segments.csv`, `error_analysis.csv`.
**Hands off to:** Member 2 (ML), Member 3 (API), Member 4 (dashboard).
**Setup first:** complete Phase 0 in `PROMPTS.md` (workspace rules and workflows).

**How to use:** open a **new Antigravity conversation per task**, use **Planning mode**, read the Implementation Plan artifact before approving, and finish with the closing check below. Commit after each task.

**Closing check (add to the end of any prompt that lacks it):**
```
Before you finish: run the tests for this task, show the exact commands and output, list every file you created or changed, and list anything in the SRS sections I named that you did NOT implement. Do not claim success unless tests ran and passed.
```

> **Shortcut:** if you already have `phase1_member1.zip` (a working pipeline), copy it in and use **Task D0** instead of D1 to D5.

---

## Task D0: Verify an existing pipeline (shortcut)

**Mode:** Planning | **Output:** an audit list

```
src/data_prep.py and tests/test_data_prep.py already exist. Read docs/SRS.md sections 4.3 to 4.5 and PROJECT_CONTEXT.md section 5. Audit the code against each requirement FR-D-1 to FR-D-10 and list every gap in a table (requirement, status, evidence). Then run: python -m src.data_prep --raw "data/raw/Online Retail.xlsx" and confirm these expected results: 5,147 rows, 100 products, 52 weeks from 2010-12-06 to 2011-11-28, closed week 2010-12-27 in closed_weeks.csv, quantity cap about 480 (415 lines capped). Run pytest and report. Fix only real gaps and explain each fix. If a number differs from expected, report why instead of forcing a match.
```
**Done when:** the audit table has no open gaps and the numbers match.

---

## Task D1: Load and cache the raw data

**SRS:** FR-D-1 | **Output:** `load_raw()` in `src/data_prep.py`

```
Read docs/SRS.md section 4.1 (raw schema) and FR-D-1.

Implement load_raw(raw_path) -> pd.DataFrame in src/data_prep.py:
- Accept .xlsx, .csv (encoding latin-1, parse InvoiceDate) and .parquet.
- For Excel: read once with openpyxl, then write a parquet cache next to the file (same name, .parquet). On later calls read the parquet instead.
- Before saving the cache, cast InvoiceNo and StockCode to str and Description to pandas "string" dtype. The real file has numeric Description cells and mixed-type columns that crash parquet writing.
- Use pathlib; no hard-coded absolute paths; must work on Windows, macOS and Linux.

Tests (tests/test_data_prep.py): a tiny DataFrame with a numeric Description round-trips through parquet without error; the cache is created on first call and reused on the second (check by monkeypatching read_excel to raise on the second call).
```
**Done when:** `pytest tests/test_data_prep.py -k load -q` passes and `data/raw/Online Retail.parquet` appears after the first real load.

---

## Task D2: Row-level cleaning with a log

**SRS:** FR-D-2, FR-D-3, FR-D-9 | **Output:** `clean_transactions()`

```
Read docs/SRS.md section 4.3 and PROJECT_CONTEXT.md section 5.

Implement clean_transactions(raw) -> tuple[pd.DataFrame, pd.DataFrame] returning the cleaned transactions and a log DataFrame with columns step and rows_remaining.

Steps, in this order, each logged with how many rows it removed:
1. raw rows (starting count)
2. drop cancellations: InvoiceNo (after strip and upper) starts with "C"
3. drop Quantity <= 0
4. drop UnitPrice <= 0
5. drop non-product StockCodes: the set {POST, D, M, DOT, BANK CHARGES, AMAZONFEE, S, B, CRUK, PADS, C2}, any code with no digit, and any code starting with GIFT or TEST (compare after strip and upper)

Rules: keep rows with missing CustomerID; strip and normalise StockCode to upper case; do not mutate the input DataFrame.

Expected on the real file: 541,909 -> 532,621 -> 531,285 -> 530,104 -> 527,734 rows. Report any difference.

Tests: a synthetic DataFrame with one row of each junk type plus normal rows; assert only the normal rows survive, the log is monotonically decreasing, and the input is unchanged.
```
**Done when:** the log numbers match the expected counts.

---

## Task D3: Weeks, closed weeks and outlier cap

**SRS:** FR-D-4, FR-D-5, FR-D-8 | **Output:** week helpers

```
Read docs/SRS.md sections 4.3 and 4.5, and PROJECT_CONTEXT.md section 4.

Implement in src/data_prep.py:
- week_start_of(dates: pd.Series) -> pd.Series: the Monday 00:00 of each timestamp's week.
- full_week_range(dates: pd.Series) -> pd.DatetimeIndex: Mondays of all FULL weeks covered by the data (first Monday on or after the data start, up to the last Monday whose Sunday is on or before the data end). Computed from the data, not hard-coded.
- find_closed_weeks(df, weeks) -> pd.DatetimeIndex: full weeks in which the whole shop recorded no transactions. These are closures, NOT zero demand.
- cap_quantity(df, train_mask, quantile=0.999) -> (df, cap_value, n_capped): cap line-level Quantity at the quantile computed on TRAINING rows only.

Tests: week_start_of maps Wed 2011-03-02 and Sun 2011-03-06 to Mon 2011-02-28 and Mon 2011-03-07 to itself; on UCI-shaped dates (2010-12-01 to 2011-12-09) full_week_range gives 2010-12-06 .. 2011-11-28 with 52 weeks; closed-week detection on a small example; the cap value does not change when test-period rows are inflated.
```
**Done when:** the tests pass; on real data the cap is about 480 and the closed week is `2010-12-27`.

---

## Task D4: Build the weekly demand table (contract C1)

**SRS:** FR-D-6, FR-D-7, FR-D-9, FR-D-10 | **Output:** `weekly_demand.parquet`

```
Read docs/SRS.md sections 4.3 to 4.5 and CONTRACTS.md C1 and C1b.

Implement build_weekly_demand(raw_path, out_path="data/processed/weekly_demand.parquet", log_path="data/processed/cleaning_log.csv", closed_weeks_path="data/processed/closed_weeks.csv", top_n=100, test_weeks=8, min_active_train_weeks=30) using the helpers from D1 to D3.

Pipeline: load -> clean -> assign week_start -> keep only full weeks (log how many rows are dropped) -> detect closed weeks and save them -> define train/test boundary (test = last test_weeks full weeks, expected to start 2011-10-10) -> cap Quantity using TRAIN rows only -> select the Top-N products by total units in the TRAINING period only, requiring sales in at least 30 training weeks -> aggregate to product x week -> fill missing weeks between each product's first and last active week.

Aggregation: units = sum(Quantity), orders = nunique(InvoiceNo), customers = nunique(CustomerID) (NaN ignored), avg_price = mean(UnitPrice). In filled weeks: units, orders, customers = 0; avg_price forward-filled. description = most frequent description per product.

Output columns, in order: product_id, description, week_start, units, orders, customers, avg_price. Types: units, orders, customers int; week_start datetime (Mondays); sorted by product_id, week_start.

Also implement validate_c1(df) raising AssertionError for any contract violation (column order, Mondays only, units >= 0, no duplicate product-week rows, int types, avg_price > 0, no gaps in each product's range, sorted) and call it before saving. Add a CLI: python -m src.data_prep --raw "data/raw/Online Retail.xlsx". Print the cleaning log and a final summary line.

Expected on the real file: 5,147 rows, 100 products, 2010-12-06 to 2011-11-28, train 44 weeks, closed week 2010-12-27.
```
**Done when:** the real run prints the expected summary and `validate_c1` passes.

---

## Task D5: Pipeline tests with synthetic data

**SRS:** section 10 (T-D-1 to T-D-4)

```
Create tests/synthetic.py with make_synthetic_raw(n_products=150, seed=42) that mimics the UCI file: dates 2010-12-01 to 2011-12-09, no Saturday trading, the first 100 products sell a lot and the rest rarely, about 25% missing CustomerID. Inject known junk: a cancellation (C999001), POST, DOT, a zero-price row, an 80,995-unit order on 2011-04-04 and a gift voucher code gift_0001_20.

Then complete tests/test_data_prep.py with these tests (run against the synthetic raw saved to a temp parquet): schema equals C1; exactly 100 products; Mondays only and no duplicates; no partial weeks (min week >= 2010-12-06, max <= 2011-11-28); no negative units; no gaps inside each product's range; zero-demand weeks exist and units are ints; junk rows removed; the outlier is capped (max weekly units below 20,000); top-100 selection is unchanged when a product's sales are inflated only in the test period; closed-week detection works on a small example.
```
**Done when:** `pytest tests/test_data_prep.py -q` is fully green. Commit: `feat(data): weekly demand pipeline and tests`.

---

## Task D6: Exploratory data analysis

**SRS:** section 4, 5.3 | **Output:** `notebooks/01_eda.py`, `reports/figures/*.png`

```
Read docs/SRS.md section 4.2. Create notebooks/01_eda.py in cell format (# %% markers, runnable top to bottom) using data/processed/weekly_demand.parquet, closed_weeks.csv, and the cleaned raw transactions.

Produce and save to reports/figures/ (dpi 150, titled, axis labels with units):
1. Rows remaining after each cleaning step (horizontal bars)
2. Total weekly units with a dashed line at the test start (2011-10-10) and a marker on the closed week
3. Top 15 products by total units (code + short description)
4. Histogram of the share of zero-demand weeks per product, and print the overall zero share
5. Weekly units distribution, raw and log(1+x), side by side
6. Volume segments: low, medium, high (tertiles of average weekly units); save data/processed/product_segments.csv with columns product_id, segment
7. Coefficient of variation per product (histogram)
8. Country share of units (print top 8)
9. Quantity distribution and the five largest raw quantities, and the cap value

Under each section add a markdown cell "What to write in the report" with 2 to 3 sentences using the real numbers. At the end add a markdown summary of what the data shows about: the October to November level shift, the closure week, intermittent demand.
```
**Done when:** the notebook runs without error and 9 figures plus `product_segments.csv` exist. Commit: `feat(data): EDA and volume segments`.

---

## Task D7: Hand-off check to the team

```
Run the contract checks: pytest tests/test_contracts.py -q against data/processed/. Then write docs/HANDOFF_M1.md for Members 2 to 4 containing: file paths and schemas delivered (C1, C1b, product_segments.csv), the exact row and product counts, the closed-week rule (week 2010-12-27 is missing data, not zero demand), the train/test boundary dates, the quantity cap value, the cleaning log table, and three known data quirks they should remember (intermittent demand, level shift in October to November, bulk wholesale orders). Do not change any code.
```
**Done when:** the hand-off note exists and the contract tests pass. Tell the team in chat.

---

## Task D8: Forecast-error analysis (after Member 2 delivers `predictions.csv`)

**SRS:** 5.3 (FR-E-1 to FR-E-6) | **Output:** `src/error_analysis.py`, `error_analysis.csv`, notebook

```
Read docs/SRS.md section 5.3 and CONTRACTS.md C2. Inputs: data/processed/predictions.csv, metrics.csv, product_segments.csv, weekly_demand.parquet.

1. Implement src/error_analysis.py with build_error_analysis(predictions, segments) writing data/processed/error_analysis.csv: one tidy table with columns level (segment, week, product), key, model_name, n, MAE, RMSE, WAPE, bias, over_forecast_units, under_forecast_units. Segment level = low, medium, high; week level = the 8 test weeks; product level = every product. Include all models, split == "test" only.
2. Create notebooks/03_error_analysis.py (cell format) for the best model and the best baseline: error by volume segment; error by test week with a comment on spike weeks; predicted vs actual scatter and residual histogram; the 10 best and 10 worst products with a one-line reason each (intermittent demand, bulk orders, level shift); total over-forecast versus under-forecast units, with one sentence on why under-forecasting causes stockouts. Save figures to reports/figures/.
3. End the notebook with "For the report: findings": 5 concrete findings with numbers.

Tests: error_analysis.csv schema; segment-level numbers reproduce from predictions.csv; over plus under forecast equals total absolute error.
```
**Done when:** `error_analysis.csv` is written and Member 3 confirms `/error-analysis` can read it. Commit: `feat(analysis): forecast-error analysis`.

---

## Task D9: Data notes for the report and viva

```
Write docs/DATA_NOTES.md from the real project files only (cleaning_log.csv, weekly_demand.parquet, closed_weeks.csv, EDA outputs). Sections: Dataset and source; Cleaning rules and why each exists (one sentence each); real row counts after each step; weekly aggregation and why weekly (daily is too sparse); product selection and why training-only; outlier handling; the closure week; limitations. Mark any number you cannot find in the files as TODO. Then list 8 likely viva questions about the data with 2-sentence model answers.
```
**Done when:** the file exists with no invented numbers. Commit: `docs(data): data notes`.
