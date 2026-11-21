# Member 1 Data Hand-off (`docs/HANDOFF_M1.md`)

**From:** Member 1 (Data & Analysis)  
**To:** Member 2 (ML Modeling), Member 3 (Risk Engine & API), Member 4 (Dashboard & QA)  
**Status:** Verified against `CONTRACTS.md` (C1, C1b, C8, C9, C10, C11, C12) via `pytest tests/test_contracts.py -q`

---

## 1. Delivered Files & Schemas

### 1.1 `data/processed/weekly_demand.parquet` (Contract C1)
- **Dimensions:** `5,067` rows × `7` columns
- **Scope:** `100` Top-N products (selected on training period only, `>= 30` active training weeks) across `51` open Monday-start weeks (`2010-12-06` through `2011-11-28`, excluding closed week `2010-12-27`)
- **Ordering & Grid:** Sorted by `product_id, week_start`; gapless Monday ranges (excluding `closed_weeks.csv`) from each product's first active week through the last full week (`2011-11-28`); zero-demand weeks filled with `units = orders = customers = 0` and forward-filled `avg_price` (two discontinued/tail-inactive SKUs, `21829` after `2011-10-17` and `22740` after `2011-09-26`, are zero-filled through `2011-11-28`)

| Column | Dtype | Constraints / Notes |
| --- | --- | --- |
| `product_id` | `str` (`object`) | Cleaned, stripped, upper-cased `StockCode` |
| `description` | `str` (`object`) | Most frequent description in training period |
| `week_start` | `datetime64[ns]` | Monday date (`weekday == 0`) |
| `units` | `int64` | Total units sold (`>= 0`); target variable |
| `orders` | `int64` | Distinct invoices (`>= 0`) |
| `customers` | `int64` | Distinct non-null `CustomerID`s (`>= 0`) |
| `avg_price` | `float64` | Mean `UnitPrice` (`> 0`, finite); forward-filled in zero weeks |

### 1.2 `data/processed/closed_weeks.csv` (Contract C1b)
- **Dimensions:** `1` row × `1` column
- **Content:**

| Column | Type | Value |
| --- | --- | --- |
| `week_start` | `date` (`YYYY-MM-DD`) | `2010-12-27` |

### 1.3 `data/processed/product_segments.csv`
- **Dimensions:** `100` rows × `2` columns
- **Definition:** Tertiles (`pd.qcut(..., q=3)`) of average weekly `units` computed strictly on **training open weeks** (`2010-12-06` to `2011-10-03`, excluding `2010-12-27`) to prevent test-set leakage when Member 2 uses pooled segment error as a fallback for `sigma_error`
- **Distribution:** `low`: `34` products (`142.8`–`174.7` avg units/week), `medium`: `32` products (`175.7`–`249.7`), `high`: `34` products (`256.5`–`856.5`)

| Column | Type | Allowed Values |
| --- | --- | --- |
| `product_id` | `str` | Matches `weekly_demand.parquet` |
| `segment` | `str` | `low`, `medium`, `high` |

### 1.4 Companion Artifacts
- `data/processed/cleaning_log.csv` — step-by-step transaction row counts (`8` steps)
- `reports/figures/01_cleaning_steps.png` through `09_quantity_cap_effect.png` — EDA figures (`150` dpi)

### 1.5 `data/processed/enrichment_weekly.parquet` (Contract C8)
- **Producer:** M1 | **Consumer:** M2
- **Dimensions:** `5,067` rows × `6` columns
- **Scope & Grid:** Strictly identical `(product_id, week_start)` grid to C1 `weekly_demand.parquet`. All values are for the named week $w$ (no shifting; shifting occurs strictly in `src/features.py`).
- **Assertions:** `gross_units` strictly equals C1 `units`; `export_units <= gross_units`; `guest_units <= gross_units`; `cancelled_units >= 0`.

| Column | Dtype | Constraints / Notes |
| --- | --- | --- |
| `product_id` | `str` (`object`) | Cleaned `StockCode` matching C1 |
| `week_start` | `datetime64[ns]` | Monday date matching C1 (`weekday == 0`) |
| `gross_units` | `int64` | Capped demand units (assert equals C1 `units`) |
| `cancelled_units` | `int64` | Absolute units on cancelled invoices (`InvoiceNo` starts with 'C') |
| `export_units` | `int64` | Capped units shipped outside UK (`Country != "United Kingdom"`) |
| `guest_units` | `int64` | Capped units where `CustomerID` was missing |

### 1.6 `data/processed/calendar_weekly.csv` (Contract C9)
- **Producer:** M1 | **Consumer:** M2
- **Dimensions:** `105` rows × `4` columns
- **Scope:** Every Monday week from `2009-12-07` through `2011-12-05` (52 weeks in 2010, 52 weeks in 2011, plus boundary horizon week).
- **Definitions:** `uk_bank_holiday_count` counts official bank holidays falling on shop trading weekdays; `trading_days_expected` is standard trading weekdays (6: Monday–Friday and Sunday, Saturday closed) minus bank holidays on trading weekdays; `is_christmas_week` is 1 if the week spans Dec 24, 25, or 26.

| Column | Dtype | Constraints / Notes |
| --- | --- | --- |
| `week_start` | `str` (`date`) | Monday date (`YYYY-MM-DD`) from `2009-12-07` to `2011-12-05` |
| `uk_bank_holiday_count` | `int64` | Bank holidays in week (`>= 0`) |
| `trading_days_expected` | `int64` | Expected trading days (`0` to `6`, typically `5` or `6`) |
| `is_christmas_week` | `int64` | Binary flag: `1` if week contains Dec 24–26, else `0` |

### 1.7 `data/processed/prior_year_weekly.parquet` (Contract C10)
- **Producer:** M1 | **Consumer:** M2
- **Dimensions:** Rows spanning `94` products across prior-year weeks (`2009-12-07` to `2010-11-29`, strictly prior to `2010-12-06`).
- **Scope & Filtering:** Sourced from UCI Online Retail II (`Year 2009-2010` sheet). Zero-filled from each product's first prior-year sale through `2010-11-29`. The 6 products with no prior-year sales have **no rows** (missing, not zero). The prior-year closure week `2009-12-28` has **no rows**.

| Column | Dtype | Constraints / Notes |
| --- | --- | --- |
| `product_id` | `str` (`object`) | Top-100 product SKU |
| `week_start` | `datetime64[ns]` | Monday date `< 2010-12-06` (excluding `2009-12-28`) |
| `units` | `int64` | Weekly units capped with main training cap (`480.0`) |
| `orders` | `int64` | Distinct invoice count (`>= 0`) |

### 1.8 `data/processed/product_intelligence.csv` (Contract C11)
- **Producer:** M1 | **Consumers:** M3 (API), M4 (Dashboard)
- **Dimensions:** `100` rows × `14` columns
- **Scope:** Computed strictly on training open weeks (`week_start < 2011-10-10`, excluding `2010-12-27`).

| Column | Dtype | Constraints / Notes |
| --- | --- | --- |
| `product_id` | `str` | Top-100 SKU |
| `description` | `str` | Most frequent description in training period |
| `adi` | `float64` | Average Demand Interval: open weeks / weeks with `units > 0` (`> 0`) |
| `cv2` | `float64` | Squared coefficient of variation of non-zero weekly units (`>= 0`) |
| `pattern_class` | `str` | Syntetos-Boylan: `smooth`, `erratic`, `intermittent`, `lumpy` |
| `abc_class` | `str` | Pareto cumulative revenue: `A` (top 80%), `B` (next 15%), `C` (tail 5%) |
| `xyz_class` | `str` | Demand variability: `X` (CV $\le 0.5$), `Y` ($0.5 <$ CV $\le 1.0$), `Z` (CV $> 1.0$) |
| `lifecycle_stage` | `str` | `new`, `growing`, `mature`, `declining`, or `inactive` |
| `weeks_since_last_sale` | `int64` | Open training weeks elapsed since last non-zero sale (`>= 0`) |
| `return_rate` | `float64` | `cancelled_units / gross_units` over training period (`>= 0`) |
| `export_share` | `float64` | Fraction of units shipped outside UK (`0.0` to `1.0`) |
| `top3_customer_share` | `float64` | Fraction of volume taken by top 3 identified customers (`0.0` to `1.0`) |
| `known_customer_coverage` | `float64` | Fraction of transactions with non-null `CustomerID` (`0.0` to `1.0`) |
| `revenue_share` | `float64` | SKU training revenue / total Top-100 training revenue (sums to `1.0`) |

### 1.9 `data/processed/basket_pairs.csv` (Contract C12)
- **Producer:** M1 | **Consumers:** M3, M4
- **Scope:** Non-cancelled training invoices dated before `2011-10-10` containing Top-100 SKUs. Restricted to pairs with $n_{AB} \ge 30$ invoices, up to 3 companions per product ranked by lift.

| Column | Dtype | Constraints / Notes |
| --- | --- | --- |
| `product_id` | `str` | Focal product SKU |
| `companion_id` | `str` | Co-occurring product SKU (`!= product_id`) |
| `companion_description` | `str` | Description of companion product |
| `n_invoices` | `int64` | Co-occurrence count ($\ge 30$) |
| `confidence` | `float64` | $P(\text{companion} \mid \text{product}) = n_{AB} / n_A$ (`0.0` to `1.0`) |
| `lift` | `float64` | $\text{confidence} / (n_B / N)$ (`> 0`) |

---

## 2. Closed-Week Rule (Critical for Member 2)

> **Week `2010-12-27` is MISSING DATA, NOT zero demand.**

- The shop recorded zero transactions across all products during the Christmas/New Year closure week (`2010-12-27`).
- It is excluded from `weekly_demand.parquet` and `enrichment_weekly.parquet` (missing, not zero demand) and listed in `data/processed/closed_weeks.csv`.
- Similarly, in prior-year data (`prior_year_weekly.parquet`), week `2009-12-28` is a Christmas closure week and has no rows.
- **Required handling in `src/features.py` & `src/train.py`:**
  1. Never include `2010-12-27` as a training target row.
  2. Mask `2010-12-27` as `NaN` before computing lags and rolling statistics so the shutdown never leaks as `0` demand into subsequent weeks' features.
  3. Exclude `2010-12-27` when computing `avg_weekly_units_train` for `next_week_forecast.csv` (C4).

---

## 3. Train / Test Boundary & Forecast Target

| Period | Weeks | Date Range (`week_start`) | Usage Rule |
| --- | --- | --- | --- |
| **Training** | `44` | `2010-12-06` to `2011-10-03` | Feature engineering, `TimeSeriesSplit` tuning, model refit, `avg_weekly_units_train`, product intelligence (C11), basket pairs (C12) |
| **Test (Hold-out)** | `8` | `2011-10-10` to `2011-11-28` | Final evaluation & error analysis only — never touch during tuning, imputation, capping, segmentation, or intelligence metrics |
| **Forecast Target** | `1` | `2011-12-05` | Next-week forecast output (`data/processed/next_week_forecast.csv`, C4) |

---

## 4. Outlier Quantity Cap & Cleaning Log

- **Line-level Quantity Cap:** **`480.0` units** (99.9th percentile fitted strictly on full-week training rows; **`415` lines capped** across full weeks).
- The exact same cap (`480.0` units) is applied to prior-year transactions in `prior_year_weekly.parquet` (C10) to maintain uniform scale.

---

## 5. Prior-Year Coverage Summary (from `docs/E0_FINDINGS.md`)

Evaluation of UCI Online Retail II (`Year 2009-2010` sheet) for the Top 100 products:
- **$\ge 1$ sale week:** **`94 / 100` products (`94.0%`)**
- **$\ge 26$ sale weeks ($\ge 50\%$ coverage):** **`81 / 100` products (`81.0%`)**
- **$\ge 40$ sale weeks:** **`69 / 100` products (`69.0%`)**
- **Zero prior-year sales:** **`6 / 100` products (`6.0%`)** (`22720`, `22993`, `22998`, `23230`, `23231`, `23232`). These represent newer product introductions launched in late 2010 or early 2011.
- **Closure week:** `2009-12-28` recorded 0 shop-wide sales and is omitted.

---

## 6. Sources and Citations

1. **Primary Dataset:**  
   UCI Machine Learning Repository — *Online Retail* (Dataset ID 352).  
   *Citation:* Chen, D. (2015). Online Retail [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5BW33
2. **Prior-Year Supporting Dataset:**  
   UCI Machine Learning Repository — *Online Retail II* (Dataset ID 502).  
   *Citation:* Chen, D. (2019). Online Retail II [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5CG6D
3. **Calendar & Bank Holidays:**  
   Python `holidays` library, United Kingdom / England calendar.  
   *Package:* `holidays>=0.40` (e.g. pinned in requirements). Verified holiday dates: `2011-04-22` (Good Friday), `2011-04-25` (Easter Monday), `2011-04-29` (Royal Wedding of Prince William and Catherine Middleton), and `2011-05-02` (Early May bank holiday).

---

## 7. Methodological Limits & Feature Exclusions

1. **Guest Share Hypothesis:**  
   `guest_share` (`guest_units / gross_units`) represents lines where `CustomerID` is null (~24.9% of transactions). In gift retail, anonymous checkouts often surge during gifting seasons. However, this is an **empirical hypothesis**, not verified buyer psychology: an unidentified checkout may represent a first-time guest or an unregistered corporate buyer. Downstream models must evaluate its ablation impact before adoption.
2. **Deliberate Exclusion of Weather and Retail-Index Series:**  
   External series such as temperature/rainfall or national macroeconomic indices (e.g., ONS monthly retail sales indices) were deliberately excluded. Because the primary dataset spans only a single 52-week period, external macro and weather variables are collinear with the monotonic time trend and Q4 calendar surge. With only 1 year of data, a model cannot statistically separate weather anomalies or macro indices from annual seasonality, leading to spurious correlations.
