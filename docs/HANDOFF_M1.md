# Member 1 Data Hand-off (`docs/HANDOFF_M1.md`)

**From:** Member 1 (Data & Analysis)  
**To:** Member 2 (ML Modeling), Member 3 (Risk Engine & API), Member 4 (Dashboard & QA)  
**Status:** Verified against `CONTRACTS.md` (C1, C1b) via `pytest tests/test_contracts.py -q`

---

## 1. Delivered Files & Schemas

### 1.1 `data/processed/weekly_demand.parquet` (Contract C1)
- **Dimensions:** `5,147` rows × `7` columns
- **Scope:** `100` Top-N products (selected on training period only, `>= 30` active training weeks) across `52` full Monday-start weeks (`2010-12-06` through `2011-11-28`)
- **Ordering & Grid:** Sorted by `product_id, week_start`; gapless Monday ranges between each product's first and last active week; zero-demand weeks filled with `units = orders = customers = 0` and forward-filled `avg_price`

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

---

## 2. Closed-Week Rule (Critical for Member 2)

> **Week `2010-12-27` is MISSING DATA, NOT zero demand.**

- The shop recorded zero transactions across all products during the Christmas/New Year closure week (`2010-12-27`).
- It remains in `weekly_demand.parquet` with `units = 0` solely so the Monday calendar grid is regular.
- **Required handling in `src/features.py` & `src/train.py`:**
  1. Never include `2010-12-27` as a training target row.
  2. Mask `2010-12-27` as `NaN` before computing lags and rolling statistics so the shutdown never leaks as `0` demand into subsequent weeks' features.
  3. Exclude `2010-12-27` when computing `avg_weekly_units_train` for `next_week_forecast.csv` (C4).

---

## 3. Train / Test Boundary & Forecast Target

| Period | Weeks | Date Range (`week_start`) | Usage Rule |
| --- | --- | --- | --- |
| **Training** | `44` | `2010-12-06` to `2011-10-03` | Feature engineering, `TimeSeriesSplit` tuning, model refit, `avg_weekly_units_train` |
| **Test (Hold-out)** | `8` | `2011-10-10` to `2011-11-28` | Final evaluation & error analysis only — never touch during tuning, imputation, capping, or segmentation |
| **Forecast Target** | `1` | `2011-12-05` | Next-week forecast output (`data/processed/next_week_forecast.csv`, C4) |

---

## 4. Outlier Quantity Cap & Cleaning Log

- **Line-level Quantity Cap:** **`480.0` units** (99.9th percentile fitted strictly on full-week training rows; **`415` lines capped** across full weeks).

| Step | Rows Remaining |
| --- | ---: |
| `raw rows` | `541,909` |
| `drop cancellations` | `532,621` |
| `drop Quantity <= 0` | `531,285` |
| `drop UnitPrice <= 0` | `530,104` |
| `drop non-product StockCodes` | `527,734` |
| `drop partial weeks` | `500,386` |
| `cap Quantity (train)` | `500,386` |
| `keep top-100 products` | `76,648` |

*(Aggregating the `76,648` Top-100 lines to `product_id × week_start` and filling internal gaps yields `5,147` rows.)*

---

## 5. Three Known Data Quirks to Remember

1. **Intermittent demand:** `3.98%` (`205 / 5,147`) of all rows have `units == 0` (`2.18%` excluding the `2010-12-27` closure); `23` of `100` products have at least one zero open week (up to `25.5%`), and per-product CV averages `0.731` (up to `1.501`). **Never use `MAPE`** — report `MAE`, `RMSE`, `WAPE`, and `bias`.
2. **Level shift in October–November:** Total demand across the Top 100 products averages `29,515.6` units/week in the 8-week test period versus `25,000.7` units/week in open training weeks (`+18.1%` / `1.18x`). Because tree models cannot extrapolate above training ranges, check test-period under-forecasting (`bias`) closely and compare against Ridge regression.
3. **Bulk wholesale orders:** Extreme single lines (`80,995` and `74,215` units raw) were capped at `480.0` units, but multi-line wholesale orders still create strong right-skew (weekly `units` median `184.0`, mean `254.7`, max `2,755`). Expect isolated spike errors and wide `sigma_error` on wholesale-heavy SKUs.

