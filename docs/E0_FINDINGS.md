# Task E0 Findings: Online Retail II Exploration & Compatibility

**Date:** 2026-11-21  
**Dataset:** `data/raw/online_retail_II.xlsx` (UCI Dataset ID: 502)  
**Status:** Read-only analysis complete; parquet caches written to `data/raw/`.

---

## 1. Sheet Inspection and Parquet Caching

`data/raw/online_retail_II.xlsx` contains two sheets. Both sheets were read and cached to parquet files located in `data/raw/` matching `load_raw` string-column normalization:
- `data/raw/online_retail_II_Year_2009-2010.parquet`
- `data/raw/online_retail_II_Year_2010-2011.parquet`

### Summary of Sheets

| Sheet Name | Row Count | Columns | Min InvoiceDate | Max InvoiceDate | Cached Parquet File |
|---|---|---|---|---|---|
| `Year 2009-2010` | 525,461 | `Invoice`, `StockCode`, `Description`, `Quantity`, `InvoiceDate`, `Price`, `Customer ID`, `Country` | 2009-12-01 07:45:00 | 2010-12-09 20:01:00 | `data/raw/online_retail_II_Year_2009-2010.parquet` |
| `Year 2010-2011` | 541,910 | `Invoice`, `StockCode`, `Description`, `Quantity`, `InvoiceDate`, `Price`, `Customer ID`, `Country` | 2010-12-01 08:26:00 | 2011-12-09 12:50:00 | `data/raw/online_retail_II_Year_2010-2011.parquet` |

---

## 2. Column Mapping & Schema Comparison

### Column Name Mapping to Shelfwatch Standards
| Online Retail II Column | Shelfwatch Primary Column | Data Type | Notes |
|---|---|---|---|
| `Invoice` | `InvoiceNo` | `object` / `string` | Transaction identifier (cancellations start with `'C'`) |
| `StockCode` | `StockCode` | `object` / `string` | Product SKU (contains letters, digits, and special codes) |
| `Description` | `Description` | `object` / `string[python]` | Product textual description |
| `Quantity` | `Quantity` | `int64` | Units transacted (negative for cancellations/disposals) |
| `InvoiceDate` | `InvoiceDate` | `datetime64[ns]` | Timestamp of transaction |
| `Price` | `UnitPrice` | `float64` | Price per unit in GBP |
| `Customer ID` | `CustomerID` | `float64` | Customer numeric ID (nullable) |
| `Country` | `Country` | `object` | Customer destination country |

### Data Differences & Missing Values

1. **Missing Values**:
   - **`Year 2009-2010`**:
     - `Customer ID`: 107,927 missing rows (20.54%).
     - `Description`: 2,928 missing rows in Excel raw data (0.56%).
     - Other columns (`Invoice`, `StockCode`, `Quantity`, `InvoiceDate`, `Price`, `Country`): 0 missing values.
   - **`Year 2010-2011`**:
     - `Customer ID`: 135,080 missing rows (24.93%).
     - `Description`: 1,454 missing rows in Excel raw data (0.27%).
     - Matches the primary dataset (`Online Retail.xlsx` / `.parquet`), which also has 135,080 missing `CustomerID` rows and 1,454 missing `Description` rows.
2. **StockCode Formats**:
   - Both sheets contain non-product codes without digits:
     - `Year 2009-2010`: 2,782 rows with non-digit stock codes (e.g., `POST`, `D`, `M`, `BANK CHARGES`, etc.).
     - `Year 2010-2011`: 2,797 rows with non-digit stock codes.
   - Cancellations (Invoice starting with `'C'`):
     - `Year 2009-2010`: 10,206 cancellation rows.
     - `Year 2010-2011`: 9,288 cancellation rows.
3. **Raw Row Discrepancy (1 Row)**:
   - `Year 2010-2011` has 541,910 rows compared to 541,909 in `Online Retail.parquet` (+1 row).
   - Upon applying the standard Shelfwatch `clean_transactions` filter (drop cancellations, Quantity > 0, UnitPrice > 0, drop non-product codes), **both datasets yield exactly 527,734 clean transaction rows**. The extra raw row was an invalid transaction correctly filtered out by the pipeline.

---

## 3. Overlap Check (2010-12-06 to 2011-03-28)

We compared weekly total units between the primary dataset (`Online Retail`) and Online Retail II (`Year 2010-2011` sheet) for the 17 full Monday-start weeks spanning `2010-12-06` to `2011-03-28` after applying standard transaction cleaning:

| Week Start (`week_start`) | Main Data Units (`main_units`) | Online Retail II Units (`ret2_units`) | Absolute Difference | Relative Difference (%) |
|---|---:|---:|---:|---:|
| `2010-12-06` | 121,597 | 121,597 | 0 | 0.0000% |
| `2010-12-13` | 107,056 | 107,056 | 0 | 0.0000% |
| `2010-12-20` | 39,312 | 39,312 | 0 | 0.0000% |
| `2010-12-27` (Closed) | 0 | 0 | 0 | 0.0000% |
| `2011-01-03` | 76,831 | 76,831 | 0 | 0.0000% |
| `2011-01-10` | 89,372 | 89,372 | 0 | 0.0000% |
| `2011-01-17` | 144,691 | 144,691 | 0 | 0.0000% |
| `2011-01-24` | 63,141 | 63,141 | 0 | 0.0000% |
| `2011-01-31` | 69,365 | 69,365 | 0 | 0.0000% |
| `2011-02-07` | 51,540 | 51,540 | 0 | 0.0000% |
| `2011-02-14` | 81,344 | 81,344 | 0 | 0.0000% |
| `2011-02-21` | 84,514 | 84,514 | 0 | 0.0000% |
| `2011-02-28` | 69,336 | 69,336 | 0 | 0.0000% |
| `2011-03-07` | 69,271 | 69,271 | 0 | 0.0000% |
| `2011-03-14` | 87,746 | 87,746 | 0 | 0.0000% |
| `2011-03-21` | 77,992 | 77,992 | 0 | 0.0000% |
| `2011-03-28` | 106,752 | 106,752 | 0 | 0.0000% |

- **Maximum Relative Difference:** **`0.0000%`**
- **Outcome:** **PERFECT MATCH** ($\le 1.0\%$). The underlying transaction records and weekly demand totals in the overlap window are completely identical.

---

## 4. Prior-Year Coverage Analysis (2009-12-07 to 2010-11-29)

Using only the `Year 2009-2010` sheet across the 52 full Monday-start weeks between `2009-12-07` and `2010-11-29`, under the exact same cleaning rules:

### Cleaning Funnel for `Year 2009-2010`
| Step | Rows Remaining |
|---|---:|
| Raw rows | 525,461 |
| Drop cancellations (`InvoiceNo` starts with 'C') | 515,255 |
| Drop `Quantity <= 0` | 513,134 |
| Drop `UnitPrice <= 0` | 511,565 |
| Drop non-product `StockCode` | 509,229 |

### Sales Week Coverage for Shelfwatch Top 100 Products
For our official Top 100 products (selected on 2010–2011 training data):
- **Products with $\ge 1$ sale week:** **94 / 100 (94.0%)**
- **Products with $\ge 26$ sale weeks ($\ge 50\%$ of weeks):** **81 / 100 (81.0%)**
- **Products with $\ge 40$ sale weeks:** **69 / 100 (69.0%)**
- **Products with 0 prior-year sales:** **6 / 100 (6.0%)**

### Products with Zero Prior-Year Sales
The following 6 products have no sales recorded in the 2009–2010 period:
1. `22720`: `SET OF 3 CAKE TINS PANTRY DESIGN` (First sale in main data: `2010-12-13`)
2. `22993`: `SET OF 4 PANTRY JELLY MOULDS` (First sale in main data: `2011-02-14`)
3. `22998`: `TRAVEL CARD WALLET KEEP CALM` (First sale in main data: `2011-02-14`)
4. `23230`: `WRAP ALPHABET DESIGN` (First sale in main data: `2011-01-10`)
5. `23231`: `WRAP DOILEY DESIGN` (First sale in main data: `2011-01-10`)
6. `23232`: `WRAP VINTAGE PETALS  DESIGN` (First sale in main data: `2011-01-17`)

**Lineage Explanation:** All 6 products possess high numeric stock codes in the 22700–23200 range and were newly introduced products in late 2010 / early 2011. Their absence in the 2009–2010 period reflects genuine catalog evolution rather than missing data.

---

## 5. Shop Closure Weeks in 2009–2010

Evaluating all 52 full weeks in `Year 2009-2010`:
- **Shop-wide zero-sales week detected:** **`2009-12-28`** (Monday 2009-12-28 to Sunday 2010-01-03).
- Just as week `2010-12-27` is a complete Christmas/New Year shutdown in the primary dataset, week `2009-12-28` had 0 transactions shop-wide in the prior year.
- This week represents a scheduled shop closure, not zero demand, and must have no rows in `prior_year_weekly.parquet` (per Contract C10).

---

## 6. Viability Recommendation for Feature Group G3

### Coverage Summary
- $\ge 1$ sale week: **94%** (94/100 products)
- $\ge 26$ sale weeks: **81%** (81/100 products)
- Required threshold: $\ge 50\%$

### Final Recommendation: **VIABLE**
1. **Strong Historical Continuity:** With 94% of top products possessing prior-year sales and 81% having at least 26 weeks of historical demand, Feature Group G3 (`yoy_mean_5w`, `yoy_seasonal_index`, `yoy_available`) is **empirically viable** and should be included as a primary feature group in the modeling pipeline.
2. **Missing Product Handling:** The 6 newer products with zero prior-year sales will cleanly carry `yoy_available = 0` and imputed prior-year averages (or category medians) in `src/features.py`, preventing data leakage and ensuring robust model training.
3. **Calendar Alignment:** The Christmas closure in week `2009-12-28` aligns with `2010-12-27` in the primary dataset, confirming consistent holiday shutdown behavior across both years.
