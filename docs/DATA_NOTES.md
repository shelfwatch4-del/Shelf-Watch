# Shelfwatch Data Notes (`docs/DATA_NOTES.md`)

> Grounded strictly in verified project outputs: `data/processed/cleaning_log.csv`, `data/processed/weekly_demand.parquet`, `data/processed/closed_weeks.csv`, `data/processed/product_segments.csv`, `data/processed/enrichment_weekly.parquet`, `data/processed/calendar_weekly.csv`, `data/processed/prior_year_weekly.parquet`, `data/processed/product_intelligence.csv`, `data/processed/basket_pairs.csv`, `src/data_prep.py`, and `docs/E0_FINDINGS.md`.

---

## 1. Datasets, Sources and Citations

1. **Primary Dataset: UCI Online Retail (Dataset ID 352)**
   - **Source:** UCI Machine Learning Repository — *Online Retail*.
   - **Citation:** Chen, D. (2015). Online Retail [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5BW33
   - **Raw coverage:** `2010-12-01 08:26:00` to `2011-12-09 12:50:00` (`541,909` transaction lines across `8` raw columns: `InvoiceNo`, `StockCode`, `Description`, `Quantity`, `InvoiceDate`, `UnitPrice`, `CustomerID`, `Country`).
   - **Product & customer profile:** `3,958` distinct raw `StockCode`s (after strip/upper), `24.93%` of raw lines missing `CustomerID` (`24.90%` after row cleaning), and `83.45%` of cleaned unit volume (`4,654,261` units) shipped to the United Kingdom (Top 8 countries account for `96.50%`).

2. **Prior-Year Supporting Dataset: UCI Online Retail II (Dataset ID 502)**
   - **Source:** UCI Machine Learning Repository — *Online Retail II*.
   - **Citation:** Chen, D. (2019). Online Retail II [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5CG6D
   - **Coverage:** `Year 2009-2010` sheet contains `525,461` raw rows from `2009-12-01` to `2010-12-09`. Overlap check against primary dataset in `2010-12-06` to `2011-03-28` yielded an exact `0.0000%` difference (perfect match).

3. **Calendar Data: Python `holidays` Package**
   - **Source:** Python `holidays` package, United Kingdom / England calendar (version `>=0.40`, e.g. `holidays==0.60`).
   - **Key Holiday Events:** Accurately models standard statutory holidays and special occasions, specifically `2011-04-22` (Good Friday), `2011-04-25` (Easter Monday), `2011-04-29` (Royal Wedding of Prince William and Catherine Middleton), and `2011-05-02` (Early May bank holiday).

---

## 2. Cleaning Rules and Why Each Exists

1. **Drop cancellations (`InvoiceNo` starting with `C`):** Cancelled credit-note invoices reverse earlier purchases rather than representing positive customer demand.
2. **Drop `Quantity <= 0`:** Non-positive quantities after removing `C`-prefixed invoices reflect damaged write-offs, stock adjustments, or manual corrections rather than sales.
3. **Drop `UnitPrice <= 0`:** Zero- or negative-price rows represent complimentary giveaways, admin adjustments, or bad debt rather than commercial retail transactions.
4. **Drop non-product `StockCode`s (`POST`, `D`, `M`, `DOT`, `BANK CHARGES`, `AMAZONFEE`, `S`, `B`, `CRUK`, `PADS`, `C2`, no-digit codes, and `GIFT*`/`TEST*` prefixes):** Service fees, postage charges, samples, gift vouchers, and test entries are not physical inventory SKUs.
5. **Keep rows with missing `CustomerID`:** Product-level demand estimation depends on units sold per SKU rather than customer identity, so dropping guest/unidentified orders (`24.90%` of cleaned rows) would discard a quarter of true demand.
6. **Drop partial boundary weeks (`2010-11-29` and `2011-12-05`):** Incomplete 5-day edge weeks artificially depress weekly totals and would distort lag, rolling, and evaluation metrics.

---

## 3. Real Row Counts After Each Step

From `data/processed/cleaning_log.csv` and `src/data_prep.py`:

| Step | Rows Removed | Rows Remaining |
| --- | --- | ---: |
| `raw rows` | — | `541,909` |
| `drop cancellations` | `9,288` | `532,621` |
| `drop Quantity <= 0` | `1,336` | `531,285` |
| `drop UnitPrice <= 0` | `1,181` | `530,104` |
| `drop non-product StockCodes` | `2,370` | `527,734` |
| `drop partial weeks` | `27,348` | `500,386` |
| `cap Quantity (train)` | `0` (values clipped) | `500,386` |
| `keep top-100 products` | `423,738` | `76,648` |

- **Post-aggregation demand table (`data/processed/weekly_demand.parquet`, C1):** Aggregating the `76,648` transaction lines to `(product_id, week_start)`, zero-filling from each product's first active week through `2011-11-28`, and excluding closed week `2010-12-27` yields **`5,067` rows** covering **`100` products** across **`51` open Monday-start weeks** (`2010-12-06` to `2011-11-28`).

---

## 4. Enrichment and Product Intelligence Tables

### 4.1 `data/processed/enrichment_weekly.parquet` (Contract C8)
- **Dimensions:** `5,067` rows × `6` columns (`product_id, week_start, gross_units, cancelled_units, export_units, guest_units`).
- **Alignment:** Strictly identical product-week grid to C1 `weekly_demand.parquet`.
- **Integrity:** `gross_units` equals C1 `units`. `export_units` ($\le \text{gross}$) isolates international orders. `guest_units` ($\le \text{gross}$) captures anonymous guest volume. `cancelled_units` reflects return pressure without reducing sales demand.

### 4.2 `data/processed/calendar_weekly.csv` (Contract C9)
- **Dimensions:** `105` rows × `4` columns (`week_start, uk_bank_holiday_count, trading_days_expected, is_christmas_week`).
- **Scope:** Covers Mondays from `2009-12-07` to `2011-12-05`.
- **Trading calendar:** Derives trading weekdays from store sales (Monday through Friday plus Sunday = 6 days; Saturday is non-trading). Expected trading days equals 6 minus bank holidays occurring on trading weekdays.

### 4.3 `data/processed/prior_year_weekly.parquet` (Contract C10)
- **Dimensions:** Rows for `94` products covering Monday weeks between `2009-12-07` and `2010-11-29` (strictly prior to `2010-12-06`).
- **Coverage Analysis (from `docs/E0_FINDINGS.md`):**
  - Products with $\ge 1$ sale week: **`94 / 100` (`94.0%`)**.
  - Products with $\ge 26$ sale weeks ($\ge 50\%$ coverage): **`81 / 100` (`81.0%`)**.
  - Products with $\ge 40$ sale weeks: **`69 / 100` (`69.0%`)**.
  - Products with 0 prior-year sales: **`6 / 100` (`6.0%`)** (`22720`, `22993`, `22998`, `23230`, `23231`, `23232`). All 6 are late-2010/early-2011 product catalog introductions.
  - Prior-year closure week: `2009-12-28` had 0 store-wide transactions and has no row.

### 4.4 `data/processed/product_intelligence.csv` (Contract C11)
- **Dimensions:** `100` rows × `14` columns.
- **Metrics (Training weeks only, `week_start < 2011-10-10`):**
  - **Demand Pattern (Syntetos-Boylan):** Classified by Average Demand Interval (ADI cutoff 1.32) and squared coefficient of variation ($CV^2$ cutoff 0.49) into `smooth`, `erratic`, `intermittent`, or `lumpy`.
  - **ABC Classification:** Pareto cumulative revenue: `A` (top 80%), `B` (next 15%), `C` (tail 5%).
  - **XYZ Classification:** Demand volatility: `X` (CV $\le 0.5$), `Y` ($0.5 <$ CV $\le 1.0$), `Z` (CV $> 1.0$).
  - **Lifecycle Stage:** `new` (first sale in last 12 training weeks), `inactive` (no sales in last 8 training weeks), `growing` (last 8 weeks / prior 8 weeks $\ge 1.2$), `declining` ($\le 0.8$), `mature` (ratio between 0.8 and 1.2).
  - **Concentration & Return Metrics:** `return_rate`, `export_share`, `top3_customer_share`, `known_customer_coverage`, and `revenue_share`.

### 4.5 `data/processed/basket_pairs.csv` (Contract C12)
- **Dimensions:** Co-occurring invoice pairs ($\ge 30$ invoices) among Top-100 products during the training period, capped at 3 companions per product ranked by lift.
- **Fields:** `product_id, companion_id, companion_description, n_invoices, confidence, lift`.

---

## 5. Methodological Boundaries and Limitations

1. **Guest Share as an Empirical Hypothesis:**  
   `guest_share` measures the fraction of weekly units from lines where `CustomerID` was missing (~24.9% of lines). In online retail, buyers without accounts frequently exhibit impulse purchases or higher price sensitivity. We treat `guest_share` strictly as an **empirical hypothesis** to test in feature ablation, acknowledging that anonymous checkouts may also include privacy-conscious returning buyers or corporate procurement.

2. **Deliberate Exclusion of Weather and Retail-Index Series:**  
   External macroeconomic series (e.g. ONS Retail Sales Index) and local weather metrics were deliberately omitted. With only one calendar year of main historical data (52 full weeks), any national macro index or seasonal weather series is mathematically collinear with the autumn/Q4 monotonic time trend. Incorporating them risks overfitting to a single seasonal surge without the multi-year variation needed to separate true external drivers from catalog seasonality.

3. **Single Year and Q4 Level Shift:**  
   Open training-week demand averages `25,000.7` units/week, while the 8-week test period averages `29,515.6` units/week (`+18.1%` / `1.18x`). Tree-based models cannot extrapolate above training target values, which causes systematic negative bias during peak holiday weeks unless trend-aware adjustments or linear models (Ridge) are employed.

4. **Simulated Inventory Levels:**  
   Because the UCI dataset logs sales invoices rather than ERP inventory ledgers, on-hand stock and supplier order histories are simulated in accordance with SRS Section 6.2 ($on\_hand = 3.0 \times avg\_train \times \text{Uniform}(0.5, 1.5)$). All stockout risk indicators demonstrate how replenishment rules respond under specified lead times and service levels, rather than real warehouse positions.
