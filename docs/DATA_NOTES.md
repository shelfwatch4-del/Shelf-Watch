# Shelfwatch Data Notes (`docs/DATA_NOTES.md`)

> Grounded strictly in verified project outputs: `data/processed/cleaning_log.csv`, `data/processed/weekly_demand.parquet`, `data/processed/closed_weeks.csv`, `data/processed/product_segments.csv`, `src/data_prep.py`, and `notebooks/01_eda.py`.

---

## 1. Dataset and Source

- **Source:** UCI Machine Learning Repository — *Online Retail* dataset (ID 352), representing transactions from a UK-based online giftware retailer.
- **Raw coverage:** `2010-12-01 08:26:00` to `2011-12-09 12:50:00` (`541,909` transaction lines across `8` raw columns: `InvoiceNo`, `StockCode`, `Description`, `Quantity`, `InvoiceDate`, `UnitPrice`, `CustomerID`, `Country`).
- **Product & customer profile:** `3,958` distinct raw `StockCode`s (after strip/upper), `24.93%` of raw lines missing `CustomerID` (`24.90%` after row cleaning), and `83.45%` of cleaned unit volume (`4,654,261` units) shipped to the United Kingdom (Top 8 countries account for `96.50%`).

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
| --- | ---: | ---: |
| `raw rows` | — | `541,909` |
| `drop cancellations` | `9,288` | `532,621` |
| `drop Quantity <= 0` | `1,336` | `531,285` |
| `drop UnitPrice <= 0` | `1,181` | `530,104` |
| `drop non-product StockCodes` | `2,370` | `527,734` |
| `drop partial weeks` | `27,348` | `500,386` |
| `cap Quantity (train)` | `0` (values clipped) | `500,386` |
| `keep top-100 products` | `423,738` | `76,648` |

- **Post-aggregation table (`data/processed/weekly_demand.parquet`):** Aggregating the `76,648` transaction lines to `(product_id, week_start)`, zero-filling from each product's first active week through `2011-11-28`, and excluding closed week `2010-12-27` yields **`5,067` rows** covering **`100` products** across **`51` open Monday-start weeks** (`2010-12-06` to `2011-11-28`). Two Top-100 products stopped selling before `2011-11-28` (`21829` last active `2011-10-17`, `22740` last active `2011-09-26`) and are zero-filled through `2011-11-28`.

---

## 4. Weekly Aggregation and Why Weekly

- **Why weekly instead of daily:** Daily SKU-level sales in gift retail are heavily sparse (zero-inflated, with no Saturday trading and intermittent weekday orders), whereas aggregating to Monday-start weeks (`week_start`) pools multi-day noise into a clean signal aligned with weekly replenishment decisions.
- **Aggregation rules:** For each `(product_id, week_start)`, `units = sum(Quantity)`, `orders = nunique(InvoiceNo)`, `customers = nunique(CustomerID)` (ignoring `NaN`), and `avg_price = mean(UnitPrice)`.
- **Gap filling & density:** Missing open weeks from each product's first active week through the last full week (`2011-11-28`), excluding `2010-12-27`, are filled with `units = orders = customers = 0` and forward-filled `avg_price`. Across the resulting `5,067` rows, `125` rows (`2.47%`) have `units == 0`.

---

## 5. Product Selection and Why Training-Only

- **Eligibility & selection:** Within the full-week training period (`44` weeks: `2010-12-06` to `2011-10-03`), `1,224` of `3,658` active SKUs sold in at least `30` distinct training weeks; the Top `100` by total training-period units were selected.
- **Volume segments (`data/processed/product_segments.csv`):** Products are split into training-period tertiles (excluding `2010-12-27`): `low` (`34` SKUs, `142.8`–`174.7` avg units/week), `medium` (`32` SKUs, `175.7`–`249.7`), and `high` (`34` SKUs, `256.5`–`856.5`).
- **Why training-only:** Selecting the product universe or segment cutoffs using the `8`-week test set (`2011-10-10` to `2011-11-28`) would introduce selection/survivorship leakage, biasing evaluation toward products known to surge during the test period.

---

## 6. Outlier Handling

- **Extreme raw orders:** The five largest raw single-line quantities are `80,995`, `74,215`, `12,540`, `5,568`, and `4,800` units (`80,995` and `74,215` survive row-level cleaning before capping).
- **Training-only capping:** Line-level `Quantity` is capped at the `99.9`th percentile learned exclusively from full-week training rows, yielding a cap of **`480.0` units** and clipping **`415` lines** without letting test-period orders influence the threshold.

---

## 7. The Closure Week

- **Detected closure (`data/processed/closed_weeks.csv`):** The full week starting **`2010-12-27`** recorded **`0` transactions** across the entire shop (Christmas/New Year shutdown).
- **Handling rule:** Week `2010-12-27` is excluded from `weekly_demand.parquet` (missing data, never zero demand), and downstream feature/training pipelines skip over it for lags, rolling calculations, and `avg_weekly_units_train`.

---

## 8. Limitations

- **Single year of history:** With only `52` full weeks (`2010-12-06` to `2011-11-28`), multi-year seasonality cannot be learned, and calendar variables (`week_of_year`, `weeks_to_christmas`) behave nearly like a linear time index and must be tested by ablation (`TODO`: ablation WAPE delta once Member 2 runs `src/train.py`).
- **October–November level shift:** Total weekly demand across the Top `100` SKUs averages `29,515.6` units/week in the `8`-week test period compared to `25,000.7` units/week in open training weeks (`+18.1%` / `1.18x`), which poses an extrapolation risk for tree-based models (`TODO`: test-period model bias once Member 2 delivers `metrics.csv` and `predictions.csv`).
- **Intermittent demand & volatility:** `23` of `100` selected products have zero-demand open weeks (up to `35.3%`, including discontinued SKUs `21829` after `2011-10-17` and `22740` after `2011-09-26`), and per-product CV averages `0.734` (range `0.415`–`1.501`), preventing the use of `MAPE`.
- **Bulk wholesale orders:** Even after capping individual lines at `480.0` units, multi-line wholesale orders produce sharp weekly demand spikes (up to `2,755` units/week).
- **No real inventory records:** The UCI dataset logs sales only, not stock-on-hand or supplier deliveries, so downstream stockout risk figures rely on simulated starting stock (`TODO`: final risk counts once Member 3 runs `src/risk_engine.py` on real forecasts).

---

## 9. Likely Viva Questions & Model Answers

1. **Why did you aggregate transactions to weekly rather than daily demand?**  
   Daily SKU-level sales in this dataset are sparse and include zero-trading days such as Saturdays, making next-day forecasts noisy and zero-inflated. Aggregating to Monday-start weeks produces `5,067` clean observations across `51` open weeks with only `2.47%` open-week zeros, matching a realistic weekly inventory review cycle.

2. **Why did you keep rows with missing `CustomerID` (~`24.9%` of lines)?**  
   Our forecasting target is total weekly units sold per SKU (`StockCode`), which does not require identifying individual buyers. Dropping guest checkouts would systematically understate true product demand by roughly a quarter.

3. **Why is week `2010-12-27` listed in `closed_weeks.csv` and treated as missing instead of zero demand?**  
   Every product recorded zero transactions during `2010-12-27` because the store was closed for the Christmas/New Year break, not because customer demand fell to zero. Keeping it in the parquet index preserves regular Monday spacing, while masking it as `NaN` in feature engineering prevents zero-demand contamination of lags and training targets.

4. **Why were both Top-100 product selection and the `99.9`th-percentile quantity cap computed on the training period only?**  
   Using test-period sales (`2011-10-10` to `2011-11-28`) to pick top sellers or learn the `480.0`-unit outlier cap would leak future information into pipeline setup and bias evaluation. Restricting both steps to the `44` training weeks guarantees strict chronological isolation, which we verify directly with a test-period inflation check in `tests/test_data_prep.py`.

5. **Why did you drop the partial boundary weeks (`2010-11-29` and `2011-12-05`)?**  
   The raw file starts mid-week on Wednesday `2010-12-01` and ends mid-week on Friday `2011-12-09`, so those edge weeks contain only partial trading days (`27,348` rows total). Retaining incomplete weeks would look like sudden demand drops at the start and end of the series and corrupt lag and test-period totals.

6. **Why is `MAPE` prohibited in favor of `MAE`, `RMSE`, `WAPE`, and `bias`?**  
   `3.98%` of product-week rows (`2.18%` on open weeks across `23` products) have zero actual demand, where `MAPE` divides by zero and becomes undefined or explodes on tiny denominators. `WAPE` weights errors by total volume, `MAE` and `RMSE` measure unit accuracy, and `bias` directly tracks systematic over- or under-forecasting.

7. **How do you handle extreme wholesale orders such as `80,995` and `74,215` units?**  
   Instead of deleting wholesale lines—which would erase real demand—we clip line-level `Quantity` at the training `99.9`th percentile (`480.0` units, affecting `415` full-week lines). This prevents single bulk anomalies from dominating model weights while keeping the product active in the series.

8. **What happens to your models given only one year of data and the October–November `+18.1%` level shift?**  
   With `52` weeks, true annual seasonality cannot be separated from a monotonic time trend, so calendar features must be validated via ablation rather than assumed helpful. In addition, because open-week demand rises from `25,000.7` units/week in training to `29,515.6` in the test period, tree models may under-forecast test peaks since they cannot extrapolate beyond training targets, making linear baselines like Ridge essential to compare.

