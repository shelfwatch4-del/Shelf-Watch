# PROMPTS_DATA.md: Data, Enrichment and Intelligence (Member 1), version 3

**You own:** `src/data_prep.py`, `src/enrichment.py`, `src/intelligence.py`, `src/error_analysis.py`, `notebooks/01_eda.py`, `03_error_analysis.py`, `05_insights.py`, and the files C1, C1b, C8 to C12 in `data/processed/`.
**Setup:** paste the Windows preface and closing check from `PROMPTS_00_INDEX.md` into every prompt. Planning mode.

## Status of earlier tasks
| Task | Status |
|---|---|
| D0 to D7, D9 | Done (pipeline, tests, EDA, hand-off, data notes). 5,067 rows, 100 products, closure week has no row, zero-fill through 2011-11-28 |
| **D8** error analysis | **Pending:** run after Member 2 delivers `predictions.csv` (task M12). Notes at the end of this file |

New tasks start at **E0**.

---

## Task E0: Check Online Retail II before using it (read-only)
**SRS:** 4.6, A-6, Appendix D issue 1

Download **Online Retail II** from UCI (dataset id 502, `online_retail_II.xlsx`) and put it in `data/raw/`.
```
Read-only analysis. Do NOT change any code or data. Write findings to docs/E0_FINDINGS.md.

1. Open data/raw/online_retail_II.xlsx (cache to parquet next to it the same way load_raw does, one parquet per sheet). Print the sheet names, the columns, the row count and the min and max InvoiceDate of each sheet.
2. Map its columns to ours (Invoice -> InvoiceNo, Price -> UnitPrice, "Customer ID" -> CustomerID). State any other difference (dtypes, StockCode formats, missing values).
3. Overlap check: compare weekly total units (after the same cleaning as the main file) for the weeks 2010-12-06 to 2011-03-28 in Online Retail II and in our main data. Report the maximum relative difference. If they differ by more than 1 percent, say what could explain it.
4. Prior-year coverage: using only the sheet that covers 2009-12-01 to 2010-12-09, apply the same cleaning rules (cancellations, quantity, price, non-product codes). For our 100 products, print how many have at least one sale, at least 26 sale weeks, and at least 40 sale weeks between 2009-12-07 and 2010-11-29. List the products with no prior-year sales.
5. Print the zero-sales weeks in that sheet (shop closure weeks around Christmas 2009).
6. End with a recommendation: is the prior-year feature group viable? Give the coverage number that supports it. Do not claim viability without the printed coverage.
```
**Done when:** `docs/E0_FINDINGS.md` has real numbers. If coverage is below about 50% of the products, tell me; group G3 becomes optional.

---

## Task E1: Calendar table (contract C9)
**SRS:** FR-X-1

```
Read docs/SRS.md section 4.6 and CONTRACTS_ADDENDUM.md C9. Add holidays to requirements.txt (pinned) and install it.

In src/enrichment.py implement build_calendar_weekly(weeks) -> DataFrame with columns week_start, uk_bank_holiday_count, trading_days_expected, is_christmas_week, covering every Monday from 2009-12-07 to 2011-12-05.

- Bank holidays: use the holidays package for the United Kingdom, England subdivision (check which constructor the installed version supports and tell me). Verify that 2011-04-22, 2011-04-25, 2011-04-29 (royal wedding) and 2011-05-02 are present; if 2011-04-29 is missing, add it explicitly and say so.
- Trading weekdays: derive from the training data which weekdays have any sales (expected Monday to Friday and Sunday, no Saturday). trading_days_expected = trading weekdays in the week minus bank holidays that fall on a trading weekday.
- is_christmas_week = 1 if the week contains 24 to 26 December.
- Write data/processed/calendar_weekly.csv and a CLI: python -m src.enrichment --calendar.

Tests in tests/test_enrichment.py: the four holiday dates; trading_days_expected is lower in the 2011-04-25 and 2011-05-02 weeks than in a normal week; no Saturday counted; print the exact number of weeks and justify it; no missing values.
```
**Done when:** the tests pass and the CSV exists.

## Task E2: Prior-year table (contract C10)
**SRS:** FR-X-2, FR-X-4

```
Read docs/SRS.md section 4.6, CONTRACTS_ADDENDUM.md C10 and docs/E0_FINDINGS.md.

In src/enrichment.py implement build_prior_year_weekly(ret2_path, cap, products) -> DataFrame (product_id, week_start, units, orders):
- Read only the sheet covering 2009-12-01 to 2010-12-09, map columns with a small normalize_columns adapter (do NOT change how clean_transactions behaves for the main file), and apply the same cleaning rules.
- Cap Quantity at the SAME cap value used for the main training data (read it from the cleaning log; do not recompute it from this sheet).
- Aggregate to Monday weeks; keep only weeks BEFORE 2010-12-06 (no overlap with the main file) and only our 100 products.
- Zero-fill from each product's first prior-year sale to the last prior-year week; products with no prior-year sales get NO rows (missing, not zero). Closure weeks get no row.
- If the file is missing: log a warning, write an empty table with the right columns, and do not crash (FR-X-4).
- Print a coverage report: products with rows, median number of weeks per product, share with at least 26 weeks.
- Write data/processed/prior_year_weekly.parquet; CLI: python -m src.enrichment --prior-year.

Tests with a synthetic second-year sheet: same cleaning applied; no rows before a product's first sale; weeks on or after 2010-12-06 excluded; missing file handled; cap applied.
```

## Task E3: Transaction-derived weekly signals (contract C8)
**SRS:** FR-X-3

```
Read CONTRACTS_ADDENDUM.md C8. In src/enrichment.py implement build_enrichment_weekly(raw_path, products, weeks) -> DataFrame (product_id, week_start, gross_units, cancelled_units, export_units, guest_units) on EXACTLY the same product-week grid as weekly_demand.parquet.

- gross_units must equal C1 units (assert this).
- export_units: capped units where Country != "United Kingdom".
- guest_units: capped units where CustomerID is missing.
- cancelled_units: absolute Quantity on invoices starting with C for these products, after the same product-code cleaning (these are the cancellations the demand pipeline removes). Assign each to the week of its InvoiceDate. They never reduce demand units.
- Write data/processed/enrichment_weekly.parquet; CLI: python -m src.enrichment --enrichment.

Tests: grid equals the C1 grid; gross_units equals C1 units; export_units and guest_units never exceed gross_units; cancelled_units never negative; C1 itself is unchanged (compare hashes before and after).
```

## Task E4: Product intelligence (contract C11)
**SRS:** 5.4, FR-I-1 to FR-I-5, FR-I-7, FR-I-8

```
Read docs/SRS.md section 5.4 and CONTRACTS_ADDENDUM.md C11. Create src/intelligence.py with build_product_intelligence(weekly, raw_path, train_end="2011-10-10") -> DataFrame. Use TRAINING weeks only (week_start < train_end) for every number.

- Demand pattern (Syntetos-Boylan): n_weeks = open training weeks from the product's first sale; ADI = n_weeks / number of weeks with units > 0; CV2 = (std / mean) squared of the non-zero weekly units. smooth: ADI < 1.32 and CV2 < 0.49; erratic: ADI < 1.32 and CV2 >= 0.49; intermittent: ADI >= 1.32 and CV2 < 0.49; lumpy: ADI >= 1.32 and CV2 >= 0.49.
- ABC: revenue = sum of units x avg_price over training weeks; sort descending; cumulative share up to 80 percent is A, up to 95 percent B, the rest C. Also revenue_share.
- XYZ: coefficient of variation of weekly units (std / mean): X up to 0.5, Y up to 1.0, Z above.
- Lifecycle: new if the first sale is within the last 12 training weeks; inactive if no sales in the last 8 training weeks; otherwise compare the mean of the last 8 weeks with the mean of the 8 weeks before: growing if the ratio is at least 1.2, declining if at most 0.8, else mature. Also weeks_since_last_sale.
- return_rate = cancelled_units / gross_units (from enrichment_weekly, training weeks); export_share; top3_customer_share and known_customer_coverage from the raw file (known customers only; state the coverage).
- ALSO compute the demand pattern class for ALL products with at least 10 active training weeks (not only the top 100) and save it to data/processed/pattern_all_products.csv, so we can show why the top 100 were chosen.
- Write data/processed/product_intelligence.csv; CLI: python -m src.intelligence.

Tests (tests/test_intelligence.py): hand-made series that fall in each of the four pattern quadrants; ABC revenue shares sum to 1 and classes follow the 80 and 95 percent rules; lifecycle cases (new, inactive, growing, declining, mature); intelligence is IDENTICAL when sales in the test weeks are inflated; no row for products outside the top 100.

Then print the distribution of each class for the top 100 and for all products. Expect the top 100 to be mostly smooth or erratic; do not force variety. Report what you find.
```

## Task E5: Bought-together pairs (contract C12)
**SRS:** FR-I-6

```
Read CONTRACTS_ADDENDUM.md C12. In src/intelligence.py implement build_basket_pairs(raw_path, products, train_end="2011-10-10") -> DataFrame.

- Baskets are non-cancelled invoices dated before train_end, product codes cleaned like the main pipeline, restricted to our 100 products. Skip invoices with fewer than 2 of our products. Ignore invoices with more than 100 distinct products and say how many were ignored.
- Use a sparse invoice x product binary matrix (scipy.sparse) and co-occurrence = X.T @ X.
- confidence(A to B) = n_AB / n_A; lift = confidence / (n_B / N), where N is the number of qualifying invoices.
- Keep pairs with n_AB >= 30. For each product keep up to 3 companions ranked by lift (ties by n_AB).
- Write data/processed/basket_pairs.csv with product_id, companion_id, companion_description, n_invoices, confidence, lift.

Tests: a tiny set of invoices with hand-computed support, confidence and lift; minimum support enforced; a product with no qualifying pairs yields no rows; training-only (inflating test-period invoices changes nothing).

Print the 10 strongest pairs with real numbers read from the file. Do not quote numbers from memory.
```

## Task E6: Insights notebook
**SRS:** 5.4, FR-I-8

```
Create notebooks/05_insights.py (cell format, runs top to bottom). Save figures to reports/figures/ with titles and axis labels:
1. Demand pattern distribution: top 100 versus all products (grouped bars), with a sentence on why the top 100 were chosen.
2. ABC Pareto curve (cumulative revenue share by product rank) and XYZ counts.
3. Lifecycle stage counts, and the list of inactive products.
4. Top 10 products by return rate and by export share (with descriptions).
5. Top 10 bought-together pairs.
6. Weekly total units with bank-holiday and Christmas weeks marked, with a comment on the royal wedding and Easter weeks only if the data shows an effect.
7. If prior_year_weekly.parquet has data: 2010 versus 2011 weekly totals for the aligned weeks, showing the October to November 2010 build-up.
Under each section add a markdown cell "What to write in the report" with real numbers, framed as a decision for the Retail Inventory and Replenishment Manager (what should they do differently because of this finding?). End with 6 findings, including anything that surprised you or that did NOT show an effect.
```

## Task E7: Contracts, tests and hand-off
```
1. Confirm CONTRACTS.md contains the merged addendum (C1 amendments, C4, C7, C8 to C13). If not, merge CONTRACTS_ADDENDUM.md now and add a change-log row.
2. Extend tests/test_contracts.py with schema checks for C8, C9, C10, C11 and C12 on data/processed/ (and on data/mock/ with small mock versions generated in scripts/make_mocks.py).
3. Update docs/HANDOFF_M1.md and docs/DATA_NOTES.md: the new tables and their schemas, row counts, the prior-year coverage figure, sources with citations (UCI Online Retail, Online Retail II, holidays package version), and the limits (guest_share is a hypothesis; weather and retail-index series were deliberately not used, with the reason).
4. Run pytest -q and show the full output. Suggested commit message: feat(data): enrichment tables and product intelligence.
```

---

## Task D8 (pending): Forecast-error analysis
Run after M12 delivers `predictions.csv` and `metrics.csv`. Use the earlier D8 prompt, with two additions: split the error by demand pattern class and by ABC class from `product_intelligence.csv`, and add the two stopped-selling products (21829 and 22740) as a labelled case study in the notebook.
