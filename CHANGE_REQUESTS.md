# Change Requests

## CR-001: SRS Section 5.1 Expected Test Row Count (`800` vs `786`) and Training Target Weeks (`44` vs `35`)

- **Date:** 2026-11-12
- **Status:** Accepted by team: extend zero-fill
- **Affected Spec:** `docs/SRS.md` Section 5.1 (line 277) and `.agents/rules/01-frozen-decisions.md` (line 12)
- **Contracts Affected:** `CONTRACTS.md` C1 (see CR-002)

### Discrepancy Observed
1. **Test rows (`800` expected vs `786` prior to CR-002):**
   Two Top-100 products selected on the training period stopped selling before `2011-11-28`:
   - `21829`: last sale `2011-10-17` (only 2 test weeks under the old fill rule).
   - `22740`: last sale `2011-09-26` (0 test weeks under the old fill rule).
2. **Training target weeks (`44` raw vs `35` target weeks):**
   The raw training period (`2010-12-06` to `2011-10-03`) spans 44 full weeks; after excluding closed week `2010-12-27` (1 week) and dropping the first 8 non-closed history weeks per product (`_LONGEST_WINDOW = 8`), `build_features` yields **35** distinct training target weeks (`2011-02-07` to `2011-10-03`, `3,467` training rows).

---

## CR-002: Extend Zero-Filling Through Last Full Week (`2011-11-28`) and Exclude Closure Week (`2010-12-27`)

- **Date:** 2026-11-12
- **Status:** Accepted by team: extend zero-fill
- **Affected Files:** `src/data_prep.py`, `CONTRACTS.md` (C1), `PROJECT_CONTEXT.md` (Section 5), `docs/HANDOFF_M1.md`, `docs/DATA_NOTES.md`, `tests/test_data_prep.py`, `tests/test_contracts.py`

### Decision
- Extend zero-filling in `_fill_product_weeks` / `build_weekly_demand` so every product is filled from its first active week **through the last full week (`2011-11-28`)**, not just to its last sale.
- Exclude shop-closed week `2010-12-27` from `weekly_demand.parquet` (missing, not zero demand).
- Do not fill rows before a product's first active week.
- Update `validate_c1` to require every product's range to end at the last full week (`2011-11-28`), with no gaps except closed weeks.
- Resulting counts: `weekly_demand.parquet` = **`5,067` rows** (`5,147 - 95` excluded closure rows `+ 15` tail zero-fill rows for `21829` and `22740`); `build_features` = **`3,467` train rows** (`35` target weeks) and **`800` test rows** (`100` products × `8` weeks).

---

## CR-003: G3 Prior-Year Window Truncation for Target Week `2011-12-05` and Boundary Test Weeks

- **Date:** 2026-11-21
- **Status:** Proposed (Task M8b audit)
- **Affected Files:** `src/features.py`, `CONTRACTS.md` (C10: `prior_year_weekly.parquet`), `data/processed/next_week_forecast.csv`
- **Contracts Affected:** `CONTRACTS.md` C10

### Discrepancy Observed
1. **Prior-Year Coverage Limit:**
   Under Contract C10, `prior_year_weekly.parquet` only contains weeks strictly before `2010-12-06` (derived from Online Retail II `Year 2009-2010`). Its latest record is Monday `2010-11-29`.
2. **Window Calculation for Forecast Week (`2011-12-05`):**
   Group G3 computes features across the 5-week window $t-54$ to $t-50$ weeks relative to target week $t$.
   For target week `2011-12-05`:
   - $t-54$: `2010-11-22` (present in C10)
   - $t-53$: `2010-11-29` (present in C10)
   - $t-52$: `2010-12-06` (**missing from C10**; in primary dataset C1)
   - $t-51$: `2010-12-13` (**missing from C10**; in primary dataset C1)
   - $t-50$: `2010-12-20` (**missing from C10**; in primary dataset C1)
3. **Impact of 3-Value Minimum Rule:**
   `src/features.py` requires at least 3 valid prior-year observations (`len(valid) >= 3`) to compute `yoy_mean_5w` and `yoy_seasonal_index`. Because only 2 weeks (`2010-11-22` and `2010-11-29`) are present in C10, `yoy_mean_5w` evaluates to `NaN` and `yoy_available` evaluates to `0.0` for **100% of products** on the exact forecast week `2011-12-05`.
4. **Boundary Test Weeks:**
   - Test week `2011-11-21` has 4 weeks in C10.
   - Test week `2011-11-28` has exactly 3 weeks in C10 (`2010-11-15`, `2010-11-22`, `2010-11-29`), so any missing prior-year sales will cause it to drop below the 3-value threshold.

### Proposed Resolution
Rather than silently modifying C10, we propose:
- In `build_features` / `build_future_features`, when evaluating weeks where $t-54 \dots t-50$ overlaps with `2010-12-06` or later, stitch the observations from the primary cleaned dataset `data/processed/weekly_demand.parquet` (C1) for weeks `2010-12-06` through `2010-12-20` into the prior-year lookup.
- Because these historical weeks are strictly in the training period (past data before target $t$), stitching them from C1 introduces zero data leakage while restoring full 5-week window coverage for forecast week `2011-12-05`.

---

## CR-004: Confirmation of `total_units` and `pattern_class` in `GET /api/products`

- **Date:** 2026-11-21
- **Status:** Ratified in CONTRACTS.md (C7) and CONTRACTS_ADDENDUM.md
- **Affected Spec:** `CONTRACTS.md` Section C7 (`GET /api/products`), `CONTRACTS_ADDENDUM.md` Section C7
- **Implementation:** Task B10 (`api/main.py`, `api/schemas.py`)

### Summary
- `GET /api/products` response contract is confirmed as `[{product_id, description, total_units, pattern_class}]`.
- `total_units` is the integer sum of sales units across the training period (`week_start < 2011-10-10`). Products are ordered descending by `total_units`.
- `pattern_class` is sourced from `data/processed/product_intelligence.csv` (`smooth`, `erratic`, `intermittent`, `lumpy`) or null if not yet classified.
- Both fields are already documented in `CONTRACTS.md` (line 136) and `CONTRACTS_ADDENDUM.md` (line 52).

