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
