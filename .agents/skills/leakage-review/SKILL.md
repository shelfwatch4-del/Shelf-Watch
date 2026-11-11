---
name: leakage-review
description: >-
  Audit named code for time-series and train/test data leakage against Shelfwatch rules. Use when the user invokes /leakage-review or asks to review code for data leakage.
---

# Leakage Review

Review the code named by the user for data leakage. Do **not** edit any code unless the user explicitly tells you to.

## Checks

1. Every lag and rolling feature uses `shift(1)` before rolling.
2. No statistic (mean, std, imputation value, outlier cap, or product selection) is computed using test-period rows.
3. Training weeks strictly precede test weeks (`week_start` < `2011-10-10` for train; `2011-10-10` to `2011-11-28` for test).
4. The shop-closed week `2010-12-27` is never a target row, and lag/rolling calculations skip it as missing data.

## Output Format

Output a markdown table with these columns:

| Finding | File and Line | Severity | Suggested Fix |
|---|---|---|---|

