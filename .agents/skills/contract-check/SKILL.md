---
name: contract-check
description: >-
  Compare files in data/processed/ and API responses against CONTRACTS.md and docs/SRS.md section 7. Use when the user invokes /contract-check or asks to validate data and API contracts.
---

# Contract Check

Compare the files in `data/processed/` and the API responses against `CONTRACTS.md` and `docs/SRS.md` section 7. Do **not** edit any code.

## Checks

1. Verify every file in `data/processed/` against its schema in `CONTRACTS.md`: column names, data types, column order, sorting, and row/null rules.
2. Verify API endpoints and responses against `CONTRACTS.md` and `docs/SRS.md` section 7: status codes, JSON field names, types, and error/empty-state shapes.

## Output

Report every mismatch found (column names, types, order, status codes, or field names), or state clearly that all checked files and endpoints match the contracts.

