---
trigger: always_on
description: "Contract compliance rules for data files and API endpoints."
---

# Contract Rules

- All produced files and API endpoints must match `CONTRACTS.md` exactly (column names, types, and order).
- Never silently alter a schema or endpoint contract; log any proposed contract change in `CHANGE_REQUESTS.md` and wait for team agreement.

