---
trigger: always_on
description: "Python coding style, library choices, and testing rules."
---

# Python Style Rules

- Follow PEP 8 across all Python files.
- Add type hints and docstrings on all public functions.
- Use `pandas`, not `polars`.
- Write pure functions where possible; avoid hidden global state.
- Every Python module must have corresponding `pytest` tests in `tests/`.

