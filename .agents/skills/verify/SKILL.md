---
name: verify
description: >-
  Run the pytest suite, print the output, diagnose and fix any failures, rerun until green, and list changed files. Use when the user invokes /verify or asks to verify tests.
---

# Verify

1. Run `pytest -q` and print the full command output.
2. If any test fails:
   - Diagnose the root cause.
   - Fix the issue.
   - Rerun `pytest -q` and repeat until all tests pass, or stop if a design or contract decision from the user is required.
3. Once green (or stopped for a user decision), list all files changed.

