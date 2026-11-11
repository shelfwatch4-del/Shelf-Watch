---
trigger: always_on
description: "Frozen project design decisions for Shelfwatch."
---

# Frozen Design Decisions

- **Granularity & Weeks:** Weekly granularity with Monday-start weeks (`week_start`).
- **Target & Horizon:** Target is `units` per product per week; forecast horizon is next week (`t+1`).
- **Product Scope:** Top 100 products selected on the **training period only**.
- **Valid Weeks:** Full weeks only, from `2010-12-06` to `2011-11-28` (52 weeks total).
- **Split:** Train = first 44 full weeks; Test = last 8 full weeks (`week_start` `2011-10-10` to `2011-11-28`).
- **Forecast Target:** Next-week forecast target week is `2011-12-05`.
- **Reproducibility:** Use random seed `42` everywhere.
- **Shop Closures:** Shop-closed week `2010-12-27` is **MISSING** data, never zero demand.

