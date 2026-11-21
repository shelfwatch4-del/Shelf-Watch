# PROMPTS_00_INDEX.md: Start here (version 3)

This replaces the older prompt order. You have finished **D1 to D9** (data) and **M1 to M5** (features and first models). Everything below continues from there. The SRS is version 3.0 (the Claude document). Export it as Markdown and replace `docs/SRS.md` before you start (click the document name, then Export, then Markdown). **Do not let the agent edit `docs/SRS.md` by hand;** the document is the source of truth.

## What changed in version 3 and why
| Change | Reason |
|---|---|
| Prior-year and calendar features, derived behaviour features, transaction-derived signals | Mentor: derive new features and use other data. Also addresses the one-year limitation |
| Product intelligence (demand pattern, ABC and XYZ, lifecycle, returns, exports, bought-together) | New information from the data, not only a forecast |
| Prediction intervals, replenishment advice, revenue at risk, stockout probability | Actionable output for stakeholders |
| MySQL database, CRUD, middleware, Python client | Mentor's class topics |
| Five separate pages (not a single-page app), interactive Predict page | Mentor: not a single-page app; show predict output |
| Kaggle publishing; local FastAPI + ngrok public URL + Postman tests | Kaggle cannot serve a live URL |
| Stakeholders and real-life use | Mentor: say who uses it and where |

## Primary stakeholder (decided)
The whole project is built around **one user: the Retail Inventory and Replenishment Manager** (SRS 1.6, `docs/STAKEHOLDER.md`). When a prompt says "the manager", this is who it means. Every page supports one step of their weekly routine. New in this decision: the `action` label (ORDER_NOW, ORDER_SOON, REVIEW_STOP, HOLD), the Home Action list, and the replenishment-policy backtest (task **B5b**), which measures whether the system helps the manager.

## Windows preface (paste at the top of EVERY prompt)
```
Environment: Windows. Use venv\Scripts\python.exe and venv\Scripts\pytest.exe for all commands. Project root is C:\ShelfWatch. Follow the workspace rules in .agent/rules. Do one task only. Do not change files outside this task's scope. Do not edit docs/SRS.md. Contract changes go in CHANGE_REQUESTS.md, never silently.
```

## Closing check (paste at the end of every prompt that lacks one)
```
Before you finish: run the tests for this task, show the exact commands and output, list every file you created or changed, and list anything in the task you did NOT implement. Do not claim success unless tests ran and passed. Do not invent numbers: print them from files.
```

## Old to new task IDs
| Old | New |
|---|---|
| ML M6 tuning | M7 (now after feature groups) |
| ML M7 ablation | M8 |
| ML M8 level shift | M9 (merged with ensemble) |
| ML M9 artifacts | M11 |
| ML M10 result files | M12 |
| ML M11 notebook and card | M13 |
| ML M12 stretch | M14 |
| Backend B1 to B14 | B1 to B21 (MySQL, CRUD, middleware, plan workflow, analysis, aliases) |
| Frontend F1 to F12 | F1 to F13 (five pages, Analysis board, plan workflow) |
| DevOps Q1 to Q11 | Q1 to Q13 (local ngrok deployment, Postman, evaluation rehearsal) |

## Order of execution
| Step | File | Task | Needs |
|---|---|---|---|
| 1 | Data | **E0** read-only check of Online Retail II | Dataset downloaded |
| 2 | ML | **M5b** diagnose the M5 results (read-only) | M5 done |
| 3 | All | Merge `CONTRACTS_ADDENDUM.md` (task E7, first half) | Step 1 |
| 4 | Data | E1, E2, E3 (calendar, prior-year, enrichment tables) | E0 |
| 5 | Backend | B1 to B8 (risk engine, MySQL layer, CRUD); install MySQL and run `sql/schema.sql` first | Mocks only |
| 6 | Frontend | F1, F2 (shell, `api.js`, mocks) | Mocks only |
| 7 | ML | M6 feature groups, then M7 tuning, M8 ablation | E1 to E3 |
| 8 | Data | E4, E5 (intelligence and basket pairs) | In parallel with ML |
| 9 | ML | M9, M10, M11, M12 (diagnostics, quantiles, artifacts, result files) | M8 |
| 10 | Data | D8 error analysis | M12 |
| 11 | Backend | **B5b** policy backtest, **B6b** load MySQL, then B9 to B18 (API, middleware, client, plans, analysis, explain, root aliases) | M12, E4, E5 |
| 12 | Frontend | F3 to F12 (five pages, no login; F7 is the Analysis board) | B9 |
| 13 | DevOps | Q5 to Q8 (start scripts, ngrok, Postman, Kaggle) | Everything above |
| 14 | DevOps | Q9 to Q13 (review, report, slides, viva, 150-minute evaluation rehearsal using `docs/EVALUATION_PLAN.md`) | Everything above |

**Cut line if you run late:** drop in this order: M14, M9 ensemble, E5 bought-together, G4 features. Never cut: the public URL (ngrok) and Postman collection, the Predict page, the Analysis board, MySQL, CRUD with database, middleware, five pages, Kaggle links, stakeholders and real-life use, JS guide.

## Paste these three first
1. **E0** from `PROMPTS_DATA.md` (read-only).
2. **M5b** from `PROMPTS_ML.md` (read-only).
3. This merge prompt (Fast mode):
```
Read CONTRACTS_ADDENDUM.md. Merge its contents into CONTRACTS.md as additive sections: amend C1, C4 and C7 where the addendum says so, and add C8 to C13. Do not delete existing text. Add a change-log row. Do not change any code in this task. Show me the diff summary.
```

## Rules that keep the agent on track
- One task per conversation, in Planning mode, and read the plan before approving.
- Do not paste one giant prompt covering many tasks; the agent will sprawl.
- If the agent invents a number, ask it to print it from the file.
- Every new feature must be evaluated (ablation) before it is kept. A feature that does not help is still reported.
