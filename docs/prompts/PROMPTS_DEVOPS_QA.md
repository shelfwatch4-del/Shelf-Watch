# PROMPTS_DEVOPS_QA.md: Environment, CI, Deployment, Kaggle, QA and Delivery, version 3

**Owner:** Member 4 leads; every member reviews. Q1 to Q4 early; Q5 to Q8 after the API works; Q9 to Q12 at the end.
**Setup:** paste the Windows preface and closing check from `PROMPTS_00_INDEX.md` into every prompt. Planning mode.

---

## Part 1: Environment and automation

## Task Q1: Cross-OS setup (skip what already exists)
```
Check whether scripts/setup.sh, scripts/setup.ps1, scripts/doctor.py and .gitattributes already exist. For anything missing: setup scripts that check Python 3.10 or 3.11, create the venv, install requirements.txt, print key library versions (pandas, numpy, scikit-learn, lightgbm, fastapi, sqlalchemy, holidays, httpx) and run pytest -q on the mock data; a .gitattributes forcing LF for text files and CRLF for .ps1; scripts/doctor.py printing OS, Python, library versions versus requirements.txt, and PASS or FAIL for the raw file, the processed tables, the model files and the database. Review code for OS-specific problems (pathlib, utf-8, no hard-coded separators). Add a README "Troubleshooting" section for Windows (py -3.11, PowerShell execution policy, LightGBM or pyarrow wheels).
```

## Task Q2: Continuous integration
```
Create .github/workflows/ci.yml: on every push and pull request, matrix ubuntu-latest and windows-latest with Python 3.10 and 3.11; install requirements.txt with pip caching; run pytest -q against data/mock (real data is not committed; tests that need real files must skip with a clear message); run python -m compileall src api; run ruff on src, api and tests with only error-level rules (E9, F63, F7, F82). Add a status badge to README.md. Explain each step in comments. No secrets.
```

## Task Q3: Pre-merge checks
```
Create the pre-merge and contract-check workflows (or docs/REVIEW_GUIDE.md if workflows are unavailable): pre-merge runs /verify, /leakage-review on changed files in src/, /contract-check against CONTRACTS.md and the addendum, then prints a table (check, result, details). Add .github/pull_request_template.md with checkboxes: tests pass, contracts unchanged or CHANGE_REQUESTS updated, no data leakage, docs updated, reviewed by a teammate.
```

## Task Q4: End-to-end smoke test
```
Write scripts/smoke.py taking a base URL (default http://localhost:8000) that checks: GET /api/health ok; /api/disclaimer contains "simulated"; /api/products has 100 items ordered by total_units; /api/forecast/{first id} has 52 history entries with exactly one null and a forecast for 2011-12-05 with p10 <= forecast <= p90; /api/forecast/NOPE is 404; POST /api/predict works; POST /api/predict/custom works and rejects 7 values with 422; POST /api/risk counts add up to 100; lead_time_weeks 9 gives 422; /api/simulate works; /api/sensitivity returns 12 rows; /api/metrics includes a baseline; /api/error-analysis and /api/intelligence respond; a write without X-API-Key gives 401 and with the key succeeds (create then delete an order); /api/logs shows the requests just made; GET / returns HTML containing "Shelfwatch"; no response contains "CustomerID". Print PASS or FAIL per check and exit non-zero on any failure. It must work against a local server and a public URL.
```

---

## Part 2: Deployment and Kaggle

## Task Q5: Local deployment, ngrok and Postman (mentor's flow)
**SRS:** 11.1 to 11.6

```
Read docs/SRS.md section 11, docs/DEPLOY_LOCAL_NGROK.md and postman/Shelfwatch.postman_collection.json.
1. Verify scripts/start_server.bat and scripts/start_ngrok.bat run from a clean terminal on Windows: they activate the venv, load .env, start uvicorn on 8000 (and ngrok http 8000); fix paths if needed.
2. Write scripts/doctor.py: checks Python version, pinned packages, .env present, MySQL reachable, every table in sql/schema.sql exists with rows where expected, models/ files present, GET /health returns ok. Exit code 1 on any failure with a plain message.
3. Run the Postman collection with newman (or describe each request for manual run) against the LOCAL environment and then against the PUBLIC environment (ngrok URL pasted into the environment); every test must pass; save a screenshot or report in docs/screens/.
4. Check that header ngrok-skip-browser-warning is sent by the collection and the pages; check CORS for the ngrok origin.
5. README "Deploy" with numbered steps; state clearly: the public URL works only while the laptop, MySQL, FastAPI and ngrok are all running, the free ngrok URL changes on restart, and a mysqldump backup (scripts/backup_db.bat) is kept on a second laptop.
Do not start long-running servers yourself in a way that blocks; give me the exact commands.
```

## Task Q6: Early smoke test of the public URL
```
Do this as soon as B9 exists: start the minimal API, run ngrok, call GET /health from Postman and from a phone on mobile data, and record the steps and any firewall or antivirus prompt in docs/DEPLOY_LOCAL_NGROK.md. Then repeat with the full app after B21.
```

## Task Q7: Prepare the Kaggle publication
**SRS:** 11.5, FR-K-1 to FR-K-5

```
Read docs/SRS.md section 11.5. Prepare, but do NOT upload, the Kaggle material in kaggle/:
1. kaggle/dataset/: the processed files only (weekly_demand, closed_weeks, calendar_weekly, prior_year_weekly, enrichment_weekly, product_intelligence, basket_pairs, product_segments, plus a small README), a dataset-metadata.json draft with title, subtitle, description, license notes and the UCI credit and link to the raw data (the raw Excel is linked, not republished), and a DATA_CARD.md describing every file and column.
2. kaggle/notebook/: shelfwatch_modeling.py (from task M13) converted to a notebook-friendly script or .ipynb, reading one DATA_DIR variable; a short markdown intro stating the problem, the data, the method and the honest results.
3. kaggle/model/: best_model.joblib, q10.joblib, q90.joblib, feature_list.json, config.json, MODEL_CARD.md (copy from docs).
4. kaggle/PUBLISH_STEPS.md: step-by-step instructions for the Kaggle website (create the dataset, create and run the notebook with the dataset attached, make it public, create the model and upload files, add the model card), what to copy back into the environment variables KAGGLE_*_URL, and a checklist to confirm the notebook runs top to bottom on Kaggle.
Include one paragraph explaining why Kaggle is used for publishing and reproducing and not for the live URL (a notebook cannot accept incoming connections, sessions end after hours, tunnel URLs change).
```
**Done when:** `kaggle/` is complete and you can follow `PUBLISH_STEPS.md` on kaggle.com. After uploading, paste the three URLs into the deployment environment.

## Task Q8: Release checklist and dry run
```
Create docs/RELEASE_CHECKLIST.md from SRS 11.4 and Appendix C with checkboxes. Do a dry run in a fresh clone in a temporary folder: create the venv with the setup script, run the pipeline commands from SRS 14.4 that can run (skip any that need the raw files and say so), pytest -q, start uvicorn, run python scripts/smoke.py and python scripts/api_client.py demo. Tick only what you verified, with the command and result next to each ticked line. List everything that failed or was skipped.
```

---

## Part 3: Review, report and viva

## Task Q9: Whole-project review and traceability
```
Act as a strict examiner and senior engineer. Read docs/SRS.md (version 3) and review the repository. Write docs/REVIEW.md with: (1) a traceability table mapping every FR and NFR to its implementing file and verifying test, marking any with none; (2) a second table mapping each row of SRS Appendix E (the mentor requirements) to evidence in the repository; (3) deviations from the SRS with severity; (4) leakage risks (run /leakage-review on src/features.py, src/enrichment.py and src/train.py); (5) every place the simulated-inventory disclaimer is missing or weak; (6) any page, endpoint or text that addresses a stakeholder other than the Retail Inventory and Replenishment Manager, or that is not tied to a step of their weekly routine; (7) any text in the code, pages or docs that overstates what the results show or claims uniqueness or company usage without a source. Do not edit code; only report.
```

## Task Q10: Report
```
Using docs/SRS.md, docs/DATA_NOTES.md, docs/MODEL_CARD.md, docs/E0_FINDINGS.md, docs/M5_DIAGNOSIS.md, the cleaning log, metrics.csv, ablation.json, interval_coverage.json, error_analysis.csv, the sensitivity results and the figures in reports/figures/, draft docs/REPORT.md with sections: Problem and objectives; The primary stakeholder: the Retail Inventory and Replenishment Manager (persona, weekly routine, decisions, success measures; say plainly that the persona is assumed; add a 'Persona validation' paragraph if anyone was interviewed) and real-life use; the replenishment-policy backtest results as the evidence of benefit; Dataset, cleaning and enrichment (real row counts; the closure week; the zero-fill decision; Online Retail II and the holidays package with citations); EDA and product intelligence findings (state class distributions honestly); Feature engineering (shift before rolling, feature groups); How we train and test (chronological split, week-based validation, why not random, why the test period is the last 8 weeks); Models, tuning and the feature-group ablation (real numbers, including what did NOT help); Prediction intervals and their measured coverage; Forecast-error analysis; Inventory simulation, user-entered stock and replenishment advice (every assumption and the disclaimer); System design (database, CRUD, middleware, Python client, five pages); Testing; Deployment and Kaggle (what is live and what is published; the free-tier reset); Limitations; Future work. Use only numbers from project files; write TODO instead of inventing any number.
```

## Task Q11: Slides and the mentor demo script
```
From docs/REPORT.md create docs/SLIDES.md with at most 14 slides (title, 3 to 5 bullets, figure file name, 2-sentence speaker note): problem and stakeholders, data and enrichment, method, results including what did not help, intervals, error analysis, inventory rule and advice, product intelligence, the five pages, backend (database, CRUD, middleware, client), deployment and Kaggle, limitations, conclusion.
Also write docs/DEMO_SCRIPT.md: a 6-minute live demo told from the manager's point of view (open with one sentence on who they are and what they must decide each week), in order: open the public URL and show /api/health and /docs; the shelf and a slider; the Predict page for a product and for typed-in sales; the what-if simulator; create and approve an order on the Inventory page and show the risk change; the Analysis board request log showing the middleware at work; the Kaggle dataset, notebook and model pages. For each step write what to say, what to click, and what to do if it fails (a fallback screenshot name).
```

## Task Q12: Viva rehearsal
```
Act as a strict college examiner. Ask me 15 hard questions one at a time and wait for my answer each time. Cover: who the single stakeholder is, why we chose them, which features serve each step of their weekly routine, what we deliberately did not build for other roles, and what the persona assumption rests on; why weekly granularity; how training and testing are split and why not random; data leakage and shift before rolling; the closure week and the zero-fill decision; why WAPE and not MAPE; why the moving average beat default LightGBM and what we did about it; what each new feature group is, why it might help, and how we tested it; prior-year features and their limits; prediction intervals and coverage; safety stock, reorder point and stockout probability in plain words; why inventory is simulated and what changes when a user enters stock; the database tables and each CRUD status code; what each middleware does; how Python calls this API; how JavaScript fetch and events work (ask me to trace a click on a bin); why Kaggle is for publishing and ngrok for the live URL; the MySQL reset (scripts/reset_demo.py) on free hosting. After each answer grade it out of 10, say what a stronger answer includes, then ask the next question. At the end give my score and my three weakest topics.
```

---

## Appendix: Recovery prompts
**Agent drifted from the SRS**
```
Stop. Compare what you just did with docs/SRS.md section [X]. List each deviation, undo it, and redo the work to match the SRS. If you believe the SRS is wrong, say so and wait.
```
**Environment broken**
```
Run python scripts/doctor.py and show the output. Identify the first failing check, explain the cause in two sentences, propose the smallest fix, and apply it only after I confirm. Do not delete the virtual environment unless I say so.
```
**CI fails but local passes**
```
Here is the failing CI log: [paste]. Reproduce it locally with the same Python version if possible, explain the cause, fix it and show the passing run. Check first: files that exist locally but are git-ignored, absolute paths, line endings, unpinned versions, tests that need real data.
```

## Task Q13: Evaluation rehearsal (added in version 3.2)
```
Read docs/EVALUATION_PLAN.md. Run a full 150-minute rehearsal with the four roles: pre-flight (doctor.py, mysqldump, reset_demo.py, ngrok URL pasted into Postman), then the segments in order, with one member playing a faculty member asking the likely questions. Also run a 30-minute soak test: call /api/risk and /api/predict every 10 seconds through the ngrok URL and confirm no errors and no memory growth. Record every failure and fix it. Output: a short checklist of what to do in the first five minutes of the real evaluation and a fallback table (laptop sleeps, ngrok URL changes, MySQL not running, internet down).
```
