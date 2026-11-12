# PROMPTS_DEVOPS_QA.md: Environment, OS Setup, CI, Deployment, QA and Delivery

**Owner:** Member 4 leads, every member reviews. Tasks Q1 to Q3 run early (Day 1 to 3); Q4 to Q7 after the API works; Q8 to Q11 at the end.
**Setup first:** Phase 0 in `PROMPTS.md`. **Mode:** Planning.

**Closing check (add to any prompt that lacks it):**
```
Before you finish: run the commands you describe, show exact commands and output, list every file you created or changed, and list anything in the SRS sections I named that you did NOT implement. Do not claim success unless it actually ran.
```

---

## Part 1: Environment and operating systems

## Task Q1: Cross-OS environment and setup scripts

**SRS:** NFR-7, C-6 | **Output:** repeatable setup on Windows, macOS and Linux

```
Read docs/SRS.md NFR-7 and section 2.4. The four team members use different operating systems.

1. Write scripts/setup.sh (macOS and Linux) and scripts/setup.ps1 (Windows PowerShell) that: check Python is 3.10 or 3.11 and stop with a clear message otherwise; create .venv; upgrade pip; install requirements.txt; print the installed versions of pandas, numpy, scikit-learn, lightgbm, fastapi; run pytest -q on the mock data. Do not require admin rights.
2. Create .gitattributes forcing LF line endings for .py, .md, .json, .html, .js, .css, .sh and CRLF for .ps1, and marking .parquet, .joblib, .png as binary.
3. Review all Python code for OS-specific problems: use pathlib everywhere, no hard-coded slashes or absolute paths, open text files with encoding="utf-8", no shell-only commands inside Python, no fork-only multiprocessing (set n_jobs conservatively). Fix what you find and list it.
4. Add a README section "Troubleshooting" covering: LightGBM or pyarrow install failures on Windows (use the pre-built wheels and the exact pip command), "python not found" (py -3.11 on Windows), PowerShell execution policy for activating the venv, and Mac Apple Silicon architecture issues.
5. Add scripts/doctor.py that prints OS, Python version, library versions compared with requirements.txt, whether data/raw/Online Retail.xlsx exists, whether data/processed and models artifacts exist, and a PASS or FAIL for each.
```
**Done when:** a teammate on a different OS runs the setup script and `python scripts/doctor.py` shows all PASS.

---

## Task Q2: Continuous integration

**SRS:** section 10.3 (quality gates)

```
Create .github/workflows/ci.yml that on every push and pull request: runs on ubuntu-latest and windows-latest with Python 3.10 and 3.11 (matrix); installs requirements.txt with pip caching; runs pytest -q against data/mock (the real data is not committed); runs python -m compileall src api; and fails on any error. Add a second job that runs a markdown link check or a simple lint (ruff, only error-level rules E9, F63, F7, F82) on src, api and tests. Add a status badge to README.md. Explain every step in comments. Do not add secrets.
```
**Done when:** a pull request shows green checks on both operating systems.

---

## Task Q3: Pre-merge review automation

```
Create .agents/workflows entries (or documented prompts in docs/REVIEW_GUIDE.md if workflows are unavailable) named pre-merge and contract-check:
- pre-merge: run /verify, then /leakage-review on changed files in src/, then /contract-check, then print a table: check, result, details. The merge is blocked unless all rows pass.
- contract-check: compare data/processed files and the running API against CONTRACTS.md and SRS section 7 and report every mismatch.
Also create a pull request template (.github/pull_request_template.md) with checkboxes: tests pass, contract unchanged or CHANGE_REQUESTS updated, no data leakage, docs updated, reviewed by a teammate.
```

---

## Part 2: Integration, deployment and release

## Task Q4: End-to-end smoke test script

**SRS:** section 10.1 (system test) | **Output:** `scripts/smoke.py`

```
  The closure week 2010-12-27 has no row in weekly_demand. In /forecast history, include it as an entry with units null (so the history has 52 weeks) and draw the chart line with a gap there. Smoke tests expect 52 history entries and one null.

Write scripts/smoke.py that takes a base URL (default http://localhost:8000) and checks: GET /health is ok; GET /disclaimer text contains "simulated"; GET /products returns 100 items ordered by total_units descending; GET /forecast/{first_product_id} returns 52 history rows, 8 test predictions and a forecast for 2011-12-05; GET /forecast/NOPE returns 404; POST /risk with defaults returns summary counts that add up to 100; POST /risk with lead_time_weeks 9 returns 422; POST /sensitivity returns 12 rows; GET /metrics includes a baseline; GET /error-analysis has segments, weeks and products; GET / returns HTML containing "Shelfwatch"; no response contains "CustomerID". Print a table of checks with PASS or FAIL and exit non-zero on any failure. It must work against a local server and a public URL.
```
**Done when:** all checks PASS locally.

## Task Q5: Docker and cloud configuration (one service)

**SRS:** section 11, Option A

```
Read docs/SRS.md section 11. Prepare Option A, where FastAPI serves both the API and the dashboard from one app.

1. Dockerfile: python:3.11-slim; set PYTHONUNBUFFERED; install requirements.txt (pinned); copy only api/, src/, dashboard/ (excluding dashboard/prototype and dashboard/mock), models/ and data/processed/; run as a non-root user; honor the PORT environment variable; CMD uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}; HEALTHCHECK calling /health.
2. .dockerignore excluding data/raw, notebooks, reports, tests, .git, .venv, __pycache__, *.xlsx.
3. Report the image size and the size of each copied folder; the artifacts must total under 50 MB (NFR-10).
4. Confirm the container starts with no data/raw folder and serves /health, /docs and the dashboard.
5. render.yaml (or exact dashboard settings documented in README) for Render: runtime Docker, health check path /health, environment variable ALLOWED_ORIGINS optional. Document the equivalent for a Hugging Face Docker Space (port 7860 handling).
6. README section "Deploy" with numbered steps and what to click.

Do not deploy anything yourself. Tell me exactly what I must do in the hosting service.
```
**Done when:** `docker build` and `docker run -p 8000:8000` work locally (or the agent states clearly that Docker is not installed and provides the commands).

## Task Q6: Hello-world deploy by Day 8

**SRS:** section 11.3 recommendation

```
Create a minimal deployment test: a branch deploy-smoke containing a tiny FastAPI app (api/hello.py) that serves GET /health and a static index page, using the same Dockerfile pattern. Give me the exact steps to deploy it on Render (or Hugging Face Spaces), and a checklist to confirm the public URL works. Then give me the steps to switch the service to the real app (api.main:app) once ready. The goal is to discover free-tier or port problems early.
```
**Done when:** a public `/health` URL responds.

## Task Q7: Release checklist and dry run

**SRS:** section 11.4, Appendix C

```
Create docs/RELEASE_CHECKLIST.md from SRS section 11.4 and Appendix C with checkboxes. Then do a dry run in a fresh clone in a temporary folder: create the venv with scripts/setup.sh, run python -m src.data_prep (skip if the raw file is unavailable and say so), python -m src.train, pytest -q, uvicorn, then python scripts/smoke.py. Tick only what you actually verified and write the evidence (command and result) next to each ticked line. List everything that failed or was skipped.
```
**Done when:** the dry run passes or every failure is listed.

---

## Part 3: Final QA, report and viva

## Task Q8: Whole-project review and traceability

```
Act as a strict examiner and senior engineer. Read docs/SRS.md and review the whole repository against it. Write docs/REVIEW.md with: (1) a traceability table mapping every FR and NFR to the implementing file and the verifying test, marking any requirement with no implementation or no test; (2) deviations from the SRS with severity; (3) data-leakage risks (run /leakage-review on src/features.py and src/train.py); (4) every place where the simulated-inventory disclaimer is missing or weak; (5) anything in code, dashboard text or documentation that overstates what the results show. Do not edit code; only report.
```

## Task Q9: Report draft

```
Using docs/SRS.md, docs/DATA_NOTES.md, docs/MODEL_CARD.md, cleaning_log.csv, metrics.csv, error_analysis.csv, the sensitivity results and the figures in reports/figures/, draft docs/REPORT.md for college submission with sections: Problem and objectives; Dataset and cleaning (real row counts); EDA findings; Feature engineering (explain shift-before-rolling and the closed-week rule); Models and evaluation (baselines versus ML with real numbers); Forecast-error analysis; Inventory simulation (every assumption plus the disclaimer); System design (API and dashboard); Testing; Deployment; Limitations (one year of data, level shift, intermittent demand, simulated stock); Future work. Use only numbers that exist in project files; write TODO instead of inventing any number. Cite figure file names where figures belong.
```

## Task Q10: Slides

```
From docs/REPORT.md create docs/SLIDES.md with at most 12 slides. Each slide: a title, 3 to 5 short bullets, the figure to show (file name), and a speaker note of 2 sentences. Include one slide on the Shelfwatch dashboard (what the shelf wall shows) and one slide on honest limitations. Order: problem, data, method, results, error analysis, inventory rule, dashboard, deployment, limitations, conclusion.
```

## Task Q11: Viva rehearsal

```
Act as a college project examiner. Ask me 12 hard questions one at a time, covering: why weekly granularity; data leakage and how we prevented it; shift before rolling; the chronological split; why WAPE and not MAPE; the closure week; why tree models can under-forecast the October to November level; how sigma_error is derived; safety stock and reorder point in plain words; why inventory is simulated and what that means for conclusions; how the API and the dashboard connect; why a static dashboard served by FastAPI. After each of my answers grade it out of 10, say what a stronger answer would include, then ask the next question. Wait for my answer each time. At the end give a score summary and my three weakest topics.
```

---

## Appendix: Recovery prompts

**Agent drifted from the SRS**
```
Stop. Compare what you just did with docs/SRS.md section [X]. List each deviation, undo the deviations, and redo the work to match the SRS. If you believe the SRS is wrong, say so and wait.
```

**Environment broken on one OS**
```
Run python scripts/doctor.py and show the output. Identify the first failing check, explain the cause in two sentences, propose the smallest fix for [Windows | macOS | Linux], and apply it only after I confirm. Do not delete the virtual environment unless I say so.
```

**CI passes locally but fails in GitHub Actions**
```
Here is the failing CI log: [paste]. Reproduce the failure locally using the same Python version and operating system settings if possible, explain the cause, fix it, and show the passing run. Typical causes to check first: files that exist locally but are git-ignored, absolute paths, line endings, and missing pinned versions.
```
