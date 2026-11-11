# PROMPTS_ML.md: Machine Learning (Member 2)

**You own:** `src/features.py`, `src/train.py`, `models/*`, `data/processed/predictions.csv`, `metrics.csv`, `next_week_forecast.csv`, `notebooks/02_modeling.py`, `docs/MODEL_CARD.md`.
**Inputs from Member 1:** `weekly_demand.parquet`, `closed_weeks.csv`, `product_segments.csv` (use `data/mock/` until they exist).
**Hands off to:** Member 1 (error analysis), Member 3 (risk engine and API).
**Setup first:** Phase 0 in `PROMPTS.md`. Use the **strongest available model** and **Planning mode** for every task here.

**Rules that apply to every task (already in `.agents/rules/02-ml-rules.md`):** chronological split only; `shift(1)` before rolling; tune only inside the training period; never use the test set for tuning, imputation, scaling, capping or selection; metrics = MAE, RMSE, WAPE, bias (never MAPE); clip predictions at 0; seed 42.

**Closing check (add to any prompt that lacks it):**
```
Before you finish: run the tests for this task, show exact commands and output, list every file you created or changed, and list anything in the SRS sections I named that you did NOT implement. Do not claim success unless tests ran and passed.
```

**Key dates:** training target weeks `< 2011-10-10`; test = 8 weeks `2011-10-10` to `2011-11-28`; forecast target week `2011-12-05`; closed week `2010-12-27` (missing data, not zero demand).

---

## Task M1: Lag and rolling features

**SRS:** FR-F-1, FR-F-2, FR-F-7 | **Output:** first part of `src/features.py`

```
Read docs/SRS.md section 5.1 and PROJECT_CONTEXT.md section 6. Input: data/processed/weekly_demand.parquet and closed_weeks.csv.

Create src/features.py with build_features(weekly: pd.DataFrame, closed_weeks) -> pd.DataFrame returning one row per product per TARGET week t, with columns product_id, week_start, target (= units at t) and features.

In this task implement only:
- lag_1, lag_2, lag_3, lag_4 (units at t-1 .. t-4)
- roll_mean_4, roll_std_4, roll_mean_8, roll_std_8: ALWAYS apply shift(1) BEFORE rolling, so week t is never inside its own window. Use min_periods=2.

Closed week 2010-12-27: set units to NaN for that week BEFORE building any feature. It must never appear as a target row, and lags and rolling windows must skip over it (use the previous non-missing weeks), not treat it as 0.

Drop rows that lack enough history for the longest window and print how many rows were dropped per product and in total.

Write a pytest test now: build features, change units for one product at week t and every later week, rebuild, and assert all features at week t are identical (the leakage test). Name it test_no_leakage_from_future.
```
**Done when:** the leakage test passes.

---

## Task M2: Calendar, commercial and product-history features

**SRS:** FR-F-3 to FR-F-5, FR-F-9

```
Extend src/features.py (do not change the lag and rolling code or its tests).

Add, all computed only from information strictly before target week t:
- week_of_year, month (of the target week's week_start; these are known in advance)
- prev_avg_price, prev_orders, prev_customers: the previous week's values (shift(1)); for the closed week's successor use the last non-missing value
- prod_past_mean, prod_past_std: expanding mean and std of the product's units over weeks strictly before t (shift(1) before expanding)

Define module-level lists:
- FEATURE_COLUMNS: the ordered default features (lags, rolling, week_of_year, month, prev_avg_price, prev_orders, prev_customers, prod_past_mean, prod_past_std)
- OPTIONAL_FEATURES = ["weeks_to_christmas"] (weeks between week_start and 2011-12-25, a time-index proxy); NOT in FEATURE_COLUMNS because with one year of data it behaves like a row counter and will be evaluated by ablation.

Also implement build_future_features(weekly, closed_weeks, target_week) -> one row per product for a week with NO target (used for the 2011-12-05 forecast): same features, computed from weeks before target_week only.

Tests: no feature uses same-week units, orders, customers or price; FEATURE_COLUMNS equals the columns produced (same order); build_future_features for 2011-12-05 uses no data from 2011-12-05 or later and has the same columns as build_features.
```
**Done when:** `pytest tests/test_features.py -q` is green.

---

## Task M3: Feature quality review and row counts

```
Run /leakage-review on src/features.py and show the table. Then print: number of training rows (target weeks before 2011-10-10), number of test rows (8 weeks), rows dropped for missing history, share of NaN per feature in train and test, and the number of rows whose rolling window skipped the closed week. Compare with expectations (training roughly 3,000 to 3,500 rows, test 800 rows) and explain any difference. Do not change code unless you find a real leakage or counting bug; if so, fix it with a test.
```
**Done when:** the review reports no HIGH-severity findings. Commit: `feat(features): leakage-safe feature pipeline`.

---

## Task M4: Split, baselines and metrics

**SRS:** FR-M-1, FR-M-5, FR-M-6 | **Output:** start of `src/train.py`

```
Read docs/SRS.md section 5.2. Create src/train.py with:
- split_chronological(df, test_start="2011-10-10") -> (train, test): train = target weeks before test_start, test = on or after; assert max(train week) < min(test week) and the test has exactly 8 distinct weeks.
- evaluate(y_true, y_pred) -> dict with MAE, RMSE, WAPE (= sum(|error|) / sum(actual)) and bias (mean of predicted minus actual). No MAPE anywhere.
- Baseline predictors that need no fitting: naive (last non-missing week = lag_1, but if lag_1 is NaN use the next available earlier week) and moving_avg_4 (= roll_mean_4). Clip predictions at 0.

Tests (tests/test_models.py): the split is strictly chronological; evaluate matches hand-computed values on a 4-row example (state the numbers in the test); WAPE handles an all-zero actual by returning NaN, not an error; predictions are clipped at 0.

Print baseline metrics on the test set as a table.
```
**Done when:** baseline test metrics print and the tests pass.

---

## Task M5: Model pipelines (Ridge, Random Forest, LightGBM)

**SRS:** FR-M-2

```
In src/train.py add a function make_model(name, params=None) returning an sklearn-compatible estimator for name in {"ridge", "random_forest", "lightgbm"}, each with seed 42:
- ridge: sklearn Pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge); imputation and scaling are inside the pipeline so they are fitted on training data only
- random_forest: Pipeline(SimpleImputer(strategy="median"), RandomForestRegressor)
- lightgbm: LGBMRegressor with native NaN handling, verbosity -1, deterministic settings

Add fit_predict(name, params, train, test, features) -> predictions clipped at 0. Train on FEATURE_COLUMNS by default. Report test metrics for all three with default hyperparameters next to the baselines in one table. Do not tune yet. Add tests: each model fits on a tiny frame and predicts finite values; the Ridge pipeline's imputer statistics come from train only (assert that fitting on train then predicting a test row with NaN uses the train median).
```
**Done when:** the comparison table prints and the tests pass.

---

## Task M6: Time-series tuning and out-of-fold validation predictions

**SRS:** FR-M-4, FR-M-10 | **Strongest model**

```
Implement tuning in src/train.py.

1. A week-based splitter: expanding-window TimeSeriesSplit over the UNIQUE target weeks of the training period (n_splits=5), so all rows of one week fall in the same fold and no week is split across train and validation. Write a test proving train weeks precede validation weeks in every fold and no week appears on both sides.
2. tune(name, train, features, n_trials=30, seed=42): random search with the splitter, scored by WAPE on the validation folds. Search spaces: ridge alpha (log 1e-3 to 1e3); random_forest n_estimators 100 to 500, max_depth 3 to 20 or None, min_samples_leaf 1 to 20; lightgbm num_leaves 8 to 64, learning_rate (log 0.01 to 0.2), n_estimators 100 to 600, min_child_samples 5 to 50, subsample 0.6 to 1.0. Return best params, mean CV WAPE and the per-trial table.
3. oof_predictions(name, params, train, features) -> DataFrame with product_id, week_start, actual, predicted from the validation folds. These are the 'val' predictions.

Never touch the test set in any of this. Print the best params and CV WAPE for each model and the time taken.
```
**Done when:** the splitter test passes and tuned params are printed (save them to `models/tuned_params.json`).

---

## Task M7: Ablation of calendar features

**SRS:** FR-F-3, section 6.4

```
Using the tuned LightGBM and Ridge from M6, run an ablation on validation WAPE (out-of-fold, never the test set) for these feature sets: (a) base FEATURE_COLUMNS, (b) base minus week_of_year and month, (c) base plus weeks_to_christmas, (d) lags only. Print a table with validation WAPE and bias for each model and set. Decide which feature set to use and justify it in two sentences. Warn explicitly if (c) improves validation but could simply be acting as a time index. Save the chosen list in models/chosen_features.json.
```
**Done when:** the table and decision are printed and `chosen_features.json` exists.

---

## Task M8: Level-shift diagnostic

**SRS:** section 5.2 (known modelling risk)

```
The October and November test weeks sit at a higher demand level than most training weeks, and tree models cannot predict above the range seen in training. Diagnose this:
1. For each model (baselines, ridge, random forest, lightgbm) compute bias and WAPE per test week and print a table of 8 weeks x models.
2. Compute the share of test rows where the actual exceeds the maximum value of that product's training target, and the share of those rows that each model under-predicts.
3. State in plain words whether tree models under-forecast on the test period.
4. If they do, implement a ratio-target variant: train on target / roll_mean_4 (guard zeros and NaN) and multiply back at prediction time; add it as candidate "lightgbm_ratio", tune it with the M6 procedure, and add it to the comparison.

Report all of it honestly, including if nothing beats the baselines.
```
**Done when:** the per-week table and a clear written conclusion exist.

---

## Task M9: Model selection and saved artifacts (contract C5)

**SRS:** FR-M-7 to FR-M-9

```
Select the best model by VALIDATION WAPE, ties broken by validation bias (not by test results). Refit it on the full training period with the chosen features, and save:
- models/best_model.joblib (the full sklearn Pipeline if one is used)
- models/feature_list.json (ordered feature names)
- models/config.json: {"model_name", "horizon_weeks": 1, "train_end": "2011-10-03", "test_start": "2011-10-10", "seed": 42, "features_path", "library_versions": {python, pandas, numpy, scikit-learn, lightgbm}}

Tests: the saved model reloads in a fresh process and reproduces the stored test predictions exactly; feature_list.json order equals the column order used in training.
```
**Done when:** the reload test passes.

---

## Task M10: Export result files (contracts C2, C3, C4)

**SRS:** FR-M-5, FR-M-10 | **Output:** the files the team consumes

```
Read CONTRACTS.md C2, C3, C4. Write:

1. data/processed/predictions.csv: columns product_id, week_start, actual, predicted, model_name, split. Rows for EVERY model (naive, moving_avg_4, ridge, random_forest, lightgbm and any variant), split "test" (models refit on the training period) and split "val" (out-of-fold).
2. data/processed/metrics.csv: columns model_name, split, MAE, RMSE, WAPE, bias for every model and both splits.
3. data/processed/next_week_forecast.csv: columns product_id, week_start (2011-12-05), forecast_weekly (best model, built with build_future_features, clipped at 0), sigma_error, avg_weekly_units_train. sigma_error = per-product standard deviation of the best model's OUT-OF-FOLD validation errors (predicted minus actual); if a product has fewer than 6 validation observations use the pooled standard deviation of its volume segment from data/processed/product_segments.csv. avg_weekly_units_train = mean weekly units in the training period excluding the closed week.

Tests: schema checks for C2 to C4 (tests/test_contracts.py must pass on real files); exactly 100 products in C4; no negative or NaN forecasts; sigma_error >= 0; test split has 800 rows per model.
```
**Done when:** `pytest tests/test_contracts.py -q` passes on the real files. Tell Members 1 and 3. Commit: `feat(model): artifacts and result files`.

---

## Task M11: Modeling notebook and model card

```
Create notebooks/02_modeling.py (cell format) that reproduces the whole flow from weekly_demand.parquet to the saved artifacts, with a final metrics table and charts (WAPE by model for validation and test; predicted vs actual for the best model on 6 sample products). Then write docs/MODEL_CARD.md: purpose and horizon; data and split dates; features and why each exists; models compared and tuning budget; final metrics table (real numbers only); the chosen model and why (validation-based); the level-shift finding; limitations (one year of data, intermittent demand, bulk orders, closure week); how sigma_error is derived; how to reload and use the model. Mark any number you cannot find as TODO.
```
**Done when:** the notebook runs top to bottom and the model card has no invented numbers.

---

## Task M12 (stretch): Poisson regression and a 4-week horizon

Only after M10 is delivered and the team is on schedule.

```
Add a PoissonRegressor pipeline (imputer, scaler) as candidate "poisson", tuned like the others, and add it to the comparison. Then build a direct 4-week-ahead variant for the best model family (target = units at t+3 using features available at t) and report test metrics for horizon 4. Do not change the saved horizon-1 artifacts or contracts; write results to reports/horizon4.md and tell me whether it is worth adding to the SRS.
```
