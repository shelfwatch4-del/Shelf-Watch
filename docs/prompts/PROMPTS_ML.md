# PROMPTS_ML.md: Machine Learning (Member 2), version 3

**You own:** `src/features.py`, `src/train.py`, `models/*`, `data/processed/predictions.csv`, `metrics.csv`, `next_week_forecast.csv`, `notebooks/02_modeling.py`, `docs/MODEL_CARD.md`, `kaggle/notebook/`.
**Inputs:** `weekly_demand.parquet`, `closed_weeks.csv`, `product_segments.csv`, and (after Member 1's E1 to E5) `calendar_weekly.csv`, `prior_year_weekly.parquet`, `enrichment_weekly.parquet`, `product_intelligence.csv`.
**Setup:** paste the Windows preface and closing check from `PROMPTS_00_INDEX.md` into every prompt. Planning mode and the strongest model.

**Rules that apply to every task:** chronological split only; `shift(1)` before rolling; tune only inside the training period; never use the test set for tuning, imputation, scaling, capping, feature selection or model selection; metrics MAE, RMSE, WAPE, bias (never MAPE); clip predictions at 0; seed 42. Enrichment tables hold the value FOR week w and only `features.py` shifts them.

## Status
| Task | Status |
|---|---|
| M1 to M5 | Done: lag, rolling, calendar, commercial and history features (15 default features), split, baselines, Ridge, Random Forest, LightGBM at defaults |
| M5 result (test) | Moving average WAPE 0.4801 (bias -5.94); Ridge 0.4913 (bias -0.18); LightGBM 0.5315 (bias +26.23). **The baseline wins; LightGBM over-forecasts** |
| **M5b** | Next: diagnose (read-only) |

Why the M5 result matters: the earlier worry was that trees would under-forecast the higher October to November level, but LightGBM over-forecast. The likelier cause is overfitting to wholesale spikes on raw-unit targets across products of very different size. The tasks below test that instead of assuming it.

---

## Task M5b: Diagnose the M5 results (read-only)
```
Read-only. Do NOT change src/. Create scripts/diagnose_m5.py and docs/M5_DIAGNOSIS.md.

Using the current M5 code and default models, print on the TEST set for naive, moving_avg_4, ridge, random_forest and lightgbm: MAE, RMSE, WAPE, bias. Then:
1. Per test week (8 weeks): WAPE and bias for each model.
2. By volume segment (data/processed/product_segments.csv): WAPE and bias for each model.
3. The share of test rows where actual exceeds that product's maximum training target, and the share of those rows each model under-predicts.
4. For LightGBM: the 10 products and the 10 (product, week) rows contributing the most absolute error, and what share of total absolute error they make up.
5. LightGBM training WAPE versus test WAPE (overfitting check).
End docs/M5_DIAGNOSIS.md with 5 findings that say, with the printed numbers, whether the problem is (a) spikes and scale, (b) level shift, (c) overfitting, or something else. Do not guess; cite the numbers.
```
**Done when:** the findings are backed by printed numbers.

## Task M6: Switchable feature groups
**SRS:** FR-F-10 to FR-F-16 | **Needs:** E1, E2, E3 (build G1 first if they are not ready)

```
Read docs/SRS.md section 5.1 (FR-F-10 to FR-F-16) and CONTRACTS_ADDENDUM.md C8, C9, C10. Do not change existing feature definitions or their tests.

In src/features.py add FEATURE_GROUPS = {"base": FEATURE_COLUMNS (the existing 15), "g1": [...], "g2": [...], "g3": [...], "g4": [...]} and extend build_features(weekly, closed_weeks, enrichment=None, groups=("base",)), build_future_features(...) with the same arguments.

Groups (all computed only from information known before target week t, except calendar facts known in advance):
- g1 (derived behaviour): order_size_lag1 = units / orders at t-1 (guard zero orders); trend_ratio = roll_mean_4 / (roll_mean_8 + 1e-9, guarded); price_index = prev_avg_price / mean of the previous 8 weeks' prices; cv_4 = roll_std_4 / (roll_mean_4 + 1); zero_streak = consecutive zero weeks before t.
- g2 (calendar, from calendar_weekly.csv): uk_bank_holiday_count and trading_days_expected of the TARGET week.
- g3 (prior year, from prior_year_weekly.parquet): yoy_mean_5w = mean of prior-year units for weeks t-54 to t-50 (weeks), needing at least 3 values else missing; yoy_seasonal_index = yoy_mean_5w divided by the product's mean over all its prior-year weeks; yoy_available (0 or 1).
- g4 (transaction-derived, shifted): return_rate_4w = sum(cancelled_units over t-4..t-1) / max(sum(gross_units over t-4..t-1), 1); export_share_4w; guest_share_4w. Skip the closure week like the other rolling features.

Also implement build_custom_features(series, target_week) for a user-entered series (units, optional prices, orders, customers): builds base and g1 and g2 features exactly like build_features; g3 and g4 columns are missing.

Tests: each group switches on and off by name; column order is stable; EXTEND the leakage test so that, for 3 products and 3 target weeks each, it changes units, orders, customers, avg_price AND the enrichment columns at every week >= t and asserts that ALL feature columns of EVERY group at week t are identical (on the fixture and on the real tables); yoy features use only weeks t-54 to t-50; custom features equal table features for a series taken from the table.

Print the row counts and the share of missing values per feature for each group.
```
**Done when:** the extended leakage test passes for every group.

## Task M7: Tuning with a scale-aware candidate
**SRS:** FR-M-4, FR-M-11

```
Read docs/SRS.md section 5.2. Use the chosen feature set "base" for now (groups are compared in M8).

1. Week-based splitter: expanding-window TimeSeriesSplit over the UNIQUE target weeks of the training period (n_splits=5), so no week appears on both sides. Test that train weeks precede validation weeks in every fold.
2. tune(name, train, features, n_trials=30, seed=42): random search scored by validation WAPE. Spaces: ridge alpha (log 1e-3 to 1e3); random_forest n_estimators 100 to 500, max_depth 3 to 20 or None, min_samples_leaf 1 to 20; lightgbm num_leaves 8 to 64, learning_rate (log 0.01 to 0.2), n_estimators 100 to 600, min_child_samples 5 to 80, subsample 0.6 to 1.0, colsample_bytree 0.6 to 1.0, reg_lambda (log 1e-3 to 10).
3. Two target modes for every model: raw units, and scale-aware (target divided by roll_mean_4 or roll_mean_8, chosen as a hyperparameter, with the denominator floored at 1; multiply back at prediction time). Report them as separate candidates.
4. Out-of-fold validation predictions for every candidate (product_id, week_start, actual, predicted).
5. Save the best parameters to models/tuned_params.json.

Never touch the test set. Print a table: candidate, mean validation WAPE, validation bias, best parameters, time taken. Say plainly whether any candidate beats the moving average on VALIDATION.
```

## Task M8: Feature-group ablation
**SRS:** FR-F-14, FR-F-15

```
Using the best tuned LightGBM and Ridge (both target modes) from M7, compare these feature sets on VALIDATION only (out-of-fold): base; base+g1; base+g2; base+g3; base+g4; all groups; also base with and without week_of_year and month; and base plus weeks_to_christmas.

For each set print validation WAPE, bias and the paired difference against base per fold (so we can see if an improvement is consistent or one lucky fold). Decide which groups to keep using validation WAPE only. Report groups that did not help, and flag guest_share_4w explicitly as a hypothesis. Warn if weeks_to_christmas or a calendar feature seems to act as a time index. Save the decision to models/chosen_groups.json and the table to models/ablation.json (rows: groups, val_wape, val_bias).

Do not use the test set. Do not drop a group because it looks good or bad on test.
```

## Task M9: Diagnostics and ensemble
```
Using the chosen groups and best candidate:
1. Over-forecast and under-forecast units separately for every candidate (the manager's costs are asymmetric: a stockout loses sales, overstock ties up cash). Then per-week bias and WAPE across the 8 test weeks for every candidate (table 8 weeks x candidates). State whether tree models under-forecast or over-forecast in October and November and by how much.
2. Error by demand pattern class and ABC class from product_intelligence.csv (if it exists).
3. Ensemble candidate "ensemble": w x best_ML + (1 - w) x moving_avg_4 with w from 0.0 to 1.0 in steps of 0.1, chosen on VALIDATION only. Report the chosen w and its validation and test metrics.
Add the ensemble to the comparison. Do not select a model here.
```

## Task M10: Quantile models (prediction intervals)
**SRS:** FR-M-13

```
Train two LightGBM quantile models (objective quantile, alpha 0.1 and 0.9) on the chosen features, with the scale-aware target if it was chosen. Produce out-of-fold validation quantiles, then refit on the full training period.
- Sort p10 <= point forecast <= p90 after prediction and clip at 0.
- Report empirical coverage (share of actuals inside [p10, p90]) on validation and on test, and the pinball loss. The target is about 80 percent. If validation coverage is far from 80 percent, apply a simple correction based on the validation residuals (document exactly what you did) and report coverage again.
- Save models/q10.joblib and models/q90.joblib and write the coverage figures to models/interval_coverage.json.
Tests: ordering holds for every row; coverage function matches a hand-computed example; saved models reload and reproduce predictions.
```

## Task M11: Select the model, save artifacts, add `predict_custom`
**SRS:** FR-M-7, FR-M-9, FR-M-14 | **Contract:** C5

```
1. Select the best candidate by VALIDATION WAPE, ties broken by validation bias. Refit on the full training period with the chosen groups and target mode. Save models/best_model.joblib, models/feature_list.json (ordered), models/config.json {model_name, target_mode, groups, horizon_weeks: 1, train_end: "2011-10-03", test_start: "2011-10-10", seed: 42, library_versions, data_version}.
2. Implement predict_custom(payload) in src/train.py (or src/predict.py): payload = {recent_units: 8 to 12 non-negative numbers oldest first, recent_prices, recent_orders, recent_customers (optional), target_week}. Build features with build_custom_features, predict with the saved model and quantile models, and return {forecast_weekly, forecast_p10, forecast_p90, features_used, notes}. The notes must say which groups were unavailable (g3, g4). Reject fewer than 8 values or negative values with ValueError.
3. Tests: the saved model reloads in a fresh process and reproduces stored test predictions exactly; predict_custom for a series taken from the table equals the table-based prediction for the shared features; bad payloads are rejected.
```

## Task M12: Result files (contracts C2, C3, C4)
**SRS:** FR-M-5, FR-M-10

```
Read CONTRACTS.md C2, C3, C4 and CONTRACTS_ADDENDUM.md C4. Write:
1. data/processed/predictions.csv: product_id, week_start, actual, predicted, model_name, split ("test" and "val") for EVERY candidate, including baselines, ensemble and the scale-aware variants.
2. data/processed/metrics.csv: model_name, split, MAE, RMSE, WAPE, bias for every candidate and both splits.
3. data/processed/next_week_forecast.csv: product_id, week_start (2011-12-05), forecast_weekly, forecast_p10, forecast_p90, sigma_error, avg_weekly_units_train, last_avg_price. sigma_error = per-product standard deviation of the best model's OUT-OF-FOLD validation errors (predicted minus actual); with fewer than 6 validation observations use the pooled value of the product's volume segment. avg_weekly_units_train excludes the closure week. last_avg_price = the product's last open-week price before the forecast week.
Tests: schema checks for C2, C3, C4; 100 products; 800 test rows per candidate; no negative or missing forecasts; sigma_error >= 0; p10 <= forecast <= p90.
Then tell Members 1 and 3 that D8 and the API can proceed.
```

## Task M13: Notebook, model card and Kaggle notebook
```
1. notebooks/02_modeling.py (cell format) reproducing the whole flow from the processed tables to the saved artifacts, with a final metrics table, WAPE by candidate, predicted versus actual for 6 sample products, and the ablation table.
2. docs/MODEL_CARD.md: purpose and horizon; data and split dates; features by group and why each exists; candidates and tuning budget; final metrics (real numbers only); the chosen model and why (validation-based); the level-shift and spike findings; interval coverage; limitations (one year of main data, intermittent demand, bulk orders, closure week, guest_share hypothesis); how to reload and use the model. Mark any number you cannot find as TODO.
3. kaggle/notebook/shelfwatch_modeling.py: a version of the notebook that reads from a single DATA_DIR variable (default /kaggle/input/shelfwatch-data) and writes to /kaggle/working, with no local-only paths. Test it locally by setting DATA_DIR to data/processed.
```

## Task M14 (stretch): Pretrained time-series model and other extras
Only after M12 is delivered and the team is on schedule.
```
In a Kaggle notebook (GPU optional), evaluate a pretrained time-series model on our 100 weekly series as one extra candidate: zero-shot first, then fine-tuned if time allows. Use the same chronological split and the same metrics. Also try a Tweedie or Poisson LightGBM objective. Report measured results only, including if it does not beat the moving average. Do not change the saved artifacts or contracts. Write results to reports/stretch.md.
```

---

## Task M15: Tuning trials, feature importance and explain helper (added in version 3.2)
**SRS:** FR-M-11, FR-M-17
```
1. Make tune() in M7 also return every trial as a row and save data/processed/tuning_trials.csv: candidate, trial_no, params (JSON string), mean_val_wape, is_best.
2. Save data/processed/feature_importance.csv for the final model: feature, group, gain_importance (LightGBM) or abs_coef (Ridge), mean_abs_contribution on the test rows.
3. Create src/explain.py: explain_row(model, feature_row) returning feature values, per-feature contributions and base_value (LightGBM pred_contrib; for Ridge coefficient x scaled value) such that base_value + sum(contributions) equals the model output within 1e-6; for scale-aware targets, return contributions in the scaled space and state the scaling factor. Tests for the sum identity on 20 random rows.
4. Document both new files as contracts in CONTRACTS_ADDENDUM.md (tuning_trials, feature_importance) and make sure scripts/load_mysql.py (B6b) loads them.
Print numbers from the files, not from memory.
```
