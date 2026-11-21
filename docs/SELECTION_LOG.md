# ShelfWatch Model Selection Log

**Document Version:** 1.0  
**Date:** 2026-11-21  
**Author:** ML Engineering Team  
**Governing SRS Rule:** FR-M-7 (Model Selection Hierarchy & Tie-breaking)  
**Applicable Contracts:** C2, C3, C4, C5  

---

## 1. Explicit Binding Statement

> **BINDING POLICY NOTICE (FR-M-7):**  
> Model selection was conducted **strictly on the training period** using expanding-window cross-validation and out-of-fold validation performance.  
> **Test numbers were NOT used to select the model.**  
> The test period (`2011-10-10` through `2011-11-28`, 8 full weeks, 800 observations) remained completely isolated throughout hyperparameter tuning, feature group ablation, and model selection. Test metrics reported in evaluation tables are strictly diagnostic and evaluative.

---

## 2. Validation Metrics & Official Selection Ranking

Under **FR-M-7**, candidate models are ranked by **validation WAPE** (lowest wins). In the event of a tie, the tie is broken by validation bias (closest to 0.0).

All candidates were evaluated across the 5 expanding-window validation folds on the 44-week training period (`2010-12-06` to `2011-10-03`, excluding shop closure week `2010-12-27`):

| Rank | Candidate / Model Name | Validation WAPE | Validation Bias | Selection Status | Notes |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **1** | **`ensemble`** | **0.5238** | **+2.0242** | **SELECTED (Served Model)** | Blends $0.8 \times \text{ridge\_raw} + 0.2 \times \text{moving\_avg\_4}$ |
| 2 | `ridge_raw` | 0.5251 | +3.6944 | Component | CV mean fold WAPE: 0.5271; $\alpha = 157.42$ |
| 3 | `random_forest_raw` | 0.5253 | +3.4597 | Evaluated | CV mean fold WAPE: 0.5273 |
| 4 | `lightgbm_raw` | 0.5285 | +3.9064 | Evaluated | CV mean fold WAPE: 0.5302 |
| 5 | `random_forest_scale_aware` | 0.5305 | +6.1945 | Evaluated | Target scaled by `roll_mean_4` |
| 6 | `moving_avg_4` (baseline) | 0.5446 | -4.6566 | Component | 4-week moving average; CV fold mean: 0.5466 |
| 7 | `lightgbm_scale_aware` | 0.5541 | +7.2061 | Evaluated | Target scaled by `roll_mean_4` |
| 8 | `naive` (baseline) | 0.6504 | -2.7880 | Evaluated | Last available open-week lag; CV fold mean: 0.6523 |
| 9 | `ridge_scale_aware` | 0.7825 | +33.4550 | Evaluated | Target scaled by `roll_mean_8` |

*(Note: Whole-split validation metrics evaluate on the 2,500 concatenated out-of-fold validation predictions across all 100 products and 5 validation folds, matching Contract C3 and `data/processed/metrics.csv`).*

---

## 3. Decision Justification

1. **FR-M-7 Compliance:**
   - The **`ensemble`** candidate achieved a validation WAPE of **0.5238**, which is strictly superior to all individual ML candidates (`ridge_raw`: 0.5251, `random_forest_raw`: 0.5253, `lightgbm_raw`: 0.5285) and baselines (`moving_avg_4`: 0.5446, `naive`: 0.6504).
   - Validation bias of the ensemble is **+2.0242**, lower than all individual ML models.
   - Therefore, the `ensemble` is the unambiguous winner under FR-M-7.

2. **Artifact Serialization:**
   - `models/config.json`: Records `model_name: "ensemble"`, `components: {"ridge_raw": 0.8, "moving_avg_4": 0.2}`, `groups: ["base", "g3"]`, and `selection_basis: "validation WAPE (FR-M-7)"`.
   - `models/best_model.joblib`: Persists the refit Ridge pipeline on the full training period with `base+g3_no_calendar` features.
   - `src/predict.py` (`predict_custom`): Computes the moving average component from user series `recent_units[-4:]` and blends with the fitted Ridge model.
   - `src/explain.py` (`explain_row`): Decomposes the ensemble prediction with exact linear feature attribution ($0.8 \times \text{ridge\_contributions}$ plus $0.2 \times \text{roll\_mean\_4}$ added to the `roll_mean_4` contribution, base value scaled by 0.8), preserving the exact identity $|\text{base} + \sum \text{contrib} - \hat{y}| < 10^{-6}$.
   - `data/processed/next_week_forecast.csv`: Generates next-week predictions for `2011-12-05` using the ensemble, and computes `sigma_error` from the ensemble's out-of-fold validation errors.

---

## 4. Test Set Evaluation (Informational Only)

Subsequent to model selection, candidates were evaluated on the held-out 8-week test set (`2011-10-10` to `2011-11-28`):

| Model Name | Test WAPE | Test Bias | Test MAE | Test RMSE | Outperformed `moving_avg_4`? |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`ensemble`** | **0.4791** | **+4.7711** | **141.3968** | **208.3836** | **YES (-0.0010 WAPE)** |
| `moving_avg_4` | 0.4801 | -5.9375 | 141.7169 | 209.2925 | Baseline |
| `ridge_raw` | 0.4882 | +7.4483 | 144.0818 | 212.8197 | No (+0.0081 WAPE) |
| `random_forest_scale_aware` | 0.4885 | +5.5638 | 144.1846 | 210.6931 | No (+0.0084 WAPE) |
| `lightgbm_raw` | 0.4903 | +2.1177 | 144.7066 | 219.0309 | No (+0.0102 WAPE) |
| `random_forest_raw` | 0.4910 | -1.5963 | 144.9359 | 223.5911 | No (+0.0109 WAPE) |
| `lightgbm_scale_aware` | 0.5009 | +4.4466 | 147.8456 | 218.1350 | No (+0.0208 WAPE) |
| `naive` | 0.5789 | +0.3412 | 170.8512 | 256.7089 | No (+0.0988 WAPE) |
| `ridge_scale_aware` | 0.8326 | +24.4348 | 245.7545 | 364.7458 | No (+0.3525 WAPE) |

The selected `ensemble` achieved a test WAPE of **0.4791**, outperforming the strong `moving_avg_4` baseline (**0.4801**) on the unseen test period by 0.0010 WAPE points. This confirms the robustness of the validation-driven selection.
