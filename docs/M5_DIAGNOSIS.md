# M5 Model Diagnosis: Test Set Evaluation and Error Analysis

**Task:** M5b (Read-only Diagnostic)  
**Date:** 2026-11-21  
**Scope:** Test set (2011-10-10 to 2011-11-28, 8 full weeks, 800 product-week observations across 100 products).  
**Models Evaluated:** `naive`, `moving_avg_4`, `ridge`, `random_forest`, `lightgbm` (all at default hyperparameters with 15 base features).

---

## 1. Executive Summary and Overall Test Performance

The benchmark evaluation on the test set confirms that the 4-week moving average baseline outperforms all machine learning models on WAPE and MAE. LightGBM and Random Forest exhibit substantial positive bias and higher error rates.

### Overall Test Metrics Table (800 rows, 8 weeks)

| Model | MAE | RMSE | WAPE | Bias |
|---|---|---|---|---|
| **naive** | 170.8512 | 256.7089 | 0.5789 | +0.3412 |
| **moving_avg_4** | **141.7169** | **209.2925** | **0.4801** | -5.9375 |
| **ridge** | 144.9976 | 217.8037 | 0.4913 | **-0.1812** |
| **random_forest** | 160.7415 | 225.5506 | 0.5446 | +39.5167 |
| **lightgbm** | 156.8811 | 228.2321 | 0.5315 | +26.2276 |

- `moving_avg_4` wins overall on WAPE (0.4801 vs LightGBM 0.5315 and Ridge 0.4913).
- Ridge is close behind the baseline (WAPE 0.4913) with virtually zero overall bias (-0.1812).
- Default tree models over-predict on average: LightGBM bias is +26.2276 units/row, while Random Forest bias is +39.5167 units/row.

---

## 2. Weekly Breakdown across the 8 Test Weeks

Demand ramps up substantially across October and November due to the Q4 holiday volume surge, ranging from 20,178 units in the first test week to 36,160 units in the peak week.

### Weekly WAPE and Bias by Model

| Week Start | Actual Units | `naive` WAPE | `naive` Bias | `moving_avg_4` WAPE | `moving_avg_4` Bias | `ridge` WAPE | `ridge` Bias | `random_forest` WAPE | `random_forest` Bias | `lightgbm` WAPE | `lightgbm` Bias |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **2011-10-10** | 20,178.0 | 0.9069 | +129.7200 | 0.6828 | +104.7725 | 0.6569 | +104.8704 | 0.7067 | +115.4563 | 0.7019 | +110.1658 |
| **2011-10-17** | 26,215.0 | 0.5474 | -60.3700 | 0.5247 | +22.3400 | 0.5031 | +20.4712 | 0.5768 | +58.2993 | 0.5319 | +42.1075 |
| **2011-10-24** | 27,792.0 | 0.5159 | -15.7700 | 0.4682 | -16.2250 | 0.5019 | -12.3950 | 0.5436 | +41.0411 | 0.5313 | +34.2595 |
| **2011-10-31** | 29,697.0 | 0.6194 | -19.0500 | 0.4968 | -28.6325 | 0.4857 | -25.8684 | 0.5627 | +34.3471 | 0.5276 | +16.7901 |
| **2011-11-07** | 33,386.0 | 0.5875 | -36.8900 | 0.4692 | -74.1550 | 0.5159 | -22.7104 | 0.5400 | +3.9634 | 0.5125 | -15.2042 |
| **2011-11-14** | 36,160.0 | 0.4447 | -27.7400 | 0.3798 | -68.8750 | 0.4197 | -46.5573 | 0.4464 | -16.3551 | 0.4444 | -27.7380 |
| **2011-11-21** | 29,820.0 | 0.5891 | +63.4000 | 0.4949 | +19.3875 | 0.4671 | +16.0954 | 0.5466 | +56.3950 | 0.5623 | +45.1338 |
| **2011-11-28** | 32,877.0 | 0.5486 | -30.5700 | 0.4234 | -6.1125 | 0.4518 | -35.3559 | 0.5149 | +22.9862 | 0.5177 | +4.3058 |

### Weekly Observations:
- **Week 1 (2011-10-10):** Every model severely over-predicted (LightGBM bias +110.17, MA4 bias +104.77, Ridge bias +104.87) following late-September training spikes, resulting in high WAPEs across all models (0.65 to 0.91).
- **Peak Weeks (2011-11-07 & 2011-11-14):** As volume peaked at 33.3k and 36.2k units, the models under-predicted (LightGBM bias -15.20 and -27.74; MA4 bias -74.16 and -68.88).
- **Post-Peak Dip (2011-11-21):** Volume temporarily dropped from 36.2k to 29.8k units, causing models to overshoot again (LightGBM bias jumped back to +45.13, RF to +56.40).

---

## 3. Breakdown by Product Volume Segment

Segments from `data/processed/product_segments.csv` group the Top 100 products into:
- `low`: 34 products, 272 test rows, 56,751 total actual units.
- `medium`: 32 products, 256 test rows, 55,575 total actual units.
- `high`: 34 products, 272 test rows, 123,799 total actual units (52.4% of total test demand).

### Segment Breakdown Table

| Segment | Products | Rows | Actual Units | Model | WAPE | Bias |
|---|---|---|---|---|---|---|
| **low** | 34 | 272 | 56,751.0 | `naive`<br>`moving_avg_4`<br>`ridge`<br>`random_forest`<br>`lightgbm` | 0.6758<br>**0.5350**<br>0.5652<br>0.6193<br>0.5873 | -4.1912<br>-7.0965<br>**-0.5841**<br>+34.3152<br>+7.2159 |
| **medium** | 32 | 256 | 55,575.0 | `naive`<br>`moving_avg_4`<br>`ridge`<br>`random_forest`<br>`lightgbm` | 0.6071<br>**0.5368**<br>0.5407<br>0.6067<br>0.5804 | -1.4297<br>**+0.2383**<br>+29.4098<br>+59.9766<br>+35.3938 |
| **high** | 34 | 272 | 123,799.0 | `naive`<br>`moving_avg_4`<br>`ridge`<br>`random_forest`<br>`lightgbm` | 0.5217<br>**0.4296**<br>0.4352<br>0.4825<br>0.4840 | +6.5404<br>-10.5910<br>-27.6287<br>+25.4618<br>+36.6122 |

### Key Segment Findings:
- `moving_avg_4` achieves the lowest WAPE in **every single volume segment** (0.5350 in low, 0.5368 in medium, and 0.4296 in high).
- High-volume products dominate total units (123.8k out of 236.1k units), and this is where LightGBM accumulates large raw-unit bias (+36.61 units/row) and an elevated WAPE (0.4840 vs MA4's 0.4296).
- LightGBM and Random Forest show severe positive bias in medium (+35.39 and +59.98) and high (+36.61 and +25.46) tiers.

---

## 4. Extreme Rows: Actual Exceeding Product Training Maximum

We examined whether test demand exceeded historical bounds observed in the training period for each product.

- **Total test rows:** 800
- **Test rows where actual > product training max:** **35 rows (4.3750%)**

### Model Under-Prediction on Exceeded Rows

| Model | Exceeded Rows | Under-Predicted Rows | Share Under-Predicted | Mean Under-Prediction Gap (units) |
|---|---|---|---|---|
| **naive** | 35 | 30 | 85.71% | 436.4000 |
| **moving_avg_4** | 35 | 34 | 97.14% | 494.2143 |
| **ridge** | 35 | 35 | **100.00%** | 602.3032 |
| **random_forest** | 35 | 35 | **100.00%** | 541.7603 |
| **lightgbm** | 35 | 35 | **100.00%** | 579.1258 |

### Observation:
- For all 35 occurrences where a product reached an all-time peak in the test set, **Ridge, Random Forest, and LightGBM under-predicted 100% of the time**.
- On these rows, LightGBM lagged actual demand by an average of **579.13 units per row**, and Ridge by **602.30 units**.
- Tree models cannot extrapolate beyond the maximum target values seen during training, guaranteeing severe under-prediction when historical peaks are surpassed.

---

## 5. LightGBM Error Concentration Analysis

**Total LightGBM Absolute Error across all 800 test rows:** **125,504.89 units**

### Top 10 Products Contributing Most Absolute Error

| Product ID | Total Abs Error | Mean Abs Error | Net Bias | Total Actual Units | Total Predicted Units | Share of Total Abs Error | Cumulative Share |
|---|---|---|---|---|---|---|---|
| **22197** | 6,909.63 | 863.70 | -863.70 | 14,472.0 | 7,562.37 | **5.51%** | 5.51% |
| **22952** | 3,427.34 | 428.42 | -214.14 | 7,719.0 | 6,005.85 | **2.73%** | 8.24% |
| **22492** | 3,335.06 | 416.88 | +30.92 | 5,331.0 | 5,578.38 | **2.66%** | 10.89% |
| **21977** | 2,880.15 | 360.02 | +360.02 | 1,993.0 | 4,873.15 | **2.29%** | 13.19% |
| **21915** | 2,754.46 | 344.31 | +125.10 | 3,979.0 | 4,979.81 | **2.19%** | 15.38% |
| **85099B** | 2,749.73 | 343.72 | +101.67 | 8,740.0 | 9,553.37 | **2.19%** | 17.57% |
| **21181** | 2,641.13 | 330.14 | +15.83 | 2,609.0 | 2,735.61 | **2.10%** | 19.68% |
| **22386** | 2,594.13 | 324.27 | +324.27 | 3,554.0 | 6,148.13 | **2.07%** | 21.75% |
| **22616** | 2,239.93 | 279.99 | -4.21 | 3,564.0 | 3,530.34 | **1.78%** | 23.53% |
| **21166** | 2,054.71 | 256.84 | -39.85 | 2,389.0 | 2,070.19 | **1.64%** | 25.17% |

- **Top 10 products make up 25.17% of all absolute errors** in the test set.
- A single product (`22197`, POPCORN HOLDER) alone accounts for **5.51% of total model error**, consistently under-forecasted by an average of -863.70 units every week.

### Top 10 (Product, Week) Rows Contributing Most Absolute Error

| Product ID | Week Start | Actual Units | Predicted Units | Abs Error | Error (Pred - Act) | Share of Total Abs Error | Cumulative Share |
|---|---|---|---|---|---|---|---|
| **22197** | 2011-11-28 | 2,755.0 | 1,112.04 | 1,642.96 | -1,642.96 | **1.31%** | 1.31% |
| **22197** | 2011-11-14 | 2,495.0 | 969.92 | 1,525.08 | -1,525.08 | **1.22%** | 2.52% |
| **22197** | 2011-11-07 | 2,340.0 | 861.85 | 1,478.15 | -1,478.15 | **1.18%** | 3.70% |
| **22197** | 2011-11-21 | 2,435.0 | 1,219.75 | 1,215.25 | -1,215.25 | **0.97%** | 4.67% |
| **21166** | 2011-11-28 | 1,142.0 | 160.00 | 982.00 | -982.00 | **0.78%** | 5.45% |
| **22952** | 2011-10-17 | 1,551.0 | 599.43 | 951.57 | -951.57 | **0.76%** | 6.21% |
| **22492** | 2011-10-17 | 1,142.0 | 346.99 | 795.01 | -795.01 | **0.63%** | 6.84% |
| **79321** | 2011-11-21 | 959.0 | 183.38 | 775.62 | -775.62 | **0.62%** | 7.46% |
| **17003** | 2011-10-24 | 1,204.0 | 440.22 | 763.78 | -763.78 | **0.61%** | 8.07% |
| **21181** | 2011-11-28 | 936.0 | 240.91 | 695.09 | -695.09 | **0.55%** | 8.62% |

- The **top 10 rows (1.25% of all test rows) account for 8.62% of total test error**.
- The top 4 error rows in the entire dataset are all from **product `22197` across November 2011**, where actual weekly sales exceeded 2,300 units while LightGBM predicted only 861 to 1,220 units.

---

## 6. Overfitting Check: LightGBM Train vs Test

| Model / Split | Split | Rows | MAE | RMSE | WAPE | Bias | Test/Train WAPE Ratio |
|---|---|---|---|---|---|---|---|
| **LightGBM** | Train | 3,467 | 84.1005 | 113.4657 | **0.3243** | +0.0250 | — |
| **LightGBM** | Test | 800 | 156.8811 | 228.2321 | **0.5315** | +26.2276 | **1.64x (+0.2072)** |
| **Random Forest** | Train | 3,467 | 45.4265 | 77.2917 | **0.2022** | +2.7854 | — |
| **Random Forest** | Test | 800 | 160.7415 | 225.5506 | **0.5446** | +39.5167 | **2.69x (+0.3424)** |
| **Ridge** | Train | 3,467 | 137.1350 | 207.2893 | **0.5289** | +0.0093 | — |
| **Ridge** | Test | 800 | 144.9976 | 217.8037 | **0.4913** | -0.1812 | **0.93x (-0.0377)** |

### Overfitting Observations:
- **LightGBM overfits substantially:** Train WAPE is 0.3243, while test WAPE expands to 0.5315 (a 64% relative increase; gap +0.2072). Test bias expands from near zero (+0.0250) to +26.2276.
- **Random Forest overfits even more severely:** Train WAPE is 0.2022, exploding to 0.5446 on test (a 169% increase; gap +0.3424), with test bias reaching +39.5167.
- **Ridge does not overfit at all:** Train WAPE is 0.5289 and test WAPE is 0.4913 (gap -0.0377, test bias -0.1812).

---

## 7. Five Diagnostic Findings

Backed by the printed numbers above, here are the 5 findings regarding the root causes of the M5 results:

### Finding 1: Overfitting is a primary flaw in default tree models, but absent in Ridge
The numbers prove substantial overfitting in default tree architectures:
- LightGBM test WAPE (0.5315) is **1.64x** its training WAPE (0.3243), a gap of **+0.2072**, while bias surges from **+0.0250** to **+26.2276**.
- Random Forest is even more overfit: test WAPE (0.5446) is **2.69x** its training WAPE (0.2022), a gap of **+0.3424**, with test bias of **+39.5167**.
- In contrast, Ridge demonstrates **zero overfitting**: its test WAPE (0.4913) is lower than its train WAPE (0.5289, gap **-0.0377**), and its test bias is negligible (**-0.1812**).
- **Conclusion:** Default tree hyperparameters (e.g. `num_leaves=31` without regularization or leaf size constraints) memorize training samples and produce high variance on unseen test distributions. Regularization and tuning (Task M7) are essential.

### Finding 2: The problem is heavily driven by (a) Spikes and Scale Disparity
The numbers demonstrate that evaluating raw unit targets across products of vastly different scales severely degrades tree performance:
- Just **10 products (10%) account for 25.17% of total test absolute error** (31,586.27 out of 125,504.89 units).
- Product `22197` alone accounts for **5.51% of all absolute errors** (6,909.63 units), with an average under-prediction of **-863.70 units/week**.
- The top 10 error rows account for **8.62% of total test error**; the top 4 error rows in the entire dataset all belong to product `22197` (errors between 1,215 and 1,643 units).
- Because loss functions (MSE/L2) square raw-unit errors, products selling thousands of units dominate split decisions, while medium/low-volume products receive inappropriate split thresholds that cause systematic over-predictions (medium segment bias +35.39; high segment bias +36.61).
- **Conclusion:** A scale-aware target transformation (dividing by rolling mean, as planned for M7) is mathematically required to normalize loss contributions across scale tiers.

### Finding 3: Tree models cannot extrapolate to new peaks, causing 100% under-prediction on extreme demand weeks
When demand exceeds historical levels, tree models fail completely due to the bounded nature of leaf outputs:
- In **35 test rows (4.38% of all test rows)**, actual demand exceeded the product's maximum training sales.
- On those 35 rows, **LightGBM, Random Forest, and Ridge under-predicted 100% of the time (35/35)**.
- LightGBM under-forecasted by an average gap of **579.13 units per row**, and Ridge by **602.30 units**.
- Even naive and moving average baselines, which carry forward recent values, under-predicted 85.71% and 97.14% of these rows.
- **Conclusion:** When a product experiences a genuine surge above historical records, tree models cannot predict higher than the training target ceiling. Scale-aware targets allow the rolling mean multiplier to scale predictions above historical training targets.

### Finding 4: Level Shift is dynamic and week-dependent, not a static additive offset
The data refutes the hypothesis that the test period has a uniform, constant level shift:
- **Week 1 (2011-10-10):** Total actual volume was relatively low (20,178 units). All models massively over-forecasted (LightGBM bias **+110.17**, MA4 bias **+104.77**, Ridge bias **+104.87**; WAPE 0.65 to 0.91).
- **Peak Weeks (2011-11-07 to 2011-11-14):** Volume surged to 33,386 and 36,160 units. Here, models under-forecasted (LightGBM bias **-15.20** and **-27.74**; MA4 bias **-74.16** and **-68.88**).
- **Post-Peak Week (2011-11-21):** Volume dipped to 29,820 units, and models over-forecasted again (LightGBM bias **+45.13**).
- **Conclusion:** Applying a global additive level shift constant would worsen errors in weeks like Oct 10 and Nov 21. The shift is seasonal and calendar-driven, requiring UK trading day and bank holiday features (G2) and YoY prior-year seasonal indexes (G3).

### Finding 5: `moving_avg_4` beats default ML models due to local responsiveness without cross-product parameter contamination
The numbers explain why the simple 4-week moving average wins:
- `moving_avg_4` achieved the lowest WAPE overall (**0.4801**) and in **every volume segment** (low: **0.5350**, medium: **0.5368**, high: **0.4296**).
- In the high-volume segment (which represents 52.4% of all test demand), `moving_avg_4` achieved **0.4296 WAPE**, outperforming Ridge (**0.4352**), Random Forest (**0.4825**), and LightGBM (**0.4840**).
- `moving_avg_4` adapts locally to each product's recent 4 weeks without cross-product feature contamination, whereas default LightGBM uses global decision trees trained on raw units, making it vulnerable to variance from other products' spikes.
- **Conclusion:** For a machine learning model to beat `moving_avg_4`, it must combine the local anchoring of a rolling mean with regularized tree complexity and calendar/prior-year signals, validating the M6-M7 roadmap.
