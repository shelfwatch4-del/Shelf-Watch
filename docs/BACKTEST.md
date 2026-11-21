# Replenishment-Policy Backtest (FR-R-9)

> **Target Stakeholder:** Retail Inventory and Replenishment Manager  
> **SRS Reference:** FR-R-9, Section 1.6, Section 6.2  
> **Contracts:** C14 (`policy_backtest.csv`), C15 (`policy_backtest_weekly.csv`)  
> **Period:** 8 full test weeks (`2011-10-10` to `2011-11-28`, 100 products, 800 product-weeks)  

## 1. Executive Summary

This backtest simulates a weekly-review order-up-to $(R, S)$ replenishment policy across the 8 test weeks under realistic retail inventory dynamics:
- **Identical Initial Position:** Every policy begins with exactly 3 weeks of the product's average training demand (deterministic, no noise).
- **Identical Demand:** Unmet customer demand is **lost** (not backordered).
- **Deterministic Ordering:** Review occurs weekly on Monday. Orders arrive after $L$ weeks.
- **Policies Evaluated:**
  1. `naive`: Orders up to $(L + 1) \times \text{lag}_1$.
  2. `moving_avg_4`: Orders up to $(L + 1) \times \text{roll\_mean}_4$.
  3. `best_model`: Orders up to $(L + 1) \times \text{forecast}_{\text{LightGBM}}$ (unbuffered point forecast).
  4. `best_model_safety`: Orders up to $(L + 1) \times \text{forecast}_{\text{LightGBM}} + z \times \sigma_{\text{error}} \times \sqrt{L + 1}$ ($z = 1.6449$ for 95% service level).

> [!IMPORTANT]
> **Simplification Note (SRS 6.4 & FR-R-9):**
> As in the risk rule, the one-step forecast is reused as the demand proxy across all $(L + 1)$ weeks of the lead-time cycle. Multi-step dynamic forecasting is not used during the lead-time window.

---

## 2. Policy Comparison Results (Overall `ALL`)

| Policy | Lead Time ($L$) | Fill Rate | Stockout Weeks | Avg On-Hand Units | Unmet Units |
|---|---|---|---|---|---|
| `naive` | 1 | **83.75%** | 172 | 342.8 | 38,372.9 |
| `moving_avg_4` | 1 | **78.92%** | 239 | 227.1 | 49,776.2 |
| `best_model` | 1 | **79.98%** | 209 | 221.2 | 47,263.1 |
| `best_model_safety` | 1 | **94.51%** | 47 | 465.4 | 12,957.8 |
| `naive` | 2 | **82.30%** | 175 | 432.2 | 41,786.9 |
| `moving_avg_4` | 2 | **78.99%** | 214 | 258.6 | 49,602.9 |
| `best_model` | 2 | **80.67%** | 191 | 247.1 | 45,649.1 |
| `best_model_safety` | 2 | **93.48%** | 54 | 521.6 | 15,390.5 |

---

## 3. Results by ABC Revenue Class

| Policy | Lead Time ($L$) | ABC Class | Fill Rate | Stockout Weeks | Avg On-Hand Units | Unmet Units |
|---|---|---|---|---|---|---|
| `naive` | 1 | `A` | 83.96% | 87 | 353.0 | 21,319.8 |
| `naive` | 1 | `B` | 83.85% | 52 | 306.0 | 9,918.8 |
| `naive` | 1 | `C` | 82.92% | 33 | 363.2 | 7,134.3 |
| `moving_avg_4` | 1 | `A` | 79.32% | 116 | 234.9 | 27,497.2 |
| `moving_avg_4` | 1 | `B` | 78.88% | 75 | 216.8 | 12,971.2 |
| `moving_avg_4` | 1 | `C` | 77.72% | 48 | 222.7 | 9,307.9 |
| `best_model` | 1 | `A` | 79.97% | 102 | 226.7 | 26,629.0 |
| `best_model` | 1 | `B` | 80.77% | 63 | 214.9 | 11,809.5 |
| `best_model` | 1 | `C` | 78.88% | 44 | 217.0 | 8,824.6 |
| `best_model_safety` | 1 | `A` | 93.86% | 24 | 479.3 | 8,162.8 |
| `best_model_safety` | 1 | `B` | 95.89% | 16 | 428.6 | 2,524.9 |
| `best_model_safety` | 1 | `C` | 94.57% | 7 | 478.5 | 2,270.0 |
| `naive` | 2 | `A` | 82.68% | 86 | 467.2 | 23,030.3 |
| `naive` | 2 | `B` | 82.18% | 51 | 370.3 | 10,942.5 |
| `naive` | 2 | `C` | 81.29% | 38 | 430.4 | 7,814.1 |
| `moving_avg_4` | 2 | `A` | 79.00% | 101 | 282.1 | 27,915.0 |
| `moving_avg_4` | 2 | `B` | 79.00% | 68 | 231.7 | 12,896.5 |
| `moving_avg_4` | 2 | `C` | 78.95% | 45 | 241.1 | 8,791.5 |
| `best_model` | 2 | `A` | 80.50% | 87 | 256.5 | 25,919.4 |
| `best_model` | 2 | `B` | 81.32% | 59 | 238.8 | 11,471.3 |
| `best_model` | 2 | `C` | 80.23% | 45 | 237.0 | 8,258.4 |
| `best_model_safety` | 2 | `A` | 92.34% | 31 | 539.2 | 10,187.9 |
| `best_model_safety` | 2 | `B` | 95.51% | 15 | 478.8 | 2,756.3 |
| `best_model_safety` | 2 | `C` | 94.14% | 8 | 533.7 | 2,446.3 |

---

## 4. Manager's Trade-Off Analysis

### 1. Which Policy Wins on Fill Rate?
**`best_model_safety` decisively wins on service level**, delivering **94.51%** fill rate at $L=1$ (and **93.48%** at $L=2$), closely tracking the manager's target 95% service level.
Stockout weeks plummet from **209** down to **47** at $L=1$ (a **77.5% reduction in stockout occurrences** across the 800 product-weeks).

### 2. At What Cost in On-Hand Stock?
Achieving this 93%+ service level requires holding **465.4 average on-hand units**, compared to **221.2 units** for raw `best_model`, **227.1 units** for `moving_avg_4`, and **342.8 units** for `naive`.
This additional inventory (~244.2 units per product over raw best model) represents the required safety buffer to absorb retail demand volatility and Q4 seasonal peaks without losing sales.

### 3. Comparison of Unbuffered Point Forecasts (Differences Are Small)
- **Point forecast differences are small:** `best_model` achieves a fill rate of **79.98%**, which is virtually identical to `moving_avg_4` at **78.92%** (a negligible difference of 1.06 percentage points), while holding similar ending inventory (221.2 vs 227.1 units).
- **Why `naive` achieves a higher fill rate (83.75%) without safety stock:** In a Q4 seasonal surge where demand rises week over week, a 1-week lag systematically orders more inventory than a smoothed or mean-reverting forecast. However, this higher fill rate comes at the cost of carrying **342.8 average units** (+121.6 units, or ~55.0% more stock than `best_model`).
- **The core empirical insight:** No unbuffered point forecast alone delivers acceptable retail availability (all hover between 78% and 84% fill rate with 172 to 239 stockout weeks). The true driver of retail availability is **safety stock buffering forecast error**, which lifts availability to >93% while keeping stockouts under 55.

---

## 5. Stakeholder Recommendation

1. **Adopt `best_model_safety` for Class A products:** The cost of lost sales on high-revenue Class A items far outweighs the carrying cost of safety stock.
2. **Maintain Lean Buffers for Class C:** For intermittent or slow-moving Class C items, the manager should dial down the target service level (e.g. to 85% or 90%) to prevent cash lock-up in dead stock.
3. **Monitor Lead Time Inflation:** Increasing supplier lead time from $L=1$ to $L=2$ increases ending stock requirements by ~12% while lowering unbuffered fill rates. Prompt ordering via the Home Action list is vital.

