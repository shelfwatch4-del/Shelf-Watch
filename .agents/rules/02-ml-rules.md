---
trigger: always_on
description: "Machine learning, time-series splitting, feature, and evaluation rules."
---

# ML Rules

- **Splitting:** Chronological split only; never use random splits.
- **Leakage Prevention:** Every lag and rolling feature must apply `shift(1)` before rolling.
- **Hyperparameter Tuning:** Tune only with `TimeSeriesSplit` inside the training period.
- **Test Set Isolation:** Never touch the test set for tuning, imputation fitting, capping, or product selection.
- **Evaluation Metrics:** Report `MAE`, `RMSE`, `WAPE`, and `bias`; never use `MAPE`.
- **Post-Processing:** Clip all demand predictions at `0`.

