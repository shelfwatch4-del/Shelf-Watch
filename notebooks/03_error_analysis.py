# %% [markdown]
# # Shelfwatch — Forecast-Error Analysis (`03_error_analysis.py`)
#
# Evaluates Member 2's test-period forecasts (`data/processed/predictions.csv`
# and `metrics.csv`) across volume segments, test weeks, and products, exports
# `data/processed/error_analysis.csv`, and saves figures to `reports/figures/`.

# %%
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.error_analysis import build_error_analysis

PROCESSED_DIR = REPO_ROOT / "data" / "processed"
FIGURES_DIR = REPO_ROOT / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

TEST_START = pd.Timestamp("2011-10-10")
CLOSED_WEEK = pd.Timestamp("2010-12-27")
BASELINE_MODELS = ("naive", "moving_avg_4")

plt.rcParams.update(
    {
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "font.size": 10,
    }
)

# %%
required_files = [
    PROCESSED_DIR / "predictions.csv",
    PROCESSED_DIR / "metrics.csv",
    PROCESSED_DIR / "product_segments.csv",
    PROCESSED_DIR / "weekly_demand.parquet",
]
missing = [str(p) for p in required_files if not p.exists()]
if missing:
    raise FileNotFoundError(
        "Cannot run forecast-error analysis until Member 2 delivers upstream "
        f"artifacts. Missing file(s): {missing}"
    )

predictions = pd.read_csv(
    PROCESSED_DIR / "predictions.csv",
    dtype={"product_id": str, "model_name": str, "split": str},
)
metrics = pd.read_csv(
    PROCESSED_DIR / "metrics.csv",
    dtype={"model_name": str, "split": str},
)
segments = pd.read_csv(
    PROCESSED_DIR / "product_segments.csv",
    dtype={"product_id": str, "segment": str},
)
weekly = pd.read_parquet(PROCESSED_DIR / "weekly_demand.parquet")
weekly["product_id"] = weekly["product_id"].astype(str)
weekly["week_start"] = pd.to_datetime(weekly["week_start"])

ea_df = build_error_analysis(
    predictions,
    segments,
    out_path=PROCESSED_DIR / "error_analysis.csv",
)
print(f"Wrote {PROCESSED_DIR / 'error_analysis.csv'} ({len(ea_df)} rows)")

# Select best overall model and best baseline by validation WAPE (tie-break |val bias|)
val_metrics = metrics[metrics["split"] == "val"].copy()
val_metrics["abs_bias"] = val_metrics["bias"].abs()
best_model = (
    val_metrics.sort_values(["WAPE", "abs_bias"], ascending=[True, True])
    .iloc[0]["model_name"]
)
val_baselines = val_metrics[val_metrics["model_name"].isin(BASELINE_MODELS)]
best_baseline = (
    val_baselines.sort_values(["WAPE", "abs_bias"], ascending=[True, True])
    .iloc[0]["model_name"]
)
print(f"Best model (val WAPE): {best_model} | Best baseline: {best_baseline}")

# %% [markdown]
# ## 1. Error by Volume Segment (Best Model vs. Best Baseline)

# %%
seg_order = ["low", "medium", "high"]
seg_ea = ea_df[
    (ea_df["level"] == "segment")
    & (ea_df["model_name"].isin([best_model, best_baseline]))
].copy()
seg_ea["key"] = pd.Categorical(seg_ea["key"], categories=seg_order, ordered=True)
seg_ea = seg_ea.sort_values(["key", "model_name"])
print(seg_ea[["key", "model_name", "n", "MAE", "RMSE", "WAPE", "bias"]].to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
x = np.arange(len(seg_order))
width = 0.35

for offset, m_name, color in [
    (-width / 2, best_baseline, "#94a3b8"),
    (width / 2, best_model, "#2563eb"),
]:
    sub = seg_ea[seg_ea["model_name"] == m_name].set_index("key").reindex(seg_order)
    axes[0].bar(x + offset, sub["WAPE"] * 100.0, width, label=m_name, color=color)
    axes[1].bar(x + offset, sub["MAE"], width, label=m_name, color=color)

axes[0].set_xticks(x, ["Low", "Medium", "High"])
axes[0].set_xlabel("Volume segment")
axes[0].set_ylabel("WAPE (%)")
axes[0].set_title("Test WAPE by Volume Segment")
axes[0].legend()

axes[1].set_xticks(x, ["Low", "Medium", "High"])
axes[1].set_xlabel("Volume segment")
axes[1].set_ylabel("MAE (units / week)")
axes[1].set_title("Test MAE by Volume Segment")
axes[1].legend()

fig.tight_layout()
fig.savefig(FIGURES_DIR / "10_error_by_segment.png", dpi=150)
plt.close(fig)

# %% [markdown]
# ## 2. Error by Test Week (Spike Weeks Highlighted)

# %%
week_ea = ea_df[
    (ea_df["level"] == "week")
    & (ea_df["model_name"].isin([best_model, best_baseline]))
].sort_values("key")
best_week_ea = week_ea[week_ea["model_name"] == best_model].copy()
spike_week = best_week_ea.loc[best_week_ea["WAPE"].idxmax()]
print(week_ea[["key", "model_name", "MAE", "WAPE", "bias"]].to_string(index=False))
print(
    f"Highest-WAPE spike week for {best_model}: {spike_week['key']} "
    f"(WAPE={spike_week['WAPE']:.3f}, bias={spike_week['bias']:.2f})"
)

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
weeks_sorted = sorted(best_week_ea["key"].unique())

for m_name, color, marker in [
    (best_baseline, "#64748b", "s"),
    (best_model, "#2563eb", "o"),
]:
    sub = week_ea[week_ea["model_name"] == m_name].set_index("key").reindex(weeks_sorted)
    axes[0].plot(weeks_sorted, sub["WAPE"] * 100.0, marker=marker, label=m_name, color=color)
    axes[1].plot(weeks_sorted, sub["bias"], marker=marker, label=m_name, color=color)

axes[1].axhline(0.0, color="#111827", linestyle="--", linewidth=1.0)
for ax, ylabel, title in [
    (axes[0], "WAPE (%)", "Test WAPE by Week (Spike Weeks Annotated)"),
    (axes[1], "Mean bias (units / week)", "Test Bias by Week (Predicted - Actual)"),
]:
    ax.set_xlabel("Test week start (Monday)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    ax.legend()

axes[0].annotate(
    f"Peak WAPE: {spike_week['key']}",
    xy=(spike_week["key"], spike_week["WAPE"] * 100.0),
    xytext=(0, 12),
    textcoords="offset points",
    ha="center",
    fontsize=8.5,
    color="#dc2626",
)

fig.tight_layout()
fig.savefig(FIGURES_DIR / "11_error_by_test_week.png", dpi=150)
plt.close(fig)

# %% [markdown]
# ## 3. Predicted vs. Actual Scatter and Residual Distribution

# %%
test_preds = predictions[
    (predictions["split"] == "test") & (predictions["model_name"] == best_model)
].copy()
test_preds["residual"] = test_preds["predicted"] - test_preds["actual"]

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
max_val = float(max(test_preds["actual"].max(), test_preds["predicted"].max())) * 1.05
axes[0].scatter(
    test_preds["actual"],
    test_preds["predicted"],
    alpha=0.45,
    s=18,
    color="#2563eb",
    edgecolors="none",
)
axes[0].plot([0, max_val], [0, max_val], color="#dc2626", linestyle="--", linewidth=1.4, label="Ideal (y = x)")
axes[0].set_xlim(0, max_val)
axes[0].set_ylim(0, max_val)
axes[0].set_xlabel("Actual weekly demand (units)")
axes[0].set_ylabel("Predicted weekly demand (units)")
axes[0].set_title(f"Predicted vs. Actual ({best_model}, Test Set)")
axes[0].legend()

axes[1].hist(test_preds["residual"], bins=40, color="#0d9488", edgecolor="white", alpha=0.86)
axes[1].axvline(0.0, color="#dc2626", linestyle="--", linewidth=1.4, label="Zero residual")
axes[1].axvline(
    float(test_preds["residual"].mean()),
    color="#111827",
    linestyle=":",
    linewidth=1.4,
    label=f"Mean bias: {test_preds['residual'].mean():.1f}",
)
axes[1].set_xlabel("Residual = predicted - actual (units)")
axes[1].set_ylabel("Product-weeks (count)")
axes[1].set_title(f"Test Residual Distribution ({best_model})")
axes[1].legend()

fig.tight_layout()
fig.savefig(FIGURES_DIR / "12_pred_vs_actual_and_residuals.png", dpi=150)
plt.close(fig)

# %% [markdown]
# ## 4. 10 Best and 10 Worst Products (with Diagnostic Reason)

# %%
def _diagnose_product_reason(pid: str, prod_weekly: pd.DataFrame) -> str:
    """Assign a one-line diagnostic reason based on observed demand behavior."""
    open_rows = prod_weekly[prod_weekly["week_start"] != CLOSED_WEEK]
    train_rows = open_rows[open_rows["week_start"] < TEST_START]
    test_rows = open_rows[open_rows["week_start"] >= TEST_START]

    zero_share = float((open_rows["units"] == 0).mean()) if len(open_rows) else 0.0
    train_mean = float(train_rows["units"].mean()) if len(train_rows) else 0.0
    test_mean = float(test_rows["units"].mean()) if len(test_rows) else 0.0
    max_to_median = (
        float(open_rows["units"].max() / max(open_rows["units"].median(), 1.0))
        if len(open_rows)
        else 1.0
    )
    shift_ratio = test_mean / max(train_mean, 1.0)

    if zero_share >= 0.08:
        return f"intermittent demand ({zero_share:.0%} zero open weeks)"
    if max_to_median >= 4.0:
        return f"bulk orders (peak {int(open_rows['units'].max()):,} vs median {int(open_rows['units'].median()):,} units)"
    if abs(shift_ratio - 1.0) >= 0.20:
        return f"level shift (test/train mean {shift_ratio:.2f}x)"
    return f"steady demand (CV={open_rows['units'].std() / max(open_rows['units'].mean(), 1.0):.2f})"


prod_ea = ea_df[
    (ea_df["level"] == "product") & (ea_df["model_name"] == best_model)
].copy()
desc_map = weekly.drop_duplicates("product_id").set_index("product_id")["description"].to_dict()
prod_ea["description"] = prod_ea["key"].map(desc_map)
prod_ea["reason"] = [
    _diagnose_product_reason(pid, weekly[weekly["product_id"] == pid])
    for pid in prod_ea["key"]
]

best10 = prod_ea.nsmallest(10, "WAPE")[["key", "description", "WAPE", "MAE", "bias", "reason"]]
worst10 = prod_ea.nlargest(10, "WAPE")[["key", "description", "WAPE", "MAE", "bias", "reason"]]

print("\n10 Best Products (lowest test WAPE):")
print(best10.to_string(index=False))
print("\n10 Worst Products (highest test WAPE):")
print(worst10.to_string(index=False))

# %% [markdown]
# ## 5. Over-Forecast vs. Under-Forecast Totals

# %%
over_under = (
    ea_df[
        (ea_df["level"] == "segment")
        & (ea_df["model_name"].isin([best_model, best_baseline]))
    ]
    .groupby("model_name")[["over_forecast_units", "under_forecast_units"]]
    .sum()
)
print("\nOver- vs. under-forecast totals (units):")
print(over_under.round(1).to_string())

fig, ax = plt.subplots(figsize=(7.8, 4.4))
models_plot = [best_baseline, best_model]
x_m = np.arange(len(models_plot))
w = 0.34
ax.bar(
    x_m - w / 2,
    [over_under.loc[m, "over_forecast_units"] for m in models_plot],
    w,
    label="Over-forecast units",
    color="#3b82f6",
)
ax.bar(
    x_m + w / 2,
    [over_under.loc[m, "under_forecast_units"] for m in models_plot],
    w,
    label="Under-forecast units",
    color="#ef4444",
)
ax.set_xticks(x_m, models_plot)
ax.set_xlabel("Model")
ax.set_ylabel("Total error magnitude across test set (units)")
ax.set_title("Over-Forecast vs. Under-Forecast Units on Test Set")
ax.legend()
fig.tight_layout()
fig.savefig(FIGURES_DIR / "13_over_vs_under_forecast.png", dpi=150)
plt.close(fig)

# %% [markdown]
# Under-forecasting is operationally more dangerous than over-forecasting because setting replenishment targets below true customer demand directly depletes on-hand stock before the next order arrives, causing stockouts and lost sales.

# %% [markdown]
# ## For the report: findings

# %%
best_test_row = metrics[(metrics["model_name"] == best_model) & (metrics["split"] == "test")].iloc[0]
base_test_row = metrics[(metrics["model_name"] == best_baseline) & (metrics["split"] == "test")].iloc[0]
best_seg = seg_ea[seg_ea["model_name"] == best_model].set_index("key")
total_over = float(over_under.loc[best_model, "over_forecast_units"])
total_under = float(over_under.loc[best_model, "under_forecast_units"])
under_share = total_under / max(total_over + total_under, 1.0)
worst_prod = worst10.iloc[0]

print("\nFor the report: findings")
print(
    f"1. Overall test accuracy: {best_model} achieves WAPE={best_test_row['WAPE']:.3f} "
    f"(MAE={best_test_row['MAE']:.1f}, bias={best_test_row['bias']:.1f}) versus best baseline "
    f"{best_baseline} (WAPE={base_test_row['WAPE']:.3f}, MAE={base_test_row['MAE']:.1f})."
)
print(
    f"2. Segment accuracy: across low/medium/high volume tertiles, {best_model} WAPE is "
    f"{best_seg.loc['low', 'WAPE']:.3f} (low), {best_seg.loc['medium', 'WAPE']:.3f} (medium), "
    f"and {best_seg.loc['high', 'WAPE']:.3f} (high)."
)
print(
    f"3. Temporal error spike: test error peaks in week {spike_week['key']} at "
    f"WAPE={spike_week['WAPE']:.3f} with bias={spike_week['bias']:.1f} units."
)
print(
    f"4. Asymmetric error & stockout exposure: {best_model} records {total_under:,.0f} under-forecast "
    f"units ({under_share:.1%} of total absolute error) vs. {total_over:,.0f} over-forecast units."
)
print(
    f"5. Product extremes: the highest-WAPE SKU is {worst_prod['key']} "
    f"(WAPE={worst_prod['WAPE']:.3f}, MAE={worst_prod['MAE']:.1f}), driven by {worst_prod['reason']}."
)

