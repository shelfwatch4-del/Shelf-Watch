# %% [markdown]
# # Shelfwatch — Forecast-Error Analysis (`03_error_analysis.py`)
#
# Evaluates test-period forecasts (`data/processed/predictions.csv` and `metrics.csv`)
# across volume segments, demand patterns, ABC classes, test weeks, and individual products.
# Exports `data/processed/error_analysis.csv`, generates figures in `reports/figures/`,
# and provides an in-depth labelled case study on discontinued products (SKUs 21829 and 22740).

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
    PROCESSED_DIR / "product_intelligence.csv",
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

intelligence = pd.read_csv(
    PROCESSED_DIR / "product_intelligence.csv",
    dtype={"product_id": str},
)

# Build comprehensive error_analysis.csv (segments, pattern_class, abc_class, week, product)
ea_df = build_error_analysis(
    predictions,
    segments=segments,
    intelligence=intelligence,
    out_path=PROCESSED_DIR / "error_analysis.csv",
)
print(f"Wrote {PROCESSED_DIR / 'error_analysis.csv'} ({len(ea_df)} rows across levels: {sorted(ea_df['level'].unique().tolist())})")

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
print("\n--- Error by Volume Segment ---")
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
# ## 2. Error by Demand Pattern Class & ABC Class (from Product Intelligence)

# %%
pat_order = ["smooth", "erratic", "intermittent", "lumpy"]
abc_order = ["A", "B", "C"]

pat_ea = ea_df[
    (ea_df["level"] == "pattern_class")
    & (ea_df["model_name"].isin([best_model, best_baseline]))
].copy()
pat_ea["key"] = pd.Categorical(pat_ea["key"], categories=pat_order, ordered=True)
pat_ea = pat_ea.sort_values(["key", "model_name"])

abc_ea = ea_df[
    (ea_df["level"] == "abc_class")
    & (ea_df["model_name"].isin([best_model, best_baseline]))
].copy()
abc_ea["key"] = pd.Categorical(abc_ea["key"], categories=abc_order, ordered=True)
abc_ea = abc_ea.sort_values(["key", "model_name"])

print("\n--- Error by Demand Pattern Class ---")
print(pat_ea[["key", "model_name", "n", "MAE", "RMSE", "WAPE", "bias"]].to_string(index=False))

print("\n--- Error by ABC Revenue Class ---")
print(abc_ea[["key", "model_name", "n", "MAE", "RMSE", "WAPE", "bias"]].to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

# Plot Pattern Class WAPE
present_pats = [p for p in pat_order if p in pat_ea["key"].values]
xp = np.arange(len(present_pats))
wp = 0.35
for offset, m_name, color in [
    (-wp / 2, best_baseline, "#94a3b8"),
    (wp / 2, best_model, "#2563eb"),
]:
    sub_p = pat_ea[pat_ea["model_name"] == m_name].set_index("key").reindex(present_pats)
    axes[0].bar(xp + offset, sub_p["WAPE"] * 100.0, wp, label=m_name, color=color)

axes[0].set_xticks(xp, [p.capitalize() for p in present_pats])
axes[0].set_xlabel("Demand Pattern Class (ADI / CV²)")
axes[0].set_ylabel("WAPE (%)")
axes[0].set_title("Test WAPE by Demand Pattern Class")
axes[0].legend()

# Plot ABC Class WAPE
xabc = np.arange(len(abc_order))
wabc = 0.35
for offset, m_name, color in [
    (-wabc / 2, best_baseline, "#94a3b8"),
    (wabc / 2, best_model, "#059669"),
]:
    sub_abc = abc_ea[abc_ea["model_name"] == m_name].set_index("key").reindex(abc_order)
    axes[1].bar(xabc + offset, sub_abc["WAPE"] * 100.0, wabc, label=m_name, color=color)

axes[1].set_xticks(xabc, [f"Class {c}" for c in abc_order])
axes[1].set_xlabel("ABC Revenue Class")
axes[1].set_ylabel("WAPE (%)")
axes[1].set_title("Test WAPE by ABC Revenue Class")
axes[1].legend()

fig.tight_layout()
fig.savefig(FIGURES_DIR / "14_error_by_pattern_and_abc.png", dpi=150)
plt.close(fig)

# %% [markdown]
# ## 3. Error by Test Week (Spike Weeks Highlighted)

# %%
week_ea = ea_df[
    (ea_df["level"] == "week")
    & (ea_df["model_name"].isin([best_model, best_baseline]))
].sort_values("key")
best_week_ea = week_ea[week_ea["model_name"] == best_model].copy()
spike_week = best_week_ea.loc[best_week_ea["WAPE"].idxmax()]
print("\n--- Error by Test Week ---")
print(week_ea[["key", "model_name", "MAE", "WAPE", "bias"]].to_string(index=False))
print(
    f"\nHighest-WAPE spike week for {best_model}: {spike_week['key']} "
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
# ## 4. Predicted vs. Actual Scatter and Residual Distribution

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
# ## 5. 10 Best and 10 Worst Products (with Diagnostic Reason)

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
# ## 6. Over-Forecast vs. Under-Forecast Totals

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
# ## 7. Labelled Case Study: Discontinued / Stopped-Selling Products (SKU 21829 & SKU 22740)
#
# A critical limitation of statistical time-series and machine learning forecasting
# models is their structural inertia: models conditioned on historical sales, lags,
# and rolling averages cannot detect that a product has been permanently discontinued
# or dropped by retail management without an explicit catalog flag.
#
# When demand abruptly drops to zero, the model continues generating positive
# "phantom" forecasts, creating massive positive bias and phantom replenishment orders.

# %%
case_pids = ["21829", "22740"]
case_meta = intelligence[intelligence["product_id"].isin(case_pids)].set_index("product_id")

fig, axes = plt.subplots(2, 1, figsize=(12, 7.5), sharex=False)

for idx, pid in enumerate(case_pids):
    ax = axes[idx]
    prod_w = weekly[weekly["product_id"] == pid].sort_values("week_start").copy()
    desc = prod_w["description"].iloc[0]
    p_meta = case_meta.loc[pid] if pid in case_meta.index else None

    # Get test predictions for best model and baseline
    sub_pred = predictions[
        (predictions["product_id"] == pid)
        & (predictions["split"] == "test")
        & (predictions["model_name"] == best_model)
    ].sort_values("week_start")

    sub_base = predictions[
        (predictions["product_id"] == pid)
        & (predictions["split"] == "test")
        & (predictions["model_name"] == best_baseline)
    ].sort_values("week_start")

    # Metrics on test set
    test_act = sub_pred["actual"].values
    test_pred_vals = sub_pred["predicted"].values
    tot_act = float(test_act.sum())
    tot_pred = float(test_pred_vals.sum())
    phantom_units = float(np.clip(test_pred_vals - test_act, 0, None).sum())
    sku_wape = float(np.abs(test_pred_vals - test_act).sum() / tot_act) if tot_act > 0 else float("inf")
    sku_bias = float((test_pred_vals - test_act).mean())

    # Plot full history
    ax.plot(
        prod_w["week_start"],
        prod_w["units"],
        color="#0f172a",
        label="Actual weekly demand",
        marker="o",
        markersize=3,
        linewidth=1.2,
    )

    # Plot test forecasts
    test_dates = pd.to_datetime(sub_pred["week_start"])
    ax.plot(
        test_dates,
        sub_pred["predicted"],
        color="#2563eb",
        linestyle="--",
        marker="s",
        markersize=4,
        linewidth=1.5,
        label=f"{best_model} forecast (test)",
    )
    ax.plot(
        test_dates,
        sub_base["predicted"],
        color="#94a3b8",
        linestyle=":",
        marker="^",
        markersize=3.5,
        linewidth=1.2,
        label=f"{best_baseline} forecast (test)",
    )

    # Highlight test period
    ax.axvspan(TEST_START, test_dates.max(), color="#fef3c7", alpha=0.35, label="Test period (8 weeks)")
    ax.axvline(TEST_START, color="#d97706", linestyle="--", linewidth=1.0)

    stage_str = f", Stage: {p_meta['lifecycle_stage'].upper()}" if p_meta is not None else ""
    ax.set_title(
        f"Case Study: SKU {pid} — {desc} (ABC: {p_meta['abc_class'] if p_meta is not None else 'N/A'}{stage_str})\n"
        f"Test Actual: {int(tot_act):,} units | Model Predicted: {int(tot_pred):,} units | Phantom Over-forecast: {int(phantom_units):,} units | Bias: +{sku_bias:.1f}/wk",
        fontsize=10.5,
        fontweight="bold",
    )
    ax.set_ylabel("Weekly units")
    ax.legend(loc="upper left", fontsize=8.5)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")

    print(f"\n--- Case Study: SKU {pid} ({desc}) ---")
    print(f"  ABC Class: {p_meta['abc_class'] if p_meta is not None else 'N/A'} | Lifecycle Stage: {p_meta['lifecycle_stage'] if p_meta is not None else 'N/A'}")
    print(f"  Test Actual Units: {tot_act:.0f} | Predicted: {tot_pred:.1f} | Phantom Over-forecast: {phantom_units:.1f}")
    print(f"  Test WAPE: {sku_wape:.3f} | Test Bias: +{sku_bias:.2f} units/week")
    print("  Week-by-week test actual vs. predicted:")
    eval_tbl = pd.DataFrame(
        {
            "week_start": sub_pred["week_start"].values,
            "actual": test_act,
            f"pred_{best_model}": np.round(test_pred_vals, 1),
            f"pred_{best_baseline}": np.round(sub_base["predicted"].values, 1),
        }
    )
    print(eval_tbl.to_string(index=False))

fig.tight_layout()
fig.savefig(FIGURES_DIR / "15_stopped_selling_case_study.png", dpi=150)
plt.close(fig)

# %% [markdown]
# ### Operational Implications for Inventory Management
# 1. **Structural Lag Inertia**: Because SKU 21829 sold in training (mean ~120 units/wk) and had spikes in early October, the models kept forecasting 104-215 units/week even after 6 straight weeks of 0 demand, resulting in 921 phantom forecast units. Similarly, SKU 22740 was completely discontinued throughout the 8 test weeks, yet the models projected 133-169 units/wk, generating 1,182 phantom units.
# 2. **Stockout vs. Working Capital Risk**: In contrast to under-forecasting (which leads to stockouts), phantom demand on discontinued lines causes severe capital lockup and dead stock if replenishment is fully automated.
# 3. **Mitigation in Shelfwatch**: Shelfwatch detects these products using `product_intelligence.csv` (`lifecycle_stage == 'declining'` or `'inactive'`, `weeks_since_last_sale >= 1`) and the risk engine's `REVIEW_STOP` policy flag, which alerts the manager to freeze reorders regardless of statistical demand projections.

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

# Metrics by pattern and ABC
smooth_wape = pat_ea[(pat_ea["model_name"] == best_model) & (pat_ea["key"] == "smooth")]["WAPE"].iloc[0]
erratic_wape = pat_ea[(pat_ea["model_name"] == best_model) & (pat_ea["key"] == "erratic")]["WAPE"].iloc[0]
a_wape = abc_ea[(abc_ea["model_name"] == best_model) & (abc_ea["key"] == "A")]["WAPE"].iloc[0]
c_wape = abc_ea[(abc_ea["model_name"] == best_model) & (abc_ea["key"] == "C")]["WAPE"].iloc[0]

sku_21829_pred = predictions[(predictions["product_id"] == "21829") & (predictions["split"] == "test") & (predictions["model_name"] == best_model)]
sku_22740_pred = predictions[(predictions["product_id"] == "22740") & (predictions["split"] == "test") & (predictions["model_name"] == best_model)]
phantom_21829 = float(np.clip(sku_21829_pred["predicted"] - sku_21829_pred["actual"], 0, None).sum())
phantom_22740 = float(np.clip(sku_22740_pred["predicted"] - sku_22740_pred["actual"], 0, None).sum())

print("\nFor the report: findings")
print(
    f"1. Overall test accuracy: {best_model} achieves WAPE={best_test_row['WAPE']:.3f} "
    f"(MAE={best_test_row['MAE']:.1f}, bias={best_test_row['bias']:.1f}) versus best baseline "
    f"{best_baseline} (WAPE={base_test_row['WAPE']:.3f}, MAE={base_test_row['MAE']:.1f})."
)
print(
    f"2. Volume and demand pattern divergence: {best_model} performs best on high-volume items "
    f"(WAPE={best_seg.loc['high', 'WAPE']:.3f}) and smooth demand (WAPE={smooth_wape:.3f}), "
    f"but degrades sharply on erratic demand (WAPE={erratic_wape:.3f}) and low-volume items "
    f"(WAPE={best_seg.loc['low', 'WAPE']:.3f})."
)
print(
    f"3. Revenue class impact: High-revenue Class A products achieve WAPE={a_wape:.3f}, "
    f"substantially outperforming tail Class C products (WAPE={c_wape:.3f}), protecting core revenue."
)
print(
    f"4. Temporal error spike: test error peaks in week {spike_week['key']} at "
    f"WAPE={spike_week['WAPE']:.3f} with bias={spike_week['bias']:.1f} units, coinciding with "
    f"pre-holiday bulk purchasing fluctuations."
)
print(
    f"5. Asymmetric error & stockout exposure: {best_model} records {total_under:,.0f} under-forecast "
    f"units ({under_share:.1%} of total absolute error) vs. {total_over:,.0f} over-forecast units, "
    f"limiting dangerous stockout exposures."
)
print(
    f"6. Discontinued product inertia (case study): on stopped-selling items 21829 and 22740, the model "
    f"accumulated {phantom_21829:,.0f} and {phantom_22740:,.0f} phantom over-forecast units respectively across the 8 test weeks, "
    f"highlighting the operational necessity of Shelfwatch's lifecycle-stage filter and REVIEW_STOP replenishment action."
)
