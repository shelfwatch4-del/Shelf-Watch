# %% [markdown]
# # Shelfwatch — Exploratory Data Analysis (`01_eda.py`)
#
# Analyzes the cleaned UCI Online Retail transactions, the 52-week C1 demand
# table (`data/processed/weekly_demand.parquet`), and shop closure records
# (`data/processed/closed_weeks.csv`), producing 9 report figures and
# `data/processed/product_segments.csv`.

# %%
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data_prep import (
    cap_quantity,
    clean_transactions,
    full_week_range,
    load_raw,
    week_start_of,
)

PROCESSED_DIR = REPO_ROOT / "data" / "processed"
RAW_PATH = REPO_ROOT / "data" / "raw" / "Online Retail.xlsx"
FIGURES_DIR = REPO_ROOT / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

TEST_START = pd.Timestamp("2011-10-10")

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
weekly = pd.read_parquet(PROCESSED_DIR / "weekly_demand.parquet")
weekly["week_start"] = pd.to_datetime(weekly["week_start"])
closed_weeks = pd.read_csv(PROCESSED_DIR / "closed_weeks.csv")
closed_dates = pd.to_datetime(closed_weeks["week_start"])
cleaning_log = pd.read_csv(PROCESSED_DIR / "cleaning_log.csv")

raw_tx = load_raw(RAW_PATH)
cleaned_tx, _ = clean_transactions(raw_tx)

print(f"weekly_demand shape: {weekly.shape}, products: {weekly['product_id'].nunique()}")
print(f"closed_weeks: {[str(d.date()) for d in closed_dates]}")
print(f"cleaned_tx shape: {cleaned_tx.shape}")

# %% [markdown]
# ## 1. Rows Remaining After Each Cleaning Step

# %%
fig, ax = plt.subplots(figsize=(9, 4.8))
log_plot = cleaning_log.iloc[::-1].reset_index(drop=True)
bars = ax.barh(log_plot["step"], log_plot["rows_remaining"], color="#2563eb", alpha=0.88)

for bar, val in zip(bars, log_plot["rows_remaining"]):
    ax.text(
        val + 6000,
        bar.get_y() + bar.get_height() / 2,
        f"{val:,}",
        va="center",
        fontsize=9,
    )

ax.set_xlim(0, cleaning_log["rows_remaining"].max() * 1.15)
ax.set_xlabel("Transaction lines remaining (rows)")
ax.set_ylabel("Pipeline cleaning step")
ax.set_title("Rows Remaining After Each Data-Cleaning Step")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "01_cleaning_steps.png", dpi=150)
plt.close(fig)

# %% [markdown]
# ### What to write in the report
# Starting from 541,909 raw UCI transaction lines, removing cancellations (9,288), non-positive quantities (1,336), non-positive prices (1,181), and non-product codes (2,370) leaves 527,734 valid product lines. Trimming the partial boundary weeks drops 27,348 lines to 500,386 full-week rows across 52 weeks, and selecting the Top 100 eligible products retains 76,648 transaction lines that aggregate into 5,147 product-week records.

# %% [markdown]
# ## 2. Total Weekly Units (Test Start & Closed Week Marked)

# %%
weekly_totals = weekly.groupby("week_start")["units"].sum().sort_index()
train_open_totals = weekly_totals[
    (weekly_totals.index < TEST_START) & (~weekly_totals.index.isin(closed_dates))
]
test_totals = weekly_totals[weekly_totals.index >= TEST_START]

print(f"Train weekly total mean (excl. closed): {train_open_totals.mean():,.1f} units/week")
print(f"Test weekly total mean (Oct-Nov): {test_totals.mean():,.1f} units/week")
print(f"Level shift ratio (test / train): {test_totals.mean() / train_open_totals.mean():.2f}x")

fig, ax = plt.subplots(figsize=(10.5, 4.6))
ax.plot(
    weekly_totals.index,
    weekly_totals.values,
    marker="o",
    markersize=4,
    linewidth=1.8,
    color="#1d4ed8",
    label="Total weekly units (Top 100 products)",
)
ax.axvline(
    TEST_START,
    color="#dc2626",
    linestyle="--",
    linewidth=1.6,
    label="Test period start (2011-10-10)",
)

for cw in closed_dates:
    cw_val = float(weekly_totals.get(cw, 0.0))
    ax.scatter(
        [cw],
        [cw_val],
        color="#f59e0b",
        edgecolor="#111827",
        s=110,
        zorder=5,
        label=f"Shop closure week ({cw.strftime('%Y-%m-%d')})",
    )
    ax.annotate(
        f"Closed ({cw.strftime('%Y-%m-%d')})",
        xy=(cw, cw_val),
        xytext=(cw + pd.Timedelta(days=14), cw_val + 4000),
        arrowprops={"arrowstyle": "->", "color": "#92400e"},
        fontsize=9,
        color="#92400e",
    )

ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
ax.set_xlabel("Week start (Monday date)")
ax.set_ylabel("Total demand (units / week)")
ax.set_title("Total Weekly Units Across Top 100 Products (2010-12-06 to 2011-11-28)")
ax.legend(loc="upper left")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "02_total_weekly_units.png", dpi=150)
plt.close(fig)

# %% [markdown]
# ### What to write in the report
# Aggregate demand across the 100 products averages 25,000.7 units per open training week and rises 18.1% to 29,515.6 units per week during the 8-week October–November test period (2011-10-10 to 2011-11-28). The week of 2010-12-27 drops cleanly to 0 units because the retailer closed over Christmas/New Year, so it is flagged in `closed_weeks.csv` as missing rather than genuine zero demand.

# %% [markdown]
# ## 3. Top 15 Products by Total Units

# %%
top15 = (
    weekly.groupby(["product_id", "description"])["units"]
    .sum()
    .nlargest(15)
    .reset_index()
)
top15["label"] = (
    top15["product_id"] + " — " + top15["description"].str.slice(0, 28)
)
top15_plot = top15.iloc[::-1].reset_index(drop=True)

fig, ax = plt.subplots(figsize=(10, 5.6))
bars = ax.barh(top15_plot["label"], top15_plot["units"], color="#0f766e", alpha=0.88)
for bar, val in zip(bars, top15_plot["units"]):
    ax.text(
        val + 500,
        bar.get_y() + bar.get_height() / 2,
        f"{val:,}",
        va="center",
        fontsize=8.5,
    )

ax.set_xlim(0, top15_plot["units"].max() * 1.14)
ax.set_xlabel("Total units sold across 52 weeks (units)")
ax.set_ylabel("Product (StockCode — description)")
ax.set_title("Top 15 Products by Total Units Sold")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "03_top15_products.png", dpi=150)
plt.close(fig)

# %% [markdown]
# ### What to write in the report
# Volume among the Top 100 products is concentrated in a handful of fast-moving gift items: `85099B` (JUMBO BAG RED RETROSPOT, 45,571 units), `22197` (SMALL POPCORN HOLDER, 40,553 units), and `84077` (WORLD WAR 2 GLIDERS ASSTD DESIGNS, 37,701 units) lead all SKUs. Even within the top 15 products, total annual volume spans more than a 2.5x range down to `47566` (PARTY BUNTING, 18,207 units).

# %% [markdown]
# ## 4. Share of Zero-Demand Weeks per Product

# %%
overall_zero_all = float((weekly["units"] == 0).mean())
weekly_open = weekly[~weekly["week_start"].isin(closed_dates)].copy()
overall_zero_open = float((weekly_open["units"] == 0).mean())
zero_share_per_product = weekly_open.groupby("product_id")["units"].apply(
    lambda s: float((s == 0).mean())
)

print(f"Overall zero-demand share (all 5,147 rows): {overall_zero_all:.2%}")
print(f"Overall zero-demand share (excl. closed week): {overall_zero_open:.2%}")
print(
    f"Products with >= 1 zero open week: {(zero_share_per_product > 0).sum()} / "
    f"{len(zero_share_per_product)} (max: {zero_share_per_product.max():.1%})"
)

fig, ax = plt.subplots(figsize=(8.5, 4.5))
ax.hist(
    zero_share_per_product * 100.0,
    bins=16,
    color="#7c3aed",
    edgecolor="white",
    alpha=0.85,
)
ax.axvline(
    overall_zero_open * 100.0,
    color="#dc2626",
    linestyle="--",
    linewidth=1.5,
    label=f"Open-week mean: {overall_zero_open:.2%}",
)
ax.set_xlabel("Share of zero-demand open weeks per product (%)")
ax.set_ylabel("Number of products (count)")
ax.set_title("Distribution of Zero-Demand Week Share per Product (Excl. Closed Week)")
ax.legend()
fig.tight_layout()
fig.savefig(FIGURES_DIR / "04_zero_demand_share.png", dpi=150)
plt.close(fig)

# %% [markdown]
# ### What to write in the report
# Across all 5,147 product-week rows, 3.98% (205 rows) have zero units; excluding the shop-wide closure week `2010-12-27`, 2.18% of open product-weeks record zero demand. Zero weeks are concentrated in 23 of the 100 products (reaching up to 25.5% of open weeks for the most intermittent SKU), which is why MAPE cannot be used and WAPE/MAE/RMSE/bias are reported instead.

# %% [markdown]
# ## 5. Weekly Units Distribution (Raw vs. `log(1 + x)`)

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))

axes[0].hist(weekly["units"], bins=40, color="#0284c7", edgecolor="white", alpha=0.88)
axes[0].set_xlabel("Weekly demand (units)")
axes[0].set_ylabel("Product-weeks (count)")
axes[0].set_title("Weekly Units Distribution (Raw)")

axes[1].hist(
    np.log1p(weekly["units"]),
    bins=35,
    color="#0d9488",
    edgecolor="white",
    alpha=0.88,
)
axes[1].set_xlabel("Transformed weekly demand (log(1 + units))")
axes[1].set_ylabel("Product-weeks (count)")
axes[1].set_title("Weekly Units Distribution (log(1 + x))")

fig.suptitle("Weekly Product-Week Demand Distribution (n = 5,147 rows)", y=1.02)
fig.tight_layout()
fig.savefig(FIGURES_DIR / "05_weekly_units_distribution.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# %% [markdown]
# ### What to write in the report
# Product-week demand is strongly right-skewed: across 5,147 rows, the median is 184.0 units while the mean is 254.7 units (std 243.8, interquartile range 100–332, maximum 2,755 units). Applying `log(1 + x)` compresses the long upper tail into an approximately unimodal distribution alongside the small zero-demand spike, motivating both tree models and linear/regularized baselines.

# %% [markdown]
# ## 6. Volume Segments (`low`, `medium`, `high` Tertiles)

# %%
# Compute tertiles on training weeks only (excluding closed weeks) to prevent test leakage.
train_open = weekly_open[weekly_open["week_start"] < TEST_START]
avg_weekly_train = train_open.groupby("product_id")["units"].mean()
segments = pd.qcut(avg_weekly_train, q=3, labels=["low", "medium", "high"])

segments_df = (
    pd.DataFrame(
        {
            "product_id": avg_weekly_train.index.to_numpy(),
            "segment": segments.astype(str).to_numpy(),
        }
    )
    .sort_values("product_id", kind="mergesort")
    .reset_index(drop=True)
)
segments_path = PROCESSED_DIR / "product_segments.csv"
segments_df.to_csv(segments_path, index=False)

seg_summary = avg_weekly_train.groupby(segments, observed=False).agg(
    ["count", "min", "mean", "max"]
)
print(f"Saved {segments_path} ({len(segments_df)} rows)")
print(seg_summary.round(1))

fig, ax = plt.subplots(figsize=(8, 4.6))
order = ["low", "medium", "high"]
grouped_vals = [avg_weekly_train[segments == s].values for s in order]
box = ax.boxplot(grouped_vals, tick_labels=["Low", "Medium", "High"], patch_artist=True)
colors = ["#93c5fd", "#3b82f6", "#1e3a8a"]
for patch, color in zip(box["boxes"], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.75)

ax.set_xlabel("Volume segment (training-period tertiles)")
ax.set_ylabel("Average weekly demand (units / week)")
ax.set_title("Product Volume Segments (100 Products: Low = 34, Medium = 32, High = 34)")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "06_volume_segments.png", dpi=150)
plt.close(fig)

# %% [markdown]
# ### What to write in the report
# Products are partitioned into tertiles using training-period average weekly demand (excluding the `2010-12-27` closure) and saved to `data/processed/product_segments.csv`: 34 `low`-volume SKUs (142.8–174.7 units/week, mean 157.3), 32 `medium`-volume SKUs (175.7–249.7 units/week, mean 210.1), and 34 `high`-volume SKUs (256.5–856.5 units/week, mean 385.5). Computing segments strictly on training weeks avoids test-set leakage when pooled segment errors are used as `sigma_error` fallbacks.

# %% [markdown]
# ## 7. Coefficient of Variation per Product

# %%
cv_per_product = weekly_open.groupby("product_id")["units"].agg(
    lambda s: float(s.std(ddof=1) / s.mean())
)
print(
    f"Product CV — mean: {cv_per_product.mean():.3f}, median: {cv_per_product.median():.3f}, "
    f"min: {cv_per_product.min():.3f}, max: {cv_per_product.max():.3f}"
)

fig, ax = plt.subplots(figsize=(8.5, 4.5))
ax.hist(cv_per_product, bins=18, color="#ea580c", edgecolor="white", alpha=0.86)
ax.axvline(
    cv_per_product.mean(),
    color="#111827",
    linestyle="--",
    linewidth=1.5,
    label=f"Mean CV: {cv_per_product.mean():.3f}",
)
ax.set_xlabel("Coefficient of variation (std / mean, dimensionless)")
ax.set_ylabel("Number of products (count)")
ax.set_title("Per-Product Demand Volatility (Coefficient of Variation on Open Weeks)")
ax.legend()
fig.tight_layout()
fig.savefig(FIGURES_DIR / "07_coefficient_of_variation.png", dpi=150)
plt.close(fig)

# %% [markdown]
# ### What to write in the report
# Even among the 100 highest-volume products, week-to-week demand volatility is substantial: the per-product coefficient of variation (`std / mean` on open weeks) averages 0.731 (median 0.668) and spans 0.415 to 1.501. High volatility implies wide forecast error intervals (`sigma_error`) and requires explicit safety-stock buffers in the downstream risk engine.

# %% [markdown]
# ## 8. Country Share of Units (Top 8)

# %%
country_units = cleaned_tx.groupby("Country")["Quantity"].sum().sort_values(ascending=False)
country_share = (country_units / country_units.sum()) * 100.0
top8_countries = pd.DataFrame(
    {"units": country_units.head(8), "share_pct": country_share.head(8)}
)
print("Top 8 countries by cleaned raw units:")
print(top8_countries.round(2).to_string())

top8_plot = top8_countries.iloc[::-1]
fig, ax = plt.subplots(figsize=(9, 4.6))
bars = ax.barh(top8_plot.index, top8_plot["share_pct"], color="#4f46e5", alpha=0.88)
for bar, val in zip(bars, top8_plot["share_pct"]):
    ax.text(
        val + 0.8,
        bar.get_y() + bar.get_height() / 2,
        f"{val:.2f}%",
        va="center",
        fontsize=9,
    )

ax.set_xlim(0, top8_plot["share_pct"].max() * 1.15)
ax.set_xlabel("Share of total cleaned units (%)")
ax.set_ylabel("Country")
ax.set_title("Top 8 Countries by Share of Cleaned Transaction Units")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "08_country_share.png", dpi=150)
plt.close(fig)

# %% [markdown]
# ### What to write in the report
# Demand is heavily concentrated in the United Kingdom, which accounts for 83.45% of all cleaned units (4,654,261 units), followed by the Netherlands (3.59%), EIRE (2.64%), Germany (2.12%), and France (2.00%). The top 8 countries together represent 96.50% of unit volume, reflecting a predominantly UK retail/wholesale customer base.

# %% [markdown]
# ## 9. Line-Level Quantity Distribution, Top 5 Raw Quantities, and Cap Effect

# %%
cleaned_full = cleaned_tx.copy()
cleaned_full["week_start"] = week_start_of(cleaned_full["InvoiceDate"])
full_weeks = full_week_range(cleaned_full["InvoiceDate"])
cleaned_full = cleaned_full[cleaned_full["week_start"].isin(set(full_weeks))].reset_index(
    drop=True
)
train_mask_lines = cleaned_full["week_start"] < TEST_START
_, cap_value, n_capped = cap_quantity(cleaned_full, train_mask_lines, quantile=0.999)

top5_raw_preclean = raw_tx["Quantity"].nlargest(5).tolist()
top5_cleaned = cleaned_tx["Quantity"].nlargest(5).tolist()
print(f"Top 5 raw quantities (pre-clean): {top5_raw_preclean}")
print(f"Top 5 cleaned quantities: {top5_cleaned}")
print(f"Training 99.9th percentile cap: {cap_value:.1f} units ({n_capped} full-week lines capped)")

fig, ax = plt.subplots(figsize=(9, 4.6))
ax.hist(
    np.log10(cleaned_full["Quantity"]),
    bins=40,
    color="#059669",
    edgecolor="white",
    alpha=0.86,
)
ax.axvline(
    np.log10(cap_value),
    color="#dc2626",
    linestyle="--",
    linewidth=1.7,
    label=f"99.9th pct cap = {cap_value:.0f} units ({n_capped} lines capped)",
)
ax.annotate(
    f"Largest raw orders:\n{top5_raw_preclean[0]:,} & {top5_raw_preclean[1]:,} units",
    xy=(np.log10(top5_raw_preclean[0]), 2),
    xytext=(3.2, 65000),
    arrowprops={"arrowstyle": "->", "color": "#991b1b"},
    fontsize=9,
    color="#991b1b",
)
ax.set_xlabel("Line-level order quantity (log10(units))")
ax.set_ylabel("Transaction lines (count)")
ax.set_title("Line-Level Quantity Distribution and Training-Only 99.9th Percentile Cap")
ax.legend()
fig.tight_layout()
fig.savefig(FIGURES_DIR / "09_quantity_cap_effect.png", dpi=150)
plt.close(fig)

# %% [markdown]
# ### What to write in the report
# Single-line quantities span orders of magnitude due to occasional wholesale bulk purchases: the five largest raw lines are 80,995, 74,215, 12,540, 5,568, and 4,800 units (80,995 and 74,215 remain in the cleaned dataset before capping). Learning the 99.9th-percentile cap solely on full-week training rows yields a threshold of 480.0 units and clips 415 extreme lines without distorting typical retail demand or leaking test-period information.

# %% [markdown]
# ## Summary: Key Patterns & Risks for Modelling
#
# - **October–November level shift:** Total weekly demand across the Top 100 products rises from a training mean of 25,000.7 units/week (excluding the closure week) to 29,515.6 units/week in the 8-week test period (+18.1%). Because tree ensembles cannot extrapolate beyond training target ranges, model evaluation must check test-period bias closely and compare linear models (Ridge) alongside tree baselines.
# - **Closure week (`2010-12-27`):** Every product recorded 0 units during the Christmas/New Year shutdown week of `2010-12-27`. Keeping the week in the grid preserves regular Monday indexing, while `closed_weeks.csv` enables feature engineering to mask it as `NaN` so it is never used as a training target or rolled into lag features.
# - **Intermittent demand & volatility:** Although overall open-week zero demand is 2.18% (3.98% including the closure week), 23 of the 100 products experience zero-demand open weeks (up to 25.5%), and per-product CV averages 0.731 (up to 1.501). This combination of intermittency and bulk wholesale spikes makes zero-safe metrics (`MAE`, `RMSE`, `WAPE`, `bias`) mandatory and directly drives safety-stock sizing.

