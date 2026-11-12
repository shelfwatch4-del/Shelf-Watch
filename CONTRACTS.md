# CONTRACTS.md

> The hand-off agreements between members. Changing anything here needs all four members to agree.
> On Day 1, each producer commits a **mock** version of their file to `data/mock/` with these exact columns.

---

## C1. `data/processed/weekly_demand.parquet`
**Producer:** M1 | **Consumers:** M2, M3 (for average demand), M4

| Column | Type | Notes |
|---|---|---|
| `product_id` | str | Original `StockCode` |
| `description` | str | Most frequent description for that product |
| `week_start` | date | Monday of the week |
| `units` | int | Total units sold that week (>= 0) |
| `orders` | int | Number of distinct invoices that week |
| `customers` | int | Number of distinct non-null CustomerIDs that week |
| `avg_price` | float | Mean UnitPrice that week. Forward-filled in zero-demand weeks |

Rules: one row per product per week; only the Top 100 products; only full weeks; zero-demand weeks included from each product's first active week through the last full week (`2011-11-28`), excluding shop-closed weeks in `closed_weeks.csv`; sorted by `product_id, week_start`.


### C1b. `data/processed/closed_weeks.csv` (companion to C1)
**Producer:** M1 | **Consumer:** M2. One column, `week_start` (date). Lists full weeks in which the whole shop made no sales. These are closures, not zero demand. Currently: `2010-12-27`.

---

## C2. `data/processed/predictions.csv`
**Producer:** M2 | **Consumers:** M1 (error analysis), M3 (sigma_error), M4

| Column | Type | Notes |
|---|---|---|
| `product_id` | str | |
| `week_start` | date | Target week being predicted |
| `actual` | float | True units |
| `predicted` | float | Model forecast, clipped at 0 |
| `model_name` | str | `naive`, `moving_avg_4`, `ridge`, `random_forest`, `lightgbm`, ... |
| `split` | str | `test` (final evaluation) or `val` (validation inside training) |

Contains predictions for **all** models so they can be compared.

---

## C3. `data/processed/metrics.csv`
**Producer:** M2 | **Consumers:** M1, M4

| Column | Type | Notes |
|---|---|---|
| `model_name` | str | Same names as C2 |
| `split` | str | `test` or `val` |
| `MAE` | float | |
| `RMSE` | float | |
| `WAPE` | float | sum(abs error) / sum(actual) |
| `bias` | float | mean(predicted - actual) |

---

## C4. `data/processed/next_week_forecast.csv`
**Producer:** M2 | **Consumers:** M3, M4

Forecast for the week **after the last full week** (week_start `2011-12-05`) using the best model.

| Column | Type | Notes |
|---|---|---|
| `product_id` | str | |
| `week_start` | date | `2011-12-05` |
| `forecast_weekly` | float | Predicted units, clipped at 0 |
| `sigma_error` | float | Std of that product's forecast error (best model, val/test) |
| `avg_weekly_units_train` | float | Mean weekly units in training period |

---

## C5. Model files in `models/`
**Producer:** M2 | **Consumers:** M3

- `best_model.joblib`: the trained model (a full sklearn Pipeline if scaling or encoding is used)
- `feature_list.json`: ordered list of feature names exactly as used in training
- `config.json`: `{"model_name": "...", "horizon_weeks": 1, "train_end": "...", "seed": 42, "library_versions": {...}}`

---

## C6. Risk engine function (`src/risk_engine.py`)
**Producer:** M3 | **Consumers:** M3 (API), M4 (tests and sensitivity)

```python
def compute_risk(forecast_df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    forecast_df: contract C4 (product_id, forecast_weekly, sigma_error,
                 avg_weekly_units_train, ...)
    params (all optional, defaults shown):
        lead_time_weeks: int = 1          # allowed 1-4
        service_level: float = 0.95       # allowed 0.80-0.99
        initial_stock_weeks: float = 3.0  # allowed 0.5-8
        seed: int = 42
    returns DataFrame with columns:
        product_id, description, forecast_weekly, forecast_lead_time,
        on_hand, safety_stock, reorder_point, days_of_cover, risk_level
    risk_level in {"HIGH", "MEDIUM", "LOW"}
    """
```

Formulas are defined in PROJECT_CONTEXT.md section 8. Invalid params raise `ValueError`.

```python
def sensitivity_table(forecast_df, lead_times=(1,2,3,4),
                      service_levels=(0.90,0.95,0.99)) -> pd.DataFrame
# columns: lead_time_weeks, service_level, n_high, n_medium, n_low
```

---

## C7. API endpoints (`api/main.py`, FastAPI)
**Producer:** M3 | **Consumer:** M4. Base URL locally: `http://localhost:8000`

### `GET /health`
```json
{"status": "ok"}
```

### `GET /products`
```json
[{"product_id": "85123A", "description": "WHITE HANGING HEART T-LIGHT HOLDER"}]
```

### `GET /forecast/{product_id}`
```json
{
  "product_id": "85123A",
  "description": "...",
  "history": [{"week_start": "2011-10-10", "units": 120}],
  "test_predictions": [{"week_start": "2011-10-10", "actual": 120, "predicted": 105.3}],
  "next_week_forecast": {"week_start": "2011-12-05", "forecast_weekly": 98.4}
}
```
Unknown product returns HTTP 404 with `{"detail": "Product not found"}`.

### `POST /risk`
Request:
```json
{"lead_time_weeks": 1, "service_level": 0.95, "initial_stock_weeks": 3}
```
Response:
```json
{
  "params": {"lead_time_weeks": 1, "service_level": 0.95, "initial_stock_weeks": 3, "seed": 42},
  "summary": {"n_high": 12, "n_medium": 20, "n_low": 68, "avg_days_of_cover": 18.4},
  "items": [{"product_id": "85123A", "description": "...", "forecast_weekly": 98.4,
             "forecast_lead_time": 98.4, "on_hand": 310.0, "safety_stock": 40.2,
             "reorder_point": 138.6, "days_of_cover": 22.1, "risk_level": "LOW"}]
}
```
Out-of-range params return HTTP 422.

### `POST /sensitivity`
Request: `{"lead_times": [1,2,3,4], "service_levels": [0.90,0.95,0.99]}`
Response: list of `{lead_time_weeks, service_level, n_high, n_medium, n_low}`

### `GET /metrics`
```json
[{"model_name": "lightgbm", "split": "test", "MAE": 0.0, "RMSE": 0.0, "WAPE": 0.0, "bias": 0.0}]
```

### `GET /disclaimer`
```json
{"text": "Inventory levels are simulated. ..."}
```

---

## C8. Definition of done for each hand-off
- File or function matches the schema exactly (column names, types, order)
- Passes a schema-check test in `tests/`
- Documented in the README with one example
- Committed on the member's branch and merged to `main` after review

## C9. Change log
| Date | Change | Agreed by |
|---|---|---|
| | Initial version | All |
