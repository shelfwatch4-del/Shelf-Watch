# Shelfwatch: Retail Product Demand Estimation and Stockout Risk Analysis

Weekly product-level demand forecasting and configurable simulated stockout-risk analysis built on the UCI Online Retail dataset.

## Sources of Truth

- [`docs/SRS.md`](docs/SRS.md) — software requirements specification
- [`PROJECT_CONTEXT.md`](PROJECT_CONTEXT.md) — frozen design decisions and project scope
- [`CONTRACTS.md`](CONTRACTS.md) — exact file schemas (C1–C5), risk engine interface (C6), and API contracts (C7)

## Environment Setup

Recommended Python versions: **Python 3.10** or **Python 3.11**.

### Windows (PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Mock Data Generation

Generate deterministic (seed `42`) mock parquet, CSV, and dashboard JSON files in `data/mock/`:

```bash
python scripts/make_mocks.py
```

Example outputs produced in `data/mock/`:
- `weekly_demand.parquet` (C1): 10 products × 20 Monday weeks (`2011-07-18` to `2011-11-28`), columns `product_id, description, week_start, units, orders, customers, avg_price`
- `closed_weeks.csv` (C1b): shop-closed week `2010-12-27`
- `predictions.csv` (C2): `val` and `test` forecasts for `naive`, `moving_avg_4`, `ridge`, `random_forest`, `lightgbm`
- `metrics.csv` (C3): `MAE`, `RMSE`, `WAPE`, `bias` per `(model_name, split)`
- `next_week_forecast.csv` (C4): next-week forecast (`2011-12-05`) with `sigma_error` and `avg_weekly_units_train`
- Dashboard mock JSON files (C7 / SRS §7): `products.json`, `forecast_<product_id>.json`, `risk.json`, `sensitivity.json`, `metrics.json`, `error_analysis.json`, `disclaimer.json`

## Data Pipeline & Analysis

Build `data/processed/weekly_demand.parquet` (C1), `closed_weeks.csv` (C1b), and `cleaning_log.csv` from the raw UCI file:

```bash
python -m src.data_prep --raw "data/raw/Online Retail.xlsx"
```

Run exploratory data analysis to generate the 9 report figures in `reports/figures/` and `data/processed/product_segments.csv`:

```bash
python notebooks/01_eda.py
```

Build test-period forecast-error aggregates (`data/processed/error_analysis.csv`) once `predictions.csv` is available:

```python
import pandas as pd
from src.error_analysis import build_error_analysis

preds = pd.read_csv("data/processed/predictions.csv", dtype={"product_id": str})
segs = pd.read_csv("data/processed/product_segments.csv", dtype={"product_id": str})
build_error_analysis(preds, segs)
```

## API

Start the server (`uvicorn api.main:app`), then browse interactive docs at <http://127.0.0.1:8000/docs>
(endpoints are grouped by tag: Meta, Products, Predictions, Risk, Inventory, Orders, Intelligence;
every request body has an example). Write endpoints (`PUT/DELETE /api/inventory/*`,
`POST/PATCH/DELETE /api/orders*`) need the `X-API-Key` header matching the server's `API_KEY`.
Errors are always `{"detail": "..."}`.

```bash
export KEY=YOUR_API_KEY BASE=http://127.0.0.1:8000

# Meta / Products
curl $BASE/api/health
curl $BASE/api/products
curl $BASE/api/forecast/85123A

# Predictions
curl -X POST $BASE/api/predict -H "Content-Type: application/json"   -d '{"product_id": "85123A", "lead_time_weeks": 2, "service_level": 0.95}'
curl -X POST $BASE/api/predict/custom -H "Content-Type: application/json"   -d '{"recent_units": [10, 12, 9, 11, 13, 10, 12, 11]}'

# Risk
curl -X POST $BASE/api/risk -H "Content-Type: application/json"   -d '{"lead_time_weeks": 2, "service_level": 0.95, "initial_stock_weeks": 3}'
curl -X POST $BASE/api/simulate -H "Content-Type: application/json"   -d '{"lead_time_weeks": 2, "delay_weeks": 1}'

# Inventory
curl $BASE/api/inventory
curl -X PUT $BASE/api/inventory/85123A -H "X-API-Key: $KEY" -H "Content-Type: application/json"   -d '{"on_hand": 75, "lead_time_weeks": 2, "service_level": 0.95}'
curl -X DELETE $BASE/api/inventory/85123A -H "X-API-Key: $KEY"

# Orders
curl -X POST $BASE/api/orders -H "X-API-Key: $KEY" -H "Content-Type: application/json"   -d '{"product_id": "85123A", "quantity": 120}'
curl -X PATCH $BASE/api/orders/1 -H "X-API-Key: $KEY" -H "Content-Type: application/json"   -d '{"status": "Approved"}'
curl "$BASE/api/orders?status=Approved"
curl -X DELETE $BASE/api/orders/1 -H "X-API-Key: $KEY"   # only Pending or Cancelled
```

Python client and demo (health, predict, create order, approve, change stock, cancel, delete):

```bash
SHELFWATCH_URL=$BASE SHELFWATCH_API_KEY=$KEY python scripts/api_client.py demo
python scripts/bench_api.py        # latency: median and p95 over 50 calls per endpoint
```

See [docs/API_CLIENT_EXPLAINED.md](docs/API_CLIENT_EXPLAINED.md) for how the client works.

## Running Tests

Run the contract schema tests (checks `data/mock/` and `data/processed/` when present):

```bash
pytest tests/test_contracts.py -q
```

Run the full test suite:

```bash
pytest -q
```

## Repository Layout

```text
ShelfWatch/
├── PROJECT_CONTEXT.md
├── CONTRACTS.md
├── GEMINI.md
├── README.md
├── requirements.txt
├── api/                  # FastAPI service (main.py, schemas.py)
├── dashboard/            # Plain HTML/CSS/JS dashboard
│   ├── css/
│   ├── js/
│   └── prototype/
├── data/
│   ├── raw/              # Raw UCI Online Retail.xlsx (gitignored)
│   ├── processed/        # Pipeline outputs (C1–C4)
│   └── mock/             # Deterministic Day-1 mock files (C1–C4, C7)
├── docs/                 # SRS and workflow prompts
├── models/               # Serialized model, feature_list.json, config.json (C5)
├── notebooks/            # EDA, modeling, error analysis, sensitivity
├── scripts/
│   └── make_mocks.py     # Deterministic mock data generator
├── src/                  # Core Python modules (data_prep, features, train, risk_engine)
└── tests/                # Pytest unit, leakage, API, and contract tests
    └── test_contracts.py
```

