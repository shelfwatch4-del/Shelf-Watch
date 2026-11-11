# Shelfwatch: Retail Product Demand Estimation and Stockout Risk Analysis

## Tech Stack
- **Data & ML:** Python 3.10/3.11, pandas, pyarrow, scikit-learn, LightGBM, scipy
- **Backend API:** FastAPI, pydantic, uvicorn
- **Testing:** pytest
- **Frontend:** Plain HTML/CSS/JavaScript dashboard (`dashboard/`) with no framework and no build step

## Sources of Truth
- [docs/SRS.md](docs/SRS.md) — software requirements specification
- [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md) — frozen design decisions and project scope
- [CONTRACTS.md](CONTRACTS.md) — exact file schemas and API endpoint contracts

## Core Rule
If code and SRS disagree, stop and ask; never silently change a contract.

