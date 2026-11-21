@echo off
REM Starts the Shelfwatch API on http://127.0.0.1:8000 (local only; ngrok forwards to it).
REM 1) Make sure MySQL is running (Services -> MySQL80 -> Running).
REM 2) .env is loaded by the app itself (api/db.py: load_dotenv) from the project root.
cd /d "%~dp0.."
if not exist venv\Scripts\activate.bat (
  echo Virtual environment not found at venv\. Create it: python -m venv venv ^& venv\Scripts\pip install -r requirements.txt
  exit /b 1
)
if not exist .env (
  echo .env not found. Copy .env.example to .env and set DATABASE_URL and API_KEY.
  exit /b 1
)
call venv\Scripts\activate.bat
REM api/db.py prefers TEST_DATABASE_URL over DATABASE_URL; blank it so the server always uses MySQL.
set TEST_DATABASE_URL=
echo Starting FastAPI on http://127.0.0.1:8000 ...
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
