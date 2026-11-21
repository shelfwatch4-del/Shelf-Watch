@echo off
REM Starts the Shelfwatch API and website on http://localhost:8000
REM 1) Make sure MySQL is running (Services -> MySQL80 -> Running).
cd /d %~dp0..
if not exist venv\Scripts\python.exe (
  echo Virtual environment not found. Run scripts\setup.ps1 first.
  exit /b 1
)
echo Starting FastAPI on port 8000 ...
venv\Scripts\python.exe -m uvicorn api.main:app --host 0.0.0.0 --port 8000
