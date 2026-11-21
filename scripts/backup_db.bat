@echo off
REM Dumps the shelfwatch database to backups\shelfwatch_YYYYMMDD_HHMMSS.sql.
REM Credentials come from DATABASE_URL in .env; the password is never printed.
setlocal
cd /d "%~dp0.."
if not exist .env (
  echo .env not found.
  exit /b 1
)
if not exist backups mkdir backups
for /f %%t in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set STAMP=%%t
set OUT=backups\shelfwatch_%STAMP%.sql
REM The helper writes a temporary option file (0 secrets on the command line) and runs mysqldump.
venv\Scripts\python.exe scripts\backup_db.py "%OUT%"
if errorlevel 1 (
  echo Backup FAILED.
  if exist "%OUT%" del "%OUT%"
  exit /b 1
)
echo Backup written to %OUT%
echo Copy it to the second laptop and restore with: mysql -u USER -p shelfwatch ^< %OUT%
