@echo off
REM Opens the public tunnel to the local server. Start scripts\start_server.bat first.
REM Free ngrok URLs change on restart unless you claim a static domain; if your dashboard
REM shows an assigned domain, run the command it displays instead:
REM   ngrok http --url=YOUR-DOMAIN.ngrok-free.app 8000   (flag name differs between versions)
where ngrok >nul 2>nul
if errorlevel 1 (
  echo ngrok not found on PATH. Install the agent from ngrok.com, then run once:
  echo   ngrok config add-authtoken YOUR_TOKEN
  exit /b 1
)
ngrok http 127.0.0.1:8000
