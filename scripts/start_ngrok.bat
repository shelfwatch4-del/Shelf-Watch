@echo off
REM Opens the public tunnel to the local server. Start scripts\start_server.bat first.
REM If your ngrok dashboard shows an assigned dev domain, use the command it displays
REM instead (the flag name for the domain differs between ngrok versions).
ngrok http 8000
