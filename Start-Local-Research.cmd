@echo off
cd /d "%~dp0"
echo EvoResearch launcher - expected backend: ml-chain-23-qwen-demo1
echo Project: %CD%
echo Website: http://127.0.0.1:8018/
set "EVORESEARCH_WEB_PORT=8018"
set "MLPY=%~dp0.venv\Scripts\python.exe"
if not exist "%MLPY%" set "MLPY=python"
if not exist "tools\open_showcase.py" (
  echo ERROR: Current project launcher was not found.
  pause
  exit /b 1
)
"%MLPY%" tools\open_showcase.py
pause
