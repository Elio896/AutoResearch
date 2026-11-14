@echo off
cd /d "%~dp0"
echo Installing dependencies into project .venv
where py >nul 2>nul
if errorlevel 1 (python tools\setup_demo.py) else (py -3 tools\setup_demo.py)
pause
