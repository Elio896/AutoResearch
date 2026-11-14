@echo off
cd /d "%~dp0"
set "MLPY=C:\workspace\EvoResearch\.venv\Scripts\python.exe"
if not exist "%MLPY%" set "MLPY=python"
"%MLPY%" tools\start_connected_ml.py
pause
