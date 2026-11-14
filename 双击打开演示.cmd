@echo off
cd /d "%~dp0"
if exist "C:\workspace\EvoResearch\.venv\Scripts\pythonw.exe" (
  start "" "C:\workspace\EvoResearch\.venv\Scripts\pythonw.exe" "%~dp0tools\open_showcase.py"
  exit /b
)
if exist "%~dp0.venv\Scripts\pythonw.exe" (
  start "" "%~dp0.venv\Scripts\pythonw.exe" "%~dp0tools\open_showcase.py"
  exit /b
)
python "%~dp0tools\open_showcase.py"
