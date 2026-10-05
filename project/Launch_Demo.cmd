@echo off
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" app.py
) else if exist "..\.venv\Scripts\python.exe" (
  "..\.venv\Scripts\python.exe" app.py
) else (
  echo First follow the Python 3.12 setup steps in README.md.
  pause
)
