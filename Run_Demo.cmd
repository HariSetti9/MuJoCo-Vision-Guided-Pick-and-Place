@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo First create the environment and install packages using the README steps.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" project\app.py
