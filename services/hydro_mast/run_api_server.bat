@echo off
setlocal

if not exist ".venv\Scripts\python.exe" (
  echo [ERROR] .venv not found. Create venv first:
  echo   python -m venv .venv
  echo   .venv\Scripts\pip install -r requirements-lock.txt
  exit /b 1
)

echo [RUN] Hydro-MAST API server on 0.0.0.0:8787
.venv\Scripts\python.exe "03_realtime_pipeline\realtime_dashboard_server.py" --host 0.0.0.0 --port 8787 --no-open
