@echo off
chcp 65001 > nul
setlocal

cd /d "%~dp0"
.\.venv\Scripts\python realtime_visual_check.py

echo.
echo 완료. 창을 닫으려면 아무 키나 누르세요.
pause > nul
