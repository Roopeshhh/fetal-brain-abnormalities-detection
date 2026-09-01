@echo off
title FetalBrain AI Launcher
echo ===================================================
echo           FetalBrain AI Web Application
echo ===================================================
echo.

cd /d "%~dp0"

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not found. Please install Python 3.9+ and check 'Add to PATH'.
    pause
    exit /b
)

REM Check if virtual environment exists
if not exist "venv" (
    echo [INFO] Creating virtual environment 'venv'...
    python -m venv venv
    call venv\Scripts\activate
    echo [INFO] Installing required dependencies...
    pip install -r requirements.txt
) else (
    call venv\Scripts\activate
)

echo [INFO] Starting Flask Server...
echo [INFO] Open your browser and go to: http://127.0.0.1:5000
echo.
python webapp/app.py
pause
