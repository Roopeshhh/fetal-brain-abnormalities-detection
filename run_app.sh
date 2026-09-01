#!/usr/bin/env bash

# Navigate to script directory
cd "$(dirname "$0")"

echo "==================================================="
echo "          FetalBrain AI Web Application"
echo "==================================================="
echo ""

# Check python
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] python3 could not be found. Please install Python 3.9+."
    exit 1
fi

# Setup venv if not present
if [ ! -d "venv" ]; then
    echo "[INFO] Creating virtual environment 'venv'..."
    python3 -m venv venv
    source venv/bin/activate
    echo "[INFO] Installing dependencies..."
    pip install -r requirements.txt
else
    source venv/bin/activate
fi

echo "[INFO] Starting Flask Server..."
echo "[INFO] Open your browser and go to: http://127.0.0.1:5000"
echo ""
python3 webapp/app.py
