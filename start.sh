#!/bin/bash
cd "$(dirname "$0")"
echo "[start.sh] Checking for updates..."
if git pull origin master; then
    echo "[start.sh] Update successful"
else
    echo "[start.sh] Update failed (no internet?), running existing code"
fi
echo "[start.sh] Starting dashboard..."
python3 dashboard_v3.py
