#!/bin/bash
cd "$(dirname "$0")"
echo "[start.sh] Checking for updates..."
git pull origin master
echo "[start.sh] Starting dashboard..."
python3 dashboard_v3.py
