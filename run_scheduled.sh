#!/usr/bin/env bash
set -euo pipefail

APP_DIR="$(cd "$(dirname "$0")" && pwd)"
PY="$APP_DIR/venv/bin/python"
REAL_USER="$(logname 2>/dev/null || echo $USER)"

exec sudo -u "$REAL_USER" "$PY" "$APP_DIR/src/audiocinema_core.py" --scheduled
