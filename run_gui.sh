#!/usr/bin/env bash
set -euo pipefail

# Detectar carpeta real del programa
APP_DIR="$(cd "$(dirname "$0")" && pwd)"

# Python del venv
PY="$APP_DIR/venv/bin/python"

# fallback si no existe el venv
if [ ! -f "$PY" ]; then
    PY="/usr/bin/python3"
fi

exec "$PY" "$APP_DIR/src/audiocinema_gui.py"
