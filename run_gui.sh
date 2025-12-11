#!/usr/bin/env bash
set -euo pipefail
APP_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV="${APP_DIR}/venv"

if [ ! -d "$VENV" ]; then
  python3 -m venv "$VENV"
  "${VENV}/bin/pip" install --upgrade pip wheel
  [ -f "${APP_DIR}/requirements.txt" ] && "${VENV}/bin/pip" install -r "${APP_DIR}/requirements.txt"
fi

exec "${VENV}/bin/python" "${APP_DIR}/audiocinema_gui.py"
