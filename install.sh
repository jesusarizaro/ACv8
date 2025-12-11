#!/usr/bin/env bash
set -euo pipefail

echo "======================================"
echo " Instalando AudioCinema en:"
echo "   $(pwd)"
echo "======================================"

APP_DIR="$(pwd)"
REAL_USER="$(logname 2>/dev/null || echo $SUDO_USER || echo $USER)"

# Crear venv local
python3 -m venv "$APP_DIR/venv"
"$APP_DIR/venv/bin/pip" install --upgrade pip wheel
"$APP_DIR/venv/bin/pip" install -r "$APP_DIR/requirements.txt"

# Asegurar permisos de ejecución
chmod +x "$APP_DIR/run_gui.sh"
chmod +x "$APP_DIR/run_scheduled.sh"

# Crear acceso directo .desktop
DESKTOP_FILE="/home/$REAL_USER/.local/share/applications/AudioCinema.desktop"
mkdir -p "$(dirname "$DESKTOP_FILE")"

cat > "$DESKTOP_FILE" <<EOF
[Desktop Entry]
Type=Application
Name=AudioCinema
Exec=$APP_DIR/run_gui.sh
Icon=$APP_DIR/assets/audiocinema.png
Terminal=false
Categories=AudioVideo;Utility;
EOF

chmod +x "$DESKTOP_FILE"
echo "✔ Instalación completada."
