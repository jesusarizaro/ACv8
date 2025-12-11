#!/usr/bin/env bash
set -euo pipefail

echo "==============================================="
echo "       Instalador oficial de AudioCinema"
echo "==============================================="

# Detectar usuario real que ejecuta GUI
REAL_USER="$(logname 2>/dev/null || echo $SUDO_USER || echo $USER)"
USER_HOME="/home/$REAL_USER"

# Directorio donde está el instalador
SOURCE_DIR="$(cd "$(dirname "$0")" && pwd)"

# Directorio de instalación final
INSTALL_DIR="/opt/audiocinema"

echo "→ Usuario detectado: $REAL_USER"
echo "→ Origen: $SOURCE_DIR"
echo "→ Instalando en: $INSTALL_DIR"
echo ""

# Borrar instalación previa segura
sudo rm -rf "$INSTALL_DIR"
sudo mkdir -p "$INSTALL_DIR"
sudo chown -R "$REAL_USER":"$REAL_USER" "$INSTALL_DIR"

# Copiar proyecto completo
echo "→ Copiando archivos..."
rsync -av "$SOURCE_DIR/" "$INSTALL_DIR/" --exclude venv >/dev/null

# Crear entorno virtual
echo "→ Creando entorno virtual..."
python3 -m venv "$INSTALL_DIR/venv"
"$INSTALL_DIR/venv/bin/pip" install --upgrade pip wheel >/dev/null

# Instalar dependencias
echo "→ Instalando dependencias..."
"$INSTALL_DIR/venv/bin/pip" install -r "$INSTALL_DIR/requirements.txt" >/dev/null

# Crear run_gui.sh correcto (SIEMPRE con rutas válidas)
echo "→ Configurando lanzador GUI..."
cat > "$INSTALL_DIR/run_gui.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
APP_DIR="\$(cd "\$(dirname "\$0")" && pwd)"
PY="\$APP_DIR/venv/bin/python"

# fallback si el venv falla
if [ ! -f "\$PY" ]; then
    PY="/usr/bin/python3"
fi

exec "\$PY" "\$APP_DIR/src/audiocinema_gui.py"
EOF

chmod +x "$INSTALL_DIR/run_gui.sh"

# Crear ejecutable scheduled
echo "→ Configurando ejecución programada..."
cat > "$INSTALL_DIR/run_scheduled.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
REAL_USER="\$(logname 2>/dev/null || echo \$USER)"
APP_DIR="\$(cd "\$(dirname "\$0")" && pwd)"
PY="\$APP_DIR/venv/bin/python"

exec sudo -u "\$REAL_USER" "\$PY" "\$APP_DIR/src/audiocinema_core.py" --scheduled
EOF

chmod +x "$INSTALL_DIR/run_scheduled.sh"

# Crear servicio systemd usando rutas dinámicas
echo "→ Instalando servicios systemd..."
sudo tee /etc/systemd/system/audiocinema.service >/dev/null <<EOF
[Unit]
Description=AudioCinema automatic measurement
After=network-online.target sound.target

[Service]
Type=oneshot
WorkingDirectory=$INSTALL_DIR
ExecStart=$INSTALL_DIR/run_scheduled.sh
EOF

sudo tee /etc/systemd/system/audiocinema.timer >/dev/null <<EOF
[Unit]
Description=Timer for AudioCinema automatic measurement

[Timer]
OnCalendar=*-*-* *:*:00
Persistent=true
Unit=audiocinema.service

[Install]
WantedBy=timers.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable --now audiocinema.timer

# Crear acceso directo GUI (.desktop)
echo "→ Creando acceso directo en el menú..."
mkdir -p "$USER_HOME/.local/share/applications/"

cat > "$USER_HOME/.local/share/applications/AudioCinema.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=AudioCinema
Comment=Evaluación automática del sistema 5.1
Exec=$INSTALL_DIR/run_gui.sh
Icon=$INSTALL_DIR/assets/audiocinema.png
Terminal=false
Categories=AudioVideo;Utility;
EOF

chown "$REAL_USER":"$REAL_USER" "$USER_HOME/.local/share/applications/AudioCinema.desktop"

echo ""
echo "==============================================="
echo "   ✔ Instalación completa. AudioCinema listo."
echo "==============================================="
echo ""
echo "Puedes abrir la aplicación desde el menú o ejecutando:"
echo "   $INSTALL_DIR/run_gui.sh"
echo ""
