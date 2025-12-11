#!/usr/bin/env bash
set -euo pipefail

echo "==============================================="
echo "     Instalador oficial de AudioCinema"
echo "==============================================="

# Detectar usuario real (importante para systemd y permisos)
REAL_USER="$(logname 2>/dev/null || echo $SUDO_USER || echo $USER)"

# Directorio de instalación
INSTALL_DIR="/opt/audiocinema"

echo "→ Usuario detectado: $REAL_USER"
echo "→ Instalando AudioCinema en: $INSTALL_DIR"
echo ""

# Crear carpeta destino
sudo mkdir -p "$INSTALL_DIR"
sudo chown -R "$REAL_USER":"$REAL_USER" "$INSTALL_DIR"

# Copiar archivos del repositorio actual
echo "→ Copiando archivos al directorio destino..."
sudo rsync -av --exclude 'venv' ./ "$INSTALL_DIR/" >/dev/null

# Crear entorno virtual
echo "→ Creando entorno virtual..."
python3 -m venv "$INSTALL_DIR/venv"
"$INSTALL_DIR/venv/bin/pip" install --upgrade pip wheel >/dev/null

# Instalar dependencias
echo "→ Instalando dependencias..."
"$INSTALL_DIR/venv/bin/pip" install -r "$INSTALL_DIR/requirements.txt" >/dev/null

# Crear wrapper GUI ejecutable
echo "→ Configurando run_gui.sh..."
cat > "$INSTALL_DIR/run_gui.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
APP_DIR="$(cd "$(dirname "\$0")" && pwd)"
exec "\$APP_DIR/venv/bin/python" "\$APP_DIR/src/audiocinema_gui.py"
EOF

sudo chmod +x "$INSTALL_DIR/run_gui.sh"

# Crear wrapper para el servicio programado
echo "→ Creando ejecutable automático run_scheduled.sh..."
cat > "$INSTALL_DIR/run_scheduled.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
REAL_USER="\$(logname 2>/dev/null || echo \$USER)"
APP_DIR="\$(cd "\$(dirname "\$0")" && pwd)"

PY="\$APP_DIR/venv/bin/python"
exec sudo -u "\$REAL_USER" "\$PY" "\$APP_DIR/src/audiocinema_core.py" --scheduled
EOF

sudo chmod +x "$INSTALL_DIR/run_scheduled.sh"

# Crear systemd service
echo "→ Generando audiocinema.service..."
sudo tee /etc/systemd/system/audiocinema.service >/dev/null <<EOF
[Unit]
Description=AudioCinema automatic measurement
After=network-online.target sound.target

[Service]
Type=oneshot
WorkingDirectory=$INSTALL_DIR
ExecStart=$INSTALL_DIR/run_scheduled.sh
EOF

# Crear systemd timer
echo "→ Generando audiocinema.timer..."
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

# Recargar systemd y habilitar timer
echo "→ Activando servicios systemd..."
sudo systemctl daemon-reload
sudo systemctl enable --now audiocinema.timer

# Crear acceso directo GUI en el menú
DESKTOP_FILE="/home/$REAL_USER/.local/share/applications/AudioCinema.desktop"

echo "→ Creando acceso directo en el menú..."
mkdir -p "$(dirname "$DESKTOP_FILE")"

cat > "$DESKTOP_FILE" <<EOF
[Desktop Entry]
Type=Application
Name=AudioCinema
Comment=Evaluación automática de sistema 5.1
Exec=$INSTALL_DIR/run_gui.sh
Icon=$INSTALL_DIR/assets/audiocinema.png
Terminal=false
Categories=AudioVideo;Utility;
StartupNotify=true
EOF

sudo chown "$REAL_USER":"$REAL_USER" "$DESKTOP_FILE"

echo ""
echo "==============================================="
echo "      ✔ Instalación de AudioCinema completa"
echo "==============================================="
echo ""
echo " Puedes abrir la aplicación desde:"
echo "   Menú → Sonido y Video → AudioCinema"
echo ""
echo " El sistema ya está programado y enviará datos a ThingsBoard."
echo ""
