#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GUI de AudioCinema

- Misma apariencia que la versión anterior.
- Popup Configuración con pestañas:
  1) Cronograma
  2) Grabación
  3) Pista de referencia
  4) Criterios de evaluación
  5) Envío de resultados
"""

from __future__ import annotations
import traceback
from datetime import datetime
from typing import Optional, List

import tkinter as tk
from tkinter import ttk, messagebox
import ttkbootstrap as tb
from ttkbootstrap.constants import *

import numpy as np
import sounddevice as sd
import soundfile as sf
import matplotlib

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from audiocinema_core import (
    APP_DIR,
    ASSETS_DIR,
    load_config,
    save_config,
    run_measurement,
    compute_next_run_from_cfg,
)

APP_NAME = "AudioCinema"


# ---------------------------
#   UTIL MIC
# ---------------------------


def list_input_devices() -> List[str]:
    """Devuelve lista de dispositivos de entrada en formato 'idx - nombre'."""
    try:
        devices = sd.query_devices()
    except Exception:
        return []
    out = []
    for idx, d in enumerate(devices):
        if d.get("max_input_channels", 0) > 0:
            out.append(f"{idx} - {d['name']}")
    return out


def pick_device_index_from_label(label: str) -> Optional[int]:
    """Extrae el índice numérico a partir de 'idx - nombre'."""
    try:
        idx_str = label.split(" - ", 1)[0]
        return int(idx_str)
    except Exception:
        return None


# Decorador para capturar errores y mostrarlos como popup


def ui_action(fn):
    def wrapper(self, *args, **kwargs):
        try:
            return fn(self, *args, **kwargs)
        except Exception:
            tb_str = traceback.format_exc()
            print(tb_str)
            messagebox.showerror(APP_NAME, tb_str)
            return None

    return wrapper


# ---------------------------
#   CLASE GUI
# ---------------------------


class AudioCinemaGUI:
    def __init__(self, root: tb.Window):
        self.root = root
        self.root.title(APP_NAME)

        tb.Style(theme="flatly")
        try:
            self.root.configure(bg="#e6e6e6")
        except Exception:
            pass

        # icono
        self._icon_img = None
        try:
            icon_path = ASSETS_DIR / "audiocinema.png"
            if icon_path.exists():
                self._icon_img = tk.PhotoImage(file=str(icon_path))
                self.root.iconphoto(True, self._icon_img)
        except Exception:
            pass

        self.cfg = load_config()

        # variables de cabecera
        self.test_name = tk.StringVar(value="—")
        self.eval_text = tk.StringVar(value="—")
        self.next_eval_text = tk.StringVar(value="—")

        # último audio
        self.last_ref = None
        self.last_cur = None
        self.last_fs = int(self.cfg["audio"]["fs"])

        # mic seleccionado
        self.input_device_label = tk.StringVar(value="(auto)")

        self._build_ui()
        self._update_next_eval_label()

    # ------------- BUILD UI -------------

    def _build_ui(self):
        root_frame = ttk.Frame(self.root, padding=8)
        root_frame.pack(fill=BOTH, expand=True)

        paned = ttk.Panedwindow(root_frame, orient=HORIZONTAL)
        paned.pack(fill=BOTH, expand=True)

        # ---------- PANEL IZQUIERDO ----------
        left = ttk.Frame(paned, padding=6)
        paned.add(left, weight=1)

        card = ttk.Frame(left, padding=6)
        card.pack(fill=Y)

        if self._icon_img is not None:
            ttk.Label(card, image=self._icon_img).pack(pady=(0, 4))

        ttk.Label(card, text="AudioCinema", font=("Segoe UI", 18, "bold")).pack()

        desc = (
            "Graba, evalúa y analiza tu sistema de audio "
            "para garantizar la mejor experiencia envolvente."
        )
        ttk.Label(card, text=desc, wraplength=220, justify="center").pack(pady=10)

        btn_style = {"bootstyle": PRIMARY, "width": 20}
        tb.Button(card, text="Información", command=self._show_info, **btn_style).pack(
            pady=5
        )
        tb.Button(
            card, text="Configuración", command=self._popup_config, **btn_style
        ).pack(pady=5)
        tb.Button(card, text="Confirmación", command=self._popup_confirm, **btn_style).pack(
            pady=5
        )
        tb.Button(card, text="Prueba ahora", command=self._run_once, **btn_style).pack(
            pady=5
        )

        # ---------- PANEL DERECHO ----------
        paned.add(ttk.Separator(root_frame, orient=VERTICAL))

        right = ttk.Frame(paned, padding=8)
        paned.add(right, weight=4)

        # cabecera
        header = ttk.Frame(right)
        header.pack(fill=X, pady=6)

        ttk.Label(
            header, text="PRUEBA:", font=("Segoe UI", 10, "bold")
        ).grid(row=0, column=0, sticky="w")
        ttk.Entry(
            header,
            textvariable=self.test_name,
            width=32,
            state="readonly",
            justify="center",
        ).grid(row=0, column=1, sticky="w")

        ttk.Label(
            header, text="RESULTADO:", font=("Segoe UI", 10, "bold")
        ).grid(row=1, column=0, sticky="w", pady=4)
        self.eval_lbl = ttk.Label(
            header, textvariable=self.eval_text, font=("Segoe UI", 11, "bold")
        )
        self.eval_lbl.grid(row=1, column=1, sticky="w")

        ttk.Label(
            header, text="PRÓXIMA EVALUACIÓN:", font=("Segoe UI", 10, "bold")
        ).grid(row=2, column=0, sticky="w")
        ttk.Label(header, textvariable=self.next_eval_text, font=("Segoe UI", 10)).grid(
            row=2, column=1, sticky="w", pady=4
        )

        # figura
        fig_card = ttk.Frame(right, padding=4)
        fig_card.pack(fill=BOTH, expand=True)

        self.fig = Figure(figsize=(5, 4), dpi=100)
        self.ax_ref = self.fig.add_subplot(2, 1, 1)
        self.ax_cur = self.fig.add_subplot(2, 1, 2)

        self.canvas = FigureCanvasTkAgg(self.fig, master=fig_card)
        self.canvas.get_tk_widget().pack(fill=BOTH, expand=True)
        self._clear_waves()
        self.fig.tight_layout()

        # mensajes
        msg_card = ttk.Frame(right, padding=4)
        msg_card.pack(fill=X)
        ttk.Label(msg_card, text="Mensajes", font=("Segoe UI", 10, "bold")).pack(
            anchor="w"
        )

        self.msg_text = tk.Text(msg_card, height=6, wrap="word")
        self.msg_text.pack(fill=BOTH)

        self._set_messages(["Listo. Presiona «Prueba ahora» para iniciar."])

    # ------------- AUX UI -------------

    def _clear_waves(self):
        for ax, title in (
            (self.ax_ref, "Pista de referencia"),
            (self.ax_cur, "Pista de prueba"),
        ):
            ax.clear()
            ax.set_title(title)
            ax.set_xlabel("Tiempo (s)")
            ax.set_ylabel("Amplitud")
            ax.grid(True, axis="x", linestyle=":")
        self.canvas.draw_idle()

    def _plot_wave(self, ax, x: np.ndarray, fs: int):
        t = np.arange(len(x)) / fs
        ax.plot(t, x, linewidth=0.8)
        if len(t):
            ax.set_xlim(0, t[-1])
        else:
            ax.set_xlim(0, 1.0)

    def _set_eval(self, overall: Optional[str]):
        if not overall:
            self.eval_text.set("—")
            self.eval_lbl.configure(foreground="#333333")
        else:
            self.eval_text.set(overall)
            if overall == "PASSED":
                self.eval_lbl.configure(foreground="#0d8a00")
            else:
                self.eval_lbl.configure(foreground="#cc0000")

    def _set_messages(self, lines):
        self.msg_text.delete("1.0", tk.END)
        for ln in lines:
            self.msg_text.insert(tk.END, f"• {ln}\n")
        self.msg_text.see(tk.END)

    def _update_next_eval_label(self):
        try:
            nxt = compute_next_run_from_cfg(self.cfg)
        except Exception:
            nxt = None
        if nxt is None:
            self.next_eval_text.set("— (programación desactivada)")
        else:
            self.next_eval_text.set(nxt.strftime("%a %Y-%m-%d %H:%M"))

    # ------------- ACCIONES -------------

    @ui_action
    def _show_info(self):
        messagebox.showinfo(
            APP_NAME,
            "AudioCinema\n\n"
            "Detecta fallas por canal en sistemas 5.1.\n"
            "Un solo micrófono escucha 6 canales contenidos en una "
            "pista con 7 banderas de frecuencia.\n"
            "Después de cada bandera (excepto la última) hay un barrido "
            "que se analiza para determinar si el canal está VIVO o MUERTO.",
        )

    @ui_action
    def _popup_confirm(self):
        cfg = self.cfg
        txt = (
            f"Archivo de referencia:\n  {cfg['reference']['path']}\n\n"
            f"Audio:\n  fs = {cfg['audio']['fs']} Hz\n"
            f"  duración = {cfg['audio']['duration_s']} s\n\n"
            f"Criterios:\n  nivel = {cfg['evaluation']['level']}\n\n"
            f"ThingsBoard:\n  host = {cfg['thingsboard']['host']}\n"
            f"  port = {cfg['thingsboard']['port']}\n"
            f"  token = {cfg['thingsboard']['token'][:6]}…\n\n"
            f"Programación:\n  habilitada = {cfg['schedule']['enabled']}\n"
            f"  modo = {cfg['schedule']['mode']}\n"
            f"  hora = {cfg['schedule']['hour']:02d}:{cfg['schedule']['minute']:02d}\n"
        )
        messagebox.showinfo("Confirmación", txt)

    @ui_action
    def _popup_config(self):
        w = tk.Toplevel(self.root)
        w.title("Configuración")
        if self._icon_img is not None:
            w.iconphoto(True, self._icon_img)

        frm = ttk.Frame(w, padding=10)
        frm.pack(fill=BOTH, expand=True)

        nb = ttk.Notebook(frm)
        nb.pack(fill=BOTH, expand=True)

        # ---------- 1. CRONOGRAMA ----------
        cron = ttk.Frame(nb)
        nb.add(cron, text="Cronograma")

        sch = self.cfg["schedule"]
        enabled_var = tk.BooleanVar(value=sch.get("enabled", False))
        mode_var = tk.StringVar(value=sch.get("mode", "daily"))
        weekday_var = tk.IntVar(value=int(sch.get("weekday", 0)))
        hour_var = tk.IntVar(value=int(sch.get("hour", 22)))
        minute_var = tk.IntVar(value=int(sch.get("minute", 0)))

        ttk.Checkbutton(
            cron,
            text="Activar programación automática",
            variable=enabled_var,
        ).grid(row=0, column=0, columnspan=2, sticky="w", pady=4)

        ttk.Label(cron, text="Modo:").grid(row=1, column=0, sticky="w", pady=4)
        cb_mode = ttk.Combobox(
            cron,
            textvariable=mode_var,
            state="readonly",
            values=["daily", "weekly"],
            width=10,
        )
        cb_mode.grid(row=1, column=1, sticky="w")

        ttk.Label(cron, text="Día (0=Lunes,…,6=Domingo):").grid(
            row=2, column=0, sticky="w", pady=4
        )
        sp_day = ttk.Spinbox(cron, from_=0, to=6, textvariable=weekday_var, width=5)
        sp_day.grid(row=2, column=1, sticky="w")

        ttk.Label(cron, text="Hora (24h):").grid(row=3, column=0, sticky="w", pady=4)
        sp_h = ttk.Spinbox(cron, from_=0, to=23, textvariable=hour_var, width=5)
        sp_h.grid(row=3, column=1, sticky="w")

        ttk.Label(cron, text="Minutos:").grid(row=4, column=0, sticky="w", pady=4)
        sp_m = ttk.Spinbox(cron, from_=0, to=59, textvariable=minute_var, width=5)
        sp_m.grid(row=4, column=1, sticky="w")

        # ---------- 2. GRABACIÓN ----------
        grab = ttk.Frame(nb)
        nb.add(grab, text="Grabación")

        fs_var = tk.IntVar(value=int(self.cfg["audio"]["fs"]))
        dur_var = tk.DoubleVar(value=float(self.cfg["audio"]["duration_s"]))
        dev_list = ["(auto)"] + list_input_devices()
        dev_var = tk.StringVar(
            value=self.input_device_label.get()
            if self.input_device_label.get() in dev_list
            else dev_list[0]
        )

        ttk.Label(grab, text="Sample rate (Hz):").grid(
            row=0, column=0, sticky="w", pady=4
        )
        ttk.Entry(grab, textvariable=fs_var, width=10).grid(row=0, column=1, sticky="w")

        ttk.Label(grab, text="Duración de las pistas (s):").grid(
            row=1, column=0, sticky="w", pady=4
        )
        ttk.Entry(grab, textvariable=dur_var, width=10).grid(
            row=1, column=1, sticky="w"
        )

        ttk.Label(
            grab, text="Micrófono / dispositivo de entrada:"
        ).grid(row=2, column=0, sticky="w", pady=4)
        cb_dev = ttk.Combobox(
            grab,
            textvariable=dev_var,
            values=dev_list,
            state="readonly",
            width=38,
        )
        cb_dev.grid(row=2, column=1, sticky="w")

        # ---------- 3. PISTA DE REFERENCIA ----------
        pref = ttk.Frame(nb)
        nb.add(pref, text="Pista de referencia")

        ref_path_var = tk.StringVar(value=self.cfg["reference"]["path"])

        ttk.Label(pref, text="Ruta del archivo de referencia (.wav):").grid(
            row=0, column=0, sticky="w", pady=4
        )
        ttk.Entry(pref, textvariable=ref_path_var, width=60).grid(
            row=0, column=1, sticky="w"
        )

        def record_ref_now():
            """Graba referencia usando los parámetros de la pestaña Grabación."""
            fs_now = int(fs_var.get())
            dur_now = float(dur_var.get())
            dev_idx = pick_device_index_from_label(dev_var.get())

            x = sd.rec(
                int(dur_now * fs_now),
                samplerate=fs_now,
                channels=1,
                dtype="float32",
                device=dev_idx,
            )
            sd.wait()

            x = x.squeeze().astype("float32")
            m = float(np.max(np.abs(x))) if x.size else 0.0
            if m > 0:
                x = x / m

            out = ASSETS_DIR / "reference_master.wav"
            sf.write(str(out), x, fs_now)

            ref_path_var.set(str(out))
            self.cfg["reference"]["path"] = str(out)
            save_config(self.cfg)

            messagebox.showinfo(
                "Pista de referencia", f"Referencia guardada:\n{out}"
            )

        ttk.Button(
            pref,
            text="Grabar referencia ahora",
            command=record_ref_now,
        ).grid(row=1, column=0, sticky="w", pady=6)

        ttk.Label(
            pref,
            text=(
                "Se usa el mismo micrófono, fs y duración\n"
                "configurados en la pestaña «Grabación»."
            ),
        ).grid(row=1, column=1, sticky="w")

        # ---------- 4. CRITERIOS DE EVALUACIÓN ----------
        crit = ttk.Frame(nb)
        nb.add(crit, text="Criterios de evaluación")

        level_var = tk.StringVar(value=self.cfg["evaluation"]["level"])

        ttk.Label(crit, text="Nivel de exigencia:").grid(
            row=0, column=0, sticky="w", pady=4
        )
        ttk.Combobox(
            crit,
            textvariable=level_var,
            state="readonly",
            values=["Bajo", "Medio", "Alto"],
            width=10,
        ).grid(row=0, column=1, sticky="w")

        ttk.Label(crit, text="Descripción:").grid(
            row=1, column=0, sticky="nw", pady=4
        )
        txt_desc = (
            "• Bajo  → permite más diferencias entre referencia y prueba.\n"
            "• Medio → recomendado (tolerancias moderadas).\n"
            "• Alto  → casi sin margen de error; cualquier variación falla."
        )
        ttk.Label(
            crit, text=txt_desc, wraplength=380, justify="left"
        ).grid(row=1, column=1, sticky="w")

        # ---------- 5. ENVÍO DE RESULTADOS ----------
        envio = ttk.Frame(nb)
        nb.add(envio, text="Envío de resultados")

        tb_cfg = self.cfg["thingsboard"]
        host_var = tk.StringVar(value=tb_cfg.get("host", "thingsboard.cloud"))
        port_var = tk.IntVar(value=int(tb_cfg.get("port", 1883)))
        tls_var = tk.BooleanVar(value=bool(tb_cfg.get("use_tls", False)))
        token_var = tk.StringVar(value=tb_cfg.get("token", ""))

        ttk.Label(envio, text="Host:").grid(row=0, column=0, sticky="w", pady=4)
        ttk.Entry(envio, textvariable=host_var, width=30).grid(
            row=0, column=1, sticky="w"
        )

        ttk.Label(envio, text="Port:").grid(row=1, column=0, sticky="w", pady=4)
        ttk.Entry(envio, textvariable=port_var, width=10).grid(
            row=1, column=1, sticky="w"
        )

        ttk.Checkbutton(
            envio, text="Usar TLS (puerto 8883)", variable=tls_var
        ).grid(row=2, column=0, columnspan=2, sticky="w", pady=4)

        ttk.Label(envio, text="Token de dispositivo:").grid(
            row=3, column=0, sticky="w", pady=4
        )
        ttk.Entry(envio, textvariable=token_var, width=40).grid(
            row=3, column=1, sticky="w"
        )

        # ---------- BOTONES GUARDAR/CANCELAR ----------
        btns = ttk.Frame(frm)
        btns.pack(fill=X, pady=8)

        def on_save():
            # schedule
            self.cfg["schedule"]["enabled"] = bool(enabled_var.get())
            self.cfg["schedule"]["mode"] = mode_var.get()
            self.cfg["schedule"]["weekday"] = int(weekday_var.get())
            self.cfg["schedule"]["hour"] = int(hour_var.get())
            self.cfg["schedule"]["minute"] = int(minute_var.get())

            # audio
            self.cfg["audio"]["fs"] = int(fs_var.get())
            self.cfg["audio"]["duration_s"] = float(dur_var.get())

            # mic
            self.input_device_label.set(dev_var.get())

            # reference
            self.cfg["reference"]["path"] = ref_path_var.get().strip()

            # evaluation
            self.cfg["evaluation"]["level"] = level_var.get()

            # thingsboard
            self.cfg["thingsboard"]["host"] = host_var.get().strip()
            self.cfg["thingsboard"]["port"] = int(port_var.get())
            self.cfg["thingsboard"]["use_tls"] = bool(tls_var.get())
            self.cfg["thingsboard"]["token"] = token_var.get().strip()

            save_config(self.cfg)
            self._update_next_eval_label()
            messagebox.showinfo(APP_NAME, "Configuración guardada.")
            w.destroy()

        tb.Button(btns, text="Guardar", bootstyle=PRIMARY, command=on_save).pack(
            side=RIGHT, padx=4
        )
        tb.Button(
            btns, text="Cancelar", bootstyle=SECONDARY, command=w.destroy
        ).pack(side=RIGHT, padx=4)

        w.transient(self.root)
        w.grab_set()

    # ------------- EJECUTAR PRUEBA -------------

    @ui_action
    def _run_once(self):
        dev_idx = pick_device_index_from_label(self.input_device_label.get())
        # Permite que run_measurement devuelva 6 o 7 valores
        res, payload, out_path, x_ref, x_cur, fs, *rest = run_measurement(
            device_index=dev_idx
        )

        self.last_ref = x_ref
        self.last_cur = x_cur
        self.last_fs = fs

        self._clear_waves()
        self._plot_wave(self.ax_ref, x_ref, fs)
        self._plot_wave(self.ax_cur, x_cur, fs)
        self.canvas.draw_idle()

        self.test_name.set(datetime.now().strftime("Test_%Y-%m-%d_%H-%M-%S"))
        self._set_eval(res.overall)

        self._set_messages(
            [
                f"Resultado global: {res.overall}.",
                f"JSON guardado en: {out_path}",
                "Resultados enviados automáticamente a ThingsBoard.",
            ]
        )

        messagebox.showinfo(
            APP_NAME,
            f"Análisis terminado.\nResultado: {res.overall}\n\nJSON:\n{out_path}",
        )


def main():
    root = tb.Window(themename="flatly")
    app = AudioCinemaGUI(root)
    root.geometry("1020x640")
    root.minsize(900, 600)
    root.mainloop()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
