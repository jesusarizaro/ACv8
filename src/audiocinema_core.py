#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Núcleo de AudioCinema (versión corregida y mejorada):
- Manejo de configuración (YAML)
- Grabación de audio
- Análisis 6 canales REAL (detección de beeps en referencia y prueba)
- Fallback si no hay 7 banderas por pista (6 canales genéricos)
- Construcción de payload ThingsBoard
- Ejecución headless (modo --once y --scheduled)
"""

from __future__ import annotations
import os
import json
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime, date, timedelta
from typing import Dict, List, Tuple, Optional

import numpy as np
import sounddevice as sd
import soundfile as sf
import yaml
import paho.mqtt.client as mqtt

# =========================
#   RUTAS BÁSICAS
# =========================

APP_DIR = Path(__file__).resolve().parent
CONFIG_DIR = APP_DIR / "config"
DATA_DIR = APP_DIR / "data"
REPORTS_DIR = DATA_DIR / "reports"
ASSETS_DIR = APP_DIR / "assets"

CONFIG_PATH = CONFIG_DIR / "config.yaml"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
CONFIG_DIR.mkdir(parents=True, exist_ok=True)
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

# =========================
#   CONFIGURACIÓN
# =========================

DEFAULT_CONFIG: Dict = {
    "audio": {
        "fs": 48000,
        "duration_s": 10.0,
        "preferred_input_name": "",
    },
    "reference": {
        "path": str((ASSETS_DIR / "reference_master.wav").resolve())
    },
    "evaluation": {
        "level": "Medio",
        "tolerances": {
            "Bajo":  {"rms_db": 6.0, "band_db": 8.0, "crest_db": 6.0, "spec95_db": 18.0},
            "Medio": {"rms_db": 3.0, "band_db": 5.0, "crest_db": 4.0, "spec95_db": 12.0},
            "Alto":  {"rms_db": 1.5, "band_db": 3.0, "crest_db": 2.0, "spec95_db": 6.0},
        }
    },
    "thingsboard": {
        "host": "thingsboard.cloud",
        "port": 1883,
        "use_tls": False,
        "token": "",
    },
    "schedule": {
        "enabled": False,
        "mode": "daily",
        "weekday": 0,
        "hour": 22,
        "minute": 0,
        "last_run_date": "",
    }
}


def load_config() -> Dict:
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    else:
        data = {}

    cfg = DEFAULT_CONFIG | data
    for k in DEFAULT_CONFIG:
        if isinstance(DEFAULT_CONFIG[k], dict):
            cfg.setdefault(k, {})
            cfg[k] = DEFAULT_CONFIG[k] | cfg[k]

    return cfg


def save_config(cfg: Dict) -> None:
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)


# =========================
#   AUDIO UTILIDADES
# =========================

def normalize_mono(x: np.ndarray) -> np.ndarray:
    if x.ndim == 2:
        x = x.mean(axis=1)
    x = x.astype(np.float32, copy=False)
    peak = np.max(np.abs(x)) if x.size else 0.0
    if peak > 1.0:
        x = x / (peak + 1e-12)
    return x


def record_audio(duration_s: float, fs: int, channels: int = 1,
                 device: Optional[int] = None) -> np.ndarray:
    duration_s = max(0.5, float(duration_s))
    kwargs = dict(samplerate=fs, channels=channels, dtype="float32")
    if device is not None:
        kwargs["device"] = device
    rec = sd.rec(int(duration_s * fs), **kwargs)
    sd.wait()
    return normalize_mono(rec.squeeze())


def rms_db(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    return 20.0 * np.log10(np.sqrt(np.mean(x**2) + 1e-20) + 1e-20)


def crest_factor_db(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    peak = np.max(np.abs(x)) + 1e-20
    rms = np.sqrt(np.mean(x**2) + 1e-20)
    return 20.0 * np.log10(peak / rms)


def welch_psd_db(x: np.ndarray, fs: int, nperseg: int = 4096) -> Tuple[np.ndarray, np.ndarray]:
    """
    Welch muy simple sin SciPy.
    """
    x = np.asarray(x, dtype=np.float32)
    if len(x) < nperseg:
        pad = np.zeros(nperseg, dtype=np.float32)
        pad[:len(x)] = x
        x = pad

    noverlap = nperseg // 2
    step = nperseg - noverlap
    n = len(x)
    nwin = 1 + max(0, (n - nperseg) // step)

    if nwin <= 0:
        f = np.linspace(0, fs/2, nperseg//2 + 1)
        return f, np.full_like(f, -300.0, dtype=np.float32)

    frames = np.zeros((nwin, nperseg), dtype=np.float32)
    for i in range(nwin):
        s = i * step
        frames[i, :] = x[s:s+nperseg]

    win = np.hanning(nperseg).astype(np.float32)
    U = (win**2).sum()
    frames *= win[None, :]

    X = np.fft.rfft(frames, n=nperseg, axis=1)
    Pxx = (np.abs(X)**2) / (fs * U)
    Pxx = Pxx.mean(axis=0)

    f = np.fft.rfftfreq(nperseg, 1.0/fs)
    Pxx = np.maximum(Pxx, 1e-30)
    Pxx_db = 10.0 * np.log10(Pxx)
    return f.astype(np.float32), Pxx_db.astype(np.float32)


# =========================
#   DETECCIÓN DE BEEPS (ENERGÍA)
# =========================

def short_time_rms(x: np.ndarray, fs: int,
                   win_s: float = 0.02, hop_s: float = 0.01):
    win = max(1, int(round(win_s * fs)))
    hop = max(1, int(round(hop_s * fs)))
    n = len(x)

    frames = 1 + max(0, (n - win) // hop)
    rms_vals = np.zeros(frames, dtype=np.float32)
    times = np.zeros(frames, dtype=np.float32)

    for i in range(frames):
        s = i * hop
        e = s + win
        seg = x[s:e]
        rms_vals[i] = np.sqrt(np.mean(seg**2) + 1e-20)
        times[i] = (s + win/2) / fs

    return times, rms_vals


def detect_beep_markers(x: np.ndarray, fs: int,
                        thr_db_over_median: float = 10.0,
                        min_sep_s: float = 0.5) -> List[int]:
    """
    Detecta beeps por energía (no por frecuencia) y devuelve índices de muestra.
    Pensado para encontrar las 7 banderas, pero NO exige frecuencias concretas.
    """
    _, r = short_time_rms(x, fs)
    r_db = 20.0 * np.log10(r + 1e-20)
    med = np.median(r_db)
    thr = med + float(thr_db_over_median)
    above = r_db > thr

    beeps = []
    i = 0
    step_t = 0.01  # hop de RMS (10 ms)
    n = len(above)
    while i < n:
        if above[i]:
            j = i
            while j < n and above[j]:
                j += 1
            k = i + int(np.argmax(r_db[i:j]))
            beeps.append(k)
            i = j
        else:
            i += 1

    markers_s = np.array(beeps, dtype=float) * step_t
    markers = (markers_s * fs).astype(int)
    markers.sort()

    final = []
    last_t = -1e9
    for m in markers:
        t = m / fs
        if t - last_t >= min_sep_s:
            final.append(int(m))
            last_t = t
    return final


# =========================
#   RESULTADOS
# =========================

@dataclass
class ChannelResult:
    index: int
    evaluacion: str      # PASSED / FAILED
    estado: str          # VIVO / MUERTO
    ref_bands: Dict[str, float]
    cine_bands: Dict[str, float]
    delta_bands: Dict[str, float]
    rms_ref_db: float
    rms_cin_db: float
    crest_ref_db: float
    crest_cin_db: float
    spec95_db: float


@dataclass
class GlobalResult:
    overall: str
    level: str
    channels: List[ChannelResult]


# =========================
#   FALLBACK (SIN 7 BEEPS)
# =========================

def analyze_fallback(x_ref: np.ndarray,
                     x_cur: np.ndarray,
                     fs: int,
                     eval_level: str,
                     tolerances: Dict[str, Dict[str, float]]) -> GlobalResult:
    """
    Análisis simple cuando NO se detectan correctamente las 7 banderas
    en referencia y/o prueba.
    - Compara RMS y crest global.
    - Usa tolerancias según el nivel elegido.
    - Genera 6 canales genéricos.
    - Opción 3:
        * overall PASSED  → canales VIVO
        * overall FAILED → canales MUERTO
    """
    rms_ref = rms_db(x_ref)
    rms_cur = rms_db(x_cur)
    crest_ref = crest_factor_db(x_ref)
    crest_cur = crest_factor_db(x_cur)

    tol = tolerances.get(eval_level, tolerances["Medio"])
    tol_rms = tol["rms_db"]
    tol_crest = tol["crest_db"]

    fail_rms = abs(rms_cur - rms_ref) > tol_rms
    fail_crest = abs(crest_cur - crest_ref) > tol_crest

    if fail_rms or fail_crest:
        evaluacion_global = "FAILED"
    else:
        evaluacion_global = "PASSED"

    if evaluacion_global == "PASSED":
        estado_ch = "VIVO"
    else:
        estado_ch = "MUERTO"

    channels: List[ChannelResult] = []
    for idx in range(1, 7):
        ch = ChannelResult(
            index=idx,
            evaluacion=evaluacion_global,
            estado=estado_ch,
            ref_bands={},
            cine_bands={},
            delta_bands={},
            rms_ref_db=float(round(rms_ref, 3)),
            rms_cin_db=float(round(rms_cur, 3)),
            crest_ref_db=float(round(crest_ref, 3)),
            crest_cin_db=float(round(crest_cur, 3)),
            spec95_db=0.0,
        )
        channels.append(ch)

    return GlobalResult(
        overall=evaluacion_global,
        level="Fallback",
        channels=channels,
    )


# =========================
#   ANÁLISIS 6 CANALES “DUAL”
# =========================

BANDS = {
    "LFE": (30.0, 100.0),
    "LF":  (30.0, 120.0),
    "MF":  (120.0, 2000.0),
    "HF":  (2000.0, 8000.0),
}


def band_energy_db(f: np.ndarray, psd_db: np.ndarray,
                   band: Tuple[float, float]) -> float:
    f1, f2 = band
    mask = (f >= f1) & (f <= f2)
    if not np.any(mask):
        return -120.0
    p_lin = 10.0 ** (psd_db[mask] / 10.0)
    return 10.0 * np.log10(np.mean(p_lin) + 1e-30)


def analyze_6channels_dual(x_ref: np.ndarray, x_cur: np.ndarray, fs: int,
                           eval_level: str,
                           tolerances: Dict[str, Dict[str, float]]) -> GlobalResult:
    """
    Detección de beeps en referencia y prueba por separado y
    emparejamiento canal a canal.
    Si faltan beeps o los segmentos salen raros, se usa fallback.
    """
    mr = detect_beep_markers(x_ref, fs)
    mc = detect_beep_markers(x_cur, fs)

    if len(mr) < 7 or len(mc) < 7:
        print(f"[AudioCinema] Beeps insuficientes (ref={len(mr)}, cur={len(mc)}). Fallback.")
        return analyze_fallback(x_ref, x_cur, fs, eval_level, tolerances)

    guard = int(0.06 * fs)  # 60 ms
    segs: List[Tuple[int, int, int, int]] = []

    for i in range(6):
        a_ref = max(0, mr[i] + guard)
        b_ref = max(0, mr[i + 1] - guard)

        a_cur = max(0, mc[i] + guard)
        b_cur = max(0, mc[i + 1] - guard)

        if b_ref <= a_ref or b_cur <= a_cur:
            print("[AudioCinema] Segmento inválido, usando fallback.")
            return analyze_fallback(x_ref, x_cur, fs, eval_level, tolerances)

        segs.append((a_ref, b_ref, a_cur, b_cur))

    tol = tolerances.get(eval_level, tolerances["Medio"])
    tol_rms = tol["rms_db"]
    tol_band = tol["band_db"]
    tol_crest = tol["crest_db"]
    tol_spec = tol["spec95_db"]

    ch_results: List[ChannelResult] = []

    for idx, (ar, br, ac, bc) in enumerate(segs, start=1):
        seg_ref = x_ref[ar:br]
        seg_cur = x_cur[ac:bc]

        # RMS y Crest
        rms_ref = rms_db(seg_ref)
        rms_cin = rms_db(seg_cur)
        crest_ref = crest_factor_db(seg_ref)
        crest_cin = crest_factor_db(seg_cur)

        # PSD y bandas
        f_ref, psd_ref = welch_psd_db(seg_ref, fs)
        f_cin, psd_cin = welch_psd_db(seg_cur, fs)

        # Interpolar cine en frecuencia de referencia
        psd_cin_i = np.interp(f_ref, f_cin, psd_cin)
        rel_db = psd_cin_i - psd_ref

        bands_ref = {k: band_energy_db(f_ref, psd_ref, bnd) for k, bnd in BANDS.items()}
        bands_cin = {k: band_energy_db(f_ref, psd_cin_i, bnd) for k, bnd in BANDS.items()}
        delta = {k: (bands_cin[k] - bands_ref[k]) for k in BANDS}

        # métrica de espectro relativo
        mask = (f_ref >= 50.0) & (f_ref <= 8000.0)
        rel_abs = np.abs(rel_db[mask]) if np.any(mask) else np.abs(rel_db)
        spec95 = float(np.percentile(rel_abs, 95))

        # chequeos
        diff_rms = rms_cin - rms_ref
        diff_crest = crest_cin - crest_ref

        fail_rms = abs(diff_rms) > tol_rms
        fail_band = any(abs(v) > tol_band for v in delta.values())
        fail_crest = abs(diff_crest) > tol_crest
        fail_spec = spec95 > tol_spec

        # canal muerto si el RMS está MUY por debajo
        dead = (rms_cin < (rms_ref - 20.0)) or (rms_cin < -70.0)

        evaluacion = "PASSED"
        if fail_rms or fail_band or fail_crest or fail_spec or dead:
            evaluacion = "FAILED"

        estado = "MUERTO" if dead else "VIVO"

        ch_results.append(
            ChannelResult(
                index=idx,
                evaluacion=evaluacion,
                estado=estado,
                ref_bands={k: float(round(v, 3)) for k, v in bands_ref.items()},
                cine_bands={k: float(round(v, 3)) for k, v in bands_cin.items()},
                delta_bands={k: float(round(v, 3)) for k, v in delta.items()},
                rms_ref_db=float(round(rms_ref, 3)),
                rms_cin_db=float(round(rms_cin, 3)),
                crest_ref_db=float(round(crest_ref, 3)),
                crest_cin_db=float(round(crest_cin, 3)),
                spec95_db=float(round(spec95, 3)),
            )
        )

    overall = "PASSED"
    if any(ch.evaluacion != "PASSED" for ch in ch_results):
        overall = "FAILED"

    return GlobalResult(
        overall=overall,
        level=eval_level,
        channels=ch_results
    )


# =========================
#   PAYLOAD THINGSBOARD
# =========================

def build_thingsboard_payload(global_res: GlobalResult,
                              extra_meta: Optional[Dict] = None) -> Dict:
    """
    Construye el payload con Canal1..CanalN y resumen global.
    """
    payload: Dict = {}

    for ch in global_res.channels:
        name = f"Canal{ch.index}"
        payload[name] = {
            "Evaluacion": ch.evaluacion,
            "Estado": ch.estado,
            "ref": ch.ref_bands,
            "cine": ch.cine_bands,
            "delta": ch.delta_bands,
            "rms": {"ref_db": ch.rms_ref_db, "cin_db": ch.rms_cin_db},
            "crest": {"ref_db": ch.crest_ref_db, "cin_db": ch.crest_cin_db},
            "spec95_db": ch.spec95_db,
        }

    payload["Resumen"] = {
        "overall": global_res.overall,
        "level": global_res.level,
        "timestamp_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }

    if extra_meta:
        payload["Meta"] = extra_meta

    return payload


def send_to_thingsboard(payload: Dict, cfg: Dict) -> bool:
    tb = cfg["thingsboard"]
    token = tb.get("token", "").strip()
    if not token:
        return False

    host = tb.get("host", "thingsboard.cloud")
    port = int(tb.get("port", 1883))
    use_tls = bool(tb.get("use_tls", False))

    try:
        client_id = f"AudioCinemaPi-{os.uname().nodename}-{os.getpid()}"
        client = mqtt.Client(client_id=client_id, clean_session=True)
        client.username_pw_set(token)
        if use_tls:
            import ssl
            client.tls_set(cert_reqs=ssl.CERT_NONE)
            client.tls_insecure_set(True)

        client.connect(host, port, keepalive=30)
        topic = "v1/devices/me/telemetry"
        client.publish(topic, json.dumps(payload), qos=1)
        client.loop(timeout=2.0)
        client.disconnect()
        return True
    except Exception as e:
        print("Error MQTT:", e)
        return False


# =========================
#   LÓGICA DE MEDICIÓN
# =========================

def _load_reference(ref_path: Path, fs_target: int) -> np.ndarray:
    if not ref_path.exists():
        raise FileNotFoundError(f"Archivo de referencia no encontrado: {ref_path}")
    x, fs = sf.read(str(ref_path), dtype="float32", always_2d=False)
    x = normalize_mono(x)
    if fs != fs_target:
        n_new = int(round(len(x) * fs_target / fs))
        idx_old = np.linspace(0, 1, len(x))
        idx_new = np.linspace(0, 1, n_new)
        x = np.interp(idx_new, idx_old, x).astype(np.float32)
    return x


def run_measurement(device_index: Optional[int] = None):
    """
    Ejecuta una medición completa leyendo config.yaml.

    Devuelve SIEMPRE 7 valores:
      - global_res:   GlobalResult
      - payload:      dict
      - out_json:     Path
      - x_ref:        np.ndarray
      - x_cur:        np.ndarray
      - fs:           int
      - sent:         bool
    """
    cfg = load_config()
    fs = int(cfg["audio"]["fs"])
    dur = float(cfg["audio"]["duration_s"])

    # 1) referencia
    ref_path = Path(cfg["reference"]["path"])
    x_ref = _load_reference(ref_path, fs)

    target_len = int(round(fs * dur))
    if len(x_ref) > target_len:
        x_ref = x_ref[:target_len]
    elif len(x_ref) < target_len:
        x_ref = np.pad(x_ref, (0, target_len - len(x_ref)))

    # 2) prueba
    x_cur = record_audio(dur, fs=fs, channels=1, device=device_index)
    m = np.max(np.abs(x_cur)) + 1e-12
    x_cur = x_cur / m

    # asegurar igual longitud
    L = min(len(x_ref), len(x_cur))
    x_ref = x_ref[:L]
    x_cur = x_cur[:L]

    # 3) análisis
    eval_level = cfg["evaluation"]["level"]
    tolerances = cfg["evaluation"]["tolerances"]

    global_res = analyze_6channels_dual(x_ref, x_cur, fs, eval_level, tolerances)

    # 4) payload
    payload = build_thingsboard_payload(global_res)

    # 5) guardar JSON
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORTS_DIR / f"analysis_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    # 6) enviar
    sent = send_to_thingsboard(payload, cfg)
    payload["Meta"] = payload.get("Meta", {})
    payload["Meta"]["sent_to_thingsboard"] = sent

    # actualizar JSON con el flag
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return global_res, payload, out, x_ref, x_cur, fs, sent


# =========================
#   PROGRAMACIÓN (SCHEDULE)
# =========================

def compute_next_run_from_cfg(cfg, now=None):
    sch = cfg.get("schedule", {})
    if not sch.get("enabled"):
        return None

    if now is None:
        now = datetime.now()

    hour = int(sch["hour"])
    minute = int(sch["minute"])
    mode = sch["mode"]

    candidate = now.replace(hour=hour, minute=minute, second=0, microsecond=0)

    if mode == "daily":
        if candidate <= now:
            candidate += timedelta(days=1)
        return candidate

    if mode == "weekly":
        weekday = int(sch["weekday"])  # 0=lunes
        delta = (weekday - now.weekday()) % 7
        if delta == 0 and candidate <= now:
            delta = 7
        return candidate + timedelta(days=delta)


def should_run_now(cfg: Dict, now: Optional[datetime] = None) -> bool:
    sch = cfg.get("schedule", {})
    if not sch.get("enabled", False):
        return False
    if now is None:
        now = datetime.now()

    mode = sch.get("mode", "daily")
    hour = int(sch.get("hour", 22))
    minute = int(sch.get("minute", 0))

    if now.hour != hour or now.minute != minute:
        return False

    if mode == "weekly":
        weekday_target = int(sch.get("weekday", 0))
        if now.weekday() != weekday_target:
            return False

    last_run_str = sch.get("last_run_date", "")
    if last_run_str:
        try:
            last_run = date.fromisoformat(last_run_str)
            if last_run == now.date():
                return False
        except Exception:
            pass

    return True


def mark_run_today(cfg: Dict, now: Optional[datetime] = None) -> None:
    if now is None:
        now = datetime.now()
    cfg.setdefault("schedule", {})
    cfg["schedule"]["last_run_date"] = now.date().isoformat()
    save_config(cfg)


# =========================
#   CLI HEADLESS
# =========================

def main_cli():
    import argparse
    parser = argparse.ArgumentParser(description="AudioCinema core")
    parser.add_argument("--once", action="store_true",
                        help="Realiza una medición inmediata")
    parser.add_argument("--scheduled", action="store_true",
                        help="Se ejecuta como parte del timer; decide si correr o no")
    args = parser.parse_args()

    if args.once:
        res, payload, out, *_ = run_measurement(device_index=None)
        print(f"[AudioCinema] Resultado global: {res.overall}  JSON: {out}")
        return

    if args.scheduled:
        cfg = load_config()
        if should_run_now(cfg):
            print("[AudioCinema] Hora programada alcanzada, ejecutando medición…")
            res, payload, out, *_ = run_measurement(device_index=None)
            mark_run_today(cfg)
            print(f"[AudioCinema] Resultado global: {res.overall}  JSON: {out}")
        else:
            print("[AudioCinema] No corresponde correr en este minuto.")
        return

    parser.print_help()


if __name__ == "__main__":
    main_cli()
