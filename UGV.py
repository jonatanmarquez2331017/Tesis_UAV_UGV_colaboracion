import time, socket, threading
from queue import Queue, Empty
import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision import models
from pynput import keyboard

# ========= CONFIGURA AQUÍ =========
HOST = "192.168.0.100"
PORT = 1234
MODEL_PATH = r"C:\Users\UPT\PycharmProjects\TesisJonatan\model_best_ResNet.pth"
CAMERA_SRC = 0
TARGET_FPS = 30
CAP_WIDTH  = 224
CAP_HEIGHT = 224

# Límites físicos del servo
PWM_MIN    = 979
PWM_MAX    = 1979
PWM_MID    = 1535   # centro nominal

# ==== CONTROL DE ACELERACIÓN ====
PWM_STOP  = 1445
PWM_SLOW  = 1633
PWM_FAST  = 1650

# Modo de aceleración: "keyboard" o "cycle"
ACC_MODE = "keyboard"
acc_pwm  = PWM_STOP

# Parámetros del ciclo al presionar 'c'
CYCLE_PWM_A     = 1635
CYCLE_PWM_B     = 1610
CYCLE_PERIOD_S  = 0.3   # cada cuánto alterna
# ============================================

# EMA simple (0.0 = sin suavizado)
SMOOTH_ALPHA = 0.7

# ====== HOLD  ======
HOLD_THR = 1900          # si cruza este valor, activar hold
HOLD_MS  = 30           # duración del hold en milisegundos
_last_hold = None        # (hold_min_pwm, tstamp)

# Normalización ImageNet (igual que entrenamiento)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# ====== MODELO ======
class ResNet18Regressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet18(weights=None)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, 1)
    def forward(self, x):
        return self.backbone(x)

def _strip_module_prefix(sd: dict) -> dict:
    if not isinstance(sd, dict):
        return sd
    needs_strip = any(k.startswith("module.") for k in sd.keys())
    if not needs_strip:
        return sd
    return {k[len("module."):]: v for k, v in sd.items()}

# ====== PRE/POST ======
def preprocess_bgr(frame_bgr):
    if frame_bgr.shape[1] != CAP_WIDTH or frame_bgr.shape[0] != CAP_HEIGHT:
        frame_bgr = cv2.resize(frame_bgr, (CAP_WIDTH, CAP_HEIGHT), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img).unsqueeze(0)  # [1,3,H,W]

def clamp_pwm(v, lo, hi):
    return lo if v < lo else (hi if v > hi else v)

# ====== MULTIPLICADOR======
def apply_side_multiplier(pwm: float) -> float:

    if pwm >= 1600:
        pwm = pwm * 1.085
        if pwm > 1977:
            pwm = 1977.0
    elif pwm <= 1300:
        pwm = pwm * 1.045
    return pwm

# ====== HOLD: mantener mínimo durante HOLD_MS ======
def apply_hold(pwm: float) -> float:
    """
    Si el PWM actual >= HOLD_THR, inicia/renueva un 'hold' (mínimo a sostener).
    Mientras dure HOLD_MS, no dejar que baje de ese mínimo.
    """
    global _last_hold
    now = time.time()
    if pwm >= HOLD_THR:
        _last_hold = (pwm, now)
        return pwm
    if _last_hold is not None:
        hold_min, t0 = _last_hold
        if (now - t0) * 1000.0 <= HOLD_MS:
            return pwm if pwm >= hold_min else hold_min
        else:
            _last_hold = None
            return pwm
    return pwm

# ====== TCP ======
def connect_tcp(host, port, retry_sec=1.0):
    while True:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            s.connect((host, port))
            print(f"🔗 Conectado a {(host, port)}")
            return s
        except Exception as e:
            print(f"❌ No conecta ({e}). Reintento en {retry_sec}s...")
            time.sleep(retry_sec)

# ====== TECLADO (aceleración) ======
def on_press(key):
    global acc_pwm, ACC_MODE
    try:
        k = key.char.lower()
        if k == 'l':
            acc_pwm = PWM_SLOW; ACC_MODE = "keyboard"; print("[ACC] Lento 1655")
        elif k == 'r':
            acc_pwm = PWM_FAST; ACC_MODE = "keyboard"; print("[ACC] Rápido 1700")
        elif k == 's':
            acc_pwm = PWM_STOP; ACC_MODE = "keyboard"; print("[ACC] Stop 1445")
        elif k == 'c':
            ACC_MODE = "cycle"; print("[ACC] Ciclo 1655↔1650 cada 1.0 s (presiona L/R/S para salir)")
    except Exception:
        pass

# ====== COLAS ======
frame_q   = Queue(maxsize=1)   # último frame disponible
command_q = Queue(maxsize=1)   # último comando (dir, acc)

# ====== HILO: CAPTURA ======
def capture_thread(cam_index, fps):
    cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAP_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, fps)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir la cámara {cam_index}")
    print("🎥 Cámara lista")
    period = 1.0 / max(1, fps)

    try:
        while True:
            t0 = time.time()
            ok, frame = cap.read()
            if not ok:
                time.sleep(period)
                continue
            if frame_q.full():
                try: frame_q.get_nowait()
                except Empty: pass
            frame_q.put(frame, block=False)
            # cadencia
            dt = time.time() - t0
            to_wait = period - dt
            if to_wait > 0:
                time.sleep(to_wait)
    finally:
        cap.release()
        print("📷 Captura terminada")

# ====== HILO: GESTOR DE ACELERACIÓN======
def accel_manager_thread():
    global acc_pwm, ACC_MODE
    phase_start = time.monotonic()
    try:
        while True:
            if ACC_MODE == "cycle":
                elapsed = time.monotonic() - phase_start
                # alterna cada CYCLE_PERIOD_S entre A y B
                idx = int(elapsed // CYCLE_PERIOD_S) % 2
                acc_pwm = CYCLE_PWM_A if idx == 0 else CYCLE_PWM_B
                time.sleep(0.01)
            else:
                # resetear fase al salir del modo ciclo
                phase_start = time.monotonic()
                time.sleep(0.05)
    except Exception as e:
        print("❌ Error en gestor de aceleración:", e)

# ====== HILO: INFERENCIA ======
def inference_thread(model, device):
    torch.set_grad_enabled(False)
    dir_prev = None
    try:
        while True:
            frame = frame_q.get()  # último frame
            x = preprocess_bgr(frame).to(device, non_blocking=True)
            with torch.no_grad():
                pwm = float(model(x).item())  # red -> PWM crudo

            # 1) clamp básico
            pwm = clamp_pwm(pwm, PWM_MIN, PWM_MAX)

            # 2) EMA opcional
            if SMOOTH_ALPHA > 0.0:
                if dir_prev is None:
                    dir_prev = pwm
                pwm = SMOOTH_ALPHA * pwm + (1.0 - SMOOTH_ALPHA) * dir_prev
                dir_prev = pwm

            # 3) multiplicador (según umbrales)
            pwm = apply_side_multiplier(pwm)

            # 4) HOLD 0.5 s si superó 1600 (evita desistir la curva)
            pwm = apply_hold(pwm)

            # 5) límites finales
            pwm = clamp_pwm(pwm, PWM_MIN, PWM_MAX)

            # 6) publicar comando
            cmd = (int(round(pwm)), int(acc_pwm))
            if command_q.full():
                try: command_q.get_nowait()
                except Empty: pass
            command_q.put(cmd, block=False)
    except Exception as e:
        print("❌ Error en inferencia:", e)

# ====== HILO: ENVÍO TCP ======
def tcp_sender_thread(host, port, fps):
    sock = connect_tcp(host, port)
    period = 1.0 / max(1, fps)
    try:
        while True:
            t0 = time.time()
            try:
                dir_pwm, acc = command_q.get(timeout=1.0)
            except Empty:
                time.sleep(0.01)
                continue
            msg = f"{dir_pwm},{acc}\n".encode()
            try:
                sock.send(msg)
            except Exception as e:
                print("⚠️ Envío falló, reconectando...", e)
                try: sock.close()
                except: pass
                sock = connect_tcp(host, port)
                continue
            # cadencia de envío
            dt = time.time() - t0
            to_wait = period - dt
            if to_wait > 0:
                time.sleep(to_wait)
    finally:
        try: sock.close()
        except: pass
        print("📡 Envío TCP terminado")

# ====== MAIN ======
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Modelo
    model = ResNet18Regressor().to(device)
    ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    state = _strip_module_prefix(state)
    model.load_state_dict(state, strict=True)
    model.eval()
    print("✅ Modelo cargado en", device)

    # Teclado (no bloqueante)
    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    # Lanzar hilos
    th_cap  = threading.Thread(target=capture_thread, args=(CAMERA_SRC, TARGET_FPS), daemon=True)
    th_inf  = threading.Thread(target=inference_thread, args=(model, device), daemon=True)
    th_tcp  = threading.Thread(target=tcp_sender_thread, args=(HOST, PORT, TARGET_FPS), daemon=True)
    th_acc  = threading.Thread(target=accel_manager_thread, daemon=True)
    th_cap.start(); th_inf.start(); th_tcp.start(); th_acc.start()

    # Mantener vivo hasta Ctrl+C
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("🛑 Interrumpido por usuario")
    finally:
        listener.stop()
        print("✅ Salida limpia")

if __name__ == "__main__":
    main()
