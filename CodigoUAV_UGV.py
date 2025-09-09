import asyncio, time, math, os, platform
import cv2, numpy as np
from collections import deque

# ==== Parches de compatibilidad (DroneKit en Python 3.10+) ====
import collections, collections.abc
if not hasattr(collections, "MutableMapping"):
    collections.MutableMapping = collections.abc.MutableMapping
if not hasattr(collections, "Mapping"):
    collections.Mapping = collections.abc.Mapping
if not hasattr(collections, "Sequence"):
    collections.Sequence = collections.abc.Sequence

# ==== Parche de event loop en Windows ====
if platform.system() == "Windows":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from dronekit import connect, VehicleMode
from pymavlink import mavutil
from pynput import keyboard

import torch
import torch.nn as nn
from torchvision import models

# ========= CONFIGURACIÓN =========
# ---- Dron ----
DRONE_CONNECTION = ['udp:0.0.0.0:14550']   # múltiples endpoints para fallback
DRONE_CAM_INDEX  = 2
ALTURA_DESEADA   = 1.10
TOLERANCIA_POS   = 0.02
THRUST_BASE      = 0.495
THRUST_MIN       = 0.30
THRUST_MAX       = 0.60
THRUST_HOVER_GUARD = 0.52   # límite superior suave cuando Z es incierta

Kp_z, Ki_z, Kd_z = 0.28, 0.001, 0.205
Kp_x, Ki_x, Kd_x = 9.0, 0.19, 5.0
Kp_y, Ki_y, Kd_y = 9.0, 0.19, 5.0

CALIB_FILE       = "parametros_calibrados.npz"
ARUCO_DICT_ID    = cv2.aruco.DICT_5X5_250
MARKER_LENGTH    = 0.12559      # m
ID_CENTRADO      = 105
IDS_ALTURA_MIN, IDS_ALTURA_MAX = 101, 108
ARUCO_MAX_DIST   = 3.0

# Fusión Z y filtros
FUSION_UMBRAL_M  = 1.20
LASER_MIN, LASER_MAX     = 0.15, 3.0
LASER_MAX_JUMP           = 0.35
ARUCO_LOSS_HOLD          = 0.35
Z_RATE_LIMIT             = 1.2
Z_ALPHA                  = 0.35

DRONE_SEND_HZ    = 20  # 50 ms

# ---- Carrito ----
CAR_HOST       = "192.168.0.100"
CAR_PORT       = 1234
MODEL_PATH     = r"C:\Users\UPT\PycharmProjects\TesisJonatan\model_best_ResNet.pth"
CAR_CAM_INDEX  = 0
TARGET_FPS     = 30
CAP_WIDTH      = 224
CAP_HEIGHT     = 224

PWM_MIN, PWM_MAX, PWM_MID =  979, 1979, 1535
PWM_STOP, PWM_SLOW, PWM_FAST = 1445, 1633, 1660

ACC_MODE         = "keyboard"  # "keyboard" o "cycle"
ACC_PWM          = PWM_STOP
CYCLE_PWM_A      = 1635
CYCLE_PWM_B      = 1610
CYCLE_PERIOD_S   = 0.3

SMOOTH_ALPHA     = 0.3
SIDE_UP_THR      = 1600
SIDE_LO_THR      = 1300
SIDE_UP_GAIN     = 1.085  # límite 1977
SIDE_LO_GAIN     = 1.047
HOLD_THR         = 1900
HOLD_MS          = 5      # ms

# ---- Previsualización ----
PREVIEW_HEIGHT   = 360
MAX_CAM_IDX_BOUNDS = (0, 9)

# ---- YOLO (Ultralytics) ----
USE_YOLO         = True
YOLO_WEIGHTS     = r"C:\Users\UPT\PycharmProjects\TesisJonatan\best.pt"
YOLO_WINDOW_NAME = "Detección YOLO"

# === Peligro (override de aceleración a 1445) ===
DANGER_CLASSES = {"Cajas_peligrosas", "Palets_peligrosos", "cono_peligroso"}
DANGER_CONF_THR = 0.70
DANGER_HOLD_S   = 0.6  # retención mínima tras última detección

# ========= ESTADO GLOBAL DRON =========
vehicle = None
vuelo_activo = True
t_inicio = time.time()

pitch_objetivo = 0.0
roll_objetivo  = 0.0
thrust_objetivo = THRUST_BASE

x_actual = 0.0
y_actual = 0.0
altura_estimada = 0.0

# Logs opcionales
valores_altura = deque(maxlen=10)
log_altura, log_altura_tiempo = [], []
log_x, log_y, log_deteccion = [], [], []
log_roll, log_pitch, log_roll_tiempo, log_pitch_tiempo = [], [], [], []

# Señal global de peligro
danger_until_ts = 0.0  # si time.time() < danger_until_ts => forzar 1445

# ========= Utilidades genéricas =========
def _clamp_idx(v: int):
    lo, hi = MAX_CAM_IDX_BOUNDS
    return max(lo, min(hi, v))

def _open_cap(index: int):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(index)
    return cap

def _reopen(cap, new_idx):
    if cap is not None:
        cap.release()
    cap2 = _open_cap(new_idx)
    if not cap2 or not cap2.isOpened():
        print(f"⚠️ No se pudo abrir cámara {new_idx}")
    return cap2

def _read_frame_or_dummy(cap, label, h=PREVIEW_HEIGHT):
    ok, frame = cap.read() if cap is not None and cap.isOpened() else (False, None)
    if not ok or frame is None:
        frame = np.zeros((h, int(h*16/9), 3), dtype=np.uint8)
        cv2.putText(frame, f"{label}: SIN SENAL", (10, h//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2, cv2.LINE_AA)
    else:
        H, W = frame.shape[:2]
        scale = h / max(1, H)
        frame = cv2.resize(frame, (int(W*scale), h), interpolation=cv2.INTER_AREA)
        cv2.putText(frame, label, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2, cv2.LINE_AA)
    return frame

# ========= Previsualización de cámaras (DRON | CARRITO) =========
def preview_cameras():
    global DRONE_CAM_INDEX, CAR_CAM_INDEX
    cap_d = _open_cap(DRONE_CAM_INDEX)
    cap_c = _open_cap(CAR_CAM_INDEX)
    win = "Preview Dron | Carrito (ajusta índices y presiona ESPACIO)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    print("\n== Previsualización de cámaras ==")
    print("A/D -> DRON índice -/+,  J/L -> CARRITO índice -/+")
    print("ESPACIO -> confirmar y comenzar   |   ESC -> salir\n")

    while True:
        fD = _read_frame_or_dummy(cap_d, f"DRON cam = {DRONE_CAM_INDEX}", PREVIEW_HEIGHT)
        fC = _read_frame_or_dummy(cap_c, f"CARRITO cam = {CAR_CAM_INDEX}", PREVIEW_HEIGHT)

        h = max(fD.shape[0], fC.shape[0])
        canvas = np.zeros((h, fD.shape[1] + fC.shape[1], 3), dtype=np.uint8)
        canvas[:fD.shape[0], :fD.shape[1]] = fD
        canvas[:fC.shape[0], fD.shape[1]:fD.shape[1]+fC.shape[1]] = fC

        cv2.putText(canvas, "A/D: Dron -/+,  J/L: Carrito -/+,  ESPACIO: Confirmar,  ESC: Salir",
                    (10, h-12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)

        cv2.imshow(win, canvas)
        k = cv2.waitKey(1) & 0xFF

        if k == 27:  # ESC
            cv2.destroyWindow(win)
            if cap_d is not None: cap_d.release()
            if cap_c is not None: cap_c.release()
            raise SystemExit("Previsualización cancelada por el usuario.")
        elif k == 32:  # SPACE
            cv2.destroyWindow(win)
            if cap_d is not None: cap_d.release()
            if cap_c is not None: cap_c.release()
            print(f"✅ Confirmado: DRON={DRONE_CAM_INDEX}, CARRITO={CAR_CAM_INDEX}")
            return
        elif k == ord('a'):
            DRONE_CAM_INDEX = _clamp_idx(DRONE_CAM_INDEX - 1)
            cap_d = _reopen(cap_d, DRONE_CAM_INDEX)
        elif k == ord('d'):
            DRONE_CAM_INDEX = _clamp_idx(DRONE_CAM_INDEX + 1)
            cap_d = _reopen(cap_d, DRONE_CAM_INDEX)
        elif k == ord('j'):
            CAR_CAM_INDEX = _clamp_idx(CAR_CAM_INDEX - 1)
            cap_c = _reopen(cap_c, CAR_CAM_INDEX)
        elif k == ord('l'):
            CAR_CAM_INDEX = _clamp_idx(CAR_CAM_INDEX + 1)
            cap_c = _reopen(cap_c, CAR_CAM_INDEX)

# ========= Conexión al dron =========
async def drone_connect_only():
    global vehicle
    print("⏳ Conectando con el dron (solo conexión)…")
    last_err = None
    for uri in DRONE_CONNECTION:
        try:
            print(f"  • Probando: {uri}")
            v = await asyncio.to_thread(connect, uri, wait_ready=False)
            # Espera de heartbeat manual (no bloqueante)
            ok = False; t0 = time.time()
            while time.time() - t0 < 10.0:
                if getattr(v, "last_heartbeat", None) is not None:
                    ok = True; break
                await asyncio.sleep(0.2)
            if not ok:
                try: v.close()
                except: pass
                print(f"    ✖ Sin heartbeat en {uri}")
                continue
            vehicle = v
            print(f"✅ Conectado vía: {uri}")
            return
        except Exception as e:
            last_err = e
            print(f"    ✖ Falló {uri}: {e}")
    raise RuntimeError(f"No se pudo conectar al dron. Último error: {last_err}")

async def drone_prepare_and_arm():
    try:
        vehicle.parameters['ARMING_CHECK'] = 0
    except Exception:
        pass
    vehicle.mode = VehicleMode("GUIDED_NOGPS")
    t0 = time.time()
    while getattr(getattr(vehicle, "mode", None), "name", None) != "GUIDED_NOGPS":
        if time.time() - t0 > 5.0:
            print("⚠️ No se confirmó GUIDED_NOGPS en 5 s (continuo)…")
            break
        await asyncio.sleep(0.2)
    if not vehicle.armed:
        print("🔐 Intentando armar…")
        vehicle.armed = True
        t0 = time.time()
        while not vehicle.armed:
            if time.time() - t0 > 12.0:
                raise RuntimeError("No se pudo armar el dron en 12 s.")
            await asyncio.sleep(0.2)
    print("✅ Dron armado y listo")

async def countdown(seconds: int = 5):
    for s in range(seconds, 0, -1):
        print(f"Arrancando en {s}…")
        await asyncio.sleep(1.0)

# ========= MAVLink helpers =========
def _yaw_deg_safe(vehicle, yaw_angle):
    if yaw_angle is None:
        try:
            y = getattr(vehicle.attitude, "yaw", None)  # rad
            if y is None or not np.isfinite(y): return 0.0
            return math.degrees(float(y))
        except Exception:
            return 0.0
    try:
        y = float(yaw_angle)
        return y if np.isfinite(y) else 0.0
    except Exception:
        return 0.0

def to_quaternion(roll_deg=0.0, pitch_deg=0.0, yaw_deg=0.0):
    cr = math.cos(math.radians(roll_deg  * 0.5))
    sr = math.sin(math.radians(roll_deg  * 0.5))
    cp = math.cos(math.radians(pitch_deg * 0.5))
    sp = math.sin(math.radians(pitch_deg * 0.5))
    cy = math.cos(math.radians(yaw_deg   * 0.5))
    sy = math.sin(math.radians(yaw_deg   * 0.5))
    w = cy*cr*cp + sy*sr*sp
    x = cy*sr*cp - sy*cr*sp
    y = cy*cr*sp + sy*sr*cp
    z = sy*cr*cp - cy*sr*sp
    return [w, x, y, z]

def send_attitude_target(roll_angle=0.0, pitch_angle=0.0, yaw_angle=None, thrust=0.5):
    try: r = float(roll_angle)
    except: r = 0.0
    try: p = float(pitch_angle)
    except: p = 0.0
    try: u = float(thrust)
    except: u = 0.5
    y_deg = _yaw_deg_safe(vehicle, yaw_angle)
    q = to_quaternion(r, p, y_deg)
    msg = vehicle.message_factory.set_attitude_target_encode(
        0, 1, 1, 0b00000111, q, 0, 0, 0, u)
    vehicle.send_mavlink(msg)
    vehicle.flush()

# ========= Detector ArUco (compatibilidad OpenCV) =========
def make_detector():
    try:
        aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID)
        params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, params)
        def detect(gray): return detector.detectMarkers(gray)
        return aruco_dict, detect, True
    except AttributeError:
        aruco_dict = cv2.aruco.Dictionary_get(ARUCO_DICT_ID)
        params = cv2.aruco.DetectorParameters_create()
        def detect(gray): return cv2.aruco.detectMarkers(gray, aruco_dict, parameters=params)
        return aruco_dict, detect, False

ARUCO_DICT, DETECT, _OPENCV_NEW = make_detector()

# ========= Filtro y validación de altura (Z) =========
class ZFilter:
    def __init__(self, z0=0.0, alpha=Z_ALPHA, rate_lim=Z_RATE_LIMIT):
        self.z = z0
        self.alpha = alpha
        self.rate = rate_lim
        self.t_last = time.time()
    def update(self, z_meas):
        t = time.time()
        dt = max(1e-3, t - self.t_last)
        max_step = self.rate * dt
        z_cand = self.z + np.clip(z_meas - self.z, -max_step, +max_step)
        self.z = self.alpha * z_cand + (1 - self.alpha) * self.z
        self.t_last = t
        return self.z

z_filter = ZFilter()

def laser_is_valid(z_now, z_prev):
    if not (LASER_MIN <= z_now <= LASER_MAX):
        return False
    if z_prev is None:
        return True
    if abs(z_now - z_prev) > LASER_MAX_JUMP:
        return False
    return True

# ========= Cámara compartida DRON -> (ArUco, YOLO) =========
drone_cam_q = asyncio.Queue(maxsize=1)
yolo_cam_q  = asyncio.Queue(maxsize=1)

async def shared_drone_cam_loop():
    cap = cv2.VideoCapture(DRONE_CAM_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(DRONE_CAM_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir la cámara del dron (índice {DRONE_CAM_INDEX})")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS,          30)
    try: cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    except Exception: pass

    period = 1.0 / 30.0
    print(f"📷 Cámara DRON/YOLO lista (índice {DRONE_CAM_INDEX})")
    try:
        while vuelo_activo:
            t0 = time.time()
            ok, frame = await asyncio.to_thread(cap.read)
            if not ok or frame is None:
                await asyncio.sleep(0.005); continue

            # Fan-out: sustituir el frame más antiguo si la cola está llena
            for q in (drone_cam_q, yolo_cam_q):
                if q.full():
                    try: q.get_nowait()
                    except asyncio.QueueEmpty: pass
                await q.put(frame)

            await asyncio.sleep(max(0.0, period - (time.time() - t0)))
    finally:
        cap.release()
        print("📷 Cámara DRON/YOLO cerrada")

# ========= Visión del dron (ArUco + fusión Z) =========
async def drone_vision_loop():
    global x_actual, y_actual, altura_estimada

    if not os.path.exists(CALIB_FILE):
        raise FileNotFoundError(f"No está {CALIB_FILE}")
    param = np.load(CALIB_FILE, allow_pickle=True)
    K_base = param["K"].copy()
    dist   = param["dist"].copy()

    # Ajuste de K a la resolución del primer frame
    frame0 = await drone_cam_q.get()
    h, w = frame0.shape[:2]
    calib_w = int(param["img_w"]) if "img_w" in param else (int(param["img_size"][0]) if "img_size" in param else w)
    calib_h = int(param["img_h"]) if "img_h" in param else (int(param["img_size"][1]) if "img_size" in param else h)
    if (w != calib_w) or (h != calib_h):
        sx, sy = w/float(calib_w), h/float(calib_h)
        K = K_base.copy(); K[0,0]*=sx; K[0,2]*=sx; K[1,1]*=sy; K[1,2]*=sy
    else:
        K = K_base.copy()

    last_aruco_t = 0.0
    last_aruco_z = None
    last_laser_z = None

    win = "Camara Dron + ArUco"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    while vuelo_activo:
        frame = await drone_cam_q.get()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = DETECT(gray)

        out = frame.copy()
        detect_centro = False
        alturas_arucos = []

        if ids is not None and len(ids) > 0:
            try: cv2.aruco.drawDetectedMarkers(out, corners, ids)
            except: pass
            pose = cv2.aruco.estimatePoseSingleMarkers(corners, MARKER_LENGTH, K, dist)
            rvecs, tvecs = (pose[:2] if len(pose)==3 else pose)

            for i in range(len(ids)):
                mid = int(ids[i][0])
                tvec = tvecs[i][0]
                rvec = rvecs[i]
                try: cv2.drawFrameAxes(out, K, dist, rvec, tvec, MARKER_LENGTH*0.5)
                except:
                    try: cv2.aruco.drawAxis(out, K, dist, rvec, tvec, MARKER_LENGTH*0.5)
                    except: pass

                if mid == ID_CENTRADO:
                    detect_centro = True
                    x_actual = float(tvec[0])
                    y_actual = float(tvec[1])
                    cv2.putText(out, "Viendo ArUco 105", (15, 35),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2, cv2.LINE_AA)
                if IDS_ALTURA_MIN <= mid <= IDS_ALTURA_MAX:
                    z = float(tvec[2])
                    if z < ARUCO_MAX_DIST:
                        alturas_arucos.append(z)
        else:
            cv2.putText(out, "No se detecta ArUco", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        # Láser y fusión
        z_laser = float(getattr(vehicle.rangefinder, "distance", 0.0) or 0.0)
        now = time.time()
        if alturas_arucos:
            z_aruco = float(np.median(alturas_arucos))
            last_aruco_t, last_aruco_z = now, z_aruco
            if z_aruco < FUSION_UMBRAL_M and laser_is_valid(z_laser, last_laser_z):
                z_meas = 0.5*z_aruco + 0.5*z_laser
            else:
                z_meas = z_aruco
        else:
            if (now - last_aruco_t) <= ARUCO_LOSS_HOLD and last_aruco_z is not None:
                z_meas = last_aruco_z
            else:
                z_meas = z_laser if laser_is_valid(z_laser, last_laser_z) else altura_estimada
        last_laser_z = z_laser if z_laser > 0 else last_laser_z

        altura_estimada = z_filter.update(z_meas)
        t = time.time() - t_inicio
        log_altura.append(altura_estimada); log_altura_tiempo.append(t)
        log_x.append(x_actual); log_y.append(y_actual)
        log_deteccion.append(1 if detect_centro else 0)

        arm_state = "ARMADO" if vehicle.armed else "DESARMADO"
        mode_name = getattr(vehicle.mode, "name", "N/A")
        cv2.putText(out, f"z_est: {altura_estimada:.2f} m | Modo: {mode_name} | {arm_state}",
                    (10, max(60, out.shape[0]-20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)

        cv2.imshow(win, out)
        cv2.waitKey(1)  # mantener ventana activa

# ========= PIDs del dron =========
async def drone_pid_altura():
    global thrust_objetivo
    integral = 0.0; prev_error = 0.0
    dt = 1.0/DRONE_SEND_HZ
    while vuelo_activo:
        error = ALTURA_DESEADA - altura_estimada
        derivada = (error - prev_error)/dt
        u = THRUST_BASE + Kp_z*error + Kd_z*derivada + Ki_z*integral
        # anti-windup
        if THRUST_MIN < u < min(THRUST_MAX, THRUST_HOVER_GUARD):
            integral += error * dt
        else:
            integral *= 0.9
        u = max(min(u, min(THRUST_MAX, THRUST_HOVER_GUARD)), THRUST_MIN)
        thrust_objetivo = float(u)
        prev_error = error
        await asyncio.sleep(dt)

async def drone_pid_roll():
    global roll_objetivo
    integral = 0.0; prev_error = 0.0
    dt = 1.0/DRONE_SEND_HZ
    while vuelo_activo:
        error = x_actual
        derivada = (error - prev_error)/dt
        u = Kp_x*error + Kd_x*derivada + Ki_x*integral
        if -5 < u < 5: integral += error * dt
        roll_objetivo = 0.0 if abs(error) < TOLERANCIA_POS else float(np.clip(u, -5, 5))
        prev_error = error
        await asyncio.sleep(dt)

async def drone_pid_pitch():
    global pitch_objetivo
    integral = 0.0; prev_error = 0.0
    dt = 1.0/DRONE_SEND_HZ
    while vuelo_activo:
        error = y_actual
        derivada = (error - prev_error)/dt
        u = Kp_y*error + Kd_y*derivada + Ki_y*integral
        if -5 < u < 5: integral += error * dt
        pitch_objetivo = 0.0 if abs(error) < TOLERANCIA_POS else float(np.clip(u, -5, 5))
        prev_error = error
        await asyncio.sleep(dt)

async def drone_attitude_sender():
    dt = 1.0/DRONE_SEND_HZ
    while vuelo_activo:
        send_attitude_target(roll_angle=roll_objetivo, pitch_angle=pitch_objetivo, thrust=thrust_objetivo)
        t = time.time() - t_inicio
        log_roll.append(roll_objetivo); log_pitch.append(pitch_objetivo)
        log_roll_tiempo.append(t);      log_pitch_tiempo.append(t)
        await asyncio.sleep(dt)

# ========= Carrito: modelo y utilidades =========
class ResNet18Regressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet18(weights=None)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, 1)
    def forward(self, x): return self.backbone(x)

def _strip_module_prefix(sd: dict) -> dict:
    if not isinstance(sd, dict): return sd
    needs_strip = any(k.startswith("module.") for k in sd.keys())
    return {k[len("module."):] if k.startswith("module.") else k: v for k, v in sd.items()} if needs_strip else sd

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def preprocess_bgr_car(frame_bgr):
    if frame_bgr.shape[1] != CAP_WIDTH or frame_bgr.shape[0] != CAP_HEIGHT:
        frame_bgr = cv2.resize(frame_bgr, (CAP_WIDTH, CAP_HEIGHT), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
    img = (img - IMAGENET_MEAN)/IMAGENET_STD
    img = np.transpose(img, (2,0,1))
    return torch.from_numpy(img).unsqueeze(0)

def clamp_pwm(v, lo, hi):
    return lo if v < lo else (hi if v > hi else v)

def apply_side_multiplier(pwm: float) -> float:
    if pwm >= SIDE_UP_THR:
        pwm = pwm * SIDE_UP_GAIN
        if pwm > 1977: pwm = 1977.0
    elif pwm <= SIDE_LO_THR:
        pwm = pwm * SIDE_LO_GAIN
    return pwm

_last_hold = None  # (hold_min_pwm, tstamp)
def apply_hold(pwm: float) -> float:
    global _last_hold
    now = time.time()
    if pwm >= HOLD_THR:
        _last_hold = (pwm, now)
        return pwm
    if _last_hold is not None:
        hold_min, t0 = _last_hold
        if (now - t0)*1000.0 <= HOLD_MS:
            return pwm if pwm >= hold_min else hold_min
        else:
            _last_hold = None
    return pwm

# ====== Colas y estado del carrito ======
frame_q_car   = asyncio.Queue(maxsize=1)
command_q_car = asyncio.Queue(maxsize=1)
last_dir_pwm  = PWM_MID  # para envío inmediato al pulsar teclas

# ====== Teclado de aceleración ======
def on_press(key):
    global ACC_PWM, ACC_MODE, last_dir_pwm
    try:
        k = key.char.lower()
        pushed = False
        if k == 'l':
            ACC_PWM = PWM_SLOW; ACC_MODE = "keyboard"; pushed = True; print("[ACC] Lento 1633")
        elif k == 'r':
            ACC_PWM = PWM_FAST; ACC_MODE = "keyboard"; pushed = True; print("[ACC] Rápido 1650")
        elif k == 's':
            ACC_PWM = PWM_STOP; ACC_MODE = "keyboard"; pushed = True; print("[ACC] Stop 1445")
        elif k == 'c':
            ACC_MODE = "cycle"; pushed = True; print(f"[ACC] Ciclo {CYCLE_PWM_A}↔{CYCLE_PWM_B} cada {CYCLE_PERIOD_S}s (L/R/S para salir)")
        if pushed:
            try:
                if command_q_car.full():
                    command_q_car.get_nowait()
                command_q_car.put_nowait((int(last_dir_pwm), int(ACC_PWM)))
                print(f"[ACC->TX] dir={int(last_dir_pwm)} acc={int(ACC_PWM)}")
            except Exception as e:
                print("[ACC->TX] no se pudo encolar:", e)
    except Exception:
        pass

async def car_camera_loop():
    cap = cv2.VideoCapture(CAR_CAM_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened(): cap = cv2.VideoCapture(CAR_CAM_INDEX)
    if not cap.isOpened(): raise RuntimeError(f"No se pudo abrir la cámara del carrito (índice {CAR_CAM_INDEX})")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAP_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
    print(f"📷 Cámara del carrito lista (índice {CAR_CAM_INDEX})")
    period = 1.0/max(1, TARGET_FPS)
    try:
        while vuelo_activo:
            t0 = time.time()
            ok, frame = await asyncio.to_thread(cap.read)
            if ok and frame is not None:
                if frame_q_car.full():
                    try: frame_q_car.get_nowait()
                    except asyncio.QueueEmpty: pass
                await frame_q_car.put(frame)
            dt = time.time()-t0
            await asyncio.sleep(max(0.0, period - dt))
    finally:
        cap.release()
        print("📷 Captura carrito terminada")

async def car_inference_loop():
    global last_dir_pwm
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet18Regressor().to(device)
    ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    state = _strip_module_prefix(state)
    model.load_state_dict(state, strict=True)
    model.eval()
    print("✅ Modelo cargado en", device)

    dir_prev = None
    while vuelo_activo:
        frame = await frame_q_car.get()
        x = preprocess_bgr_car(frame).to(device, non_blocking=True)
        pwm = await asyncio.to_thread(lambda: float(model(x).item()))
        pwm = clamp_pwm(pwm, PWM_MIN, PWM_MAX)
        if SMOOTH_ALPHA > 0.0:
            if dir_prev is None: dir_prev = pwm
            pwm = SMOOTH_ALPHA*pwm + (1.0-SMOOTH_ALPHA)*dir_prev
            dir_prev = pwm
        pwm = apply_side_multiplier(pwm)
        pwm = apply_hold(pwm)
        pwm = clamp_pwm(pwm, PWM_MIN, PWM_MAX)

        last_dir_pwm = int(round(pwm))  # para envío inmediato al pulsar teclas
        # La aceleración se decide en el sender (puede forzarse a 1445 por peligro)
        cmd = (last_dir_pwm, int(ACC_PWM))
        if command_q_car.full():
            try: command_q_car.get_nowait()
            except asyncio.QueueEmpty: pass
        await command_q_car.put(cmd)

async def car_accel_manager_loop():
    global ACC_PWM, ACC_MODE
    phase_start = time.monotonic()
    while vuelo_activo:
        if ACC_MODE == "cycle":
            elapsed = time.monotonic() - phase_start
            idx = int(elapsed // CYCLE_PERIOD_S) % 2
            ACC_PWM = CYCLE_PWM_A if idx == 0 else CYCLE_PWM_B
            await asyncio.sleep(0.01)
        else:
            phase_start = time.monotonic()
            await asyncio.sleep(0.05)

async def car_tcp_sender_loop():
    global danger_until_ts
    last_cmd = (PWM_MID, ACC_PWM)
    while vuelo_activo:
        try:
            reader, writer = await asyncio.open_connection(CAR_HOST, CAR_PORT)
            print(f"🔗 Conectado a {(CAR_HOST, CAR_PORT)}")
            # Opciones de socket
            try:
                sock = writer.get_extra_info('socket')
                if sock is not None:
                    import socket as _s
                    sock.setsockopt(_s.IPPROTO_TCP, _s.TCP_NODELAY, 1)
                    sock.setsockopt(_s.SOL_SOCKET, _s.SO_KEEPALIVE, 1)
            except Exception:
                pass

            period = 1.0/max(1, TARGET_FPS)
            while vuelo_activo:
                t0 = time.time()
                try:
                    dir_pwm, acc = await asyncio.wait_for(command_q_car.get(), timeout=0.2)
                    last_cmd = (dir_pwm, acc)
                except asyncio.TimeoutError:
                    dir_pwm, acc = last_cmd[0], ACC_PWM  # keep-alive

                # Override por peligro
                if time.time() < danger_until_ts:
                    acc = PWM_STOP  # 1445
                else:
                    acc = ACC_PWM

                writer.write(f"{dir_pwm},{acc}\n".encode())
                await writer.drain()
                await asyncio.sleep(max(0.0, period - (time.time()-t0)))
        except Exception as e:
            print(f"⚠️ TCP caído: {e}. Reintentando en 1 s…")
            await asyncio.sleep(1.0)

# ========= YOLO (Ultralytics) leyendo de la misma cámara =========
async def yolo_loop():
    global danger_until_ts
    try:
        from ultralytics import YOLO
    except Exception as e:
        print("⚠️ Ultralytics no disponible:", e)
        return
    if not os.path.exists(YOLO_WEIGHTS):
        print(f"⚠️ No se encontró el modelo YOLO en {YOLO_WEIGHTS}")
        return

    model = YOLO(YOLO_WEIGHTS)
    cv2.namedWindow(YOLO_WINDOW_NAME, cv2.WINDOW_NORMAL)

    # Helper para evaluar peligro
    def result_has_danger(res):
        """
        True si hay detección con clase peligrosa y confianza >= umbral.
        """
        try:
            boxes = res.boxes
            if boxes is None or len(boxes) == 0:
                return False
            names = res.names if hasattr(res, "names") and isinstance(res.names, dict) else getattr(model, "names", {})
            cls_idx = boxes.cls.cpu().numpy().astype(int)
            confs   = boxes.conf.cpu().numpy()
            for c, cf in zip(cls_idx, confs):
                label = names.get(int(c), str(int(c)))
                if (label in DANGER_CLASSES) and (float(cf) >= DANGER_CONF_THR):
                    return True
            return False
        except Exception:
            return False

    while vuelo_activo:
        frame = await yolo_cam_q.get()

        # Inferencia en thread para no bloquear
        def infer_one(img):
            res = model(img)
            return res[0]  # objeto Results

        res = await asyncio.to_thread(infer_one, frame)

        # Actualizar peligro y dibujar
        if result_has_danger(res):
            danger_until_ts = time.time() + DANGER_HOLD_S

        annotated = res.plot()
        status = "PELIGRO -> 1445" if time.time() < danger_until_ts else "normal"
        cv2.putText(annotated, f"Estado: {status}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)

        cv2.imshow(YOLO_WINDOW_NAME, annotated)
        cv2.waitKey(1)  # mantener ventana activa

# ========= MAIN =========
async def main():
    global vuelo_activo

    # 0) Conectar dron
    await drone_connect_only()

    # 1) Arrancar TCP del carrito y teclado (L/R/S sin delay)
    kb_listener = keyboard.Listener(on_press=on_press)
    kb_listener.start()
    tcp_task = asyncio.create_task(car_tcp_sender_loop(), name="car_tcp")

    # 2) Previsualización para ajustar índices
    await asyncio.to_thread(preview_cameras)

    # 3) Cuenta regresiva y armado
    await countdown(5)
    await drone_prepare_and_arm()

    # 4) Tareas principales
    tasks = [
        tcp_task,
        asyncio.create_task(shared_drone_cam_loop(),  name="cam_shared"),
        asyncio.create_task(drone_vision_loop(),      name="drone_vision"),
        asyncio.create_task(drone_pid_altura(),       name="drone_pid_z"),
        asyncio.create_task(drone_pid_roll(),         name="drone_pid_x"),
        asyncio.create_task(drone_pid_pitch(),        name="drone_pid_y"),
        asyncio.create_task(drone_attitude_sender(),  name="drone_att_send"),
        asyncio.create_task(car_camera_loop(),        name="car_cam"),
        asyncio.create_task(car_inference_loop(),     name="car_infer"),
        asyncio.create_task(car_accel_manager_loop(), name="car_accel"),
    ]
    if USE_YOLO:
        tasks.append(asyncio.create_task(yolo_loop(), name="yolo"))

    # 5) Esperar fin
    try:
        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        pass
    except SystemExit as e:
        print(e)
    except Exception as e:
        print("❌ Excepción principal:", e)
    finally:
        # Failsafe
        try:
            print("🛬 Aterrizando (failsafe)…")
            for _ in range(30):
                send_attitude_target(thrust=THRUST_MIN)
                await asyncio.sleep(0.2)
            vehicle.mode = VehicleMode("STABILIZE")
            vehicle.armed = False
            vehicle.close()
        except Exception as e:
            print("⚠️ Failsafe con advertencia:", e)
        vuelo_activo = False
        try: kb_listener.stop()
        except: pass
        print("✅ Salida limpia")

if __name__ == "__main__":
    asyncio.run(main())
