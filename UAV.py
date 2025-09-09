import asyncio, time, math
import cv2, numpy as np
from collections import deque
import matplotlib.pyplot as plt
import collections, collections.abc
if not hasattr(collections, "MutableMapping"):
    collections.MutableMapping = collections.abc.MutableMapping
if not hasattr(collections, "Mapping"):
    collections.Mapping = collections.abc.Mapping
if not hasattr(collections, "Sequence"):
    collections.Sequence = collections.abc.Sequence
from dronekit import connect, VehicleMode

# ========= ÍNDICE/RESOLUCIÓN DE CÁMARA (FIJOS) =========
Indice_cam = 1
CAM_WIDTH  = 640
CAM_HEIGHT = 480
CAM_FPS    = 30

# ========= CONFIGURACIÓN GLOBAL =========
ALTURA_DESEADA     = 1.1      # m
TOLERANCIA_POS     = 0.02      # m
THRUST_BASE        = 0.495
THRUST_MIN         = 0.30
THRUST_MAX         = 0.60
THRUST_HOVER_GUARD = 0.52

Kp_z, Ki_z, Kd_z = 0.28, 0.001, 0.205
Kp_x, Ki_x, Kd_x = 9.0, 0.19, 5.0
Kp_y, Ki_y, Kd_y = 9.0, 0.19, 5.0

# Fusión de sensores / validaciones
FUSION_UMBRAL_M    = 1.20
LASER_MIN, LASER_MAX   = 0.15, 3.0
LASER_MAX_JUMP         = 0.35
ARUCO_MAX_DIST         = 3.0   # m
ARUCO_LOSS_HOLD        = 0.35  # s
Z_RATE_LIMIT           = 1.2   # m/s
Z_ALPHA                = 0.35  # EMA

# ========= CALIBRACIÓN =========
CALIB_FILE = "parametros_calibrados.npz"
param = np.load(CALIB_FILE, allow_pickle=True)
K_base = param["K"].copy()
dist   = param["dist"].copy()

if "img_w" in param and "img_h" in param:
    calib_w, calib_h = int(param["img_w"]), int(param["img_h"])
elif "img_size" in param:
    size = param["img_size"]
    calib_w, calib_h = int(size[0]), int(size[1])
else:
    calib_w = calib_h = None

# ========= ARUCO NUEVA LÓGICA =========
ARUCO_DICT     = cv2.aruco.DICT_5X5_250   # IDs de 0..249
MARKER_LENGTH  = 0.12559                  # lado del ArUco en metros
ID_CENTRADO    = 105                      # ID para centrar X-Y (ajústalo si quieres)
IDS_ALTURA_MIN = 101
IDS_ALTURA_MAX = 108

# ========= ESTADO GLOBAL =========
pitch_objetivo = 0.0
roll_objetivo  = 0.0
thrust_objetivo = THRUST_BASE
x_actual = 0.0
y_actual = 0.0
altura_estimada = 0.0
vuelo_activo = True

# Buffers / logs
valores_altura = deque(maxlen=10)
log_altura, log_altura_tiempo = [], []
log_x, log_y = [], []
log_deteccion = []
log_pitch, log_roll, log_pitch_tiempo, log_roll_tiempo = [], [], [], []

# ========= CONEXIÓN ROBUSTA =========
def connect_drone(uri='udp:0.0.0.0:14550', hb_wait_s=12.0):
    print(f"⏳ Conectando a {uri} (wait_ready=False)...")
    v = connect(uri, wait_ready=False)
    t0 = time.time()
    while time.time() - t0 < hb_wait_s:
        if getattr(v, 'last_heartbeat', None) is not None:
            print(f"✅ Conectado a {uri}")
            return v
        time.sleep(0.2)
    try: v.close()
    except: pass
    raise TimeoutError(f"Sin heartbeat en {uri} tras {hb_wait_s} s")

# ========= MAVLink helpers (robustos) =========
def _yaw_deg_safe(vehicle, yaw_angle):
    import numpy as _np
    if yaw_angle is None:
        try:
            y = getattr(vehicle.attitude, "yaw", None)  # radianes
            if y is None or not _np.isfinite(y):
                return 0.0
            return math.degrees(float(y))               # a grados
        except Exception:
            return 0.0
    try:
        y = float(yaw_angle)  # asumimos que el caller pasa grados
        return y if _np.isfinite(y) else 0.0
    except Exception:
        return 0.0

def to_quaternion(roll_deg=0.0, pitch_deg=0.0, yaw_deg=0.0):
    """roll/pitch/yaw en GRADOS → [w, x, y, z]"""
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
    """roll/pitch en GRADOS; yaw en GRADOS (None → se toma del vehículo, rad→deg)."""
    try: r = float(roll_angle)
    except: r = 0.0
    try: p = float(pitch_angle)
    except: p = 0.0
    try: u = float(thrust)
    except: u = 0.5

    y_deg = _yaw_deg_safe(vehicle, yaw_angle)
    q = to_quaternion(r, p, y_deg)

    msg = vehicle.message_factory.set_attitude_target_encode(
        0, 1, 1,
        0b00000111,   # ignorar body rates
        q,
        0, 0, 0,
        u
    )
    vehicle.send_mavlink(msg)
    vehicle.flush()

# ========= UTILIDADES Z =========
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

# ========= VISIÓN Y ALTURA (NUEVA LÓGICA ARUCO) =========
async def vision_loop():
    global x_actual, y_actual, altura_estimada, vuelo_activo, K

    # Cámara (sin redimensionar, fijamos 640x480)
    cap = cv2.VideoCapture(Indice_cam, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(Indice_cam)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ No se pudo abrir la cámara."); vuelo_activo = False; return

    # Forzamos resolución/params
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS,          CAM_FPS)
    try: cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    except: pass
    try: cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
    except: pass

    # Leer un frame para saber W,H finales (sin redimensionar)
    ret, frame = cap.read()
    if not ret:
        print("❌ Cámara sin frames."); vuelo_activo = False; return
    h, w = frame.shape[:2]

    # Ajuste de K si la calibración fue en otra resolución
    if calib_w and calib_h and (w != calib_w or h != calib_h):
        sx, sy = w / float(calib_w), h / float(calib_h)
        K = K_base.copy()
        K[0,0] *= sx; K[0,2] *= sx
        K[1,1] *= sy; K[1,2] *= sy
    else:
        K = K_base.copy()

    # ArUco (tal cual tu verificar_aruco_pose.py)
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    try:
        params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(dictionary, params)
        use_new = True
    except AttributeError:
        # Fallback OpenCV viejas
        params = cv2.aruco.DetectorParameters_create()
        use_new = False

    last_aruco_t = 0.0
    last_aruco_z = None
    last_laser_z = None

    while vuelo_activo:
        ret, frame = cap.read()
        if not ret:
            await asyncio.sleep(0.01);
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detección simple (sin preproc agresivo)
        if use_new:
            corners, ids, _ = detector.detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary, parameters=params)

        out = frame.copy()
        detectado_centro = False
        alturas_arucos = []

        if ids is not None and len(ids) > 0:
            try: cv2.aruco.drawDetectedMarkers(out, corners, ids)
            except: pass

            # Pose de todos los marcadores
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, MARKER_LENGTH, K, dist)

            for i in range(len(ids)):
                mid = int(ids[i][0])
                tvec = tvecs[i][0]
                rvec = rvecs[i]

                # Ejes (visual)
                try:
                    cv2.drawFrameAxes(out, K, dist, rvec, tvec, MARKER_LENGTH*0.5)
                except:
                    try: cv2.aruco.drawAxis(out, K, dist, rvec, tvec, MARKER_LENGTH*0.5)
                    except: pass

                # Selección para control XY (ID_CENTRADO)
                if mid == ID_CENTRADO:
                    detectado_centro = True
                    # Convención: tvec = [x, y, z] en metros (sistema de cámara)
                    x_actual = float(tvec[0])
                    y_actual = float(tvec[1])

                # Selección para altura (IDs dentro de rango)
                if IDS_ALTURA_MIN <= mid <= IDS_ALTURA_MAX:
                    z = float(tvec[2])
                    if z < ARUCO_MAX_DIST:
                        alturas_arucos.append(z)

            if detectado_centro:
                # Busca índice del ID_CENTRADO para calcular Euler y distancia
                idx = None
                for i in range(len(ids)):
                    if int(ids[i][0]) == ID_CENTRADO:
                        idx = i; break
                if idx is not None:
                    R,_ = cv2.Rodrigues(rvecs[idx])
                    sy = np.sqrt(R[0,0]**2 + R[1,0]**2)
                    yaw   = np.degrees(np.arctan2(R[1,0], R[0,0]))
                    pitch = np.degrees(np.arctan2(-R[2,0], sy))
                    roll  = np.degrees(np.arctan2(R[2,1], R[2,2]))
                    dist_m = float(np.linalg.norm(tvecs[idx][0]))
                    txt = f"dist={dist_m:.2f} m  yaw={yaw:.1f}  pitch={pitch:.1f}  roll={roll:.1f}"
                    cv2.putText(out, txt, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        else:
            cv2.putText(out, "No se detecta ArUco", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        # Lectura láser
        z_laser = float(getattr(vehicle.rangefinder, "distance", 0.0) or 0.0)

        # Fusión robusta de altura
        now = time.time()
        if alturas_arucos:
            z_aruco = float(np.median(alturas_arucos))
            last_aruco_t = now
            last_aruco_z = z_aruco

            if z_aruco < FUSION_UMBRAL_M:
                if laser_is_valid(z_laser, last_laser_z):
                    z_meas = 0.5*z_aruco + 0.5*z_laser
                else:
                    z_meas = z_aruco
            else:
                z_meas = z_aruco
        else:
            if (now - last_aruco_t) <= ARUCO_LOSS_HOLD and last_aruco_z is not None:
                z_meas = last_aruco_z
            else:
                if laser_is_valid(z_laser, last_laser_z):
                    z_meas = z_laser
                else:
                    z_meas = altura_estimada

        last_laser_z = z_laser if z_laser > 0 else last_laser_z

        # Rate-limit + EMA
        altura_estimada = z_filter.update(z_meas)
        valores_altura.append(altura_estimada)

        # Logs
        t = time.time() - t_inicio
        log_altura.append(altura_estimada); log_altura_tiempo.append(t)
        log_x.append(x_actual);            log_y.append(y_actual)
        log_deteccion.append(1 if detectado_centro else 0)

        # Overlay final
        cv2.putText(out, f"z_est: {altura_estimada:.2f} m", (10, h-20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.imshow("Camara + ArUco (640x480, sin resize)", out)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        await asyncio.sleep(0.01)

    cap.release()
    cv2.destroyAllWindows()

# ========= CONTROL DE ALTURA =========
async def control_altura():
    global thrust_objetivo
    integral = 0.0
    prev_error = 0.0
    while vuelo_activo:
        error = ALTURA_DESEADA - altura_estimada
        derivada = (error - prev_error) / 0.05

        u = THRUST_BASE + Kp_z*error + Kd_z*derivada + Ki_z*integral
        if THRUST_MIN < u < THRUST_MAX:
            integral += error * 0.05
        else:
            integral *= 0.9  # anti-windup (bleed-off)

        u = max(min(u, min(THRUST_MAX, THRUST_HOVER_GUARD)), THRUST_MIN)
        thrust_objetivo = float(u)

        prev_error = error
        await asyncio.sleep(0.05)

# ========= PID ROLL (X) y PITCH (Y) =========
async def control_roll():
    global roll_objetivo
    integral = 0.0; prev_error = 0.0
    while vuelo_activo:
        error = x_actual
        derivada = (error - prev_error) / 0.05
        u = Kp_x*error + Kd_x*derivada + Ki_x*integral
        if -5 < u < 5: integral += error * 0.05
        roll_objetivo = 0.0 if abs(error) < TOLERANCIA_POS else float(np.clip(u, -5, 5))
        prev_error = error
        await asyncio.sleep(0.05)

async def control_pitch():
    global pitch_objetivo
    integral = 0.0; prev_error = 0.0
    while vuelo_activo:
        error = y_actual
        derivada = (error - prev_error) / 0.05
        u = Kp_y*error + Kd_y*derivada + Ki_y*integral
        if -5 < u < 5: integral += error * 0.05
        pitch_objetivo = 0.0 if abs(error) < TOLERANCIA_POS else float(np.clip(u, -5, 5))
        prev_error = error
        await asyncio.sleep(0.05)

# ========= LOOP DE ACTITUD =========
async def actitud_loop():
    while vuelo_activo:
        send_attitude_target(roll_angle=roll_objetivo,
                             pitch_angle=pitch_objetivo,
                             yaw_angle=None,
                             thrust=thrust_objetivo)
        t = time.time() - t_inicio
        log_roll.append(roll_objetivo);   log_pitch.append(pitch_objetivo)
        log_roll_tiempo.append(t);        log_pitch_tiempo.append(t)
        await asyncio.sleep(0.05)

# ========= MAIN =========
async def main():
    await asyncio.gather(
        vision_loop(),
        control_altura(),
        control_roll(),
        control_pitch(),
        actitud_loop()
    )

# ========= EJECUCIÓN =========
t_inicio = time.time()
vehicle = None
try:
    connection_string = 'udp:0.0.0.0:14550'
    vehicle = connect_drone(connection_string, hb_wait_s=12.0)

    try: vehicle.parameters['ARMING_CHECK'] = 0
    except: pass
    vehicle.mode = VehicleMode("GUIDED_NOGPS")
    time.sleep(2)

    if not vehicle.armed:
        vehicle.armed = True
        t0 = time.time()
        while not vehicle.armed:
            if time.time() - t0 > 12.0:
                raise RuntimeError("No se pudo armar en 12 s")
            time.sleep(0.2)
    print("✅ Dron armado")

    asyncio.get_event_loop().run_until_complete(asyncio.wait_for(main(), timeout=120))

except asyncio.TimeoutError:
    print("🛬 Fin de prueba (timeout), iniciando aterrizaje...")
finally:
    vuelo_activo = False
    try:
        for _ in range(30):
            send_attitude_target(roll_angle=0.0, pitch_angle=0.0, yaw_angle=0.0, thrust=THRUST_MIN)
            time.sleep(0.2)
        vehicle.mode = VehicleMode("STABILIZE")
        vehicle.armed = False
        vehicle.close()
    except Exception as e:
        print("⚠️ Failsafe con advertencia:", e)

# ========= GRAFICADO =========
plt.figure("Altura y posición")
plt.subplot(3, 1, 1)
plt.plot(log_altura_tiempo, log_altura, color='purple', label="Altura estimada")
plt.axhline(ALTURA_DESEADA, color='gray', linestyle='--', label="Altura deseada")
plt.legend(); plt.grid()

plt.subplot(3, 1, 2)
plt.plot(log_altura_tiempo, log_x, label="X", color='red')
plt.plot(log_altura_tiempo, log_y, label="Y", color='orange')
plt.axhline(0, color='gray', linestyle='--')
plt.legend(); plt.grid()

plt.subplot(3, 1, 3)
plt.plot(log_altura_tiempo, log_deteccion, label=f"Aruco {ID_CENTRADO}", color='black')
plt.ylim(-0.1, 1.1); plt.grid(); plt.legend(); plt.xlabel("Tiempo (s)")
plt.tight_layout(); plt.show()

plt.figure("Control")
plt.subplot(2, 1, 1)
plt.plot(log_roll_tiempo, log_roll, color='green', label="Roll (X)")
plt.legend(); plt.grid()
plt.subplot(2, 1, 2)
plt.plot(log_pitch_tiempo, log_pitch, color='blue', label="Pitch (Y)")
plt.legend(); plt.grid(); plt.tight_layout(); plt.show()
