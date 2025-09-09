import cv2, os, time
import numpy as np

# ===== Calibración =====
CALIB_FILE = "parametros_calibrados.npz"
if not os.path.exists(CALIB_FILE):
    raise FileNotFoundError("Falta 'parametros_calibrados.npz' con claves 'K' y 'dist'.")
param = np.load(CALIB_FILE)
K, dist = param["K"], param["dist"]

# ===== Preset =====
# Preprocesado
GAMMA, CLAHE_CLIP, CLAHE_TILE = 0.9, 4.9, (9, 9)
UNSHARP_AMT, BLUR_KERNEL, BLUR_SIGMA = 0.68, 7, 0.7
BILATERAL_D, BILATERAL_SIG, USE_BILATERAL = 10, 35.0, True

# Detector
DICT = cv2.aruco.DICT_5X5_250
ATH_WIN_MIN, ATH_WIN_MAX, ATH_WIN_STEP, ATH_CONST = 3, 75, 1, 14
MIN_PERIM, POLY_ACC, USE_SUBPIX = 0.069, 0.030, True

# Marcador
tam_aruco_mm = 125.59
tam_aruco_m  = tam_aruco_mm / 1000.0

# ===== Utilidades =====
def apply_gamma(gray, gamma):
    inv = 1.0 / max(0.1, gamma)
    table = np.array([((i/255.0)**inv)*255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(gray, table)

def unsharp_mask(gray, amount=1.0, ksize=3, sigma=0.0):
    if amount <= 0.0: return gray
    blur = cv2.GaussianBlur(gray, (ksize, ksize), sigma)
    return cv2.addWeighted(gray, 1.0+amount, blur, -amount, 0)

def preprocesar(bgr):
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    g = apply_gamma(g, GAMMA)
    g = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_TILE).apply(g)
    g = unsharp_mask(g, amount=UNSHARP_AMT, ksize=BLUR_KERNEL, sigma=BLUR_SIGMA)
    g = cv2.GaussianBlur(g, (BLUR_KERNEL, BLUR_KERNEL), BLUR_SIGMA)
    if USE_BILATERAL:
        d = BILATERAL_D if BILATERAL_D > 0 else -1
        g = cv2.bilateralFilter(g, d, BILATERAL_SIG, BILATERAL_SIG)
    return g

def build_detector():
    try:
        aruco_dict = cv2.aruco.getPredefinedDictionary(DICT)
        pars = cv2.aruco.DetectorParameters()
        pars.adaptiveThreshWinSizeMin = ATH_WIN_MIN
        pars.adaptiveThreshWinSizeMax = ATH_WIN_MAX
        pars.adaptiveThreshWinSizeStep = ATH_WIN_STEP
        pars.adaptiveThreshConstant   = ATH_CONST
        pars.minMarkerPerimeterRate   = MIN_PERIM
        pars.maxMarkerPerimeterRate   = 4.0
        pars.polygonalApproxAccuracyRate = POLY_ACC
        pars.cornerRefinementMethod   = cv2.aruco.CORNER_REFINE_SUBPIX if USE_SUBPIX else cv2.aruco.CORNER_REFINE_NONE
        pars.cornerRefinementWinSize  = 5
        pars.cornerRefinementMaxIterations = 50
        pars.cornerRefinementMinAccuracy   = 0.01
        detector = cv2.aruco.ArucoDetector(aruco_dict, pars)
        def detect(gray): return detector.detectMarkers(gray)
        return aruco_dict, detect
    except AttributeError:
        aruco_dict = cv2.aruco.Dictionary_get(DICT)
        pars = cv2.aruco.DetectorParameters_create()
        pars.adaptiveThreshWinSizeMin = ATH_WIN_MIN
        pars.adaptiveThreshWinSizeMax = ATH_WIN_MAX
        pars.adaptiveThreshWinSizeStep = ATH_WIN_STEP
        pars.adaptiveThreshConstant   = ATH_CONST
        pars.minMarkerPerimeterRate   = MIN_PERIM
        pars.maxMarkerPerimeterRate   = 4.0
        pars.polygonalApproxAccuracyRate = POLY_ACC
        pars.cornerRefinementMethod   = cv2.aruco.CORNER_REFINE_SUBPIX if USE_SUBPIX else cv2.aruco.CORNER_REFINE_NONE
        pars.cornerRefinementWinSize  = 5
        pars.cornerRefinementMaxIterations = 50
        pars.cornerRefinementMinAccuracy   = 0.01
        def detect(gray): return cv2.aruco.detectMarkers(gray, aruco_dict, parameters=pars)
        return aruco_dict, detect

_, DETECT = build_detector()

# ===== Captura y loop (solo "Overlay") =====
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)   # cambia índice si es necesario
if not cap.isOpened():
    raise RuntimeError("No se pudo abrir la cámara (índice 1).")

print("Presiona 'q' para salir.")
t_prev, fps = time.time(), 0.0

while True:
    ok, frame = cap.read()
    if not ok: break

    gray = preprocesar(frame)
    corners, ids, _ = DETECT(gray)

    out = frame.copy()
    if ids is not None and len(ids) > 0:
        cv2.aruco.drawDetectedMarkers(out, corners, ids)
        pose = cv2.aruco.estimatePoseSingleMarkers(corners, tam_aruco_m, K, dist)
        rvecs, tvecs = (pose[:2]) if len(pose) == 3 else pose
        for i in range(len(ids)):
            try:
                cv2.aruco.drawAxis(out, K, dist, rvecs[i], tvecs[i], tam_aruco_m/2.0)
            except AttributeError:
                cv2.drawFrameAxes(out, K, dist, rvecs[i], tvecs[i], tam_aruco_m/2.0)
            z = tvecs[i][0][2]
            cv2.putText(out, f"ID {int(ids[i][0])}  z:{z:.2f} m",
                        (10, 30 + 30*(i % 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    else:
        cv2.putText(out, "No se detecta ArUco", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

    # HUD mínimo con FPS y preset
    t = time.time(); dt = t - t_prev; t_prev = t
    fps = 0.9*fps + 0.1*((1.0/dt) if dt > 0 else 0.0)
    cv2.putText(out, f"FPS:{fps:4.1f}  [g:{GAMMA} clip:{CLAHE_CLIP} un:{UNSHARP_AMT} k:{BLUR_KERNEL} s:{BLUR_SIGMA}]",
                (10, out.shape[0]-12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

    cv2.imshow("Overlay", out)
    if (cv2.waitKey(1) & 0xFF) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
