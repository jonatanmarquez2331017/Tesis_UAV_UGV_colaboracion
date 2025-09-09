import cv2, os
import numpy as np

CHECKERBOARD = (8, 5)           # 8x5 esquinas internas
SAVE_DIR = "calib_raw"          # carpeta destino
CAM_INDEX = 1                   # índice de tu cámara (ajusta)

os.makedirs(SAVE_DIR, exist_ok=True)

def laplacian_variance(gray):
    return cv2.Laplacian(gray, cv2.CV_64F).var()

cap = cv2.VideoCapture(CAM_INDEX)
if not cap.isOpened():
    raise RuntimeError("No se pudo abrir la cámara")

print("Vista en vivo: 's' guarda CRUDO cuando hay detección, 'q' sale.")
counter = 0
HAS_SB = hasattr(cv2, "findChessboardCornersSB")
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)

while True:
    ok, frame = cap.read()
    if not ok:
        continue
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_raw = frame.copy()
    ret, corners = False, None

    if HAS_SB:
        try:
            ret, corners = cv2.findChessboardCornersSB(
                gray, CHECKERBOARD, flags=cv2.CALIB_CB_NORMALIZE_IMAGE
            )
        except Exception:
            ret = False

    if not ret:
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, flags=flags)
        if ret:
            corners = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)

    disp = frame.copy()
    sharp = laplacian_variance(gray)
    color = (0,255,0) if ret else (0,0,255)
    msg = f"{'✔ Detectado' if ret else '✖ No detectado'}  {CHECKERBOARD[0]}x{CHECKERBOARD[1]}  nitidez={sharp:.0f}"
    cv2.putText(disp, msg, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
    if ret:
        cv2.drawChessboardCorners(disp, CHECKERBOARD, corners, ret)

    cv2.imshow("Checkerboard (preview)", disp)
    k = cv2.waitKey(1) & 0xFF

    if k == ord('s') and ret:
        out = os.path.join(SAVE_DIR, f"raw_{CHECKERBOARD[0]}x{CHECKERBOARD[1]}_{counter:03d}.png")
        cv2.imwrite(out, frame_raw)
        print(f"💾 Guardada cruda: {out}")
        counter += 1
    elif k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
