import cv2, glob, os
import numpy as np

CHECKERBOARD = (8, 5)        # (cols, rows) de esquinas internas
SQUARE_SIZE  = 27.59           # tamaño real de cada cuadro
PATTERNS = ["calib_raw/raw_8x5_*.png", "calib_raw/raw_8x5_*.jpg"]

cols, rows = CHECKERBOARD
objp = np.zeros((cols*rows,3), np.float32)
objp[:,:2] = np.mgrid[0:cols,0:rows].T.reshape(-1,2)
objp *= SQUARE_SIZE

objpoints, imgpoints, used = [], [], []
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
HAS_SB = hasattr(cv2, "findChessboardCornersSB")

files = []
for p in PATTERNS: files += glob.glob(p)
files = sorted(files)
if not files: raise FileNotFoundError("No hay imágenes en calib_raw/ con el patrón indicado.")

print(f"Detectando {cols}x{rows} en {len(files)} imágenes…")
for f in files:
    img = cv2.imread(f, cv2.IMREAD_COLOR);  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = False, None
    if HAS_SB:
        try:
            ret, corners = cv2.findChessboardCornersSB(gray, CHECKERBOARD, flags=cv2.CALIB_CB_NORMALIZE_IMAGE)
        except: ret = False
    if not ret:
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, flags=flags)
        if ret:
            corners = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
    if ret and corners is not None and len(corners) == cols*rows:
        objpoints.append(objp.copy()); imgpoints.append(corners); used.append(f)
        print(f"✔ {f}")
    else:
        print(f"✖ {f}")

if len(objpoints) < 12:
    raise RuntimeError(f"Solo {len(objpoints)} imágenes válidas; toma ≥12–20 para buena calibración.")

h, w = cv2.imread(used[0]).shape[:2]
print(f"\nCalibrando con {len(used)} imágenes (tamaño {w}x{h})…")
rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (w, h), None, None)

print("\n=== Resultados ===")
print("RMS global (px):", rms)
print("K:\n", K)
print("dist:\n", dist.ravel())

errs = []
for objp_i, imgp_i, rv, tv in zip(objpoints, imgpoints, rvecs, tvecs):
    proj,_ = cv2.projectPoints(objp_i, rv, tv, K, dist)
    err = cv2.norm(imgp_i, proj, cv2.NORM_L2)/len(proj)
    errs.append(err)
print("Error medio por imagen (px):", float(np.mean(errs)))

np.savez("parametros_calibrados.npz", K=K, dist=dist, rms=rms, per_image_err=np.array(errs), img_size=(w,h))
print("✅ Guardado: parametros_calibrados.npz")

und = cv2.undistort(cv2.imread(used[0]), K, dist)
cv2.imwrite("undistorted_preview.png", und)
print("✅ Guardado: undistorted_preview.png")
