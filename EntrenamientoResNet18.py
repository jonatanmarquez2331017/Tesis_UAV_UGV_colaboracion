
import os, random, math, csv, time
from datetime import datetime
import argparse
import json
import numpy as np
import pandas as pd
from PIL import Image
import cv2

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from tqdm import tqdm

# =========================
# Configuración por defecto
# =========================
CSV_PATH    = r"D:\Datasets\Carretera\labels_pwm.csv"   # <-- EDITA
IMAGES_DIR  = r"D:\Datasets\Carretera\images"           # <-- EDITA
OUT_DIR     = r".\salidas_regresion"
BATCH_SIZE  = 64
EPOCHS      = 50
LR          = 1e-4
WEIGHT_DECAY= 1e-4
NUM_WORKERS = 4
SEED        = 42
TEST_PCT    = 0.10
VAL_PCT     = 0.10   # del total; train = 1 - TEST_PCT - VAL_PCT
PIN_MEMORY  = True if torch.cuda.is_available() else False
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

# =================
# Utilidades
# =================
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def r2_score(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1.0 - ss_res / (ss_tot + 1e-12)

def compute_metrics(targets, preds):
    targets = np.asarray(targets, dtype=np.float64)
    preds   = np.asarray(preds,   dtype=np.float64)
    diff    = preds - targets
    mae     = np.mean(np.abs(diff))
    rmse    = math.sqrt(np.mean(diff**2))
    r2      = r2_score(targets, preds)
    maxerr  = np.max(np.abs(diff))
    within20 = np.mean(np.abs(diff) <= 20.0) * 100.0
    within10 = np.mean(np.abs(diff) <= 10.0) * 100.0
    return dict(MAE=mae, RMSE=rmse, R2=r2, MaxErr=maxerr, Pct_<=20=within20, Pct_<=10=within10)

# =================
# Dataset
# =================
class PWMDataset(Dataset):
    """
    CSV esperado con columnas: filename,pwm
    Las imágenes están en IMAGES_DIR y ya son 224x224 (no se recorta ni reescala).
    """
    def __init__(self, df, images_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.images_dir = images_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def _load_image(self, path):
        # Carga con PIL en RGB para usar transform.normalize de torchvision
        img = Image.open(path).convert("RGB")
        # No redimensionamos: asumimos 224x224 ya en disco
        return img

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.images_dir, row["filename"])
        img = self._load_image(img_path)
        pwm = float(row["pwm"])
        if self.transform:
            img = self.transform(img)
        pwm = torch.tensor([pwm], dtype=torch.float32)
        return img, pwm, row["filename"]

# =================
# Modelo
# =================
class ResNet18Regressor(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, 1)  # Salida escalar (PWM)
    def forward(self, x):
        return self.backbone(x)

# =================
# Early Stopping
# =================
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best = None
        self.counter = 0
        self.should_stop = False

    def step(self, metric):
        if self.best is None or (self.best - metric) > self.min_delta:
            self.best = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

# =================
# Entrenamiento
# =================
def split_dataframe(df, test_pct=0.10, val_pct=0.10, seed=SEED):
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n = len(df)
    n_test = int(round(n * test_pct))
    n_val  = int(round(n * val_pct))
    test_df = df.iloc[:n_test]
    val_df  = df.iloc[n_test:n_test+n_val]
    train_df= df.iloc[n_test+n_val:]
    return train_df, val_df, test_df

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    losses, gts, prs = [], [], []
    pbar = tqdm(loader, desc="Entrenando", leave=False)
    for xb, yb, _ in pbar:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        gts.extend(yb.detach().cpu().numpy().flatten())
        prs.extend(pred.detach().cpu().numpy().flatten())
        pbar.set_postfix(loss=np.mean(losses))
    metrics = compute_metrics(gts, prs)
    return float(np.mean(losses)), metrics

@torch.no_grad()
def eval_one_epoch(model, loader, criterion, device, desc="Val"):
    model.eval()
    losses, gts, prs = [], [], []
    pbar = tqdm(loader, desc=desc, leave=False)
    for xb, yb, _ in pbar:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        pred = model(xb)
        loss = criterion(pred, yb)
        losses.append(loss.item())
        gts.extend(yb.detach().cpu().numpy().flatten())
        prs.extend(pred.detach().cpu().numpy().flatten())
        pbar.set_postfix(loss=np.mean(losses))
    metrics = compute_metrics(gts, prs)
    return float(np.mean(losses)), metrics, (gts, prs)

def annotate_and_save(img_path, out_path, real_pwm, pred_pwm):
    img = cv2.imread(img_path)
    if img is None:
        return
    diff = abs(pred_pwm - real_pwm)
    text = f"real {int(real_pwm)}, pred {int(pred_pwm)}, diff {int(diff)}"
    # Caja semitransparente
    overlay = img.copy()
    cv2.rectangle(overlay, (5, 5), (5+420, 5+38), (0,0,0), -1)
    img = cv2.addWeighted(overlay, 0.5, img, 0.5, 0)
    # Texto
    cv2.putText(img, text, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
    cv2.imwrite(out_path, img)

def main():
    set_seed(SEED)
    ensure_dir(OUT_DIR)
    with open(os.path.join(OUT_DIR, "config.json"), "w") as f:
        json.dump({
            "CSV_PATH": CSV_PATH, "IMAGES_DIR": IMAGES_DIR, "OUT_DIR": OUT_DIR,
            "BATCH_SIZE": BATCH_SIZE, "EPOCHS": EPOCHS, "LR": LR,
            "WEIGHT_DECAY": WEIGHT_DECAY, "SEED": SEED, "TEST_PCT": TEST_PCT, "VAL_PCT": VAL_PCT
        }, f, indent=2)

    # ====== Carga CSV ======
    df = pd.read_csv(CSV_PATH)
    assert {"filename","pwm"}.issubset(df.columns), "CSV debe tener columnas: filename,pwm"
    # Filtra los que existan
    df["exists"] = df["filename"].apply(lambda x: os.path.exists(os.path.join(IMAGES_DIR, x)))
    df = df[df["exists"]].drop(columns=["exists"]).reset_index(drop=True)
    print(f"Registros válidos: {len(df)}")

    train_df, val_df, test_df = split_dataframe(df, TEST_PCT, VAL_PCT, SEED)
    print(f"Split -> train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")

    # ====== Transforms (sin resize; solo normalización ImageNet) ======
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],
                             std =[0.229,0.224,0.225])
    ])

    # ====== Datasets y Loaders ======
    train_set = PWMDataset(train_df, IMAGES_DIR, transform)
    val_set   = PWMDataset(val_df,   IMAGES_DIR, transform)
    test_set  = PWMDataset(test_df,  IMAGES_DIR, transform)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, drop_last=False)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    test_loader  = DataLoader(test_set,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    # ====== Modelo, criterio, optimizador, scheduler ======
    model = ResNet18Regressor(pretrained=True).to(DEVICE)
    criterion = nn.SmoothL1Loss(beta=1.0)  # Huber
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                                           patience=3, verbose=True)

    early = EarlyStopping(patience=10, min_delta=0.0)

    # ====== Loop de entrenamiento ======
    history = []
    best_val_mae = float("inf")
    best_path = os.path.join(OUT_DIR, "model_best.pth")

    for epoch in range(1, EPOCHS+1):
        print(f"\n===== Época {epoch}/{EPOCHS} =====")
        t0 = time.time()
        tr_loss, tr_metrics = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        va_loss, va_metrics, _ = eval_one_epoch(model, val_loader, criterion, DEVICE, desc="Validando")

        scheduler.step(va_metrics["MAE"])

        row = {
            "epoch": epoch,
            "train_loss": tr_loss, **{f"train_{k}": v for k,v in tr_metrics.items()},
            "val_loss": va_loss,   **{f"val_{k}": v   for k,v in va_metrics.items()},
            "lr": optimizer.param_groups[0]["lr"],
            "time_sec": time.time() - t0
        }
        history.append(row)
        pd.DataFrame(history).to_csv(os.path.join(OUT_DIR, "history.csv"), index=False)

        print(f"MAE train: {tr_metrics['MAE']:.2f} | MAE val: {va_metrics['MAE']:.2f} | R2 val: {va_metrics['R2']:.4f}")

        # Guardar mejor por MAE de validación
        if va_metrics["MAE"] < best_val_mae:
            best_val_mae = va_metrics["MAE"]
            torch.save({
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "best_val_mae": best_val_mae,
                "config": {
                    "mean": [0.485,0.456,0.406],
                    "std":  [0.229,0.224,0.225],
                    "arch": "resnet18",
                }
            }, best_path)
            print(f"✅ Nuevo mejor modelo guardado: {best_path} (MAE val={best_val_mae:.2f})")

        # Early stopping
        early.step(va_metrics["MAE"])
        if early.should_stop:
            print("⛔ Early stopping activado.")
            break

    # ====== Evaluación en TEST con el mejor modelo ======
    print("\n====== Evaluando en TEST ======")
    ckpt = torch.load(best_path, map_location=DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    te_loss, te_metrics, (gts, prs) = eval_one_epoch(model, test_loader, criterion, DEVICE, desc="Test")
    print(f"TEST -> MAE: {te_metrics['MAE']:.2f}, RMSE: {te_metrics['RMSE']:.2f}, R2: {te_metrics['R2']:.4f}, MaxErr: {te_metrics['MaxErr']:.1f}, ≤20PWM: {te_metrics['Pct_<=20']:.1f}%")

    # Guardar CSV de test
    test_pred_csv = os.path.join(OUT_DIR, "test_predictions.csv")
    all_fns = []
    for _, _, f in test_loader:
        all_fns.extend(list(f))
    pd.DataFrame({"filename": all_fns, "pwm_true": gts, "pwm_pred": prs, "diff": np.asarray(prs)-np.asarray(gts)}).to_csv(test_pred_csv, index=False)
    print(f"Predicciones de test guardadas en {test_pred_csv}")

    # Guardar imágenes anotadas
    anno_dir = os.path.join(OUT_DIR, "test_results_images")
    ensure_dir(anno_dir)

    # Para anotar necesitamos iterar de nuevo sobre el set de test 1 a 1
    single_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0)
    preds_list = []
    model.eval()
    with torch.no_grad():
        for (xb, yb, fn) in tqdm(single_loader, desc="Anotando imágenes de TEST"):
            xb = xb.to(DEVICE)
            pred = model(xb).cpu().numpy().flatten()[0]
            real = float(yb.numpy().flatten()[0])
            in_path  = os.path.join(IMAGES_DIR, fn[0])
            out_path = os.path.join(anno_dir, f"pred_{int(round(pred))}_{fn[0]}")
            annotate_and_save(in_path, out_path, real, pred)

    print(f"Imágenes anotadas en: {anno_dir}")
    print("✅ Listo. Usa model_best.pth para inferencia en tu script de envío por WiFi.")

if __name__ == "__main__":
    main()
