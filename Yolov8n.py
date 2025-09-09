from ultralytics import YOLO
from roboflow import Roboflow

# === 1) Descarga dataset desde Roboflow (rellena tus datos) ===
RF_API_KEY = "TU_API_KEY"        # <- pon tu API key
WORKSPACE  = "tu-workspace"
PROJECT    = "tu-proyecto"
VERSION    = 1

rf = Roboflow(api_key=RF_API_KEY)
dataset = rf.workspace(WORKSPACE).project(PROJECT).version(VERSION).download("yolov8")  # genera carpeta con .yaml

# === 2) Entrena YOLOv8n con los hiperparámetros de tu tabla ===
model = YOLO("yolov8n.pt")
model.train(
    data=dataset.location + "/data.yaml",  # ruta del YAML generado por Roboflow
    imgsz=640,
    epochs=100,
    batch=16,
    optimizer="SGD",
    lr0=0.001,
    lrf=0.01,            # lr_final = lr0 * lrf
    warmup_epochs=3,
    momentum=0.937,
    weight_decay=0.0005,
    patience=10,         # early stopping
    device=0,            # usa 0 o 'cpu'
    project="runs_yolo",
    name="yolov8n_tesis"
)

# (opcional) Validación rápida
model.val(data=dataset.location + "/data.yaml", imgsz=640, device=0)
