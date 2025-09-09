# Navegación Autónoma Colaborativa UAV–UGV (Visión + Deep Learning)

Implementación reproducible del sistema de cooperación **UAV–UGV** basado exclusivamente en **visión artificial** y **aprendizaje profundo**.  
El **UGV** sigue un carril con una **ResNet-18** que predice el **PWM de dirección**; el **UAV** se centra sobre el UGV mediante **ArUco** y asiste con **detección de obstáculos (YOLOv8n)**.  
La orquestación es **off-board**, con comunicación de baja latencia por Wi-Fi.  

Este repositorio contiene los códigos desarrollados durante la tesis *“Navegación Autónoma Colaborativa UAV–UGV Mediante Visión Artificial”*.

---

## 📂 Mapa del repositorio

| Archivo | Función principal |
|---------|------------------|
| `Calibracion_Camara.py` | Calibración intrínseca de cámara con tablero (chessboard/charuco). Estima matriz K y coeficientes de distorsión. |
| `Captura_Img_Tablero.py` | Captura imágenes del tablero para el dataset de calibración. |
| `Deteccion_Aruco.py` | Detección y estimación de pose de marcadores ArUco. Sirve para el centrado del UAV sobre el UGV. |
| `EntrenamientoResNet18.py` | Entrenamiento de ResNet-18 para predecir el PWM de dirección del UGV a partir de imágenes. |
| `Yolov8n.py` | Entrenamiento y validación de YOLOv8n para detección de obstáculos y del UGV. |
| `UGV.py` | Inferencia en el vehículo terrestre: abre cámara, carga modelo ResNet-18 y publica comandos de dirección. |
| `UAV.py` | Inferencia en el dron: detección ArUco y YOLOv8n para centrado y advertencias. |
| `CodigoUAV_UGV.py` | Script de integración: coordina UAV y UGV, aplica política de freno seguro y registra métricas. |
| `requirements.txt` | Dependencias necesarias para ejecutar los códigos. |

---

## ⚙️ Requisitos

- Python 3.8+
- OpenCV (contrib), NumPy, Pandas, Matplotlib
- PyTorch y Torchvision
- Ultralytics YOLOv8
- (Opcional) Roboflow para gestión de datasets

---

## 🚀 Instalación

```bash
# Clonar repositorio
git clone https://github.com/JonatanM28/Tesis_UAV_UGV.git
cd Tesis_UAV_UGV

# Crear entorno virtual (opcional)
python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
