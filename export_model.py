from ultralytics import YOLO
import shutil
import os

# Obtener la ruta correcta al directorio de modelos
MODELS_DIR = "models"
if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

# 1. Obtener el modelo base (YOLOv8 nano)
model_name = "yolov8n"
model = YOLO(f"{model_name}.pt") 

# 2. Exportar al formato ONNX compatible con OpenCV DNN
success = model.export(format="onnx")

# 3. Mover al directorio final
source = f"{model_name}.onnx"
destination = os.path.join(MODELS_DIR, f"{model_name}.onnx")

if os.path.exists(source):
    shutil.move(source, destination)
    print(f"Modelo exportado exitosamente a: {destination}")
else:
    print(f"Error: Falló la exportación del modelo YOLOv8 a ONNX. Archivo {source} no encontrado.")
