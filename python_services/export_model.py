from ultralytics import YOLO
import shutil
import os

# 1. Obtener el modelo base
model = YOLO("yolov8n.pt") 

# 2. Exportar al formato ONNX compatible con OpenCV DNN
success = model.export(format="onnx")

# 3. Mover al directorio final
source = "yolov8n.onnx"
destination = "models/yolov8n.onnx"

if os.path.exists(source):
    # Asegúrate de que la carpeta models exista
    if not os.path.exists("models"):
        os.makedirs("models")
        
    shutil.move(source, destination)
    print(f"Modelo exportado exitosamente a: {destination}")
else:
    print("Error: Falló la exportación del modelo YOLOv8 a ONNX.")
