# UDC: Sistema Modular de Seguridad

Sistema de detección y reconocimiento de intrusos de alto rendimiento. Combina el poder de **C++** (YOLOv8 + Tracking) y **Python** (Reconocimiento Facial + Base de Datos) comunicados vía **ZeroMQ**.

## Arquitectura

1.  **C++ Core (Máximo FPS):** Carga video, ejecuta la inferencia YOLOv8 (ONNX) con OpenCV DNN, aplica seguimiento (Kalman/IoU) y envía las regiones de personas por ZMQ.
2.  **Python Service (Reconocimiento):** Recibe las imágenes, utiliza FaceNet para generar un embedding, consulta la base de datos (SQLite) y clasifica a la persona (Empleado, VIP, Problema).

## Requisitos e Instalación

### Requisitos de Sistema (C++)
- **CMake** (v3.10+)
- **OpenCV** (v4.5+ con soporte DNN)
- **ZeroMQ** (`libzmq3-dev` en Linux)
- **nlohmann/json** (header-only library, descargar `json.hpp`)

### Requisitos de Python (Servicio)
Se recomienda usar un entorno virtual.

```bash
pip install pyzmq numpy opencv-python torch torchvision scikit-learn facenet-pytorch




1. Preparar el Modelo ONNX
Ejecuta el script para obtener el modelo:
python export_model.py
2. Preparar la Base de Datos
La base de datos face_database.db se crea automáticamente al iniciar el servidor Python por primera vez. Se debe usar la función register_face dentro del script recognition_server.py para ingresar embeddings de personas conocidas.
3. Compilar C++
mkdir build && cd build
cmake ..
cmake --build .

4. Iniciar servidor python (terminal 1)
python python_services/recognition_server.py
5. Iniciar el Detector C++ (Terminal 2):
./UDC
