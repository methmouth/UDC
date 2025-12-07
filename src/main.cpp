#include "../include/Yolov8Detector.h"
#include "../include/KalmanTracker.h"
#include "../include/ZmqPublisher.h"
#include <iostream>
#include <fstream>
#include <algorithm>

using namespace cv;
using namespace std;

// --- Configuración Global ---
const int PERSON_CLASS_ID = 0; // ID de la clase 'persona' en el coco.names traducido
const int ALARM_CLASS_ID = 26; // ID de la clase 'arma de fuego' (o similar), si entrenaste una. 
                               // Usaremos 26 como ejemplo para 'backpack' si no hay una clase arma real.

// Función para dibujar los resultados en el frame
void drawDetections(Mat& frame, const std::vector<Detection>& detections, const std::vector<std::string>& classNames) {
    for (const auto& det : detections) {
        // Rojo para alarma (si la clase coincide con ALARM_CLASS_ID), Verde para personas/otros
        Scalar color = (det.classId == ALARM_CLASS_ID) ? Scalar(0, 0, 255) : Scalar(0, 255, 0); 

        rectangle(frame, det.box, color, 2);
        
        // Formato: NombreClase ID:42 (Conf: 0.85)
        string label = classNames[det.classId] + format(" ID:%d (Conf: %.2f)", det.trackId, det.confidence);

        // Disparo de Alarma Visual en la parte superior del frame
        if (det.classId == ALARM_CLASS_ID) {
             putText(frame, "!!! ALERTA: OBJETO PELIGROSO DETECTADO !!!", Point(50, 50), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 0, 255), 3);
        }

        // Etiqueta sobre el bounding box
        putText(frame, label, Point(det.box.x, det.box.y - 10), FONT_HERSHEY_SIMPLEX, 0.5, color, 2);
    }
}

int main() {
    // --- CONFIGURACIÓN DE ARCHIVOS ---
    string modelPath = "models/yolov8n.onnx";
    string classesPath = "models/coco.names";
    string videoPath = "0"; // 0 para webcam

    try {
        // 1. Inicializar Componentes C++
        Yolov8Detector detector(modelPath, classesPath);
        KalmanTracker tracker;
        ZmqPublisher zmq_sender; 
        
        // Cargar nombres de clases
        vector<string> classNames;
        ifstream ifs(classesPath.c_str());
        string line;
        while (getline(ifs, line)) { classNames.push_back(line); }

        // 2. Inicializar Video
        VideoCapture cap;
        (videoPath == "0") ? cap.open(0) : cap.open(videoPath);

        if (!cap.isOpened()) {
            cerr << "Error: No se pudo abrir la fuente de video." << endl;
            return -1;
        }

        // 3. Bucle de Procesamiento
        while (waitKey(1) < 0) {
            Mat frame;
            cap >> frame;
            if (frame.empty()) break;

            // A. Detección
            std::vector<Detection> detections = detector.detect(frame);

            // B. Seguimiento (Asignación de IDs)
            tracker.updateTracks(detecciones);

            // C. Envío Asíncrono a Python (Reconocimiento Facial)
            for (const auto& det : detections) {
                // Solo enviamos si es una 'persona' y su ID es estable (trackId != -1)
                if (det.classId == PERSON_CLASS_ID && det.trackId != -1) { 
                    
                    // Recortar la región completa de la persona (Python se encarga de detectar el rostro)
                    // Usamos un tamaño fijo para evitar errores de recorte si el bbox está en el borde
                    Rect safe_box = det.box & Rect(0, 0, frame.cols, frame.rows);

                    if (safe_box.area() > 0) {
                        cv::Mat person_region = frame(safe_box).clone(); 
                        zmq_sender.sendFaceForRecognition(det.trackId, person_region); 
                    }
                }
            }
            
            // D. Visualización
            drawDetections(frame, detections, classNames);

            imshow("Intrusion Detector (C++ YOLO + ZMQ)", frame);
        }

    } catch (const exception& e) {
        cerr << "Ocurrió un error en C++: " << e.what() << endl;
        return -1;
    }
    return 0;
}
