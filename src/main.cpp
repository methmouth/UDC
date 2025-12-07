#include "../include/Yolov8Detector.h"
#include "../include/KalmanTracker.h"
#include "../include/ZmqPublisher.h"
#include <iostream>
#include <fstream>
#include <algorithm>

using namespace cv;
using namespace std;

// Función auxiliar para dibujar los resultados
void drawDetections(Mat& frame, const std::vector<Detection>& detections, const std::vector<std::string>& classNames) {
    // Implementación de visualización con alarmas en español
    // ... (El código provisto en la respuesta anterior de traducción)
}

int main() {
    // --- CONFIGURACIÓN DE ARCHIVOS ---
    string modelPath = "models/yolov8n.onnx";
    string classesPath = "models/coco.names";
    string videoPath = "0"; 
    const int PERSON_CLASS_ID = 0; // Asumiendo que 'persona' es la primera clase

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
            tracker.updateTracks(detections);

            // C. Envío Asíncrono a Python (Reconocimiento Facial)
            for (const auto& det : detections) {
                // Solo enviamos si es una 'persona' y su ID es conocido (seguimiento estable)
                if (det.classId == PERSON_CLASS_ID && det.trackId != -1) { 
                    // Se recorta la región completa de la persona (simplificado)
                    cv::Mat person_region = frame(det.box).clone(); 
                    
                    // Envío a Python para reconocimiento (asíncrono)
                    zmq_sender.sendFaceForRecognition(det.trackId, person_region); 
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
