#include "../include/Yolov8Detector.h"
#include <fstream>
#include <iostream>

using namespace cv;
using namespace cv::dnn;
using namespace std;

Yolov8Detector::Yolov8Detector(const std::string& modelPath, const std::string& classesPath, float conf, float nms) 
    : confThreshold(conf), nmsThreshold(nms) {
    
    // 1. Cargar el modelo ONNX
    net = readNet(modelPath);
    net.setPreferableBackend(DNN_BACKEND_OPENCV);
    net.setPreferableTarget(DNN_TARGET_CPU); // Usar DNN_TARGET_CUDA para GPU
}

std::vector<Detection> Yolov8Detector::detect(const Mat& frame) {
    Mat blob;
    // Preprocesamiento estándar para YOLOv8 (640x640, escalado 1/255)
    blobFromImage(frame, blob, 1/255.0, Size(640, 640), Scalar(), true, false); 
    
    net.setInput(blob);
    
    std::vector<Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames()); 
    
    std::vector<Detection> detections;
    postProcess(frame, outputs, detections);
    
    return detections;
}

void Yolov8Detector::postProcess(const Mat& frame, const std::vector<Mat>& outputs, std::vector<Detection>& detections) {
    // La salida de YOLOv8 ONNX es típicamente un tensor de forma (1, 84, N)
    Mat output = outputs[0].reshape(1, outputs[0].size[2]); 

    std::vector<Rect> boxes;
    std::vector<float> confidences;
    std::vector<int> classIds;
    int frameWidth = frame.cols;
    int frameHeight = frame.rows;

    for (int i = 0; i < output.rows; ++i) {
        // Obtenemos las puntuaciones de clase (columnas 4 en adelante)
        Mat scores = output.row(i).colRange(4, output.cols);
        Point classIdPoint;
        double confidence;
        
        // Encontramos la clase con la puntuación más alta
        minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
        
        if (confidence > confThreshold) {
            // Decodificación de los bounding boxes (formato [cx, cy, w, h])
            float cx = output.at<float>(i, 0);
            float cy = output.at<float>(i, 1);
            float w = output.at<float>(i, 2);
            float h = output.at<float>(i, 3);

            // Conversión a coordenadas [x, y, w, h] y escalado al tamaño original del frame
            int x = static_cast<int>((cx - 0.5f * w) * (frameWidth / 640.0f));
            int y = static_cast<int>((cy - 0.5f * h) * (frameHeight / 640.0f));
            int width = static_cast<int>(w * (frameWidth / 640.0f));
            int height = static_cast<int>(h * (frameHeight / 640.0f));

            boxes.push_back(Rect(x, y, width, height));
            confidences.push_back((float)confidence);
            classIds.push_back(classIdPoint.x);
        }
    }

    // Aplicar NMS (Non-Maximum Suppression)
    std::vector<int> indices;
    NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);

    for (int idx : indices) {
        detections.push_back({boxes[idx], confidences[idx], classIds[idx]});
    }
}
