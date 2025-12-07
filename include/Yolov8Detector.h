#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <vector>

// Estructura para contener los resultados de la detección y tracking
struct Detection {
    cv::Rect box;
    float confidence;
    int classId;
    int trackId = -1; // ID asignado por el tracker
};

class Yolov8Detector {
public:
    Yolov8Detector(const std::string& modelPath, const std::string& classesPath, float confThreshold = 0.25f, float nmsThreshold = 0.45f);
    std::vector<Detection> detect(const cv::Mat& frame);

private:
    cv::dnn::Net net;
    float confThreshold;
    float nmsThreshold;

    void postProcess(const cv::Mat& frame, const std::vector<cv::Mat>& outputs, std::vector<Detection>& detections);
};
