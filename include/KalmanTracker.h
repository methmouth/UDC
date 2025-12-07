#pragma once
#include <opencv2/video/tracking.hpp>
#include <vector>
// Requerido para usar la estructura Detection
#include "Yolov8Detector.h" 

// Estructura para representar un rastro (track)
struct Track {
    int id;
    cv::KalmanFilter kf;
    int age; // Contador de frames sin detección
    cv::Rect lastBox;
};

class KalmanTracker {
public:
    KalmanTracker();
    void updateTracks(std::vector<Detection>& detections);

private:
    std::vector<Track> tracks;
    int nextTrackId;

    cv::KalmanFilter initKalman(const cv::Rect& box);
    float calculateIoU(const cv::Rect& box1, const cv::Rect& box2);
};
