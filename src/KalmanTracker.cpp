#include "../include/KalmanTracker.h"
#include <iostream>
#include <algorithm>

using namespace cv;
using namespace std;

KalmanTracker::KalmanTracker() : nextTrackId(1) {}

// Implementación del Filtro de Kalman (Modelo de Movimiento Constante de Velocidad)
KalmanFilter KalmanTracker::initKalman(const Rect& box) {
    // Estado (4x1): [cx, cy, w, h]
    // Medición (2x1): [cx, cy]
    KalmanFilter kf(4, 2, 0); 
    
    // Matriz de Transición A (4x4): Identidad
    setIdentity(kf.transitionMatrix); 
    
    // Matriz de Medición H (2x4)
    kf.measurementMatrix = (Mat_<float>(2, 4) << 
        1, 0, 0, 0,
        0, 1, 0, 0);
    
    // Inicialización del estado (Post-estado)
    kf.statePost.at<float>(0, 0) = box.x + box.width / 2.0f;
    kf.statePost.at<float>(1, 0) = box.y + box.height / 2.0f;
    kf.statePost.at<float>(2, 0) = (float)box.width;
    kf.statePost.at<float>(3, 0) = (float)box.height;

    // Covarianza de Ruido de Proceso Q y Medición R
    setIdentity(kf.processNoiseCov, Scalar::all(1e-4));
    setIdentity(kf.measurementNoiseCov, Scalar::all(1e-2));

    return kf;
}

float KalmanTracker::calculateIoU(const Rect& box1, const Rect& box2) {
    Rect intersection = box1 & box2;
    float intersectionArea = intersection.area();
    float unionArea = box1.area() + box2.area() - intersectionArea;
    return (unionArea > 0) ? (intersectionArea / unionArea) : 0.0f;
}

void KalmanTracker::updateTracks(std::vector<Detection>& detections) {
    // 1. Predicción: Actualizar todos los tracks existentes con el Filtro de Kalman
    for (auto& track : tracks) {
        track.kf.predict(); 
        track.age++;
    }

    // 2. Asociación (IoU Matching)
    std::vector<int> trackIndices(tracks.size(), -1); 
    std::vector<bool> matchedDetections(detections.size(), false);
    
    // Emparejar tracks existentes con las detecciones del frame actual
    for (size_t i = 0; i < tracks.size(); ++i) {
        if (tracks[i].age > 10) continue; // Descartar tracks perdidos
        
        float maxIou = 0.0f;
        int bestDetectionIdx = -1;

        // Buscar la mejor coincidencia IoU
        for (size_t j = 0; j < detections.size(); ++j) {
            if (matchedDetections[j]) continue;

            float iou = calculateIoU(tracks[i].lastBox, detections[j].box);
            
            if (iou > maxIou && iou > 0.5) { // Umbral IoU ajustable (0.5)
                maxIou = iou;
                bestDetectionIdx = j;
            }
        }

        // Si hay match: Corregir el estado con la medición real
        if (bestDetectionIdx != -1) {
            Rect matchedBox = detections[bestDetectionIdx].box;
            
            Mat measurement = (Mat_<float>(2, 1) << 
                matchedBox.x + matchedBox.width / 2.0f,
                matchedBox.y + matchedBox.height / 2.0f);
            
            tracks[i].kf.correct(measurement); // Corregir el estado
            tracks[i].lastBox = matchedBox;
            tracks[i].age = 0;
            detections[bestDetectionIdx].trackId = tracks[i].id;
            matchedDetections[bestDetectionIdx] = true;
        }
    }

    // 3. Inicializar Nuevos Tracks
    for (size_t i = 0; i < detections.size(); ++i) {
        if (!matchedDetections[i]) {
            Track newTrack;
            newTrack.id = nextTrackId++;
            newTrack.kf = initKalman(detections[i].box);
            newTrack.lastBox = detections[i].box;
            newTrack.age = 0;
            detections[i].trackId = newTrack.id;
            tracks.push_back(newTrack);
        }
    }

    // 4. Limpieza (Eliminar tracks perdidos)
    tracks.erase(std::remove_if(tracks.begin(), tracks.end(), 
                                [](const Track& t){ return t.age > 10; }), 
                 tracks.end());
}
