#pragma once
#include <string>
#include <opencv2/opencv.hpp>
#include <zmq.hpp>

class ZmqPublisher {
public:
    ZmqPublisher(const std::string& connect_addr = "tcp://127.0.0.1:5555");
    ~ZmqPublisher();

    // Envía la imagen del rostro como un string codificado a través del socket PUSH
    void sendFaceForRecognition(int track_id, const cv::Mat& face_image);

private:
    zmq::context_t context;
    zmq::socket_t sender;
};
