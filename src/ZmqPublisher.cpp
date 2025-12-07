#include "../include/ZmqPublisher.h"
#include <iostream>
#include <string>
#include <vector>

// Necesario para serializar la imagen de OpenCV a JSON.
// Requiere la librería nlohmann/json.hpp en tu path de inclusión.
#include "json.hpp" 

using json = nlohmann::json;

ZmqPublisher::ZmqPublisher(const std::string& connect_addr) : context(1), sender(context, ZMQ_PUSH) {
    try {
        sender.connect(connect_addr);
        //std::cout << "ZMQ Publisher conectado a: " << connect_addr << std::endl;
    } catch (const zmq::error_t& e) {
        std::cerr << "Error ZMQ al conectar. Verifica que el servidor Python esté iniciado: " << e.what() << std::endl;
    }
}

ZmqPublisher::~ZmqPublisher() {
    if (sender.handle() != nullptr) {
        sender.close();
    }
}

void ZmqPublisher::sendFaceForRecognition(int track_id, const cv::Mat& face_image) {
    if (face_image.empty()) return;

    // 1. Codificar la imagen OpenCV a formato JPEG
    std::vector<uchar> buf;
    cv::imencode(".jpg", face_image, buf);
    
    // 2. Convertir el buffer de bytes a un string binario
    std::string img_data(buf.begin(), buf.end());
    
    // 3. Crear el mensaje JSON
    json msg;
    msg["track_id"] = track_id;
    // Se envía el string binario. Python debe decodificarlo.
    msg["face_img_b64"] = img_data; 
    
    std::string json_str = msg.dump();
    
    // 4. Enviar mensaje a través de ZMQ (no bloqueante)
    try {
        zmq::message_t message(json_str.size());
        memcpy(message.data(), json_str.data(), json_str.size());
        sender.send(message, zmq::send_flags::none);
    } catch (const zmq::error_t& e) {
        // Manejar errores de envío sin detener el bucle principal de video
        // std::cerr << "Error ZMQ al enviar: " << e.what() << std::endl;
    }
}
