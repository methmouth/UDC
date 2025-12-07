#include "../include/ZmqPublisher.h"
#include <iostream>
#include <string>
#include <vector>

// Necesario para serializar la imagen de OpenCV a JSON.
// Se asume que este archivo ha sido descargado y colocado en un directorio accesible (como 'include').
#include "json.hpp" 

using json = nlohmann::json;

// Usamos latin1 en Python para la codificación, así que usamos el mismo enfoque para decodificación/envío si es posible.
// Esto es una simplificación; en entornos de producción se usaría Base64 en C++ y Python.

ZmqPublisher::ZmqPublisher(const std::string& connect_addr) : context(1), sender(context, ZMQ_PUSH) {
    try {
        // Conectar al socket PULL del servidor Python
        sender.connect(connect_addr);
        std::cout << "ZMQ Publisher conectado a: " << connect_addr << std::endl;
    } catch (const zmq::error_t& e) {
        std::cerr << "Error ZMQ al conectar. Asegúrate que el servidor Python esté iniciado o verifica la dirección: " << e.what() << std::endl;
    }
}

ZmqPublisher::~ZmqPublisher() {
    // El destructor cierra el socket
    if (sender.handle() != nullptr) {
        sender.close();
    }
}

void ZmqPublisher::sendFaceForRecognition(int track_id, const cv::Mat& face_image) {
    if (face_image.empty()) return;

    // 1. Codificar la imagen OpenCV a formato JPEG (bytes binarios)
    std::vector<uchar> buf;
    // La compresión JPEG es clave para reducir el tamaño de los datos enviados por ZMQ
    cv::imencode(".jpg", face_image, buf);
    
    // 2. Convertir el buffer de bytes a un string binario para JSON
    // Se utiliza un constructor de string que toma un rango de iteradores
    std::string img_data(buf.begin(), buf.end());
    
    // 3. Crear el mensaje JSON con la información de seguimiento y la imagen
    json msg;
    msg["track_id"] = track_id;
    // Nota: enviamos el string de bytes binarios. El servidor Python lo recibirá y decodificará.
    msg["face_img_b64"] = img_data; 
    
    std::string json_str = msg.dump();
    
    // 4. Enviar mensaje a través de ZMQ
    try {
        zmq::message_t message(json_str.size());
        // Copiar el contenido de la cadena JSON al buffer de mensajes ZMQ
        memcpy(message.data(), json_str.data(), json_str.size());
        
        // El socket PUSH envía el mensaje de forma no bloqueante
        sender.send(message, zmq::send_flags::none);
        // std::cout << "Enviado ID " << track_id << " a Python." << std::endl; // Descomentar para debug
    } catch (const zmq::error_t& e) {
        std::cerr << "Error ZMQ al enviar mensaje: " << e.what() << std::endl;
    }
}
