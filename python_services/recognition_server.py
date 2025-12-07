import zmq
import numpy as np
import cv2
import sqlite3
import json
import torch
import io
import base64
from facenet_pytorch import InceptionResnetV1, MTCNN
from sklearn.metrics.pairwise import cosine_similarity 

# --- 1. Inicialización de Deep Learning y CUDA ---
# Utiliza la GPU si está disponible
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Usando dispositivo: {device}")

# MTCNN: Detector y alineador de rostros
mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
    device=device
)
# InceptionResnetV1 (FaceNet): Modelo para generar el embedding (vector de características)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# --- 2. Base de Datos (SQLite) ---
DB_NAME = 'python_services/face_database.db' # Ruta relativa dentro del proyecto

def setup_database():
    """Crea la tabla de rostros conocidos si no existe."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS known_faces (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            category TEXT NOT NULL,  -- Ej: 'Empleado', 'VIP', 'Problema'
            embedding BLOB NOT NULL 
        )
    """)
    conn.commit()
    conn.close()

def get_best_match(embedding):
    """Compara el embedding de entrada con la BD usando Distancia Coseno."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT name, category, embedding FROM known_faces")
    
    known_embeddings = []
    known_labels = []
    
    for name, category, stored_embedding_blob in cursor.fetchall():
        # Deserializar el BLOB (bytes) de la BD a un array numpy (float32)
        stored_embedding = np.frombuffer(stored_embedding_blob, dtype=np.float32)
        known_embeddings.append(stored_embedding)
        known_labels.append((name, category))
        
    conn.close()

    if not known_embeddings:
        return "Desconocido", "Intruso", 0.0

    known_embeddings = np.array(known_embeddings)
    
    # Calcular Distancia Coseno (similarity = 1 es idéntico)
    # El embedding de entrada debe ser 2D para cosine_similarity
    similarities = cosine_similarity(embedding.reshape(1, -1), known_embeddings)
    best_similarity = np.max(similarities)
    
    # Umbral de reconocimiento (Ajuste este valor: 0.75-0.85 es común)
    RECOGNITION_THRESHOLD = 0.78 
    
    if best_similarity > RECOGNITION_THRESHOLD:
        best_match_index = np.argmax(similarities)
        name, category = known_labels[best_match_index]
        return name, category, best_similarity
    else:
        return "Desconocido", "Intruso", best_similarity

# --- 4. Funciones de Ayuda (Para llenar la BD) ---
def register_face(name, category, image_path):
    """Función de ejemplo para registrar un rostro en la BD."""
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: No se pudo cargar la imagen en {image_path}")
        return

    face_tensor = mtcnn(img, return_prob=False)
    if face_tensor is None:
        print("Error: No se detectó un rostro en la imagen de registro.")
        return

    # Generar el Embedding
    face_tensor = face_tensor.to(device).unsqueeze(0)
    embedding = resnet(face_tensor).detach().cpu().numpy().flatten()

    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    # Convertir array numpy a bytes (BLOB)
    embedding_blob = embedding.tobytes()
    
    cursor.execute("INSERT INTO known_faces (name, category, embedding) VALUES (?, ?, ?)", 
                   (name, category, embedding_blob))
    conn.commit()
    conn.close()
    print(f"Registrado: {name} ({category})")
    
# --- 5. Servidor ZMQ ---
def start_server():
    setup_database()
    
    # Ejecuta el ejemplo de registro una vez para probar (requiere una imagen local, ej: 'test_empleado.jpg')
    # register_face("Juan Pérez", "Empleado", "images/juan_perez.jpg") 

    context = zmq.Context()
    receiver = context.socket(zmq.PULL) 
    receiver.bind("tcp://127.0.0.1:5555") 
    print("Python Recognition Server (ZMQ PULL) iniciado en puerto 5555...")

    while True:
        try:
            # Recibir datos del C++ (JSON en formato de bytes)
            msg_bytes = receiver.recv()
            msg = json.loads(msg_bytes.decode('utf-8')) # Decodificar bytes ZMQ a JSON

            track_id = msg['track_id']
            img_data_bytes = msg['face_img_b64']
            
            # 1. Decodificar la imagen de string de bytes a array NumPy de OpenCV
            # Usamos latin-1 o iso-8859-1 para la codificación/decodificación binaria de bytes en JSON
            np_arr = np.frombuffer(img_data_bytes.encode('latin1'), np.uint8) 
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if img is None or img.size == 0:
                print(f"ID {track_id}: Error al decodificar imagen o imagen vacía.")
                continue

            # 2. Pre-procesamiento y Detección de Rostro (MTCNN)
            # Solo procesamos si el rostro es detectado por MTCNN (más confiable que solo el recorte de YOLO)
            face_tensor = mtcnn(img, return_prob=False)
            
            if face_tensor is not None:
                # 3. Generar el Embedding
                face_tensor = face_tensor.to(device).unsqueeze(0)
                embedding = resnet(face_tensor).detach().cpu().numpy().flatten()
                
                # 4. Consulta a la Base de Datos
                name, category, similarity = get_best_match(embedding)
                
                # 5. Lógica de Alarma y Logging:
                log_message = f"INFO: ID {track_id} -> {name} ({category}). Sim: {similarity:.4f}"
                
                # ALARMA: Si es clasificado como 'Problema' o 'Intruso Desconocido'
                if category == "Problema" or (name == "Desconocido"):
                    print(f"!!! ALERTA INMEDIATA DETECTADA !!! {log_message}")
                else:
                    print(log_message)
                
            else:
                print(f"ID {track_id}: Rostro no detectado/alineado en la región enviada.")

        except Exception as e:
            print(f"Error fatal en el servidor Python: {e}")

if __name__ == '__main__':
    start_server()
