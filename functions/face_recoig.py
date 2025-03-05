# face_recognition.py
import torch
import cv2
import numpy as np
import hashlib
from datetime import datetime
from facenet_pytorch import MTCNN, InceptionResnetV1
from functions.database_manager import DatabaseManager

# Inicializar modelos y base de datos
db = DatabaseManager()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=True, device=device, min_face_size=40, thresholds=[0.6, 0.7, 0.7], factor=0.8)
facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

def procesar_reconocimiento_facial(frame, embeddings_bd, nombres_empleados):
    """ Detecta y reconoce rostros en un frame. """
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = mtcnn.detect(frame_rgb, landmarks=False)

    if result[0] is None:
        return frame

    for box in result[0]:
        x1, y1, x2, y2 = map(int, box)
        rostro = frame[y1:y2, x1:x2]
        
        if rostro.shape[0] < 20 or rostro.shape[1] < 20:
            continue

        rostro_resized = cv2.resize(rostro, (160, 160))
        rostro_rgb = cv2.cvtColor(rostro_resized, cv2.COLOR_BGR2RGB)
        rostro_tensor = torch.tensor(rostro_rgb).permute(2, 0, 1).unsqueeze(0).float().to(device)
        rostro_tensor = (rostro_tensor - 127.5) / 128.0

        embedding_detectado = facenet(rostro_tensor).detach().cpu().numpy()[0]
        embedding_detectado = embedding_detectado / np.linalg.norm(embedding_detectado)
        embedding_hash = hashlib.md5(embedding_detectado.round(0).tobytes()).hexdigest()

        emp_id, nombre_detectado = identificar_empleado(embedding_detectado, embeddings_bd, nombres_empleados)

        color = (0, 255, 0) if emp_id else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{nombre_detectado}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        if not emp_id:
            guardar_rostro_desconocido(rostro, embedding_hash)

    return frame

def identificar_empleado(embedding_detectado, embeddings_bd, nombres_empleados, threshold=0.6):
    """ Compara el embedding detectado con la base de datos para encontrar una coincidencia. """
    menor_distancia = float('inf')
    emp_id, nombre_detectado = None, "Desconocido"

    for embedding_bd, nombre in zip(embeddings_bd, nombres_empleados):
        distancia = np.linalg.norm(embedding_detectado - embedding_bd.cpu().numpy())
        if distancia < threshold and distancia < menor_distancia:
            menor_distancia = distancia
            nombre_detectado = nombre
            emp_id = obtener_id_empleado(nombre_detectado)

    return emp_id, nombre_detectado

def guardar_rostro_desconocido(rostro, embedding_hash):
    """ Guarda una imagen de un rostro desconocido en la base de datos. """
    ultima_deteccion = db.get_unknown_face_by_id(embedding_hash)

    if ultima_deteccion is None or (datetime.now().timestamp() - ultima_deteccion > 10):
        _, buffer = cv2.imencode(".jpg", rostro)
        db.save_unknown_face(buffer.tobytes())
        print("✅ Imagen de rostro desconocido almacenada.")

def obtener_id_empleado(nombre_completo):
    """ Obtiene el ID del empleado a partir de su nombre. """
    return db.get_empleado_id(nombre_completo)
