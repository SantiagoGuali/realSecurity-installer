# recoignation_logic.py
import os
import sys
import time
import hashlib
import logging
import concurrent.futures

import cv2
import numpy as np
import tensorflow as tf

# Ajustar sys.path si tu proyecto lo requiere
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import mediapipe as mp
except (ImportError, OSError) as e:
    print(f"❌ Error al importar MediaPipe: {e}")
    print("Asegúrate de tener mediapipe==0.10.21 correctamente instalado, ")
    print("y si estás en Windows 10/11 con Python 3.11, comprueba Visual C++ Redistributable.")
    sys.exit(1)

from functions.database_manager import DatabaseManager
from functions.face_validator import GestorPersistenciaRostros
from utils.settings_controller import (
    SIMILARITY_THRESHOLD,
    TIME_RECOGNITION,       # Para registrar asistencia tras X seg
    DETECTION_CONFIDENCE    # Umbral detección
)
from utils.foto_validate import detectar_parpadeo, detectar_reflejos
from embeddings_models.facenet_model import align_face, detect_faces

# ========== AJUSTES DE LOG ==========
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.get_logger().setLevel("ERROR")
logging.getLogger("mediapipe").setLevel(logging.ERROR)

# ========== PARÁMETROS GLOBALES ==========
FRAME_SKIP = 2                # Saltar frames para optimizar
FAIL_PERCENT = 10             # Umbral mínimo de "confianza"
umbral_movimiento_ojos = 0.005

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# Control de detecciones para asistencia
detecciones_continuas = {}        # {emp_id: {"inicio": t_inicial, "registrado": bool, "tipo": None}}
registro_desconocidos = {}        # hashing de embeddings desconocidos
TIEMPO_MINIMO_ENTRE_DESCONOCIDOS = 10
ultima_deteccion_global = 0

posicion_ojos_anterior = None
db = DatabaseManager()

# ========== INICIALIZAR MEDIAPIPE ==========

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,   # Ajustable
    min_tracking_confidence=0.5
)

mp_face_detection = mp.solutions.face_detection
face_detector = mp_face_detection.FaceDetection(
    model_selection=0,
    min_detection_confidence=DETECTION_CONFIDENCE or 0.7
)

# Gestor de rostros (para estabilizar la identidad)
gestor_rostros = GestorPersistenciaRostros()

# ========== CARGA FACE NET ==========
FACENET_MODEL_PATH = "files/facenet_model/20180402-114759.pb"

def load_facenet_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No se encontró el modelo FaceNet en: {model_path}")
    try:
        graph = tf.Graph()
        with graph.as_default():
            with tf.io.gfile.GFile(model_path, "rb") as f:
                graph_def = tf.compat.v1.GraphDef()
                graph_def.ParseFromString(f.read())
                tf.import_graph_def(graph_def, name="")
        return graph
    except Exception as e:
        raise RuntimeError(f"Error al cargar FaceNet: {e}")

graph = load_facenet_model(FACENET_MODEL_PATH)
sess = tf.compat.v1.Session(graph=graph)

# ========== FUNCIONES AUXILIARES DE EMBEDDINGS ==========
def comparar_embeddings(embedding1, embedding2, threshold=SIMILARITY_THRESHOLD):
    """
    Compara dos embeddings (R=512) y devuelve:
     - es_similar (bool)
     - distancia
     - porcentaje_confianza (0..100)
    """
    if embedding1.shape != (512,) or embedding2.shape != (512,):
        return False, float("inf"), 0.0

    dist = np.linalg.norm(embedding1 - embedding2)
    similitud = np.dot(embedding1, embedding2)
    porc_conf = max(0, min(100, (1 - dist) * 100))

    # Ajuste dinámico leve
    if dist > 1.0:
        threshold += 0.2
    elif dist > 0.7:
        threshold += 0.1

    es_similar = (dist < threshold) or (similitud > 0.25)
    return es_similar, dist, porc_conf

def preprocess_image(image, required_size=(160, 160)):
    """
    Reescalar y normalizar un recorte de rostro antes de FaceNet
    """
    if image is None or image.shape[0] < 30 or image.shape[1] < 30:
        return None

    h, w = image.shape[:2]
    if h < required_size[0] or w < required_size[1]:
        factor = required_size[0] / min(h, w)
        image = cv2.resize(image, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)

    image = cv2.GaussianBlur(image, (3, 3), 0)
    image = cv2.resize(image, required_size)
    image = image.astype("float32")

    mean, std = image.mean(), image.std()
    if std < 1e-6:
        return None
    image = (image - mean) / std

    return np.expand_dims(image, axis=0)

def obtener_embedding_facenet(frame, x1, y1, x2, y2):
    """
    Recibe bounding box, recorta el frame, lo preprocesa y extrae embedding con FaceNet
    """
    rostro = frame[y1:y2, x1:x2]
    if rostro.shape[0] == 0 or rostro.shape[1] == 0:
        return None

    procesado = preprocess_image(rostro)
    if procesado is None or procesado.shape != (1, 160, 160, 3):
        return None

    input_tensor = graph.get_tensor_by_name("input:0")
    embeddings_tensor = graph.get_tensor_by_name("embeddings:0")
    phase_train_tensor = graph.get_tensor_by_name("phase_train:0")

    feed_dict = {input_tensor: procesado, phase_train_tensor: False}
    try:
        embedding = sess.run(embeddings_tensor, feed_dict=feed_dict)[0]
        return embedding / np.linalg.norm(embedding)
    except:
        return None

# ========== GUARDAR DESCONOCIDO ==========
def guardar_rostro_desconocido(rostro_bgr):
    """Guarda el recorte del rostro desconocido en DB (caras_desconocidas)."""
    try:
        _, buffer = cv2.imencode(".jpg", rostro_bgr)
        db.save_unknown_face(buffer.tobytes())
        print("✅ Cara desconocida almacenada en DB.")
    except Exception as e:
        print(f"Error guardando rostro desconocido: {e}")

# ========== PROCESAR UN ROSTRO ==========
def procesar_rostro(
    frame,
    x1, y1, x2, y2,
    embeddings_bd, nombres_empleados,
    umbral_confianza=FAIL_PERCENT,
    identidad_actual="Desconocido",
    confianza_actual=0.0,
    ultimo_embedding=None,
    identidad_confirmada=False,
    tipo="ENTRADA"
):
    """
    Extrae embedding, compara con BD, registra asistencia o guarda desconocido.
    Retorna (identidad_actualizada, confianza, embedding_detectado, id_confirmada)
    """
    global detecciones_continuas, ultima_deteccion_global, registro_desconocidos

    emb_detectado = obtener_embedding_facenet(frame, (x1, y1, x2, y2))
    if emb_detectado is None:
        return identidad_actual, confianza_actual, ultimo_embedding, identidad_confirmada

    # Revisa cambio brusco
    cambio_rostro = False
    if ultimo_embedding is not None:
        if np.linalg.norm(emb_detectado - ultimo_embedding) > 0.7:
            cambio_rostro = True
    else:
        cambio_rostro = True

    if cambio_rostro:
        identidad_actual = "Desconocido"
        confianza_actual = 0.0
        identidad_confirmada = False

    # Buscar coincidencia en BD
    nombre_detectado = "Desconocido"
    menor_dist = float("inf")
    confianza_detectada = 0.0
    emp_id = None

    for emb_bd, nombre_bd in zip(embeddings_bd, nombres_empleados):
        es_similar, dist, conf = comparar_embeddings(emb_detectado, emb_bd)
        if dist < menor_dist:
            menor_dist = dist
            nombre_detectado = nombre_bd
            confianza_detectada = conf

    if identidad_confirmada:
        # Ya estaba confirmada, no actualizamos
        return identidad_actual, confianza_actual, emb_detectado, identidad_confirmada

    if confianza_detectada >= umbral_confianza:
        # Reconocido
        identidad_actual = nombre_detectado
        confianza_actual = confianza_detectada
        identidad_confirmada = True

        # Manejar detección continua
        emp_id = db.obtener_id_empleado(nombre_detectado)
        if emp_id:
            if emp_id not in detecciones_continuas:
                detecciones_continuas[emp_id] = {"inicio": time.time(), "registrado": False, "tipo": None}
            t_detectado = time.time() - detecciones_continuas[emp_id]["inicio"]
            if t_detectado >= TIME_RECOGNITION:
                if (not detecciones_continuas[emp_id]["registrado"]
                    or detecciones_continuas[emp_id]["tipo"] != tipo):
                    db.registrar_asistencia(emp_id, tipo)
                    detecciones_continuas[emp_id]["registrado"] = True
                    detecciones_continuas[emp_id]["tipo"] = tipo
                    print(f"Asistencia registrada para {nombre_detectado} ({tipo}).")
    else:
        # Desconocido
        identidad_actual = "Desconocido"
        confianza_actual = 0.0
        identidad_confirmada = False

        emb_hash = hashlib.md5(emb_detectado.round(0).tobytes()).hexdigest()
        ultima_det = registro_desconocidos.get(emb_hash, None)

        ahora = time.time()
        if ((ultima_det is None or (ahora - ultima_det > TIEMPO_MINIMO_ENTRE_DESCONOCIDOS))
            and (ahora - ultima_deteccion_global > TIEMPO_MINIMO_ENTRE_DESCONOCIDOS)):
            print("🟢 Rostro desconocido nuevo. Guardando en DB...")
            desconocido_crop = frame[y1:y2, x1:x2]
            guardar_rostro_desconocido(desconocido_crop)
            registro_desconocidos[emb_hash] = ahora
            ultima_deteccion_global = ahora
        else:
            print("🔴 Rostro desconocido repetido en tiempo de espera.")

    return identidad_actual, confianza_actual, emb_detectado, identidad_confirmada

# ========== DETECTAR MOV. OJOS ==========
def detectar_movimiento_ojos(landmarks):
    global posicion_ojos_anterior
    # Escoger landmarks de ojo izq / der
    left_eye_center = np.mean([[landmarks[idx].x, landmarks[idx].y] for idx in LEFT_EYE], axis=0)
    right_eye_center = np.mean([[landmarks[idx].x, landmarks[idx].y] for idx in RIGHT_EYE], axis=0)
    eye_center = (left_eye_center + right_eye_center) / 2

    if posicion_ojos_anterior is None:
        posicion_ojos_anterior = eye_center
        return False

    movimiento = np.linalg.norm(np.array(eye_center) - np.array(posicion_ojos_anterior))
    posicion_ojos_anterior = eye_center
    return movimiento > umbral_movimiento_ojos

# ========== PROCESAR FRAME (SINGLE-FRAME) ==========
def procesar_reconocimiento_facial(frame, embeddings_bd, nombres_empleados, tipo="ENTRADA"):
    """
    Procesa UN SOLO FRAME con FaceNet+MediaPipe:
    - FaceMesh => parpadeo, reflejos, mov_ojos
    - FaceDetection => bounding boxes
    - Concurrency para varios rostros
    - Dibuja bounding boxes con identidades
    Retorna frame BGR final
    """
    h, w, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # FaceMesh
    mesh_results = face_mesh.process(frame_rgb)
    if mesh_results.multi_face_landmarks:
        for fl in mesh_results.multi_face_landmarks:
            detectar_parpadeo(frame, fl)
            detectar_reflejos(frame)
            detectar_movimiento_ojos(fl.landmark)

    # FaceDetection
    detect_results = face_detector.process(frame_rgb)
    faces = detect_results.detections if (detect_results and detect_results.detections) else []

    nuevos_rostros = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        future_to_bbox = {}
        for face in faces:
            bboxC = face.location_data.relative_bounding_box
            x1 = max(int(bboxC.xmin * w) - 10, 0)
            y1 = max(int(bboxC.ymin * h) - 20, 0)
            x2 = min(int((bboxC.xmin + bboxC.width)*w)+10, w)
            y2 = min(int((bboxC.ymin + bboxC.height)*h)+20, h)

            if (x2 - x1) < 50 or (y2 - y1) < 50:
                continue

            centro = ((x1 + x2)//2, (y1 + y2)//2)
            fut = executor.submit(
                procesar_rostro,
                frame,
                x1, y1, x2, y2,
                embeddings_bd, nombres_empleados,
                FAIL_PERCENT,
                "Desconocido", 0.0,
                None, False,
                tipo
            )
            future_to_bbox[fut] = (centro, (x1, y1, x2, y2))

        for fut in concurrent.futures.as_completed(future_to_bbox):
            (centro, (x1,y1,x2,y2)) = future_to_bbox[fut]
            try:
                nombre_asignado, conf_asignada, emb_detectado, id_confirm = fut.result()
                nuevos_rostros[centro] = (nombre_asignado, conf_asignada)
            except Exception as e:
                print(f"❌ Error en procesar_rostro: {e}")

    # Estabilizar identidades
    rostros_filtrados = gestor_rostros.actualizar_rostros(nuevos_rostros)

    # Dibujar
    for centro, (nombre_detectado, conf_detectada) in rostros_filtrados.items():
        x1 = centro[0] - 60
        y1 = centro[1] - 70
        x2 = centro[0] + 60
        y2 = centro[1] + 70
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, w), min(y2, h)

        color = (0,255,0) if nombre_detectado != "Desconocido" else (0,0,255)
        cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
        cv2.putText(frame, f"{nombre_detectado} ({conf_detectada:.1f}%)",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    return frame

# ========== INICIALIZAR RECONOCIMIENTO ==========
def inicializar_reconocimiento():
    """
    Carga embeddings BD y retorna (embeddings_bd, nombres_empleados).
    """
    emb_list, nom_list = db.get_embeddings()
    return emb_list, nom_list
