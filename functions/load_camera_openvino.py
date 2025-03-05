import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import dlib
import numpy as np
import time
from datetime import datetime
from scipy.spatial.distance import cosine
from facenet_pytorch import MTCNN
from plyer import notification

import torch
import openvino.runtime as ov

# Importar configuraciones y base de datos
from utils.settings_controller import (
    SIMILARITY_THRESHOLD,
    TIME_RECOGNITION,
    NOTIFICATION_ASISTENCIA,
    ENTRY_INTERVAL_START,
    ENTRY_INTERVAL_END,
    EXIT_INTERVAL_START,
    EXIT_INTERVAL_END
)
from functions.database_manager import DatabaseManager

# --------------------------------------------------------------------------
# CONFIGURACIONES GLOBALES
# --------------------------------------------------------------------------
db = DatabaseManager()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DETECTION_CONFIDENCE = 0.95
FPS_PROCESAMIENTO = 10
ultimo_procesamiento = time.time()

# Intervalos de registro (usando las variables de settings_controller)
INTERVALO_ENTRADA = (ENTRY_INTERVAL_START, ENTRY_INTERVAL_END)
INTERVALO_SALIDA = (EXIT_INTERVAL_START, EXIT_INTERVAL_END)

# Parámetros para suplantación
SUPLANTACION_REFLEJO_UMBRAL = 30   # Varianza de Laplacian mínima
SUPLANTACION_PARPADEO_UMBRAL = 3   # Diferencia de altura en ojos

# -------------------- MTCNN - Detección de rostros ------------------------
mtcnn = MTCNN(
    keep_all=True,
    device=DEVICE,
    min_face_size=40,
    thresholds=[0.6, 0.7, 0.7],
    post_process=False
)

# -------------------- OpenVINO - FaceNet Embeddings -----------------------
ie = ov.Core()
facenet_model_path = "model_ir/facenet.xml"  # Asegúrate de tener también el .bin
compiled_model_facenet = ie.compile_model(facenet_model_path, "CPU")
input_layer_facenet = compiled_model_facenet.input(0)
output_layer_facenet = compiled_model_facenet.output(0)

# -------------------- OpenVINO - Anti-Spoofing model ----------------------
anti_spoof_xml = "model_ir/anti_spoofing.xml"
anti_spoof_bin = "model_ir/anti_spoofing.bin"
compiled_model_anti_spoof = ie.compile_model(model=anti_spoof_xml, device_name="CPU")
input_layer_anti_spoof = compiled_model_anti_spoof.input(0)
output_layer_anti_spoof = compiled_model_anti_spoof.output(0)

# -------------------- Dlib - Parpadeo / Spoof check -----------------------
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("files/shape_predictor_68_face_landmarks.dat")

# Estructura para detectar si el rostro permanece cierto tiempo
detecciones_continuas = {}

# --------------------------------------------------------------------------
# 1) inicializar_reconocimiento()
# --------------------------------------------------------------------------
def inicializar_reconocimiento():
    try:
        embeddings_bd, nombres_empleados = db.get_embeddings()
        # Convertir cada embedding a numpy array (float32), por seguridad
        embeddings_bd = [np.array(e, dtype=np.float32) for e in embeddings_bd]
        return embeddings_bd, nombres_empleados
    except Exception as e:
        print(f"Error al inicializar reconocimiento: {e}")
        return [], []

# --------------------------------------------------------------------------
# 2) Checks tradicionales de suplantación
# --------------------------------------------------------------------------
def detectar_reflejo(rostro_bgr):
    """
    True si la varianza del Laplacian es menor a SUPLANTACION_REFLEJO_UMBRAL
    => posible foto/pantalla (sin textura).
    """
    gray = cv2.cvtColor(rostro_bgr, cv2.COLOR_BGR2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return lap_var < SUPLANTACION_REFLEJO_UMBRAL

def detectar_parpadeo(frame_bgr):
    """
    Retorna False si no se detecta parpadeo => ojo_izq y ojo_der difieren
    menos de SUPLANTACION_PARPADEO_UMBRAL en altura => posible foto.
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    rostros = detector(gray, 0)
    for rostro in rostros:
        landmarks = predictor(gray, rostro)
        ojo_izq = (landmarks.part(36).x, landmarks.part(36).y)
        ojo_der = (landmarks.part(45).x, landmarks.part(45).y)
        if abs(ojo_izq[1] - ojo_der[1]) < SUPLANTACION_PARPADEO_UMBRAL:
            return False
    return True

# --------------------------------------------------------------------------
# 3) Detección de suplantación con Anti-Spoofing (OpenVINO)
# --------------------------------------------------------------------------
def detectar_suplantacion_anti_spoofing(rostro_bgr, threshold=0.5):
    """
    Usa el modelo de OpenVINO anti-spoofing para clasificar el rostro.
    Return: True si se considera SPOOF (suplantación).
    """
    input_size = (128, 128)  # Ajustar a tu modelo
    rostro_rgb = cv2.cvtColor(rostro_bgr, cv2.COLOR_BGR2RGB)
    rostro_rgb = cv2.resize(rostro_rgb, input_size)

    blob = rostro_rgb.transpose((2, 0, 1))
    blob = np.expand_dims(blob, axis=0).astype(np.float32)

    request = compiled_model_anti_spoof.create_infer_request()
    request.infer({input_layer_anti_spoof: blob})
    output_data = request.get_output_tensor(0).data
    score = output_data[0][0]  # Un ejemplo de cómo leer si es 1D

    # Ejemplo de umbral
    if score < threshold:
        return True
    return False

# --------------------------------------------------------------------------
# 4) Función combinada
# --------------------------------------------------------------------------
def es_suplantacion(rostro_bgr, frame_bgr):
    # 1) Modelo anti-spoofing
    if detectar_suplantacion_anti_spoofing(rostro_bgr):
        print("[Suplantación] El modelo anti-spoofing indicó SPOOF.")
        return True

    # 2) Chequeo de reflejo
    if detectar_reflejo(rostro_bgr):
        print("[Suplantación] Reflejo anormal => posible foto/pantalla.")
        return True

    # 3) Chequeo de parpadeo
    if not detectar_parpadeo(frame_bgr):
        print("[Suplantación] No se detecta parpadeo => posible foto.")
        return True

    return False

# --------------------------------------------------------------------------
# 5) Intervalos de horario
# --------------------------------------------------------------------------
def dentro_de_intervalo(hora_str, intervalo):
    """
    Verifica si hora_str (HH:MM) está en el rango [intervalo[0], intervalo[1]].
    """
    h1, m1 = map(int, intervalo[0].split(':'))
    h2, m2 = map(int, intervalo[1].split(':'))
    hi, mi = map(int, hora_str.split(':'))

    inicio = h1 * 60 + m1
    fin = h2 * 60 + m2
    actual = hi * 60 + mi
    return inicio <= actual <= fin

def tipo_permitido_auto():
    """
    Retorna 'entrada', 'salida' o None según la hora actual.
    """
    hora_str = datetime.now().strftime("%H:%M")
    if dentro_de_intervalo(hora_str, INTERVALO_ENTRADA):
        return "entrada"
    elif dentro_de_intervalo(hora_str, INTERVALO_SALIDA):
        return "salida"
    else:
        return None

# --------------------------------------------------------------------------
# 6) Funciones de embeddings (FaceNet OpenVINO)
# --------------------------------------------------------------------------
def get_embedding_openvino(face_bgr):
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    face_rgb = cv2.resize(face_rgb, (160, 160))

    blob = face_rgb.transpose((2, 0, 1))
    blob = np.expand_dims(blob, axis=0).astype(np.float32)
    blob = (blob - 127.5) / 128.0

    request = compiled_model_facenet.create_infer_request()
    request.infer({input_layer_facenet: blob})
    output_data = request.get_output_tensor(0).data

    embedding = output_data[0]
    embedding = embedding / np.linalg.norm(embedding)
    return embedding

def comparar_embeddings(embedding_detectado, embedding_bd):
    dist = np.linalg.norm(embedding_detectado - embedding_bd)
    return dist < SIMILARITY_THRESHOLD, dist

def calcular_confianza(embedding_detectado, embeddings_bd):
    """
    Mide cuán cerca está embedding_detectado del conjunto de embeddings_bd.
    Retorna un valor en [0, 100].
    """
    if not embeddings_bd:
        return 0.0
    distancias = []
    for emb_bd in embeddings_bd:
        dist = np.linalg.norm(embedding_detectado - emb_bd)
        distancias.append(dist)
    min_dist = min(distancias)
    c = max(0, 1 - (min_dist / SIMILARITY_THRESHOLD)) * 100
    return c

def obtener_id_empleado(nombre_completo):
    """
    Retorna el ID del empleado, dada la concatenación
    'nombres_emp + espacio + apellidos_emp'.
    """
    try:
        cursor = db.connection.cursor()
        cursor.execute(
            "SELECT id FROM empleados WHERE (nombres_emp + ' ' + apellidos_emp) = ?",
            (nombre_completo,)
        )
        res = cursor.fetchone()
        cursor.close()
        return res[0] if res else None
    except Exception as e:
        print(f"Error al obtener ID del empleado: {e}")
        return None

# --------------------------------------------------------------------------
# 7) Lógica de asistencia con DB
# --------------------------------------------------------------------------
def validar_registro_asistencia_especial(emp_id, nuevo_tipo):
    """
    Llama a la validación en la BD (requiere que exista entrada previa
    si se trata de 'salida', y que no exista entrada repetida).
    """
    return db.validar_registro_asistencia_vino(emp_id, nuevo_tipo)

def registrar_asistencia_especial(emp_id, tipo):
    """
    Llama al método que registra o actualiza la salida.
    Retorna True si se concretó la inserción/actualización, False si se ignoró.
    """
    return db.registrar_asistencia_vino(emp_id, tipo)

# --------------------------------------------------------------------------
# 8) Función principal => procesar_reconocimiento_facial
# --------------------------------------------------------------------------
def procesar_reconocimiento_facial(frame, embeddings_bd, nombres_empleados, tipo=None):
    """
    - Se procesa el frame con MTCNN para obtener rostros.
    - Por cada rostro, se verifica suplantación y se compara con embeddings.
    - Se registra asistencia según las reglas de horario y BD.
    """
    global ultimo_procesamiento
    ahora = time.time()
    # Control de FPS
    if (ahora - ultimo_procesamiento) < (1 / FPS_PROCESAMIENTO):
        return frame
    ultimo_procesamiento = ahora

    height, width = frame.shape[:2]
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 1) Detección de rostros
    try:
        detection_result = mtcnn.detect(frame_rgb, landmarks=False)
        if not detection_result or detection_result[0] is None:
            return frame
        boxes, probs = detection_result
    except Exception as e:
        print(f"Error detectando rostros con MTCNN: {e}")
        return frame

    # 2) Iterar cada rostro detectado
    for box, prob in zip(boxes, probs):
        if prob < DETECTION_CONFIDENCE:
            continue

        x1, y1, x2, y2 = map(int, box)
        # Ajustar límites por seguridad
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, width), min(y2, height)
        if x2 <= x1 or y2 <= y1:
            continue

        rostro_bgr = frame[y1:y2, x1:x2]
        if rostro_bgr.size == 0:
            continue

        # 2a) Chequeo de suplantación
        if es_suplantacion(rostro_bgr, frame):
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, "Suplantacion", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            continue

        # 2b) Obtener embedding y comparar con BD
        emb_detectado = get_embedding_openvino(rostro_bgr)
        nombre_detectado = "Desconocido"
        emp_id = None
        min_dist = 999.0

        for emb_bd, nombre_bd in zip(embeddings_bd, nombres_empleados):
            es_similar, dist = comparar_embeddings(emb_detectado, emb_bd)
            if es_similar and dist < min_dist:
                min_dist = dist
                nombre_detectado = nombre_bd
                emp_id = obtener_id_empleado(nombre_bd)

        # 2c) Calcular confianza (opcional)
        confianza = calcular_confianza(emb_detectado, embeddings_bd) if emp_id else 0.0

        # 2d) Dibujar recuadro
        color_rect = (0, 255, 0) if emp_id else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color_rect, 2)
        cv2.putText(frame,
                    f"{nombre_detectado} {confianza:.1f}%",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, color_rect, 2)

        # 3) Registro de asistencia
        if emp_id is not None:
            hora_str = datetime.now().strftime("%H:%M")
            print(f"[DEBUG] Hora actual: {hora_str}")

            # 3a) Determinar modo (entrada / salida) de forma manual o automática
            if tipo == "entrada":
                # Verificar que esté dentro del horario de entrada
                if not dentro_de_intervalo(hora_str, INTERVALO_ENTRADA):
                    print("[DEBUG] Fuera de rango ENTRADA => no registra.")
                    modo = None
                else:
                    modo = "entrada"

            elif tipo == "salida":
                # Verificar que esté dentro del horario de salida
                if not dentro_de_intervalo(hora_str, INTERVALO_SALIDA):
                    print("[DEBUG] Fuera de rango SALIDA => no registra.")
                    modo = None
                else:
                    modo = "salida"
            else:
                # Modo automático
                modo = tipo_permitido_auto()
                print(f"[DEBUG] tipo_permitido_auto => {modo}")

            # 3b) Intentar registrar si hay un modo válido (entrada/salida)
            if modo:
                # Si no tenemos un registro previo de detección para este emp_id, iniciamos
                if emp_id not in detecciones_continuas:
                    detecciones_continuas[emp_id] = {
                        "inicio": ahora
                    }

                tiempo_detect = ahora - detecciones_continuas[emp_id]["inicio"]

                # Solo registrar si llevamos TIME_RECOGNITION seg viendo el rostro
                if tiempo_detect >= TIME_RECOGNITION:
                    # Llamamos primero a la validación en BD
                    if validar_registro_asistencia_especial(emp_id, modo):
                        # Intentamos registrar/actualizar
                        registro_ok = registrar_asistencia_especial(emp_id, modo)
                        if registro_ok:
                            print(f"Se registró {modo} para {nombre_detectado}")
                            # Notificar solo si se registró de verdad
                            if NOTIFICATION_ASISTENCIA:
                                notification.notify(
                                    title="Registro de Asistencia",
                                    message=f"{nombre_detectado} => {modo}",
                                    app_name="FaceID System",
                                    timeout=4
                                )
                        else:
                            # Ocurre si la BD lo ignora (por la regla de 5 min en 'salida')
                            print(f"[DEBUG] Se intentó registrar {modo}, pero se ignoró. (<5min).")
                    else:
                        print(f"[DEBUG] No se puede registrar {modo} => la BD indica que no procede.")

    return frame
