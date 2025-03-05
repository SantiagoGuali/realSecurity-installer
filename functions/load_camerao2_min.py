import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import dlib
import torch
import numpy as np
import time
from datetime import datetime
from scipy.spatial.distance import cosine
from facenet_pytorch import MTCNN, InceptionResnetV1
from plyer import notification
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
SUPLANTACION_PARPADEO_UMBRAL = 1   # Diferencia de altura en ojos

# Inicialización de MTCNN
mtcnn = MTCNN(
    keep_all=True,
    device=DEVICE,
    min_face_size=40,
    thresholds=[0.6, 0.7, 0.7],
    post_process=False
)
# Y FaceNet
facenet = InceptionResnetV1(pretrained='vggface2').eval().to(DEVICE)

# Dlib para parpadeo
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("files/shape_predictor_68_face_landmarks.dat")

# Estructura para detecciones continuas
detecciones_continuas = {}


# --------------------------------------------------------------------------
# 1) inicializar_reconocimiento()
#    Carga embeddings BD y nombres empleados
# --------------------------------------------------------------------------
def inicializar_reconocimiento():
    try:
        embeddings_bd, nombres_empleados = db.get_embeddings()
        # Convertir a tensores en la device
        embeddings_bd = [torch.tensor(e).to(DEVICE) for e in embeddings_bd]
        return embeddings_bd, nombres_empleados
    except Exception as e:
        print(f"Error al inicializar reconocimiento: {e}")
        return [], []


# --------------------------------------------------------------------------
# 2) Suplantación
# --------------------------------------------------------------------------
def detectar_reflejo(rostro_bgr):
    """True si varianza Laplacian < SUPLANTACION_REFLEJO_UMBRAL."""
    gray = cv2.cvtColor(rostro_bgr, cv2.COLOR_BGR2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return lap_var < SUPLANTACION_REFLEJO_UMBRAL

def detectar_parpadeo(frame_bgr):
    """False si ojo_izq y ojo_der difieren < SUPLANTACION_PARPADEO_UMBRAL en altura."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    rostros = detector(gray, 0)
    for rostro in rostros:
        landmarks = predictor(gray, rostro)
        ojo_izq = (landmarks.part(36).x, landmarks.part(36).y)
        ojo_der = (landmarks.part(45).x, landmarks.part(45).y)
        if abs(ojo_izq[1] - ojo_der[1]) < SUPLANTACION_PARPADEO_UMBRAL:
            return False
    return True

def es_suplantacion(rostro_bgr, frame_bgr):
    """Combina reflejo + parpadeo."""
    if detectar_reflejo(rostro_bgr):
        print("[Suplantación] Reflejo anormal => posible foto/pantalla.")
        return True
    if not detectar_parpadeo(frame_bgr):
        print("[Suplantación] No se detecta parpadeo => posible foto.")
        return True
    return False


# --------------------------------------------------------------------------
# 3) Intervalos de horario
# --------------------------------------------------------------------------
def dentro_de_intervalo(hora_str, intervalo):
    """Verifica si hora_str está en el rango [intervalo[0], intervalo[1]]."""
    h1, m1 = map(int, intervalo[0].split(':'))
    h2, m2 = map(int, intervalo[1].split(':'))
    hi, mi = map(int, hora_str.split(':'))

    inicio = h1 * 60 + m1
    fin = h2 * 60 + m2
    actual = hi * 60 + mi
    return inicio <= actual <= fin

def tipo_permitido_auto():
    """Retorna 'entrada', 'salida' o None según la hora actual y los intervalos."""
    hora_str = datetime.now().strftime("%H:%M")
    if dentro_de_intervalo(hora_str, INTERVALO_ENTRADA):
        return "entrada"
    elif dentro_de_intervalo(hora_str, INTERVALO_SALIDA):
        return "salida"
    else:
        return None


# --------------------------------------------------------------------------
# 4) Funciones de embeddings
# --------------------------------------------------------------------------
def comparar_embeddings(embedding_detectado, embedding_bd):
    """
    Distancia L2 con SIMILARITY_THRESHOLD.
    Retorna (bool, dist).
    """
    e1 = embedding_detectado / np.linalg.norm(embedding_detectado)
    e2 = embedding_bd / np.linalg.norm(embedding_bd)
    dist = np.linalg.norm(e1 - e2)
    return dist < SIMILARITY_THRESHOLD, dist

def calcular_confianza(embedding_detectado, embeddings_bd):
    """
    Mide cuán cerca está embedding_detectado de la BD.
    """
    if not embeddings_bd:
        return 0.0
    distancias = []
    for emb_bd in embeddings_bd:
        emb_bd_np = emb_bd.cpu().numpy()
        emb_bd_norm = emb_bd_np / np.linalg.norm(emb_bd_np)
        dist = np.linalg.norm(embedding_detectado - emb_bd_norm)
        distancias.append(dist)
    min_dist = min(distancias)
    c = max(0, 1 - (min_dist / SIMILARITY_THRESHOLD)) * 100
    return c

def obtener_id_empleado(nombre_completo):
    """Consulta la BD por ID del empleado según 'nombre completo'."""
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
# 5) Nueva lógica: validación + registro => "primera entrada" + "última salida"
# --------------------------------------------------------------------------
def validar_registro_asistencia_especial(emp_id, nuevo_tipo):
    """
    - Sólo 1 entrada al día (si ya hubo, ignora).
    - Salida => actualiza la última si existe, si no existe => crea.
    """
    return db.validar_registro_asistencia_o(emp_id, nuevo_tipo)


def registrar_asistencia_especial(emp_id, tipo):
    """
    1 entrada => si no hay, insertar.
    1 salida => si hay, se actualiza (la última); si no, insertar.
    """
    db.registrar_asistencia_o(emp_id, tipo)


# --------------------------------------------------------------------------
# 6) Función principal => procesar_reconocimiento_facial
# --------------------------------------------------------------------------
def procesar_reconocimiento_facial(frame, embeddings_bd, nombres_empleados, tipo=None):
    """
    1) Detección de rostros con MTCNN
    2) CLAMP bounding box
    3) Suplantación
    4) Comparación embeddings
    5) Intervalo horario => registrar ENTRADA / SALIDA con la nueva lógica:
       - primera entrada del día
       - última salida del día (actualizando la salida previa)
    """
    global ultimo_procesamiento
    ahora = time.time()
    if ahora - ultimo_procesamiento < 1 / FPS_PROCESAMIENTO:
        return frame
    ultimo_procesamiento = ahora

    height, width = frame.shape[:2]
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detección
    try:
        detection_result = mtcnn.detect(frame_rgb, landmarks=False)
        if not detection_result or detection_result[0] is None:
            return frame
        boxes, probs = detection_result
    except Exception as e:
        print(f"Error detectando rostros con MTCNN: {e}")
        return frame

    for box, prob in zip(boxes, probs):
        if prob < DETECTION_CONFIDENCE:
            continue

        x1, y1, x2, y2 = map(int, box)
        # Clamp
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(x2, width)
        y2 = min(y2, height)

        if x2 <= x1 or y2 <= y1:
            continue

        # Rostro BGR
        rostro_bgr = frame[y1:y2, x1:x2]
        if rostro_bgr.size == 0:
            continue

        # Chequeo suplantación
        if es_suplantacion(rostro_bgr, frame):
            cv2.rectangle(frame, (x1,y1),(x2,y2),(0,0,255),2)
            cv2.putText(frame, "Suplantacion",(x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255),2)
            continue

        # Embedding con FaceNet
        rostro_rgb = cv2.resize(rostro_bgr, (160,160))
        rostro_tensor = torch.tensor(rostro_rgb).permute(2,0,1).unsqueeze(0).float().to(DEVICE)
        rostro_tensor = (rostro_tensor - 127.5) / 128.0

        with torch.no_grad():
            emb_detectado = facenet(rostro_tensor).cpu().numpy()[0]
        emb_detectado = emb_detectado / np.linalg.norm(emb_detectado)

        # Comparar con BD
        nombre_detectado = "Desconocido"
        emp_id = None
        min_dist = 999
        for emb_bd, nombre_bd in zip(embeddings_bd, nombres_empleados):
            emb_bd_np = emb_bd.cpu().numpy()
            es_similar, dist = comparar_embeddings(emb_detectado, emb_bd_np)
            if es_similar and dist < min_dist:
                min_dist = dist
                nombre_detectado = nombre_bd
                emp_id = obtener_id_empleado(nombre_bd)

        # Confianza
        confianza = calcular_confianza(emb_detectado, embeddings_bd) if emp_id else 0.0
        color = (0,255,0) if emp_id else (0,0,255)
        cv2.rectangle(frame, (x1,y1),(x2,y2), color, 2)
        cv2.putText(frame, f"{nombre_detectado} {confianza:.1f}%", (x1,y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color,2)

        # Registro de asistencia si corresponde
        if emp_id is not None:
            # Hora actual
            hora_str = datetime.now().strftime("%H:%M")
            print(f"[DEBUG] Hora actual: {hora_str}")

            # Determinar si 'tipo' (forzado) o autodetectar
            if tipo == "entrada":
                print(f"[DEBUG] Revisa intervalo {INTERVALO_ENTRADA} vs {hora_str}")
                if not dentro_de_intervalo(hora_str, INTERVALO_ENTRADA):
                    print("[DEBUG] Fuera de rango ENTRADA => no registra.")
                    modo = None
                else:
                    modo = "entrada"
            elif tipo == "salida":
                print(f"[DEBUG] Revisa intervalo {INTERVALO_SALIDA} vs {hora_str}")
                if not dentro_de_intervalo(hora_str, INTERVALO_SALIDA):
                    print("[DEBUG] Fuera de rango SALIDA => no registra.")
                    modo = None
                else:
                    modo = "salida"
            else:
                # Auto
                modo = tipo_permitido_auto()
                print(f"[DEBUG] tipo_permitido_auto => {modo}")

            # Lógica de detecciones continuas
            if modo:
                if emp_id not in detecciones_continuas:
                    detecciones_continuas[emp_id] = {"inicio": ahora, "registrado": False, "tipo": None}

                tiempo_detect = ahora - detecciones_continuas[emp_id]["inicio"]
                if tiempo_detect >= TIME_RECOGNITION:
                    if (not detecciones_continuas[emp_id]["registrado"]) or (detecciones_continuas[emp_id]["tipo"] != modo):
                        # Usar las nuevas funciones
                        if validar_registro_asistencia_especial(emp_id, modo):
                            registrar_asistencia_especial(emp_id, modo)
                            detecciones_continuas[emp_id]["registrado"] = True
                            detecciones_continuas[emp_id]["tipo"] = modo
                            print(f"Se registró {modo} (primera entrada / ultima salida) para {nombre_detectado}")
                            # Notificación
                            if NOTIFICATION_ASISTENCIA:
                                notification.notify(
                                    title="Registro de Asistencia",
                                    message=f"{nombre_detectado} => {modo}",
                                    app_name="FaceID System",
                                    timeout=4
                                )
                        else:
                            print(f"[DEBUG] No se puede registrar {modo} => ya hay una {modo} en el día.")

    return frame
