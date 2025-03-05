import os
import sys
import traceback  # Para imprimir el stack trace de errores
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

INTERVALO_ENTRADA = (ENTRY_INTERVAL_START, ENTRY_INTERVAL_END)
INTERVALO_SALIDA = (EXIT_INTERVAL_START, EXIT_INTERVAL_END)

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
try:
    compiled_model_facenet = ie.compile_model(facenet_model_path, "CPU")
    input_layer_facenet = compiled_model_facenet.input(0)
    output_layer_facenet = compiled_model_facenet.output(0)
except Exception as e:
    print("[DEBUG] Error compilando el modelo FaceNet:", e)
    traceback.print_exc()

# -------------------- OpenVINO - Anti-Spoofing model ----------------------
anti_spoof_xml = "model_ir/anti_spoofing.xml"
anti_spoof_bin = "model_ir/anti_spoofing.bin"
try:
    compiled_model_anti_spoof = ie.compile_model(model=anti_spoof_xml, device_name="CPU")
    input_layer_anti_spoof = compiled_model_anti_spoof.input(0)
    output_layer_anti_spoof = compiled_model_anti_spoof.output(0)
except Exception as e:
    print("[DEBUG] Error compilando el modelo anti-spoofing:", e)
    traceback.print_exc()

# -------------------- Dlib - Parpadeo / Spoof check -----------------------
detector = dlib.get_frontal_face_detector()
try:
    predictor = dlib.shape_predictor("files/shape_predictor_68_face_landmarks.dat")
except Exception as e:
    print("[DEBUG] Error cargando el predictor de landmarks:", e)
    traceback.print_exc()

detecciones_continuas = {}

# --------------------------------------------------------------------------
# 1) inicializar_reconocimiento()
# --------------------------------------------------------------------------
def inicializar_reconocimiento():
    try:
        embeddings_bd, nombres_empleados = db.get_embeddings()
        embeddings_bd = [np.array(e, dtype=np.float32) for e in embeddings_bd]
        print("[DEBUG] Inicialización de reconocimiento completada correctamente.")
        return embeddings_bd, nombres_empleados
    except Exception as e:
        print("[DEBUG] Error en inicializar_reconocimiento:", e)
        traceback.print_exc()
        return [], []

# --------------------------------------------------------------------------
# 2) Checks tradicionales de suplantación
# --------------------------------------------------------------------------
def detectar_reflejo(rostro_bgr):
    try:
        gray = cv2.cvtColor(rostro_bgr, cv2.COLOR_BGR2GRAY)
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return lap_var < SUPLANTACION_REFLEJO_UMBRAL
    except Exception as e:
        print("[DEBUG] Error en detectar_reflejo:", e)
        traceback.print_exc()
        return True  # Por seguridad, se considera un fallo

def detectar_parpadeo(frame_bgr):
    try:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        rostros = detector(gray, 0)
        for rostro in rostros:
            landmarks = predictor(gray, rostro)
            ojo_izq = (landmarks.part(36).x, landmarks.part(36).y)
            ojo_der = (landmarks.part(45).x, landmarks.part(45).y)
            if abs(ojo_izq[1] - ojo_der[1]) < SUPLANTACION_PARPADEO_UMBRAL:
                return False
        return True
    except Exception as e:
        print("[DEBUG] Error en detectar_parpadeo:", e)
        traceback.print_exc()
        return False

# --------------------------------------------------------------------------
# 3) Detección de suplantación con Anti-Spoofing (OpenVINO)
# --------------------------------------------------------------------------
def detectar_suplantacion_anti_spoofing(rostro_bgr, threshold=0.5):
    try:
        input_size = (128, 128)
        rostro_rgb = cv2.cvtColor(rostro_bgr, cv2.COLOR_BGR2RGB)
        rostro_rgb = cv2.resize(rostro_rgb, input_size)

        blob = rostro_rgb.transpose((2, 0, 1))
        blob = np.expand_dims(blob, axis=0).astype(np.float32)

        request = compiled_model_anti_spoof.create_infer_request()
        request.infer({input_layer_anti_spoof: blob})
        output_data = request.get_output_tensor(0).data
        score = output_data[0][0]
        if score < threshold:
            return True
        return False
    except Exception as e:
        print("[DEBUG] Error en detectar_suplantacion_anti_spoofing:", e)
        traceback.print_exc()
        return True  # En caso de error, se asume suplantación para evitar falsos positivos

# --------------------------------------------------------------------------
# 4) Función combinada
# --------------------------------------------------------------------------
def es_suplantacion(rostro_bgr, frame_bgr):
    if detectar_suplantacion_anti_spoofing(rostro_bgr):
        print("[DEBUG] Anti-spoofing indica suplantación.")
        return True
    if detectar_reflejo(rostro_bgr):
        print("[DEBUG] Chequeo de reflejo indica posible foto/pantalla.")
        return True
    if not detectar_parpadeo(frame_bgr):
        print("[DEBUG] Falta detección de parpadeo; posible imagen estática.")
        return True
    return False

# --------------------------------------------------------------------------
# 5) Intervalos de horario
# --------------------------------------------------------------------------
def dentro_de_intervalo(hora_str, intervalo):
    try:
        h1, m1 = map(int, intervalo[0].split(':'))
        h2, m2 = map(int, intervalo[1].split(':'))
        hi, mi = map(int, hora_str.split(':'))
        inicio = h1 * 60 + m1
        fin = h2 * 60 + m2
        actual = hi * 60 + mi
        return inicio <= actual <= fin
    except Exception as e:
        print("[DEBUG] Error en dentro_de_intervalo:", e)
        traceback.print_exc()
        return False

def tipo_permitido_auto():
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
    try:
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        face_rgb = cv2.resize(face_rgb, (160, 160))
        blob = face_rgb.transpose((2, 0, 1))
        blob = np.expand_dims(blob, axis=0).astype(np.float32)
        blob = (blob - 127.5) / 128.0
        request = compiled_model_facenet.create_infer_request()
        request.infer({input_layer_facenet: blob})
        output_data = request.get_output_tensor(0).data
        embedding = output_data[0]
        norm_val = np.linalg.norm(embedding)
        if norm_val > 1e-6:
            embedding = embedding / norm_val
        else:
            print("[DEBUG] Norma del embedding demasiado baja:", norm_val)
        return embedding
    except Exception as e:
        print("[DEBUG] Error en get_embedding_openvino:", e)
        traceback.print_exc()
        return None

def comparar_embeddings(embedding_detectado, embedding_bd):
    try:
        dist = np.linalg.norm(embedding_detectado - embedding_bd)
        return dist < SIMILARITY_THRESHOLD, dist
    except Exception as e:
        print("[DEBUG] Error en comparar_embeddings:", e)
        traceback.print_exc()
        return False, float('inf')

def calcular_confianza(embedding_detectado, embeddings_bd):
    try:
        if not embeddings_bd:
            return 0.0
        distancias = [np.linalg.norm(embedding_detectado - emb_bd) for emb_bd in embeddings_bd]
        min_dist = min(distancias)
        c = max(0, 1 - (min_dist / SIMILARITY_THRESHOLD)) * 100
        return c
    except Exception as e:
        print("[DEBUG] Error en calcular_confianza:", e)
        traceback.print_exc()
        return 0.0

def obtener_id_empleado(nombre_completo):
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
        print("[DEBUG] Error en obtener_id_empleado:", e)
        traceback.print_exc()
        return None

# --------------------------------------------------------------------------
# 7) Lógica de asistencia con DB
# --------------------------------------------------------------------------
def validar_registro_asistencia_especial(emp_id, nuevo_tipo):
    try:
        return db.validar_registro_asistencia_vino(emp_id, nuevo_tipo)
    except Exception as e:
        print("[DEBUG] Error en validar_registro_asistencia_especial:", e)
        traceback.print_exc()
        return False

def registrar_asistencia_especial(emp_id, tipo):
    try:
        return db.registrar_asistencia_vino(emp_id, tipo)
    except Exception as e:
        print("[DEBUG] Error en registrar_asistencia_especial:", e)
        traceback.print_exc()
        return False

# --------------------------------------------------------------------------
# 8) Función principal => procesar_reconocimiento_facial
# --------------------------------------------------------------------------
def procesar_reconocimiento_facial(frame, embeddings_bd, nombres_empleados, tipo=None):
    global ultimo_procesamiento
    ahora = time.time()
    if (ahora - ultimo_procesamiento) < (1 / FPS_PROCESAMIENTO):
        return frame
    ultimo_procesamiento = ahora

    try:
        height, width = frame.shape[:2]
    except Exception as e:
        print("[DEBUG] Error obteniendo dimensiones del frame:", e)
        traceback.print_exc()
        return frame

    try:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print("[DEBUG] Error en conversión de color del frame:", e)
        traceback.print_exc()
        return frame

    # 1) Detección de rostros
    try:
        detection_result = mtcnn.detect(frame_rgb, landmarks=False)
        if not detection_result or detection_result[0] is None:
            print("[DEBUG] No se detectaron rostros en el frame.")
            return frame
        boxes, probs = detection_result
    except Exception as e:
        print("[DEBUG] Error detectando rostros con MTCNN:", e)
        traceback.print_exc()
        return frame

    # 2) Procesamiento de cada rostro detectado
    for box, prob in zip(boxes, probs):
        if prob < DETECTION_CONFIDENCE:
            continue
        try:
            x1, y1, x2, y2 = map(int, box)
            x1, y1 = max(x1, 0), max(y1, 0)
            x2, y2 = min(x2, width), min(y2, height)
            if x2 <= x1 or y2 <= y1:
                print("[DEBUG] Coordenadas inválidas para la caja del rostro:", box)
                continue
        except Exception as e:
            print("[DEBUG] Error ajustando las coordenadas de la caja:", e)
            traceback.print_exc()
            continue

        rostro_bgr = frame[y1:y2, x1:x2]
        if rostro_bgr.size == 0:
            print("[DEBUG] La región del rostro tiene tamaño 0.")
            continue

        # 2a) Chequeo de suplantación
        try:
            if es_suplantacion(rostro_bgr, frame):
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, "Suplantacion", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                continue
        except Exception as e:
            print("[DEBUG] Error en el chequeo de suplantación:", e)
            traceback.print_exc()
            continue

        # 2b) Obtener embedding y comparar con BD
        try:
            emb_detectado = get_embedding_openvino(rostro_bgr)
            if emb_detectado is None:
                print("[DEBUG] No se pudo obtener embedding para el rostro.")
                continue
        except Exception as e:
            print("[DEBUG] Error obteniendo el embedding:", e)
            traceback.print_exc()
            continue

        nombre_detectado = "Desconocido"
        emp_id_result = None
        min_dist = 999.0

        try:
            for emb_bd, nombre_bd in zip(embeddings_bd, nombres_empleados):
                es_similar, dist = comparar_embeddings(emb_detectado, emb_bd)
                if es_similar and dist < min_dist:
                    min_dist = dist
                    nombre_detectado = nombre_bd
                    emp_id_result = obtener_id_empleado(nombre_bd)
        except Exception as e:
            print("[DEBUG] Error comparando embeddings:", e)
            traceback.print_exc()

        # 2c) Calcular confianza (opcional)
        try:
            confianza = calcular_confianza(emb_detectado, embeddings_bd) if emp_id_result else 0.0
        except Exception as e:
            print("[DEBUG] Error calculando la confianza:", e)
            traceback.print_exc()
            confianza = 0.0

        # 2d) Dibujar recuadro
        color_rect = (0, 255, 0) if emp_id_result else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color_rect, 2)
        cv2.putText(frame,
                    f"{nombre_detectado} {confianza:.1f}%",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, color_rect, 2)

        # 3) Registro de asistencia
        if emp_id_result is not None:
            hora_str = datetime.now().strftime("%H:%M")
            print(f"[DEBUG] Hora actual: {hora_str}")
            try:
                if tipo == "entrada":
                    if not dentro_de_intervalo(hora_str, INTERVALO_ENTRADA):
                        print("[DEBUG] Fuera de rango ENTRADA => no registra.")
                        modo = None
                    else:
                        modo = "entrada"
                elif tipo == "salida":
                    if not dentro_de_intervalo(hora_str, INTERVALO_SALIDA):
                        print("[DEBUG] Fuera de rango SALIDA => no registra.")
                        modo = None
                    else:
                        modo = "salida"
                else:
                    modo = tipo_permitido_auto()
                    print(f"[DEBUG] Modo determinado automáticamente: {modo}")
            except Exception as e:
                print("[DEBUG] Error determinando el modo de registro:", e)
                traceback.print_exc()
                modo = None

            if modo:
                if emp_id_result not in detecciones_continuas:
                    detecciones_continuas[emp_id_result] = {"inicio": ahora}
                tiempo_detect = ahora - detecciones_continuas[emp_id_result]["inicio"]
                if tiempo_detect >= TIME_RECOGNITION:
                    try:
                        if validar_registro_asistencia_especial(emp_id_result, modo):
                            registro_ok = registrar_asistencia_especial(emp_id_result, modo)
                            if registro_ok:
                                print(f"[DEBUG] Se registró {modo} para {nombre_detectado}")
                                if NOTIFICATION_ASISTENCIA:
                                    notification.notify(
                                        title="Registro de Asistencia",
                                        message=f"{nombre_detectado} => {modo}",
                                        app_name="FaceID System",
                                        timeout=4
                                    )
                            else:
                                print("[DEBUG] Registro no efectuado (posiblemente por regla de 5 min).")
                        else:
                            print("[DEBUG] Validación en BD rechaza el registro.")
                    except Exception as e:
                        print("[DEBUG] Error durante el registro de asistencia:", e)
                        traceback.print_exc()

    return frame
