import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import cv2
import numpy as np
from functions.database_manager import DatabaseManager
from utils.settings_controller import *
import time
import hashlib
from datetime import datetime
from colorama import Fore, Style, init
from scipy.spatial.distance import cosine
import dlib
from collections import deque
from plyer import notification
from dotenv import load_dotenv
from functions.send_grid import enviar_alerta_correo

load_dotenv()
db = DatabaseManager()

# --- Variables Globales y Umbrales ---
UMBRAL_VALIDACION_DESCONOCIDO = 0.7       # (0.1 a 1; a mayor valor, más estricto para desconocidos)
UMBRAL_REFLEJO = 10                      # Umbral para la varianza del laplaciano (imagen sin detalles)
UMBRAL_PARPADEO = 5                      # Umbral para la diferencia vertical de los ojos
UMBRAL_RECONOCIMIENTO_EMPLEADO = 0.9       # Umbral para considerar que el embedding coincide con un empleado
MAX_LEN = 30                             # Tamaño del historial para micromovimientos (frames)
THRESHOLD = 15                           # Mínima variación en píxeles para considerar movimiento

detecciones_continuas = {}
registro_desconocidos = {}
ultima_deteccion_global = 0
SAVE_IMG_D = True
TIEMPO_MINIMO_ENTRE_DETECCIONES = 3        # Segundos entre almacenar rostros desconocidos
MTCNN_MIN_FACE_SIZE = 35

# --- Configuración del dispositivo y modelos ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=True, device=device, min_face_size=40, thresholds=[0.5, 0.6, 0.7], post_process=False)
facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("files/shape_predictor_68_face_landmarks.dat")

# Historial para reconocimiento múltiple y para evaluar micromovimientos
historial_embeddings = {}
historial_posiciones = deque(maxlen=MAX_LEN)

# --- Funciones de Preprocesamiento ---
def aplicar_clahe(imagen):
    if len(imagen.shape) == 2:
        imagen = cv2.cvtColor(imagen, cv2.COLOR_GRAY2RGB)
    lab = cv2.cvtColor(imagen, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

def ajustar_brillo_contraste(imagen, alpha=1.2, beta=30):
    return cv2.convertScaleAbs(imagen, alpha=alpha, beta=beta)

def convertir_a_gris_si_es_necesario(imagen):
    brillo_medio = np.mean(imagen)
    if brillo_medio < 50 or brillo_medio > 200:
        return cv2.cvtColor(imagen, cv2.COLOR_RGB2GRAY)
    return imagen

brillo_anterior = None
brillo_medio = None
estado_luz_actual = None

def detectar_condiciones_de_luz(imagen):
    global brillo_anterior
    if imagen is None or not isinstance(imagen, np.ndarray):
        return "normal", 100
    if len(imagen.shape) == 3 and imagen.shape[2] == 3:
        imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_RGB2GRAY)
    elif len(imagen.shape) == 2:
        imagen_gris = imagen
    else:
        return "normal", 100
    brillo_medio = np.mean(imagen_gris)
    if brillo_anterior is not None and abs(brillo_medio - brillo_anterior) < 10:
        return estado_luz_actual, brillo_medio
    brillo_anterior = brillo_medio
    if brillo_medio < 50:
        return "poca_luz", brillo_medio
    elif brillo_medio > 200:
        return "mucha_luz", brillo_medio
    else:
        return "normal", brillo_medio

# --- Funciones de Validación de Fotografía y Rostro ---
def detectar_reflejo(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var < UMBRAL_REFLEJO

def detectar_parpadeo(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    rostros = detector(gray, 0)
    for rostro in rostros:
        landmarks = predictor(gray, rostro)
        ojo_izq = (landmarks.part(36).x, landmarks.part(36).y)
        ojo_der = (landmarks.part(45).x, landmarks.part(45).y)
        if abs(ojo_izq[1] - ojo_der[1]) < UMBRAL_PARPADEO:
            return False
    return True

def es_suplantacion(rostro, frame):
    # Si la imagen tiene poca información o no se detecta parpadeo, se asume foto
    if detectar_reflejo(rostro):
        print("Suplantación detectada: Reflejo anormal.")
        return True
    if not detectar_parpadeo(frame):
        print("Suplantación detectada: No se detectó parpadeo.")
        return True
    return False

def es_rostro_completo(rostro):
    try:
        gray = cv2.cvtColor(rostro, cv2.COLOR_BGR2GRAY)
    except Exception as e:
        print("Error en es_rostro_completo():", e)
        return False
    rects = detector(gray, 1)
    if len(rects) == 0:
        print("No se detectó ningún rostro en es_rostro_completo().")
        return False
    for rect in rects:
        landmarks = predictor(gray, rect)
        if landmarks.num_parts == 68:
            return True
    print("Rostro incompleto: No tiene 68 puntos faciales.")
    return False

def mejorar_resolucion(rostro):
    if rostro.shape[0] < 80 or rostro.shape[1] < 80:
        rostro = cv2.resize(rostro, (160, 160), interpolation=cv2.INTER_CUBIC)
    return rostro

def calcular_confianza(embedding_detectado, embeddings_bd):
    distancias = [1 - cosine(embedding_detectado, emb) for emb in embeddings_bd]
    confianza = max(distancias) if distancias else 0
    return confianza * 100

def comparar_embeddings(embedding1, embedding2, threshold=SIMILARITY_THRESHOLD):
    embedding1 = embedding1 / np.linalg.norm(embedding1)
    embedding2 = embedding2 / np.linalg.norm(embedding2)
    distancia = np.linalg.norm(embedding1 - embedding2)
    return distancia < threshold, distancia

def limpiar_detecciones_continuas(tiempo_actual, tiempo_limite=5):
    eliminados = []
    for emp_id, datos in detecciones_continuas.items():
        if tiempo_actual - datos["inicio"] > tiempo_limite and not datos.get("registrado", False):
            eliminados.append(emp_id)
    for emp_id in eliminados:
        del detecciones_continuas[emp_id]
        print(f"Eliminando detección inactiva para empleado {emp_id}.")

def reconocimiento_multiple(rostro_embedding, emp_id, threshold=0.70):
    key = emp_id if emp_id is not None else "unknown"
    if key not in historial_embeddings:
        historial_embeddings[key] = []
    historial_embeddings[key].append(rostro_embedding)
    if len(historial_embeddings[key]) < 2:
        return False, 0
    similitudes = [1 - cosine(rostro_embedding, e) for e in historial_embeddings[key]]
    confianza = np.mean(similitudes)
    return confianza > threshold, confianza * 100

def limpiar_registro_desconocidos():
    global registro_desconocidos
    ahora = time.time()
    TIEMPO_LIMITE_REGISTRO = 60
    claves_a_eliminar = [key for key, datos in registro_desconocidos.items() if ahora - datos["timestamp"] > TIEMPO_LIMITE_REGISTRO]
    for key in claves_a_eliminar:
        del registro_desconocidos[key]
    print(f"Se limpiaron {len(claves_a_eliminar)} registros de desconocidos.")

def validar_rostro_desconocido(embedding_detectado, embeddings_bd):
    global registro_desconocidos
    TIEMPO_LIMITE_REGISTRO = 60
    ahora = time.time()
    claves_a_eliminar = [key for key, datos in registro_desconocidos.items() if ahora - datos["timestamp"] > TIEMPO_LIMITE_REGISTRO]
    for key in claves_a_eliminar:
        del registro_desconocidos[key]
    if not registro_desconocidos:
        print("No hay registros previos de desconocidos. Registrando el primero.")
        return True
    for emb_hash, datos in registro_desconocidos.items():
        distancia = cosine(embedding_detectado, datos["embedding"])
        print(f"Comparando con desconocido previo: Distancia = {distancia}")
        if distancia < UMBRAL_VALIDACION_DESCONOCIDO:
            print("Este rostro ya fue registrado como desconocido. Ignorando.")
            return False
    return True

def ajustar_mtcnn():
    global mtcnn
    mtcnn = MTCNN(keep_all=True, device=device, min_face_size=20, thresholds=[0.4, 0.6, 0.7])

def inicializar_reconocimiento():
    try:
        embeddings_bd, nombres_empleados = db.get_embeddings()
        embeddings_bd = [torch.tensor(e).to(device) for e in embeddings_bd]
        return embeddings_bd, nombres_empleados
    except Exception as e:
        print(f"Error al inicializar reconocimiento facial: {e}")
        return [], []

def obtener_id_empleado(nombre_completo):
    try:
        cursor = db.connection.cursor()
        cursor.execute("SELECT id FROM empleados WHERE (nombres_emp + ' ' + apellidos_emp) = ?", (nombre_completo,))
        resultado = cursor.fetchone()
        cursor.close()
        return resultado[0] if resultado else None
    except Exception as e:
        print(f"Error al obtener ID del empleado: {e}")
        return None

def guardar_rostro_desconocido(rostro, embedding):
    global registro_desconocidos
    ahora = time.time()
    if rostro is None or rostro.size == 0 or len(rostro.shape) != 3:
        print("Imagen inválida en guardar_rostro_desconocido(). No se guardará.")
        return
    emb_hash = hashlib.sha256(embedding.tobytes()).hexdigest()
    if emb_hash in registro_desconocidos:
        print("Este rostro ya fue registrado recientemente. Ignorando.")
        return
    try:
        _, buffer = cv2.imencode(".jpg", rostro)
        imagen_binaria = buffer.tobytes()
        if not imagen_binaria or len(imagen_binaria) < 1000:
            print("Imagen demasiado pequeña o vacía. No se guardará.")
            return
        db.save_unknown_face(imagen_binaria)
        print("✅ Cara desconocida guardada correctamente.")
        registro_desconocidos[emb_hash] = {"embedding": embedding, "timestamp": ahora}
    except Exception as e:
        print(f"Error al guardar el rostro desconocido: {e}")

# --- Función para validar micromovimientos ---
def validar_micro_movimientos(historial_posiciones, threshold):
    if len(historial_posiciones) < 2:
        return False
    xs = [pos[0] for pos in historial_posiciones]
    ys = [pos[1] for pos in historial_posiciones]
    movimiento_x = max(xs) - min(xs)
    movimiento_y = max(ys) - min(ys)
    return (movimiento_x >= threshold) or (movimiento_y >= threshold)

# --- Variables para Procesamiento ---
FPS_PROCESAMIENTO = 10
ultimo_procesamiento = time.time()
contador_frames = 0

from concurrent.futures import ThreadPoolExecutor
thread_executor = ThreadPoolExecutor(max_workers=2)

def procesar_frame_en_hilo(frame, embeddings_bd, nombres_empleados, tipo, tiempo_minimo):
    if frame is None or frame.size == 0:
        print("Frame inválido en procesar_frame_en_hilo()")
        return
    frame_copia = frame.copy()
    thread_executor.submit(procesar_reconocimiento_facial, frame_copia, embeddings_bd, nombres_empleados, tipo, tiempo_minimo)

def es_movimiento_recto(x1, x2):
    if len(historial_posiciones) < 5:
        return False
    posiciones_x = [pos[0] for pos in historial_posiciones]
    return max(posiciones_x) - min(posiciones_x) < 10

def ajustar_mtcnn_dinamicamente():
    global mtcnn
    mtcnn = MTCNN(keep_all=True, device=device, min_face_size=30, thresholds=[0.6, 0.7, 0.8])

def procesar_reconocimiento_facial(frame, embeddings_bd, nombres_empleados, tipo, tiempo_minimo=TIME_RECOGNITION):
    global ultimo_procesamiento, registro_desconocidos, detecciones_continuas, ultima_deteccion_global, mtcnn, estado_luz_actual, contador_frames, brillo_anterior, brillo_medio
    if frame is None or frame.size == 0:
        print("Frame inválido recibido.")
        return frame
    ahora = time.time()
    FRAME_SKIP = 2 if STATE_APP else 10
    if contador_frames % FRAME_SKIP != 0:
        contador_frames += 1
        return frame
    contador_frames += 1

    # Preprocesamiento general
    frame_resized = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_rgb = ajustar_brillo_contraste(frame_rgb)
    frame_rgb = convertir_a_gris_si_es_necesario(frame_rgb)
    frame_rgb = aplicar_clahe(frame_rgb)

    condicion_luz, brillo_medio_temp = detectar_condiciones_de_luz(frame_rgb)
    if brillo_anterior is None:
        brillo_anterior = brillo_medio_temp
    brillo_medio = brillo_medio_temp
    if abs(brillo_medio - brillo_anterior) > 20:
        print(f"Ajustando MTCNN por cambio de luz: {condicion_luz}")
        if condicion_luz == "poca_luz":
            mtcnn = MTCNN(keep_all=True, device=device, min_face_size=15, thresholds=[0.3, 0.5, 0.7])
        elif condicion_luz == "mucha_luz":
            mtcnn = MTCNN(keep_all=True, device=device, min_face_size=35, thresholds=[0.6, 0.7, 0.8])
        else:
            mtcnn = MTCNN(keep_all=True, device=device, min_face_size=25, thresholds=[0.5, 0.6, 0.7])
        estado_luz_actual = condicion_luz
        brillo_anterior = brillo_medio

    if ahora - ultimo_procesamiento < 1 / FPS_PROCESAMIENTO:
        return frame
    ultimo_procesamiento = ahora

    try:
        result = mtcnn.detect(frame_rgb, landmarks=True)
        if result is None or result[0] is None:
            print("No se detectó ningún rostro en el frame.")
            return frame
        boxes, probs = result[:2] if len(result) >= 2 else (None, None)
        if boxes is None or probs is None:
            print("No se encontraron coordenadas de rostros.")
            return frame
    except Exception as e:
        print(f"Error en la detección de rostros: {e}")
        return frame

    for box, prob in zip(boxes, probs):
        if prob < DETECTION_CONFIDENCE:
            continue

        x1, y1, x2, y2 = map(int, box)
        # Actualizamos el historial de posiciones para evaluar micromovimientos
        historial_posiciones.append((x1, y1))

        # Ajuste de la caja si es muy pequeña
        if (x2 - x1) < 80 or (y2 - y1) < 80:
            centro_x, centro_y = (x1 + x2) // 2, (y1 + y2) // 2
            zoom_factor = 1.3
            x1, x2 = int(centro_x - (x2 - x1) * zoom_factor), int(centro_x + (x2 - x1) * zoom_factor)
            y1, y2 = int(centro_y - (y2 - y1) * zoom_factor), int(centro_y + (y2 - y1) * zoom_factor)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

        rostro = frame[y1:y2, x1:x2]
        if rostro is None or rostro.size == 0 or rostro.shape[0] < 20 or rostro.shape[1] < 20:
            print("Rostro inválido detectado. No se procesará.")
            continue

        # Extraer el embedding del rostro
        rostro_preparado = cv2.resize(rostro, (160, 160))
        rostro_rgb_face = cv2.cvtColor(rostro_preparado, cv2.COLOR_BGR2RGB)
        with torch.no_grad():
            rostro_tensor = torch.tensor(rostro_rgb_face).permute(2, 0, 1).unsqueeze(0).float().to(device)
            rostro_tensor = (rostro_tensor - 127.5) / 128.0
            embedding_detectado = facenet(rostro_tensor).detach().cpu().numpy()[0]
            embedding_detectado = embedding_detectado / np.linalg.norm(embedding_detectado)

        # Comparar embedding con base de datos
        emp_id = None
        nombre_detectado = "Desconocido"
        menor_distancia = float('inf')
        embeddings_bd_cpu = [e.cpu().numpy() for e in embeddings_bd]
        for embedding_bd, nombre in zip(embeddings_bd_cpu, nombres_empleados):
            es_similar, distancia = comparar_embeddings(embedding_detectado, embedding_bd)
            print(f"Comparando con {nombre}: Distancia = {distancia}")
            if es_similar and distancia < menor_distancia:
                menor_distancia = distancia
                nombre_detectado = nombre
                emp_id = obtener_id_empleado(nombre_detectado)

        print(f"Resultado final: {nombre_detectado} con distancia {menor_distancia}")

        # Si se reconoce al empleado (y la distancia es suficientemente baja), registrar asistencia
        if emp_id is not None and menor_distancia < UMBRAL_RECONOCIMIENTO_EMPLEADO:
            if db.validar_registro_asistencia(emp_id, tipo):
                if emp_id not in detecciones_continuas:
                    detecciones_continuas[emp_id] = {"inicio": ahora, "registrado": False, "tipo": None}
                tiempo_detectado = ahora - detecciones_continuas[emp_id]["inicio"]
                print(f"Empleado {nombre_detectado} detectado desde hace {tiempo_detectado:.2f} s")
                if tiempo_detectado >= tiempo_minimo:
                    if (not detecciones_continuas[emp_id]["registrado"] or 
                        detecciones_continuas[emp_id]["tipo"] != tipo):
                        db.registrar_asistencia(emp_id, tipo)
                        detecciones_continuas[emp_id]["registrado"] = True
                        detecciones_continuas[emp_id]["tipo"] = tipo
                        print(f"Asistencia registrada para {nombre_detectado} ({tipo}).")
                        if NOTIFICATION_ASISTENCIA:
                            notification.notify(
                                title="Registro de Asistencia",
                                message=f"{nombre_detectado} ha registrado su ({tipo}).",
                                app_name="Sistema de Reconocimiento Facial",
                                timeout=4
                            )
            else:
                print(f"Registro rechazado: {nombre_detectado} no puede registrar un {tipo} consecutivo.")
            continue  # Se procesa el siguiente rostro

        # Si no se reconoce, validar condiciones de "vivacidad" para descartar fotografías
        if not validar_micro_movimientos(historial_posiciones, threshold=THRESHOLD):
            print("No se detectaron micromovimientos. Posible fotografía, se descarta.")
            continue

        if not detectar_parpadeo(frame):
            print("No se detectó parpadeo. Posible fotografía, se descarta.")
            continue

        if not es_rostro_completo(rostro):
            print("Rostro incompleto. Se descarta.")
            continue

        # Si pasa las validaciones, se procesa como rostro desconocido
        if validar_rostro_desconocido(embedding_detectado, embeddings_bd) and (ahora - ultima_deteccion_global >= TIEMPO_MINIMO_ENTRE_DETECCIONES):
            print("VALIDACIÓN DE NIVEL 1 DESCONOCIDOS SUPERADA")
            limpiar_registro_desconocidos()
            emb_hash = hashlib.sha256(embedding_detectado.tobytes()).hexdigest()
            if emb_hash not in registro_desconocidos:
                print("Nuevo rostro desconocido detectado. Registrando...")
                if SAVE_IMG_D:
                    guardar_rostro_desconocido(rostro, embedding_detectado)
                registro_desconocidos[emb_hash] = {"embedding": embedding_detectado, "timestamp": ahora}
                ultima_deteccion_global = ahora
                if NOTIFICACION_INTRUSO:
                    print(f"Rostro desconocido registrado. Hash: {emb_hash}")
                    enviar_alerta_correo(rostro)
                    notification.notify(
                        title="Alerta de seguridad",
                        message="Precaución: Se ha detectado un intruso.",
                        timeout=5
                    )
            else:
                print("Este rostro ya fue registrado previamente como desconocido. Ignorando.")

        torch.cuda.empty_cache()

    return frame