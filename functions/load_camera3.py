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
import threading
from collections import deque
from plyer import notification 

# Inicialización de la base de datos
db = DatabaseManager()

# Configuración global y de control de detecciones
detecciones_continuas = {}       # Controla las detecciones activas para cada empleado
registro_desconocidos = {}         # Registro de rostros desconocidos (para evitar múltiples capturas)
ultima_deteccion_global = 0        # Tiempo de la última detección de rostro desconocido
SAVE_IMG_D = True
TIEMPO_MINIMO_ENTRE_DETECCIONES = 5  # Segundos entre almacenamiento de rostros desconocidos
MTCNN_MIN_FACE_SIZE = 40              # Evita falsos positivos en rostros muy pequeños
PARPADEOS = 3
REFLEJO = 30
DESCONOCIDO_VALIDATE = 3  # Este umbral se revisa en la función de validación básica

# Configuración del dispositivo para torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Inicialización de modelos de detección y reconocimiento facial
mtcnn = MTCNN(keep_all=True, device=device, min_face_size=40, thresholds=[0.5, 0.6, 0.7], post_process=False)
facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Inicialización de componentes de Dlib para la detección de parpadeo
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("files/shape_predictor_68_face_landmarks.dat")

# Diccionario global para almacenar el historial de embeddings por empleado
historial_embeddings = {}

def aplicar_clahe(imagen):
    # Si la imagen es en escala de grises, convertirla a 3 canales antes de CLAHE
    if len(imagen.shape) == 2:  # Si solo tiene 1 canal (escala de grises)
        imagen = cv2.cvtColor(imagen, cv2.COLOR_GRAY2RGB)  # Convertir a 3 canales
    lab = cv2.cvtColor(imagen, cv2.COLOR_RGB2LAB)  # Convertir a LAB
    l, a, b = cv2.split(lab)  # Separar los canales
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))  # Aplicar CLAHE
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))  # Fusionar los canales de nuevo
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)  # Convertir de nuevo a RGB

def ajustar_brillo_contraste(imagen, alpha=1.2, beta=30):
    return cv2.convertScaleAbs(imagen, alpha=alpha, beta=beta)

def convertir_a_gris_si_es_necesario(imagen):
    brillo_medio = np.mean(imagen)  # Calcula el brillo medio de la imagen
    if brillo_medio < 50 or brillo_medio > 200:  # Muy oscuro o muy brillante
        return cv2.cvtColor(imagen, cv2.COLOR_RGB2GRAY)  # Convertir a escala de grises
    return imagen  # Mantener en color si la iluminación es normal

historial_posiciones = deque(maxlen=5)  # Guarda las últimas 5 posiciones del rostro

def es_movimiento_recto(x1, x2):
    if len(historial_posiciones) < 5:
        return False  # No hay suficientes datos
    posiciones_x = [pos[0] for pos in historial_posiciones]
    return max(posiciones_x) - min(posiciones_x) < 10  # Si la variación en X es pequeña, la persona va en línea recta

estado_luz_actual = None  
brillo_anterior = None
brillo_medio = None

def detectar_condiciones_de_luz(imagen):
    global brillo_anterior
    if imagen is None or not isinstance(imagen, np.ndarray):
        print("❌ Error: Imagen inválida en detectar_condiciones_de_luz().")
        return "normal", 100  # Valor por defecto de brillo
    if len(imagen.shape) == 3 and imagen.shape[2] == 3:
        imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_RGB2GRAY)  
    elif len(imagen.shape) == 2:
        imagen_gris = imagen  
    else:
        print("❌ Error: Formato de imagen no compatible.")
        return "normal", 100  
    brillo_medio = np.mean(imagen_gris)  # Calcular brillo medio
    if brillo_anterior is not None and abs(brillo_medio - brillo_anterior) < 10:
        return estado_luz_actual, brillo_medio  
    brillo_anterior = brillo_medio
    if brillo_medio < 50:
        return "poca_luz", brillo_medio
    elif brillo_medio > 200:
        return "mucha_luz", brillo_medio
    else:
        return "normal", brillo_medio

def detectar_reflejo(image):
    """
    Detecta imágenes con poco detalle (posible suplantación con fotos).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if laplacian_var < REFLEJO:  # Antes era 50, reducido a 30 para evitar falsos positivos
        return True  # Imagen con poco detalle, posible foto
    return False

def detectar_parpadeo(frame):
    """
    Verifica si hay parpadeo detectando cambios en la altura de los ojos.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)  # Mejorar contraste para detección de ojos
    rostros = detector(gray, 0)
    for rostro in rostros:
        landmarks = predictor(gray, rostro)
        ojo_izq = (landmarks.part(36).x, landmarks.part(36).y)
        ojo_der = (landmarks.part(45).x, landmarks.part(45).y)
        if abs(ojo_izq[1] - ojo_der[1]) < PARPADEOS:  # Reducir el umbral de validación
            return False  # No se detectó parpadeo
    return True  # Se detectó parpadeo correctamente

def es_suplantacion(rostro, frame):
    if detectar_reflejo(rostro):
        print("Suplantación detectada: Reflejo anormal (posible pantalla o foto impresa).")
        return True
    if not detectar_parpadeo(frame):
        print("Suplantación detectada: No se detectó parpadeo, posible intento con foto o pantalla.")
        return True
    return False  # Si pasa ambas pruebas, no es suplantación

def mejorar_resolucion(rostro):
    if rostro.shape[0] < 80 or rostro.shape[1] < 80:  # Solo mejora si el rostro es menor de 80px
        rostro = cv2.resize(rostro, (160, 160), interpolation=cv2.INTER_CUBIC)  # Escala con interpolación de alta calidad
    return rostro

def calcular_confianza(embedding_detectado, embeddings_bd):
    distancias = [1 - cosine(embedding_detectado, emb) for emb in embeddings_bd]
    confianza = max(distancias) if distancias else 0
    return confianza * 100 

def enhance_face_resolution(rostro):
    return cv2.resize(rostro, (160, 160), interpolation=cv2.INTER_CUBIC)

def get_dynamic_threshold(self):
    embeddings, _ = self.get_embeddings()
    if len(embeddings) < 5:
        return 0.75  # Valor base si hay pocos datos
    similarities = [
        1 - cosine(embeddings[i], embeddings[j])
        for i in range(len(embeddings))
        for j in range(i + 1, len(embeddings))
    ]
    return max(0.7, min(0.9, np.percentile(similarities, 90)))  # Ajusta el umbral

def comparar_embeddings(embedding1, embedding2, threshold=SIMILARITY_THRESHOLD):
    """
    Compara dos embeddings normalizados usando la distancia euclidiana.
    """
    embedding1 = embedding1 / np.linalg.norm(embedding1)
    embedding2 = embedding2 / np.linalg.norm(embedding2)
    distancia = np.linalg.norm(embedding1 - embedding2)
    return distancia < threshold, distancia

def limpiar_detecciones_continuas(tiempo_actual, tiempo_limite=5):
    """
    Elimina detecciones activas que hayan estado inactivas por más de 'tiempo_limite' segundos.
    """
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

def es_rostro_completo(rostro):
    try:
        gray = cv2.cvtColor(rostro, cv2.COLOR_BGR2GRAY)
    except Exception as e:
        print("Error convirtiendo a gris:", e)
        return False
    rects = detector(gray, 1)
    for rect in rects:
        landmarks = predictor(gray, rect)
        if landmarks.num_parts == 68:
            return True
    return False

def validar_rostro_desconocido(embedding_detectado):
    """
    Validación básica: se compara el embedding actual contra los ya registrados.
    Si la distancia coseno es menor al umbral (DESCONOCIDO_VALIDATE), se considera que ya fue detectado.
    """
    global registro_desconocidos
    for emb_hash in registro_desconocidos.keys():
        if cosine(embedding_detectado, registro_desconocidos[emb_hash]["embedding"]) < DESCONOCIDO_VALIDATE:
            return False  # El rostro ya fue detectado previamente
    return True

def validar_rostro_desconocido_mejorado(rostro, embedding_detectado, umbral_similitud=0.3, min_frames=3):
    """
    Validación mejorada de rostro desconocido que utiliza la información del embedding en múltiples frames.
    1. Verifica tamaño mínimo del rostro.
    2. Verifica que el rostro es completo (68 landmarks detectados).
    3. Acumula embeddings en un historial y solo valida si se han acumulado suficientes frames y la consistencia es alta.
    """
    # 1. Verificar tamaño mínimo del rostro
    if rostro.shape[0] < 80 or rostro.shape[1] < 80:
        print("[DEBUG] Rostro descartado: tamaño insuficiente.")
        return False
    # 2. Verificar que el rostro es completo (por ejemplo, 68 landmarks detectados)
    if not es_rostro_completo(rostro):
        print("[DEBUG] Rostro descartado: no se detectó un rostro completo.")
        return False

    key = "unknown"
    if key not in historial_embeddings:
        historial_embeddings[key] = []
    historial_embeddings[key].append(embedding_detectado)

    # Si aún no tenemos suficientes frames, no se valida
    if len(historial_embeddings[key]) < min_frames:
        print(f"[DEBUG] Esperando más frames para validar rostro desconocido ({len(historial_embeddings[key])}/{min_frames}).")
        return False

    # Calcular consistencia promedio entre el embedding actual y los almacenados
    similitudes = [1 - cosine(embedding_detectado, e) for e in historial_embeddings[key]]
    consistencia = np.mean(similitudes)
    print(f"[DEBUG] Consistencia de rostro desconocido: {consistencia:.2f}")
    # Se valida si la consistencia es superior a (1 - umbral_similitud)
    if consistencia > (1 - umbral_similitud):
        historial_embeddings[key] = []  # Reinicia el historial tras validación exitosa
        return True
    else:
        return False

def ajustar_mtcnn():
    global mtcnn
    mtcnn = MTCNN(keep_all=True, device=device, min_face_size=20, thresholds=[0.4, 0.6, 0.7])

def inicializar_reconocimiento():
    """
    Inicializa el reconocimiento facial obteniendo los embeddings y nombres de la base de datos.
    """
    try:
        embeddings_bd, nombres_empleados = db.get_embeddings()
        embeddings_bd = [torch.tensor(e).to(device) for e in embeddings_bd]
        return embeddings_bd, nombres_empleados
    except Exception as e:
        print(f"Error al inicializar reconocimiento facial: {e}")
        return [], [] 

def obtener_id_empleado(nombre_completo):
    """
    Obtiene el ID del empleado basado en su nombre completo.
    """
    try:
        cursor = db.connection.cursor()
        cursor.execute(
            "SELECT id FROM empleados WHERE (nombres_emp + ' ' + apellidos_emp) = ?", 
            (nombre_completo,)
        )
        resultado = cursor.fetchone()
        cursor.close()  
        return resultado[0] if resultado else None
    except Exception as e:
        print(f"Error al obtener ID del empleado: {e}")
        return None

def guardar_rostro_desconocido(rostro):
    try:
        _, buffer = cv2.imencode(".jpg", rostro)
        imagen_binaria = buffer.tobytes()  # Asegura que se mantiene en formato binario
        db.save_unknown_face(imagen_binaria)
        print("✅ Imagen de rostro desconocido almacenada.")
    except Exception as e:
        print(f"Error al guardar el rostro desconocido: {e}")

FPS_PROCESAMIENTO = 10
ultimo_procesamiento = time.time()
contador_frames = 0

def procesar_frame_en_hilo(frame, embeddings_bd, nombres_empleados, tipo, tiempo_minimo):
    """Ejecuta la detección y reconocimiento en un hilo separado para reducir latencia."""
    hilo = threading.Thread(target=procesar_reconocimiento_facial, args=(frame, embeddings_bd, nombres_empleados, tipo, tiempo_minimo))
    hilo.start()

def ajustar_mtcnn_dinamicamente():
    """Si la persona está en movimiento recto, reduce la precisión para mejorar la velocidad."""
    global mtcnn
    mtcnn = MTCNN(keep_all=True, device=device, min_face_size=30, thresholds=[0.5, 0.6, 0.7])

def procesar_reconocimiento_facial(frame, embeddings_bd, nombres_empleados, tipo, tiempo_minimo=TIME_RECOGNITION):
    global ultimo_procesamiento, registro_desconocidos, detecciones_continuas, ultima_deteccion_global, mtcnn, estado_luz_actual, contador_frames, brillo_anterior, brillo_medio
    ahora = time.time()
    FRAME_SKIP = 10 if STATE_APP else 5  # Reducir FPS cuando la app está minimizada

    if contador_frames % FRAME_SKIP != 0:
        contador_frames += 1
        return frame
    contador_frames += 1

    frame_resized = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))

    # Ajuste de brillo y contraste
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convertir solo una vez
    frame_rgb = ajustar_brillo_contraste(frame_rgb)
    frame_rgb = convertir_a_gris_si_es_necesario(frame_rgb)
    frame_rgb = aplicar_clahe(frame_rgb)

    # Detección de condiciones de luz y ajuste de MTCNN
    condicion_luz, brillo_medio_temp = detectar_condiciones_de_luz(frame_rgb)
    if brillo_anterior is None:
        brillo_anterior = brillo_medio_temp  # Asignar primer valor de brillo
    brillo_medio = brillo_medio_temp  # Actualizar brillo medio

    # Solo cambiar MTCNN si la luz cambia significativamente
    if abs(brillo_medio - brillo_anterior) > 20:
        print(f"⚡ Ajustando MTCNN por cambio de luz: {condicion_luz}")
        if condicion_luz == "poca_luz":
            mtcnn = MTCNN(keep_all=True, device=device, min_face_size=15, thresholds=[0.3, 0.5, 0.7])
        elif condicion_luz == "mucha_luz":
            mtcnn = MTCNN(keep_all=True, device=device, min_face_size=35, thresholds=[0.6, 0.7, 0.8])
        else:
            mtcnn = MTCNN(keep_all=True, device=device, min_face_size=25, thresholds=[0.5, 0.6, 0.7])
        estado_luz_actual = condicion_luz  
        brillo_anterior = brillo_medio

    # Evitar procesamientos innecesarios si el FPS es alto
    if ahora - ultimo_procesamiento < 1 / FPS_PROCESAMIENTO:
        return frame
    ultimo_procesamiento = ahora

    # Intentar detección de rostro
    try:
        result = mtcnn.detect(frame_rgb, landmarks=True)
        if result is None or result[0] is None:
            print(f"{Fore.YELLOW} ⚠️ No se detectó ningún rostro en el frame. {Style.RESET_ALL}")
            return frame  # No continuar si no se detectan rostros
        boxes, probs = result[:2] if len(result) >= 2 else (None, None)
        if boxes is None or probs is None:
            print(f"{Fore.YELLOW} ⚠️ No se encontraron coordenadas de rostros. {Style.RESET_ALL}")
            return frame
    except Exception as e:
        print(f"❌ Error en la detección de rostros: {e}")
        return frame

    # Procesar cada rostro detectado
    for box, prob in zip(boxes, probs):
        if prob < DETECTION_CONFIDENCE:
            continue

        x1, y1, x2, y2 = map(int, box)
        historial_posiciones.append((x1, y1))

        if es_movimiento_recto(x1, x2):
            ajustar_mtcnn_dinamicamente()

        if (x2 - x1) < 80 or (y2 - y1) < 80:
            centro_x, centro_y = (x1 + x2) // 2, (y1 + y2) // 2
            zoom_factor = 1.3  # Reducido para evitar cortar el rostro
            x1, x2 = int(centro_x - (x2 - x1) * zoom_factor), int(centro_x + (x2 - x1) * zoom_factor)
            y1, y2 = int(centro_y - (y2 - y1) * zoom_factor), int(centro_y + (y2 - y1) * zoom_factor)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

        rostro = frame[y1:y2, x1:x2]
        if rostro.shape[0] < 20 or rostro.shape[1] < 20:
            continue

        if es_suplantacion(rostro, frame):
            print(f"{Fore.RED} ❌ Rostro descartado por suplantación. {Style.RESET_ALL}")
            continue

        rostro = mejorar_resolucion(rostro)

        # Normalización y extracción del embedding
        rostro_resized = cv2.resize(rostro, (160, 160))
        rostro_rgb = cv2.cvtColor(rostro_resized, cv2.COLOR_BGR2RGB)

        with torch.no_grad():
            rostro_tensor = torch.tensor(rostro_rgb).permute(2, 0, 1).unsqueeze(0).float().to(device)
            rostro_tensor = (rostro_tensor - 127.5) / 128.0
            embedding_detectado = facenet(rostro_tensor).detach().cpu().numpy()[0]
            embedding_detectado = embedding_detectado / np.linalg.norm(embedding_detectado)

        torch.cuda.empty_cache()

        emp_id = None
        nombre_detectado = "Desconocido"
        menor_distancia = float('inf')

        for embedding_bd, nombre in zip(embeddings_bd, nombres_empleados):
            es_similar, distancia = comparar_embeddings(embedding_detectado, embedding_bd.cpu().numpy())
            if es_similar and distancia < menor_distancia:
                menor_distancia = distancia
                nombre_detectado = nombre
                emp_id = obtener_id_empleado(nombre_detectado)

        # Cálculo de confianza del reconocimiento
        confianza = calcular_confianza(embedding_detectado, embeddings_bd) if emp_id else 0.0

        # Verificar la consistencia usando reconocimiento múltiple
        es_valido, confianza = reconocimiento_multiple(embedding_detectado, emp_id)
        uncertain = not es_valido

        # Seleccionar color según el caso:
        if emp_id is not None:
            color = (0, 255, 0) if not uncertain else (0, 255, 255)  # Verde si es consistente, amarillo si no
        else:
            color = (0, 0, 255)  # Rojo para desconocido

        # Dibujar recuadro, barra de confianza y etiqueta
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        barra_x1, barra_x2 = x2 + 10, x2 + 30
        barra_y1, barra_y2 = y1, y2
        barra_altura = int((confianza / 100) * (y2 - y1))
        cv2.rectangle(frame, (barra_x1, barra_y1), (barra_x2, barra_y2), (50, 50, 50), -1)
        cv2.rectangle(frame, (barra_x1, barra_y2 - barra_altura), (barra_x2, barra_y2), color, -1)
        cv2.putText(frame, f"{nombre_detectado}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, f"{confianza:.2f}%", (barra_x1 - 5, barra_y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Registro de asistencia para empleados reconocidos
        if emp_id is not None and not uncertain:
            if db.validar_registro_asistencia(emp_id, tipo):
                if emp_id not in detecciones_continuas:
                    detecciones_continuas[emp_id] = {"inicio": ahora, "registrado": False, "tipo": None}
                tiempo_detectado = ahora - detecciones_continuas[emp_id]["inicio"]
                print(f"[DEBUG] Empleado {nombre_detectado} detectado desde hace {tiempo_detectado:.2f} s")
                if tiempo_detectado >= tiempo_minimo:
                    if (not detecciones_continuas[emp_id]["registrado"] or detecciones_continuas[emp_id]["tipo"] != tipo):
                        db.registrar_asistencia(emp_id, tipo)
                        detecciones_continuas[emp_id]["registrado"] = True
                        detecciones_continuas[emp_id]["tipo"] = tipo
                        print(f"{Fore.GREEN} ✅ Asistencia registrada para {nombre_detectado} ({tipo}). {Style.RESET_ALL}")
                        if NOTIFICATION_ASISTENCIA:
                            notification.notify(
                                title="Registro de Asistencia",
                                message=f"{nombre_detectado} ha registrado su ({tipo}).",
                                app_name="Sistema de Reconocimiento Facial",
                                timeout=4
                            )
            else:
                print(f"[DEBUG] Registro rechazado: El empleado {nombre_detectado} no puede registrar un {tipo} consecutivo.")

        # Procesamiento para rostros desconocidos
        elif emp_id is None:
            print("[DEBUG] Rostro desconocido detectado.")
            # Validar tamaño mínimo y completitud del rostro
            if rostro.shape[0] < 80 or rostro.shape[1] < 80:
                print("[DEBUG] Rostro desconocido descartado por tamaño insuficiente.")
                continue
            if not es_rostro_completo(rostro):
                print("[DEBUG] Rostro desconocido descartado por no ser completo.")
                continue

            # Validación de rostro desconocido utilizando dos métodos:
            validacion_basica = validar_rostro_desconocido(embedding_detectado)
            validacion_mejorada = validar_rostro_desconocido_mejorado(rostro, embedding_detectado, umbral_similitud=0.3, min_frames=3)
            if validacion_basica and validacion_mejorada:
                print("[DEBUG] Rostro desconocido validado como nuevo.")
                if ahora - ultima_deteccion_global >= TIEMPO_MINIMO_ENTRE_DETECCIONES:
                    if SAVE_IMG_D:
                        guardar_rostro_desconocido(rostro)
                        ultima_deteccion_global = ahora
                        emb_hash = hashlib.sha256(embedding_detectado.tobytes()).hexdigest()
                        registro_desconocidos[emb_hash] = {
                            "embedding": embedding_detectado,
                            "timestamp": ahora
                        }
                        print(f"{Fore.CYAN} [DEBUG] Rostro desconocido registrado y almacenado. Hash: {emb_hash}{Style.RESET_ALL}")
                        notification.notify(
                            title="Alerta de seguridad",
                            message="Precaución: se ha detectado un intruso.",
                            timeout=5
                        )
            else:
                print("[DEBUG] Rostro desconocido ya detectado previamente o no cumple criterios de validación.")

    torch.cuda.empty_cache()  # Liberar memoria de GPU
    return frame
