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
from dotenv import load_dotenv
from functions.send_grid import enviar_alerta_correo

load_dotenv()

db = DatabaseManager()


detecciones_continuas = {}       # Controla las detecciones activas para cada empleado
registro_desconocidos = {}         # Registro de rostros desconocidos (para evitar múltiples capturas)
ultima_deteccion_global = 0        # Tiempo de la última detección de rostro desconocido
SAVE_IMG_D = True
TIEMPO_MINIMO_ENTRE_DETECCIONES = 2  # Segundos entre almacenamiento de rostros desconocidos
MTCNN_MIN_FACE_SIZE = 40              # Evita falsos positivos en rostros muy pequeños
STATE_APP = True
PARPADEOS = 3
REFLEJO = 30
DESCONOCIDO_VALIDATE = 3
# Configuración del dispositivo para torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Inicialización de modelos de detección y reconocimiento facial

# 🚀 Optimizar MTCNN con batch_size y device
mtcnn = MTCNN(keep_all=True, device=device, min_face_size=40, thresholds=[0.5, 0.6, 0.7], post_process=False)
facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Inicialización de componentes de Dlib para la detección de parpadeo
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("files/shape_predictor_68_face_landmarks.dat")

# Diccionario global para almacenar el historial de embeddings por empleado
historial_embeddings = {}



def aplicar_clahe(imagen):
    # 🟢 **Si la imagen es en escala de grises, convertirla a 3 canales antes de CLAHE**
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

    if laplacian_var < REFLEJO:  # Antes era 50, reducirlo a 30 para evitar falsos positivos
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
        print("❌ Error convirtiendo a gris en es_rostro_completo():", e)
        return False

    rects = detector(gray, 1)
    if len(rects) == 0:
        print("[DEBUG] ❌ No se detectó ningún rostro en es_rostro_completo().")
        return False

    for rect in rects:
        landmarks = predictor(gray, rect)
        if landmarks.num_parts == 68:
            print("[DEBUG] ✅ Se detectó un rostro completo.")
            return True

    print("[DEBUG] ❌ Rostro no tiene 68 puntos faciales. No es completo.")
    return False


def limpiar_registro_desconocidos():
    global registro_desconocidos
    ahora = time.time()
    TIEMPO_LIMITE_REGISTRO = 60  # Mantener solo los registros de los últimos 60 segundos

    claves_a_eliminar = [key for key, datos in registro_desconocidos.items() if ahora - datos["timestamp"] > TIEMPO_LIMITE_REGISTRO]
    
    for key in claves_a_eliminar:
        del registro_desconocidos[key]

    print(f"[DEBUG] Se limpiaron {len(claves_a_eliminar)} registros de rostros desconocidos.")




def validar_rostro_desconocido(embedding_detectado, embeddings_bd):
    global registro_desconocidos

    TIEMPO_LIMITE_REGISTRO = 60  # 🔹 Limpiar registros cada 60 segundos
    ahora = time.time()

    # 🔥 Limpiar registros antiguos (evita llenado de memoria)
    claves_a_eliminar = [key for key, datos in registro_desconocidos.items() if ahora - datos["timestamp"] > TIEMPO_LIMITE_REGISTRO]
    for key in claves_a_eliminar:
        del registro_desconocidos[key]

    if not registro_desconocidos:
        print("⚠️ No hay registros previos de desconocidos. Registrando el primero...")
        return True  # ✅ Permite registrar el primer rostro desconocido

    # 🔹 Comparar el nuevo rostro con los ya registrados
    for emb_hash, datos in registro_desconocidos.items():
        distancia = cosine(embedding_detectado, datos["embedding"])
        print(f"🔎 Comparando con desconocido previo: Distancia = {distancia}")

        if distancia < 0.5:  # 🔹 Si la distancia es menor, es un duplicado
            print("❌ Este rostro ya fue registrado como desconocido. Ignorando.")
            return False  

    return True  # ✅ Si no coincide con ningún desconocido previo, registrar







def es_suplantacion(rostro, frame):
    if detectar_reflejo(rostro):
        print("Suplantación detectada: Reflejo anormal (posible pantalla o foto impresa).")
        return True

    if not detectar_parpadeo(frame):
        print("Suplantación detectada: No se detectó parpadeo, posible intento con foto o pantalla.")
        return True

    return False  # Si pasa ambas pruebas, no es suplantación


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
    

def guardar_rostro_desconocido(rostro, embedding):
    global registro_desconocidos
    ahora = time.time()

    # 🔹 Verificar si la imagen es válida
    if rostro is None or rostro.size == 0 or len(rostro.shape) != 3:
        print("❌ Error: Imagen inválida en `guardar_rostro_desconocido()`. No se guardará.")
        return

    # 🔹 Generar hash único del embedding
    emb_hash = hashlib.sha256(embedding.tobytes()).hexdigest()

    # 🔹 Evitar duplicados recientes
    if emb_hash in registro_desconocidos:
        print("⚠️ Este rostro ya fue registrado recientemente. Ignorando.")
        return  

    try:
        _, buffer = cv2.imencode(".jpg", rostro)
        imagen_binaria = buffer.tobytes()

        # ✅ Evitar guardar imágenes vacías
        if not imagen_binaria or len(imagen_binaria) < 1000:
            print("❌ Imagen demasiado pequeña o vacía. No se guardará.")
            return

        db.save_unknown_face(imagen_binaria)  # 🔹 Guardar la imagen en la BD
        print("✅ Imagen de rostro desconocido almacenada.")

        # 🔹 Guardar el embedding en `registro_desconocidos`
        registro_desconocidos[emb_hash] = {
            "embedding": embedding,
            "timestamp": ahora
        }

    except Exception as e:
        print(f"❌ Error al guardar el rostro desconocido: {e}")








FPS_PROCESAMIENTO = 10
ultimo_procesamiento = time.time()




historial_posiciones = deque(maxlen=5)

from concurrent.futures import ThreadPoolExecutor

thread_executor = ThreadPoolExecutor(max_workers=2)  # 🔹 Máximo 4 hilos activos

def procesar_frame_en_hilo(frame, embeddings_bd, nombres_empleados, tipo, tiempo_minimo):
    if frame is None or frame.size == 0:
        print("❌ Error: Frame inválido recibido en procesar_frame_en_hilo()")
        return  

    frame_copia = frame.copy()
    thread_executor.submit(procesar_reconocimiento_facial, frame_copia, embeddings_bd, nombres_empleados, tipo, tiempo_minimo)



def es_movimiento_recto(x1, x2):
    """Verifica si el rostro se mueve en línea recta hacia la cámara."""
    if len(historial_posiciones) < 5:
        return False
    posiciones_x = [pos[0] for pos in historial_posiciones]
    return max(posiciones_x) - min(posiciones_x) < 10

def ajustar_mtcnn_dinamicamente():
    """Si la persona está en movimiento recto, reduce la precisión para mejorar la velocidad."""
    global mtcnn
    mtcnn = MTCNN(keep_all=True, device=device, min_face_size=30, thresholds=[0.6, 0.7, 0.8])

FPS_PROCESAMIENTO = 10
ultimo_procesamiento = time.time()


mtcnn_rapido = MTCNN(keep_all=True, device=device, min_face_size=60, thresholds=[0.7, 0.8, 0.9]) 
mtcnn_preciso = MTCNN(keep_all=True, device=device, min_face_size=30, thresholds=[0.5, 0.6, 0.7]) 
estado_luz_actual = None  
 
contador_frames = 0



def procesar_reconocimiento_facial(frame, embeddings_bd, nombres_empleados, tipo, tiempo_minimo=TIME_RECOGNITION):
    global ultimo_procesamiento, registro_desconocidos, detecciones_continuas, ultima_deteccion_global, mtcnn, estado_luz_actual, contador_frames, brillo_anterior, brillo_medio
    if frame is None or frame.size == 0:
        print("Error: frame inválido recibido.")
        return frame  # Retorna sin procesar
    
    ahora = time.time()
    FRAME_SKIP = 10 if STATE_APP else 5  # Reducir FPS cuando la app está minimizada

    if contador_frames % FRAME_SKIP != 0:
        contador_frames += 1
        return frame
    contador_frames += 1

    frame_resized = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))

    # 🔹 **Ajuste de brillo y contraste**
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convertir solo una vez
    frame_rgb = ajustar_brillo_contraste(frame_rgb)
    frame_rgb = convertir_a_gris_si_es_necesario(frame_rgb)
    frame_rgb = aplicar_clahe(frame_rgb)

    # 🔹 **Detección de condiciones de luz y ajuste de MTCNN**
    condicion_luz = detectar_condiciones_de_luz(frame_rgb)
    # 🔹 **Solo cambiar `MTCNN` si la luz cambia significativamente**

    # 🔹 **Detección de condiciones de luz y ajuste de MTCNN**
    condicion_luz, brillo_medio_temp = detectar_condiciones_de_luz(frame_rgb)

    # 🛑 **Evitar error si `brillo_anterior` o `brillo_medio` no tienen un valor inicial**
    if brillo_anterior is None:
        brillo_anterior = brillo_medio_temp  # Asignar primer valor de brillo

    brillo_medio = brillo_medio_temp  # Actualizar `brillo_medio` con el valor de la función

    # 🔹 **Solo cambiar `MTCNN` si la luz cambia significativamente**
    if abs(brillo_medio - brillo_anterior) > 20:
        print(f"⚡ Ajustando MTCNN por cambio de luz: {condicion_luz}")

        if condicion_luz == "poca_luz":
            mtcnn = MTCNN(keep_all=True, device=device, min_face_size=15, thresholds=[0.3, 0.5, 0.7])
        elif condicion_luz == "mucha_luz":
            mtcnn = MTCNN(keep_all=True, device=device, min_face_size=35, thresholds=[0.6, 0.7, 0.8])
        else:
            mtcnn = MTCNN(keep_all=True, device=device, min_face_size=25, thresholds=[0.5, 0.6, 0.7])

        estado_luz_actual = condicion_luz  
        brillo_anterior = brillo_medio  #  **Actualizar el valor correctamente**


    #  **Evitar procesamientos innecesarios si el FPS es alto**
    if ahora - ultimo_procesamiento < 1 / FPS_PROCESAMIENTO:
        return frame
    ultimo_procesamiento = ahora

    # 🏎️ **Intentar primero con detección rápida, luego con detección precisa**
    try:
        result = mtcnn.detect(frame_rgb, landmarks=True)
        if result is None or result[0] is None:
            print(f"{Fore.YELLOW} ⚠️ No se detectó ningún rostro en el frame. {Style.RESET_ALL}")
            return frame  # No continuar si no se detectan rostros

        # 🟢 Asegurar que `boxes` y `probs` no sean None antes de usarlos
        boxes, probs = result[:2] if len(result) >= 2 else (None, None)
        if boxes is None or probs is None:
            print(f"{Fore.YELLOW} ⚠️ No se encontraron coordenadas de rostros. {Style.RESET_ALL}")
            return frame  # Evita seguir procesando si no hay detección de rostros

    except Exception as e:
        print(f"❌ Error en la detección de rostros: {e}")
        return frame

    # 🔹 **Procesar cada rostro detectado**
    for box, prob in zip(boxes, probs):
        if prob < DETECTION_CONFIDENCE:
            continue

        x1, y1, x2, y2 = map(int, box)
        historial_posiciones.append((x1, y1))

        if es_movimiento_recto(x1, x2):
            ajustar_mtcnn_dinamicamente()

        if (x2 - x1) < 80 or (y2 - y1) < 80:
            centro_x, centro_y = (x1 + x2) // 2, (y1 + y2) // 2
            zoom_factor = 1.3  # Reducido de 1.5 a 1.3 para evitar cortar el rostro
            x1, x2 = int(centro_x - (x2 - x1) * zoom_factor), int(centro_x + (x2 - x1) * zoom_factor)
            y1, y2 = int(centro_y - (y2 - y1) * zoom_factor), int(centro_y + (y2 - y1) * zoom_factor)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)


        rostro = frame[y1:y2, x1:x2]

        # ✅ Verificar si la imagen es válida ANTES de procesarla
        if rostro is None or rostro.size == 0 or rostro.shape[0] < 20 or rostro.shape[1] < 20:
            print("[DEBUG] ❌ Rostro inválido detectado. No se guardará.")
            continue

        if es_suplantacion(rostro, frame):
            print(f"{Fore.RED} ❌ Rostro descartado por suplantación. {Style.RESET_ALL}")
            continue


        rostro = mejorar_resolucion(rostro)

        # 🔍 **Normalización y extracción del embedding**
        rostro_resized = cv2.resize(rostro, (160, 160))
        rostro_rgb = cv2.cvtColor(rostro_resized, cv2.COLOR_BGR2RGB)

        with torch.no_grad():
            # Conversión del rostro a tensor y normalización
            rostro_tensor = torch.tensor(rostro_rgb).permute(2, 0, 1).unsqueeze(0).float().to(device)
            rostro_tensor = (rostro_tensor - 127.5) / 128.0
            embedding_detectado = facenet(rostro_tensor).detach().cpu().numpy()[0]
            embedding_detectado = embedding_detectado / np.linalg.norm(embedding_detectado)


        embedding_detectado = embedding_detectado / np.linalg.norm(embedding_detectado)

        emp_id = None
        nombre_detectado = "Desconocido"
        menor_distancia = float('inf')

        embeddings_bd_cpu = [e.cpu().numpy() for e in embeddings_bd]  # ✅ Convertir solo una vez

        for embedding_bd, nombre in zip(embeddings_bd_cpu, nombres_empleados):
            es_similar, distancia = comparar_embeddings(embedding_detectado, embedding_bd)
            
            # 🔍 Debug para ver qué está pasando
            print(f"🔎 Comparando con {nombre}: Distancia = {distancia}")

            if es_similar and distancia < menor_distancia:
                menor_distancia = distancia
                nombre_detectado = nombre
                emp_id = obtener_id_empleado(nombre_detectado)

        print(f"🛑 Resultado final: {nombre_detectado} con distancia {menor_distancia}")


                

        # 📊 **Cálculo de confianza del reconocimiento**
        confianza = calcular_confianza(embedding_detectado, embeddings_bd) if emp_id else 0.0


        # Verificar la consistencia usando reconocimiento múltiple
        es_valido, confianza = reconocimiento_multiple(embedding_detectado, emp_id)
        # Si la detección no es consistente, marcarla como incierta
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



        elif emp_id is None:
            print("[DEBUG] Rostro desconocido detectado.")
            if rostro.shape[0] < 80 or rostro.shape[1] < 80:
                print("[DEBUG] Descarta rostro desconocido por tamaño insuficiente.")
                continue
            if not es_rostro_completo(rostro):
                print("[DEBUG] Descarta rostro desconocido por no ser completo.")
                continue



        if validar_rostro_desconocido(embedding_detectado, embeddings_bd) and (ahora - ultima_deteccion_global >= TIEMPO_MINIMO_ENTRE_DETECCIONES):
            if not es_rostro_completo(rostro):
                print("[DEBUG] Descarta rostro desconocido por no ser completo.")
                continue
            
            if es_suplantacion(rostro, frame):
                print(f"{Fore.RED} ❌ Rostro descartado por suplantación. {Style.RESET_ALL}")
                continue
            print("✅ VALIDACIÓN DE NIVEL 1 DESCONOCIDOS SUPERADA")
            limpiar_registro_desconocidos()
            # 🔹 Generar un hash único para el rostro desconocido
            emb_hash = hashlib.sha256(embedding_detectado.tobytes()).hexdigest()

            # 🔹 Verificar si ya está en `registro_desconocidos`
            if emb_hash not in registro_desconocidos:
                print(f"🔍 Nuevo rostro desconocido detectado. Registrando...")

                if SAVE_IMG_D:
                    guardar_rostro_desconocido(rostro, embedding_detectado)  # ✅ Ahora se pasa el embedding correctamente

                # 🔹 Guardar el nuevo desconocido en `registro_desconocidos`
                registro_desconocidos[emb_hash] = {
                    "embedding": embedding_detectado,
                    "timestamp": ahora
                }

                # 🔹 Actualizar el tiempo de la última detección
                ultima_deteccion_global = ahora  

                # 🔹 Enviar alerta solo si la opción está activada
                if NOTIFICACION_INTRUSO:
                    print(f"[DEBUG] Rostro desconocido registrado. Hash: {emb_hash}")
                    enviar_alerta_correo(rostro)
                    notification.notify(
                        title="Alerta de seguridad",
                        message="⚠️ Precaución: Se ha detectado un intruso.",
                        timeout=5
                    )
            else:
                print("❌ Este rostro ya fue registrado previamente como desconocido. Ignorando.")


  
        torch.cuda.empty_cache()

    return frame