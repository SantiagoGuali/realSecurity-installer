import sys
import os
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import concurrent.futures
import logging
from PyQt6.QtCore import QThread, pyqtSignal, Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QFrame, QLabel

# Ajustar si tus módulos están en carpetas específicas:
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.model_loader import *
from functions.database_manager import DatabaseManager
from utils.settings_controller import SIMILARITY_THRESHOLD
from functions.face_validator import GestorPersistenciaRostros
from embeddings_models.facenet_model import align_face, detect_faces
from utils.foto_validate import detectar_parpadeo, detectar_reflejos

# Configuraciones y parámetros
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.get_logger().setLevel("ERROR")
logging.getLogger("mediapipe").setLevel(logging.ERROR)

FRAME_SKIP = 2
FAIL_PERCENT = 10
umbral_movimiento_ojos = 0.005

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

posicion_ojos_anterior = None

# -------------------------------------------------
# Carga (global) de la BD y FaceNet (una sola vez)
# -------------------------------------------------
db = DatabaseManager()
raw_embeddings, nombres_empleados = db.get_embeddings()
embeddings_bd = [e.copy() for e in raw_embeddings if e is not None and e.shape == (512,)]

FACENET_MODEL_PATH = "files/facenet_model/20180402-114759.pb"

def load_facenet_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"El modelo FaceNet no se encontró en: {model_path}")
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

# -------------------------------------------------
# Funciones Auxiliares (idénticas a tu código)
# -------------------------------------------------
def comparar_embeddings(embedding1, embedding2, threshold=SIMILARITY_THRESHOLD):
    if embedding1.shape != (512,) or embedding2.shape != (512,):
        return False, float("inf"), 0.0

    distancia = np.linalg.norm(embedding1 - embedding2)
    similitud = np.dot(embedding1, embedding2)
    porcentaje_confianza = max(0, min(100, (1 - distancia) * 100))

    if distancia > 1.0:
        threshold += 0.2
    elif distancia > 0.7:
        threshold += 0.1

    es_similar = (distancia < threshold) or (similitud > 0.25)
    return es_similar, distancia, porcentaje_confianza

def preprocess_image(image, required_size=(160, 160)):
    if image is None or image.shape[0] < 30 or image.shape[1] < 30:
        return None

    h, w = image.shape[:2]
    if h < required_size[0] or w < required_size[1]:
        scale_factor = required_size[0] / min(h, w)
        image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)

    image = cv2.GaussianBlur(image, (3, 3), 0)
    image = cv2.resize(image, required_size)
    image = image.astype("float32")
    mean, std = image.mean(), image.std()
    if std < 1e-6:
        return None
    image = (image - mean) / std

    return np.expand_dims(image, axis=0)

def obtener_embedding_facenet(frame, bbox):
    x1, y1, x2, y2 = bbox
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
    except Exception:
        return None

def procesar_rostro(frame, x1, y1, x2, y2, embeddings_bd, nombres_empleados,
                    umbral_confianza=FAIL_PERCENT,
                    identidad_actual="Desconocido", confianza_actual=0,
                    ultimo_embedding=None, identidad_confirmada=False):
    embedding_detectado = obtener_embedding_facenet(frame, (x1, y1, x2, y2))
    if embedding_detectado is None:
        return identidad_actual, confianza_actual, ultimo_embedding, identidad_confirmada

    # Chequear cambio brusco
    if ultimo_embedding is not None:
        cambio_rostro = np.linalg.norm(embedding_detectado - ultimo_embedding) > 0.7
    else:
        cambio_rostro = True

    if cambio_rostro:
        identidad_actual = "Desconocido"
        confianza_actual = 0
        identidad_confirmada = False

    # Buscar mejor coincidencia
    nombre_detectado = "Desconocido"
    menor_distancia = float("inf")
    confianza_detectada = 0.0

    for emb_bd, nombre_bd in zip(embeddings_bd, nombres_empleados):
        _, distancia, confianza = comparar_embeddings(embedding_detectado, emb_bd)
        if distancia < menor_distancia:
            menor_distancia = distancia
            nombre_detectado = nombre_bd
            confianza_detectada = confianza

    # Si ya estaba confirmada, se mantiene
    if identidad_confirmada:
        return identidad_actual, confianza_actual, embedding_detectado, identidad_confirmada

    # Confirmar si pasa umbral
    if confianza_detectada >= umbral_confianza:
        identidad_actual = nombre_detectado
        confianza_actual = confianza_detectada
        identidad_confirmada = True
    else:
        identidad_actual = "Desconocido"
        confianza_actual = 0

    return identidad_actual, confianza_actual, embedding_detectado, identidad_confirmada

def dibujar_detalles(frame, x1, y1, x2, y2, nombre, confianza):
    color = (0, 255, 0) if nombre != "Desconocido" else (0, 0, 255)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, f"{nombre} ({confianza:.1f}%)", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    bar_x = x2 + 10
    bar_y = y1
    bar_height = (y2 - y1)
    confidence_height = int(bar_height * (confianza / 100))

    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + 20, y2), (200, 200, 200), 2)
    cv2.rectangle(frame, (bar_x, y2 - confidence_height), (bar_x + 20, y2), color, -1)

def detectar_movimiento_ojos(landmarks):
    global posicion_ojos_anterior
    left_eye_center = np.mean([[landmarks[idx].x, landmarks[idx].y] for idx in LEFT_EYE], axis=0)
    right_eye_center = np.mean([[landmarks[idx].x, landmarks[idx].y] for idx in RIGHT_EYE], axis=0)
    eye_center = (left_eye_center + right_eye_center) / 2

    if posicion_ojos_anterior is None:
        posicion_ojos_anterior = eye_center
        return False

    movimiento = np.linalg.norm(np.array(eye_center) - np.array(posicion_ojos_anterior))
    posicion_ojos_anterior = eye_center
    return movimiento > umbral_movimiento_ojos

# -------------------------------------------------
# Clase CameraThread: corre en un hilo de PyQt6
# -------------------------------------------------
class CameraThread(QThread):
    """
    Hilo que captura frames de una cámara dada, realiza el reconocimiento facial
    (incluyendo la lógica con concurrent.futures) y emite el frame final procesado.
    """
    frame_processed = pyqtSignal(QImage)  # Señal que emitimos con la imagen procesada

    def __init__(self, camera_index=0, camera_type="Entrada", parent=None):
        super().__init__(parent)
        self.camera_index = camera_index
        self.camera_type = camera_type
        self.running = True

        # MediaPipe
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True
        )
        self.face_detector = mp.solutions.face_detection.FaceDetection(
            min_detection_confidence=0.7
        )

        # Gestor de rostros (asegúrate de que retorne SOLO 2 valores (nombre, confianza))
        self.gestor_rostros = GestorPersistenciaRostros()

        # Estado de identidad global
        self.identidad_actual = "Desconocido"
        self.confianza_actual = 0
        self.ultimo_embedding = None
        self.identidad_confirmada = False

        self.frame_count = 0

        # Creamos el ThreadPoolExecutor (puedes ajustar max_workers)
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

    def run(self):
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            print(f"No se pudo abrir la cámara {self.camera_type} (índice: {self.camera_index})")
            return

        while self.running:
            ret, frame = cap.read()
            if not ret:
                continue

            self.frame_count += 1
            if self.frame_count % FRAME_SKIP != 0:
                # Saltamos frames para optimizar
                continue

            h, w, _ = frame.shape
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # FaceMesh => parpadeo, reflejos, mov. ojos
            mesh_results = self.face_mesh.process(frame_rgb)
            if mesh_results.multi_face_landmarks:
                for face_landmarks in mesh_results.multi_face_landmarks:
                    # Detección de parpadeo, reflejos y mov. ojos
                    _parpadeo = detectar_parpadeo(frame, face_landmarks)
                    _reflejo = detectar_reflejos(frame)
                    _mov_ojos = detectar_movimiento_ojos(face_landmarks.landmark)

            # FaceDetection => bounding boxes
            face_results = self.face_detector.process(frame_rgb)
            faces = face_results.detections if (face_results and face_results.detections) else []

            nuevos_rostros = {}
            future_to_bbox = {}

            # Enviamos a procesar rostros en paralelo
            for face in faces:
                bboxC = face.location_data.relative_bounding_box
                x1 = max(int(bboxC.xmin * w) - 10, 0)
                y1 = max(int(bboxC.ymin * h) - 20, 0)
                x2 = min(int((bboxC.xmin + bboxC.width) * w) + 10, w)
                y2 = min(int((bboxC.ymin + bboxC.height) * h) + 20, h)

                if (x2 - x1) < 50 or (y2 - y1) < 50:
                    continue

                centro_actual = ((x1 + x2) // 2, (y1 + y2) // 2)

                future = self.executor.submit(
                    procesar_rostro,
                    frame,
                    x1, y1, x2, y2,
                    embeddings_bd, nombres_empleados,
                    FAIL_PERCENT,
                    self.identidad_actual,
                    self.confianza_actual,
                    self.ultimo_embedding,
                    self.identidad_confirmada
                )
                future_to_bbox[future] = (centro_actual, (x1, y1, x2, y2))

            # Recogemos resultados
            for future in concurrent.futures.as_completed(future_to_bbox):
                (centro_actual, (x1, y1, x2, y2)) = future_to_bbox[future]
                try:
                    nombre_asignado, conf_asignada, emb_detectado, id_confirm = future.result()
                    nuevos_rostros[centro_actual] = (nombre_asignado, conf_asignada)

                    # Actualizamos estado (para un solo rostro principal)
                    self.identidad_actual = nombre_asignado
                    self.confianza_actual = conf_asignada
                    self.ultimo_embedding = emb_detectado
                    self.identidad_confirmada = id_confirm

                except Exception as e:
                    print(f"Error procesando embedding: {e}")

            # Estabilizar con el gestor
            rostros_filtrados = self.gestor_rostros.actualizar_rostros(nuevos_rostros)

            # Dibujamos
            for centro, (nombre_detectado, confianza_detectada) in rostros_filtrados.items():
                x1 = centro[0] - 60
                y1 = centro[1] - 70
                x2 = centro[0] + 60
                y2 = centro[1] + 70

                x1, y1 = max(x1, 0), max(y1, 0)
                x2, y2 = min(x2, w), min(y2, h)

                dibujar_detalles(frame, x1, y1, x2, y2, nombre_detectado, confianza_detectada)

            # (Opcional) Ponemos un texto que indique el tipo de cámara
            cv2.putText(frame, f"Cam: {self.camera_type}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

            # Convertir BGR->RGB->QImage para emitir al GUI
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width, ch = frame_rgb.shape
            bytes_per_line = ch * width
            qimg = QImage(frame_rgb.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
            self.frame_processed.emit(qimg)

        cap.release()

    def stop(self):
        self.running = False
        self.quit()
        self.wait()
        self.executor.shutdown(wait=False)

# -------------------------------------------------
# Interfaz PyQt6: muestra 2 QFrames con 2 cámaras
# -------------------------------------------------
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dos Cámaras con Reconocimiento Facial")

        # QFrames (o QLabels) para mostrar el video de cada cámara
        self.frameEntrada = QLabel(self)
        self.frameEntrada.setFrameShape(QFrame.Shape.Box)
        self.frameEntrada.setStyleSheet("background-color: #000; color: #FFF;")
        self.frameEntrada.setText("Cámara de Entrada")

        self.frameSalida = QLabel(self)
        self.frameSalida.setFrameShape(QFrame.Shape.Box)
        self.frameSalida.setStyleSheet("background-color: #000; color: #FFF;")
        self.frameSalida.setText("Cámara de Salida")

        layout = QVBoxLayout()
        layout.addWidget(self.frameEntrada)
        layout.addWidget(self.frameSalida)
        self.setLayout(layout)

        # Creamos los hilos de cámara
        self.thread_entrada = CameraThread(camera_index=0, camera_type="Entrada")
        self.thread_salida = CameraThread(camera_index=1, camera_type="Salida")

        # Conectamos la señal frame_processed a slots que actualizan el QFrame
        self.thread_entrada.frame_processed.connect(self.updateFrameEntrada)
        self.thread_salida.frame_processed.connect(self.updateFrameSalida)

        # Iniciar los hilos de captura
        self.thread_entrada.start()
        self.thread_salida.start()

    def updateFrameEntrada(self, qimg):
        """
        Muestra la imagen resultante en el frameEntrada.
        """
        pixmap = QPixmap.fromImage(qimg)
        self.frameEntrada.setPixmap(pixmap.scaled(
            self.frameEntrada.width(),
            self.frameEntrada.height(),
            Qt.AspectRatioMode.KeepAspectRatio
        ))

    def updateFrameSalida(self, qimg):
        """
        Muestra la imagen resultante en el frameSalida.
        """
        pixmap = QPixmap.fromImage(qimg)
        self.frameSalida.setPixmap(pixmap.scaled(
            self.frameSalida.width(),
            self.frameSalida.height(),
            Qt.AspectRatioMode.KeepAspectRatio
        ))

    def closeEvent(self, event):
        """
        Al cerrar la ventana, detenemos los hilos de cámara.
        """
        self.thread_entrada.stop()
        self.thread_salida.stop()
        super().closeEvent(event)

# -------------------------------------------------
# Lanzar la aplicación
# -------------------------------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
