import os
import numpy as np
import cv2
import tensorflow as tf
import torch
from tqdm import tqdm
from facenet_pytorch import MTCNN
from functions.database_manager import DatabaseManager
from sklearn.decomposition import PCA

tf.compat.v1.disable_eager_execution()

FACENET_MODEL_PATH = "files/facenet_model/20180402-114759.pb"

db = DatabaseManager()
face_detector = MTCNN()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
face_detector = MTCNN(keep_all=True, device=device)

# Umbral de “calidad” de la norma del embedding
EMBED_THRESHOLD = 0.70

def detect_faces(image):
    boxes, probs, landmarks = face_detector.detect(image, landmarks=True)
    if boxes is None:
        return []
    faces = []
    for i, box in enumerate(boxes):
        face_data = {
            "box": list(map(int, box)),
            "landmarks": landmarks[i]
        }
        faces.append(face_data)
    return faces

def align_face(image, box, landmarks):
    x, y, w, h = list(map(int, box))
    left_eye, right_eye = landmarks[0], landmarks[1]

    # Asegurar que left_eye sea el izquierdo
    if left_eye[0] > right_eye[0]:
        left_eye, right_eye = right_eye, left_eye

    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    angle = np.degrees(np.arctan2(dy, dx))

    center = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)
    rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Agregamos padding para evitar recortes
    padded_image = cv2.copyMakeBorder(image, 50, 50, 50, 50, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    aligned_image = cv2.warpAffine(padded_image, rot_matrix, (padded_image.shape[1], padded_image.shape[0]))

    # Ajustar la caja con el padding
    x, y = x + 50, y + 50
    return aligned_image[y:y+h, x:x+w]

def preprocess_image(image, required_size=(160, 160)):
    image = cv2.resize(image, required_size)
    image = image.astype("float32")
    mean, std = image.mean(), image.std()
    if std < 1e-6:  # Evita división por cero
        return None
    image = (image - mean) / std
    return np.expand_dims(image, axis=0)

def load_facenet_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"El modelo FaceNet no se encontró en la ruta: {model_path}")

    print("✅ Cargando modelo FaceNet desde archivo...")
    graph = tf.Graph()
    with graph.as_default():
        with tf.io.gfile.GFile(model_path, "rb") as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name="")
    print("✅ Modelo FaceNet cargado correctamente.")
    return graph

def generate_embeddings_for_employee(emp_id, graph):
    """
    Genera un embedding final para el empleado 'emp_id'.
    Si la norma del embedding final es menor a EMBED_THRESHOLD (0.85),
    se lanza una excepción para que NO se guarde.
    """
    result = db.folder_validate(emp_id)
    if not result or not result[0]:
        raise ValueError(f"No hay carpeta asociada para el empleado ID {emp_id}.")
    ruta_carpeta_rel = result[0]

    # Convertir ruta relativa a absoluta (si fuera necesario)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    folder_path = os.path.join(project_root, ruta_carpeta_rel)
    if not os.path.exists(folder_path):
        raise ValueError(f"La carpeta {folder_path} no existe en disco.")

    image_files = os.listdir(folder_path)
    if not image_files:
        raise ValueError(f"No hay imágenes en la carpeta {folder_path}.")

    with tf.compat.v1.Session(graph=graph) as sess:
        input_tensor = graph.get_tensor_by_name("input:0")
        embeddings_tensor = graph.get_tensor_by_name("embeddings:0")
        phase_train_tensor = graph.get_tensor_by_name("phase_train:0")

        all_embeddings = []

        for file_name in image_files:
            file_path = os.path.join(folder_path, file_name)
            image = cv2.imread(file_path)
            if image is None:
                continue

            faces = detect_faces(image)
            if len(faces) == 0:
                continue

            for face in faces:
                face_crop = align_face(image, face["box"], face["landmarks"])
                if face_crop is None or face_crop.size == 0:
                    continue

                # Probamos algunas escalas
                scales = [1.0, 0.85]  # Ejemplo de 2 escalas
                for scale in scales:
                    h_scaled = int(160 * scale)
                    w_scaled = int(160 * scale)
                    resized_face = cv2.resize(face_crop, (w_scaled, h_scaled))
                    preprocessed_face = preprocess_image(resized_face)
                    if preprocessed_face is None:
                        continue
                    feed_dict = {
                        input_tensor: preprocessed_face,
                        phase_train_tensor: False
                    }
                    embedding = sess.run(embeddings_tensor, feed_dict=feed_dict)[0]
                    # Normalizar el embedding
                    norm_val = np.linalg.norm(embedding)
                    if norm_val > 1e-6:
                        embedding = embedding / norm_val
                    all_embeddings.append(embedding)

        if not all_embeddings:
            raise ValueError("No se generó ningún embedding válido (0 rostros detectados).")

        # Calculamos un embedding final (promedio)
        final_embedding = np.mean(all_embeddings, axis=0)
        final_norm = np.linalg.norm(final_embedding)

        # Verificamos la norma con el umbral EMBED_THRESHOLD
        if final_norm < EMBED_THRESHOLD:
            # Lanzamos excepción para que form_addEmp maneje la reversión
            raise ValueError(
                f"La norma del embedding ({final_norm:.2f}) es menor a {EMBED_THRESHOLD}."
            )
        print(f"La normalizacion final es {final_norm:.2f}")

        # Si pasó el umbral, guardamos en la BD
        db.add_embedding(emp_id, final_embedding)
        print(f"✅ Embedding final para emp_id={emp_id} guardado correctamente (norma={final_norm:.2f}).")
