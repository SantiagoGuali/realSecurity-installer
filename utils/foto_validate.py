import cv2
import mediapipe as mp
import numpy as np

# Inicializar el detector de malla facial de MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# Índices de los ojos en MediaPipe
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

def calcular_ear(landmarks, eye_indices):
    """ Calcula el Eye Aspect Ratio (EAR) basado en los puntos clave del ojo. """
    p1 = np.array([landmarks[eye_indices[1]].x, landmarks[eye_indices[1]].y])
    p2 = np.array([landmarks[eye_indices[5]].x, landmarks[eye_indices[5]].y])
    p3 = np.array([landmarks[eye_indices[2]].x, landmarks[eye_indices[2]].y])
    p4 = np.array([landmarks[eye_indices[4]].x, landmarks[eye_indices[4]].y])
    p5 = np.array([landmarks[eye_indices[0]].x, landmarks[eye_indices[0]].y])
    p6 = np.array([landmarks[eye_indices[3]].x, landmarks[eye_indices[3]].y])

    # Distancias euclidianas
    vertical_1 = np.linalg.norm(p2 - p4)
    vertical_2 = np.linalg.norm(p3 - p5)
    horizontal = np.linalg.norm(p1 - p6)

    # Cálculo de EAR
    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
    return ear

def detectar_parpadeo(frame, face_landmarks):
    """ Detecta si una persona parpadea y dibuja los puntos en los ojos. """
    h, w, _ = frame.shape  # Dimensiones de la imagen

    # Calcular EAR para ambos ojos
    left_ear = calcular_ear(face_landmarks.landmark, LEFT_EYE)
    right_ear = calcular_ear(face_landmarks.landmark, RIGHT_EYE)
    ear_avg = (left_ear + right_ear) / 2.0

    # Dibujar los puntos en los ojos
    for eye in [LEFT_EYE, RIGHT_EYE]:
        for idx in eye:
            x = int(face_landmarks.landmark[idx].x * w)
            y = int(face_landmarks.landmark[idx].y * h)
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)  # Puntos verdes en los ojos

    return ear_avg < 0.2  # True si hay parpadeo

def detectar_reflejos(frame, bbox=None):
    """
    Si bbox se provee (x1, y1, x2, y2),
    se extrae esa región para buscar reflejos.
    """
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        roi = frame[y1:y2, x1:x2]
    else:
        roi = frame

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY)

    white_pixels = cv2.countNonZero(thresh)
    # Ajustar umbral según sea menor ROI
    return white_pixels > 200  # o un valor menor
