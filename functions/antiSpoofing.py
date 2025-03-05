import onnxruntime as ort
import numpy as np
import cv2

class AntiSpoofingDetector:
    def __init__(self, model_bin_path="models/AntiSpoofing_bin_1.5_128.onnx",
                 model_print_replay_path="models/AntiSpoofing_print-replay_1.5_128.onnx"):
        self.session_bin = ort.InferenceSession(model_bin_path, providers=['CPUExecutionProvider'])
        self.session_print_replay = ort.InferenceSession(model_print_replay_path, providers=['CPUExecutionProvider'])

    def preprocess_image(self, frame, bbox, target_size=(128, 128)):
        x1, y1, x2, y2 = bbox
        face = frame[y1:y2, x1:x2]
        if face.shape[0] == 0 or face.shape[1] == 0:
            return None
        face_resized = cv2.resize(face, target_size)
        face_resized = face_resized.astype(np.float32) / 255.0
        face_resized = np.transpose(face_resized, (2, 0, 1))  # Convertir a formato (C, H, W)
        return np.expand_dims(face_resized, axis=0)

    def predict(self, frame, bbox):
        face_input = self.preprocess_image(frame, bbox)
        if face_input is None:
            return False, 0.0  # No se pudo procesar el rostro

        output_bin = self.session_bin.run(None, {"input": face_input})[0]
        output_print_replay = self.session_print_replay.run(None, {"input": face_input})[0]

        score_bin = output_bin[0][0]
        score_print_replay = output_print_replay[0][0]
        score_final = (score_bin + score_print_replay) / 2

        is_real = score_final > 0.5
        return is_real, score_final
