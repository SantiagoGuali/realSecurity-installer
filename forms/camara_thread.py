import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtGui import QImage
import cv2
import numpy as np

# Tu método de reconocimiento
from functions.openvino import procesar_reconocimiento_facial

class CameraThread(QThread):
    frame_ready = pyqtSignal(QImage)

    def __init__(self, camera_index, embeddings_bd, nombres_empleados, tipo, use_mjpeg=True):
        super().__init__()
        self.camera_index = camera_index
        self.embeddings_bd = embeddings_bd
        self.nombres_empleados = nombres_empleados
        self.tipo = tipo
        self.cap = None
        self.use_mjpeg = use_mjpeg
        self.running = True

    def run(self):
        # Determinar si es cámara local (int) o IP/RTSP/HTTP (str)
        if isinstance(self.camera_index, int):
            self.cap = cv2.VideoCapture(self.camera_index)
            if self.use_mjpeg:
                fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                success_fourcc = self.cap.set(cv2.CAP_PROP_FOURCC, fourcc)
                if success_fourcc:
                    print(f"✔ MJPEG activado en la cámara local {self.camera_index}.")
                else:
                    print(f"⚠ No se pudo activar MJPEG en la cámara local {self.camera_index}.")
        else:
            # Asumimos IP/RTSP
            self.cap = cv2.VideoCapture(self.camera_index)

        if not self.cap or not self.cap.isOpened():
            print(f"⚠️ No se pudo abrir la cámara {self.camera_index}. El hilo termina.")
            return

        # Bucle principal de lectura
        while not self.isInterruptionRequested():
            ret, frame = self.cap.read()
            if not ret:
                print(f"❌ Error en cámara {self.camera_index}. Reintentando abrir...")
                self.cap.release()
                self.cap = cv2.VideoCapture(self.camera_index)

                if self.use_mjpeg and isinstance(self.camera_index, int):
                    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                    self.cap.set(cv2.CAP_PROP_FOURCC, fourcc)

                if not self.cap.isOpened():
                    print(f"⚠️ No se pudo reabrir la cámara {self.camera_index}. Saliendo hilo.")
                    break
                continue

            # Procesamos
            frame_procesado = procesar_reconocimiento_facial(
                frame, self.embeddings_bd, self.nombres_empleados, self.tipo
            )
            if frame_procesado is None or frame_procesado.size == 0:
                continue

            # Convertir a QImage
            frame_rgb = cv2.cvtColor(frame_procesado, cv2.COLOR_BGR2RGB)
            h, w, ch = frame_rgb.shape
            bytes_per_line = ch * w
            q_img = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)

            self.frame_ready.emit(q_img)

        # Al salir del bucle, liberar
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        print(f"✅ Cámara {self.camera_index} liberada en el hilo.")

    def stop(self):
        # Llamado externo si quisieras
        self.requestInterruption()
        self.running = False
