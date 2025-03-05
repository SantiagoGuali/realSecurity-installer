import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt6.QtWidgets import QFrame, QVBoxLayout, QLabel
from PyQt6.QtCore import Qt, QProcess
from PyQt6.QtGui import QPixmap
from forms.camara_thread import CameraThread

class CameraWindow(QFrame):
    def __init__(self, camera_index, title, embeddings_bd, nombres_empleados, tipo):
        super().__init__()
        self.setStyleSheet("border: 2px solid #444; background-color: black;")
        self.camera_index = camera_index
        self.embeddings_bd = embeddings_bd
        self.nombres_empleados = nombres_empleados
        self.tipo = tipo

        self.video_label = QLabel("Cámara desactivada")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("background-color: black;")

        layout = QVBoxLayout()
        layout.addWidget(self.video_label)
        self.setLayout(layout)

        self.thread = None

    def startCamera(self):
        if self.thread is not None and self.thread.isRunning():
            print(f"La cámara {self.camera_index} ya está en ejecución.")
            return

        print(f"Iniciando cámara {self.camera_index}...")
        self.thread = CameraThread(
            camera_index=self.camera_index,
            embeddings_bd=self.embeddings_bd,
            nombres_empleados=self.nombres_empleados,
            tipo=self.tipo,
            use_mjpeg=True
        )
        self.thread.frame_ready.connect(self.update_frame)
        self.thread.start()
        self.video_label.setText("")

    def stopCamera(self):
        if self.thread is not None:
            if self.thread.isRunning():
                print(f"Deteniendo cámara {self.camera_index}...")

                self.thread.requestInterruption()
                self.thread.quit()
                # Esperar hasta 3 s
                finished = self.thread.wait(3000)
                if not finished:
                    print("⚠️ El hilo no se detuvo a tiempo. Forzando terminate()...")
                    self.thread.terminate()
                    self.thread.wait(1000)

            self.thread.deleteLater()
            self.thread = None

        self.video_label.setText("Cámara detenida")
        print("Cámara detenida correctamente.")

    def update_frame(self, q_img):
        if q_img is not None and not q_img.isNull():
            pixmap = QPixmap.fromImage(q_img)
            self.video_label.setPixmap(pixmap)
        else:
            print("Aviso: QImage es nulo/invalid.")

    def check_camera_status(self):
        if not self.thread or not self.thread.isRunning():
            print(f"Cámara {self.camera_index} se detuvo. Podrías reiniciarla...")

    def restart_application(self):
        self.stopCamera()
        QProcess.startDetached(sys.executable, sys.argv)
        sys.exit()

    def closeEvent(self, event):
        self.stopCamera()
        event.accept()
