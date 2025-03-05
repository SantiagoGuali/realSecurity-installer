import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PyQt6.QtWidgets import QDialog, QVBoxLayout, QLabel, QProgressBar
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QGuiApplication

class LoadingScreen(QDialog):
    def __init__(self, parent=None, message="Generando embeddings..."):
        super().__init__(parent)

        self.setWindowTitle("Por favor, espere")
        self.setWindowModality(Qt.WindowModality.ApplicationModal)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
        self.setFixedSize(400, 120)

        self.setStyleSheet("""
            QDialog {
                background-color: rgba(30, 30, 46, 230);
                border-radius: 12px;
            }
            QLabel {
                font-size: 18px;
                font-weight: bold;
                color: #ffffff;
            }
            QProgressBar {
                border: 2px solid #444;
                border-radius: 10px;
                background-color: #2b2b3e;
                height: 15px;
            }
            QProgressBar::chunk {
                background-color: qlineargradient(
                    spread:pad, x1:0, y1:0, x2:1, y2:1,
                    stop:0 #4a90e2, stop:1 #9f70fd
                );
                border-radius: 10px;
                width: 5px;
            }
        """)

        # ===== Layout principal =====
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(20, 20, 20, 20)
        self.layout.setSpacing(10)

        # ===== Etiqueta de mensaje =====
        self.label = QLabel(message, self)
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.layout.addWidget(self.label)

        # ===== Barra de progreso =====
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setRange(0, 0)  # Indeterminada (modo infinito)
        self.progress_bar.setFixedHeight(20)
        self.layout.addWidget(self.progress_bar)

        # Centrar ventana en la pantalla
        self.center_window()

    def center_window(self):
        screen = QGuiApplication.primaryScreen()
        screen_geometry = screen.geometry()
        center_point = screen_geometry.center()

        self.move(center_point.x() - self.width() // 2, center_point.y() - self.height() // 2)
