from PyQt6.QtWidgets import QDialog, QVBoxLayout, QLabel
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QMovie

class LoadingDialog(QDialog):
    def __init__(self, parent=None, message="Cargando..."):
        super().__init__(parent)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
        self.setModal(True)
        self.setFixedSize(150, 150)
        
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.label_spinner = QLabel(self)
        self.label_spinner.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # Asegúrate de tener el archivo "spinner.gif" en la carpeta "media"
        self.movie = QMovie("media/spinner.gif")
        self.label_spinner.setMovie(self.movie)
        self.movie.start()
        layout.addWidget(self.label_spinner)
        
        self.label_message = QLabel(message, self)
        self.label_message.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.label_message)
        
        self.setLayout(layout)
