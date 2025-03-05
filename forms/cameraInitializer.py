# Puedes colocar este código en el mismo archivo o en uno separado
from PyQt6.QtCore import QObject, pyqtSignal

class CameraInitializer(QObject):
    finished = pyqtSignal()  # Señal que se emite cuando la inicialización termina

    def __init__(self, entrada_widget, salida_widget):
        super().__init__()
        self.entrada_widget = entrada_widget
        self.salida_widget = salida_widget

    def run(self):
        # Se asume que startCamera() es no bloqueante, pero si tarda en iniciar, se ejecuta aquí en el hilo de trabajo.
        self.entrada_widget.startCamera()
        self.salida_widget.startCamera()
        self.finished.emit()
