import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import qtawesome as qta
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QPushButton, QLabel, QWidget,
    QHBoxLayout
)

# Importamos los formularios de solo lectura
from forms.form_reports_view import formReports
from forms.form_employees_view import formEmployeesView

class EmployeeReportsForm(QMainWindow):
    """
    Ventana principal para el usuario (ROL USUARIO).
    Muestra únicamente:
      - Botón para ver empleados (solo lectura).
      - Botón para ver reportes (solo lectura).
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Panel de Usuario (Vista Empleados y Asistencias)")
        self.setGeometry(300, 100, 900, 600)

        # Estilos generales
        self.setStyleSheet("""
            QWidget {
                background-color: #1e1e2e; /* Fondo un poco más oscuro */
                color: white;
                font-family: Arial, sans-serif;
            }
            QLabel {
                font-size: 20px;
                font-weight: bold;
                color: white;
            }
            QPushButton {
                background-color: #3a3a4a;
                color: white;
                border: 1px solid #555;
                padding: 10px;
                font-size: 16px;
                text-align: left;
            }
            QPushButton:hover {
                background-color: #555;
            }
        """)

        # Ventanas secundarias
        self.window_employees = None
        self.window_reports = None

        self.init_ui()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout_principal = QVBoxLayout(central_widget)

        # Barra superior con título
        barra_superior = QHBoxLayout()
        titulo = QLabel("Panel de Usuario")
        titulo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        barra_superior.addWidget(titulo)
        layout_principal.addLayout(barra_superior)

        # Botón para ver Empleados
        btn_empleados = QPushButton("  Ver Empleados")
        btn_empleados.setIcon(qta.icon("fa5s.user", color="white"))
        btn_empleados.setIconSize(QSize(30, 30))
        btn_empleados.clicked.connect(self.ver_empleados)
        layout_principal.addWidget(btn_empleados)

        # Botón para ver Reportes (Asistencias)
        btn_reportes = QPushButton("  Ver Reportes")
        btn_reportes.setIcon(qta.icon("fa5s.chart-bar", color="white"))
        btn_reportes.setIconSize(QSize(30, 30))
        btn_reportes.clicked.connect(self.ver_reportes)
        layout_principal.addWidget(btn_reportes)

    def ver_empleados(self):
        if not self.window_employees or not self.window_employees.isVisible():
            self.window_employees = formEmployeesView()
            self.window_employees.show()
        elif self.window_employees.isMinimized():
            self.window_employees.showMaximized()

    def ver_reportes(self):
        if not self.window_reports or not self.window_reports.isVisible():
            self.window_reports = formReports()
            self.window_reports.show()
        elif self.window_reports.isMinimized():
            self.window_reports.showMaximized()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = EmployeeReportsForm()
    window.show()
    sys.exit(app.exec())
