import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QTableWidget,
    QTableWidgetItem, QHeaderView
)
from PyQt6.QtCore import Qt

# Importa tu DatabaseManager con las funciones para acceder a la BD
from functions.database_manager import DatabaseManager

db = DatabaseManager()

class formEmployeesView(QWidget):
    """
    Vista de Empleados en modo lectura:
     - Tabla con ID, Cédula, Apellidos, Nombres, Correo, Teléfono, Fecha Nacimiento, Género, Área, Estado
     - Un buscador para filtrar por texto
     - Sin botones de Agregar, Actualizar, Activar/Desactivar, ni regenerar embeddings
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Listado de Empleados (Sólo Lectura)")
        self.setStyleSheet("background-color: #2b2b2b; color: white;")
        self.setWindowState(Qt.WindowState.WindowMaximized)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        # Título
        title = QLabel("Listado de Empleados")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-size: 24px; font-weight: bold;")
        layout.addWidget(title)

        # Barra de búsqueda
        search_layout = QHBoxLayout()
        self.search_bar = QLineEdit()
        self.search_bar.setPlaceholderText("Buscar por nombre, cédula o área...")
        self.search_bar.setStyleSheet("padding: 5px; font-size: 14px;")
        self.search_bar.textChanged.connect(self.filter_employees)
        search_layout.addWidget(self.search_bar)
        layout.addLayout(search_layout)

        # Tabla de empleados (10 columnas)
        self.table = QTableWidget()
        self.table.setColumnCount(10)
        self.table.setHorizontalHeaderLabels([
            "ID", "Cédula", "Apellidos", "Nombres", "Correo", "Teléfono",
            "Fecha Nac.", "Género", "Área", "Estado"
        ])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setStyleSheet("font-size: 14px; color: black; background-color: white;")
        layout.addWidget(self.table)

        # Cargar datos
        self.load_employees()

    def load_employees(self):
        """
        Carga los empleados en la tabla, sin mostrar botones ni permitir edición.
        """
        empleados = db.get_empleados()
        self.table.setRowCount(len(empleados))

        for row, emp in enumerate(empleados):
            # emp -> (id, ci_emp, apellidos_emp, nombres_emp, mail_emp, phone_emp, fechaN_emp, genero_emp, area_emp, estado_emp)
            emp_id, ci, apellidos, nombres, correo, telefono, fecha_nac, genero, area, estado = emp

            # ID
            item_id = QTableWidgetItem(str(emp_id))
            item_id.setFlags(Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled)
            self.table.setItem(row, 0, item_id)

            # Cédula
            item_ci = QTableWidgetItem(str(ci))
            item_ci.setFlags(Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled)
            self.table.setItem(row, 1, item_ci)

            # Apellidos
            item_apellidos = QTableWidgetItem(str(apellidos))
            item_apellidos.setFlags(Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled)
            self.table.setItem(row, 2, item_apellidos)

            # Nombres
            item_nombres = QTableWidgetItem(str(nombres))
            item_nombres.setFlags(Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled)
            self.table.setItem(row, 3, item_nombres)

            # Correo
            item_correo = QTableWidgetItem(str(correo))
            item_correo.setFlags(Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled)
            self.table.setItem(row, 4, item_correo)

            # Teléfono
            item_tel = QTableWidgetItem(str(telefono))
            item_tel.setFlags(Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled)
            self.table.setItem(row, 5, item_tel)

            # Fecha Nac.
            item_fnac = QTableWidgetItem(str(fecha_nac) if fecha_nac else "")
            item_fnac.setFlags(Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled)
            self.table.setItem(row, 6, item_fnac)

            # Género
            item_gen = QTableWidgetItem(str(genero) if genero else "")
            item_gen.setFlags(Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled)
            self.table.setItem(row, 7, item_gen)

            # Área
            item_area = QTableWidgetItem(str(area) if area else "")
            item_area.setFlags(Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled)
            self.table.setItem(row, 8, item_area)

            # Estado (1=Activo, 0=Inactivo)
            estado_str = "Activo" if estado == 1 else "Inactivo"
            item_estado = QTableWidgetItem(estado_str)
            item_estado.setFlags(Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled)
            self.table.setItem(row, 9, item_estado)

    def filter_employees(self):

        search_text = self.search_bar.text().lower()
        for row in range(self.table.rowCount()):
            match = False
            # Recorremos columnas 1..9 para nombre, cedula, area, etc.
            for col in range(1, self.table.columnCount()):
                item = self.table.item(row, col)
                if item and search_text in item.text().lower():
                    match = True
                    break
            self.table.setRowHidden(row, not match)
