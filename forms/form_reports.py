import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QTableWidget,
    QTableWidgetItem, QHeaderView, QFileDialog, QMessageBox, QLineEdit, QComboBox, QDateEdit, QTimeEdit
)
from PyQt6.QtCore import Qt
from functions.database_manager import *
from datetime import datetime

db = DatabaseManager()

class formReports(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Reportes de Asistencias")
        self.setStyleSheet("background-color: #2b2b2b; color: white;")
        self.setWindowState(Qt.WindowState.WindowMaximized)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        # Título
        title = QLabel("Listado de Asistencias")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-size: 24px; font-weight: bold;")
        layout.addWidget(title)

        # Barra de filtros
        filter_layout = QHBoxLayout()

        self.filter_day = QLineEdit()
        self.filter_day.setPlaceholderText("Filtrar por día (YYYY-MM-DD)")
        self.filter_day.setStyleSheet("padding: 5px; font-size: 14px;")
        filter_layout.addWidget(self.filter_day)

        self.filter_month = QComboBox()
        self.filter_month.addItem("Seleccionar Mes")
        for i in range(1, 13):
            self.filter_month.addItem(datetime(1900, i, 1).strftime("%B"))
        self.filter_month.setStyleSheet("padding: 5px; font-size: 14px;")
        filter_layout.addWidget(self.filter_month)

        self.filter_year = QLineEdit()
        self.filter_year.setPlaceholderText("Filtrar por año (YYYY)")
        self.filter_year.setStyleSheet("padding: 5px; font-size: 14px;")
        filter_layout.addWidget(self.filter_year)

        # Botón para aplicar filtros
        self.btn_apply_filters = QPushButton("Aplicar Filtros")
        self.btn_apply_filters.setStyleSheet("background-color: #4CAF50; padding: 10px; font-size: 14px; color: white;")
        self.btn_apply_filters.clicked.connect(self.apply_filters)
        filter_layout.addWidget(self.btn_apply_filters)

        layout.addLayout(filter_layout)

        # Tabla de asistencias
        self.table = QTableWidget()
        self.table.setColumnCount(7)
        self.table.setHorizontalHeaderLabels(["ID", "Empleado", "Fecha", "Hora", "Tipo", "Actualizar", "Eliminar"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setStyleSheet("font-size: 14px; color: black; background-color: white;")
        layout.addWidget(self.table)

        # Botón para actualizar la lista
        self.btn_refresh = QPushButton("Actualizar Lista")
        self.btn_refresh.setStyleSheet("background-color: #4a4a4a; padding: 10px; font-size: 14px;")
        self.btn_refresh.clicked.connect(self.load_asistencias)
        layout.addWidget(self.btn_refresh)

        # Botón para generar reporte
        self.btn_generate_report = QPushButton("Generar Reporte")
        self.btn_generate_report.setStyleSheet("background-color: #4a4a4a; padding: 10px; font-size: 14px;")
        self.btn_generate_report.clicked.connect(self.generate_report)
        layout.addWidget(self.btn_generate_report)

        self.load_asistencias()

    def apply_filters(self):
        """Filtra las asistencias según los valores ingresados por el usuario."""
        filters = {}

        day = self.filter_day.text().strip()
        if day:
            filters["day"] = day

        month = self.filter_month.currentIndex()
        if month > 0:
            filters["month"] = month

        year = self.filter_year.text().strip()
        if year:
            filters["year"] = year

        self.load_asistencias(filters)

    def load_asistencias(self, filters=None):
        """Carga las asistencias desde la base de datos aplicando filtros."""
        asistencias = db.get_asistencias(filters)
        self.table.setRowCount(len(asistencias))

        for row, asistencia in enumerate(asistencias):
            id_, empleado, fecha, hora, tipo = asistencia

            # ID - No editable
            item_id = QTableWidgetItem(str(id_))
            item_id.setFlags(Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled)
            self.table.setItem(row, 0, item_id)

            # Empleado - No editable
            item_empleado = QTableWidgetItem(str(empleado))
            item_empleado.setFlags(Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled)
            self.table.setItem(row, 1, item_empleado)

            # Fecha - No editable
            item_fecha = QTableWidgetItem(str(fecha))
            item_fecha.setFlags(Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled)
            self.table.setItem(row, 2, item_fecha)

            # Hora - Editable con QTimeEdit
            time_edit = QTimeEdit()
            if isinstance(hora, str):
                try:
                    hora = datetime.strptime(hora, "%H:%M:%S").time()
                except ValueError:
                    hora = datetime.now().time()
            time_edit.setTime(hora)
            self.table.setCellWidget(row, 3, time_edit)

            # Tipo - No editable
            item_tipo = QTableWidgetItem(str(tipo))
            item_tipo.setFlags(Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled)
            self.table.setItem(row, 4, item_tipo)

            # Botón para actualizar hora
            update_button = QPushButton("Actualizar")
            update_button.setStyleSheet("background-color: blue; color: white;")
            update_button.clicked.connect(lambda _, row=row: self.update_time(row))
            self.table.setCellWidget(row, 5, update_button)

            # Botón para eliminar asistencia
            delete_button = QPushButton("Eliminar")
            delete_button.setStyleSheet("background-color: red; color: white;")
            delete_button.clicked.connect(lambda _, row=row: self.delete_record(row))
            self.table.setCellWidget(row, 6, delete_button)

    def generate_report(self):
        """Genera un reporte de asistencias y lo guarda en un archivo Excel."""
        file_path, _ = QFileDialog.getSaveFileName(self, "Guardar Reporte", "", "Archivos Excel (*.xlsx)")
        if file_path:
            db.generate_report_asistencias({}, file_path)
            QMessageBox.information(self, "Éxito", "El reporte ha sido generado correctamente.")

    def update_time(self, row):
        """Actualiza la hora de la asistencia en la base de datos."""
        asis_id = int(self.table.item(row, 0).text())
        time_edit = self.table.cellWidget(row, 3)
        formatted_time = time_edit.time().toString("HH:mm:ss")

        try:
            hora_valida = datetime.strptime(formatted_time, "%H:%M:%S").time()
            db.update_asistencia_hora(asis_id, hora_valida)
            QMessageBox.information(self, "Éxito", "Hora actualizada correctamente.")
        except ValueError:
            QMessageBox.critical(self, "Error", "Formato de hora inválido.")

    def delete_record(self, row):
        """Elimina el registro de asistencia de la base de datos."""
        asis_id = int(self.table.item(row, 0).text())
        confirmation = QMessageBox.question(
            self, "Eliminar Asistencia",
            "¿Estás seguro de que quieres eliminar este registro?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        if confirmation == QMessageBox.StandardButton.Yes:
            db.delete_asistencia(asis_id)
            self.table.removeRow(row)
            QMessageBox.information(self, "Éxito", "Registro eliminado correctamente.")
