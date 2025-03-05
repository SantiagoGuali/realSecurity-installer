import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sys
import cv2
import pyodbc
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
    QTableWidget, QTableWidgetItem, QHeaderView, QFileDialog, QMessageBox, QLineEdit
)
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import Qt
from functions.database_manager import DatabaseManager

db = DatabaseManager()

class formUnknownFaces(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Gestión de Rostros Desconocidos")
        self.setStyleSheet("background-color: #2b2b2b; color: white;")
        self.setWindowState(Qt.WindowState.WindowMaximized)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        # Título
        title = QLabel("Rostros Desconocidos")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-size: 24px; font-weight: bold; color: white;")
        layout.addWidget(title)

        # Campo de filtro por fecha
        filter_layout = QHBoxLayout()
        self.filter_date = QLineEdit()
        self.filter_date.setPlaceholderText("Filtrar por fecha (YYYY-MM-DD)")
        self.filter_date.setStyleSheet("padding: 5px; font-size: 14px;")
        filter_layout.addWidget(self.filter_date)

        filter_button = QPushButton("Filtrar")
        filter_button.clicked.connect(self.apply_filter)
        filter_layout.addWidget(filter_button)

        layout.addLayout(filter_layout)

        # Tabla para mostrar rostros desconocidos
        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["ID", "Fecha", "Imagen", "Acciones"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.MultiSelection)
        self.table.setStyleSheet("font-size: 14px; color: black; background-color: white;")
        layout.addWidget(self.table)

        # Botones de CRUD
        button_layout = QHBoxLayout()

        self.btn_delete = QPushButton("Eliminar Seleccionados")
        self.btn_delete.setStyleSheet("background-color: red; padding: 10px; font-size: 14px;")
        self.btn_delete.clicked.connect(self.delete_selected)
        button_layout.addWidget(self.btn_delete)

        self.btn_refresh = QPushButton("Actualizar Lista")
        self.btn_refresh.setStyleSheet("background-color: #4a4a4a; padding: 10px; font-size: 14px;")
        self.btn_refresh.clicked.connect(self.load_faces)
        button_layout.addWidget(self.btn_refresh)

        layout.addLayout(button_layout)

        # Cargar datos iniciales
        self.load_faces()

    def load_faces(self, date_filter=None):
        """Carga los rostros desconocidos desde la base de datos."""
        self.table.setRowCount(0)
        faces = db.get_unknown_faces(date_filter)  # Obtener datos de la base de datos

        for row, (face_id, fecha, imagen) in enumerate(faces):
            self.table.insertRow(row)
            self.table.setItem(row, 0, QTableWidgetItem(str(face_id)))
            self.table.setItem(row, 1, QTableWidgetItem(str(fecha)))

            # Convertir imagen de base de datos a formato QPixmap
            pixmap = self.convert_image(imagen)
            img_label = QLabel()
            img_label.setPixmap(pixmap.scaled(100, 100, Qt.AspectRatioMode.KeepAspectRatio))
            img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setCellWidget(row, 2, img_label)

            # Botones de acción
            btn_view = QPushButton("Ver")
            btn_view.clicked.connect(lambda _, f=imagen: self.view_image(f))
            btn_download = QPushButton("Descargar")
            btn_download.clicked.connect(lambda _, i=face_id: self.download_image(i))

            action_layout = QHBoxLayout()
            action_layout.addWidget(btn_view)
            action_layout.addWidget(btn_download)
            action_widget = QWidget()
            action_widget.setLayout(action_layout)

            self.table.setCellWidget(row, 3, action_widget)

    def convert_image(self, image_data):
        """Convierte datos binarios a un QPixmap."""
        image_array = bytearray(image_data)
        pixmap = QPixmap()
        pixmap.loadFromData(image_array)
        return pixmap

    def delete_selected(self):
        """Elimina los rostros seleccionados de la base de datos."""
        selected_rows = self.table.selectionModel().selectedRows()
        if not selected_rows:
            QMessageBox.warning(self, "Advertencia", "Selecciona al menos un rostro para eliminar.")
            return

        confirmation = QMessageBox.question(
            self, "Eliminar Rostros",
            "¿Estás seguro de que quieres eliminar los rostros seleccionados?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )

        if confirmation == QMessageBox.StandardButton.Yes:
            for row in sorted(selected_rows, reverse=True):
                face_id = int(self.table.item(row.row(), 0).text())
                db.delete_unknown_face(face_id)
                self.table.removeRow(row.row())
            QMessageBox.information(self, "Éxito", "Rostros eliminados correctamente.")

    def view_image(self, image_data):
        """Muestra la imagen en una ventana emergente."""
        pixmap = self.convert_image(image_data)

        self.image_window = QWidget() 
        self.image_window.setWindowTitle("Vista de Imagen")
        self.image_window.setGeometry(100, 100, 400, 400)

        layout = QVBoxLayout(self.image_window)
        label = QLabel()
        label.setPixmap(pixmap)
        layout.addWidget(label)

        self.image_window.show() 


    def download_image(self, face_id):
        """Descarga la imagen seleccionada."""
        face_data = db.get_unknown_face_by_id(face_id)
        if face_data is None:
            QMessageBox.critical(self, "Error", "No se pudo encontrar la imagen en la base de datos.")
            return

        file_path, _ = QFileDialog.getSaveFileName(self, "Guardar Imagen", f"rostro_{face_id}.jpg", "Archivos de Imagen (*.jpg *.png)")
        if file_path:
            with open(file_path, "wb") as file:
                file.write(face_data)
            QMessageBox.information(self, "Éxito", "Imagen guardada correctamente.")

    def apply_filter(self):
        """Aplica el filtro por fecha."""
        date_filter = self.filter_date.text().strip()
        self.load_faces(date_filter)
