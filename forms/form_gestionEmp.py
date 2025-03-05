import os, sys

from utils.settings_controller import FACES_FOLDER
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QTableWidget,
    QTableWidgetItem, QHeaderView, QFileDialog, QMessageBox, QLineEdit,
    QDialog, QListWidget, QListWidgetItem, QStackedLayout, QFrame
)
from PyQt6.QtCore import Qt, QDate, QThread, pyqtSignal
from PyQt6.QtGui import QIcon, QPixmap, QGuiApplication

import shutil
import winsound

from functions.database_manager import DatabaseManager
from forms.form_addEmp import formAddEmp  # Para abrir el formAddEmp
# Si usas tu propio LoadingScreen, descomenta:
from forms.load_screen import LoadingScreen

# Importamos la lógica de embeddings
from embeddings_models.facenet_model import load_facenet_model, generate_embeddings_for_employee

# Ajusta si tu modelo está en otra ruta:
FACENET_MODEL_PATH = "files/facenet_model/20180402-114759.pb"

db = DatabaseManager()

class gestionEmp(QWidget):
    def __init__(self, db):
        super().__init__()
        self.db = db  # Guardar la instancia de la base de datos
        self.setWindowTitle("Gestión de Empleados")
        self.setStyleSheet("background-color: #2b2b2b; color: white;")
        self.setWindowState(Qt.WindowState.WindowMaximized)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        # Título
        title = QLabel("Gestión de Empleados")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-size: 24px; font-weight: bold;")
        layout.addWidget(title)

        # Barra de búsqueda y botón de agregar empleado
        search_layout = QHBoxLayout()
        self.search_bar = QLineEdit()
        self.search_bar.setPlaceholderText("Buscar empleado por nombre, cédula o área...")
        self.search_bar.setStyleSheet("padding: 5px; font-size: 14px;")
        self.search_bar.textChanged.connect(self.filter_employees)
        search_layout.addWidget(self.search_bar)

        add_button = QPushButton("Agregar Empleado")
        add_button.setStyleSheet("background-color: #4a4a4a; padding: 10px; font-size: 14px;")
        add_button.clicked.connect(self.open_add_employee_form)
        search_layout.addWidget(add_button)

        # Botón para generar reporte
        report_button = QPushButton("Generar Reporte")
        report_button.setStyleSheet("background-color: #4a4a4a; padding: 10px; font-size: 14px;")
        report_button.clicked.connect(self.generate_report)
        search_layout.addWidget(report_button)

        layout.addLayout(search_layout)

        # Tabla de empleados
        # Aumentamos el setColumnCount para la nueva columna "Regenerar Embeddings"
        self.table = QTableWidget()
        self.table.setColumnCount(14)
        self.table.setHorizontalHeaderLabels([
            "ID", "Cédula", "Apellidos", "Nombres", "Correo", "Teléfono",
            "Fecha Nacimiento", "Género", "Área", "Estado", "Horas Trabajadas",
            "Actualizar", "Activar/Desactivar", "Regenerar Embeddings"
        ])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setStyleSheet("font-size: 14px; color: black; background-color: white;")
        layout.addWidget(self.table)

        # Actualizar la tabla
        self.load_employees()

    def load_employees(self):
        empleados = db.get_empleados()
        self.table.setRowCount(len(empleados))

        columnas_no_editables = [0, 9, 10]  # ID, Estado, Horas trabajadas (ej. no editables)

        for row, empleado in enumerate(empleados):
            # empleado -> (id, ci_emp, apellidos_emp, nombres_emp, mail_emp, phone_emp, fechaN_emp, genero_emp, area_emp, estado_emp, ...)
            self.table.setItem(row, 0, QTableWidgetItem(str(empleado[0])))  # ID
            self.table.setItem(row, 1, QTableWidgetItem(str(empleado[1])))  # Cédula
            self.table.setItem(row, 2, QTableWidgetItem(str(empleado[2])))  # Apellidos
            self.table.setItem(row, 3, QTableWidgetItem(str(empleado[3])))  # Nombres
            self.table.setItem(row, 4, QTableWidgetItem(str(empleado[4])))  # Correo
            self.table.setItem(row, 5, QTableWidgetItem(str(empleado[5])))  # Teléfono

            # Fecha nacimiento en col 6
            fecha_nac_str = ""
            if empleado[6]:
                fecha_nac_str = str(empleado[6])
            self.table.setItem(row, 6, QTableWidgetItem(fecha_nac_str))

            # Género en col 7
            genero_str = "No especificado" if not empleado[7] else str(empleado[7])
            self.table.setItem(row, 7, QTableWidgetItem(genero_str))

            # Área en col 8
            area_str = "Sin asignar" if not empleado[8] else str(empleado[8])
            self.table.setItem(row, 8, QTableWidgetItem(area_str))

            # Estado (Activo=1, Inactivo=0) en col 9
            estado_str = "Activo" if empleado[9] == 1 else "Inactivo"
            self.table.setItem(row, 9, QTableWidgetItem(estado_str))

            # Calcular horas trabajadas (col 10)
            horas_trabajadas = db.calculate_hours(empleado[0])  # ID del empleado
            self.table.setItem(row, 10, QTableWidgetItem(str(horas_trabajadas)))

            # Botón para "Actualizar"
            update_button = QPushButton("Actualizar")
            update_button.setStyleSheet("background-color: #4a4a4a; padding: 5px; font-size: 14px;")
            update_button.clicked.connect(lambda checked, row=row: self.update_employee(row))
            self.table.setCellWidget(row, 11, update_button)

            # Botón para "Activar/Desactivar"
            state = "Activar" if empleado[9] == 0 else "Desactivar"
            toggle_button = QPushButton(state)
            toggle_button.setStyleSheet("background-color: red; padding: 5px; font-size: 14px;")
            toggle_button.clicked.connect(lambda checked, row=row: self.toggle_employee(row))
            self.table.setCellWidget(row, 12, toggle_button)

            # Botón para "Regenerar Embeddings"
            regen_button = QPushButton("Regenerar")
            regen_button.setStyleSheet("background-color: #4a4a4a; padding: 5px; font-size: 14px;")
            regen_button.clicked.connect(lambda checked, row=row: self.open_regen_embeddings_form(row))
            self.table.setCellWidget(row, 13, regen_button)

            # Hacer que ciertas columnas no sean editables
            for col_no_edit in columnas_no_editables:
                item = self.table.item(row, col_no_edit)
                if item:
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)

    def update_employee(self, row):
        try:
            emp_id = self.table.item(row, 0).text().strip()
            ci = self.table.item(row, 1).text().strip()
            apellidos = self.table.item(row, 2).text().strip()
            nombres = self.table.item(row, 3).text().strip()
            correo = self.table.item(row, 4).text().strip()
            telefono = self.table.item(row, 5).text().strip()
            fecha_nacimiento = self.table.item(row, 6).text().strip()
            genero = self.table.item(row, 7).text().strip()
            area = self.table.item(row, 8).text().strip()

            if not all([emp_id, apellidos, nombres, correo, telefono, fecha_nacimiento, genero, area]):
                QMessageBox.warning(self, "Error", "Todos los campos son obligatorios.")
                return

            db.update_empleado(emp_id, apellidos, nombres, correo, telefono, fecha_nacimiento, genero, area)
            QMessageBox.information(self, "Éxito", f"Empleado {nombres} {apellidos} actualizado correctamente.")
            self.load_employees()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error al actualizar empleado: {e}")

    def toggle_employee(self, row):
        try:
            emp_id = self.table.item(row, 0).text()
            current_state = self.table.item(row, 9).text()
            new_state = 1 if current_state == "Inactivo" else 0

            cursor = self.db.connection.cursor()
            cursor.execute("UPDATE empleados SET estado_emp = ? WHERE id = ?", (new_state, emp_id))
            cursor.execute("UPDATE embeddings SET estado_emb = ? WHERE emp_id = ?", (new_state, emp_id))
            self.db.connection.commit()

            self.load_employees()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"No se pudo cambiar el estado del empleado: {e}")

    def filter_employees(self):
        search_text = self.search_bar.text().lower()
        for row in range(self.table.rowCount()):
            match = False
            for col in range(self.table.columnCount() - 3):  # Las últimas columnas son botones
                item = self.table.item(row, col)
                if item and search_text in item.text().lower():
                    match = True
                    break
            self.table.setRowHidden(row, not match)

    def open_add_employee_form(self):
        self.add_employee_window = formAddEmp()
        self.add_employee_window.show()

        # Si quieres recargar la tabla al cerrar formAddEmp
        # self.add_employee_window.finished.connect(self.load_employees)
        # (o maneja señales específicas)

    def generate_report(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Guardar Reporte", "", "Archivos Excel (*.xlsx)")
        if file_path:
            try:
                result = db.generate_report_asistencias({}, file_path)
                if result:
                    QMessageBox.information(self, "Reporte", "Reporte generado exitosamente.")
                else:
                    QMessageBox.warning(self, "Reporte", "No se generó el reporte (sin datos o error).")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error al generar el reporte: {e}")

    # --------------------------------------
    #   NUEVO: Botón "Regenerar Embeddings"
    # --------------------------------------
    def open_regen_embeddings_form(self, row):
        emp_id_item = self.table.item(row, 0)
        if not emp_id_item:
            QMessageBox.warning(self, "Error", "No se encontró el ID del empleado.")
            return

        emp_id = emp_id_item.text().strip()
        if not emp_id:
            QMessageBox.warning(self, "Error", "ID de empleado inválido.")
            return

        # Abrimos el nuevo formulario de regeneración
        self.regen_window = formRegenEmp(emp_id)
        self.regen_window.show()


# -------------------------------------------------------------------------
#  CLASE DIALOGO PARA REGENERAR EMBEDDINGS
# -------------------------------------------------------------------------
class formRegenEmp(QDialog):
    def __init__(self, emp_id):
        super().__init__()
        self.emp_id = emp_id
        self.setWindowTitle(f"Regenerar Embeddings - Empleado #{emp_id}")

        screen = QGuiApplication.primaryScreen()
        screen_geometry = screen.availableGeometry()
        width = int(screen_geometry.width() * 0.5)
        height = int(screen_geometry.height() * 0.35)
        x = (screen_geometry.width() - width) // 2
        y = (screen_geometry.height() - height) // 3
        self.setGeometry(x, y, width, height)

        # Estilo
        self.setStyleSheet("""
            QWidget {
                background-color: #2b2b2b;
                color: white;
                font-family: Arial, Helvetica, sans-serif;
            }
            QLabel {
                font-size: 16px;
                color: white;
            }
            QPushButton {
                background-color: #3a3a3a;
                color: white;
                border: 1px solid #555;
                border-radius: 8px;
                padding: 8px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #555;
            }
            QListWidget {
                background-color: #2C2C3E;
                color: #FFFFFF;
            }
            QLineEdit, QComboBox, QDateEdit {
                background-color: #2C2C3E;
                color: #FFFFFF;
                border: 2px solid #4C4C6E;
                border-radius: 8px;
                padding: 5px;
                font-size: 14px;
            }
        """)

        self.main_layout = QVBoxLayout(self)
        self.setLayout(self.main_layout)

        # Mensaje de advertencia
        label_warning = QLabel(
            "Esta operación eliminará todas las fotos y embeddings anteriores del empleado.\n"
            "Luego podrás subir nuevas fotos y se generarán los embeddings actualizados."
        )
        label_warning.setStyleSheet("color: #FF8888; font-size: 15px; font-weight: bold;")
        self.main_layout.addWidget(label_warning)

        # Botón para confirmar que queremos realizar el proceso
        self.btn_confirm = QPushButton("Estoy de acuerdo. Continuar")
        self.btn_confirm.clicked.connect(self.on_confirm)
        self.main_layout.addWidget(self.btn_confirm)

        # Sección de subida de imágenes (oculta hasta que confirmemos)
        self.stacked_layout = QStackedLayout()
        self.main_layout.addLayout(self.stacked_layout)

        # Vista inicial vacía o de relleno
        empty_frame = QFrame()
        self.stacked_layout.addWidget(empty_frame)

        # Vista para subir imágenes
        self.upload_frame = QFrame()
        vbox_upload = QVBoxLayout(self.upload_frame)

        info_upload = QLabel("Selecciona las nuevas fotos para el empleado:")
        vbox_upload.addWidget(info_upload)

        self.btn_subir_foto = QPushButton("Seleccionar imágenes")
        self.btn_subir_foto.clicked.connect(self.upload_images)
        vbox_upload.addWidget(self.btn_subir_foto)

        self.image_list = QListWidget()
        self.image_list.setFixedHeight(200)
        vbox_upload.addWidget(self.image_list)

        # Botón guardar
        self.btn_guardar = QPushButton("Guardar imágenes y Generar Embeddings")
        self.btn_guardar.clicked.connect(self.start_regeneration)
        vbox_upload.addWidget(self.btn_guardar)

        self.stacked_layout.addWidget(self.upload_frame)

        # Para cargar y mostrar la pantalla de carga
        self.loading_screen = None
        self.fotos_seleccionadas = []

    def on_confirm(self):
        # Primero confirmamos con un QMessageBox
        reply = QMessageBox.question(
            self,
            "Confirmar Regeneración",
            "¿Estás seguro de que deseas eliminar fotos y embeddings previos y continuar?",
            QMessageBox.Yes | QMessageBox.Cancel
        )
        if reply == QMessageBox.Yes:
            # Eliminar embeddings y fotos anteriores
            self.delete_old_data()
            # Cambiamos la vista al "frame" de subida de imágenes
            self.stacked_layout.setCurrentWidget(self.upload_frame)
        else:
            self.close()

    def delete_old_data(self):
        """
        Elimina:
          - Los embeddings anteriores en DB (tabla embeddings).
          - Los registros de fotos en DB (tabla empleado_fotos).
          - Los archivos en la carpeta del empleado en disco.
        """
        try:
            # 1) Eliminar embeddings en la BD
            db.delete_embeddings(self.emp_id)

            # 2) Eliminar registros de fotos en la BD
            db.delete_foto_emp_ci_emp(self.emp_id)

            # 3) Eliminar carpeta y fotos antiguas (si existe)
            result = db.folder_validate(self.emp_id)
            if result and result[0]:
                ruta_carpeta_rel = result[0]
                project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
                folder_path = os.path.join(project_root, ruta_carpeta_rel)
                if os.path.exists(folder_path):
                    shutil.rmtree(folder_path, ignore_errors=True)
                    print(f"Carpeta {folder_path} eliminada.")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error al eliminar datos previos: {e}")

    def upload_images(self):
        self.fotos_seleccionadas, _ = QFileDialog.getOpenFileNames(
            self, "Seleccionar imágenes", "", "Images (*.png *.jpg *.jpeg)"
        )
        self.image_list.clear()
        if not self.fotos_seleccionadas:
            return

        for foto_path in self.fotos_seleccionadas:
            pixmap = QPixmap(foto_path).scaled(80, 80, Qt.AspectRatioMode.KeepAspectRatio)
            item = QListWidgetItem(QIcon(pixmap), os.path.basename(foto_path))
            item.setToolTip(foto_path)
            self.image_list.addItem(item)

    def start_regeneration(self):
        """
        Crea la carpeta del empleado, copia las fotos y genera los embeddings.
        """
        if not self.fotos_seleccionadas:
            QMessageBox.warning(self, "Sin imágenes", "No has seleccionado imágenes.")
            return

        # 1) Crear carpeta de empleado
        folder_path = self.create_folder(self.emp_id)
        if not folder_path:
            QMessageBox.critical(self, "Error", "No se pudo crear la carpeta del empleado.")
            return

        # 2) Copiar las fotos a la carpeta
        try:
            for foto_path in self.fotos_seleccionadas:
                shutil.copy2(foto_path, folder_path)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Ocurrió un error copiando las imágenes: {e}")
            return

        # 3) Generar embeddings con un hilo (QThread)
        self.generate_embeddings()

    def create_folder(self, emp_id):
        """
        Crea la carpeta para el empleado y actualiza empleado_fotos con su ruta relativa.
        Reutiliza la función get_name_folder(...) del DatabaseManager.
        """
        name_folder = db.get_name_folder(emp_id)
        if not name_folder:
            return None

        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        base_folder = os.path.join(project_root,  FACES_FOLDER)  # O usa FACES_FOLDER si lo tienes configurado
        folder_path = os.path.join(base_folder, name_folder)

        try:
            os.makedirs(folder_path, exist_ok=True)
        except Exception as e:
            print(f"Error al crear carpeta: {e}")
            return None

        # Guardar la ruta relativa en BD
        relative_path = os.path.relpath(folder_path, project_root)
        db.add_ruta_carpeta_emp(emp_id, relative_path)

        return folder_path

    def generate_embeddings(self):
        self.loading_screen = LoadingScreen(self, message="Generando embeddings...")
        self.loading_screen.show()
        # Carga del modelo
        try:
            self.graph = load_facenet_model(FACENET_MODEL_PATH)
        except Exception as e:
            self.loading_screen.close()
            QMessageBox.critical(self, "Error", f"Error al cargar FaceNet: {e}")
            return

        self.worker = EmbeddingWorkerRegen(self.emp_id, self.graph)
        self.worker.finished.connect(self.on_embeddings_finished)
        self.worker.error.connect(self.on_embeddings_error)
        self.worker.start()

    def on_embeddings_finished(self):
        self.loading_screen.close()
        winsound.MessageBeep(winsound.MB_OK)
        QMessageBox.information(self, "Éxito", "Embeddings regenerados correctamente.")
        self.close()

    def on_embeddings_error(self, error_message):
        self.loading_screen.close()
        QMessageBox.warning(
            self,
            "Error en Embeddings",
            f"El proceso falló: {error_message}\nNo se generaron embeddings."
        )
        # Aquí no revertimos al empleado, pues ya existía. El usuario debe reintentar o usar otras fotos.
        self.close()


# -------------------------------------------------------------------------
#   QThread para generar embeddings (similar a EmbeddingWorker de form_addEmp)
# -------------------------------------------------------------------------
class EmbeddingWorkerRegen(QThread):
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, emp_id, graph):
        super().__init__()
        self.emp_id = emp_id
        self.graph = graph

    def run(self):
        try:
            generate_embeddings_for_employee(self.emp_id, self.graph)
            self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))
