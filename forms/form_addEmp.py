import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.settings_controller import FACES_FOLDER
from PyQt6.QtWidgets import (
    QApplication, QWidget, QPushButton, QVBoxLayout, QHBoxLayout,
    QSpacerItem, QSizePolicy, QLabel, QLineEdit, QComboBox, QDateEdit,
    QFormLayout, QFileDialog, QGridLayout, QFrame, QListWidget, QListWidgetItem,
    QMessageBox, QDialog, QStackedLayout
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QPixmap, QGuiApplication, QIcon

import qtawesome as qta 
import winsound
import shutil

from forms.load_screen import LoadingScreen
from functions.database_manager import DatabaseManager
from functions.validate_form import ValidadorEmpleado

# Importamos nuestra versión de FaceNet que lanza excepción si el embedding no cumple umbral
from embeddings_models.facenet_model import load_facenet_model, generate_embeddings_for_employee

from utils.settings_controller import *
FACENET_MODEL_PATH = "files/facenet_model/20180402-114759.pb"

db = DatabaseManager()
MODEL_USE = 'facenet'

class EmbeddingWorker(QThread):
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, emp_id, graph):
        super().__init__()
        self.emp_id = emp_id
        self.graph = graph

    def run(self):
        """
        Genera embeddings. Si no cumple el umbral, lanza excepción ValueError,
        que se capturará en on_embeddings_error de formAddEmp.
        """
        try:
            generate_embeddings_for_employee(self.emp_id, self.graph)
            self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))


class formAddEmp(QDialog):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Agregar Nuevo Empleado")
        self.folder_path = None
        self.emp_id = None
        self.fotos_seleccionadas = []

        screen = QGuiApplication.primaryScreen()
        screen_geometry = screen.availableGeometry()
        width = int(screen_geometry.width() * 0.8)
        height = int(screen_geometry.height() * 0.0001)
        x = (screen_geometry.width() - width) // 2
        y = int(screen_geometry.height() * 0.001)
        self.setGeometry(x, y, width, height)

        # -----------------------------------------------------------------
        # 1) Desactivar botón de cierre (X) y maximizar, dejando minimizar
        # -----------------------------------------------------------------
        # Tomamos los flags actuales:
        current_flags = self.windowFlags()
        # Quitamos el botón Close:
        current_flags = current_flags & ~Qt.WindowType.WindowCloseButtonHint
        # Quitamos el botón Maximize:
        current_flags = current_flags & ~Qt.WindowType.WindowMaximizeButtonHint
        # Dejamos Minimizar activo (o lo añadimos explícitamente):
        current_flags = current_flags | Qt.WindowType.WindowMinimizeButtonHint
        # Finalmente lo aplicamos:
        self.setWindowFlags(current_flags)

   
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)

        self.stacked_layout = QStackedLayout()
        self.main_layout.addLayout(self.stacked_layout)

        self.form_view = QFrame()
        self.form_view.setLayout(self.create_form())
        self.stacked_layout.addWidget(self.form_view)

        self.image_upload_view = QFrame()
        self.image_upload_view.setLayout(self.create_image_upload_view())
        self.stacked_layout.addWidget(self.image_upload_view)

        self.setLayout(self.main_layout)

    def create_form(self):
        layout = QGridLayout()
        form_layout = QFormLayout()
        form_layout.setSpacing(70)

        self.input_ci = QLineEdit()
        self.input_ci.setPlaceholderText("CI")
        self.input_apellidos = QLineEdit()
        self.input_apellidos.setPlaceholderText("Apellidos")
        self.input_nombres = QLineEdit()
        self.input_nombres.setPlaceholderText("Nombres")
        self.input_correo = QLineEdit()
        self.input_correo.setPlaceholderText("example@hotmail.com")
        self.input_telf = QLineEdit()
        self.input_telf.setPlaceholderText("0912345678")

        self.fecha_nacimiento = QDateEdit()
        self.fecha_nacimiento.setCalendarPopup(True)
        self.fecha_nacimiento.setDisplayFormat("dd/MM/yyyy")

        self.combo_genero = QComboBox()
        self.combo_genero.addItems(["Masculino", "Femenino"])

        self.combo_area = QComboBox()
        self.combo_area.addItems(["Cocina", "Restaurante", "Poli funcional"])

        # Botones
        button_layout = self.create_buttons()

        form_layout.addRow("Número de cédula", self.input_ci)
        form_layout.addRow("Apellidos", self.input_apellidos)
        form_layout.addRow("Nombres", self.input_nombres)
        form_layout.addRow("Correo", self.input_correo)
        form_layout.addRow("Teléfono", self.input_telf)
        form_layout.addRow("Fecha de nacimiento", self.fecha_nacimiento)
        form_layout.addRow("Género", self.combo_genero)
        form_layout.addRow("Área de trabajo", self.combo_area)

        layout.addLayout(form_layout, 0, 0)
        layout.addLayout(button_layout, 1, 0, 1, Qt.AlignmentFlag.AlignLeft)
        return layout

    def create_buttons(self):
        self.btn_agregar = QPushButton("Guardar Empleado")
        self.btn_agregar.clicked.connect(self.guardar_empleado)
        # El botón "Cancelar" cierra la ventana y revierte si es necesario
        self.btn_eliminar = QPushButton("Cancelar")
        self.btn_eliminar.clicked.connect(self.cancelar_operacion)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.btn_eliminar)
        button_layout.addWidget(self.btn_agregar)
        return button_layout

    def create_image_upload_view(self):
        layout = QVBoxLayout()
        label_info = QLabel("Sube las imágenes del empleado:")
        label_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(label_info)

        self.btn_subir_foto = QPushButton("Seleccionar imágenes")
        self.btn_subir_foto.clicked.connect(self.upload_and_save_images)
        layout.addWidget(self.btn_subir_foto)

        self.image_list = QListWidget()
        self.image_list.setFixedHeight(400)
        layout.addWidget(self.image_list)

        # Botón “Cancelar” en la segunda vista
        self.btn_abort = QPushButton("Cancelar")
        self.btn_abort.clicked.connect(self.cancelar_operacion)
        layout.addWidget(self.btn_abort)

        self.btn_guardar_fotos = QPushButton("Guardar imágenes")
        self.btn_guardar_fotos.clicked.connect(self.guardar_imagenes_empleado)
        layout.addWidget(self.btn_guardar_fotos)

        return layout

    def guardar_empleado(self):
        ci = self.input_ci.text().strip()
        apellidos = self.input_apellidos.text().strip()
        nombres = self.input_nombres.text().strip()
        correo = self.input_correo.text().strip()
        telf = self.input_telf.text().strip()
        fecha_nac = self.fecha_nacimiento.date().toString("yyyy-MM-dd")
        genero = self.combo_genero.currentText()
        area = self.combo_area.currentText()

        # Validación con tu validador
        valido, mensaje = ValidadorEmpleado.validar_campos(ci, apellidos, nombres, correo, telf, fecha_nac)
        if not valido:
            QMessageBox.warning(self, "Validación de Datos", mensaje)
            return

        try:
            db.create_empleado(ci, apellidos, nombres, correo, telf, fecha_nac, genero, area)
            self.emp_id = db.get_end_empleado()
            self.folder_path = self.create_folder(self.emp_id)
            if not self.folder_path:
                raise Exception("No se pudo crear la carpeta del empleado.")
            QMessageBox.information(self, "Éxito", "Empleado guardado correctamente.")
            self.stacked_layout.setCurrentWidget(self.image_upload_view)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error al guardar el empleado: {e}")
            self.revert_changes()

    def create_folder(self, id_empleado):
        name_folder = db.get_name_folder(id_empleado)
        if name_folder:
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            base_folder = os.path.join(project_root, FACES_FOLDER)
            folder_path = os.path.join(base_folder, name_folder)
            if not os.path.exists(folder_path):
                try:
                    os.makedirs(folder_path)
                    print(f"Carpeta creada: {folder_path}")
                except Exception as e:
                    print(f"Error al crear carpeta: {e}")
                    return None
            # Guardar la ruta relativa en BD
            relative_path = os.path.relpath(folder_path, project_root)
            db.add_ruta_carpeta_emp(id_empleado, relative_path)
            return folder_path
        else:
            print(f"No se encontró un nombre de carpeta para ID={id_empleado}")
            return None

    def upload_and_save_images(self):
        self.fotos_seleccionadas, _ = QFileDialog.getOpenFileNames(
            self, "Seleccionar imágenes", "", "Images (*.png *.xpm *.jpg *.jpeg)"
        )
        self.image_list.clear()
        if not self.fotos_seleccionadas:
            QMessageBox.warning(self, "Sin selección", "No se seleccionaron imágenes.")
            return

        for foto_path in self.fotos_seleccionadas:
            pixmap = QPixmap(foto_path).scaled(100, 100, Qt.AspectRatioMode.KeepAspectRatio)
            item = QListWidgetItem(QIcon(pixmap), os.path.basename(foto_path))
            item.setToolTip(foto_path)
            self.image_list.addItem(item)

    def guardar_imagenes_empleado(self):
        if not self.fotos_seleccionadas:
            QMessageBox.warning(self, "Sin imágenes", "No se seleccionaron imágenes para guardar.")
            return
        if not self.folder_path:
            QMessageBox.critical(self, "Error", "La carpeta del empleado no existe.")
            return
        try:
            for foto_path in self.fotos_seleccionadas:
                file_name = os.path.basename(foto_path)
                destino = os.path.join(self.folder_path, file_name)
                shutil.copy2(foto_path, destino)
            QMessageBox.information(self, "Éxito", "Imágenes guardadas correctamente.")
            # Luego generamos embeddings
            self.generar_embeddings()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error al guardar imágenes: {e}")

    def generar_embeddings(self):
        self.loading_screen = LoadingScreen(self, message="Espere un momento por favor")
        self.loading_screen.show()
        QApplication.processEvents()
        try:
            graph = load_facenet_model(FACENET_MODEL_PATH)
        except Exception as e:
            self.loading_screen.close()
            QMessageBox.critical(self, "Error", f"Error al cargar FaceNet: {e}")
            return

        self.worker = EmbeddingWorker(self.emp_id, graph)
        self.worker.finished.connect(self.on_embeddings_finished)
        self.worker.error.connect(self.on_embeddings_error)
        self.worker.start()

    def on_embeddings_finished(self):
        self.loading_screen.close()
        QApplication.processEvents()
        winsound.MessageBeep(winsound.MB_OK)
        QMessageBox.information(self, "Éxito", "Empleado registrado correctamente.")
        # Cierra la ventana luego de éxito
        self.close()

    def on_embeddings_error(self, error_message):
        self.loading_screen.close()
        QApplication.processEvents()
        # Se detecta que no cumple umbral => revertimos y cerramos
        QMessageBox.warning(self, "Error en Embeddings",
            f"El proceso falló: {error_message}\nSe revertirán los datos y se cerrará la ventana."
        )
        self.revert_changes()
        # Llamamos a cancelar_operacion() para cerrar
        self.cancelar_operacion()

    def revert_changes(self):
        if self.emp_id:
            try:
                db.revert_employee_data(self.emp_id)  # Nueva función en database_manager
            except Exception as e:
                print(f"Error al revertir en BD: {e}")

        if self.folder_path and os.path.exists(self.folder_path):
            try:
                shutil.rmtree(self.folder_path)
                print(f"Carpeta eliminada: {self.folder_path}")
            except Exception as e:
                print(f"Error al eliminar carpeta: {e}")

        # Se omite el mensaje "Revertido" en favor del warning anterior

    def cancelar_operacion(self):
        self.revert_changes()
        self.close()


if __name__ == "__main__":
    try:
        app = QApplication(sys.argv)
        window = formAddEmp()
        window.show()
        sys.exit(app.exec())
    except Exception as e:
        print(f"Error en la ejecución de la aplicación: {e}")
