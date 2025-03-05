import cv2
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import qtawesome as qta

from PyQt6.QtCore import Qt, QSize, QTimer
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QHBoxLayout, QVBoxLayout, QWidget,
    QLabel, QPushButton, QSpacerItem, QSizePolicy, QFrame,
    QMessageBox
)
from PyQt6.QtGui import QIcon, QAction, QGuiApplication

# Módulos locales
from functions.database_manager import DatabaseManager
from forms.form_addEmp import formAddEmp
from forms.form_settings import formSettings
from forms.form_reports import formReports
from forms.form_gestionEmp import gestionEmp
from forms.form_faces import formUnknownFaces
from forms.camara_window import CameraWindow
from functions.openvino import inicializar_reconocimiento
from utils.settings_controller import (
    DEFAULT_CONFIG, load_config, get_config_value
    # Si no lo usas, quita por completo
)

db = DatabaseManager()


class mainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.secondary_window = None
        self.tercer_window = None
        self.cuarto_window = None

        self.setWindowTitle("RealSecurity")
        self.setWindowIcon(QIcon("media/iconApp.ico"))

        # Intentamos cargar la configuración
        try:
            self.config = load_config()
        except Exception as e:
            self.show_error_message("Error de configuración", f"No se pudo cargar la configuración: {e}")
            self.config = DEFAULT_CONFIG

        # Ventana sin bordes
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)

        # Inicializamos reconocimiento facial
        try:
            self.embeddings_bd, self.nombres_empleados = inicializar_reconocimiento()
        except Exception as e:
            self.show_error_message(
                "Error en reconocimiento",
                f"No se pudo inicializar el reconocimiento facial: {e}"
            )
            self.embeddings_bd, self.nombres_empleados = [], []

        # Configura las cámaras de entrada/salida
        self.camara_entrada_modo = get_config_value("ENTRANCE_CAMERA_MODE", "local")  # local/ip
        self.camara_salida_modo = get_config_value("EXIT_CAMERA_MODE", "local")

        if self.camara_entrada_modo == "local":
            self.camara_entrada_index = get_config_value("ENTRANCE_CAMERA_INDEX", 0)
        else:
            self.camara_entrada_index = get_config_value("ENTRANCE_CAMERA_URL", "")

        if self.camara_salida_modo == "local":
            self.camara_salida_index = get_config_value("EXIT_CAMERA_INDEX", 1)
        else:
            self.camara_salida_index = get_config_value("EXIT_CAMERA_URL", "")

        # Timer para refrescar la barra lateral
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_lateral_bar)
        self.timer.start(5000)

        # Ajustar tamaño a pantalla
        try:
            screen = QGuiApplication.primaryScreen()
            screen_geometry = screen.geometry()
            self.setGeometry(screen_geometry)
        except Exception as e:
            print("Error obteniendo la geometría de pantalla:", e)

        self.setWindowState(Qt.WindowState.WindowMaximized)

        # Widget central
        wd_central = QWidget()
        self.setCentralWidget(wd_central)

        self.setStyleSheet("""
            QWidget {
                background-color: #2b2b2b;
                color: white;
            }
            QLabel {
                font-size: 18px;
                color: white;
            }
            QPushButton {
                background-color: #3a3a3a;
                color: white;
                border: 1px solid #555;
                padding: 10px;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #555;
            }
            QFrame {
                border: 2px solid #444;
            }
        """)

        self.UI()

    def UI(self):
        """Arma la interfaz principal."""
        central_widget = self.centralWidget()
        layout_principal = QHBoxLayout(central_widget)

        # Barra lateral con asistencias
        layout_principal.addWidget(self.lateral_bar())

        # Layout vertical principal (derecha)
        contenedor_principal = QVBoxLayout()
        layout_principal.addLayout(contenedor_principal)

        # Barra superior (botones minimizar/cerrar)
        contenedor_principal.addLayout(self.top_bar())

        # Sección para las cámaras
        self.contenedor_camaras = QHBoxLayout()
        contenedor_principal.addLayout(self.contenedor_camaras)

        # Widgets de cámara
        try:
            self.entrada_camera_widget = CameraWindow(
                self.camara_entrada_index, "Cámara de Entrada",
                self.embeddings_bd, self.nombres_empleados, "entrada"
            )
            self.salida_camera_widget = CameraWindow(
                self.camara_salida_index, "Cámara de Salida",
                self.embeddings_bd, self.nombres_empleados, "salida"
            )
        except Exception as e:
            self.show_error_message(
                "Error al iniciar cámaras",
                f"No se pudieron crear los widgets de cámara: {e}"
            )
            return

        self.entrada_camera_widget.setFixedSize(600, 480)
        self.salida_camera_widget.setFixedSize(600, 480)
        self.contenedor_camaras.addWidget(self.entrada_camera_widget)
        self.contenedor_camaras.addWidget(self.salida_camera_widget)

        # Panel inferior con herramientas y opciones
        contenedor_inferior = QHBoxLayout()
        contenedor_inferior.addWidget(self.panel_herramientas("Barra de Herramientas"))
        contenedor_inferior.addWidget(self.panel_opciones("Opciones de Cámara"))
        contenedor_principal.addLayout(contenedor_inferior)

    # ---------------------------------------------------------------------
    # Barra Lateral
    # ---------------------------------------------------------------------
    def lateral_bar(self):
        """Barra lateral para asistencias recientes."""
        barra_lateral = QFrame()
        barra_lateral.setStyleSheet("background-color: #3a3a3a;")
        # Ajusta el ancho a tu gusto
        barra_lateral.setFixedWidth(self.width() // 8)

        layout_lateral = QVBoxLayout()
        title = QLabel("Asistencias Recientes")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-size: 16px; font-weight: bold; color: white;")
        layout_lateral.addWidget(title)

        try:
            recent_attendances = db.get_recent_attendances(limit=6)
        except Exception as e:
            self.show_error_message("Error de base de datos", f"No se pudieron cargar registros recientes: {e}")
            recent_attendances = []

        if recent_attendances:
            for attendance in recent_attendances:
                employee_id, name, fecha, hora_asis, tipo_asis = attendance
                label_text = f"{name} (ID: {employee_id})\nFecha: {fecha}\n{tipo_asis.capitalize()}: {hora_asis}"
                empleado_label = QLabel(label_text)
                empleado_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
                empleado_label.setStyleSheet("font-size: 14px; color: white;")
                layout_lateral.addWidget(empleado_label)
        else:
            no_data_label = QLabel("No hay registros recientes.")
            no_data_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            no_data_label.setStyleSheet("font-size: 14px; color: gray;")
            layout_lateral.addWidget(no_data_label)

        barra_lateral.setLayout(layout_lateral)
        self.lateral_bar_layout = layout_lateral
        return barra_lateral

    # ---------------------------------------------------------------------
    # Barra Superior (Min/Close)
    # ---------------------------------------------------------------------
    def top_bar(self):
        """Barra superior: minimizar/cerrar."""
        icon_close = qta.icon('fa5s.window-close', color='red')
        icon_min = qta.icon('fa5s.window-minimize', color='white')

        barra_superior = QHBoxLayout()
        # Empuja los botones a la derecha
        barra_superior.addItem(QSpacerItem(20, 0, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum))

        boton_minimizar = QPushButton("")
        boton_minimizar.setIcon(icon_min)
        boton_minimizar.setFixedSize(25, 25)
        boton_minimizar.clicked.connect(self.showMinimized)  # Simplificado
        barra_superior.addWidget(boton_minimizar)

        boton_cerrar = QPushButton("")
        boton_cerrar.setIcon(icon_close)
        boton_cerrar.setFixedSize(25, 25)
        boton_cerrar.clicked.connect(self.closeAll)
        barra_superior.addWidget(boton_cerrar)

        return barra_superior

    # ---------------------------------------------------------------------
    # Panel Inferior: Herramientas y Opciones
    # ---------------------------------------------------------------------
    def panel_herramientas(self, titulo):
        """Panel con botones para gestiones."""
        panel = QFrame()
        panel.setFrameShape(QFrame.Shape.StyledPanel)
        panel.setStyleSheet("background-color: #3a3a3a; border: 2px solid #444;")
        panel.setFixedHeight(self.height() // 4)

        layout = QVBoxLayout()
        botones_layout = QHBoxLayout()

        botones = [
            ("Gestión de empleados", "fa5s.user", self.form_gestionEmp),
            ("Ver reportes",        "fa5s.save", self.form_reports),
            ("Rostros Desconocidos","fa5s.user", self.form_faces),
            ("Configuraciones",     "fa5s.wrench", self.form_settings),
        ]

        for texto, icono, accion in botones:
            boton = QPushButton("\n" + texto)
            boton.setIcon(qta.icon(icono, color="white"))
            boton.setIconSize(QSize(60, 60))
            boton.setStyleSheet("""
                QPushButton {
                    background-color: #4a4a4a;
                    color: white;
                    font-size: 12px;
                    border: none;
                }
                QPushButton:hover {
                    background-color: #555;
                }
            """)
            boton.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

            if accion:
                boton.clicked.connect(accion)

            botones_layout.addWidget(boton)

        layout.addLayout(botones_layout)
        panel.setLayout(layout)
        return panel

    def panel_opciones(self, titulo):
        """Panel para iniciar/detener cámaras."""
        panel = QFrame()
        panel.setFrameShape(QFrame.Shape.StyledPanel)
        panel.setStyleSheet("background-color: #3a3a3a; border: 2px solid #444;")
        panel.setFixedWidth(self.width() // 4)
        panel.setFixedHeight(self.height() // 4)

        layout = QVBoxLayout()
        title = QLabel(titulo)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        btn_iniciar = QPushButton("Iniciar")
        btn_iniciar.setStyleSheet("""
            QPushButton {
                background-color: #4a4a4a;
                color: white;
                font-size: 18px;
                border: none;
            }
            QPushButton:hover {
                background-color: #555;
            }
        """)
        btn_iniciar.clicked.connect(self.iniciar_camaras)
        layout.addWidget(btn_iniciar)

        btn_detener = QPushButton("Detener")
        btn_detener.setStyleSheet("""
            QPushButton {
                background-color: #4a4a4a;
                color: white;
                font-size: 18px;
                border: none;
            }
            QPushButton:hover {
                background-color: #555;
            }
        """)
        btn_detener.clicked.connect(self.detener_camaras)
        layout.addWidget(btn_detener)

        panel.setLayout(layout)
        return panel

    # ---------------------------------------------------------------------
    # Métodos para cargar ventanas secundarias
    # ---------------------------------------------------------------------
    def form_addEmp(self):
        if self.secondary_window is None or not self.secondary_window.isVisible():
            self.secondary_window = formAddEmp()
            self.secondary_window.show()
        elif self.secondary_window.isMinimized():
            self.secondary_window.showMaximized()

    def form_gestionEmp(self):
        if self.secondary_window is None or not self.secondary_window.isVisible():
            self.secondary_window = gestionEmp(db)
            self.secondary_window.show()
        elif self.secondary_window.isMinimized():
            self.secondary_window.showMaximized()

    def form_settings(self):
        if self.tercer_window is None or not self.tercer_window.isVisible():
            self.tercer_window = formSettings()
            self.tercer_window.show()
        elif self.tercer_window.isMinimized():
            self.tercer_window.showMaximized()

    def form_reports(self):
        if self.cuarto_window is None or not self.cuarto_window.isVisible():
            self.cuarto_window = formReports()
            self.cuarto_window.show()
        elif self.cuarto_window.isMinimized():
            self.cuarto_window.showMaximized()

    def form_faces(self):
        if self.cuarto_window is None or not self.cuarto_window.isVisible():
            self.cuarto_window = formUnknownFaces()
            self.cuarto_window.show()
        elif self.cuarto_window.isMinimized():
            self.cuarto_window.showMaximized()

    # ---------------------------------------------------------------------
    # Slots
    # ---------------------------------------------------------------------
    def closeAll(self):
        """Cierra todas las ventanas abiertas y luego la principal."""
        if self.tercer_window is not None:
            self.tercer_window.close()
        if self.secondary_window is not None:
            self.secondary_window.close()
        if self.cuarto_window is not None:
            self.cuarto_window.close()
        self.close()

    def show_error_message(self, title, message):
        msg_box = QMessageBox(self)
        msg_box.setIcon(QMessageBox.Icon.Critical)
        msg_box.setWindowTitle(title)
        msg_box.setText(message)
        msg_box.exec()

    def update_lateral_bar(self):
        """Refresca la lista de asistencias en la barra lateral."""
        if not hasattr(self, "lateral_bar_layout"):
            return

        # Elimina widgets previos (excepto el primero, que es el título)
        while self.lateral_bar_layout.count() > 1:
            item = self.lateral_bar_layout.takeAt(1)
            widget_to_remove = item.widget()
            if widget_to_remove:
                widget_to_remove.deleteLater()

        try:
            recent_attendances = db.get_recent_attendances(limit=6)
        except Exception as e:
            self.show_error_message("Error de base de datos", f"No se pudieron obtener registros recientes: {e}")
            recent_attendances = []

        if recent_attendances:
            for attendance in recent_attendances:
                employee_id, name, fecha, hora_asis, tipo_asis = attendance
                label_text = f"{name} (ID: {employee_id})\nFecha: {fecha}\n{tipo_asis.capitalize()}: {hora_asis}"
                empleado_label = QLabel(label_text)
                empleado_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
                empleado_label.setStyleSheet("font-size: 14px; color: white;")
                self.lateral_bar_layout.addWidget(empleado_label)
        else:
            no_data_label = QLabel("No hay registros recientes.")
            no_data_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            no_data_label.setStyleSheet("font-size: 14px; color: gray;")
            self.lateral_bar_layout.addWidget(no_data_label)

    # ---------------------------------------------------------------------
    # Lógica de cámaras
    # ---------------------------------------------------------------------
    def validar_camaras(self):
        """Verifica si se pueden abrir las cámaras locales (si están en modo 'local')."""
        if isinstance(self.camara_entrada_index, int):
            cap_entrada = cv2.VideoCapture(self.camara_entrada_index)
            entrada_ok = cap_entrada.isOpened()
            cap_entrada.release()
        else:
            # Si es una URL IP, no validamos de esta forma
            entrada_ok = True

        if isinstance(self.camara_salida_index, int):
            cap_salida = cv2.VideoCapture(self.camara_salida_index)
            salida_ok = cap_salida.isOpened()
            cap_salida.release()
        else:
            salida_ok = True

        return entrada_ok and salida_ok

    def iniciar_camaras(self):
        """Inicia la captura en cada CameraWindow."""
        if not self.validar_camaras():
            self.show_error_message(
                "Error de cámara",
                "Se requieren al menos 2 cámaras conectadas (o URLs IP válidas) para iniciar."
            )
            return
        self.entrada_camera_widget.startCamera()
        self.salida_camera_widget.startCamera()

    def detener_camaras(self):
        """Detiene la captura en cada CameraWindow."""
        if self.entrada_camera_widget:
            self.entrada_camera_widget.stopCamera()
        if self.salida_camera_widget:
            self.salida_camera_widget.stopCamera()
        print("Todas las cámaras han sido detenidas correctamente.")

    def closeEvent(self, event):
        """Al cerrar la app, detenemos cámaras."""
        try:
            if self.entrada_camera_widget:
                self.entrada_camera_widget.stopCamera()
            if self.salida_camera_widget:
                self.salida_camera_widget.stopCamera()
        except Exception as e:
            print("Error al detener las cámaras:", e)
        event.accept()


if __name__ == "__main__":
    try:
        app = QApplication(sys.argv)
        window = mainWindow()
        window.show()
        sys.exit(app.exec())
    except Exception as e:
        print(f"Error en la ejecución de la aplicación: {e}")
