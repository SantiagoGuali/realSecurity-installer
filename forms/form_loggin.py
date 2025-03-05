import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os, sys
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QLabel, QLineEdit, QPushButton, QVBoxLayout,
    QWidget, QMessageBox, QHBoxLayout
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QIcon
import qtawesome as qta
from functions.database_manager import DatabaseManager
from forms.form_main import mainWindow
from forms.main_level2 import EmployeeReportsForm
db = DatabaseManager()


class formLoggin(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Inicio de Sesión")
        self.setFixedSize(400, 450)  # Ajuste del tamaño de ventana
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)  # Sin bordes para un diseño más limpio

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(15)
        self.setStyleSheet("""
    /* ======= Estilo Global ======= */
    QWidget {
        background-color: #1e1e2e;  /* Fondo oscuro */
        color: #ffffff;  /* Texto blanco */
        font-family: 'Arial', sans-serif;
        font-size: 16px;
    }

    /* ======= Estilo para Labels ======= */
    QLabel {
        font-size: 18px;
        color: #ffffff;
        font-weight: bold;
    }

    /* ======= Botones Principales ======= */
    QPushButton {
        background-color: #3a3a4a;  /* Color de fondo del botón */
        color: #ffffff;
        border: 2px solid #4c4c6e;  /* Borde moderno */
        border-radius: 8px;
        padding: 10px;
        font-size: 16px;
        transition: all 0.3s ease-in-out;
    }

    QPushButton:hover {
        background-color: #5a5a7e;  /* Cambio de color al pasar el cursor */
        border: 2px solid #7F7FBF;
    }

    QPushButton:pressed {
        background-color: #202030;
        border: 2px solid #9F9FD1;
    }

    /* ======= Botón de Cerrar ======= */
    QPushButton.button-close {
        background-color: #ff4c4c;
        border-radius: 8px;
        padding: 8px;
        font-size: 14px;
        font-weight: bold;
    }

    QPushButton.button-close:hover {
        background-color: #ff3333;
    }

    QPushButton.button-close:pressed {
        background-color: #b30000;
    }

    /* ======= Botón de Minimizar ======= */
    QPushButton.button-minimize {
        background-color: #ffcc00;
        border-radius: 8px;
        padding: 8px;
        font-size: 14px;
        font-weight: bold;
    }

    QPushButton.button-minimize:hover {
        background-color: #ffb700;
    }

    QPushButton.button-minimize:pressed {
        background-color: #b38f00;
    }

    /* ======= Estilo para Inputs (QLineEdit) ======= */
    QLineEdit {
        background-color: #2b2b3e;
        color: #ffffff;
        border: 2px solid #4c4c6e;
        border-radius: 8px;
        padding: 6px;
        font-size: 14px;
    }

    QLineEdit:focus {
        border: 2px solid #7F7FBF;
        background-color: #38384A;
    }

    /* ======= Estilo para Frames ======= */
    QFrame {
        border: 2px solid #444;
        border-radius: 8px;
    }

    /* ======= ScrollBar Vertical ======= */
    QScrollBar:vertical {
        border: none;
        background: #2b2b3e;
        width: 10px;
        margin: 0px 0px 0px 0px;
    }
    
    QScrollBar::handle:vertical {
        background: #555;
        min-height: 20px;
        border-radius: 5px;
    }

    QScrollBar::handle:vertical:hover {
        background: #7F7FBF;
    }

    /* ======= Menús desplegables (QComboBox) ======= */
    QComboBox {
        background-color: #2b2b3e;
        color: white;
        border: 2px solid #4c4c6e;
        border-radius: 8px;
        padding: 5px;
        font-size: 14px;
    }

    QComboBox:focus {
        border: 2px solid #7F7FBF;
    }

    QComboBox QAbstractItemView {
        background-color: #2b2b3e;
        selection-background-color: #5A5A7E;
        color: white;
        border: 1px solid #4C4C6E;
        font-size: 14px;
    }

""")


        # ====== Etiqueta del Título ======
        self.label_titulo = QLabel("Iniciar Sesión")
        self.label_titulo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label_titulo.setStyleSheet("font-size: 24px; font-weight: bold;")

        layout.addWidget(self.label_titulo)

        # ====== Campo de Usuario ======
        self.label_usuario = QLabel("Usuario:")
        self.input_usuario = QLineEdit()
        self.input_usuario.setPlaceholderText("Ingrese su usuario")
        self.input_usuario.setFixedHeight(40)
        self.input_usuario.setStyleSheet("padding-left: 10px;")
        self.input_usuario.setClearButtonEnabled(True)

        layout.addWidget(self.label_usuario)
        layout.addWidget(self.input_usuario)

        # ====== Campo de Contraseña ======
        self.label_contraseña = QLabel("Contraseña:")
        self.input_contraseña = QLineEdit()
        self.input_contraseña.setPlaceholderText("Ingrese su contraseña")
        self.input_contraseña.setFixedHeight(40)
        self.input_contraseña.setEchoMode(QLineEdit.EchoMode.Password)
        self.input_contraseña.setStyleSheet("padding-left: 10px;")

        # ====== Botón para Mostrar/Ocultar Contraseña ======
        self.btn_toggle_password = QPushButton()
        self.btn_toggle_password.setFixedSize(40, 40)
        self.btn_toggle_password.setIcon(qta.icon("fa5s.eye-slash", color="white"))
        self.btn_toggle_password.setStyleSheet("border: none;")
        self.btn_toggle_password.clicked.connect(self.toggle_password_visibility)

        password_layout = QHBoxLayout()
        password_layout.addWidget(self.input_contraseña)
        password_layout.addWidget(self.btn_toggle_password)

        layout.addWidget(self.label_contraseña)
        layout.addLayout(password_layout)

        # ====== Botón de Iniciar Sesión ======
        self.btn_login = QPushButton("Iniciar Sesión")
        self.btn_login.setFixedHeight(45)
        self.btn_login.setStyleSheet("font-size: 16px; font-weight: bold;")
        self.btn_login.setIcon(qta.icon("fa5s.sign-in-alt", color="white"))
        self.btn_login.clicked.connect(self.iniciar_sesion)

        layout.addWidget(self.btn_login)

        # ====== Botones de Cerrar y Minimizar ======
        buttons_layout = QHBoxLayout()
        buttons_layout.setAlignment(Qt.AlignmentFlag.AlignRight)

        self.btn_minimize = QPushButton()
        self.btn_minimize.setFixedSize(30, 30)
        self.btn_minimize.setIcon(qta.icon("fa5s.window-minimize", color="white"))
        self.btn_minimize.setStyleSheet("border: none;")
        self.btn_minimize.clicked.connect(self.showMinimized)

        self.btn_close = QPushButton()
        self.btn_close.setFixedSize(30, 30)
        self.btn_close.setIcon(qta.icon("fa5s.times-circle", color="red"))
        self.btn_close.setStyleSheet("border: none;")
        self.btn_close.clicked.connect(self.close)

        buttons_layout.addWidget(self.btn_minimize)
        buttons_layout.addWidget(self.btn_close)

        layout.addLayout(buttons_layout)

        # ====== Contenedor Principal ======
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.input_contraseña.returnPressed.connect(self.iniciar_sesion)

    def toggle_password_visibility(self):
        """Muestra u oculta la contraseña."""
        if self.input_contraseña.echoMode() == QLineEdit.EchoMode.Password:
            self.input_contraseña.setEchoMode(QLineEdit.EchoMode.Normal)
            self.btn_toggle_password.setIcon(qta.icon("fa5s.eye", color="white"))
        else:
            self.input_contraseña.setEchoMode(QLineEdit.EchoMode.Password)
            self.btn_toggle_password.setIcon(qta.icon("fa5s.eye-slash", color="white"))


    def iniciar_sesion(self):
        """Manejo de inicio de sesión con validación y redirección por rol."""
        usuario = self.input_usuario.text().strip()
        contraseña = self.input_contraseña.text().strip()

        if not usuario or not contraseña:
            QMessageBox.warning(self, "Error", "Ingrese su usuario y contraseña.")
            return

        rol = db.autenticar_usuario(usuario, contraseña)  # Obtener el rol

        if rol:  # Si autenticación es exitosa, `rol` no será None
            if rol == "GERENTE":
                self.home = mainWindow()  # Redirigir a la ventana principal
            elif rol == "USUARIO":
                self.home = EmployeeReportsForm()  # Redirigir a la ventana de reportes
            else:
                QMessageBox.critical(self, "Error", "Rol no válido.")
                return

            self.home.show()
            self.close()
        else:
            QMessageBox.critical(self, "Error", "Usuario o contraseña incorrectos.")



