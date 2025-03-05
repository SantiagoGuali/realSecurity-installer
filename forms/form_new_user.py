import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PyQt6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QMessageBox, QComboBox
from PyQt6.QtCore import Qt
from argon2 import PasswordHasher
from functions.database_manager import DatabaseManager
from functions.reload_app import reiniciar_aplicacion

class RegisterFirstUser(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Registrar Primer Usuario")
        self.setFixedSize(400, 350)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)

        self.db = DatabaseManager()
        # if self.db.existen_usuarios():
        #     self.close()
        #     return

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        self.label_titulo = QLabel("Registrar Usuario")
        self.label_titulo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label_titulo.setStyleSheet("font-size: 18px; font-weight: bold;")
        layout.addWidget(self.label_titulo)

        self.input_usuario = QLineEdit()
        self.input_usuario.setPlaceholderText("Ingrese un usuario")
        layout.addWidget(self.input_usuario)

        self.input_password = QLineEdit()
        self.input_password.setPlaceholderText("Ingrese una contraseña")
        self.input_password.setEchoMode(QLineEdit.EchoMode.Password)
        layout.addWidget(self.input_password)

        self.rol_selector = QComboBox()
        self.rol_selector.addItems(["GERENTE", "USUARIO"])
        layout.addWidget(self.rol_selector)

        # Contenedor de botones (Registro y Cancelar)
        button_layout = QHBoxLayout()

        self.btn_registrar = QPushButton("Registrar Usuario")
        self.btn_registrar.clicked.connect(self.registrar_usuario)
        button_layout.addWidget(self.btn_registrar)

        self.btn_cancelar = QPushButton("Cancelar")
        self.btn_cancelar.clicked.connect(self.close)  # Cierra el formulario sin registrar
        button_layout.addWidget(self.btn_cancelar)

        layout.addLayout(button_layout)  # Agregar botones al layout principal

        self.setLayout(layout)

    def registrar_usuario(self):
        usuario = self.input_usuario.text().strip()
        password = self.input_password.text().strip()
        rol = self.rol_selector.currentText()

        # if not usuario or not password:
        #     QMessageBox.warning(self, "Error", "Debe ingresar un usuario y una contraseña.")
        #     return

        # if self.db.existen_usuarios():
        #     QMessageBox.critical(self, "Error", "Ya existe un usuario registrado.")
        #     self.close()
        #     return

        # if rol == "GERENTE" and self.db.existe_gerente():
        #     QMessageBox.warning(self, "Error", "Solo puede existir un GERENTE registrado.")
        #     return

        self.db.registrar_usuario(usuario, password, rol)
        QMessageBox.information(self, "Éxito", f"Usuario '{usuario}' registrado correctamente con rol '{rol}'.")
        reiniciar_aplicacion()
        self.close()

# Métodos en DatabaseManager para verificar si existen usuarios o un gerente
# DatabaseManager.existen_usuarios = lambda self: self.connection.cursor().execute("SELECT COUNT(*) FROM usuarios").fetchone()[0] > 0
# DatabaseManager.existe_gerente = lambda self: self.connection.cursor().execute("SELECT COUNT(*) FROM usuarios WHERE rol = 'GERENTE'").fetchone()[0] > 0
