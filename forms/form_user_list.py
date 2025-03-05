import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PyQt6.QtWidgets import (
    QApplication, QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QMessageBox, QTableWidget, QTableWidgetItem, QHeaderView
)
from PyQt6.QtCore import Qt
from functions.database_manager import DatabaseManager
from forms.form_new_user import RegisterFirstUser

class UserManagement(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Administración de Usuarios")
        self.setWindowState(Qt.WindowState.WindowMaximized)
        self.setStyleSheet("""
            QWidget {
                background-color: #2b2b2b;
                color: white;
                font-family: Arial, Helvetica, sans-serif;
            }
            QLineEdit {
                background-color: #2C2C3E;
                color: #FFFFFF;
                border: 2px solid #4C4C6E;
                border-radius: 8px;
                padding: 5px;
                font-size: 14px;
                selection-background-color: #5A5A7E;
            }
            QLineEdit:focus {
                border: 2px solid #7F7FBF;
                background-color: #38384A;
            }
            QLabel {
                font-size: 18px;
                color: white;
            }
            QPushButton {
                background-color: #3a3a3a;
                color: white;
                border: 1px solid #555;
                border-radius: 8px;
                padding: 10px;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #555;
            }
            QComboBox {
                background-color: #2C2C3E;
                color: white;
                border: 2px solid #4C4C6E;
                border-radius: 8px;
                padding: 5px;
                font-size: 14px;
            }
            QComboBox:focus {
                border: 2px solid #7F7FBF;
            }
            QComboBox QAbstractItemView {
                background-color: #2C2C3E;
                selection-background-color: #5A5A7E;
                color: white;
                border: 1px solid #4C4C6E;
                font-size: 14px;
            }
            QDateEdit {
                background-color: #2C2C3E;
                color: white;
                border: 2px solid #4C4C6E;
                border-radius: 8px;
                padding: 5px;
                font-size: 14px;
            }
            QDateEdit:focus {
                border: 2px solid #7F7FBF;
            }
            QDateEdit::drop-down {
                background-color: #3a3a3a;
            }
        """)
        
        self.db = DatabaseManager()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        self.label_titulo = QLabel("Usuarios Registrados")
        self.label_titulo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label_titulo.setStyleSheet("font-size: 24px; font-weight: bold;")
        layout.addWidget(self.label_titulo)

        self.table = QTableWidget()
        self.table.setColumnCount(2)
        self.table.setHorizontalHeaderLabels(["Usuario", "Eliminar"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.table)

        button_layout = QHBoxLayout()
        self.btn_nuevo_usuario = QPushButton("Nuevo Usuario")
        self.btn_nuevo_usuario.setStyleSheet("background-color: blue; color: white;")
        self.btn_nuevo_usuario.clicked.connect(self.abrir_formulario_usuario)
        button_layout.addWidget(self.btn_nuevo_usuario)

        layout.addLayout(button_layout)
        self.cargar_usuarios()
        self.setLayout(layout)

    def cargar_usuarios(self):
        self.table.setRowCount(0)
        usuarios = self.db.obtener_usuarios()
        for row, usuario in enumerate(usuarios):
            self.table.insertRow(row)
            self.table.setItem(row, 0, QTableWidgetItem(usuario))
            btn_eliminar = QPushButton("Eliminar")
            btn_eliminar.setStyleSheet("background-color: red; color: white;")
            btn_eliminar.clicked.connect(lambda _, u=usuario: self.eliminar_usuario(u))
            self.table.setCellWidget(row, 1, btn_eliminar)

    def eliminar_usuario(self, usuario):
        confirmation = QMessageBox.question(
            self, "Eliminar Usuario",
            f"¿Está seguro de que desea eliminar al usuario '{usuario}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        if confirmation == QMessageBox.StandardButton.Yes:
            if self.db.eliminar_usuario(usuario):
                QMessageBox.information(self, "Éxito", f"Usuario '{usuario}' eliminado correctamente.")
                self.cargar_usuarios()
            else:
                QMessageBox.critical(self, "Error", f"No se pudo eliminar al usuario '{usuario}'.")

    def abrir_formulario_usuario(self):
        self.form_usuario = RegisterFirstUser()
        self.form_usuario.exec()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = UserManagement()
    window.show()
    sys.exit(app.exec())
