import datetime
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import re
import cv2

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QWidget, QSlider,
    QComboBox, QCheckBox, QListWidget, QStackedWidget, QMessageBox,
    QApplication, QLineEdit, QFileDialog, QRadioButton
)
from PyQt6.QtCore import Qt, QProcess
from PyQt6.QtGui import QPixmap, QImage

from utils.settings_controller import (
    load_config, save_config, get_config_value, set_config_value
)
from forms.form_user_list import UserManagement
from functions.database_manager import DatabaseManager
from functions.send_grid import enviar_reporte_por_correo

db = DatabaseManager()

class formSettings(QDialog):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Configuraciones")
        self.setGeometry(300, 100, 900, 600)

        self.config = load_config()
        self.camara_activa = None
        self.camaras_modificadas = False  # Flag p/indicar si se cambian cámaras/horarios/etc.

        # --- Control de reportes automáticos basados en fecha ---
        # Se puede guardar en config.json la "LAST_REPORT_DATE" para no repetir
        # en el mismo día si ya se envió con éxito.
        self.last_report_date = self.config.get("LAST_REPORT_DATE", "")

        self.initUI()

    def initUI(self):
        main_layout = QHBoxLayout(self)

        # -------------------------
        # Barra lateral (menú)
        # -------------------------
        sidebar_layout = QVBoxLayout()
        self.menu_items = [
            "Notificaciones", "CUDA", "Reconocimiento",
            "Embeddings", "Cámaras", "Horarios", "Reportes"
        ]
        self.menu_list = QListWidget()
        self.menu_list.addItems(self.menu_items)
        self.menu_list.setFixedWidth(200)
        self.menu_list.currentRowChanged.connect(self.change_page)
        sidebar_layout.addWidget(self.menu_list)

        sidebar_layout.addStretch()

        self.btn_register_user = QPushButton("Registrar Usuario")
        self.btn_register_user.setStyleSheet("""
            QPushButton {
                background-color: #3a3a3a;
                color: white;
                border: 1px solid #555;
                padding: 10px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #555;
            }
        """)
        self.btn_register_user.clicked.connect(self.abrir_registro_usuario)
        sidebar_layout.addWidget(self.btn_register_user)

        sidebar_container = QWidget()
        sidebar_container.setLayout(sidebar_layout)
        main_layout.addWidget(sidebar_container)

        # -------------------------
        # Contenido derecho
        # -------------------------
        self.stack = QStackedWidget()
        self.create_pages()

        # Ajustar campos cámara (IP vs local) según el radio button
        self.toggle_camera_fields(self.rb_entry_local, self.entry_camera_selector, self.entry_ip_edit)
        self.toggle_camera_fields(self.rb_exit_local, self.exit_camera_selector, self.exit_ip_edit)

        right_layout = QVBoxLayout()
        right_layout.addWidget(self.stack)

        # Botones "Guardar" y "Cancelar"
        button_layout = QHBoxLayout()
        save_button = QPushButton("Guardar")
        save_button.clicked.connect(self.save_configurations)
        cancel_button = QPushButton("Cancelar")
        cancel_button.clicked.connect(self.close)
        button_layout.addWidget(save_button)
        button_layout.addWidget(cancel_button)

        right_layout.addLayout(button_layout)
        main_layout.addLayout(right_layout)

        # Página inicial
        self.menu_list.setCurrentRow(0)

        self.setLayout(main_layout)

    def create_pages(self):
        """Construye las páginas del QStackedWidget en el orden del self.menu_items."""

        # ----------------------
        # (0) Notificaciones
        # ----------------------
        notif_layout = QVBoxLayout()
        notif_layout.addWidget(QLabel("Configuración de Notificaciones"))

        self.notifications_checkbox_asistencia = QCheckBox("Notificación de Asistencias")
        self.notifications_checkbox_intruso    = QCheckBox("Notificación de Intrusos")
        self.phone_number_input = QLineEdit()
        self.phone_number_input.setPlaceholderText("Número de Teléfono")

        self.notifications_checkbox_asistencia.setChecked(
            self.config.get("NOTIFICATION_ASISTENCIA", False)
        )
        self.notifications_checkbox_intruso.setChecked(
            self.config.get("NOTIFICACION_INTRUSO", False)
        )
        self.phone_number_input.setText(self.config.get("EMAIL_USER", ""))

        self.toggle_phone_input()
        self.notifications_checkbox_asistencia.stateChanged.connect(self.toggle_phone_input)
        self.notifications_checkbox_intruso.stateChanged.connect(self.toggle_phone_input)

        notif_layout.addWidget(self.notifications_checkbox_asistencia)
        notif_layout.addWidget(self.notifications_checkbox_intruso)
        notif_layout.addWidget(self.phone_number_input)

        notif_page = QWidget()
        notif_page.setLayout(notif_layout)
        self.stack.addWidget(notif_page)

        # ----------------------
        # (1) CUDA
        # ----------------------
        cuda_layout = QVBoxLayout()
        cuda_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        cuda_layout.addWidget(QLabel("Activar/Desactivar CUDA para reconocimiento:"))

        self.cuda_checkbox = QCheckBox("Habilitar uso de CUDA")
        self.cuda_checkbox.setChecked(self.config.get("USE_CUDA", False))
        self.cuda_info = QLabel(
            "Beneficios:\n- Aceleración facial.\n\n"
            "Implicaciones:\n- Requiere hardware compatible.\n- Mayor consumo energía."
        )
        self.cuda_checkbox.stateChanged.connect(self.update_cuda_info)

        cuda_layout.addWidget(self.cuda_checkbox)
        cuda_layout.addWidget(self.cuda_info)

        cuda_page = QWidget()
        cuda_page.setLayout(cuda_layout)
        self.stack.addWidget(cuda_page)

        # ----------------------
        # (2) Reconocimiento
        # ----------------------
        recog_layout = QVBoxLayout()

        # Sliders MTCNN, Haar, etc.
        self.min_face_size_slider = QSlider(Qt.Orientation.Horizontal)
        self.min_face_size_slider.setRange(20, 100)
        self.min_face_size_slider.setValue(self.config.get("MTCNN_MIN_FACE_SIZE", 40))
        self.min_face_size_label = QLabel(f"Tamaño mínimo de rostro (MTCNN): {self.min_face_size_slider.value()}")
        self.min_face_size_slider.valueChanged.connect(
            lambda: self.update_label(self.min_face_size_label, "Tamaño mínimo de rostro (MTCNN):", self.min_face_size_slider.value())
        )
        recog_layout.addWidget(self.min_face_size_label)
        recog_layout.addWidget(self.min_face_size_slider)

        self.haar_scale_factor_slider = QSlider(Qt.Orientation.Horizontal)
        self.haar_scale_factor_slider.setRange(101, 150)
        self.haar_scale_factor_slider.setValue(int(self.config.get("HAAR_SCALE_FACTOR", 1.1) * 100))
        self.haar_scale_factor_label = QLabel(f"Factor escala (Haar): {self.haar_scale_factor_slider.value() / 100:.2f}")
        self.haar_scale_factor_slider.valueChanged.connect(
            lambda: self.update_label(self.haar_scale_factor_label, "Factor escala (Haar):", self.haar_scale_factor_slider.value() / 100)
        )
        recog_layout.addWidget(self.haar_scale_factor_label)
        recog_layout.addWidget(self.haar_scale_factor_slider)

        self.haar_min_neighbors_slider = QSlider(Qt.Orientation.Horizontal)
        self.haar_min_neighbors_slider.setRange(3, 10)
        self.haar_min_neighbors_slider.setValue(self.config.get("HAAR_MIN_NEIGHBORS", 5))
        self.haar_min_neighbors_label = QLabel(f"Vecinos mínimos (Haar): {self.haar_min_neighbors_slider.value()}")
        self.haar_min_neighbors_slider.valueChanged.connect(
            lambda: self.update_label(self.haar_min_neighbors_label, "Vecinos mínimos (Haar):", self.haar_min_neighbors_slider.value())
        )
        recog_layout.addWidget(self.haar_min_neighbors_label)
        recog_layout.addWidget(self.haar_min_neighbors_slider)

        self.haar_min_size_slider = QSlider(Qt.Orientation.Horizontal)
        self.haar_min_size_slider.setRange(20, 100)
        self.haar_min_size_slider.setValue(self.config.get("HAAR_MIN_SIZE", [30,30])[0])
        self.haar_min_size_label = QLabel(f"Tamaño mínimo (Haar): {self.haar_min_size_slider.value()}")
        self.haar_min_size_slider.valueChanged.connect(
            lambda: self.update_label(self.haar_min_size_label, "Tamaño mínimo (Haar):", self.haar_min_size_slider.value())
        )
        recog_layout.addWidget(self.haar_min_size_label)
        recog_layout.addWidget(self.haar_min_size_slider)

        # Similarity threshold
        self.similarity_threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.similarity_threshold_slider.setRange(70, 95)
        self.similarity_threshold_slider.setValue(int(self.config.get("SIMILARITY_THRESHOLD", 0.85) * 100))
        self.similarity_threshold_label = QLabel(f"Umbral de Similitud: {self.similarity_threshold_slider.value() / 100:.2f}")
        self.similarity_threshold_slider.valueChanged.connect(
            lambda: self.update_label(self.similarity_threshold_label, "Umbral de Similitud:", self.similarity_threshold_slider.value() / 100)
        )
        recog_layout.addWidget(self.similarity_threshold_label)
        recog_layout.addWidget(self.similarity_threshold_slider)

        recog_page = QWidget()
        recog_page.setLayout(recog_layout)
        self.stack.addWidget(recog_page)

        # ----------------------
        # (3) Embeddings
        # ----------------------
        embed_layout = QVBoxLayout()
        self.preprocess_size_combo = QComboBox()
        self.preprocess_size_combo.addItems(["160", "224"])
        current_size = self.config.get("PREPROCESS_RESIZE", [160,160])[0]
        self.preprocess_size_combo.setCurrentText(str(current_size))

        embed_layout.addWidget(QLabel("Resolución de preprocesamiento de embeddings:"))
        embed_layout.addWidget(self.preprocess_size_combo)

        embed_page = QWidget()
        embed_page.setLayout(embed_layout)
        self.stack.addWidget(embed_page)

        # ----------------------
        # (4) Cámaras
        # ----------------------
        cameras_layout = QVBoxLayout()
        available_cameras = self.discover_cameras()

        # Entrada
        self.rb_entry_local = QRadioButton("Cámara Local")
        self.rb_entry_ip    = QRadioButton("Cámara IP")
        entry_mode = self.config.get("ENTRANCE_CAMERA_MODE", "local")
        if entry_mode == "ip":
            self.rb_entry_ip.setChecked(True)
        else:
            self.rb_entry_local.setChecked(True)

        self.entry_ip_edit = QLineEdit()
        self.entry_ip_edit.setPlaceholderText("URL RTSP/HTTP IP")
        self.entry_ip_edit.setText(self.config.get("ENTRANCE_CAMERA_URL",""))

        self.entry_camera_selector = QComboBox()
        saved_ent_index = self.config.get("ENTRANCE_CAMERA_INDEX", 0)
        self.load_cameras_into_combobox(self.entry_camera_selector, available_cameras, saved_index=saved_ent_index)

        self.rb_entry_local.toggled.connect(self.mark_cameras_modified)
        self.rb_entry_ip.toggled.connect(self.mark_cameras_modified)
        self.entry_camera_selector.currentIndexChanged.connect(self.mark_cameras_modified)
        self.entry_ip_edit.textChanged.connect(self.mark_cameras_modified)

        cameras_layout.addWidget(QLabel("Cámara de ENTRADA:"))
        cameras_layout.addWidget(self.rb_entry_local)
        cameras_layout.addWidget(self.rb_entry_ip)
        cameras_layout.addWidget(self.entry_camera_selector)
        cameras_layout.addWidget(self.entry_ip_edit)

        # Salida
        self.rb_exit_local = QRadioButton("Cámara Local")
        self.rb_exit_ip    = QRadioButton("Cámara IP")
        exit_mode = self.config.get("EXIT_CAMERA_MODE", "local")
        if exit_mode == "ip":
            self.rb_exit_ip.setChecked(True)
        else:
            self.rb_exit_local.setChecked(True)

        self.exit_ip_edit = QLineEdit()
        self.exit_ip_edit.setPlaceholderText("URL RTSP/HTTP IP")
        self.exit_ip_edit.setText(self.config.get("EXIT_CAMERA_URL",""))

        self.exit_camera_selector = QComboBox()
        saved_ext_index = self.config.get("EXIT_CAMERA_INDEX", 1)
        self.load_cameras_into_combobox(self.exit_camera_selector, available_cameras, saved_index=saved_ext_index)

        self.rb_exit_local.toggled.connect(self.mark_cameras_modified)
        self.rb_exit_ip.toggled.connect(self.mark_cameras_modified)
        self.exit_camera_selector.currentIndexChanged.connect(self.mark_cameras_modified)
        self.exit_ip_edit.textChanged.connect(self.mark_cameras_modified)

        cameras_layout.addWidget(QLabel("Cámara de SALIDA:"))
        cameras_layout.addWidget(self.rb_exit_local)
        cameras_layout.addWidget(self.rb_exit_ip)
        cameras_layout.addWidget(self.exit_camera_selector)
        cameras_layout.addWidget(self.exit_ip_edit)

        cameras_page = QWidget()
        cameras_page.setLayout(cameras_layout)
        self.stack.addWidget(cameras_page)

        # ----------------------
        # (5) Horarios
        # ----------------------
        interval_layout = QVBoxLayout()
        interval_layout.addWidget(QLabel("Configuración de intervalos de Horario (Entrada/Salida)"))

        self.entry_start_input = QLineEdit()
        self.entry_end_input   = QLineEdit()
        self.entry_start_input.setPlaceholderText("HH:MM")
        self.entry_end_input.setPlaceholderText("HH:MM")

        self.entry_start_input.setText(self.config.get("ENTRY_INTERVAL_START", ""))
        self.entry_end_input.setText(self.config.get("ENTRY_INTERVAL_END",""))

        self.exit_start_input = QLineEdit()
        self.exit_end_input   = QLineEdit()
        self.exit_start_input.setPlaceholderText("HH:MM")
        self.exit_end_input.setPlaceholderText("HH:MM")

        self.exit_start_input.setText(self.config.get("EXIT_INTERVAL_START",""))
        self.exit_end_input.setText(self.config.get("EXIT_INTERVAL_END",""))

        self.entry_start_input.textChanged.connect(self.mark_cameras_modified)
        self.entry_end_input.textChanged.connect(self.mark_cameras_modified)
        self.exit_start_input.textChanged.connect(self.mark_cameras_modified)
        self.exit_end_input.textChanged.connect(self.mark_cameras_modified)

        interval_layout.addWidget(QLabel("Entrada: inicio-fin (HH:MM). Si se deja vacío => sin restricción."))
        interval_layout.addWidget(self.entry_start_input)
        interval_layout.addWidget(self.entry_end_input)

        interval_layout.addWidget(QLabel("Salida: inicio-fin (HH:MM). Si se deja vacío => sin restricción."))
        interval_layout.addWidget(self.exit_start_input)
        interval_layout.addWidget(self.exit_end_input)

        interval_page = QWidget()
        interval_page.setLayout(interval_layout)
        self.stack.addWidget(interval_page)

        # ----------------------
        # (6) Reportes
        # ----------------------
        report_layout = QVBoxLayout()
        self.report_period_selector = QComboBox()
        self.report_period_selector.addItems(["Diario", "Semanal", "Mensual", "Anual"])
        self.report_period_selector.setCurrentText(self.config.get("REPORT_PERIOD", "Mensual"))
        report_layout.addWidget(QLabel("Período de reportes automáticos:"))
        report_layout.addWidget(self.report_period_selector)

        self.enable_email_reports = QCheckBox("Enviar reportes por correo")
        self.enable_email_reports.setChecked(self.config.get("ENABLE_EMAIL_REPORTS", False))
        report_layout.addWidget(self.enable_email_reports)

        self.email_input = QLineEdit()
        self.email_input.setPlaceholderText("Correo para reportes")
        self.email_input.setText(self.config.get("REPORT_EMAIL",""))
        report_layout.addWidget(self.email_input)

        self.btn_generate_report = QPushButton("Generar Reporte Ahora")
        self.btn_generate_report.setStyleSheet("background-color: #4a4a4a; padding:10px; font-size:14px;")
        self.btn_generate_report.clicked.connect(self.generate_manual_report)
        report_layout.addWidget(self.btn_generate_report)

        report_page = QWidget()
        report_page.setLayout(report_layout)
        self.stack.addWidget(report_page)

    def mark_cameras_modified(self):
        self.camaras_modificadas = True

    def load_cameras_into_combobox(self, combo, camera_list, saved_index):
        combo.clear()
        index_to_select = 0
        for i, label in enumerate(camera_list):
            if "Cámara" in label:
                parts = label.split()
                if len(parts)==2 and parts[0]=="Cámara":
                    try:
                        real_index = int(parts[1])
                    except ValueError:
                        real_index = -1
                else:
                    real_index = -1

                combo.addItem(label, real_index)
                if real_index == saved_index:
                    index_to_select = i
            else:
                combo.addItem(label, -1)

        combo.setCurrentIndex(index_to_select)

    def toggle_phone_input(self):
        if self.notifications_checkbox_intruso.isChecked():
            self.phone_number_input.setEnabled(True)
        else:
            self.phone_number_input.setEnabled(False)
            self.phone_number_input.clear()

    def change_page(self, index):
        self.stack.setCurrentIndex(index)

    def toggle_camera_fields(self, rb_local, combo, ip_edit):
        if rb_local.isChecked():
            combo.setEnabled(True)
            ip_edit.setEnabled(False)
        else:
            combo.setEnabled(False)
            ip_edit.setEnabled(True)

    def save_configurations(self):
        if not self.validate_camera_selection():
            return

        # Guardar intervalos
        if not self.validate_and_save_intervals():
            return

        # Otras configs
        self.config["PREPROCESS_RESIZE"] = [
            int(self.preprocess_size_combo.currentText()),
            int(self.preprocess_size_combo.currentText())
        ]
        self.config["SIMILARITY_THRESHOLD"] = self.similarity_threshold_slider.value()/100
        self.config["MTCNN_MIN_FACE_SIZE"]  = self.min_face_size_slider.value()
        self.config["HAAR_SCALE_FACTOR"]    = self.haar_scale_factor_slider.value()/100
        self.config["HAAR_MIN_NEIGHBORS"]   = self.haar_min_neighbors_slider.value()
        self.config["HAAR_MIN_SIZE"] = [
            self.haar_min_size_slider.value(),
            self.haar_min_size_slider.value()
        ]
        self.config["USE_CUDA"]       = self.cuda_checkbox.isChecked()
        self.config["NOTIFICATION_ASISTENCIA"] = self.notifications_checkbox_asistencia.isChecked()
        self.config["NOTIFICACION_INTRUSO"]    = self.notifications_checkbox_intruso.isChecked()
        self.config["EMAIL_USER"]             = self.phone_number_input.text()

        self.config["REPORT_PERIOD"]        = self.report_period_selector.currentText()
        self.config["ENABLE_EMAIL_REPORTS"] = self.enable_email_reports.isChecked()
        self.config["REPORT_EMAIL"]         = self.email_input.text()

        # Cámaras
        if self.rb_entry_local.isChecked():
            self.config["ENTRANCE_CAMERA_MODE"] = "local"
            chosen_index = self.entry_camera_selector.currentData()
            if chosen_index is None or chosen_index<0:
                chosen_index=0
            self.config["ENTRANCE_CAMERA_INDEX"] = chosen_index
        else:
            self.config["ENTRANCE_CAMERA_MODE"] = "ip"
            self.config["ENTRANCE_CAMERA_URL"]  = self.entry_ip_edit.text()

        if self.rb_exit_local.isChecked():
            self.config["EXIT_CAMERA_MODE"] = "local"
            chosen_index2 = self.exit_camera_selector.currentData()
            if chosen_index2 is None or chosen_index2<0:
                chosen_index2 = 1
            self.config["EXIT_CAMERA_INDEX"] = chosen_index2
        else:
            self.config["EXIT_CAMERA_MODE"] = "ip"
            self.config["EXIT_CAMERA_URL"]  = self.exit_ip_edit.text()

        # Guardar
        save_config(self.config)

        if self.camaras_modificadas:
            self.prompt_restart()

        self.close()

    def validate_and_save_intervals(self):
        pattern = re.compile(r"^(?:[01]\d|2[0-3]):[0-5]\d$")
        def check_time_or_empty(value):
            if not value.strip():
                return True
            return bool(pattern.match(value.strip()))

        ent_start = self.entry_start_input.text().strip()
        ent_end   = self.entry_end_input.text().strip()
        sal_start = self.exit_start_input.text().strip()
        sal_end   = self.exit_end_input.text().strip()

        if not check_time_or_empty(ent_start):
            QMessageBox.warning(self, "Error Intervalo",
                f"Hora inicial de Entrada '{ent_start}' no es válido (HH:MM).")
            return False
        if not check_time_or_empty(ent_end):
            QMessageBox.warning(self, "Error Intervalo",
                f"Hora final de Entrada '{ent_end}' no es válido (HH:MM).")
            return False
        if not check_time_or_empty(sal_start):
            QMessageBox.warning(self, "Error Intervalo",
                f"Hora inicial de Salida '{sal_start}' no es válido (HH:MM).")
            return False
        if not check_time_or_empty(sal_end):
            QMessageBox.warning(self, "Error Intervalo",
                f"Hora final de Salida '{sal_end}' no es válido (HH:MM).")
            return False

        self.config["ENTRY_INTERVAL_START"] = ent_start
        self.config["ENTRY_INTERVAL_END"]   = ent_end
        self.config["EXIT_INTERVAL_START"]  = sal_start
        self.config["EXIT_INTERVAL_END"]    = sal_end
        return True

    def prompt_restart(self):
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Icon.Warning)
        msg_box.setText("Los cambios en las cámaras requieren reiniciar el sistema.")
        msg_box.setInformativeText("¿Desea reiniciar ahora?")
        msg_box.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        msg_box.setDefaultButton(QMessageBox.StandardButton.Yes)
        response = msg_box.exec()
        if response == QMessageBox.StandardButton.Yes:
            self.restart_application()

    def restart_application(self):
        QApplication.quit()
        QProcess.startDetached(sys.executable, sys.argv)
        sys.exit()

    def validate_camera_selection(self):
        if self.rb_entry_local.isChecked() and self.rb_exit_local.isChecked():
            e_data = self.entry_camera_selector.currentData()
            x_data = self.exit_camera_selector.currentData()
            if e_data==-1 or x_data==-1:
                QMessageBox.warning(self,"Advertencia","Debe seleccionar cámaras locales válidas.")
                return False
            if e_data == x_data:
                QMessageBox.warning(self,"Advertencia","La cámara de entrada y de salida deben ser distintas.")
                return False
        return True

    def confirm_camera_selection(self):
        if not self.validate_camera_selection():
            return
        save_config(self.config)
        QMessageBox.information(self,"Éxito","Cámaras configuradas correctamente.")
        self.close()

    def abrir_registro_usuario(self):
        self.registro_usuario_window = UserManagement()
        self.registro_usuario_window.exec()

    def generate_manual_report(self):
        file_path,_ = QFileDialog.getSaveFileName(self,"Guardar Reporte","","Archivos Excel (*.xlsx)")
        if file_path:
            period = self.report_period_selector.currentText().lower()
            db.auto_generate_employee_report(period, file_path)
            QMessageBox.information(self,"Éxito","Reporte generado correctamente.")
            if self.enable_email_reports.isChecked():
                email_to = self.email_input.text().strip()
                if email_to:
                    enviar_reporte_por_correo(file_path, email_to)
                    QMessageBox.information(self,"Correo Enviado",f"El reporte se ha enviado a {email_to}.")
                else:
                    QMessageBox.warning(self,"Error","Ingrese un correo válido para enviar reportes.")

    def auto_generate_report(self):
        """Reportes automáticos según fecha, evitando duplicar si ya se envió hoy."""
        # 1) Ver si ya se envió un reporte hoy
        today_str = datetime.datetime.now().strftime("%Y-%m-%d")
        last_report = self.config.get("LAST_REPORT_DATE","")
        if last_report == today_str:
            print("[AutoReporte] Ya se envió un reporte hoy. Se omite.")
            return
        # 2) Generar
        period = get_config_value("REPORT_PERIOD","mensual").lower()
        dir_reports = "reportes"
        if not os.path.exists(dir_reports):
            os.makedirs(dir_reports)
        file_path = os.path.join(dir_reports, f"reporte_{period}_{today_str}.xlsx")

        db.auto_generate_employee_report(period,file_path)
        print(f"Reporte autogenerado en {file_path}")

        # 3) Enviar si corresponde
        if get_config_value("ENABLE_EMAIL_REPORTS",False):
            email_to = get_config_value("REPORT_EMAIL","").strip()
            if email_to:
                # Tratar de enviar. Si falla => no actualizamos LAST_REPORT_DATE
                try:
                    enviar_reporte_por_correo(file_path, email_to)
                    print(f"[AutoReporte] Enviado a {email_to}")
                    # 4) Guardar que ya se envió hoy
                    self.config["LAST_REPORT_DATE"] = today_str
                    save_config(self.config)
                except Exception as e:
                    print(f"[AutoReporte] Error al enviar: {e}")
            else:
                print("[AutoReporte] No hay correo configurado. No se envía.")
        else:
            print("[AutoReporte] Envío de correo desactivado.")

    def discover_cameras(self):
        available_cameras = []
        max_check_range = 10
        for idx in range(max_check_range):
            cap = cv2.VideoCapture(idx, cv2.CAP_ANY)
            if cap.isOpened():
                available_cameras.append(f"Cámara {idx}")
                cap.release()
            else:
                cap.release()
        if not available_cameras:
            available_cameras.append("No hay cámaras disponibles")
        return available_cameras

    def update_cuda_info(self):
        if self.cuda_checkbox.isChecked():
            self.cuda_info.setText(
                "Beneficios:\n- Aceleración reconocimiento.\n\n"
                "Implicaciones:\n- Hardware GPU.\n- Mayor consumo."
            )
        else:
            self.cuda_info.setText("CUDA deshabilitado. Usando CPU.")

    def update_label(self,label,prefix,value):
        if isinstance(value,float):
            label.setText(f"{prefix} {value:.2f}")
        else:
            label.setText(f"{prefix} {value}")

if __name__=="__main__":
    app = QApplication(sys.argv)
    form = formSettings()
    form.show()
    sys.exit(app.exec())
