import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import re
from PyQt6.QtWidgets import QMessageBox
import cv2
class ValidadorEmpleado:

    @staticmethod
    def validar_cedula(cedula):
        if not cedula.strip():
            return False, "El campo de cédula está vacío."

        if not cedula.isdigit() or len(cedula) != 10:
            return False, "La cédula debe tener 10 dígitos numéricos."
        
        provincia = int(cedula[:2])
        if provincia < 1 or provincia > 24:
            return False, "La cédula ingresada no pertenece a una provincia válida en Ecuador."

        coeficientes = [2, 1, 2, 1, 2, 1, 2, 1, 2]
        suma = sum([(int(cedula[i]) * coeficientes[i]) if (int(cedula[i]) * coeficientes[i]) < 10 
                    else (int(cedula[i]) * coeficientes[i]) - 9 for i in range(9)])
        
        digito_verificador = (10 - (suma % 10)) % 10
        if digito_verificador != int(cedula[9]):
            return False, "La cédula ingresada no es válida."

        return True, ""

    @staticmethod
    def validar_nombres_apellidos(nombre):
        if not nombre.strip():
            return False, "El campo de nombres/apellidos está vacío."
        if not re.match(r"^[A-Za-zÁÉÍÓÚÑáéíóúñ\s]{2,50}$", nombre):
            return False, "El nombre/apellido debe contener solo letras y espacios, y tener al menos 2 caracteres."
        return True, ""

    @staticmethod
    def validar_correo(correo):
        if not correo.strip():
            return False, "El campo de correo está vacío."
        if not re.match(r"^[\w\.-]+@[\w\.-]+\.\w+$", correo):
            return False, "Ingrese un correo electrónico válido."
        return True, ""

    @staticmethod
    def validar_telefono(telefono):
        if not telefono.strip():
            return False, "El campo de teléfono está vacío."
        if not telefono.isdigit() or not re.match(r"^(09\d{8}|0[2-7]\d{7})$", telefono):
            return False, "Ingrese un número de teléfono válido."
        return True, ""

    @staticmethod
    def validar_fecha(fecha):
        """Valida que la fecha sea razonable (mayor de 18 años y menor de 100 años)."""
        from datetime import datetime
        
        if not fecha.strip():
            return False, "El campo de fecha de nacimiento está vacío."

        try:
            fecha_nac = datetime.strptime(fecha, "%Y-%m-%d")
            edad = (datetime.today() - fecha_nac).days // 365
            if edad < 18:
                return False, "El empleado debe ser mayor de 18 años."
            if edad > 100:
                return False, "Ingrese una fecha de nacimiento válida."
            return True, ""
        except ValueError:
            return False, "Formato de fecha inválido."

    @staticmethod
    def validar_campos(ci, apellidos, nombres, correo, telefono, fecha):
        """Valida todos los campos y muestra el error correspondiente."""

        # ✅ Si todos los campos están vacíos, muestra un mensaje general
        if not ci and not apellidos and not nombres and not correo and not telefono and not fecha:
            return False, "Debe completar al menos un campo antes de continuar."

        validaciones = [
            ValidadorEmpleado.validar_cedula(ci),
            ValidadorEmpleado.validar_nombres_apellidos(apellidos),
            ValidadorEmpleado.validar_nombres_apellidos(nombres),
            ValidadorEmpleado.validar_correo(correo),
            ValidadorEmpleado.validar_telefono(telefono),
            ValidadorEmpleado.validar_fecha(fecha)
        ]

        for valido, mensaje in validaciones:
            if not valido:
                return False, mensaje

        return True, ""
    
def validar_camara(cam_index=0):
    """Verifica si la cámara está disponible antes de usarla."""
    cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
    if cap.isOpened():
        cap.release()
        return True
    return False
