# attendance_manager.py
from datetime import datetime
from functions.database_manager import DatabaseManager

db = DatabaseManager()

def registrar_asistencia(emp_id, tipo):
    """ Registra la asistencia de un empleado en la base de datos. """
    fecha_actual = datetime.now().strftime('%Y-%m-%d')
    hora_actual = datetime.now().strftime('%H:%M:%S')

    if not validar_asistencia(emp_id, tipo):
        db.registrar_asistencia(emp_id, tipo)
        print(f"✅ Asistencia registrada para empleado ID {emp_id} ({tipo}).")
    else:
        print(f"⏳ Registro duplicado evitado para empleado ID {emp_id}.")

def validar_asistencia(emp_id, tipo):
    """ Verifica si el empleado ya registró asistencia recientemente. """
    asistencias = db.get_asistencias_ci_emp(emp_id)
    if not asistencias:
        return False

    ultima_asistencia = asistencias[-1]
    ultima_fecha, ultima_hora, ultimo_tipo = ultima_asistencia[1], ultima_asistencia[2], ultima_asistencia[3]

    ahora = datetime.now()
    diferencia = (ahora - datetime.strptime(f"{ultima_fecha} {ultima_hora}", '%Y-%m-%d %H:%M:%S')).total_seconds()

    return diferencia < 60 and ultimo_tipo == tipo
