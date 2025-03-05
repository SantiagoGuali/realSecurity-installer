from datetime import datetime
import os
from utils.settings_controller import load_config, set_config_value
from functions.database_manager import DatabaseManager
    

db = DatabaseManager()

def check_and_generate_report_on_start():
    config = load_config()
    last_date_str = config.get("LAST_REPORT_DATE","")
    report_period = config.get("REPORT_PERIOD","mensual").lower()
    
    # Ejemplo: si es diario => comparamos con 'YYYY-MM-DD' de hoy
    # si es mensual => comparamos con 'YYYY-MM' o con la 1ra del mes

    hoy_str = datetime.now().strftime("%Y-%m-%d")
    
    if report_period == "diario":
        # si last_date_str == hoy_str => no generes => return
        if last_date_str == hoy_str:
            print("Ya se generó el reporte diario hoy. No se repite.")
            return
    elif report_period == "mensual":
        # Compare YYYY-MM:
        hoy_mes = datetime.now().strftime("%Y-%m")
        if last_date_str.startswith(hoy_mes):
            print("Ya se generó el reporte mensual este mes.")
            return
    # ...similar para semanal, anual, etc.

    # Si llegamos acá => generarlo
    file_path = os.path.join("reportes", f"reporte_{report_period}_{hoy_str}.xlsx")
    if not os.path.exists("reportes"):
        os.makedirs("reportes")

    success = db.auto_generate_employee_report(report_period,file_path)
    if not success:
        print("No se pudo generar el reporte.")
        return

    # si success => mandar correo
    if config.get("ENABLE_EMAIL_REPORTS",False):
        email = config.get("REPORT_EMAIL","").strip()
        if email:
            try:
                from functions.send_grid import enviar_reporte_por_correo
                enviar_reporte_por_correo(file_path,email)
                print("Reporte enviado con éxito.")
                # marcar la fecha en config => 'LAST_REPORT_DATE'
                if report_period == "diario":
                    set_config_value("LAST_REPORT_DATE", hoy_str)
                elif report_period == "mensual":
                    # guardamos 'YYYY-MM-DD' o 'YYYY-MM' si gustas
                    set_config_value("LAST_REPORT_DATE", hoy_mes+"-01")  # p.ej
            except Exception as e:
                print(f"Fallo al enviar correo => {e}")
                # No marcamos last_date => reintentará proximavez
        else:
            print("No hay correo configurado => no se envía.")
    else:
        # no se envían correos => igual marcamos la fecha en config
        if report_period == "diario":
            set_config_value("LAST_REPORT_DATE", hoy_str)
        elif report_period == "mensual":
            hoy_mes = datetime.now().strftime("%Y-%m")
            set_config_value("LAST_REPORT_DATE", hoy_mes+"-01")
