from PyQt6.QtCore import QProcess
import sys

def reiniciar_aplicacion():
    """ Reinicia la aplicación cerrando la instancia actual y ejecutando una nueva. """
    try:
        python = sys.executable  
        QProcess.startDetached(python, sys.argv)  
        sys.exit(0) 
    except Exception as e:
        print(f"Error al reiniciar la aplicación: {e}")
