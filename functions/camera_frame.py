import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import threading
from functions.recoignation_logic_optimized import reconocimiento_facial_optimizado

# Índices de las cámaras (modifica si es necesario)
camera_entrada = 0  # Primera cámara
camera_salida = 1   # Segunda cámara

# Crear hilos para cada cámara
thread1 = threading.Thread(target=reconocimiento_facial_optimizado, args=(camera_entrada, "Entrada"))
thread2 = threading.Thread(target=reconocimiento_facial_optimizado, args=(camera_salida, "Salida"))

# Iniciar ambos hilos
thread1.start()
thread2.start()

# Esperar a que ambos hilos terminen
thread1.join()
thread2.join()

print("✅ Ambos procesos de reconocimiento facial han terminado.")
